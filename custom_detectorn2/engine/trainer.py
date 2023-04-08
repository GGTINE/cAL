# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# for ema scheduler
import os
from collections import OrderedDict

import numpy as np

from .label_tools import add_label, remove_label

import torch
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel

from fvcore.nn.precise_bn import get_bn_modules

import detectron2.utils.comm as comm
from detectron2.engine import DefaultTrainer, hooks, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.evaluation import (
    COCOEvaluator,
    verify_results,
    DatasetEvaluator,
    print_csv_format,
)
from detectron2.utils.events import EventStorage

from custom_detectorn2.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from custom_detectorn2.data.build import (
    build_detection_semisup_train_loader_two_crops,
    build_detection_test_loader,
)
from custom_detectorn2.data.dataset_mapper import DatasetMapperTwoCropSeparate
from custom_detectorn2.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from custom_detectorn2.modeling.pseudo_generator import PseudoGenerator
from custom_detectorn2.solver.build import build_lr_scheduler
from custom_detectorn2.evaluation.evaluator import inference_on_dataset


class ActiveTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        self.accumulation = 2
        self.accumulate_steps = 0

        # Student 모델 생성
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # Teacher 모델 생성
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher
        self.model_teacher.eval()

        data_loader = self.build_train_loader(cfg)

        # Multi-GPU 대응
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(self.model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.pseudo_generator = PseudoGenerator(cfg)
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        if cfg.TEST.EVALUATOR == "COCOeval":
            return COCOEvaluator(dataset_name, cfg, True, output_dir=output_folder)
        else:
            raise NotImplementedError("Not Implemented")

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                raise
            finally:
                self.after_train()

    # ===============[Training Flow]===============
    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[ActiveLearning] model was changed to eval mode!"

        data = next(self._trainer._data_loader_iter)
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data

        label_data_q.extend(label_data_k)

        # burn_in stage
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            with autocast(enabled=self.cfg.SOLVER.AMP.ENABLED):
                record_dict = self.model(label_data_q, branch="labeled")
                losses = sum(record_dict.values()) / self.accumulation

            if self.cfg.SOLVER.AMP.ENABLED:
                self._trainer.grad_scaler.scale(losses).backward()
            else:
                losses.backward()

        # label + unlabel train stage
        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                self._update_teacher_model(keep_rate=0.00)

                unlabel_data_q = remove_label(unlabel_data_q)
                unlabel_data_k = remove_label(unlabel_data_k)

            with torch.no_grad():
                pred_teacher, raw_pred_teacher = self.model_teacher(
                    unlabel_data_k,
                    output_raw=True,
                    nms_method=self.cfg.MODEL.FCOS.NMS_CRITERIA_TRAIN,
                    branch="teacher_weak",
                )

            self.create_pseudo_label(
                unlabel_data=unlabel_data_q,
                pred_teacher=pred_teacher,
                raw_pred_teacher=raw_pred_teacher,
                nms_method=self.cfg.MODEL.FCOS.NMS_CRITERIA_REG_TRAIN,
            )

            with autocast(self.cfg.SOLVER.AMP.ENABLED):
                record_dict = self.model(label_data_q, branch="labeled")
                record_all_unlabel_data = self.model(unlabel_data_q, branch="unlabeled")
                record_dict.update(record_all_unlabel_data)
                losses = sum(record_dict.values()) / self.accumulation

            if self.cfg.SOLVER.AMP.ENABLED:
                self._trainer.grad_scaler.scale(losses).backward()
            else:
                losses.backward()

        self._write_metrics(record_dict)
        self.accumulate_steps += 1

        if self.accumulate_steps % self.accumulation == 0:
            if self.cfg.SOLVER.AMP.ENABLED:
                self._trainer.grad_scaler.step(self.optimizer)
                self._trainer.grad_scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.accumulate_steps = 0

            if self.iter > self.cfg.SEMISUPNET.BURN_UP_STEP:
                self._update_teacher_model(keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

    def _write_metrics(self, metrics_dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_teacher))

        if comm.is_main_process():
            # ret.append(hooks.BestCheckpointer(cfg.TEST.EVAL_PERIOD, self.checkpointer, "bbox/AP"))
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def set_threshold(self):
        if self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE == "thresholding":
            threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD
        elif self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE == "thresholding_cls_ctr":
            threshold = (
                self.cfg.SEMISUPNET.BBOX_THRESHOLD,
                self.cfg.SEMISUPNET.BBOX_CTR_THRESHOLD,
            )
        else:
            raise ValueError("Invalid value for pseudo_bounding_box_sample.")

        if self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE_REG == "thresholding":
            threshold_reg = self.cfg.SEMISUPNET.BBOX_THRESHOLD_REG
        elif self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE_REG == "thresholding_cls_ctr":
            threshold_reg = (
                self.cfg.SEMISUPNET.BBOX_THRESHOLD_REG,
                self.cfg.SEMISUPNET.BBOX_CTR_THRESHOLD_REG,
            )
        else:
            raise ValueError("Invalid value for pseudo_bounding_box_sample_reg.")

        return threshold, threshold_reg

    def create_pseudo_label(
        self, unlabel_data, pred_teacher, raw_pred_teacher, nms_method
    ):
        # pred_teacher_loc = self.pseudo_generator.nms_from_dense(
        #     raw_pred_teacher, nms_method=self.cfg.MODEL.FCOS.NMS_CRITERIA_REG_TRAIN
        # )

        threshold, threshold_reg = self.set_threshold()

        joint_proposal_dict = {}

        (
            pesudo_proposals_roih_unsup_k,
            _,
        ) = self.pseudo_generator.process_pseudo_label(
            pred_teacher,
            threshold,
            "roih",
            self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE,
        )
        joint_proposal_dict["proposals_pseudo_cls"] = pesudo_proposals_roih_unsup_k

        # (
        #     pesudo_proposals_roih_unsup_k_reg,
        #     _,
        # ) = self.pseudo_generator.process_pseudo_label(
        #     pred_teacher_loc,
        #     threshold_reg,
        #     "roih",
        #     self.cfg.SEMISUPNET.PSEUDO_BBOX_SAMPLE_REG,
        # )
        # joint_proposal_dict["proposals_pseudo_reg"] = pesudo_proposals_roih_unsup_k_reg

        unlabel_data = add_label(
            unlabel_data, joint_proposal_dict["proposals_pseudo_cls"], ""
        )
        # unlabel_data = add_label(
        #     unlabel_data, joint_proposal_dict["proposals_pseudo_reg"], "reg"
        # )

        # unlabeled data pseudo-labeling
        for data in unlabel_data:
            assert len(data) != 0, "unlabeled data must have at least one pseudo-box"

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator, cfg)
            # results_i = inference_on_dataset(model, data_loader, evaluator)

            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results
