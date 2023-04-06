from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

# hacky way to register
from custom_detectorn2.modeling import *
from custom_detectorn2.engine import *

from custom_detectorn2 import add_ubteacher_config


def setup(args):
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    active_trainer = ActiveTrainer(cfg)

    if args.eval_only:
        model = active_trainer.build_model(cfg)
        model_teacher = active_trainer.build_model(cfg)
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        DetectionCheckpointer(ensem_ts_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = active_trainer.test(cfg, ensem_ts_model.modelTeacher)

        return res

    active_trainer.resume_or_load(resume=args.resume)

    return active_trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.num_gpus = 1
    args.config_file = "./configs/fcos/sup1.yaml"

    print("Command Line Args:", args)
    launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank, dist_url=args.dist_url, args=(args,))
