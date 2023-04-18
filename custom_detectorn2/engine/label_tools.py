def remove_label(label_data):
    for label_datum in label_data:
        if "instances" in label_datum.keys():
            del label_datum["instances"]
    return label_data


def add_label(unlabled_data, label, labeltype=""):
    for unlabel_datum, lab_inst in zip(unlabled_data, label):
        if labeltype == "class":
            unlabel_datum["instances_class"] = lab_inst
        elif labeltype == "reg":
            unlabel_datum["instances_reg"] = lab_inst
        else:
            unlabel_datum["instances"] = lab_inst
    return unlabled_data
