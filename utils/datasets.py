def nb_dims(dataset):
    if dataset is "CBF":
        return 1
    return 2


def nb_classes(dataset):
    if dataset == "ELECTRICITY":
        return 9
    if dataset == "ENERGY":
        return 6
    if dataset == "HOUSE":
        return 5
    if dataset == "APARTMENT":
        return 20
    if dataset == "WATER":
        return 10
    if dataset == "CBF":
        return 128
    print("Missing dataset: %s" % dataset)
    return 2


def class_offset(y, dataset):
    return (y + class_modifier_add(dataset)) * class_modifier_multi(dataset)


def class_modifier_add(dataset):
    if dataset == "CBF":
        return -1  # 128
    return 0


def class_modifier_multi(dataset):
    if dataset == "ECG200":
        return 0.5  # 96
    return 1