import os

import numpy as np

def pad_sequences(sequences, maxlen=None, dtype='float32', padding='post', truncating='post', value=0.0):
    """
    Pad each sequence to the same length.
    If maxlen is provided, pad to this length.
    If maxlen is not provided, pad to the length of the longest sequence.

    Parameters:
    sequences (list of arrays): List of sequences to pad
    maxlen (int): Maximum length to pad the sequences
    dtype (str): Desired output data-type
    padding (str): 'pre' or 'post', pad either before or after each sequence
    truncating (str): 'pre' or 'post', remove values from sequences larger than maxlen either in the beginning or in the end
    value (float): Padding value

    Returns:
    np.array: Padded sequences array
    """
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def load_data_from_file(data_file, label_file=None, delimiter=" "):
    if label_file:
        data = np.genfromtxt(data_file, delimiter=delimiter)
        labels = np.genfromtxt(label_file, delimiter=delimiter)
        if labels.ndim > 1:
            labels = labels[:,1]
    else:
        data = np.genfromtxt(data_file, delimiter=delimiter)
        labels = data[:,0]
        data = data[:,1:]
    return data, labels
    
def read_data_sets(train_file, train_label=None, test_file=None, test_label=None, test_split=0.1, delimiter=" "):
    train_data, train_labels = load_data_from_file(train_file, train_label, delimiter)
    if test_file:
        test_data, test_labels = load_data_from_file(test_file, test_label, delimiter)
    else:
        test_size = int(test_split * float(train_labels.shape[0]))
        test_data = train_data[:test_size]
        test_labels = train_labels[:test_size]
        train_data = train_data[test_size:]
        train_labels = train_labels[test_size:]
    return train_data, train_labels, test_data, test_labels


def get_datasets(args):
    # Load data
    if args.preset_files:
        if args.dataset == 'CBF':
            train_data_file = os.path.join(args.data_dir, args.dataset, "CBF_TRAIN.tsv")
            test_data_file = os.path.join(args.data_dir, args.dataset, "CBF_TEST.tsv")
            x_train, y_train, x_test, y_test = read_data_sets(train_data_file, "", test_data_file, "", delimiter="\t")
        elif args.ucr:
            train_data_file = os.path.join(args.data_dir, args.dataset, "%s_TRAIN.txt" % args.dataset)
            test_data_file = os.path.join(args.data_dir, args.dataset, "%s_TEST.txt" % args.dataset)
            x_train, y_train, x_test, y_test = read_data_sets(train_data_file, "", test_data_file, "", delimiter="")
        elif args.ucr2018:
            train_data_file = os.path.join(args.data_dir, args.dataset, "%s_TRAIN.tsv" % args.dataset)
            test_data_file = os.path.join(args.data_dir, args.dataset, "%s_TEST.tsv" % args.dataset)
            x_train, y_train, x_test, y_test = read_data_sets(train_data_file, "", test_data_file, "", delimiter="\t")
        else:
            x_train_file = os.path.join(args.data_dir, "train-%s-data.txt" % (args.dataset))
            y_train_file = os.path.join(args.data_dir, "train-%s-labels.txt" % (args.dataset))
            x_test_file = os.path.join(args.data_dir, "test-%s-data.txt" % (args.dataset))
            y_test_file = os.path.join(args.data_dir, "test-%s-labels.txt" % (args.dataset))
            x_train, y_train, x_test, y_test = read_data_sets(x_train_file, y_train_file, x_test_file, y_test_file,
                                                              test_split=args.test_split, delimiter=args.delimiter)
    else:
        x_train, y_train, x_test, y_test = read_data_sets(args.train_data_file, args.train_labels_file,
                                                          args.test_data_file, args.test_labels_file,
                                                          test_split=args.test_split, delimiter=args.delimiter)

    # Pad sequences to the same length
    x_train = pad_sequences(x_train)
    x_test = pad_sequences(x_test)

    # Normalize
    if args.normalize_input:
        x_train_max = np.nanmax(x_train)
        x_train_min = np.nanmin(x_train)
        x_train = 2. * (x_train - x_train_min) / (x_train_max - x_train_min) - 1.
        # Test is secret
        x_test = 2. * (x_test - x_train_min) / (x_train_max - x_train_min) - 1.

    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)
    return x_train, y_train, x_test, y_test


def run_augmentation(x, y, args):
    print("Augmenting %s" % args.dataset)
    np.random.seed(args.seed)
    x_aug = x
    y_aug = y
    augmentation_tags = args.extra_tag

    if args.augmentation_ratio > 0:
        augmentation_tags = "%d" % args.augmentation_ratio
        for n in range(args.augmentation_ratio):
            if 'sequential' in args.augmentation_method:
                # For sequential methods, use the updated data from the previous round
                x_temp, y_temp, temp_tags = augment_sequential(x_aug, y_aug, args)
            else:
                # For simple and parallel methods, use the original data
                if args.augmentation_method == 'simple':
                    x_temp, temp_tags = augment_data_simple(x, args)
                elif 'parallel_magnitude' in args.augmentation_method:
                    method_num = args.augmentation_method[-1]
                    x_temp, y_temp, temp_tags = globals()[f'augment_data_parallel_magnitude{method_num}'](x, y)
                elif 'parallel_time' in args.augmentation_method:
                    method_num = args.augmentation_method[-1]
                    x_temp, y_temp, temp_tags = globals()[f'augment_data_parallel_time{method_num}'](x, y)
                elif 'parallel_combined' in args.augmentation_method:
                    method_num = args.augmentation_method[-1]
                    x_temp, y_temp, temp_tags = globals()[f'augment_data_parallel_combined{method_num}'](x, y)
                else:
                    raise ValueError("Unknown augmentation method")

            x_aug = np.append(x_aug, x_temp, axis=0)
            y_aug = np.append(y_aug, y_temp, axis=0)
            print("Round %d: %s done" % (n, augmentation_tags + temp_tags))

        augmentation_tags += temp_tags
        if args.extra_tag:
            augmentation_tags += "_" + args.extra_tag
    else:
        augmentation_tags = args.extra_tag

    return x_aug, y_aug, augmentation_tags


def augment_sequential(x, y, args):
    if 'sequential_magnitude' in args.augmentation_method:
        method_num = args.augmentation_method[-1]
        return globals()[f'augment_data_sequential_magnitude{method_num}'](x, y)
    elif 'sequential_time' in args.augmentation_method:
        method_num = args.augmentation_method[-1]
        return globals()[f'augment_data_sequential_time{method_num}'](x, y)
    elif 'sequential_combined' in args.augmentation_method:
        method_num = args.augmentation_method[-1]
        return globals()[f'augment_data_sequential_combined{method_num}'](x, y)
    else:
        raise ValueError("Unknown sequential augmentation method")


def augment_data_simple(x, args):
    import utils.augmentation as aug
    augmentation_tags = ""
    if args.jitter:
        x = aug.jitter(x)
        augmentation_tags += "_jitter"
    elif args.scaling:
        x = aug.scaling(x)
        augmentation_tags += "_scaling"
    elif args.rotation:
        x = aug.rotation(x)
        augmentation_tags += "_rotation"
    elif args.permutation:
        x = aug.permutation(x)
        augmentation_tags += "_permutation"
    elif args.randompermutation:
        x = aug.permutation(x, seg_mode="random")
        augmentation_tags += "_randomperm"
    elif args.magwarp:
        x = aug.magnitude_warp(x)
        augmentation_tags += "_magwarp"
    elif args.timewarp:
        x = aug.time_warp(x)
        augmentation_tags += "_timewarp"
    elif args.windowslice:
        x = aug.window_slice(x)
        augmentation_tags += "_windowslice"
    elif args.windowwarp:
        x = aug.window_warp(x)
        augmentation_tags += "_windowwarp"
    return x, augmentation_tags


'''Sequential Augmentation Methods'''


def augment_data_sequential_magnitude1(x, y):
    import utils.augmentation as aug
    x = aug.jitter(x)
    x = aug.scaling(x)
    x = aug.rotation(x)
    x = aug.magnitude_warp(x)
    augmentation_tags = "_augment_data_sequential_magnitude1"
    return x, y, augmentation_tags


def augment_data_sequential_magnitude2(x, y):
    import utils.augmentation as aug
    x = aug.scaling(x)
    x = aug.rotation(x)
    x = aug.magnitude_warp(x)
    x = aug.jitter(x)
    augmentation_tags = "_augment_data_sequential_magnitude2"
    return x, y, augmentation_tags


def augment_data_sequential_magnitude3(x, y):
    import utils.augmentation as aug
    x = aug.rotation(x)
    x = aug.magnitude_warp(x)
    x = aug.scaling(x)
    x = aug.jitter(x)
    augmentation_tags = "_augment_data_sequential_magnitude3"
    return x, y, augmentation_tags


def augment_data_sequential_magnitude4(x, y):
    import utils.augmentation as aug
    x = aug.magnitude_warp(x)
    x = aug.scaling(x)
    x = aug.jitter(x)
    x = aug.rotation(x)
    augmentation_tags = "_augment_data_sequential_magnitude4"
    return x, y, augmentation_tags


def augment_data_sequential_time1(x, y):
    import utils.augmentation as aug
    x = aug.permutation(x)
    x = aug.time_warp(x)
    x = aug.window_warp(x)
    x = aug.window_slice(x)
    augmentation_tags = "_augment_data_sequential_time1"
    return x, y, augmentation_tags


def augment_data_sequential_time2(x, y):
    import utils.augmentation as aug
    x = aug.time_warp(x)
    x = aug.window_warp(x)
    x = aug.window_slice(x)
    x = aug.permutation(x)
    augmentation_tags = "_augment_data_sequential_time2"
    return x, y, augmentation_tags


def augment_data_sequential_time3(x, y):
    import utils.augmentation as aug
    x = aug.window_warp(x)
    x = aug.window_slice(x)
    x = aug.permutation(x)
    x = aug.time_warp(x)
    augmentation_tags = "_augment_data_sequential_time3"
    return x, y, augmentation_tags


def augment_data_sequential_time4(x, y):
    import utils.augmentation as aug
    x = aug.window_slice(x)
    x = aug.permutation(x)
    x = aug.time_warp(x)
    x = aug.window_warp(x)
    augmentation_tags = "_augment_data_sequential_time4"
    return x, y, augmentation_tags


def augment_data_sequential_combined1(x, y):
    import utils.augmentation as aug
    x = aug.jitter(x)
    x = aug.permutation(x)
    x = aug.scaling(x)
    x = aug.window_warp(x)
    augmentation_tags = "_augment_data_sequential_combined1"
    return x, y, augmentation_tags


def augment_data_sequential_combined2(x, y):
    import utils.augmentation as aug
    x = aug.permutation(x)
    x = aug.scaling(x)
    x = aug.window_slice(x)
    x = aug.rotation(x)
    augmentation_tags = "_augment_data_sequential_combined2"
    return x, y, augmentation_tags


def augment_data_sequential_combined3(x, y):
    import utils.augmentation as aug
    x = aug.scaling(x)
    x = aug.window_slice(x)
    x = aug.rotation(x)
    x = aug.magnitude_warp(x)
    augmentation_tags = "_augment_data_sequential_combined3"
    return x, y, augmentation_tags


def augment_data_sequential_combined4(x, y):
    import utils.augmentation as aug
    x = aug.permutation(x)
    x = aug.rotation(x)
    x = aug.magnitude_warp(x)
    x = aug.time_warp(x)
    augmentation_tags = "_augment_data_sequential_combined4"
    return x, y, augmentation_tags


def augment_data_sequential_combined5(x, y):
    import utils.augmentation as aug
    x = aug.rotation(x)
    x = aug.magnitude_warp(x)
    x = aug.time_warp(x)
    x = aug.window_slice(x)
    augmentation_tags = "_augment_data_sequential_combined5"
    return x, y, augmentation_tags


def augment_data_sequential_combined6(x, y):
    import utils.augmentation as aug
    x = aug.magnitude_warp(x)
    x = aug.time_warp(x)
    x = aug.window_slice(x)
    x = aug.window_warp(x)
    augmentation_tags = "_augment_data_sequential_combined6"
    return x, y, augmentation_tags


def augment_data_sequential_combined7(x, y):
    import utils.augmentation as aug
    x = aug.time_warp(x)
    x = aug.window_slice(x)
    x = aug.window_warp(x)
    x = aug.permutation(x)
    augmentation_tags = "_augment_data_sequential_combined7"
    return x, y, augmentation_tags


def augment_data_sequential_combined8(x, y):
    import utils.augmentation as aug
    x = aug.window_slice(x)
    x = aug.window_warp(x)
    x = aug.permutation(x)
    x = aug.jitter(x)
    augmentation_tags = "_augment_data_sequential_combined8"
    return x, y, augmentation_tags


def augment_data_sequential_combined9(x, y):
    import utils.augmentation as aug
    x = aug.window_warp(x)
    x = aug.rotation(x)
    x = aug.jitter(x)
    x = aug.permutation(x)
    augmentation_tags = "_augment_data_sequential_combined9"
    return x, y, augmentation_tags


def augment_data_sequential_combined10(x, y):
    import utils.augmentation as aug
    x = aug.rotation(x)
    x = aug.jitter(x)
    x = aug.permutation(x)
    x = aug.scaling(x)
    augmentation_tags = "_augment_data_sequential_combined10"
    return x, y, augmentation_tags


'''Parallel Augmentation Methods'''


def augment_data_parallel_magnitude1(x, y):
    import utils.augmentation as aug
    x_jitter = aug.jitter(x.copy())
    x_scaling = aug.scaling(x.copy())
    x_rotation = aug.rotation(x.copy())
    x_magnitude_warp = aug.magnitude_warp(x.copy())

    x_combined = np.concatenate((x_jitter, x_scaling, x_rotation, x_magnitude_warp))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_magnitude1"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_magnitude2(x, y):
    import utils.augmentation as aug
    x_scaling = aug.scaling(x.copy())
    x_rotation = aug.rotation(x.copy())
    x_magnitude_warp = aug.magnitude_warp(x.copy())
    x_jitter = aug.jitter(x.copy())

    x_combined = np.concatenate((x_scaling, x_rotation, x_magnitude_warp, x_jitter))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_magnitude2"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_magnitude3(x, y):
    import utils.augmentation as aug
    x_rotation = aug.rotation(x.copy())
    x_magnitude_warp = aug.magnitude_warp(x.copy())
    x_scaling = aug.scaling(x.copy())
    x_jitter = aug.jitter(x.copy())

    x_combined = np.concatenate((x_rotation, x_magnitude_warp, x_scaling, x_jitter))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_magnitude3"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_magnitude4(x, y):
    import utils.augmentation as aug
    x_magnitude_warp = aug.magnitude_warp(x.copy())
    x_scaling = aug.scaling(x.copy())
    x_jitter = aug.jitter(x.copy())
    x_rotation = aug.rotation(x.copy())

    x_combined = np.concatenate((x_magnitude_warp, x_scaling, x_jitter, x_rotation))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_magnitude4"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_time1(x, y):
    import utils.augmentation as aug
    x_permutation = aug.permutation(x.copy())
    x_time_warp = aug.time_warp(x.copy())
    x_window_warp = aug.window_warp(x.copy())
    x_window_slice = aug.window_slice(x.copy())

    x_combined = np.concatenate((x_permutation, x_time_warp, x_window_warp, x_window_slice))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_time1"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_time2(x, y):
    import utils.augmentation as aug
    x_time_warp = aug.time_warp(x.copy())
    x_window_warp = aug.window_warp(x.copy())
    x_window_slice = aug.window_slice(x.copy())
    x_permutation = aug.permutation(x.copy())

    x_combined = np.concatenate((x_time_warp, x_window_warp, x_window_slice, x_permutation))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_time2"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_time3(x, y):
    import utils.augmentation as aug
    x_window_warp = aug.window_warp(x.copy())
    x_window_slice = aug.window_slice(x.copy())
    x_permutation = aug.permutation(x.copy())
    x_time_warp = aug.time_warp(x.copy())

    x_combined = np.concatenate((x_window_warp, x_window_slice, x_permutation, x_time_warp))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_time3"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_time4(x, y):
    import utils.augmentation as aug
    x_window_slice = aug.window_slice(x.copy())
    x_permutation = aug.permutation(x.copy())
    x_time_warp = aug.time_warp(x.copy())
    x_window_warp = aug.window_warp(x.copy())

    x_combined = np.concatenate((x_window_slice, x_permutation, x_time_warp, x_window_warp))
    y_combined = np.concatenate((y, y, y, y))
    augmentation_tags = "_augment_data_parallel_time4"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined1(x, y):
    import utils.augmentation as aug
    x_jitter = aug.jitter(x.copy())
    x_permutation = aug.permutation(x.copy())

    x_combined = np.concatenate((x_jitter, x_permutation))
    y_combined = np.concatenate((y, y))
    augmentation_tags = "_augment_data_parallel_combined1"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined2(x, y):
    import utils.augmentation as aug
    x_scaling = aug.scaling(x.copy())
    x_permutation = aug.permutation(x.copy())

    x_combined = np.concatenate((x_scaling, x_permutation))
    y_combined = np.concatenate((y, y))
    augmentation_tags = "_augment_data_parallel_combined2"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined3(x, y):
    import utils.augmentation as aug
    x_rotation = aug.rotation(x.copy())
    x_time_warp = aug.time_warp(x.copy())

    x_combined = np.concatenate((x_rotation, x_time_warp))
    y_combined = np.concatenate((y, y))
    augmentation_tags = "_augment_data_parallel_combined3"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined4(x, y):
    import utils.augmentation as aug
    x_magnitude_warp = aug.magnitude_warp(x.copy())
    x_window_slice = aug.window_slice(x.copy())

    x_combined = np.concatenate((x_magnitude_warp, x_window_slice))
    y_combined = np.concatenate((y, y))
    augmentation_tags = "_augment_data_parallel_combined4"
    return x_combined, y_combined, augmentation_tags


def augment_data_parallel_combined5(x, y):
    import utils.augmentation as aug
    x_window_warp = aug.window_warp(x.copy())
    x_window_slice = aug.window_slice(x.copy())

    x_combined = np.concatenate((x_window_warp, x_window_slice))
    y_combined = np.concatenate((y, y))
    augmentation_tags = "_augment_data_parallel_combined5"
    return x_combined, y_combined, augmentation_tags
