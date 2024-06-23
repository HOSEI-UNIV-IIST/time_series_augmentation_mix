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

