
import os
import numpy as np
import pandas as pd

def pad_sequences(sequences, maxlen=None, dtype='float32', padding='post', truncating='post', value=0.0):
    """
    Pad each sequence to the same length.
    If maxlen is provided, pad to this length.
    If maxlen is not provided, pad to the length of the longest sequence.
    """
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        trunc = s[-maxlen:] if truncating == 'pre' else s[:maxlen]
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')
    return x

def load_data_from_file(data_file, label_file=None, is_augmented=False):
    """
    Load data and labels from CSV files, ignoring headers and handling only values.
    If is_augmented is True, assumes labels are in the last column of data.
    """
    if label_file:
        # Load data and labels from separate files, assuming they have no headers
        data = pd.read_csv(data_file, sep=",").values
        labels = pd.read_csv(label_file, sep=",").values
        if labels.ndim > 1:
            labels = labels[:, 0]  # Use only the first column if labels has multiple columns
    else:
        # Load data and labels from a single file, assuming no headers
        raw_data = pd.read_csv(data_file)
        raw_data = raw_data.values
        if is_augmented:
            labels = raw_data[:, -1]  # Labels are in the last column for augmented data
            data = raw_data[:, :-1]   # Features are all columns except the last
        else:
            labels = raw_data[:, 0]   # Labels are in the first column for non-augmented data
            data = raw_data[:, :]    # Features are all columns except the first

    return data, labels

def read_data_sets(train_file, train_label=None, test_file=None, test_label=None, test_split=0.1, is_train_augmented=False):
    train_data, train_labels = load_data_from_file(train_file, train_label, is_augmented=is_train_augmented)
    if test_file:
        test_data, test_labels = load_data_from_file(test_file, test_label)
    else:
        # Split train data into train and test sets
        test_size = int(test_split * len(train_labels))
        test_data = train_data[:test_size]
        test_labels = train_labels[:test_size]
        train_data = train_data[test_size:]
        train_labels = train_labels[test_size:]
    return train_data, train_labels, test_data, test_labels


def check_for_augmented_train_file(train_file, augmentation_method):
    """
    Check if an augmented version of the train file exists and return the appropriate file path.

    Parameters:
    - train_file: The original training data file path.
    - augmentation_method: The augmentation method identifier to check in the filename.

    Returns:
    - augmented_file (str): Path to the augmented file if it exists, otherwise the original train file path.
    - is_augmented (bool): True if the augmented file exists, False otherwise.
    - method_used (str or None): The augmentation method used if an augmented file exists, otherwise None.
    """
    # Split the file path into the base name and extension
    base, ext = os.path.splitext(train_file)

    # Construct the augmented file path based on the augmentation method
    augmented_file = f"{base}_{augmentation_method}{ext}"

    # Check if the augmented file exists
    if os.path.exists(augmented_file):
        return augmented_file, True, augmentation_method
    else:
        return train_file, False, None


def get_datasets(args):
    """
    Main function to get train and test datasets, optionally checking for augmented versions and normalizing as needed.
    """
    # Determine file paths
    train_data_file = os.path.join(args.data_dir, args.dataset, f"{args.dataset}_TRAIN.CSV")
    test_data_file = os.path.join(args.data_dir, args.dataset, f"{args.dataset}_TEST.CSV")

    # Check for augmented train data file if requested
    is_train_augmented = False
    augmentation_method = None
    if args.read_augmented:
        train_data_file, is_train_augmented, augmentation_method = check_for_augmented_train_file(
            train_data_file, args.augmentation_method
        )

    # Load data
    x_train, y_train, x_test, y_test = read_data_sets(
        train_data_file, args.train_labels_file, test_data_file, args.test_labels_file,
        test_split=args.test_split, is_train_augmented=is_train_augmented
    )

    # Pad sequences to the same length
    x_train = pad_sequences(x_train)
    x_test = pad_sequences(x_test)

    # Normalize train data if needed
    if args.normalize_input and not is_train_augmented:
        x_train_min, x_train_max = np.nanmin(x_train), np.nanmax(x_train)
        y_train_min, y_train_max = np.nanmin(y_train), np.nanmax(y_train)

        if args.normalize_input_positive:
            x_train = (x_train - x_train_min) / (x_train_max - x_train_min)
            y_train = (y_train - y_train_min) / (y_train_max - y_train_min)
        else:
            x_train = 2. * (x_train - x_train_min) / (x_train_max - x_train_min) - 1.
            y_train = 2. * (y_train - y_train_min) / (y_train_max - y_train_min) - 1.

    # Normalize test data independently
    if test_data_file and args.normalize_input:
        x_test_min, x_test_max = np.nanmin(x_test), np.nanmax(x_test)
        y_test_min, y_test_max = np.nanmin(y_test), np.nanmax(y_test)

        if args.normalize_input_positive:
            x_test = (x_test - x_test_min) / (x_test_max - x_test_min)
            y_test = (y_test - y_test_min) / (y_test_max - y_test_min)
        else:
            x_test = 2. * (x_test - x_test_min) / (x_test_max - x_test_min) - 1.
            y_test = 2. * (y_test - y_test_min) / (y_test_max - y_test_min) - 1.

    # Replace NaNs with zeroes
    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)

    return x_train, y_train, x_test, y_test, is_train_augmented, augmentation_method
