import numpy as np


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
                    x_temp, y_temp, temp_tags = augment_data_simple(x, y, args)
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


def augment_data_simple(x, y, args):
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
    return x, y, augmentation_tags


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


'''
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
'''

'''Parallel Augmentation Methods'''


def augment_data_parallel_magnitude1(x, y):
    import utils.augmentation as aug
    x_1, x_2, x_3, x_4 = x[:len(x) // 4], x[len(x) // 4:len(x) // 2], x[len(x) // 2:3 * len(x) // 4], x[
                                                                                                      3 * len(x) // 4:]
    y_1, y_2, y_3, y_4 = y[:len(y) // 4], y[len(y) // 4:len(y) // 2], y[len(y) // 2:3 * len(y) // 4], y[
                                                                                                      3 * len(y) // 4:]
    x_jitter = aug.jitter(x_1.copy())
    x_scaling = aug.scaling(x_2.copy())
    x_rotation = aug.rotation(x_3.copy())
    x_magnitude_warp = aug.magnitude_warp(x_4.copy())

    x_combined = np.concatenate((x_jitter, x_scaling, x_rotation, x_magnitude_warp))
    y_combined = np.concatenate((y_1, y_2, y_3, y_4))
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


'''
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
'''
