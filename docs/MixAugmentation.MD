# New Augmentation Techniques by Mixing methods

Combining multiple data augmentation techniques for time series classification add significance enhancement to the
robustness and
performance of the model by creating a diverse and extensive training dataset. Here’s a structured approach to
combining the techniques we’ve implemented:

## 1. Sequential Combination

Sequential combination involves applying augmentation techniques one after the other. This approach simulate complex
variations in the data. For example:

```
def augment_data_sequential(x, y):
    x = jitter(x)
    x = scaling(x)
    x = rotation(x)
    x = permutation(x)
    x = randompermutation(x)
    x = magnitude_warp(x)
    x = time_warp(x)
    x = window_slice(x)
    x = window_warp(x)
    x = spawner(x, y)
    return x
```

## 2. Random Combination

Randomly apply a subset of augmentation techniques to each data instance to introduce stochastic variability.

```
def augment_data_random(x, y, num_augmentations=3):
    augmentations = [jitter, scaling, rotation, permutation, randompermutation, magnitude_warp, time_warp, window_slice, window_warp]
    
    for i in range(x.shape[0]):
        chosen_augmentations = np.random.choice(augmentations, num_augmentations, replace=False)
        for aug in chosen_augmentations:
            x[i] = aug(x[i])
    return x
```

## 3. Parallel Combination

Apply different augmentation techniques to different portions of the dataset and then combine the results.

```
def augment_data_parallel(x, y):
    x_jitter = jitter(x.copy())
    x_scaling = scaling(x.copy())
    x_rotation = rotation(x.copy())
    x_permutation = permutation(x.copy())
    x_randompermutation = randompermutation(x.copy())
    x_magnitude_warp = magnitude_warp(x.copy())
    x_time_warp = time_warp(x.copy())
    x_window_slice = window_slice(x.copy())
    x_window_warp = window_warp(x.copy())

    x_combined = np.concatenate((x_jitter, x_scaling,x_rotation, x_permutation, x_randompermutation, x_magnitude_warp, x_time_warp, x_window_slice, x_window_warp))
    y_combined = np.concatenate((y, y, y, y, y, y, y, y, y, y, y, y, y))
    
    return x_combined, y_combined
```

## 4. Custom Combination Strategy

Create a custom strategy where we design specific rules for combining different augmentation methods based on the
dataset characteristics and task requirements.

```
def custom_augmentation_strategy(x, y):
    for i in range(x.shape[0]):
        if i % 2 == 0:
            x[i] = jitter(scaling(x[i]))
        else:
            x[i] = permutation(magnitude_warp(x[i]))
    return x
```

## Implementation Example

A complete example of integrating these combinations into the data pipeline:

```
def augment_data(x, y, method='sequential', num_augmentations=3):
    if method == 'sequential':
        return augment_data_sequential(x, y)
    elif method == 'random':
        return augment_data_random(x, y, num_augmentations)
    elif method == 'parallel':
        return augment_data_parallel(x, y)
    elif method == 'custom':
        return custom_augmentation_strategy(x, y)
    else:
        raise ValueError("Unknown augmentation method")

```

# Transformation-Based Methods

- Transformation-Based Methods: Magnitude and Time

| Magnitude         | Time               |
|-------------------|--------------------|
| Jittering         | Permutation        |
| Rotation          | Window Slicing     |
| Scaling           | Time Warping       |
| Magnitude Warping | Window Warping     |

- Pattern-Based: Not used

| Magnitude | Time                             |
|-----------|----------------------------------|
|           | SPAWNER                          |
|           | WDBA                             |
|           | Random Guided Warp               |
|           | Random Guided Warp Shape         |
|           | Discriminative Guided Warp       |
|           | Discriminative Guided Warp Shape |

# Selected datasets from the 2018 UCR Time Series Classification Archive

| Dataset         | Type      | Train | Test | Class | Length |
|-----------------|-----------|-------|------|-------|--------|
| CBF             | Simulated | 30    | 900  | 3     | 128    |
| ECG200          | ECG       | 100   | 100  | 2     | 96     |
| ECG5000         | ECG       | 500   | 4500 | 5     | 140    |
| FordB           | Sensor    | 3636  | 810  | 2     | 500    |
| GunPointAgeSpan | Motion    | 135   | 316  | 2     | 150    |
| ScreenType      | Device    | 375   | 375  | 3     | 720    |
| Strawberry      | Spectro   | 613   | 370  | 2     | 235    |
| Yoga            | Image     | 300   | 3000 | 2     | 426    |



