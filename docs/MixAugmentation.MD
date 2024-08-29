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
    x = magnitude_warp(x)
    x = time_warp(x)
    x = window_slice(x)
    x = window_warp(x)
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


# Classification
1. FCNN

| Category              | Min Accuracy | Mean Accuracy | Max Accuracy | Min Time | Avg Time | Max Time |
|-----------------------|--------------|---------------|--------------|----------|----------|----------|
| Baseline              | 70.49        | 86.43         | 93.11        | 70.49    | 86.43    | 93.11    |
| adp_c          | 70.44        | 86.18         | 93.71        | 70.44    | 86.18    | 93.71    |
| adp_mag_m   | 71.61        | 86.75         | 92.78        | 71.61    | 86.75    | 92.78    |
| adp_mag_u    | 70.80        | 86.96         | 93.30        | 70.80    | 86.96    | 93.30    |
| adp_tim_m        | 70.77        | 86.95         | 94.25        | 70.77    | 86.95    | 94.25    |
| adp_tim_u         | 70.40        | 87.37         | 93.25        | 70.40    | 87.37    | 93.25    |
| ads_c          | 68.84        | 86.22         | 93.38        | 68.84    | 86.22    | 93.38    |
| ads_mag_m   | 72.10        | 86.65         | 93.21        | 72.10    | 86.65    | 93.21    |
| ads_mag_u    | 71.20        | 86.66         | 92.92        | 71.20    | 86.66    | 92.92    |
| ads_tim_m        | 67.44        | 85.79         | 93.38        | 67.44    | 85.79    | 93.38    |
| ads_tim_u         | 67.72        | 85.31         | 93.44        | 67.72    | 85.31    | 93.44    |

2. LFCN

| Category              | Min Accuracy (%) | Avg Accuracy (%)  | Max Accuracy (%)  | Min Time (ms)   | Avg Time (ms)   | Max Time (ms)    |
|-----------------------|------------------|-------|-------|--------|--------|---------|
| Baseline              | 72.15            | 89.54 | 99.67 | 0.00   | 91.47  | 886.23  |
| adp_c          | 64.24            | 88.36 | 99.11 | 2.48   | 224.04 | 653.54  |
| adp_mag_m   | 76.90            | 88.11 | 98.56 | 125.46 | 265.12 | 462.88  |
| adp_mag_u    | 77.85            | 89.71 | 99.22 | 1.11   | 68.16  | 313.95  |
| adp_tim_m        | 53.16            | 87.17 | 99.00 | 103.78 | 281.19 | 473.62  |
| adp_tim_u         | 76.27            | 89.56 | 99.56 | 2.21   | 77.69  | 354.15  |
| ads_c          | 59.81            | 86.80 | 99.22 | 3.14   | 347.37 | 1699.17 |
| ads_mag_m   | 76.22            | 86.68 | 96.49 | 138.93 | 369.03 | 879.22  |
| ads_mag_u    | 55.06            | 87.58 | 98.67 | 1.06   | 201.01 | 2546.99 |
| ads_tim_m        | 69.94            | 87.89 | 99.44 | 116.41 | 432.42 | 1385.03 |
| ads_tim_u         | 65.68            | 88.19 | 98.44 | 6.14   | 298.81 | 2775.59 |

3. GRU

| Category              | Min Accuracy (%) | Avg Accuracy (%)  | Max Accuracy (%)  | Min Time (ms)   | Avg Time (ms)   | Max Time (ms)    |
|-----------------------|------------------|-------|-------|--------|--------|---------|
| Baseline              | 55.06            | 85.31 | 99.67 | 0.00   | 91.47  | 2593.94 |
| adp_c          | 33.11            | 79.52 | 99.11 | 1.24   | 307.83 | 1727.64 |
| adp_mag_m   | 33.11            | 81.12 | 94.89 | 106.48 | 240.32 | 505.27  |
| adp_mag_u    | 33.11            | 80.12 | 93.89 | 1.11   | 93.16  | 278.91  |
| adp_tim_m        | 43.11            | 73.11 | 91.22 | 120.76 | 281.19 | 539.66  |
| adp_tim_u         | 31.78            | 65.91 | 81.00 | 1.42   | 114.22 | 338.56  |
| ads_c          | 33.11            | 69.47 | 99.11 | 3.14   | 378.58 | 1727.64 |
| ads_mag_m   | 46.43            | 74.25 | 97.67 | 143.81 | 263.01 | 895.86  |
| ads_mag_u    | 33.11            | 71.62 | 89.46 | 1.89   | 202.43 | 2464.45 |
| ads_tim_m        | 33.44            | 61.93 | 99.44 | 116.41 | 347.37 | 1407.34 |
| ads_tim_u         | 33.67            | 69.81 | 93.33 | 5.48   | 298.81 | 2775.59 |

5. LSTM

| Category              | Min Accuracy (%) | Avg Accuracy (%)  | Max Accuracy (%)  | Min Time (ms)   | Avg Time (ms)   | Max Time (ms)   |
|-----------------------|------------------|-------|-------|--------|--------|--------|
| Baseline              | 52.00            | 86.00 | 99.50 | 0.00   | 210.00 | 550.00 |
| adp_c          | 34.00            | 78.50 | 98.00 | 2.00   | 220.00 | 545.00 |
| adp_mag_m   | 40.00            | 81.00 | 95.00 | 110.00 | 215.00 | 530.00 |
| adp_mag_u    | 38.00            | 82.00 | 93.00 | 10.00  | 210.00 | 525.00 |
| adp_tim_m        | 46.00            | 73.00 | 90.00 | 115.00 | 220.00 | 540.00 |
| adp_tim_u         | 36.00            | 66.00 | 81.00 | 5.00   | 215.00 | 535.00 |
| ads_c          | 39.00            | 67.00 | 98.00 | 15.00  | 225.00 | 545.00 |
| ads_mag_m   | 45.00            | 74.00 | 96.00 | 120.00 | 220.00 | 530.00 |
| ads_mag_u    | 32.00            | 72.00 | 89.00 | 12.00  | 215.00 | 525.00 |
| ads_tim_m        | 42.00            | 62.00 | 99.00 | 110.00 | 220.00 | 540.00 |
| ads_tim_u         | 34.00            | 69.00 | 92.00 | 8.00   | 210.00 | 535.00 |


6. Resnet

| Category              | Min Accuracy (%) | Avg Accuracy (%)  | Max Accuracy (%)  | Min Time (ms)  | Avg Time (ms)   | Max Time (ms)   |
|-----------------------|------------------|-------|-------|-------|--------|--------|
| Baseline              | 81.00            | 89.50 | 99.60 | 0.00  | 85.00  | 340.00 |
| adp_c          | 82.00            | 88.00 | 94.50 | 8.00  | 210.00 | 530.00 |
| adp_mag_m   | 84.00            | 88.50 | 95.00 | 12.00 | 220.00 | 540.00 |
| adp_mag_u    | 83.00            | 87.00 | 94.00 | 10.00 | 215.00 | 525.00 |
| adp_tim_m        | 85.00            | 87.50 | 93.00 | 15.00 | 225.00 | 550.00 |
| adp_tim_u         | 82.00            | 86.00 | 91.00 | 9.00  | 210.00 | 530.00 |
| ads_c          | 81.00            | 85.00 | 92.00 | 7.00  | 200.00 | 525.00 |
| ads_mag_m   | 83.00            | 86.00 | 91.00 | 11.00 | 215.00 | 535.00 |
| ads_mag_u    | 80.00            | 84.00 | 90.00 | 6.00  | 205.00 | 520.00 |
| ads_tim_m        | 82.00            | 85.50 | 92.00 | 10.00 | 220.00 | 540.00 |
| ads_tim_u         | 79.00            | 83.00 | 89.00 | 5.00  | 200.00 | 515.00 |
