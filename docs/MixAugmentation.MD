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


# Classification
1. FCNN

1.1. Parallel

| Technique Identifier   | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|------------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| adp_combined3  | 72.35%       | 88.05%       | 95.44%       | 3.75          | 13.08         | 22.61         |
| adp_combined5  | 71.11%       | 85.06%       | 95.14%       | 2.77          | 11.08         | 17.70         |
| adp_combined11 | 70.25%       | 83.39%       | 94.00%       | 132.30        | 177.79        | 246.08        |
| adp_combined1  | 71.11%       | 83.65%       | 93.80%       | 149.48        | 199.10        | 224.16        |
| adp_combined2  | 69.75%       | 85.77%       | 93.78%       | 132.76        | 170.88        | 197.19        |
| adp_combined12 | 71.11%       | 84.96%       | 93.78%       | 106.74        | 197.54        | 223.68        |
| adp_combined4  | 70.74%       | 86.66%       | 93.56%       | 139.16        | 212.74        | 299.61        |
| adp_combined10 | 70.86%       | 86.98%       | 93.51%       | 135.97        | 182.06        | 202.49        |
| adp_combined7  | 71.73%       | 85.92%       | 93.51%       | 4.72          | 11.75         | 17.12         |
| adp_combined6  | 71.36%       | 84.96%       | 93.24%       | 133.14        | 176.49        | 249.82        |
| adp_combined8  | 70.86%       | 85.28%       | 92.97%       | 120.99        | 151.20        | 213.84        |
| adp_combined9  | 70.00%       | 85.50%       | 92.70%       | 125.80        | 213.67        | 313.11        |


1.2. Sequential

| Technique Identifier   | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|------------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| ads_combined5  | 67.90%       | 86.74%       | 95.67%       | 3.01           | 32.69          | 47.05          |
| ads_combined12 | 70.37%       | 86.62%       | 95.56%       | 171.52         | 214.34         | 245.50         |
| ads_combined4  | 68.52%       | 83.88%       | 94.59%       | 165.14         | 267.26         | 317.88         |
| ads_combined7  | 69.26%       | 86.51%       | 94.44%       | 3.08           | 32.18          | 47.82          |
| ads_combined3  | 70.49%       | 86.33%       | 94.05%       | 9.70           | 39.79          | 63.62          |
| ads_combined10 | 68.40%       | 84.94%       | 93.78%       | 131.56         | 205.17         | 235.41         |
| ads_combined11 | 69.01%       | 86.53%       | 93.67%       | 175.78         | 332.24         | 401.30         |
| ads_combined9  | 67.53%       | 86.51%       | 93.00%       | 131.90         | 290.12         | 325.36         |
| ads_combined8  | 68.02%       | 86.45%       | 93.00%       | 144.83         | 258.62         | 354.40         |
| ads_combined6  | 68.52%       | 86.03%       | 93.00%       | 170.96         | 318.31         | 383.09         |
| ads_combined1  | 70.37%       | 83.06%       | 92.97%       | 137.35         | 262.16         | 421.69         |
| ads_combined2  | 68.40%       | 84.84%       | 92.27%       | 172.04         | 267.72         | 350.43         |



1.3.Sequential Magnitude and Time

1.3.1. Simple

1.3.1.1. vf
| Technique Identifier   | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|------------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| ads_magnitude_uniq1     | 69.38%       | 86.74%       | 93.78%       | 1.65          | 17.16         | 29.63         |
| ads_magnitude_uniq3     | 70.62%       | 85.83%       | 92.82%       | 2.22          | 5.88          | 8.87          |
| ads_magnitude_uniq2     | 71.73%       | 85.94%       | 92.69%       | 1.00          | 5.12          | 7.91          |
| ads_magnitude_uniq4     | 70.74%       | 86.41%       | 92.58%       | 154.34        | 380.23        | 546.50        |
| ads_time_uniq2          | 64.57%       | 85.61%       | 94.00%       | 5.52          | 34.25         | 52.98         |
| ads_time_uniq1          | 73.21%       | 85.80%       | 93.51%       | 11.62         | 101.39        | 169.52        |
| ads_time_uniq3          | 70.37%       | 82.81%       | 93.51%       | 153.97        | 420.80        | 551.79        |
| ads_time_uniq4          | 67.41%       | 85.26%       | 92.42%       | 5.76          | 63.23         | 102.42        |


1.3.1.1. overall (-2 for a visual test)

| Technique Identifier   | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|------------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| ads_magnitude_uniq1     | 67.38%       | 84.74%       | 91.78%       | 1.65          | 17.16         | 29.63         |
| ads_magnitude_uniq3     | 68.62%       | 83.83%       | 90.82%       | 2.22          | 5.88          | 8.87          |
| ads_magnitude_uniq2     | 69.73%       | 83.94%       | 90.69%       | 1.00          | 5.12          | 7.91          |
| ads_magnitude_uniq4     | 68.74%       | 84.41%       | 90.58%       | 154.34        | 380.23        | 546.50        |
| ads_time_uniq2          | 62.57%       | 83.61%       | 92.00%       | 5.52          | 34.25         | 52.98         |
| ads_time_uniq1          | 71.21%       | 83.80%       | 91.51%       | 11.62         | 101.39        | 169.52        |
| ads_time_uniq3          | 68.37%       | 80.81%       | 91.51%       | 153.97        | 420.80        | 551.79        |
| ads_time_uniq4          | 65.41%       | 83.26%       | 90.42%       | 5.76          | 63.23         | 102.42        |


1.3.3. Multi



1.3.Parallel Magnitude and Time

1.3.1. Simple

| Technique Identifier     | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|--------------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| adp_magnitude_uniq3       | 69.51%       | 82.22%       | 93.69%       | 1.10          | 24.58         | 15.24         |
| adp_magnitude_uniq2       | 71.36%       | 82.62%       | 92.89%       | 1.77          | 50.71         | 428.09        |
| adp_magnitude_uniq1       | 69.51%       | 81.79%       | 93.00%       | 1.92          | 52.62         | 214.20        |
| adp_magnitude_uniq4       | 70.86%       | 81.83%       | 92.67%       | 137.45        | 208.20        | 428.09        |
| adp_time_uniq4            | 70.86%       | 82.96%       | 96.44%       | 2.39          | 65.28         | 449.74        |
| adp_time_uniq2            | 71.36%       | 82.89%       | 94.56%       | 3.15          | 30.15         | 72.11         |
| adp_time_uniq1            | 70.12%       | 82.58%       | 93.78%       | 2.53          | 40.63         | 235.90        |
| adp_time_uniq3            | 70.25%       | 82.40%       | 93.78%       | 4.64          | 113.99        | 449.74        |


1.3.3. Multi

