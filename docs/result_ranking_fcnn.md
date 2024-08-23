# Result Classification

> for optimal computation and performance, prioritizing accuracy is important, but efficiency should also be balanced to ensure faster, consistent results. The refined ranking order would be:

> - Max Accuracy: Prioritize highest potential performance.
> - Avg Accuracy: Ensure consistent performance.
> - Min Time (ms): Favor the fastest execution.
> - Avg Time (ms): Ensure overall efficiency.
> - Min Accuracy: Ensure the worst-case scenario is still acceptable.
> - Max Time (ms): Control variability in execution time.

> This order balances both accuracy and computational efficiency, making it more suitable for performance-critical applications.

> Highest (Max Accuracy;Avg Accuracy); Lowest (Min Time (ms);Avg Time (ms);Min Accuracy); Highest (Max Time (ms))
——————————————————————————
## 1.   Simple

### 1.1.    by accuracy and efficiency

| Technique Identifier | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| original             | 69.49%       | 88.61%       | 92.46%       | 0.00          | 0.00          | 0.00          |
| jitter               | 82.12%       | 88.00%       | 92.46%       | 1.80          | 3.53          | 11.22         |
| magwarp              | 68.50%       | 87.69%       | 92.55%       | 264.10        | 404.68        | 813.67        |
| windowwarp           | 69.74%       | 88.34%       | 91.39%       | 7.11          | 21.70         | 163.01        |
| scaling              | 82.55%       | 88.21%       | 91.59%       | 1.20          | 3.99          | 14.10         |
| permutation          | 83.45%       | 86.83%       | 91.48%       | 7.16          | 25.96         | 249.97        |
| rotation             | 80.25%       | 87.28%       | 91.48%       | 1.26          | 7.34          | 18.45         |
| windowslice          | 66.28%       | 87.13%       | 92.35%       | 4.25          | 31.68         | 94.48         |
| timewarp             | 68.01%       | 85.83%       | 91.75%       | 128.59        | 303.15        | 813.67        |


### 1.2.    by datasets

| Dataset          | Jitter | Scaling | Permutation | Magwarp | Timewarp | Windowslice | Windowwarp | Rotation | Original |
|------------------|--------|---------|-------------|---------|----------|-------------|------------|----------|----------|
| ECG5000          | 92.46% | 91.59%  | 91.48%      | 92.55%  | 91.24%   | 92.35%      | 91.39%     | 91.48%   | 91.59%   |
| Strawberry       | 89.29% | 91.45%  | 91.18%      | 89.29%  | 88.48%   | 90.91%      | 88.75%     | 91.45%   | 90.64%   |
| ECG200           | 90.75% | 89.75%  | 86.75%      | 91.75%  | 91.75%   | 88.75%      | 87.75%     | 89.75%   | 88.75%   |
| GunPointAgeSpan  | 87.99% | 85.14%  | 87.04%      | 88.31%  | 87.36%   | 87.36%      | 87.67%     | 87.36%   | 89.89%   |
| CBF              | 90.75% | 89.42%  | 83.97%      | 90.19%  | 87.08%   | 89.97%      | 90.42%     | 86.86%   | 90.75%   |
| Yoga             | 82.12% | 82.55%  | 83.45%      | 81.52%  | 79.55%   | 78.95%      | 83.08%     | 80.25%   | 81.88%   |
| FordB            | 68.38% | 71.47%  | 70.60%      | 68.50%  | 68.01%   | 66.28%      | 69.74%     | 70.23%   | 69.49%   |


——————————————————————————
## 2.   Sequential Magnitude & Time
### 2.1. Unique


| Technique Identifier       | Max Accuracy | Avg Accuracy | Min Time (ms) | Avg Time (ms) | Min Accuracy | Max Time (ms) |
|----------------------------|--------------|--------------|---------------|---------------|--------------|---------------|
| ads_mag_u3                 | 93.71%       | 89.10%       | 1.68          | 91.90         | 71.36%       | 2488.52       |
| ads_mag_u2                 | 92.16%       | 90.91%       | 1.68          | 3.97          | 70.99%       | 607.00        |
| ads_mag_u1                 | 92.71%       | 87.33%       | 2.85          | 14.84         | 70.62%       | 425.78        |
| ads_mag_u4                 | 91.00%       | 85.29%       | 5.53          | 840.14        | 70.99%       | 2488.52       |
|----------------------------|--------------|--------------|---------------|---------------|--------------|---------------|
| adp_tim_u4                 | 94.56%       | 79.63%       | 4.40          | 126.67        | 67.04%       | 2744.76       |
| adp_tim_u3                 | 93.78%       | 86.57%       | 16.97         | 208.95        | 67.04%       | 689.32        |
| adp_tim_u2                 | 92.70%       | 85.05%       | 4.40          | 50.64         | 62.96%       | 1016.16       |
| adp_tim_u1                 | 93.02%       | 85.92%       | 10.20         | 76.09         | 62.96%       | 1016.16       |

### 2.2. Multiple

| Technique Identifier      | Max Accuracy | Avg Accuracy | Min Time (ms) | Avg Time (ms) | Min Accuracy | Max Time (ms) |
|---------------------------|--------------|--------------|---------------|---------------|--------------|---------------|
| adp_tim_m3                | 93.78%       | 88.22%       | 172.86        | 301.93        | 67.04%       | 1379.30       |
| adp_tim_m2                | 93.78%       | 88.16%       | 151.75        | 301.93        | 67.04%       | 1379.30       |
| adp_tim_m4                | 93.24%       | 87.25%       | 136.18        | 296.60        | 67.04%       | 1379.30       |
| adp_tim_m1                | 92.70%       | 87.04%       | 136.18        | 307.42        | 67.04%       | 1379.30       |
|---------------------------|--------------|--------------|---------------|---------------|--------------|---------------|
| ads_mag_m3                | 93.76%       | 87.10%       | 118.78        | 257.41        | 71.48%       | 851.92        |
| ads_mag_m4                | 92.69%       | 86.77%       | 118.78        | 276.01        | 71.48%       | 851.92        |
| ads_mag_m2                | 93.60%       | 86.62%       | 118.78        | 240.09        | 70.89%       | 851.92        |
| ads_mag_m1                | 91.00%       | 83.64%       | 118.78        | 273.51        | 67.04%       | 851.92        |

——————————————————————————

## 3.   Parallel Magnitude & Time
### 3.1. Unique

| Technique Identifier       | Max Accuracy | Avg Accuracy | Min Time (ms) | Avg Time (ms) | Min Accuracy | Max Time (ms) |
|----------------------------|--------------|--------------|---------------|---------------|--------------|---------------|
| ads_tim_u1                 | 93.49%       | 85.30%       | 5.22          | 30.14         | 70.12%       | 454.30        |
| ads_tim_u3                 | 93.00%       | 85.87%       | 1.96          | 219.43        | 70.12%       | 689.32        |
| ads_tim_u2                 | 92.78%       | 85.75%       | 3.02          | 10.33         | 70.12%       | 300.70        |
| ads_tim_u4                 | 92.70%       | 85.30%       | 2.15          | 15.36         | 67.04%       | 2744.76       |
|----------------------------|--------------|--------------|---------------|---------------|--------------|---------------|
| adp_mag_u4                 | 93.96%       | 86.60%       | 5.53          | 686.56        | 70.62%       | 2488.52       |
| adp_mag_u3                 | 93.73%       | 85.89%       | 1.68          | 81.81         | 71.36%       | 2488.52       |
| adp_mag_u2                 | 92.80%       | 85.52%       | 1.19          | 4.68          | 70.86%       | 607.00        |
| adp_mag_u1                 | 92.71%       | 85.44%       | 2.85          | 13.94         | 69.88%       | 425.78        |

### 3.2. Multiple

| Technique Identifier       | Max Accuracy | Avg Accuracy | Min Time (ms) | Avg Time (ms) | Min Accuracy | Max Time (ms) |
|----------------------------|--------------|--------------|---------------|---------------|--------------|---------------|
| ads_tim_m1                 | 97.33%       | 89.79%       | 212.69        | 272.80        | 69.38%       | 515.99        |
| ads_tim_m3                 | 93.42%       | 90.08%       | 143.13        | 267.26        | 81.37%       | 527.47        |
| ads_tim_m4                 | 93.00%       | 89.54%       | 210.48        | 281.34        | 81.17%       | 546.27        |
| ads_tim_m2                 | 93.24%       | 88.85%       | 157.32        | 266.64        | 72.59%       | 546.27        |
|----------------------------|--------------|--------------|---------------|---------------|--------------|---------------|
| adp_mag_m4                 | 93.11%       | 88.42%       | 161.58        | 254.73        | 70.62%       | 432.14        |
| adp_mag_m2                 | 92.78%       | 88.41%       | 206.84        | 256.82        | 70.62%       | 432.14        |
| adp_mag_m1                 | 92.69%       | 88.10%       | 157.64        | 273.59        | 70.62%       | 432.14        |
| adp_mag_m3                 | 91.00%       | 87.35%       | 170.26        | 269.31        | 71.73%       | 432.14        |

——————————————————————————

## 4.   Combined Magnitude & Time
### 4.1.    Sequential

| Technique Identifier          | Max Accuracy | Avg Accuracy | Min Time (ms) | Avg Time (ms) | Min Accuracy | Max Time (ms) |
|-------------------------------|--------------|--------------|---------------|---------------|--------------|---------------|
| ads_c3                        | 94.59%       | 89.96%       | 5.62          | 122.02        | 83.33%       | 240.21        |
| ads_c5                        | 93.47%       | 87.50%       | 9.97          | 141.83        | 68.15%       | 374.04        |
| ads_c7                        | 92.51%       | 87.46%       | 9.42          | 163.73        | 68.77%       | 400.22        |
| ads_c10                       | 92.56%       | 87.97%       | 9.42          | 175.22        | 78.93%       | 432.56        |
| ads_c1                        | 93.24%       | 83.47%       | 10.06         | 159.17        | 67.16%       | 470.08        |
| ads_c8                        | 93.24%       | 81.44%       | 10.06         | 183.63        | 67.16%       | 470.08        |
| ads_c9                        | 93.24%       | 81.40%       | 14.51         | 180.06        | 67.16%       | 470.08        |
| ads_c12                       | 93.24%       | 80.33%       | 12.53         | 161.17        | 67.16%       | 470.08        |

### 4.2.    Parallel

| Technique Identifier        | Max Accuracy | Avg Accuracy | Min Time (ms) | Avg Time (ms) | Min Accuracy | Max Time (ms) |
|-----------------------------|--------------|--------------|---------------|---------------|--------------|---------------|
| adp_c12                     | 94.59%       | 82.61%       | 2.62          | 243.76        | 69.14%       | 615.83        |
| adp_c10                     | 94.00%       | 82.52%       | 2.62          | 242.96        | 69.14%       | 615.83        |
| adp_c6                      | 93.80%       | 81.98%       | 2.62          | 241.84        | 69.14%       | 615.83        |
| adp_c1                      | 93.78%       | 81.88%       | 2.62          | 243.02        | 69.14%       | 615.83        |
| adp_c8                      | 93.78%       | 81.71%       | 2.62          | 245.83        | 69.14%       | 615.83        |
| adp_c3                      | 93.51%       | 81.35%       | 2.62          | 241.99        | 69.14%       | 615.83        |
| adp_c5                      | 93.51%       | 81.28%       | 2.62          | 244.25        | 69.14%       | 615.83        |
| adp_c7                      | 93.51%       | 81.01%       | 2.62          | 244.78        | 69.14%       | 615.83        |

### 4.3.    Both

| Technique Identifier       | Max Accuracy | Avg Accuracy | Min Time (ms) | Avg Time (ms) | Min Accuracy | Max Time (ms) |
|----------------------------|--------------|--------------|---------------|---------------|--------------|---------------|
| ads_c3                     | 94.59%       | 89.96%       | 5.62          | 122.02        | 83.33%       | 240.21        |
| ads_c10                    | 92.56%       | 87.97%       | 9.42          | 175.22        | 78.93%       | 432.56        |
| ads_c5                     | 93.47%       | 87.50%       | 9.97          | 141.83        | 68.15%       | 374.04        |
| ads_c7                     | 92.51%       | 87.46%       | 9.42          | 163.73        | 68.77%       | 400.22        |
| ads_c1                     | 93.24%       | 83.47%       | 10.06         | 159.17        | 67.16%       | 470.08        |
| ads_c8                     | 93.24%       | 81.44%       | 10.06         | 183.63        | 67.16%       | 470.08        |
| ads_c9                     | 93.24%       | 81.40%       | 14.51         | 180.06        | 67.16%       | 470.08        |
| ads_c12                    | 93.24%       | 80.33%       | 12.53         | 161.17        | 67.16%       | 470.08        |
|----------------------------|--------------|--------------|---------------|---------------|--------------|---------------|
| adp_c12                    | 94.59%       | 82.61%       | 2.62          | 243.76        | 69.14%       | 615.83        |
| adp_c10                    | 94.00%       | 82.52%       | 2.62          | 242.96        | 69.14%       | 615.83        |
| adp_c6                     | 93.80%       | 81.98%       | 2.62          | 241.84        | 69.14%       | 615.83        |
| adp_c1                     | 93.78%       | 81.88%       | 2.62          | 243.02        | 69.14%       | 615.83        |
| adp_c8                     | 93.78%       | 81.71%       | 2.62          | 245.83        | 69.14%       | 615.83        |
| adp_c3                     | 93.51%       | 81.35%       | 2.62          | 241.99        | 69.14%       | 615.83        |
| adp_c5                     | 93.51%       | 81.28%       | 2.62          | 244.25        | 69.14%       | 615.83        |
| adp_c7                     | 93.51%       | 81.01%       | 2.62          | 244.78        | 69.14%       | 615.83        |

## 5. Ovverall
| Category               | Avg Max Accuracy | Avg Avg Accuracy | Avg Min Time (ms) | Avg Avg Time (ms) | Avg Min Accuracy | Avg Max Time (ms) |
|------------------------|------------------|------------------|-------------------|-------------------|------------------|-------------------|
| sequential_combined    | 94.59%           | 89.96%           | 5.62              | 122.02            | 83.33%           | 432.56            |
| parallel_combined      | 94.03%           | 82.42%           | 2.62              | 243.55            | 69.14%           | 615.83            |
| time_multi             | 93.63%           | 87.67%           | 149.24            | 301.72            | 67.04%           | 1379.30           |
| time_uniq              | 93.77%           | 84.79%           | 8.49              | 115.09            | 65.50%           | 1366.60           |
| magnitude_uniq         | 92.90%           | 88.66%           | 2.93              | 237.71            | 70.99%           | 1477.21           |
| magnitude_multi        | 92.76%           | 86.03%           | 118.78            | 261.25            | 70.48%           | 851.92            |

# Refinement
Abbreviations Explanation:
- adp: Augmented Data Parallel
- ads: Augmented Data Sequential
- tim: Time-based method
- mag: Magnitude-based method
- u: Unique technique (applies a single type of transformation)
- m: Multiple techniques combined
- c: Combined method (sequential or parallel)

## Table: Group | Enumerated Abbreviations

| **Group**                   | **Enumerated Abbreviations**                                           |
|-----------------------------|------------------------------------------------------------------------|
| **Sequential Methods**      | ads_c1, ads_c2, ads_c3, ads_c4, ads_c5, ads_c6,                        |
|                             | ads_c7, ads_c8, ads_c9, ads_c10, ads_c11, ads_c12                      |
| --------------------------- |------------------------------------------------------------------------|
| **Parallel Methods**        | adp_c1, adp_c2, adp_c3, adp_c4, adp_c5, adp_c6,                        |
|                             | adp_c7, adp_c8, adp_c9, adp_c10, adp_c11, adp_c12                      |
| --------------------------- |------------------------------------------------------------------------|
| **Time-based Methods**      | ads_tim_u1, ads_tim_u2, ads_tim_u3, ads_tim_u4, ads_tim_m1, ads_tim_m2,|
|                             | ads_tim_m3, ads_tim_m4, adp_tim_u1, adp_tim_u2, adp_tim_u3, adp_tim_u4,|
|                             | adp_tim_m1, adp_tim_m2, adp_tim_m3, adp_tim_m4                         |
| --------------------------- |------------------------------------------------------------------------|
| **Magnitude-based Methods** | ads_mag_u1, ads_mag_u2, ads_mag_u3, ads_mag_u4, ads_mag_m1, ads_mag_m2,|
|                             | ads_mag_m3, ads_mag_m4, adp_mag_u1, adp_mag_u2, adp_mag_u3, adp_mag_u4,|
|                             | adp_mag_m1, adp_mag_m2, adp_mag_m3, adp_mag_m4                        |
| --------------------------- |------------------------------------------------------------------------|

## Table: Group | Enumerated Abbreviations Parallel

| **Abbreviation**  | **Augmentations (Order is Important)**                          |
|-------------------|-----------------------------------------------------------------|
| **ads_c1**        | Permutation, Rotation, Time Warping, Scaling                    |
| **ads_c2**        | Permutation, Jittering, Time Warping, Magnitude Warping         |
| **ads_c3**        | Permutation, Scaling, Window Slicing, Rotation                  |
| **ads_c4**        | Permutation, Magnitude Warping, Window Slicing, Jittering       |
| **ads_c5**        | Window Slicing, Jittering, Window Warping, Scaling              |
| **ads_c6**        | Window Slicing, Rotation, Time Warping, Magnitude Warping       |
| **ads_c7**        | Window Slicing, Scaling, Window Warping, Jittering              |
| **ads_c8**        | Time Warping, Rotation, Window Warping, Scaling                 |
| **ads_c9**        | Time Warping, Jittering, Window Warping, Magnitude Warping      |
| **ads_c10**       | Time Warping, Scaling, Window Slicing, Rotation                 |
| **ads_c11**       | Time Warping, Rotation, Window Slicing, Magnitude Warping       |
| **ads_c12**       | Window Slicing, Jittering, Time Warping, Rotation               |
| **adp_c1**        | Permutation, Rotation, Time Warping, Scaling                    |
| **adp_c2**        | Permutation, Jittering, Time Warping, Magnitude Warping         |
| **adp_c3**        | Permutation, Scaling, Window Slicing, Rotation                  |
| **adp_c4**        | Permutation, Magnitude Warping, Window Slicing, Jittering       |
| **adp_c5**        | Window Slicing, Jittering, Window Warping, Scaling              |
| **adp_c6**        | Window Slicing, Rotation, Time Warping, Magnitude Warping       |
| **adp_c7**        | Window Slicing, Scaling, Window Warping, Jittering              |
| **adp_c8**        | Time Warping, Rotation, Window Warping, Scaling                 |
| **adp_c9**        | Time Warping, Jittering, Window Warping, Magnitude Warping      |
| **adp_c10**       | Time Warping, Scaling, Window Slicing, Rotation                 |
| **adp_c11**       | Time Warping, Rotation, Window Slicing, Magnitude Warping       |
| **adp_c12**       | Window Slicing, Jittering, Time Warping, Rotation               |
