# Result Classification

> for optimal computation and performance, prioritizing accuracy is important, but efficiency should also be balanced to ensure faster, consistent results. The refined ranking order would be:

> - Max Accuracy: Prioritize highest potential performance.
> - Avg Accuracy: Ensure consistent performance.
> - Min Time (ms): Favor the fastest execution.
> - Avg Time (ms): Ensure overall efficiency.
> - Min Accuracy: Ensure the worst-case scenario is still acceptable.
> - Max Time (ms): Control variability in execution time.

> This order balances both accuracy and computational efficiency, making it more suitable for performance-critical applications.

> Max Accuracy;Avg Accuracy;Min Time (ms);Avg Time (ms);Min Accuracy;Max Time (ms)
——————————————————————————
## 1.   Simple

| Technique Identifier     | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|--------------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| original                 | 70.74%       | 86.16%       | 92.84%       | 0.00          | 0.00          | 0.00          |
| jitter                   | 83.37%       | 89.25%       | 93.71%       | 1.80          | 3.53          | 11.22         |
| scaling                  | 83.80%       | 89.46%       | 92.84%       | 1.20          | 3.99          | 14.10         |
| permutation              | 84.70%       | 88.08%       | 92.73%       | 7.16          | 25.96         | 249.97        |
| randomperm               | 84.20%       | 88.70%       | 92.70%       | 8.40          | 27.56         | 310.69        |
| magwarp                  | 69.75%       | 88.94%       | 93.80%       | 264.10        | 404.68        | 813.67        |
| timewarp                 | 69.26%       | 87.08%       | 93.00%       | 128.59        | 303.15        | 813.67        |
| windowslice              | 67.53%       | 88.38%       | 93.60%       | 4.25          | 31.68         | 94.48         |
| windowwarp               | 70.99%       | 89.59%       | 92.64%       | 7.11          | 21.70         | 163.01        |
| rotation                 | 81.50%       | 88.53%       | 92.73%       | 1.26          | 7.34          | 18.45         |

——————————————————————————
## 2.   Sequential Magnitude & Time
### 2.1. Unique

| Technique Identifier     | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|--------------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| magnitude_uniq3           | 71.36%       | 89.10%       | 93.71%       | 1.68          | 91.90         | 2488.52       |
| magnitude_uniq2           | 70.99%       | 90.91%       | 92.16%       | 1.68          | 3.97          | 607.00        |
| magnitude_uniq1           | 70.62%       | 87.33%       | 92.71%       | 2.85          | 14.84         | 425.78        |
| magnitude_uniq4           | 70.99%       | 85.29%       | 91.00%       | 5.53          | 840.14        | 2488.52       |
| time_uniq4                | 67.04%       | 79.63%       | 94.56%       | 4.40          | 126.67        | 2744.76       |
| time_uniq3                | 67.04%       | 86.57%       | 93.78%       | 16.97         | 208.95        | 689.32        |
| time_uniq2                | 62.96%       | 85.05%       | 92.70%       | 4.40          | 50.64         | 1016.16       |
| time_uniq1                | 62.96%       | 85.92%       | 93.02%       | 10.20         | 76.09         | 1016.16       |

### 2.2. Multiple

| Technique Identifier     | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|--------------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| magnitude_multi3          | 71.48%       | 87.10%       | 93.76%       | 118.78        | 257.41        | 851.92        |
| magnitude_multi4          | 71.48%       | 86.77%       | 92.69%       | 118.78        | 276.01        | 851.92        |
| magnitude_multi2          | 70.89%       | 86.62%       | 93.60%       | 118.78        | 240.09        | 851.92        |
| magnitude_multi1          | 67.04%       | 83.64%       | 91.00%       | 118.78        | 273.51        | 851.92        |
| time_multi3               | 67.04%       | 88.22%       | 93.78%       | 172.86        | 301.93        | 1379.30       |
| time_multi2               | 67.04%       | 88.16%       | 93.78%       | 151.75        | 301.93        | 1379.30       |
| time_multi4               | 67.04%       | 87.25%       | 93.24%       | 136.18        | 296.60        | 1379.30       |
| time_multi1               | 67.04%       | 87.04%       | 92.70%       | 136.18        | 307.42        | 1379.30       |

——————————————————————————

## 3.   Parallel Magnitude & Time
### 3.1. Unique

### 3.2. Multiple

——————————————————————————

## 4.   Combined Magnitude & Time
### 4.1.    Sequential

| Technique Identifier          | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|-------------------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| sequential_combined3          | 83.33%       | 89.96%       | 94.59%       | 5.62          | 122.02        | 240.21        |
| sequential_combined10         | 78.93%       | 87.97%       | 92.56%       | 9.42          | 175.22        | 432.56        |
| sequential_combined5          | 68.15%       | 87.50%       | 93.47%       | 9.97          | 141.83        | 374.04        |
| sequential_combined7          | 68.77%       | 87.46%       | 92.51%       | 9.42          | 163.73        | 400.22        |
| sequential_combined1          | 67.16%       | 83.47%       | 93.24%       | 10.06         | 159.17        | 470.08        |
| sequential_combined8          | 67.16%       | 81.44%       | 93.24%       | 10.06         | 183.63        | 470.08        |
| sequential_combined9          | 67.16%       | 81.40%       | 93.24%       | 14.51         | 180.06        | 470.08        |
| sequential_combined12         | 67.16%       | 80.33%       | 93.24%       | 12.53         | 161.17        | 470.08        |

### 4.1.    Parallel

| Technique Identifier        | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|-----------------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| adp_parallel_combined12     | 69.14%       | 82.61%       | 94.59%       | 2.62          | 243.76        | 615.83        |
| adp_parallel_combined10     | 69.14%       | 82.52%       | 94.00%       | 2.62          | 242.96        | 615.83        |
| adp_parallel_combined6      | 69.14%       | 81.98%       | 93.80%       | 2.62          | 241.84        | 615.83        |
| adp_parallel_combined1      | 69.14%       | 81.88%       | 93.78%       | 2.62          | 243.02        | 615.83        |
| adp_parallel_combined8      | 69.14%       | 81.71%       | 93.78%       | 2.62          | 245.83        | 615.83        |
| adp_parallel_combined3      | 69.14%       | 81.35%       | 93.51%       | 2.62          | 241.99        | 615.83        |
| adp_parallel_combined5      | 69.14%       | 81.28%       | 93.51%       | 2.62          | 244.25        | 615.83        |
| adp_parallel_combined7      | 69.14%       | 81.01%       | 93.51%       | 2.62          | 244.78        | 615.83        |
