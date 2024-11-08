# Result Classification

> for optimal computation and performance, prioritizing accuracy is important, but efficiency should also be balanced to ensure faster, consistent results. The refined ranking order would be:

> - Max Accuracy: Prioritize highest potential performance.
> - Avg Accuracy: Ensure consistent performance.
> - Min Time (ms): Favor the fastest execution.
> - Avg Time (ms): Ensure overall efficiency.
> - Min Accuracy: Ensure the worst-case scenario is still acceptable.
> - Max Time (ms): Control variability in execution time.

> This order balances both accuracy and computational efficiency, making it more suitable for performance-critical applications.

	1.	Max Accuracy: The primary criterion is to prioritize techniques with the highest Max Accuracy.
	2.	Avg Accuracy: Among techniques with the same Max Accuracy, those with higher Avg Accuracy are favored.
	3.	Min Time (ms): Techniques with lower Min Time are ranked higher when Max and Avg Accuracy are similar.
	4.	Avg Time (ms): If Min Time is also similar, Avg Time is the next criterion.
	5.	Min Accuracy: Lower Min Accuracy is less desirable, so techniques with higher Min Accuracy are prioritized later.
	6.	Max Time (ms): Finally, techniques with the lowest Max Time are ranked higher if all other factors are equal.

> Highest (Max Accuracy;Avg Accuracy); Lowest (Min Time (ms);Avg Time (ms);Min Accuracy); Highest (Max Time (ms))

——————————————————————————
## 1.   Simple

### 1.1.    by accuracy and efficiency

| Technique Identifier        | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|-----------------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| magwarp                     | 84.80%       | 93.63%       | 99.67%       | 178.97        | 384.69        | 870.38        |
| timewarp                    | 85.00%       | 91.20%       | 98.89%       | 117.53        | 298.23        | 886.23        |
| windowwarp                  | 78.89%       | 90.10%       | 97.84%       | 2.73          | 37.80         | 161.30        |
| windowslice                 | 76.79%       | 90.18%       | 97.30%       | 2.67          | 25.94         | 89.44         |
| original                    | 78.52%       | 89.94%       | 98.33%       | 0             | 0             | 0             |
| rotation                    | 78.78%       | 88.82%       | 98.33%       | 1.77          | 14.85         | 16.41         |
| jitter                      | 77.78%       | 88.76%       | 99.11%       | 2.16          | 13.00         | 49.08         |
| permutation                 | 79.26%       | 88.23%       | 97.03%       | 3.68          | 57.94         | 253.39        |
| scaling                     | 72.15%       | 89.94%       | 97.67%       | 1.22          | 5.46          | 13.87         |


### 1.2.    by datasets

| Dataset          | Original | Scaling | Permutation | Magwarp | Timewarp | Windowslice | Windowwarp | Rotation | Jitter |
|------------------|----------|---------|-------------|---------|----------|-------------|------------|----------|---------|
| CBF1             | 98.33%   | 97.67%  | 95.11%      | 99.67%  | 98.89%   | 98.78%      | 98.33%     | 78.78%   | 99.11%  |
| Strawberry1      | 97.30%   | 97.30%  | 97.03%      | 97.30%  | 97.03%   | 97.30%      | 97.84%     | 95.68%   | 96.22%  |
| ECG50001         | 91.71%   | 94.27%  | 94.04%      | 94.44%  | 92.53%   | 93.78%      | 94.16%     | 92.69%   | 93.11%  |
| GunPointAgeSpan1 | 97.47%   | 72.15%  | 81.01%      | 94.94%  | 87.34%   | 96.84%      | 83.86%     | 87.97%   | 89.24%  |
| ECG2001          | 90.00%   | 88.00%  | 87.00%      | 85.00%  | 90.00%   | 91.00%      | 91.00%     | 88.00%   | 88.00%  |
| Yoga1            | 85.00%   | 87.43%  | 85.93%      | 84.80%  | 85.00%   | 84.33%      | 86.53%     | 83.53%   | 84.37%  |
| FordB1           | 78.52%   | 78.77%  | 79.26%      | 79.14%  | 78.89%   | 76.79%      | 78.89%     | 78.89%   | 77.78%  |


——————————————————————————
## 2.   Sequential Magnitude & Time
### 2.1. Unique

| Technique Identifier | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| ads_tim_u4           | 65.68%       | 89.45%       | 98.44%       | 6.14          | 198.17        | 2775.59       |
| ads_tim_u1           | 78.15%       | 89.63%       | 96.76%       | 14.45         | 274.39        | 985.46        |
| ads_tim_u2           | 65.68%       | 89.08%       | 96.49%       | 6.14          | 107.58        | 729.87        |
| ads_tim_u3           | 65.68%       | 88.78%       | 94.44%       | 21.43         | 268.54        | 2775.59       |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| ads_mag_u3           | 55.06%       | 86.62%       | 98.67%       | 1.06          | 10.28         | 515.70        |
| ads_mag_u4           | 76.58%       | 89.79%       | 98.44%       | 6.51          | 615.14        | 2546.99       |
| ads_mag_u1           | 78.52%       | 90.36%       | 97.30%       | 3.17          | 44.92         | 187.28        |
| ads_mag_u2           | 80.12%       | 90.31%       | 96.22%       | 1.05          | 83.20         | 540.23        |




### 2.2. Multiple

| Technique Identifier | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| ads_tim_m3           | 76.42%       | 89.91%       | 99.44%       | 178.92        | 269.24        | 1363.32       |
| ads_tim_m4           | 78.00%       | 89.25%       | 99.11%       | 179.05        | 266.67        | 1372.43       |
| ads_tim_m1           | 78.15%       | 89.13%       | 97.44%       | 116.41        | 302.96        | 1228.20       |
| ads_tim_m2           | 69.94%       | 86.61%       | 97.44%       | 208.11        | 263.73        | 1385.03       |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| ads_mag_m3           | 77.65%       | 90.45%       | 95.57%       | 161.83        | 336.13        | 879.22        |
| ads_mag_m4           | 77.28%       | 89.64%       | 95.14%       | 135.37        | 308.04        | 774.47        |
| ads_mag_m1           | 76.22%       | 88.83%       | 95.68%       | 201.68        | 415.65        | 873.97        |
| ads_mag_m2           | 78.52%       | 89.04%       | 96.49%       | 138.93        | 330.73        | 857.19        |




——————————————————————————

## 3.   Parallel Magnitude & Time
### 3.1. Unique

| Technique Identifier | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| adp_mag_u1           | 78.48%       | 92.16%       | 99.22%       | 1.11          | 7.18          | 23.18         |
| adp_mag_u3           | 80.37%       | 91.02%       | 98.00%       | 1.36          | 10.32         | 21.87         |
| adp_mag_u4           | 77.85%       | 91.32%       | 97.89%       | 4.04          | 116.59        | 313.95        |
| adp_mag_u2           | 79.63%       | 90.63%       | 98.33%       | 1.68          | 10.82         | 16.12         |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| adp_tim_u3           | 78.89%       | 90.21%       | 99.56%       | 2.21          | 213.88        | 354.15        |
| adp_tim_u4           | 76.27%       | 88.59%       | 97.78%       | 6.52          | 43.07         | 252.36        |
| adp_tim_u1           | 79.38%       | 88.57%       | 98.78%       | 2.21          | 35.03         | 70.01         |
| adp_tim_u2           | 79.14%       | 88.50%       | 98.78%       | 2.21          | 15.66         | 28.41         |



### 3.2. Multiple

| Technique Identifier | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| adp_mag_m1           | 76.90%       | 88.71%       | 98.56%       | 125.46        | 241.79        | 318.27        |
| adp_mag_m2           | 77.53%       | 88.93%       | 96.49%       | 174.35        | 271.30        | 369.59        |
| adp_mag_m3           | 76.90%       | 88.90%       | 97.03%       | 153.39        | 233.42        | 330.78        |
| adp_mag_m4           | 77.53%       | 88.87%       | 98.11%       | 135.37        | 266.48        | 372.06        |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| adp_tim_m1           | 70.57%       | 84.09%       | 98.22%       | 175.43        | 270.80        | 490.64        |
| adp_tim_m2           | 77.65%       | 85.43%       | 97.67%       | 171.09        | 245.15        | 459.83        |
| adp_tim_m3           | 60.44%       | 85.66%       | 98.56%       | 156.31        | 262.02        | 470.64        |
| adp_tim_m4           | 53.16%       | 84.57%       | 99.00%       | 103.78        | 251.87        | 473.62        |



——————————————————————————
| Technique Identifier | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| adp_mag_u1           | 78.48%       | 92.16%       | 99.22%       | 1.11          | 7.18          | 23.18         |
| adp_mag_u3           | 80.37%       | 91.02%       | 98.00%       | 1.36          | 10.32         | 21.87         |
| adp_mag_u2           | 79.63%       | 90.63%       | 98.33%       | 1.68          | 10.82         | 16.12         |
| adp_mag_u4           | 77.85%       | 91.32%       | 97.89%       | 4.04          | 116.59        | 313.95        |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| adp_tim_u3           | 78.89%       | 90.21%       | 99.56%       | 2.21          | 213.88        | 354.15        |
| adp_tim_u1           | 79.38%       | 88.57%       | 98.78%       | 2.21          | 35.03         | 70.01         |
| adp_tim_u2           | 79.14%       | 88.50%       | 98.78%       | 2.21          | 15.66         | 28.41         |
| adp_tim_u4           | 76.27%       | 88.59%       | 97.78%       | 6.52          | 43.07         | 252.36        |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| ads_mag_u1           | 78.52%       | 90.36%       | 97.30%       | 3.17          | 44.92         | 187.28        |
| ads_mag_u2           | 80.12%       | 90.31%       | 96.22%       | 1.05          | 83.20         | 540.23        |
| ads_mag_u4           | 76.58%       | 89.79%       | 98.44%       | 6.51          | 615.14        | 2546.99       |
| ads_mag_u3           | 55.06%       | 86.62%       | 98.67%       | 1.06          | 10.28         | 515.70        |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| ads_tim_u1           | 78.15%       | 89.63%       | 96.76%       | 14.45         | 274.39        | 985.46        |
| ads_tim_u4           | 65.68%       | 89.45%       | 98.44%       | 6.14          | 198.17        | 2775.59       |
| ads_tim_u2           | 65.68%       | 89.08%       | 96.49%       | 6.14          | 107.58        | 729.87        |
| ads_tim_u3           | 65.68%       | 88.78%       | 94.44%       | 21.43         | 268.54        | 2775.59       |


| Technique Identifier | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| adp_mag_m1           | 76.90%       | 88.71%       | 98.56%       | 125.46        | 241.79        | 318.27        |
| adp_mag_m3           | 76.90%       | 88.90%       | 97.03%       | 153.39        | 233.42        | 330.78        |
| adp_mag_m2           | 77.53%       | 88.93%       | 96.49%       | 174.35        | 271.30        | 369.59        |
| adp_mag_m4           | 77.53%       | 88.87%       | 98.11%       | 135.37        | 266.48        | 372.06        |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| adp_tim_m3           | 60.44%       | 85.66%       | 98.56%       | 156.31        | 262.02        | 470.64        |
| adp_tim_m2           | 77.65%       | 85.43%       | 97.67%       | 171.09        | 245.15        | 459.83        |
| adp_tim_m4           | 53.16%       | 84.57%       | 99.00%       | 103.78        | 251.87        | 473.62        |
| adp_tim_m1           | 70.57%       | 84.09%       | 98.22%       | 175.43        | 270.80        | 490.64        |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| ads_mag_m3           | 77.65%       | 90.45%       | 95.57%       | 161.83        | 336.13        | 879.22        |
| ads_mag_m4           | 77.28%       | 89.64%       | 95.14%       | 135.37        | 308.04        | 774.47        |
| ads_mag_m1           | 76.22%       | 88.83%       | 95.68%       | 201.68        | 415.65        | 873.97        |
| ads_mag_m2           | 78.52%       | 89.04%       | 96.49%       | 138.93        | 330.73        | 857.19        |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| ads_tim_m3           | 76.42%       | 89.91%       | 99.44%       | 178.92        | 269.24        | 1363.32       |
| ads_tim_m4           | 78.00%       | 89.25%       | 99.11%       | 179.05        | 266.67        | 1372.43       |
| ads_tim_m1           | 78.15%       | 89.13%       | 97.44%       | 116.41        | 302.96        | 1228.20       |
| ads_tim_m2           | 69.94%       | 86.61%       | 97.44%       | 208.11        | 263.73        | 1385.03       |

——————————————————————————

## 4.   Combined Magnitude & Time

| Technique Identifier | Min Accuracy | Avg Accuracy | Max Accuracy | Min Time (ms) | Avg Time (ms) | Max Time (ms) |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| adp_c4               | 88.44%       | 94.65%       | 99.11%       | 2.48          | 176.12        | 251.91        |
| adp_c12              | 86.08%       | 92.72%       | 98.89%       | 7.61          | 188.12        | 234.44        |
| adp_c3               | 86.08%       | 91.35%       | 99.11%       | 7.61          | 189.38        | 251.91        |
| adp_c10              | 87.00%       | 90.59%       | 98.56%       | 7.61          | 157.39        | 251.91        |
| adp_c7               | 81.65%       | 90.35%       | 98.56%       | 6.48          | 154.32        | 186.69        |
| adp_c1               | 84.00%       | 89.14%       | 99.22%       | 2.48          | 158.37        | 251.91        |
| adp_c2               | 79.14%       | 89.79%       | 98.56%       | 7.61          | 159.01        | 181.38        |
| adp_c11              | 88.44%       | 89.78%       | 98.89%       | 3.36          | 181.38        | 231.83        |
| adp_c8               | 78.44%       | 88.79%       | 97.56%       | 7.61          | 153.91        | 223.40        |
| adp_c6               | 85.13%       | 87.49%       | 96.76%       | 7.61          | 158.37        | 251.91        |
| adp_c5               | 64.24%       | 89.32%       | 97.78%       | 3.36          | 175.32        | 210.21        |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| ads_c7               | 85.13%       | 90.30%       | 99.22%       | 3.14          | 80.45         | 302.56        |
| ads_c4               | 72.15%       | 89.82%       | 99.00%       | 10.76         | 202.20        | 402.28        |
| ads_c5               | 75.95%       | 88.41%       | 98.56%       | 4.17          | 64.18         | 296.94        |
| ads_c12              | 76.00%       | 88.39%       | 98.89%       | 26.02         | 235.36        | 516.15        |
| ads_c9               | 69.62%       | 88.80%       | 98.22%       | 48.71         | 252.52        | 528.49        |
| ads_c8               | 73.42%       | 87.91%       | 97.33%       | 117.47        | 203.50        | 375.71        |
| ads_c10              | 80.80%       | 88.49%       | 90.51%       | 143.06        | 275.92        | 393.50        |
| ads_c3               | 66.77%       | 87.23%       | 95.95%       | 4.54          | 65.30         | 357.21        |
| ads_c11              | 62.00%       | 86.82%       | 96.33%       | 135.32        | 265.53        | 489.73        |
| ads_c6               | 62.00%       | 86.15%       | 95.95%       | 14.28         | 320.77        | 1564.93       |
| ads_c1               | 59.81%       | 85.43%       | 95.95%       | 3.14          | 158.26        | 1188.47       |
| ads_c2               | 61.08%       | 85.79%       | 95.68%       | 7.93          | 139.79        | 480.03        |

## 5. Ovverall

|------|--------------------------------|----------------------------------------------------------------------------|
| Rank | Technique Classifications      | Reasoning                                                                  |
|------|--------------------------------|----------------------------------------------------------------------------|
| 1    | ads_tim_m                      | Highest Max Accuracy (99.44%), high Avg Accuracy, reasonable time          |
|------|--------------------------------|----------------------------------------------------------------------------|
| 2    | adp_tim_u                      | Very high Max Accuracy (99.56%), strong Avg Accuracy, low time overhead    |
|------|--------------------------------|----------------------------------------------------------------------------|
| 3    | adp_c                          | High Max Accuracy (99.11%), very high Avg Accuracy, excellent time         |
|------|--------------------------------|----------------------------------------------------------------------------|
| 4    | ads_c                          | High Max Accuracy (99.22%), strong Avg Accuracy, low time                  |
|------|--------------------------------|----------------------------------------------------------------------------|
| 5    | adp_mag_u                      | High Max Accuracy (99.22%), high Avg Accuracy, lowest time                 |
|------|--------------------------------|----------------------------------------------------------------------------|
| 6    | ads_mag_u                      | High Max Accuracy (98.67%), reasonable time, strong Avg Accuracy           |
|------|--------------------------------|----------------------------------------------------------------------------|
| 7    | adp_mag_m                      | High Max Accuracy (98.56%), balanced Avg Accuracy, reasonable time         |
|------|--------------------------------|----------------------------------------------------------------------------|
| 8    | ads_mag_m                      | Strong Max Accuracy (95.57%), reasonable time, balanced Avg Accuracy       |
|------|--------------------------------|----------------------------------------------------------------------------|
| 9    | adp_tim_m                      | Good Max Accuracy (98.56%), decent Avg Accuracy, slightly higher time      |
|------|--------------------------------|----------------------------------------------------------------------------|
| 10   | ads_tim_u                      | Strong Max Accuracy (98.44%), balanced Avg Accuracy, moderate time         |
|------|--------------------------------|----------------------------------------------------------------------------|
| 11   | baseline                       | Solid performance but generally outclassed by specialized methods          |
|------|--------------------------------|----------------------------------------------------------------------------|
