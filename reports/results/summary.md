# Results — VB regression, leave-one-case-out CV

145 labelled runs, 16 folds (one held-out insert per fold). CNN: mean ± std over 3 seeds.

## Headline

| Inputs | Model | MAE (mm) | RMSE (mm) | R² |
|---|---|---|---|---|
| process_time | gbr | 0.078 | 0.124 | 0.772 |
| sensor_process (raw signals) | cnn | 0.107 ± 0.002 | 0.152 ± 0.006 | 0.658 ± 0.027 |
| sensor_process | ridge | 0.126 | 0.177 | 0.533 |
| sensor_only | ridge | 0.136 | 0.204 | 0.381 |
| — (predict mean) | mean | 0.200 | 0.264 | -0.037 |
| process_only | ridge | 0.200 | 0.266 | -0.052 |

## All classical experiments

| Feature set | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| process_only | mean | 0.200 | 0.264 | -0.037 |
| process_only | ridge | 0.200 | 0.266 | -0.052 |
| process_only | rf | 0.212 | 0.292 | -0.264 |
| process_only | svr | 0.213 | 0.293 | -0.275 |
| process_only | gbr | 0.223 | 0.304 | -0.374 |
| process_time | gbr | 0.078 | 0.124 | +0.772 |
| process_time | svr | 0.081 | 0.132 | +0.740 |
| process_time | rf | 0.087 | 0.147 | +0.681 |
| process_time | ridge | 0.133 | 0.199 | +0.412 |
| process_time | mean | 0.200 | 0.264 | -0.037 |
| sensor_only | ridge | 0.136 | 0.204 | +0.381 |
| sensor_only | svr | 0.137 | 0.198 | +0.416 |
| sensor_only | rf | 0.137 | 0.209 | +0.355 |
| sensor_only | gbr | 0.143 | 0.216 | +0.310 |
| sensor_only | mean | 0.200 | 0.264 | -0.037 |
| sensor_process | ridge | 0.126 | 0.177 | +0.533 |
| sensor_process | svr | 0.128 | 0.188 | +0.477 |
| sensor_process | rf | 0.137 | 0.208 | +0.357 |
| sensor_process | gbr | 0.143 | 0.219 | +0.291 |
| sensor_process | mean | 0.200 | 0.264 | -0.037 |
