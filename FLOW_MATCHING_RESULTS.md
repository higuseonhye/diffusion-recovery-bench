# Flow matching vs DDIM

Generated: 2026-10-16T14:34:23+09:00

Same 1D temporal denoiser. DDIM regresses noise along a cosine schedule. Flow matching regresses the straight-line velocity from noise to the action chunk and integrates it with Euler steps. Behavior numbers use the saved DDIM checkpoint and a flow model trained for the same epoch budget on the same synthetic corridors.

## Latency, action head only

Random weights, batch 1, action dim 7, observation dim 128, horizon 16. 100 Hz requires P99 < 10 ms. 50 Hz requires P99 < 20 ms. One Euler step and one DDIM step are both a single forward pass.

| sampler | steps | p50_ms | p99_ms | 100Hz | 50Hz |
| --- | --- | --- | --- | --- | --- |
| ddim | 1 | 1.03 | 1.56 | pass | pass |
| ddim | 2 | 2.00 | 26.13 | fail | fail |
| ddim | 4 | 3.99 | 4.94 | pass | pass |
| ddim | 10 | 8.65 | 11.31 | fail | pass |
| ddim | 16 | 13.14 | 16.14 | fail | pass |
| flow | 1 | 0.62 | 1.31 | pass | pass |
| flow | 2 | 1.68 | 2.22 | pass | pass |
| flow | 4 | 2.65 | 4.31 | pass | pass |
| flow | 10 | 7.37 | 9.45 | pass | pass |
| flow | 16 | 11.27 | 13.88 | fail | pass |

## Undisturbed corridors

32 stochastic samples from the origin. Success means the goal ball is reached and the disk is missed.

| sampler | steps | collision | success | median_error_cm | left | right | collapsed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ddim | 2 | 31.2% | 0.0% | 19.23 | 40.6% | 50.0% | 9.4% |
| ddim | 4 | 21.9% | 43.8% | 2.03 | 43.8% | 53.1% | 3.1% |
| ddim | 10 | 9.4% | 90.6% | 0.40 | 53.1% | 43.8% | 3.1% |
| flow | 1 | 100.0% | 0.0% | 1.01 | 25.0% | 0.0% | 75.0% |
| flow | 2 | 43.8% | 56.2% | 0.61 | 46.9% | 37.5% | 15.6% |
| flow | 4 | 18.8% | 81.2% | 0.46 | 56.2% | 40.6% | 3.1% |

## 6 cm jump

Same seeds as the diffusion recovery track (`seed + 1000`).

| sampler | steps | mode | T_exec | success | collision | median_error_cm |
| --- | --- | --- | --- | --- | --- | --- |
| ddim | 2 | CONTINUE | 16 | 0.0% | 37.5% | 18.57 |
| ddim | 2 | REPLAN | 4 | 0.0% | 34.4% | 33.57 |
| ddim | 2 | REPLAN | 8 | 3.1% | 53.1% | 21.61 |
| ddim | 10 | CONTINUE | 16 | 0.0% | 50.0% | 5.69 |
| ddim | 10 | REPLAN | 4 | 78.1% | 21.9% | 1.46 |
| ddim | 10 | REPLAN | 8 | 56.2% | 43.8% | 1.47 |
| flow | 1 | CONTINUE | 16 | 0.0% | 3.1% | 6.51 |
| flow | 1 | REPLAN | 4 | 96.9% | 3.1% | 1.28 |
| flow | 1 | REPLAN | 8 | 96.9% | 3.1% | 1.33 |
| flow | 2 | CONTINUE | 16 | 0.0% | 50.0% | 6.22 |
| flow | 2 | REPLAN | 4 | 84.4% | 15.6% | 1.31 |
| flow | 2 | REPLAN | 8 | 59.4% | 40.6% | 1.35 |
| flow | 4 | CONTINUE | 16 | 0.0% | 50.0% | 6.00 |
| flow | 4 | REPLAN | 4 | 81.2% | 18.8% | 0.94 |
| flow | 4 | REPLAN | 8 | 59.4% | 40.6% | 1.25 |

## Notes

- Latency does not measure trajectory quality. A 2-step sampler is fast whether or not the chunk clears the disk.
- The earlier expectation that flow matching at 2 steps would land near 2.5 ms restates the measured DDIM K=2 cost. This file is the direct Euler timing.
- 100 Hz is a 10 ms control period. It describes this action head, not a camera encoder plus a robot.
- The planar point mass, the 4 cm disk, and the instantaneous 6 cm jump are unchanged. Isaac Sim was not run.
- Flow at 4 Euler steps was scored for the 6 cm jump afterward, on the saved `results/flow_policy.pt` weights and the same trial seeds.
- This is a software benchmark, not a surgical device.
