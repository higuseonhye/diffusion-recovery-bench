# Diffusion recovery benchmark results

Generated: 2026-10-16T14:04:17+09:00

Synthetic proxy for action-chunk diffusion latency, multimodal corridor selection, and receding-horizon recovery after a 6 cm lateral jump. The action head met the 20 ms / 50 Hz P99 budget on 3 of 5 DDIM step settings. MLP collision rate was 100.0%; diffusion collision rate was 9.4%.

## Environment

| python | torch | numpy | device | torch threads | platform |
| --- | --- | --- | --- | --- | --- |
| 3.14.3 | 2.13.0+cpu | 2.4.3 | cpu | 6 | Windows-11-10.0.26200-SP0 |

## Task

| start_m | goal_m | obstacle_radius_m | goal_tol_m | perturbation_m | disturb_after_steps |
| --- | --- | --- | --- | --- | --- |
| [0.00, 0.00] | [0.24, 0.00] | 0.04 | 0.02 | 0.06 | 4 |

Dataset windows: 2555 (nominal corridors 800, recovery bridges 1755). Diffusion epsilon loss 0.5533 -> 0.0326. MLP MSE 0.7483 -> 0.1638. Trained for 50 epochs, diffusion schedule length 50.

## Exp 1 — DDIM latency

Batch 1. P99 is over the timed runs after warmup. 50 Hz requires P99 < 20 ms. 20 Hz requires P99 < 50 ms. Weights are randomly initialized; latency does not depend on fitting the corridors.

| condition | K | executed | p50_ms | p95_ms | p99_ms | 50Hz | 20Hz |
| --- | --- | --- | --- | --- | --- | --- | --- |
| action_head | 2 | 2 | 1.88 | 2.40 | 2.48 | pass | pass |
| action_head | 5 | 5 | 4.19 | 5.44 | 5.81 | pass | pass |
| action_head | 10 | 10 | 8.43 | 9.86 | 10.96 | pass | pass |
| action_head | 16 | 16 | 12.86 | 16.99 | 29.98 | fail | pass |
| action_head | 50 | 50 | 40.10 | 48.34 | 49.89 | fail | pass |
| eye_and_action_head | 2 | 2 | 4.18 | 4.91 | 5.03 | pass | pass |
| eye_and_action_head | 5 | 5 | 6.33 | 7.74 | 8.03 | pass | pass |
| eye_and_action_head | 10 | 10 | 10.59 | 12.54 | 12.83 | pass | pass |
| eye_and_action_head | 16 | 16 | 16.99 | 18.78 | 18.89 | pass | pass |
| eye_and_action_head | 50 | 50 | 44.16 | 51.61 | 53.27 | fail | fail |

## Exp 2 — Multimodal separation

Same start state, two demonstration sides. MLP regresses the chunk with MSE. Diffusion draws 32 DDIM samples (K=10). A collapsed path is one whose mean y stays inside ±1 cm.

| method | trials | collision | success | median_error_cm | left | right | collapsed | median_clearance_cm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp_bc | 1 | 100.0% | 0.0% | 0.08 | 0.0% | 0.0% | 100.0% | -3.55 |
| diffusion | 32 | 9.4% | 90.6% | 0.40 | 53.1% | 43.8% | 3.1% | 4.04 |

## Exp 3 — 6 cm jump, CONTINUE vs REPLAN

The same diffusion policy is executed two ways. After 4 steps the state jumps +6 cm in y. CONTINUE keeps the original chunk. REPLAN reads the new state every T_exec steps. Median steps and the 20 Hz time column are computed on successes only.

| mode | T_exec | trials | success | collision | median_error_cm | median_steps | median_s_at_20hz |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CONTINUE | 16 | 32 | 0.0% | 50.0% | 5.69 |  |  |
| REPLAN | 4 | 32 | 78.1% | 21.9% | 1.46 | 11.0 | 0.55 |
| REPLAN | 8 | 32 | 56.2% | 43.8% | 1.47 | 12.0 | 0.60 |

## Notes

- The planar task is a point mass in meters. Isaac Sim, contact dynamics, and a 7-DoF arm were not executed.
- Exp 1 times a randomly initialized temporal-conv denoiser (action dim 7, observation dim 128, horizon 16). The eye row adds a random 128x128 conv stub. Neither row is the surgical-vision serving benchmark.
- Exp 2 and Exp 3 train that same denoiser family on synthetic left/right corridors. The MLP baseline is MSE regression on the same chunks.
- CONTINUE denoises once and integrates that chunk. REPLAN denoises again every T_exec steps from the measured state. The 6 cm event is an instantaneous +y state jump after 4 steps.
- Success means the final point is inside the 2 cm goal ball and the polyline never intersects the 4 cm obstacle.
- steps_to_goal is a control-cycle count. The 20 Hz seconds column multiplies that count by 50 ms; it is not a measured wall-clock duration.
- This is a software benchmark, not a clinical validation and not a surgical device.
