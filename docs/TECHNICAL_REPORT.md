# Technical report: planar action-chunk benchmark

This is an engineering report of one CPU benchmark. It records what the code measured, how the units should be read, and which operating point the numbers support.

It is not a product requirements document. It is not a manuscript, and it is not evidence for a clinical claim.

The run logs are [BENCHMARK_RESULTS.md](../BENCHMARK_RESULTS.md) and [FLOW_MATCHING_RESULTS.md](../FLOW_MATCHING_RESULTS.md). This file is the reading of those logs.

## 1. Scope

The plant is a point mass in a plane. The state is position in meters. A policy emits a chunk of 16 future positions, expressed as offsets from the position at which it was sampled. CONTINUE integrates that chunk and does not look again. REPLAN samples a new chunk every 4 or 8 executed steps.

The obstacle is a disk. Two demonstration corridors pass on opposite sides of it. Their average goes through the disk. After four steps the plant can be moved by a fixed lateral jump.

Isaac Sim, contact, a robot arm, and a camera were not run. The latency rows time a small temporal convolution on this CPU. One row also times a small random image network in front of that convolution. Neither row is the serving measurement in `surgical-vision-bench`.

## 2. Units

The stored geometry is:

| Quantity | Meters | Millimeters |
| --- | --- | --- |
| Path, origin to goal | 0.24 | 240 |
| Lateral jump | 0.06 | 60 |
| Obstacle radius | 0.04 | 40 |
| Goal ball radius | 0.02 | 20 |
| Corridor peak | 0.09 | 90 |

A label of 6 cm is `0.06 m × 100`. That is a conversion from meters to centimeters. The slider and the figures use it. Nothing in the plant is magnified for display, and the network was not trained in a hidden millimeter frame that the plots then scaled up.

The jump is 25% of the 240 mm path (`0.06 / 0.24`). On this figure that is a large disturbance. It is not a microsurgical amplitude.

The network never sees centimeters. Observations and action chunks are standardized with the training mean and standard deviation before the loss, and decoded back to meters afterward. Because of that standardization, the success and collision rates are unchanged if every length in the task is multiplied by the same factor. The physical millimeters change. The rates do not transfer if only the jump is reduced and the path, disk, and goal ball stay at the recorded size. That experiment was not run.

An illustrative uniform scale, not a second run, sets the path to 12 mm. `similitude()` in `diffusion_bench/task.py` applies that factor to every length:

| Quantity | Recorded run | Same geometry on a 12 mm path |
| --- | --- | --- |
| Path | 240 mm | 12 mm |
| Jump | 60 mm | 3.0 mm |
| Obstacle radius | 40 mm | 2.0 mm |
| Goal ball radius | 20 mm | 1.0 mm |
| Corridor peak | 90 mm | 4.5 mm |

12 mm is an example tool-tip travel chosen for this note. It is not an anatomical measurement. A 0.6 mm jump would be a hundred times smaller than the recorded 60 mm jump, and this repository does not contain that run.

## 3. What the plane is showing

The drawing is a top view of the point mass, not a photograph of an instrument.

The mass starts at the origin and is scored at the goal. The disk sits on the straight line between them. Training trajectories go above the disk or below it. A sample that stays within 10 mm of `y = 0` is counted as collapsed.

Mean-squared regression on those two families returns their average. On the recorded run that average reaches the goal (median error 0.08 cm) and intersects the disk (clearance −3.6 cm). Collision rate for that single deterministic chunk is 100%. Success, which requires both the goal ball and a miss of the disk, is 0%.

A diffusion sampler does not emit that average when it is given enough denoising steps. At 10 DDIM steps, 32 draws from the origin split 53% above and 44% below, with a 9% collision rate and a 91% success rate. Median clearance is +4.0 cm.

The jump is applied to the plant after four executed steps, along `+y`. CONTINUE keeps integrating the chunk chosen at the origin, so the jump remains in the final position. REPLAN reads the new position on its next cycle and samples again. For 10-step DDIM, CONTINUE success is 0% with median error 5.7 cm. REPLAN every 4 steps succeeds on 78% of trials with median error 1.5 cm. REPLAN every 8 steps succeeds on 56%.

The 4.3 ms figure below is not the duration of that recovery. It is the 99th-percentile time to evaluate the action head four times. The recovery rates are a separate count of geometric success.

## 4. Sampler comparison

Both samplers use the same temporal-convolution backbone. DDIM is trained to predict the noise added by a cosine schedule. Flow matching is trained to predict the straight-line velocity from a noise sample to the action chunk, and the chunk is produced by Euler steps. The flow model was trained for the same 50 epochs. Latency is a batch-1 timing of a randomly initialized head with action dimension 7, observation dimension 128, and horizon 16. Behavior uses the saved planar checkpoint and the flow checkpoint trained in that comparison. The 4-step flow recovery row was measured afterward on `results/flow_policy.pt` with the same trial seeds.

| Sampler | Steps | Action-head P99 | Origin success | 60 mm jump, REPLAN every 4 steps |
| --- | --- | --- | --- | --- |
| DDIM | 10 | 11.3 ms | 91% | 78% |
| Flow | 4 | 4.3 ms | 81% | 81% |
| Flow | 2 | 2.2 ms | 56% | 84% |
| Flow | 1 | 1.3 ms | 0% | 97% |

One Euler step from the origin is the unresolved average of the two corridors: 75% of those samples are collapsed, the collision rate is 100%, and the median goal error is still only 1.0 cm. The same one-step field, after the jump has broken the left/right symmetry, succeeds on 97% of replans. Two Euler steps sit between those regimes. Four Euler steps keep origin success at 81% and recovery at 81%, with median recovery error 0.9 cm.

DDIM at 10 steps remains the highest origin success. Its action-head P99 of 11.3 ms fits a 20 ms period (50 Hz) and does not fit a 10 ms period (100 Hz). Four Euler steps fit the 10 ms period. The per-step cost of the two samplers is similar. What changes is how many steps are required before the sample stops taking the average corridor.

These times do not include a camera encoder from another process. The serving P99 in `surgical-vision-bench` is a different measurement and is not added to the numbers above.

## 5. Operating choice on this bench

If the control period is 20 ms and the start state is still the symmetric origin, 10-step DDIM is the measured default: 11.3 ms and 91% origin success.

If the period is 10 ms, 4-step flow matching is the measured candidate inside that budget: 4.3 ms, 81% origin success, 81% recovery. Moving from 10-step DDIM to 2-step flow matching saves latency and gives up origin success (91% to 56%). That trade is not justified by a 20 ms budget that 10-step DDIM already meets.

One Euler step is a recovery update for a state that is already off the center line. It is not a policy for the symmetric start.

## 6. Limits

The mass has no inertia, friction, or contact. The jump is inserted into the state. It is not a breath, a heartbeat, or a measured tissue motion.

A uniform rescaling, such as the 12 mm example above, preserves the rates only as a statement about this normalized geometry. It does not create a new log at that size.

The repository does not show that a foundation-model checkpoint can be reused across diffusion and flow matching. Octo publishes a diffusion action head, OpenVLA emits discrete action tokens, and π0 is trained with flow matching. Those are different interfaces.

Nothing here is a verification package for a surgical device.
