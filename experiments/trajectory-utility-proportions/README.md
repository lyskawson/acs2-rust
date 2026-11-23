# Trajectory utility proportions

The draft in `plan.json` pins the original S+1 and trajectory-utility registrations.
The six new arms cross Mean value and Random goal with
`--tu-share 0.25`, `--tu-share 0.75` and `--tu-share 1`.
The share is conditional on requesting a copy. The original-goal probability stays
one third. The other two requested probabilities appear beside the realized shares
and fallbacks in the row diagnostics. The final grid labels append the fraction,
for example `acs2tuher_mean_value_p1of4`. The final plan records the mapping from
the submission labels in the pilot. Share one disables hindsight requests.
Share one half uses the original route stream and row schema. Share zero runs S+1.

For the registered maze pair protocols, `--second-test-cap 100` adds
`test_caps.registered` and `test_caps.extended` at each checkpoint.
Each extended rollout takes the test pair and the evaluation RNG state from before
the corresponding registered rollout. Maze pair reset and transitions are deterministic.
The added fields record weighted success, the mean shortest path and the weighted
mean steps with failures charged the applicable cap. Their ratio is the ratio of
the two means. Extended pair records follow the registered test pair order.
The cumulative thread CPU timer covers the alternate environment, RNG forks,
rollouts and their per-pair attribution. The ordinary evaluation, including its
registered output fields, retains its stream and configuration.

## Cost pilot

`pilot-cells.json` fixes the seed-42 checkpoints and bounds for the 168 cells.
Pin the dedicated cluster checkout and run `pilot.py build --commit COMMIT`.
After the musl build completes, run `pilot.py submit --wave 0`. The worker derives
the same manifest from the pinned registration, validates the allocation and
checks every checkpoint. `pilot.py status --wave 0` reads allocation accounting.
`pilot.py seal --wave 0` seals terminal accounting. After a STOP, a subsequent wave
uses the same seed, binary and targets for unfinished cells and compares preserved
checkpoints in binary64 bits. The pilot ceiling includes the build, spent CPU and
remaining reservations across waves. The complete seal covers the originals and
accounting for all attempts.

`cost_forecast.py` accepts the sealed pilot and the original
`trajectory-utility-measures` archive. It writes cost-only tables with matched
checkpoint ratios, extended test CPU, memory, per-seed projections and proposed
limits. It uses all twenty archived seeds at one half for each measure and maze.
The MazeE2 and MazeE3 envelopes require full-budget seed-42 scouts before the
dependent seeds can start. Reused full-budget pilot runs remain in the 3,360
cell forecast and are excluded from the forecast of remaining work.

`regression.py` takes the original ablation and 6a analysis indices for short
identity checks. It verifies the source checksums and compares original pair
records. It also verifies all existing presets with the second test cap toggled.
`analysis_rules.py` selects the share carried into adaptive replay from validation
only. The full grid requires an explicit approval and CPU-hour limit.
