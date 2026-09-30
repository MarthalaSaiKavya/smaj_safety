# SMAJ v4 results summary

Generated 2026-09-30 14:38. Every number below is computed by smaj_v4.py from the saved episodes; see the E*.json files for the full statistics.

## E0 Behaviour-level reanalysis of the v3 closed loop

| condition | success | 95% CI | RMST (steps) | mean steps (successes) | distractor lifted | success if target lifted |
|---|---|---|---|---|---|---|
| clean | 58/60 |  [0.89, 0.99] | 109.3 | 103.5 | 0/60 | 1.00 |
| covered | 10/60 |  [0.09, 0.28] | 254.8 | 129.0 | 7/60 | 0.56 |
| screened | 33/60 |  [0.42, 0.67] | 192.8 | 121.5 | 0/60 | 0.62 |
| painted | 44/60 |  [0.61, 0.83] | 156.9 | 112.1 | 5/60 | 0.96 |
| restore_mlp | 59/60 |  [0.91, 1.00] | 107.5 | 104.6 | 0/60 | 1.00 |
| restore_attr | 8/60 |  [0.07, 0.24] | 262.3 | 147.0 | 6/60 | 0.33 |
| restore_random | 8/60 |  [0.07, 0.24] | 259.9 | 129.0 | 7/60 | 0.44 |
| inject_attr | 60/60 |  [0.94, 1.00] | 104.9 | 104.9 | 0/60 | 1.00 |
| restore_sel | 11/60 |  [0.11, 0.30] | 254.4 | 140.5 | 7/60 | 0.55 |

Per-task distractor lifting: covered: task 2 7; painted: task 2 5; restore_attr: task 2 6; restore_random: task 2 7; restore_sel: task 2 7

- covered_vs_clean: success diff -0.80 [-0.97, -0.60], RMST diff 145.5 [102.3, 178.9] steps, jointly successful pairs 10 (completion diff 18.4 [0.6, 57.5], conditional)
- screened_vs_clean: success diff -0.42 [-0.67, -0.18], RMST diff 83.5 [32.2, 138.8] steps, jointly successful pairs 32 (completion diff 15.0 [3.8, 44.1], conditional)
- painted_vs_clean: success diff -0.23 [-0.53, -0.05], RMST diff 47.5 [12.5, 103.0] steps, jointly successful pairs 42 (completion diff 6.3 [1.4, 10.6], conditional)
- restore_mlp_vs_clean: success diff 0.02 [-0.05, 0.10], RMST diff -1.8 [-14.8, 8.9] steps, jointly successful pairs 57 (completion diff 0.5 [0.1, 0.9], conditional)
- restore_attr_vs_clean: success diff -0.83 [-0.97, -0.70], RMST diff 152.9 [125.2, 180.4] steps, jointly successful pairs 8 (completion diff 37.1 [0.5, 75.0], conditional)
- restore_random_vs_clean: success diff -0.83 [-0.98, -0.65], RMST diff 150.5 [114.7, 176.0] steps, jointly successful pairs 8 (completion diff 25.8 [2.0, 63.3], conditional)
- inject_attr_vs_clean: success diff 0.03 [0.00, 0.10], RMST diff -4.5 [-15.1, 1.1] steps, jointly successful pairs 58 (completion diff 0.7 [0.5, 1.1], conditional)
- restore_sel_vs_clean: success diff -0.78 [-0.95, -0.60], RMST diff 145.1 [101.8, 178.6] steps, jointly successful pairs 11 (completion diff 32.0 [3.6, 102.3], conditional)
- Reproducibility vs the archived v3 episodes: 540/540 same outcome, 540/540 same length.

## E2 Restoration controls under the box occluder (paired with v3)

| condition | success | 95% CI | recovered fraction |
|---|---|---|---|
| clean | 58/60 |  [0.89, 0.99] | -- |
| covered | 10/60 |  [0.09, 0.28] | -- |
| restore_mlp | 59/60 |  [0.91, 1.00] | 1.02 [0.95, 1.16] |
| restore_attr | 8/60 |  [0.07, 0.24] | -0.04 [-0.24, 0.06] |
| restore_random | 8/60 |  [0.07, 0.24] | -0.04 [-0.15, 0.04] |
| inject_attr | 60/60 |  [0.94, 1.00] | -- |
| restore_sel | 11/60 |  [0.11, 0.30] | 0.02 [0.00, 0.06] |
| restore_mlp_L0_13 | 58/60 |  [0.89, 0.99] | 1.00 [0.93, 1.14] |
| restore_sel_top100 | 10/60 |  [0.09, 0.28] | 0.00 [0.00, 0.00] |
| restore_attr_k300 | 18/60 |  [0.20, 0.43] | 0.17 [0.04, 0.36] |
| restore_attr_k1000 | 18/60 |  [0.20, 0.43] | 0.17 [-0.14, 0.37] |
| restore_random_k1000 | 12/60 |  [0.12, 0.32] | 0.04 [-0.05, 0.15] |
| restore_random_k300 | 8/60 |  [0.07, 0.24] | -0.04 [-0.15, 0.04] |
- restore_attr_k1000_minus_restore_random_k1000: 0.10 [-0.13, 0.30] (n=60)
- restore_attr_k300_minus_restore_random_k300: 0.17 [0.03, 0.30] (n=60)
- restore_mlp_L0_13_minus_restore_mlp: -0.02 [-0.05, 0.00] (n=60)
- restore_sel_top100_minus_restore_attr: 0.03 [-0.05, 0.15] (n=60)

## E4 Gripper and speed features (chosen on discovery episodes, tested on held-out ones)

- gripper-closure set (16 features): held-out AUROC for 'closes within 16 steps' 1.00 vs proprioception 1.00 vs both 1.00
- speed sets: held-out corr with future EE speed fast 0.73, slow -0.79; current-speed baseline 0.41; partial corr given current speed + task: fast 0.70

## E5 Steering vs prompting vs action scaling

| comparison | Δ success | Δ mean speed (m/s) | Δ first close (steps) | Δ unwanted contact |
|---|---|---|---|---|
| prompt_slow_vs_base | -0.020 [-0.120, 0.060] | -0.006 [-0.011, -0.000] | 0.120 [-0.060, 0.300] | 0.060 [-0.040, 0.140] |
| act_scale_vs_base | -0.760 [-0.980, -0.520] | -0.097 [-0.106, -0.088] | 4.560 [0.000, 10.640] | 0.220 [-0.060, 0.440] |
| speed_ablate_vs_base | 0.000 [-0.060, 0.060] | -0.001 [-0.004, 0.003] | 0.040 [0.000, 0.080] | 0.000 [-0.060, 0.060] |
| speed_boost_vs_base | 0.040 [0.000, 0.080] | -0.000 [-0.007, 0.004] | -0.060 [-0.260, 0.140] | 0.000 [-0.060, 0.060] |
| speed_random_vs_base | 0.000 [0.000, 0.000] | -0.002 [-0.004, -0.000] | -0.020 [-0.060, 0.000] | 0.020 [-0.060, 0.120] |
| grip_ablate_vs_base | 0.000 [-0.060, 0.060] | -0.003 [-0.006, 0.000] | 0.000 [0.000, 0.000] | 0.040 [0.000, 0.120] |
| grip_boost_vs_base | 0.000 [-0.060, 0.060] | -0.004 [-0.008, -0.000] | -0.240 [-0.620, 0.180] | 0.100 [0.000, 0.260] |
| grip_random_vs_base | 0.020 [0.000, 0.060] | -0.002 [-0.008, 0.003] | 0.760 [0.160, 1.660] | 0.020 [-0.040, 0.080] |
| covered_prompt_slow_vs_covered | 0.020 [-0.040, 0.080] | -0.001 [-0.007, 0.004] | 0.420 [0.120, 0.720] | 0.040 [0.000, 0.080] |
| covered_act_scale_vs_covered | -0.120 [-0.220, -0.020] | -0.036 [-0.053, -0.020] | 9.940 [1.940, 17.940] | -0.260 [-0.580, 0.000] |
| speed_ablate_vs_speed_random | 0.000 [-0.060, 0.060] | 0.001 [-0.003, 0.005] | 0.060 [0.020, 0.100] | -0.020 [-0.120, 0.080] |
| grip_boost_vs_grip_random | -0.020 [-0.080, 0.040] | -0.002 [-0.010, 0.006] | -1.000 [-1.560, -0.540] | 0.080 [-0.040, 0.220] |
| speed_ablate_vs_prompt_slow | 0.020 [-0.040, 0.080] | 0.005 [0.000, 0.010] | -0.080 [-0.220, 0.080] | -0.060 [-0.140, 0.020] |
| speed_ablate_vs_act_scale | 0.760 [0.540, 0.940] | 0.096 [0.089, 0.105] | -4.520 [-10.560, 0.040] | -0.220 [-0.480, 0.060] |
| speed_boost_vs_prompt_slow | 0.060 [0.000, 0.140] | 0.006 [0.000, 0.012] | -0.180 [-0.360, 0.060] | -0.060 [-0.140, 0.040] |
| covered_prompt_slow_vs_covered_act_scale | 0.140 [0.040, 0.240] | 0.036 [0.023, 0.052] | -9.520 [-17.600, -1.440] | 0.300 [0.020, 0.600] |

Manipulation check: {"grip_boost": {"set": "grip", "relative_intervention_norm": 0.3091164999757994, "later_probe_mean_img_max": 0.09847207359513846, "base_later_probe_mean_img_max": 0.09541752605211165}, "grip_ablate": {"set": "grip", "relative_intervention_norm": 0.0015270481711547673, "later_probe_mean_img_max": 0.09427173343251695, "base_later_probe_mean_img_max": 0.09541752605211165}, "speed_ablate": {"set": "speed_fast", "relative_intervention_norm": 0.004423587916977657, "later_probe_mean_img_max": 0.11447495532385159, "base_later_probe_mean_img_max": 0.1150780694008747}, "speed_boost": {"set": "speed_slow", "relative_intervention_norm": 0.1492343997797914, "later_probe_mean_img_max": 0.08002915650409656, "base_later_probe_mean_img_max": 0.08441053374455525}, "grip_random": {"set": "grip", "relative_intervention_norm": 0.3046896468366242, "later_probe_mean_img_max": 0.09097506840019437, "base_later_probe_mean_img_max": 0.09541752605211165}, "speed_random": {"set": "speed_fast", "relative_intervention_norm": 0.0005485046510399931, "later_probe_mean_img_max": 0.11447196322788686, "base_later_probe_mean_img_max": 0.1150780694008747}}

**Decision:** {"gripper_steering_beyond_random": true, "speed_steering_slows_beyond_random": false, "speed_steering_slows_vs_base": false, "prompt_slows_vs_base": true, "action_scaling_slows_vs_base": true, "rule": "an effect counts when the 95% task-clustered CI of the paired difference excludes 0 in the stated direction"}
