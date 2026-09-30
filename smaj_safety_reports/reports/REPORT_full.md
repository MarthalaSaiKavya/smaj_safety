# SMAJ v4: full report on what we did and what we found

**Project:** interpretable features of the pi0.5 vision-language-action policy under visual occlusion: restoration, monitoring and steering.
**Policy / benchmark:** `lerobot/pi05_libero_finetuned` on LIBERO-Spatial (10 tasks), MuJoCo / robosuite.
**Date of the runs:** 29–30 September 2026 (Google Colab, A100). The E6 dictionary was trained in an earlier session of the same notebook.

Every number in this report was read from the result files in this repository or from logged notebook output. Each section names its source file. Where a result is missing or a planned step did not run, the report says so. Nothing has been estimated or filled in.

---

## 0. Where the results are in this repository

| Folder / file | What it holds |
|---|---|
| `smaj_v4/` | The final result set: E0, E2, E3, E4b, E5 and E7 JSONs, `SUMMARY.md`, the analysis config (`config_v4.json`), the frozen feature sets (`sets.json`, `discover/`), E3 episode metadata (`monitor/episodes.json`, `probe_layout.json`), figures for E3 and E7, and `paper_v4/` (auto-generated LaTeX tables and a draft). |
| `smaj_v4 2/` | Output of the GPU session that ran the rollouts: E0, E2, E4b and E5 JSONs (byte-identical to the copies in `smaj_v4/`), `run_v4.log`, `run_extra.log`, `smoke/report.json`, `status/*.json` for every step, `archived/goal3.json` (v3 closed-loop episodes used for the reproducibility check), and figures for E0, E2, E4 and E5. Its `config_v4.json` holds the **pre-specified** analysis settings. |
| `SMAJ_v4_final_experiments_(2) (2).ipynb` | The notebook with the E6 (8192-feature dictionary) training log in its output. |
| `SMAJ_v4_final_experiments_(2) (3).ipynb`, `... (3) copy.ipynb` | Later copies of the same notebook: extra experiments (E4b and the failed dose run), the quick report, the OpenCV fix, and the one-cell E3 analysis. |
| `reports/` | This report, the CoRL safety-workshop report, and copies of the figures. |

The per-episode arrays (`ep/*.npz`) and the videos are **not** in the repository. They are on Google Drive. All analyses below were run on them there, and only the summaries are here.

---

## 1. Setup shared by all experiments

**Policy.** pi0.5 (PaliGemma backbone plus a flow-matching action expert) as ported in LeRobot. The log confirms the policy details:

- bfloat16, action chunk 50, `n_action_steps` = 50, so each predicted chunk is executed open-loop in full;
- three image inputs;
- robot state is not fed to `sample_actions` (log line `state input to sample_actions: False`);
- `torch.compile` disabled so the forward hooks work.

**Where we look inside.** Hooks sit on the outputs of the 18 decoder-layer MLPs of the PaliGemma language model. This is the prefix that processes image and text tokens. It is not the action expert.

**Transcoders (v3).** TopK transcoders (4096 features, k = 64) were trained per layer on these MLPs in the earlier v3 pipeline (`tc_circuit_v3_main`). The v3 faithfulness check ("check 5") **failed** for this dictionary: median held-out FVU 0.204 against a threshold of 0.20, and 0.294 on held-out test frames.

**Tasks.** All LIBERO-Spatial tasks are "pick up the black bowl … and place it on the plate". Each scene contains two identical black bowls. The log identifies the target as `akita_black_bowl_1` and the **distractor** as `akita_black_bowl_2`. "Distractor lifted" therefore means the robot picked up the wrong bowl.

**Occluders.** Occluders are **render-only**: they are extra geoms drawn at render time, with no physics. The world and the dynamics are unchanged, so every behavioural change comes from what the cameras see. Five kinds were used:

| Name | What it is |
|---|---|
| `covered` (box) | Opaque gray box around the target. It follows the target and is visible to both cameras. |
| `screened` | Thin panel between the agent-view camera and the target. It hides the target from that camera only; the wrist camera still sees it. |
| `cover_miss` | The same box, displaced so it hides nothing of the target (checked with segmentation at t = 0). Controls for "a gray box is in the image". |
| `covered_static` | A larger box fixed in the world around the target's initial position. Its centre is randomly jittered so it does not give away the object position. |
| `painted` | v1/v2 condition: target pixels painted gray in both cameras. |

![conditions](figures/fig_conditions_task0.png)

*Task 0 at t = 0 under the five E3 conditions (snapshots from `smaj_v4/monitor/snapshots/`; images are shown as rendered).*

**Episodes.**

- Control runs at 20 Hz. The maximum is 280 steps (420 in E5).
- Each episode's seed is `seed + 100000 + 1000·task + init`. The same seed, initial state and noise are used across conditions, so all condition comparisons are **paired**.
- Success is LIBERO's success check.
- Behaviour metrics are logged per step: end-effector speed and path, gripper open/close onsets, target and distractor height, and MuJoCo contact forces between robot and movable non-target objects.
- **Unwanted contact** means contact force above a threshold. The pre-specified threshold is 1 N; it includes routine touches of the plate while placing.
- A **drop** means the target leaves a two-finger grasp and falls more than 3 cm.

**Statistics.**

- Success rates carry Wilson 95% CIs.
- Paired differences use a **task-clustered** bootstrap (5000 resamples). E0, E2 and E5 have 5 tasks, so these CIs are coarse.
- RMST is the restricted mean completion time, with failures counted at the step limit.
- McNemar exact tests are reported for success.
- Monitoring uses out-of-fold predictions with GroupKFold by (task, initial state), and the bootstrap resamples those groups.

---

## 2. E0: behaviour reanalysis of the v3 closed loop

**Source:** `smaj_v4/E0_behaviour_v3.json`. **Data:** the 540 archived v3 Goal-3 episodes: 5 tasks (0, 2, 6, 7, 9) × 12 initial states (0–11) × 9 conditions.

**Reproducibility.** A re-run of all 540 episodes matched the archived run in **540/540 outcomes and 540/540 episode lengths**. Rollouts are deterministic.

| Condition | Success | Wilson 95% | RMST (steps) | Distractor lifted |
|---|---|---|---|---|
| clean | 58/60 | [0.89, 0.99] | 109.4 | 0 |
| covered (box) | 10/60 | [0.09, 0.28] | 254.8 | 7 |
| screened | 33/60 | [0.42, 0.67] | 192.9 | 0 |
| painted | 44/60 | [0.61, 0.83] | 156.9 | 5 |
| restore_mlp (all 18 dense MLP outputs from the clean pass) | 59/60 | [0.91, 1.00] | 107.5 | 0 |
| restore_attr (100 attribution features) | 8/60 | [0.07, 0.24] | 262.3 | 6 |
| restore_random (norm-matched random features) | 8/60 | [0.07, 0.24] | 259.9 | 7 |
| restore_sel (129 selective features) | 11/60 | [0.11, 0.30] | 254.4 | 7 |
| inject_attr (clean scene; attribution features set to their *covered* values) | 60/60 | [0.94, 1.00] | 104.9 | 0 |

Paired differences against clean (task-clustered 95% CI):

| Comparison | Success difference | RMST difference (steps) |
|---|---|---|
| covered − clean | −0.80 [−0.97, −0.60] | +145.5 [102.3, 178.9] |
| screened − clean | −0.42 [−0.67, −0.18] | +83.5 [32.2, 138.8] |
| painted − clean | −0.23 [−0.53, −0.05] | +47.5 [12.5, 103.0] |

Paired differences against covered:

| Comparison | Success difference |
|---|---|
| restore_attr − covered | −0.03 [−0.15, 0.05] |
| restore_random − covered | −0.03 [−0.10, 0.03] |
| restore_sel − covered | +0.02 [0.00, 0.05] |
| restore_mlp − covered | +0.82 [0.70, 0.93] |

In these 5 tasks, distractor lifting happens **only in task 2**.

**Reading.**

- The box occluder is the strongest perturbation.
- Restoring the full dense MLP output recovers clean behaviour.
- Restoring the 100 attribution features or the 129 selective features does nothing measurable.
- Injecting the attribution features' "covered" values into a clean run does not hurt it (60/60). These features are neither sufficient to cause the failure nor, restored alone, enough to fix it.

![E0](figures/fig_E0_success_by_step.png)

---

## 3. E2: restoration controls under the box occluder

**Source:** `smaj_v4/E2_controls.json`, `run_v4.log`. **Design:** the same 5 tasks × initial states 0–11 as E0, paired with v3 Goal 3 (360 new episodes). The recovered fraction is (condition − covered) / (clean − covered).

| Condition | Set size | Success | RMST | Distractor lifted | Recovered fraction [95% CI] |
|---|---|---|---|---|---|
| clean (v3) | — | 58/60 | 109.4 | 0 | — |
| covered (v3) | — | 10/60 | 254.8 | 7 | 0 |
| restore_mlp, all 18 layers (v3) | dense | 59/60 | 107.5 | 0 | 1.02 [0.95, 1.16] |
| **restore_mlp_L0_13** (dense, layers 0–13 only) | dense | **58/60** | 110.3 | 0 | **1.00 [0.93, 1.14]** |
| restore_sel_top100 | 100 | 10/60 | 257.9 | 8 | 0.00 [0.00, 0.00] |
| restore_attr_k300 | 300 | 18/60 | 232.0 | 2 | 0.17 [0.04, 0.36] |
| restore_attr_k1000 | 1000 | 18/60 | 228.2 | 0 | 0.17 [−0.14, 0.37] |
| restore_random_k300 (norm-matched) | 300 | 8/60 | 262.6 | 8 | −0.04 [−0.15, 0.04] |
| restore_random_k1000 (norm-matched) | 1000 | 12/60 | 252.9 | 11 | 0.04 [−0.05, 0.15] |

Paired contrasts (task-clustered 95% CI):

| Contrast | Difference |
|---|---|
| attr_k300 − random_k300 | **+0.17 [0.03, 0.30]** |
| attr_k1000 − random_k1000 | +0.10 [−0.13, 0.30] |
| attr_k300 − covered | +0.13 [0.03, 0.23] (McNemar p = 0.096) |
| restore_mlp_L0_13 − restore_mlp | −0.02 [−0.05, 0.00] |
| restore_sel_top100 − restore_attr (100) | +0.03 [−0.05, 0.15] |

Per task, restore_attr_k300 succeeded 5/12, 0/12, 0/12, 5/12 and 8/12 on tasks 0, 2, 6, 7 and 9. On task 6 no transcoder-feature condition succeeded at all.

**Reading.**

- Dense restoration of layers 0–13 is enough to recover everything.
- Restoring 300 attribution-ranked transcoder features recovers about 17% of the gap, and beats norm-matched random features at k = 300.
- Going to 1000 features does not increase recovery, and the k = 1000 difference from random is not significant.
- Most of the causal signal is either spread over many features or sits in the transcoder error term.

![E2 dose](figures/fig_E2_dose.png)

---

## 4. E3: monitoring rollouts (behaviour + features), 600 episodes

**Source:** `smaj_v4/E3_monitoring.json`, `SUMMARY.md`, `status/analyze_monitor.json`. **Design:**

- 10 tasks × 5 conditions (clean, cover_miss, covered, covered_static, screened) × 12 initial states (12–23) = 600 episodes, all run to completion.
- Probes every 8 steps record:
  - pooled transcoder activations (agent-image mean, wrist-image mean, image max, language mean, language max) at layers 3, 6, 9, 12 and 15, all 4096 features;
  - a 2361-column feature bank for the pre-registered sets;
  - dense MLP activations;
  - privileged target visibility (segmentation pixel counts).

### 4.1 Behaviour under occlusion (all 10 tasks)

| Condition | Success | Wilson 95% | RMST | Distractor lifted | Unwanted contact >1 N | Grasped target | Close onsets / episode | Mean speed (m/s) | Path (m) |
|---|---|---|---|---|---|---|---|---|---|
| clean | 106/120 (0.88) | [0.81, 0.93] | 127.9 | 2 | 0.24 | 0.93 | 1.35 | 0.180 | 1.08 |
| cover_miss | 105/120 (0.88) | [0.80, 0.92] | 133.2 | 4 | 0.29 | 0.93 | 1.38 | 0.177 | 1.11 |
| covered | 14/120 (0.12) | [0.07, 0.19] | 260.4 | 25 | 0.67 | 0.29 | 2.64 | 0.124 | 1.54 |
| covered_static | 18/120 (0.15) | [0.10, 0.22] | 260.5 | 23 | 0.63 | 0.16 | 2.74 | 0.125 | 1.56 |
| screened | 43/120 (0.36) | [0.28, 0.45] | 221.6 | 7 | 0.50 | 0.58 | 2.66 | 0.141 | 1.47 |

Paired against clean (task-clustered bootstrap over 10 tasks):

| Comparison | Success difference | McNemar p | RMST difference (steps) |
|---|---|---|---|
| cover_miss − clean | −0.01 [−0.09, 0.05] | 1.0 | +5.3 [−6.1, 21.0] |
| covered − clean | −0.77 [−0.89, −0.63] | 1.2e−25 | +132.5 [105.4, 157.8] |
| covered_static − clean | −0.73 [−0.89, −0.53] | 6.5e−27 | +132.6 [98.2, 163.0] |
| screened − clean | −0.53 [−0.71, −0.35] | 1.9e−16 | +93.7 [63.3, 126.5] |

**Contact force.** The table gives the fraction of episodes whose maximum robot–object force exceeds each threshold, plus the median of that maximum.

| Condition | >1 N | >5 N | >10 N | >20 N | Median max force |
|---|---|---|---|---|---|
| clean | 0.24 | 0.24 | 0.18 | 0.02 | 0.0 N |
| cover_miss | 0.29 | 0.26 | 0.20 | 0.04 | 0.0 N |
| covered | 0.67 | 0.65 | 0.51 | 0.19 | 10.2 N |
| covered_static | 0.63 | 0.62 | 0.50 | 0.14 | 10.0 N |
| screened | 0.50 | 0.47 | 0.43 | 0.17 | 1.1 N |

**Where distractor lifting happens.** It occurs in tasks 1, 2 and 5, plus once in task 7:

| Condition | Episodes with distractor lifted, by task |
|---|---|
| covered | task 1: 7, task 2: 9, task 5: 9 |
| covered_static | task 1: 6, task 2: 9, task 5: 7, task 7: 1 |
| screened | task 1: 6, task 5: 1 |
| clean | task 5: 2 |
| cover_miss | task 5: 4 |

Task 5 is the hardest task even when clean (7/12). Success under the box stays non-trivial only in tasks 8 and 9 (5/12 each).

**Reading.**

- A render-only box over the target drops success from 88% to 12%.
- The box roughly triples the rate of contacts above 10 N (0.18 → 0.51).
- One episode in five ends up lifting the *other* bowl.
- The same box placed where it hides nothing (cover_miss) changes nothing. The effect comes from losing sight of the target, not from a box appearing in the image.
- Hiding the target from the third-person camera only (screened) is less harmful than hiding it from both cameras, but still costly.

### 4.2 Episode-level failure prediction

**Predictors.** All are cross-validated logistic regressions.

- **simple:** robot-available state at step τ: end-effector pose, displacement from start, gripper opening, recent speed, mean and absolute-mean of recent actions, and task one-hot.
- **Feature families:** the last probe plus the mean over probes up to τ. The families are:

| Family | Contents |
|---|---|
| cover_sel | 129 box-selective features |
| cover_attr | 100 attribution features |
| cover_sel_top100 | top-100 selective features |
| cover_sel_all | 452 features |
| screen_sel | 128 features |
| visibility | privileged segmentation visibility |
| sae_all | all 4096 × 5 layers, PCA-128 inside each fold |
| dense | dense MLP activations |

- **Random draws:** layer-matched random sets.
- **Label:** episode failure. 314 of the 600 episodes fail.

**Deviation from the pre-specified plan.**

| Setting | Pre-specified (`smaj_v4 2/config_v4.json`) | Actually run (FAST mode, `smaj_v4/config_v4.json`) |
|---|---|---|
| Observation windows | 0, 24, 48, 80 | 0 and 48 |
| C grid | 0.01 / 0.1 / 1 | fixed C = 0.1 |
| Bootstrap resamples | 1000 | 300 |
| Random draws per set actually evaluated | 5 | 2 (`rand_*_0`, `rand_*_1` in the E3 JSON; both configs still list `n_random_sets: 5`) |
| Hard-contact threshold for E7 | — | `hard_contact_n: 10` added |

The change was made because the full analysis ran at about 52 min per window on Colab. The primary window (48) and the decision rule were unchanged.

**Window 48 (primary; pre-specified), n = 600, 314 failures:**

| Predictor | AUROC [95% CI] | AUPRC |
|---|---|---|
| simple | 0.831 [0.786, 0.872] | 0.863 |
| cover_sel | 0.877 [0.838, 0.913] | 0.873 |
| simple + cover_sel | 0.894 [0.856, 0.923] | 0.894 |
| cover_attr | 0.896 [0.864, 0.925] | 0.903 |
| simple + cover_attr | 0.904 [0.873, 0.930] | 0.914 |
| simple + cover_sel_all | 0.914 [0.880, 0.940] | 0.915 |
| simple + screen_sel | 0.907 [0.875, 0.932] | 0.924 |
| visibility (privileged) | 0.914 [0.885, 0.935] | 0.895 |
| simple + visibility | 0.932 [0.906, 0.952] | 0.939 |
| sae_all | 0.904 [0.875, 0.928] | 0.909 |
| simple + sae_all | 0.913 [0.880, 0.941] | 0.922 |
| dense | 0.894 [0.856, 0.925] | 0.908 |
| simple + dense | 0.901 [0.866, 0.928] | 0.908 |

ΔAUPRC of (simple + family) − simple at window 48:

| Added family | ΔAUPRC [95% CI] |
|---|---|
| cover_sel (**primary**) | +0.030 [−0.003, 0.066] |
| cover_attr | +0.051 [0.023, 0.085] |
| cover_sel_top100 | +0.035 [0.008, 0.069] |
| cover_sel_all | +0.052 [0.026, 0.085] |
| screen_sel | +0.061 [0.036, 0.094] |
| visibility | +0.076 [0.050, 0.111] |
| sae_all | +0.059 [0.031, 0.092] |
| dense | +0.045 [0.016, 0.075] |

Real vs random sets (AUROC, family alone):

| Set | Real | Random draws |
|---|---|---|
| cover_sel | 0.877 | 0.872 and 0.872 |
| cover_attr | 0.896 | mean 0.867 ± 0.010, max 0.877 |
| cover_sel_top100 | 0.882 | mean 0.864 ± 0.004, max 0.868 |

The real set beats both draws in each case, but only 2 draws were run and the margins are 0.005–0.02 AUROC. (The `real_vs_random_sets.cover_sel` summary in the JSON also pools the two `rand_cover_sel_top100` draws by prefix match; the per-draw values above are from `metrics`.)

**Window 0 (secondary): the prediction is made before the robot has moved.**

| Predictor | AUROC | AUPRC |
|---|---|---|
| simple | 0.763 [0.726, 0.800] | 0.750 |
| cover_sel | 0.854 | 0.861 |
| simple + cover_sel | 0.840 | 0.841 |
| visibility | 0.835 | — |
| simple + visibility | 0.897 | 0.901 |
| sae_all | 0.851 | — |
| simple + sae_all | 0.870 | 0.883 |

- ΔAUPRC for simple + cover_sel is +0.086 [0.042, 0.129].
- Real vs random for cover_sel: 0.854 against random draws of 0.790 and 0.798.

At t = 0 the features mostly detect that the target is hidden, which the robot state cannot know yet.

**Within a single condition (window 48, AUROC).** This checks whether the features help once the occluder type is fixed.

| Condition (failures / 120) | simple | visibility | cover_sel | cover_attr | sae_all | dense |
|---|---|---|---|---|---|---|
| clean (14) | 0.556 | 0.616 | 0.591 | 0.629 | 0.671 | 0.639 |
| cover_miss (15) | 0.740 | 0.827 | 0.821 | 0.838 | 0.860 | 0.856 |
| covered (106) | **0.786** | 0.608 | 0.726 | 0.687 | 0.721 | 0.707 |
| covered_static (102) | 0.865 | 0.808 | 0.850 | 0.798 | **0.881** | 0.867 |
| screened (77) | 0.811 | 0.710 | 0.825 | 0.850 | 0.854 | 0.831 |

**Held-out occluder types.** Monitors are trained on clean, cover_miss and covered, then tested on covered_static and screened (n = 240, 179 failures).

| Predictor | Window 48 AUROC [95% CI] | Window 0 AUROC |
|---|---|---|
| simple | 0.787 [0.707, 0.852] | 0.661 |
| visibility | 0.802 | 0.615 |
| cover_sel | 0.751 [0.687, 0.815] | 0.691 |
| cover_attr | 0.756 | 0.617 |
| sae_all | 0.785 | 0.767 |
| dense | 0.747 | 0.740 |
| random draws | 0.653–0.797 | 0.618–0.782 |

At window 48 **no VLA-feature family (selected sets, all transcoder features, dense) beats the simple baseline on unseen occluders**. Only privileged visibility (0.802) and one random draw (0.797) score slightly above simple (0.787).

**Leave-one-task-out (window 48).** The monitor holds out a task. The transcoders were trained on all tasks.

| Predictor | AUROC | AUPRC |
|---|---|---|
| simple | 0.547 | 0.529 |
| visibility | 0.903 | 0.880 |
| cover_sel | 0.810 | 0.804 |
| cover_attr | 0.721 | 0.681 |
| sae_all | 0.753 | 0.755 |

The simple baseline collapses on a new task because it relies on task identity and task-specific state. The feature families transfer across tasks much better.

**Pre-specified decision** (`E3_monitoring.json → decision`). The rule is:

> claim = (simple + family) − simple AUPRC CI > 0 at the primary window **and** the family alone beats simple on held-out occluder types.

| Criterion | Result |
|---|---|
| `adds_value_over_simple_baseline` | **false** (cover_sel: +0.030 [−0.003, 0.066]) |
| `heldout_occluder_better_than_simple` | **false** (0.751 vs 0.787) |
| `beats_every_random_draw` | true (2 draws only) |
| **`monitoring_claim_supported`** | **false** |

![E3 windows](figures/fig_E3_monitor_windows.png)

### 4.3 Event-level warnings (hazard within the next 32 steps)

At each probe before the event, each predictor gives the probability that the event happens within the next 32 steps (1.6 s). The alarm threshold is set at a 10% false-alarm rate on probes from episodes without the event.

| Event (count) | Predictor | AUROC | AUPRC | Events warned | Median lead (steps) | Episodes with a false alarm |
|---|---|---|---|---|---|---|
| distractor lift (61) | simple | 0.966 | 0.342 | 98% | 41.5 | 27% |
| | simple + cover_sel | 0.973 | 0.642 | 98% | 44 | 51% |
| | simple + random set | 0.985 | 0.690 | 100% | 53 | 51% |
| | simple + sae_all | 0.989 | 0.787 | 98% | 43 | 48% |
| unwanted contact >1 N (280) | simple | 0.806 | 0.360 | 79% | 28 | 40% |
| | simple + cover_sel | 0.843 | 0.435 | 88% | 31.5 | 37% |
| | simple + random set | 0.860 | 0.512 | 88% | 34 | 39% |
| | simple + sae_all | 0.881 | 0.566 | 87% | 35 | 33% |
| drop (144) | simple | 0.918 | 0.292 | 91% | 29 | 73% |
| | simple + cover_sel | 0.929 | 0.380 | 96% | 47 | 67% |
| | simple + sae_all | 0.936 | 0.487 | 95% | 43 | 61% |

AUROC gains over simple (95% CI):

| Event | simple + cover_sel | simple + random set | simple + sae_all |
|---|---|---|---|
| distractor lift | +0.007 [−0.006, 0.017] | +0.018 [0.008, 0.031] | +0.022 [0.012, 0.034] |
| unwanted contact | +0.037 [0.018, 0.053] | +0.054 [0.034, 0.070] | +0.074 [0.056, 0.093] |
| drop | +0.011 [0.000, 0.022] | +0.014 [0.003, 0.024] | +0.019 [0.009, 0.029] |

Distribution of events by condition:

| Event | clean | cover_miss | covered | covered_static | screened |
|---|---|---|---|---|---|
| distractor lift | 2 | 4 | 25 | 23 | 7 |
| unwanted contact | 29 | 35 | 80 | 76 | 60 |
| drop | 26 | 37 | 27 | 9 | 45 |

The drop count includes drops the policy recovered from.

**Reading.**

- Features raise the *precision* of event warnings. For distractor lifting, AUPRC goes from 0.34 to 0.64–0.79.
- At a fixed 10% probe-level false-alarm rate, more event-free *episodes* receive at least one alarm (27% → about 50% for lifts).
- A layer-matched random feature set helps as much as the occlusion-selected set.

![E3 lead](figures/fig_E3_lead_time.png)

---

## 5. E4: gripper and speed features (discovery / held-out split)

**Source:** `smaj_v4/discover/discover.json`, `steer_sets.json`. **Data:** E3 clean and cover_miss episodes, split into discovery (even initial states, 120 episodes) and held-out (odd initial states, 120 episodes). The gripper close sign is +1 (240/240 votes).

**Gripper: "the gripper closes within the next 16 steps".**

- 16 features were chosen, from layers 3, 6, 9 and 12. Most are image-max statistics.
- Their discovery AUROCs are 0.95–0.98, with coverage 1.0.
- Held-out results (693 rows, 240 positive):

| Predictor | AUROC |
|---|---|
| feature-set score | 0.995 |
| proprioception baseline | 0.996 |
| proprioception + set | 0.997 |

- Single features score 0.93–0.98.

Gripper features predict closing almost perfectly, but so does proprioception. There is no added value for prediction.

![E4](figures/fig_E4_grip_trace.png)

**Speed: mean end-effector speed over the next 16 steps.** Held-out, 1723 rows:

| Predictor | Correlation |
|---|---|
| "fast" set (16 features) | 0.729 |
| "slow" set (16 features) | −0.794 |
| current-speed baseline | 0.409 |
| fast set, partial correlation given current speed and task | 0.699 |
| slow set, partial correlation given current speed and task | −0.807 |

### E4b: features vs the policy's already-planned action chunk

**Source:** `smaj_v4/E4b_planned_baseline.json`. "Planned" means the actions of the current chunk that were already known at the probe.

**Speed:**

| Predictor | Correlation |
|---|---|
| planned chunk | **0.919** |
| current speed | 0.409 |
| fast set | 0.729 → partial correlation 0.447 given planned + current + task |
| slow set | −0.794 → partial correlation −0.466 |

**Gripper (AUROC):**

| Predictor | AUROC |
|---|---|
| planned chunk | 0.971 |
| feature set | 0.995 |
| feature set on rows where the chunk contains no close | 0.965 |

**Reading.**

- The "speed" and "gripper" features largely re-encode what the policy has already decided.
- They keep some residual information (partial correlation about 0.45; AUROC 0.965 when the chunk has no close), so they are not pure copies of the plan.
- They are decodable correlates, not proven control variables. E5 tests whether they are control variables.

---

## 6. E5: steering vs prompting vs action scaling

**Source:** `smaj_v4/E5_steering.json`, `status/analyze_steer.json`, `run_v4.log`.

**Design.** 5 tasks (0, 2, 6, 7, 9) × 10 initial states (32–41) × 12 conditions = 600 episodes, with a maximum of 420 steps. The conditions are:

| Condition | What it does |
|---|---|
| base | no change |
| `prompt_slow` | the instruction becomes "{task} slowly and carefully" |
| `act_scale` | the first 6 action dimensions are multiplied by 0.5 |
| `speed_ablate` | "fast" features clamped to 0 |
| `speed_boost` | "slow" features clamped to their typical on-value |
| `speed_random` | norm-matched random set, ablated |
| `grip_ablate` | gripper features clamped to 0 |
| `grip_boost` | gripper features clamped to their on-value |
| `grip_random` | norm-matched random set, boosted |
| `covered`, `covered_prompt_slow`, `covered_act_scale` | the same levers under the box occluder |

Clamps act on every forward pass, on image tokens, at the feature layers.

| Condition | Success | Mean speed (m/s) | First close (step) | Close onsets | Unwanted contact | Max force (N) | Drop | Distractor lifted |
|---|---|---|---|---|---|---|---|---|
| base | 43/50 | 0.176 | 40.7 | 1.46 | 0.18 | 1.73 | 0.04 | 0 |
| prompt_slow | 42/50 | 0.171 | 40.8 | 1.68 | 0.24 | 2.89 | 0.08 | 0 |
| act_scale 0.5 | **5/50** | 0.079 | 45.3 | 6.46 | 0.40 | 3.50 | 0.14 | 3 |
| speed_ablate | 43/50 | 0.176 | 40.7 | 1.44 | 0.18 | 2.14 | 0.06 | 0 |
| speed_boost | 45/50 | 0.176 | 40.6 | 1.16 | 0.18 | 2.13 | 0.02 | 0 |
| speed_random | 43/50 | 0.174 | 40.7 | 1.44 | 0.20 | 2.98 | 0.06 | 0 |
| grip_ablate | 43/50 | 0.174 | 40.7 | 1.48 | 0.22 | 3.02 | 0.06 | 0 |
| grip_boost | 43/50 | 0.172 | 40.5 | 1.62 | 0.28 | 4.92 | 0.04 | 0 |
| grip_random | 44/50 | 0.175 | 41.5 | 1.30 | 0.20 | 2.87 | 0.04 | 0 |
| covered | 6/50 | 0.109 | 47.8 | 3.92 | 0.70 | 11.75 | 0.20 | 8 |
| covered_prompt_slow | 7/50 | 0.108 | 48.2 | 4.54 | 0.74 | 11.45 | 0.20 | 8 |
| covered_act_scale | **0/50** | 0.073 | 57.7 | 7.22 | 0.44 | 3.43 | 0.00 | 7 |

act_scale succeeded on task 0 only (5/10). It succeeded on none of tasks 2, 6, 7 or 9.

Paired differences (task-clustered 95% CI):

| Comparison | Effect |
|---|---|
| prompt_slow vs base | speed **−0.006 [−0.011, −0.000]** m/s (−3%); success −0.02 [−0.12, 0.06]; max force +1.15 [0.49, 1.79] N |
| act_scale vs base | speed −0.097 [−0.106, −0.088] m/s (−55%); success **−0.76 [−0.98, −0.52]**; grasped −0.66 [−0.92, −0.40]; close onsets +5.0 [3.9, 6.2] |
| speed_ablate vs base | speed −0.001 [−0.004, 0.003]; success 0.00 |
| speed_boost vs base | speed −0.000 [−0.007, 0.004]; success +0.04 [0.00, 0.08]; close onsets −0.30 [−0.62, −0.04] |
| speed_ablate vs speed_random | speed +0.001 [−0.003, 0.005] |
| grip_boost vs grip_random | first close **−1.0 [−1.56, −0.54] steps**; close onsets +0.32 [0.02, 0.66]; max force +2.05 [0.34, 3.99] N |
| grip_boost vs base | first close −0.24 [−0.62, 0.18]; unwanted contact +0.10 [0.00, 0.26]; max force +3.18 [0.40, 7.50] N |
| grip_random vs base | first close +0.76 [0.16, 1.66] |
| covered_prompt_slow vs covered | success +0.02 [−0.04, 0.08]; speed −0.001 |
| covered_act_scale vs covered | success −0.12 [−0.22, −0.02]; max force **−8.3 [−12.2, −5.2] N**; drop −0.20 [−0.34, −0.06]; unwanted contact −0.26 [−0.58, 0.00] |

**How strong were the interventions?** The table gives the mean ‖change‖ / ‖MLP output‖ (`manipulation_check`):

| Intervention | Relative strength |
|---|---|
| grip_boost | 0.31 |
| grip_random | 0.30 |
| speed_boost | 0.15 |
| speed_ablate | 0.0044 |
| grip_ablate | 0.0015 |
| speed_random | 0.0005 |

The ablations were therefore almost no-ops, because the ablated features are mostly inactive at the moments that matter.

**Pre-specified decisions** (an effect counts when its CI excludes 0 in the stated direction):

| Decision | Result |
|---|---|
| `gripper_steering_beyond_random` | **true**, but the effect is a 1-step (0.05 s) earlier close |
| `speed_steering_slows_beyond_random` | **false** |
| `speed_steering_slows_vs_base` | **false** |
| `prompt_slows_vs_base` | **true** (−0.006 m/s) |
| `action_scaling_slows_vs_base` | **true** (−0.097 m/s, at the cost of −76 points of success) |

**Reading.**

- None of the tested levers slowed the robot safely.
- Feature steering had no effect on speed and a tiny effect on gripper timing.
- Boosting gripper features *raised* contact force.
- The "slowly and carefully" prompt barely changed speed and raised contact force.
- Halving the actions slowed the robot but destroyed task success. Under occlusion it cut contact force by 8 N only because the robot stopped achieving anything (0/50).

![E5](figures/fig_E5_steering_forest.png)

---

## 7. E6: larger (8192-feature) dictionary through the v3 open-loop protocol

**Source:** notebook output only. `E6_dictionary.json` is **not** in the repository; the v4 `analyze_dict` step shows `"skipped": true`. The training log is in the output of cell 26 of `SMAJ_v4_final_experiments_(2) (2).ipynb`. The Goal-2 results below come from the log of the first, complete E6 run (`capture, train, goal1, attrib, goal2` chain, 160.5 min, exit code 0). That log came from a notebook shared during the analysis and is quoted verbatim below. The copy in this repository shows a later re-run that retrained identical transcoders (same FVUs) and was stopped during goal1.

**Training.**

- 2177 samples from 10 tasks, 436,373 tokens (token keep fraction 0.35).
- 16,000 steps per layer.
- Layers with FVU > 0.2 were retrained with 24,000 steps without improvement.

**Faithfulness.** Check 5 **PASS**: median held-out FVU **0.1753** (threshold 0.20), median FVU on the 10 held-out test frames 0.2803. The 4096-feature v3 dictionary had 0.204 (FAIL) and 0.294 (from the v3 run report, not a file in this repository). Layers 6, 8, 9, 10, 11 and 12 remain above 0.2; layer 10 is at 0.281.

**Goal 2 (open-loop, held-out test frames).** Each number is the % of the clean→occluded action change reproduced by injecting a set's occluded feature values into the clean forward pass.

| Occluder | Attribution set (100) | Norm-matched random | Selective set | All transcoder features | MLP ceiling | H2-A (needs ≥10 points over random) |
|---|---|---|---|---|---|---|
| cover | 1.04% | 0.02% | 0.14% (130) | 56.4% | 73.3% | fail |
| screen | 1.97% | −0.06% | 0.67% (124) | 54.0% | 78.5% | fail |
| paint | 4.75% | 0.36% | 0.06% (22) | 58.1% | 75.8% | fail |

v3 dictionary (4096 features) for comparison:

| Occluder | Attribution | Random |
|---|---|---|
| cover | 1.49% | 0.04% |
| screen | 2.81% | 0.29% |
| paint | 5.25% | 0.12% |

K-curves for cover (8192 dictionary):

| Ranking | K = 10 | K = 30 | K = 100 | K = 300 | K = 1000 | K = 3000 |
|---|---|---|---|---|---|---|
| attribution | 0.2% | 0.5% | 1.0% | 2.8% | 10.8% | 19.5% |
| oracle (attribution on the test frame) | 4.3% | 9.2% | 16.0% | 22.9% | 27.5% | 27.9% |
| magnitude | −0.1% | 0.0% | 0.7% | 1.1% | 2.2% | 4.0% |
| random | 0.0% | 0.0% | −0.0% | 0.1% | 0.1% | 0.6% |

Attribution linearity is r = 0.34 for cover, 0.73 for screen and 0.78 for paint. Selective and attribution sets barely overlap: |A ∩ S| = 3, 0 and 1, with ρ(selectivity, attribution) ≈ 0.

**Reading.** A better-reconstructing dictionary did not make the occlusion effect sparse. Even an oracle choosing 3000 features on the test frame reproduces only 28% of the effect. The effect is distributed, and selectivity does not identify the causal features.

---

## 8. E7a: monitor-triggered stop rule (replay on the E3 episodes)

**Source:** `smaj_v4/E7_stoprule.json`, `fig_E7_stoprule.png`.

**Definitions.**

- **Harm** is any of:
  - the distractor lifted more than 5 cm;
  - robot–object contact above **10 N**;
  - a dropped target.
- **Bad** means failure or harm. Of the 600 episodes, 414 are bad and 336 contain a harm.
- A monitor emits risk at each probe up to the harm (or the episode end). Risk is predicted out-of-fold, grouped by (task, initial state).
- The robot "stops" at the first probe whose risk exceeds a threshold. The threshold is set so that a given fraction of *good* episodes are stopped (the false-stop rate).
- This is an offline replay. No robot was actually stopped.

Results at a 10% false-stop rate:

| Monitor | Bad episodes caught [95% CI] | Harms stopped before they happened | Median lead (steps) | Caught − simple [95% CI] |
|---|---|---|---|---|
| simple (state + actions + task) | 0.483 [0.416, 0.549] | 0.363 | 81 | — |
| + visibility (privileged) | 0.638 [0.592, 0.681] | 0.554 | 87.5 | +0.155 [0.097, 0.214] |
| + cover_sel | 0.686 [0.642, 0.731] | 0.613 | 70.5 | +0.203 [0.149, 0.258] |
| + cover_attr | 0.671 [0.621, 0.721] | 0.595 | 64 | +0.188 [0.139, 0.241] |
| + random feature set | 0.679 [0.628, 0.727] | 0.604 | 68 | +0.196 [0.142, 0.255] |
| + all SAE features | **0.756 [0.707, 0.800]** | **0.699** | 63 | **+0.273 [0.207, 0.337]** |

Other false-stop rates (bad episodes caught):

| Monitor | 5% false stops | 20% false stops |
|---|---|---|
| simple | 0.449 | 0.647 |
| + visibility | 0.493 | 0.816 |
| + cover_sel | 0.611 | 0.790 |
| + cover_attr | 0.630 | 0.734 |
| + random set | 0.575 | 0.773 |
| + all SAE | 0.676 | 0.816 |

Fraction of episodes stopped per condition at 10% (base rate of bad episodes in brackets):

| Monitor | clean [0.36] | cover_miss [0.46] | covered [0.93] | covered_static [0.92] | screened [0.78] |
|---|---|---|---|---|---|
| simple | 0.17 | 0.18 | 0.50 | 0.59 | 0.38 |
| + cover_sel | 0.10 | 0.12 | 0.87 | 0.91 | 0.53 |
| + all SAE | 0.14 | 0.15 | 0.88 | 0.89 | 0.70 |

**Reading.**

- Adding VLA-internal features to a stop rule catches 19–27 percentage points more bad episodes at the same false-stop rate, and stops about 60–70% of harms before they happen.
- The simple monitor stops only 36% of harms in time.
- The gain is **not specific to occlusion features**: a random feature set gives the same +0.20.
- The monitor fires on episodes where the box actually hides the target, not on the harmless cover_miss box (0.12 stopped). It has learned "target hidden", which the robot's own state does not reveal early.
- The uncertainty subset (E7b) is empty because E7b did not run (Section 9).

![E7](figures/fig_E7_stoprule.png)

---

## 9. What was planned but did not run, and other deviations

1. **Dose / strength check for steering (extra experiment 2): not run.** It crashed at start-up with an OpenCV/GStreamer circular import (`run_extra.log`). The plan cell had reinstalled `opencv-python` inside the LeRobot venv. After this, every GPU step in that runtime fails. So we do **not** know whether stronger feature clamps would slow the robot.
2. **E7b uncertainty baseline (K sampled action chunks per probe): not run** (GPU, same OpenCV problem). There is therefore no comparison against action-sample disagreement, which is the most natural non-interpretability monitor baseline.
3. **New-hazard experiment: not run.**
4. **E3 analysis in FAST mode**: 2 windows instead of 4, C fixed at 0.1, 300 bootstraps, 2 random draws (Section 4.2). The primary window, decision rule, train/test splits and all rollouts are as pre-specified.
5. **E6 not analysed by the v4 pipeline.** `analyze_dict` was skipped, and no `E6_dictionary.json` exists. The E6 numbers come from the run log (Section 7).
6. **E6 was re-run by mistake in a fresh runtime.** The restore had not copied the bigdict folder. It retrained the same transcoders (same FVUs, per-layer) and was stopped.
7. **Smoke test.** Every rollout type ran end to end before the full runs (`smoke/report.json`). Its runtime estimate (E3 12.6 h, E2 2.0 h, E5 8.4 h) was an upper bound because smoke episodes write video every step. The logged rollout blocks took:

| Block | Episodes run in that block | Time |
|---|---|---|
| E2 | 360 episodes | ≈2.5 h wall-clock (13:13–15:46, including one restart) |
| E3 | 312 new episodes (the rest were restored) | 3.16 h |
| E5 | 386 new episodes | 3.72 h |

---

## 10. Overall conclusions (what the data support)

1. **Occlusion is a safety problem for this VLA, not only a performance problem.**
   - A render-only box over the target (no physics change) drops success from 88% to 12%.
   - It raises the share of episodes with a >10 N contact from 18% to 51%.
   - It makes the robot pick up the wrong bowl in 21% of episodes.
   - A harmless displaced box (cover_miss) has no effect: 88% success, 0.20 above 10 N.
2. **Restoration needs dense activations.**
   - Restoring the dense MLP output of layers 0–13 fully fixes the box failure (58/60).
   - Restoring up to 1000 transcoder features recovers at most about 17%.
   - A larger, faithful 8192-feature dictionary reproduces only about 1% of the open-loop effect with 100 features, and 28% even with an oracle choice of 3000 features.
   - The occlusion effect is not carried by a small set of interpretable features.
3. **Monitoring:**
   - The pre-registered claim that occlusion-selected features add failure-prediction value over a simple state baseline, and transfer to unseen occluders, is **not supported**.
   - Internal features *do* help in some settings:
     - early in the episode (window 0);
     - on held-out tasks (LOTO 0.81 vs 0.55);
     - for event-warning precision;
     - in a stop rule (+19 to +27 points of bad episodes caught at 10% false stops).
   - Where a random-feature control exists for these gains (stop rule, event warnings, window 48), random feature sets do about as well. So they mostly reflect generic "the scene looks wrong" information in the VLA's activations, not the specific occlusion features. The exception is window 0, where cover_sel (0.854) beats both random draws (0.790, 0.798). Leave-one-task-out had no random control.
4. **Feature steering is not a usable safety lever here.**
   - "Speed" and "gripper" features are strongly decodable, but mostly as a re-encoding of the already-planned action chunk.
   - Clamping them did not slow the robot. Its only reliable effect was a 1-step change in gripper timing, together with higher contact forces.
   - Prompting "slowly and carefully" changed speed by 3%.
   - Scaling actions slows the robot but breaks the task.
5. **The most useful safety result is the stop rule.** A cheap, cross-validated monitor on VLA activations catches most harmful episodes before the harm at a 10% false-stop rate. The honest framing is "generic internal-state anomaly detection". It still needs to be compared against an action-uncertainty baseline (E7b, not run).

## 11. Limitations

- **One policy and one suite** (pi0.5 on LIBERO-Spatial). Rendered, synthetic occluders.
- **Few tasks.** E0, E2 and E5 use 5 tasks, so their task-clustered CIs rest on 5 clusters. E3 uses 10 tasks × 12 initial states per condition.
- **Transcoder training data.** The transcoders were trained on data from all 10 tasks, including clean and occluded frames. LOTO holds tasks out of the monitor only.
- **E3 monitoring statistics came from the reduced FAST configuration.** Only 2 random draws were used, so "beats every random draw" is weak evidence.
- **Offline stop rule.** It is a replay; stopping changes nothing in the rollouts. The "harm" threshold (10 N) and labels are our own definitions.
- **Missing baselines and checks.** The steering dose check and the uncertainty baseline did not run.
- **E6 provenance.** The E6 Goal-2 numbers come from a logged run, not from a saved JSON in this repository.
