# Seeing Is Not Stopping: Occlusion Hazards in a Vision-Language-Action Policy, and What Its Internal Features Can and Cannot Do About Them

*Draft for a CoRL 2026 workshop on robot safety. Every number is taken from the result files in this repository (`smaj_v4/*.json`; see `REPORT_full.md` for sources and full tables). Items marked **[TODO before submission]** are gaps we know about.*

---

## Abstract

Vision-language-action (VLA) policies are increasingly proposed for open-world manipulation, but it is unclear how they fail when their view of the task is degraded, and whether their internal representations can be used to catch or correct those failures. We study pi0.5 on LIBERO-Spatial under *render-only* occluders. These change only what the cameras see, never the physics.

**Occlusion is a safety problem, not just a performance problem.** A gray box over the target cuts success from 88% to 12% (600 paired episodes, 10 tasks). It raises the share of episodes with a robot–object contact above 10 N from 18% to 51%, and makes the robot lift the wrong, identical-looking object in 21% of episodes. The same box, moved so that it hides nothing, has no effect.

**Three ways to use the policy's internals for safety.** We test sparse transcoder features on the 18 language-model MLPs, and compare them against controls:

1. **Repair.** Restoring clean *dense* MLP outputs fixes the failure (58/60). Restoring up to 1000 sparse features recovers at most 17%.
2. **Monitoring.** A pre-registered test finds that occlusion-selected features do not add significant failure-prediction value over a robot-state baseline at the primary time point, and do not transfer to unseen occluders. However, a stop rule that adds VLA activations to the state baseline stops 19–27 percentage points more bad episodes at a 10% false-stop rate, and stops 60–70% of harms before they occur (36% for the baseline). This gain is matched by *random* feature sets, so it reflects generic internal "scene is off" information rather than interpretable occlusion features.
3. **Steering.** Clamping features that decode future speed and gripper closure (partly re-encodings of the already-planned action chunk) does not slow the robot. Its one reliable effect was on gripper timing: closing one step earlier relative to a random-feature control, with higher contact force. Prompting "slowly and carefully" changes speed by 3%. Halving actions slows the robot but drops success by 76 points.

VLA internals look promising as *monitors* and not as *levers*, and interpretability-specific claims need random-feature controls.

---

## 1. Introduction

Generalist robot policies will face degraded perception: occlusion, glare, and sensor dropout. Safety cases for such systems need to know three things:

- what the policy *does* when perception fails (keeps going, stops, or does something harmful);
- whether the failure can be *detected* early enough to intervene;
- whether there are *levers* to make behaviour more conservative.

Interpretability tools for VLAs (sparse dictionaries on VLM activations, activation probes for failure prediction, feature steering) are often motivated by exactly these needs. We ask whether they deliver, measured by *task-level and safety-relevant outcomes* rather than by reconstruction or probe accuracy alone.

**Contributions.**

1. **A controlled occlusion stress test with safety metrics.** Render-only occluders (moving box, static box, camera-specific screen) are paired with a *harmless-box* control that separates "target hidden" from "odd object in view". Episodes are paired (same seed, initial state and noise across conditions). We log contact forces, wrong-object lifting and drops.
2. **Negative result on sparse repair.** The occlusion effect is carried by dense, distributed activations. This holds even for a larger 8192-feature dictionary that passes a reconstruction-faithfulness check.
3. **A pre-registered monitoring test plus an offline stop-rule evaluation.** Internal activations improve harm interception, but random feature sets do equally well and the pre-registered claim is not supported.
4. **A comparison of three "be careful" levers:** feature steering, prompting and action scaling. None gives safe slowing.

## 2. Setup

**Policy.**

- `lerobot/pi05_libero_finetuned` (pi0.5, PaliGemma + flow-matching action expert).
- 50-step action chunks executed open-loop at 20 Hz; 280-step limit (420 in the steering study).

**Hooks and dictionaries.**

- Hooks sit on the outputs of the 18 MLPs of the PaliGemma language model, which processes image and text tokens. The action expert is not hooked.
- TopK transcoders: 4096 features, k = 64, one per layer. An 8192-feature dictionary is used in the Appendix.

**Tasks.** LIBERO-Spatial, all 10 tasks: "pick up the black bowl … and place it on the plate". Every scene contains **two identical black bowls**, the target and a distractor.

**Occluders.** All are drawn at render time only; there is no physics.

| Occluder | What it does |
|---|---|
| box | Follows the target; visible to both cameras. |
| static box | Larger box fixed around the target's start position, randomly jittered. |
| screen | Panel that hides the target from the third-person camera only. |
| miss box | The same box displaced so it hides nothing (segmentation-checked). |

![conditions](figures/fig_conditions_task0.png)

*Figure 1. Task 0 at t = 0 in the five conditions (third-person and wrist camera).*

**Safety outcomes.**

- *Unwanted contact*: the maximum robot force on any movable non-target object, reported at 1 N, 10 N and 20 N.
- *Wrong-object lift*: distractor raised more than 5 cm.
- *Drop*: target leaves a two-finger grasp and falls more than 3 cm.
- *Harm*: wrong-object lift, contact above 10 N, or drop.
- *Bad episode*: failure or harm.

**Statistics.**

- Paired differences use a task-clustered bootstrap; success rates use Wilson intervals.
- Monitors are logistic regressions with out-of-fold predictions (GroupKFold by task × initial state).
- The monitoring decision rule, primary time window and train/test occluder split were fixed before the data existed.

## 3. Occlusion makes the policy unsafe, not just unsuccessful

Monitoring set: 10 tasks × 12 initial states × 5 conditions = 600 episodes. Differences are relative to clean.

| Condition | Success | Δ success [95% CI] | Contact >10 N | Contact >20 N | Wrong object lifted | Target grasped | Gripper close onsets |
|---|---|---|---|---|---|---|---|
| clean | 106/120 (88%) | — | 18% | 2% | 2 (2%) | 93% | 1.35 |
| miss box (control) | 105/120 (88%) | −0.01 [−0.09, 0.05] | 20% | 4% | 4 (3%) | 93% | 1.38 |
| box | 14/120 (12%) | −0.77 [−0.89, −0.63] | 51% | 19% | 25 (21%) | 29% | 2.64 |
| static box | 18/120 (15%) | −0.73 [−0.89, −0.53] | 50% | 14% | 23 (19%) | 16% | 2.74 |
| screen (3rd-person only) | 43/120 (36%) | −0.53 [−0.71, −0.35] | 43% | 17% | 7 (6%) | 58% | 2.66 |

**Why this matters for safety.**

- The policy does **not** stop or wait when it cannot see the target. It keeps moving: path length rises from 1.08 m to 1.54 m, even though mean speed falls from 0.18 to 0.12 m/s. It re-grasps repeatedly and pushes into objects; median peak contact force goes from 0 N to 10 N.
- In three of the ten tasks it often picks up the *other* bowl: under the box, 7, 9 and 9 of 12 episodes in tasks 1, 2 and 5.
- The miss-box control shows the harm comes from losing sight of the target, not from an unfamiliar object in view.
- Hiding the target from the third-person camera alone is enough to cause 53 points of success loss and more hard contacts. The wrist camera does not compensate.

The same pattern appears in an independent 5-task closed-loop set (v3 episodes, reproduced 540/540 exactly): box 10/60 vs clean 58/60, and gray pixel paint 44/60.

## 4. Can the policy's internals repair the failure?

In a clean-vs-occluded paired forward pass, we overwrite the occluded run's activations with the clean run's. This uses privileged clean renders, so it is a diagnostic, not a deployable fix. Setting: box occluder, 5 tasks × 12 initial states.

| What is restored | Success | Recovered fraction of the gap |
|---|---|---|
| none (box) | 10/60 | 0 |
| dense MLP outputs, all 18 layers | 59/60 | 1.02 [0.95, 1.16] |
| dense MLP outputs, layers 0–13 | 58/60 | 1.00 [0.93, 1.14] |
| 100 selective transcoder features | 10/60 | 0.00 |
| 300 attribution-ranked features | 18/60 | 0.17 [0.04, 0.36] |
| 1000 attribution-ranked features | 18/60 | 0.17 [−0.14, 0.37] |
| 300 / 1000 random features (norm-matched) | 8/60 and 12/60 | −0.04 and 0.04 |

- Attribution-ranked features beat random at k = 300 (+0.17 [0.03, 0.30]) but not at k = 1000 (+0.10 [−0.13, 0.30]).
- Injecting the 100 attribution features' *occluded* values into clean runs leaves success at 60/60.
- An 8192-feature dictionary (median held-out FVU 0.175, passing our faithfulness bar; the 4096 dictionary was at 0.204) does not change the picture in open-loop tests on held-out frames:
  - its 100 attribution features reproduce 1.0% of the box's effect on the action (random 0.02%);
  - even 3000 features chosen with oracle attribution on the test frame reproduce 28%, against 73% for the dense-MLP ceiling.

**Take-away.** The information the policy loses under occlusion lives in dense, distributed activations. Sparse interpretable features are not a handle for repairing it.

## 5. Can the internals warn us in time?

### 5.1 Pre-registered failure prediction (negative)

Before the data existed, we fixed the following test. The occlusion-selected feature set (129 features) must:

- raise failure-prediction AUPRC over a robot-state baseline at step 48 (end-effector pose, gripper, recent speed and actions, task identity); and
- beat that baseline on occluder types held out of training (train on clean / miss / box, test on static box / screen).

Results (600 episodes, 314 failures):

| Test | Baseline | Occlusion features | Other predictors |
|---|---|---|---|
| Step 48, AUROC | 0.83 | + selected features 0.89 | + all features 0.91; + privileged visibility 0.93 |
| Step 48, ΔAUPRC over baseline | — | **+0.030 [−0.003, 0.066]**, criterion not met | — |
| Held-out occluders, AUROC | 0.79 | **0.75**, criterion not met | random-feature draws 0.65–0.80 |

**The pre-registered monitoring claim is not supported.** Two secondary observations:

- At step 0, before the robot moves, features add +0.086 [0.042, 0.129] AUPRC. They mostly detect that the target is hidden.
- When the monitor is tested on a held-out *task*, the state baseline collapses (AUROC 0.55) while the selected features hold up (0.81).

### 5.2 Offline stop rule (positive, but non-specific)

The monitor outputs a risk score every 8 steps. The robot is "stopped" the first time risk crosses a threshold; the threshold is set so that a chosen fraction of *good* episodes (success, no harm) are stopped. Of the 600 episodes, 414 are bad and 336 contain a harm. Results at a 10% false-stop rate:

| Monitor | Bad episodes caught [95% CI] | Harms stopped *before* they occur | vs baseline |
|---|---|---|---|
| robot state + actions + task | 0.48 [0.42, 0.55] | 0.36 | — |
| + privileged target visibility | 0.64 [0.59, 0.68] | 0.55 | +0.16 [0.10, 0.21] |
| + occlusion-selected features | 0.69 [0.64, 0.73] | 0.61 | +0.20 [0.15, 0.26] |
| + **random** feature set (same size, same layers) | 0.68 [0.63, 0.73] | 0.60 | +0.20 [0.14, 0.26] |
| + all transcoder features | **0.76 [0.71, 0.80]** | **0.70** | **+0.27 [0.21, 0.34]** |

- At a 5% false-stop rate: baseline 0.45, all features 0.68.
- At a 20% false-stop rate: baseline 0.65, all features 0.82.
- With the selected features, the stop rule stops 10% of clean episodes and 12% of harmless-miss-box episodes, but 87% of box and 91% of static-box episodes.

![stop rule](figures/fig_E7_stoprule.png)

*Figure 2. Fraction of bad episodes stopped as a function of the fraction of good episodes stopped.*

**Hazard-level early warning** (will the event happen within 1.6 s?) shows the same pattern:

- For wrong-object lifting (61 events), AUPRC rises from 0.34 (state) to 0.64 (+ selected features), 0.69 (+ random features) and 0.79 (+ all features). Median warning lead is 41–53 steps (about 2–2.7 s).
- The number of harmless episodes that ever receive an alarm also rises, from 27% to about 50%.

**Take-away.** VLA activations carry actionable information about impending harm that robot state lacks. A cheap monitor on them intercepts most harms before they happen. The information is **generic**: random features work as well as "occlusion features". Interpretability adds no specific monitoring value here, and we recommend random-feature controls as standard for such claims.

## 6. Can the internals make the robot more careful?

We find transcoder features that predict future end-effector speed and gripper closure on held-out episodes:

| Target | Features | Baselines |
|---|---|---|
| speed | r = 0.73 ("fast" set) / −0.79 ("slow" set) | current speed r = 0.41 |
| gripper close within 16 steps | AUROC 0.995 | proprioception 0.996 |

Against the policy's *already-planned* action chunk:

| Target | Planned chunk | Features, given the plan |
|---|---|---|
| speed | r = 0.92 | partial r = 0.45 |
| gripper | AUROC 0.97 | AUROC 0.965 on rows whose chunk has no close |

So the features are partly re-encodings of what the policy has already decided.

We compare three levers (5 tasks × 10 initial states per condition; 95% task-clustered CIs vs the unmodified policy):

| Lever | Δ speed (m/s) | Δ success | Other safety-relevant change |
|---|---|---|---|
| Prompt "…slowly and carefully" | −0.006 [−0.011, −0.000] (−3%) | −0.02 [−0.12, 0.06] | max contact force +1.2 [0.5, 1.8] N |
| Clamp "slow" features on | −0.000 [−0.007, 0.004] | +0.04 [0.00, 0.08] | — |
| Ablate "fast" features | −0.001 [−0.004, 0.003] | 0.00 | intervention ≈ 0.4% of MLP output norm |
| Boost gripper features (vs random control) | — | −0.02 | closes 1.0 [0.5, 1.6] steps earlier; max force +2.0 [0.3, 4.0] N |
| Scale actions × 0.5 | −0.097 [−0.106, −0.088] (−55%) | **−0.76 [−0.98, −0.52]** | grasp rate −0.66 |
| Under the box: prompt | −0.001 | +0.02 [−0.04, 0.08] | — |
| Under the box: scale × 0.5 | −0.036 | −0.12 [−0.22, −0.02] (6/50 → 0/50) | max force −8.3 [−12.2, −5.2] N |

**Take-away.**

- Decodable speed and gripper features are not control knobs.
- Language-level caution is almost ignored by the policy.
- Crude action scaling reduces contact forces only by making the robot fail. Under occlusion it lowers peak force by 8 N and success to zero.

A dose–response check with stronger clamps did not run (Section 8), so the steering null holds only at the strengths tested.

## 7. Discussion

For this VLA, the safety-relevant use of internals is **detection, not control**. Three lessons for safety evaluations of interpretability methods:

1. **Measure outcomes, not proxies.** High decoding accuracy (AUROC 0.995 for gripper closure) coexisted with no useful steering effect. Faithful reconstruction (FVU 0.175) coexisted with no sparse causal handle.
2. **Always include random-feature and harmless-perturbation controls.** Without the random-feature stop rule we would have credited occlusion features with a +20-point safety gain that any feature set provides. Without the miss-box control we could not attribute the harms to lost target visibility.
3. **Pre-register the monitoring claim.** Our pre-registered test was negative even though several post-hoc views (step 0, held-out tasks, event precision) look positive. Reporting both avoids over-claiming.

## 8. Limitations and gaps before submission

**Limitations.**

- One policy and one simulated suite.
- Occluders are synthetic, render-only boxes and panels.
- 5-task clusters in the repair and steering studies make those CIs coarse.
- Transcoders were trained on all tasks, so leave-one-task-out holds tasks out of the monitor only.
- The stop rule is an offline replay; a real stop changes what happens next.
- "Harm" thresholds are our own definitions.

**Gaps, all [TODO before submission].**

- (i) **Uncertainty baseline.** Disagreement across K sampled action chunks is the most natural non-interpretability monitor. It was implemented but did not run, because of a GPU runtime failure.
- (ii) **Steering dose–response.** Also implemented; did not run for the same reason.
- (iii) **Full monitoring analysis.** It was run in a reduced configuration: windows 0/48 instead of 0/24/48/80, C fixed at 0.1 instead of a grid, 300 bootstraps instead of 1000, and 2 random-feature draws evaluated instead of the planned 5. It should be re-run at full settings. The stop rule's random-feature comparison uses one draw (`rand_cover_sel_0`).
- (iv) **Citations.** Titles and authors of related work must be checked (references below are placeholders by identifier).

## References (to be completed)

- [Swann et al., arXiv:2603.19183] Sparse features of the pi0.5 VLM and steering. **TODO**: verify title and authors.
- [OpenReview Ln0yjcSVku] VLA steering evaluated by task-completion speed, with a prompting control. **TODO**: verify.
- [OpenReview ZVY1lSGcnZ] Action-expert activation probes predict task failure. **TODO**: verify.
- [arXiv:2609.06326] *Rethinking Safety for Generalist Robots*. **TODO**: verify authors.
- LIBERO benchmark; pi0.5 / openpi; LeRobot; TopK sparse autoencoders / transcoders. **TODO**: add standard citations.

---

## Appendix A: Additional numbers

- **Closed-loop reproducibility.** Re-running the 540 v3 closed-loop episodes gives identical outcomes and lengths (540/540).
- **Contact-force sensitivity.**

  | Condition | >1 N | >5 N | >10 N | >20 N |
  |---|---|---|---|---|
  | clean | 0.24 | 0.24 | 0.18 | 0.02 |
  | miss box | 0.29 | 0.26 | 0.20 | 0.04 |
  | box | 0.67 | 0.65 | 0.51 | 0.19 |
  | static box | 0.63 | 0.62 | 0.50 | 0.14 |
  | screen | 0.50 | 0.47 | 0.43 | 0.17 |

- **Within-condition failure prediction** (step 48, AUROC). Under the box itself the state baseline is best (0.79 vs 0.73 for the selected features). Under the screen the features are slightly better (0.83–0.87 vs 0.81).
- **Unwanted-contact warnings (>1 N, 280 events).** AUROC 0.81 (state), 0.84 (+ selected), 0.86 (+ random), 0.88 (+ all features).
- **Drop warnings (144 events).** AUROC 0.92 (state) vs 0.93–0.94 with features. Episode false-alarm rates are high for every monitor (58–73%).
- **8192-feature dictionary, open loop, cover occluder.** Attribution top-K reproduces 0.2 / 1.0 / 10.8 / 19.5% of the effect at K = 10 / 100 / 1000 / 3000. Oracle ranking reaches 4.3 / 16.0 / 27.5 / 27.9%. Random reaches 0.0 / −0.0 / 0.1 / 0.6%. Selective and attribution sets share 3 of 100 features.

![E3 windows](figures/fig_E3_monitor_windows.png)
![E5 forest](figures/fig_E5_steering_forest.png)
![E2 dose](figures/fig_E2_dose.png)
