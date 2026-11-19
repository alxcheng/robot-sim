# Walker Prototype (spike) — Results

Headless biped walker built to de-risk legged locomotion before Milestone 1 (spec §5.4, §8.4).

* **Code:** `crates/walker_lab` (Rust, `rapier3d` 0.36.1, no Bevy) and `tools/walker_viewer` (three.js + Playwright renderer).
* **Robot:** concept-art proportions: 3.8 m tall, 3,560 kg, pelvis reactor (CoM ≈ 52% of height), 2 legs × 6 revolute joints
  (hip yaw/roll/pitch, knee, ankle pitch/roll) as one Rapier **multibody**, no self-collision.
* **Sim:** 60 Hz × 4 substeps (spec §4.1). `ForceBased` position motors with per-joint torque caps.
* **Controller:** SIMBICON-style (Yin et al. 2007): two-state gait; swing-hip target in the heading frame corrected by
  CoM offset/velocity from the stance ankle (Raibert-style foot placement); stance hip servos the trunk; ankles keep feet
  level (ankle balance strategy when standing); hip yaw steers. Optional upright-assist torque on the trunk = spec's
  "Normal law" helper.
* **Tuning:** (1+4) random search over 19 gait parameters on 5 scenarios, run separately per assist level.

## Results (from `out/report.json`)

| Scenario | Assist 0 (pure physics) | Assist 0.35 (weak helper) | Assist 1.0 (full helper) |
|---|---|---|---|
| Walk 0.7 m/s | ✅ 0.82 m/s | ✅ 0.74 | ✅ 0.70 |
| Walk 1.2 m/s | ✅ 1.11 | ✅ 1.18 | ✅ 1.18 |
| Walk 2.0 m/s command | ✅ 1.61 (undershoots) | ✅ 1.76 | ✅ 1.54 |
| Turn 46° then back | ✅ heading err 0.01 rad | ✅ | ✅ |
| 10 cm step down | ✅ | ✅ | ✅ |
| Pushes ±0.3 side, ±0.4 fwd/back | ✅ | ✅ | ✅ |
| Max push, worst of 4 step phases (side / fwd / back, m/s of whole-robot Δv) | 0.67 / 0.89 / 2.38 | 2.55 / 3.15 / 4.12 | ≥5 / ≥5 / ≥5 (search cap) |
| 10-min soak, random speed/heading/pushes | 10 falls (auto-reset) | 1 fall | 0 falls |
| Run (flight phase) | ❌ no flight phase (3%), tops out ~1.9 m/s, nearly falls | ❌ falls | ❌ falls |
| Non-finite values / explosions | none | none | none |

Stance-foot slip averages 0.1–0.4 m/s (higher at speed). Sim speed: ~50–90× real time single-threaded.

## Findings

1. **An assisted walk is solid, and pure-physics walking works too.** All three tuned gaits walk, turn, step down and
   recover from moderate pushes with no solver blow-ups.
2. **Full-strength assist is a hand of god.** At assist 1.0 the trunk can't tip, so big pushes make the robot skid
   instead of fall (see video 05). That breaks spec rule "assists weaker than real impacts". **Recommendation: ~0.35 as
   the Normal-law level**, 0 for Direct law.
3. **Running is not solved.** The SIMBICON walking structure does not produce a flight phase; it saturates at a fast walk.
   Running needs its own state machine (stance push-off, flight, landing), likely with the impact-absorption work in §6.4.
4. **Rapier specifics learned:**
   * Default `AccelerationBased` motors scale with the link's own mass, so knees couldn't hold a 3.6 t body; use
     `ForceBased` with load-sized gains.
   * Instant impulses on multibody links are lost when link velocities sync from the multibody; apply pushes/hits as
     short force pulses.
5. **Fixed-step stepping from rest** needs the ankle strategy only while standing; using it while walking fights
   forward progress.

## Not done / next
* Running gait; turning in place; slopes and stairs; arms swinging; damage-degraded control laws driving `assist`.
* Visual feel was judged only from rendered video; needs hands-on play in Bevy.

## Reproduce
```
cargo run --release -p walker_lab -- report               # metrics for all assist levels -> out/report.json
cargo run --release -p walker_lab -- tune 250 0.35        # retune a gait -> out/gait_a0.35.json
cargo run --release -p walker_lab -- run showcase 0 out/rec.json
cd tools/walker_viewer && npm install && node capture.js ../../out/rec.json ../../out/video 30
```
Tuned gaits are committed in `crates/walker_lab/gaits/` (copy to `out/` to use).

---

# Running (round 2): better parts + a running controller

Goal: an actual run (flight phase), with assistance allowed to close the gap.

## What was added
* **Parts** (`RobotSpec`): bigger feet (×1.3), **motor-driven swinging arms** (counter-swing against the same-side leg),
  **digitigrade multi-joint legs** (7 joints: hip yaw/roll/pitch, knee, metatarsal, toe pitch/roll; the metatarsal is
  coupled opposite to the knee so the leg folds like a spring), and a **reaction-wheel back unit** (pitch + roll
  wheels, 2 × 100 kg) that gives the trunk internal stabilising torque (legit physics, unlike the external assist).
  Total mass held at ~3.6 t.
* **Running controller** (`run_ctl.rs`): each leg is a *virtual leg* (length + angle) solved to joint angles by IK, so the
  same controller drives biped and digitigrade legs. Stance (compress → extend + toe push-off) → flight → landing at the
  Raibert angle φ = asin((v·T/2 + k(v − v_cmd))/ℓ). Stance hip servos the trunk.
* **Flight metrics:** number and mean duration of airborne phases, centre-of-mass bounce.

Builds: **A** baseline biped · **B** biped + swinging arms + big feet · **C** digitigrade + arms + big feet ·
**D** = C + reaction wheels. Gaits tuned by random search on 2.2 m/s, 3.0 m/s and a turn-and-push run.

## Results (`crates/walker_lab/gaits/run_report.json`)

| Build | Assist | 3.0 m/s cmd: speed | Airborne | Flight phase | Bounce | Turn + push run | Side push (worst phase) | 5-min random soak |
|---|---|---|---|---|---|---|---|---|
| A baseline | 0.35 | 2.76 | 37% | 126 ms | 4 cm | ❌ falls | 1.08 m/s | 98 falls |
| B arms + feet | 0.35 | 2.82 | 44% | 293 ms | 15 cm | ✅ but doesn't turn, feet slide ~0.9 m/s | 1.15 m/s | 2 falls |
| C digitigrade | 0.35 | 2.75 | 49% | 134 ms | 3 cm | ✅ | 2.38 m/s | 21 falls |
| **C digitigrade** | **0 (pure physics)** | **2.66** | **45%** | 134 ms | 4 cm | ✅ | 1.30 m/s | 111 falls |
| D + reaction wheels | 0.35 | 1.98 | 36% | 194 ms | 13 cm | ✅ (slow) | 1.57 m/s | 8 falls |
| **D + reaction wheels** | **0 (pure physics)** | **2.26** | **43%** | 161 ms | 5 cm | ✅ | **2.10 m/s** | **5 falls** |

No non-finite values in any run.

## Findings
1. **A real run exists now**, with dozens of genuine airborne phases of 130–290 ms per run, up to ~3 m/s.
2. **Digitigrade legs were the biggest single win** at assist 0.35: smoother tracking and lower tilt. They also ran with
   **no assist at all**. (The biped builds A/B were not tuned at zero assist, so that comparison is still open.)
3. **Reaction wheels replace most of the external assist**: build D with *zero* assist was the most robust runner
   (5 falls in 5 min of random commands and pushes; 2.1 m/s side push), at the cost of top speed. This is the
   physically honest way to "close the gap".
4. **Swinging arms + big feet on a biped (B)** gave the highest bounce and survived the soak, but feet slide badly and it
   stopped tracking heading; arms alone don't fix the biped.
5. **Gaits are narrow.** C passes its tuned scenarios but falls often under random speed/heading changes; robustness
   needs the soak in the tuning objective.
6. **Looks:** the optimizer only rewarded survival, speed and air time, so the runs are crouched, strongly forward-leaning
   sprints with large arm swings. Style terms (posture, arm range, bounce height) are the next tuning step.

---

# Learned policy, step 1: distilling the hand-built runner (behaviour cloning + DAgger)

* **Network:** MLP 56 → 192 → 192 → 30 (tanh), trained in Rust (`policy.rs`, hand-written backprop + Adam, 4 threads).
* **Observations (56):** command, trunk tilt/rates, centre-of-mass velocity/height, per-leg limb angles and rates,
  foot positions relative to the CoM, foot contacts, contact-history phase cues, arm angles. All sensor-derived.
* **Actions (30):** per-joint position targets and stiffness scales for both 7-joint legs, plus two arm targets.
  Reaction wheels (and assist) stay as hand-coded reflexes.
* **Training:** teacher = build D running controller, no assist. 24 teacher runs, then 5 DAgger rounds (the network
  drives, the teacher labels every visited state). ~3 min total on 4 CPU cores.

Fixed evaluation set (turn-and-push run + 9 random 20 s runs; `crates/walker_lab/policies/distill_eval.json`):

| | Falls | Survival | Speed error | Airborne |
|---|---|---|---|---|
| Teacher | 2/10 | 94% | 0.51 m/s | 48% |
| Round 0 (BC only) | 10/10 | 24% | 1.09 | 27% |
| Round 1 | 7/10 | 78% | 0.63 | 36% |
| **Round 2** | **0/10** | **100%** | 0.60 | 43% |
| Round 3 | 2/10 | 92% | 0.39 | 40% |
| Round 5 | 2/10 | 95% | 0.52 | 44% |

Result: the network reproduces the teacher from round 2. Imitation plateaus at teacher level, so further gains need
RL fine-tuning (step 2). Round-2 weights: `crates/walker_lab/policies/run_D_a0_distilled_round2.json`.
Reproduce: `walker_lab distill D 0 5` then `walker_lab evaldistill D 0`.
