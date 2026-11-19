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
