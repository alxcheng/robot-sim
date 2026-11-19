# Technical Design Specification: 3D Physics Robot Battler (Rust) — Rev. 2

> Rev. 2 folds in the design review: verified Rapier API surface, scope cuts, damage rules,
> bracing/impact response, layered control with degradable "control laws", enemy AI,
> keyboard + mouse control map, and real-robotics research references.

---

## 1. Summary & Pillars

A single-player, third-person robot combat game in **Rust**. Genre reference point: modular mech
action (build from parts, boost-driven movement, mixed ranged/melee) — but where that genre uses
abstract stats, **this game simulates them**:

| Genre convention (abstract) | This game (physical) |
|---|---|
| HP / armor points bar | Plates dent, bolts shear, limbs detach |
| Stagger meter | Real balance loss — knocked down, shoved, spun by recoil |
| Weight / load stats | Real mass and inertia |
| Boost with no consequence | Thrust applied at mount points; losing a pod yaws you off course |

**Pillars**
1. **Stability first.** Zero jitter, no explosions, no `NaN`/infinite velocities. Combat feel beats engineering accuracy.
2. **Every stat is physical.** If a number matters, it should be a mass, a torque limit, a stiffness or a break threshold.
3. **The player commands intent, the robot handles the joints.** Layered control, like real piloted robots (§8).
4. **Losing parts is gameplay, not just failure.** Parts detach, can be recovered, and reattached (§9).

**Materials are deliberately unrealistic** (tuned for feel), but control and structural ideas are borrowed from real robotics (§13).

---

## 2. Tech Stack

* **Physics:** `rapier3d` **0.36.1** used directly as a Bevy `Resource` (no `bevy_rapier`).
  * Verified in the 0.36.1 source: `SoftBodySet`, `SoftBodyBuilder::volumetric` / `volumetric_skinned` / `volumetric_with`,
    `SoftBodySolver::Fem`, `SoftBodyCellModel::{Corotational, NeoHookean}`, `SoftBodyMaterial`
    (`young_modulus`, `poisson_ratio`, `elastic_damping_ratio`, `plastic_yield`, `plastic_creep`, `plastic_max`,
    `deformation_damping`, `tear_strain`, edge plasticity), clusters/soft-frames (`set_cluster_stiffness_scale`,
    `set_cluster_pinned`, `set_cluster_kinematic_target`), `SoftBodyTearEvent`, `RopeJoint`, `SpringJoint`,
    `JointMotor { max_force, .. }`, `ImpulseJoint::impulses`.
  * Rapier depends on `glamx` + `nalgebra`. **Check its `glam` version against Bevy's** at setup; keep all conversion at one boundary module.
* **Engine / rendering:** `bevy` (ECS + `wgpu`). Fixed-step physics decoupled from render; render transforms interpolated.
* **Debug draw:** Rapier `DebugRenderPipeline` → Bevy gizmos (we own this since we skip `bevy_rapier`).
* **Tuning UI:** egui inspector for materials, joint thresholds, motor gains, AI scores.

---

## 3. Physics Architecture: Three Layers + Armor Tiers

### Layer 1 — Rigid articulated skeleton & actuators
* Chassis, limb segments, locomotion modules, weapon shafts as rigid bodies.
* **Joint policy:**
  * **Multibody (reduced-coordinate) joints** for internal limb chains (arms, 4–5-joint legs): stable on long chains, tolerant of heavy-chassis / light-foot mass ratios.
  * **Impulse joints** for every *breakable connection* (limb↔chassis, armor bolts, weapon mounts, thruster pods), because their reaction impulses are directly readable (`ImpulseJoint::impulses`).
* Motors: `JointMotor` targets with `max_force` caps. Part stats map onto motor stiffness/damping/max_force (see "gear ratio", §5).

### Layer 2 — Armor, in three cost tiers
FEM is the biggest performance risk, so it is used only where it shows.

| Tier | Implementation | Where |
|---|---|---|
| **A — Hero FEM plates** | Volumetric tetrahedral soft body, `Fem` + `Corotational`, plasticity + `tear_strain`, attached via soft-frames → breakable bolt joints | 1–2 plates per robot (chest, one large panel) |
| **B — Shock-mounted plates** | Rigid plate on a `SpringJoint`/soft impulse joint with a break threshold (combat-robot "shock mount") | Most armor |
| **C — Cosmetic dents** | Rigid plate + visual vertex offset at impact points | Small/secondary panels |

**FEM material notes**
* `plastic_max` bounds plastic flow so crushed cells can't collapse to slivers (covers the "minimum element volume" guard).
* `deformation_damping` (a few 1/s) kills residual sway on stiff plates without damping rigid motion — use it against "jelly metal".
* Torn fragments: convert small pieces to rigid convex hulls or despawn quickly; don't keep them as soft bodies.
* Laser "softening": per-cluster `set_cluster_stiffness_scale` on heated clusters. Per-cluster plastic yield is **not** confirmed in the API — prototype before committing.

### Layer 3 — Visual skinning
* High-detail meshes bound via `volumetric_skinned`; update vertex buffers on render frames; recompute normals only for dirty meshes.

---

## 4. Stability Guardrails

1. **Fixed step:** Bevy `FixedUpdate` / `Time<Fixed>` at **60 Hz with 4 substeps** (pick this *or* 120 Hz, not both). Clamp accumulated frame delta.
2. **Post-step safety pass:** clamp linear/angular velocity per body class; `NaN`/`Inf` check → reset body to last good state and log.
3. **Mass ratios:** ≤ 10–20× across impulse joints. Multibody chains are more forgiving but feet still get real mass and wide colliders.
4. **CCD** on melee heads, slugs, grapple hooks, and feet during high-speed landings. Solid inner chassis collider under all armor.
5. **Safe detachment:** no artificial repulsion on break/tear; preserve velocities; brief collision grace for overlapping fragments; sleep then pool/despawn debris.
6. **Self-collision filtering:** collision groups so a robot's own weapons don't hit its own parts.
7. **Determinism:** not required (no multiplayer), but keep input record/replay working for reproducing physics bugs.

---

## 5. Robot Design & Modular Parts

### 5.1 Slots
| Slot | Examples | Notes |
|---|---|---|
| **Core / chassis** | Strut frame, box core, bulwark hull | Defines mount points, base mass |
| **Head / control core** | Mono-eye, twin sensor, armored dome | Drives control-law level (§8.4) |
| **Left arm / Right arm** | Autocannon, hammer, grapple, shield | Each arm = one mouse button |
| **Locomotion** | Wheels, treads, biped (4–5 joints/leg), hover skid | Wheels/hover first; walker is a spike |
| **Back** | Thruster pack, inertial tail, reaction wheel | Mid-air control options |
| **Armor** | Tier A/B/C plates on chassis/limbs | |

### 5.2 Build budget
* **Mass limit** and **energy limit** per robot. More armor ⇒ less weapon/thrust (combat-robot weight-budget trade-off).
* Heavy builds are physically slower to accelerate/turn — no stat penalty needed.

### 5.3 Part stats map to physics
* **Gear ratio** (low ↔ high): low = backdrivable, absorbs impacts, weaker; high = strong/precise, stiff, shears on impact. Maps to motor stiffness/damping and `max_force`. (MIT Cheetah "impact mitigation".)
* **Stiffness mode** (series-elastic / variable-stiffness analogue): parts can switch stiff↔soft — the basis of brace mode.
* **Shear threshold & durability pool** per mount.
* **Persistent part identity:** each part is an entity with an ID and saved damage state, so detached parts can be recovered and reattached as the *same* damaged part.

### 5.4 Locomotion
* Baseline: wheeled/skid or hover (stable, predictable).
* Walker: 2 legs × 4–5 joints as multibody chains. Balance via **Raibert foot placement** (neutral point) + **capture point** for step selection; capture point outside the support area ⇒ trigger stagger/knockdown.
* Part/DOF budget is relaxed from Rev. 1 (4–8 links) to allow walkers; cap total joints per robot at ~20 and profile.

---

## 6. Mobility & Impact Response

### 6.1 Thrusters
* Force at mount offsets each substep; off-centre mounts induce torque. Damaged pod ⇒ reduced/sputtering thrust; sheared pod ⇒ disabled.
* Draws from an **energy/boost gauge**. Clamp max thrust and terminal velocity.

### 6.2 Grapple / rope swing
* Phase 1: CCD hook projectile, slack tether.
* Phase 2: latch → anchor joint at contact; tension loads the target part's joint (can rip parts off).
* Phase 3: Rapier **`RopeJoint`** for max length. Soften the last few % of length (or cap rope impulse) so snapping taut doesn't jerk the anchor joint. Reel in/out via mouse wheel.

### 6.3 Mid-air reorientation
* Thrusters, or fuel-free via **inertial tail** / **reaction wheel** back parts (moving internal mass). These are breakable parts.

### 6.4 Bracing & impact response
* **Impact prediction:** shapecast the chassis along velocity each tick; if solid contact within ~0.3 s, cue the player (HUD flash, audio).
* **Two-phase brace (from humanoid fall research):**
  * *Pre-impact:* orient strongest armor or **feet** toward the surface; tuck arms; optional retro-thrust; temporarily lock forearms to chassis with a `FixedJoint` (load sharing).
  * *Impact:* switch motors to **soft** (low stiffness, high damping). Torque caps act as slip clutches.
* **Leg landings:** bent knees (never straight — axial load bypasses the motors); IK places feet on the predicted contact plane; leave joint range-of-motion headroom so limits don't take the spike.
* **Wall kick:** after absorption, stiffen and push off (release brace at contact).
* **Readable payoff:** while braced, break thresholds × 1.5–2. Full bonus within ~150 ms of impact, partial if early. Holding brace costs mobility (no attacks).
* Also works against enemy melee strikes.

---

## 7. Weapons

| Archetype | Physics | Armor & joint interaction |
|---|---|---|
| Autocannon / MG | Hitscan or swept shapecast per substep | Point impulses, cumulative joint fatigue |
| Heavy cannon / railgun | CCD rigid slug (pooled) | High momentum; craters, tears, shears joints |
| Shotgun / flak | Ray/shapecast fan | Spreads impulse across many bolts |
| Rockets / grenades | Thrust-driven projectile + sphere query | Falloff radial impulse, clamped peak |
| Laser / plasma | Continuous ray + cluster softening | Lowers stiffness in heated clusters |
| Harpoon / tether | CCD dart + breakable `RopeJoint` | Anchors so dashes can tear parts off |
| Spinner (melee) | Motor-driven flywheel rigid body | Stored energy dump on hit; gyroscopic turning resistance comes free |
| Hammer / blade (melee) | PD-tracked arc, torque-capped | Momentum drives denting and shear |

* **Recoil** applied at barrel transform; mounts and barrels are breakable/bendable.
* **Shock-mounted weapons** option: weapon on a spring mount to protect own frame.

---

## 8. Control Architecture

### 8.1 Layers
1. **Input** (keys/mouse) →
2. **Intent / command interface** (`Drive`, `Aim`, `Fire(L/R)`, `Melee`, `Brace`, `Boost`, `Grapple`, `Interact`…) — **shared by player and AI** →
3. **Behaviors** (poses, IK, motor-mode switches, thruster bursts) →
4. **Controllers** (PD motors with torque caps) →
5. **Physics.**

Missing parts make their commands no-ops (graceful degradation).

### 8.2 Reflexes / Skills / Direct control
| Class | Trigger | Examples | Rule |
|---|---|---|---|
| **Reflex** | Automatic | Balance, foot placement, upright assist, grapple grip | Invisible, reliable |
| **Skill** | Player, timed | Brace, melee, wall kick, grapple fire/release, overdrive | Few, consistent, one button each |
| **Direct** | Player, continuous | Drive, aim | Minimum layers in between |

Skills never become automatic for the player (AI and easy difficulty may auto-brace).

### 8.3 Feel rules
* **Same-frame feedback:** audio/VFX on the input frame even if physics responds later.
* **Latency budget:** < 100 ms input → visible motion for every skill. Measure it.
* **Authored poses over live IK** for skills; IK only for small corrections.
* **Assist forces:** upright torque, aim/melee assist, weaker than real impacts. Blend as `cmd = α·assist + (1−α)·player`, α higher when target intent is unambiguous.
* **Explain failure:** blocked swings and late braces get a clear stagger/VFX.
* **Camera** follows a smoothed chassis proxy, not raw physics.
* **Strain cues:** joint near break ⇒ creak audio + sparks (KB+M has no rumble).

### 8.4 Control laws (damage degrades assistance)
Inspired by fly-by-wire Normal/Alternate/Direct laws. Level is driven by the **head/control-core** part's health:

| Law | Player experience |
|---|---|
| **Normal** | Full assists: upright, aim assist, joint protection, auto foot placement |
| **Degraded** | Weaker stabilizer, reduced aim assist, joint protection off |
| **Direct** | Raw thrust/motors, no limits — hard to fly, but enables moves assists would block |

* Joint protection uses **soft stops**: holding melee (**overdrive**) pushes past safe torque at the risk of self-shearing. Self-shearing arms are intentional.
* Assist strength is also a menu setting (like drone Angle/Horizon/Acro modes).

### 8.5 Control map (keyboard + mouse)
| Input | Action |
|---|---|
| **W A S D** | Drive; steer thrust when airborne |
| **Mouse** | Aim (torso yaw + task-space arm aim) |
| **LMB / RMB** | Fire left / right arm weapon (lost arm ⇒ dead button) |
| **Space** | Jump / vertical thrust (hold); get up when knocked down |
| **Shift** | Boost / dash |
| **Ctrl (hold)** | Brace; release at contact = wall kick |
| **F** | Melee; hold = overdrive |
| **Q** | Fire grapple / release |
| **Mouse wheel** | Reel in/out while grappled |
| **MMB** | Lock-on toggle |
| **E** | Pick up / attach part |

Reserved if needed: **R** eject part, **Tab** cycle targets, **1–4** stance/thruster presets.
Optional: mouse motion during melee windup picks swing direction. Joint-level arm control is dropped (doesn't fit KB+M).

---

## 9. Damage, Self-Damage & Repair

* **Breakage reads peak load:** per-step impulse (or a short rolling peak) feeds each joint's **durability pool**; single-frame spikes are smoothed. Absorption (bracing, bent legs, shock mounts) therefore actually reduces damage.
* **Source weighting:** contact events tag hits as *enemy / arena / own part*; arena hits drain durability at a reduced factor (tunable, e.g. 0.3×). Physics is unchanged; only bookkeeping differs.
* **Self-damage is allowed:** crashes, grapple snaps, recoil fatigue and overdriven motors can break your own parts. Tuning targets: top self-propelled speed dents but doesn't shear core joints; overdrive can shear arms.
* **Crumple zones:** front bumpers absorb crash energy before core joints.
* **Repair:** detached parts persist (ID + damage state). Press **E** near a part to pull it in and reattach (supervisory "context action": player picks the target, robot executes). Reattachment creates a new breakable joint at the slot's mount. Losing/recovering the control core changes the control law.

---

## 10. Enemy AI

* **Same command interface as the player** (§8.1). No physics cheats; damage degrades AI exactly like players.
* **Tactical layer (5–10 Hz):** utility AI scoring *approach, hold range & shoot, melee, retreat, recover part, deny player's part, grapple*. Inputs: own damage/law level, target damage, range, energy.
* **Steering (every tick):** seek/strafe/avoid via raycasts; add navmesh (`oxidized_navigation`) only if arenas need it.
* **Aim/attack:** projectile lead, range-based melee selection, pre-fire for turn lag.
* **Exploit damage data:** target near-break bolts/joints.
* **Self-preservation:** use impact prediction + thresholds to brace or avoid suicidal rams (with deliberate occasional mistakes).
* **Grapple AI last;** start with grapple as a pull-to-target attack.
* **Difficulty:** reaction delay, aim error, re-decision rate, weak-spot knowledge, brace timing — not HP/damage multipliers.
* **Debug:** gizmo overlay of chosen action, scores, targets.
* Keep the sim headless-capable so ML/RL tuning stays possible later.

---

## 11. Scope

**v1 (vertical slice)**
* 1v1 vs AI in 1–2 arenas. No missions/story/multiplayer.
* One frame family with swappable parts: ~2–3 options per slot.
* Wheels or hover locomotion; walker as a parallel spike.
* FEM on 1–2 plates per robot; the rest tier B/C.

**Non-goals (for now):** multiplayer/netcode, cross-platform determinism, campaign content, gamepad-specific features.

---

## 12. Roadmap

0. **Tooling (continuous):** debug render, egui inspector, input record/replay, perf HUD.
1. **M1 — Sandbox, feel & perf gate**
   * Bevy + Rapier resource, `FixedUpdate` 60 Hz × 4 substeps, render interpolation, glam boundary.
   * Arena, wheeled/hover chassis, mouse aim, smoothed camera, upright assist, boost gauge.
   * **FEM stress test:** N plates × M cells × 2 robots — record ms/step. Decides tier A budget.
   * **Gate:** driving and aiming alone must feel good.
2. **M2 — Breakage, melee, first weapons, first AI**
   * Impulse-joint breakage with peak-load durability pools and source weighting; self-collision groups.
   * `Windup → Strike → Recovery` PD arcs with torque caps and overdrive; hitscan weapon; thrusters with mount offsets.
   * Basic AI: approach + swing + shoot.
3. **M3 — Impact response, parts & repair**
   * Impact prediction, brace (two-phase, threshold multiplier, timing window), wall kick.
   * Part slots, mass/energy budget, persistent part identity, pick-up/reattach.
   * Control-law degradation via control core. AI utility layer expanded.
4. **M4 — Armor tiers**
   * Tier B shock-mounted plates; tier A FEM hero plates (soft-frames + bolts, dirty-normal skinning); tier C dent offsets; fragment conversion/despawn.
5. **M5 — Advanced mobility & weapons**
   * Grapple (`RopeJoint`, reel), slugs, rockets, shotgun, laser softening, recoil, spinner.
   * Walker locomotion (multibody legs, Raibert/capture point, leg-first landings); tail / reaction-wheel parts.
   * AI: grapple, part denial.

---

## 13. Research References

**Actuation & impact**
* MIT Cheetah proprioceptive actuators / impact mitigation factor — https://dspace.mit.edu/handle/1721.1/119863
* Variable-stiffness actuator survey (Grioli, Wolf et al. 2015) — https://www.centropiaggio.unipi.it/sites/default/files/2015_gwg_ijrr.pdf
* Overload protection in humanoid joints — https://humanoid.guide/?p=18827

**Falling & landing**
* Tripod Fall (Yun & Goswami) — https://www.sri.com/publication/information-computer-science-pubs/tripod-fall-concept-and-experiments-of-a-novel-approach-to-humanoid-robot-fall-damage-reduction/
* Fall on Backpack (Lee & Goswami) — https://lava.kaist.ac.kr/wp-content/uploads/2017/06/2013_JCND_Lee_Goswami_Humanoid_Fall_on_Backpack.pdf
* Damage Reduction Strategies for a Falling Humanoid (VT / ESCHER) — https://vtechworks.lib.vt.edu/handle/10919/78765

**Balance & reorientation**
* Raibert heuristic / simplest balance controller — https://arxiv.org/pdf/2211.06223
* Quadruped with inertial tail (mid-air righting) — https://arxiv.org/abs/2209.15337

**Combat robotics**
* RioBotz Combot Tutorial (energy, shock mounts) — https://www.scribd.com/doc/65786336/Riobotz-Combot-Tutorial

**Piloting & control abstraction**
* Eagle Prime two-pilot controls (IEEE Spectrum) — https://spectrum.ieee.org/watch-the-worlds-first-giant-robot-fight
* Spot operation — https://www.generationrobots.com/media/spot-boston-dynamics/spot-user-guide-r2.0-va.pdf
* Excavator coordinated control — https://imdl.gatech.edu/currentproj/haptic_interface/haptic_interface.html
* Fly-by-wire control laws — https://skybrary.aero/articles/fly-wire
* FPV flight modes — https://betaflight.com/docs/wiki/guides/current/Modes
* Shared autonomy blending — https://arxiv.org/html/1503.05451
* Affordance templates / supervisory control (NASA Valkyrie) — https://ntrs.nasa.gov/citations/20140012413

---

## 14. Open Questions
1. Tier A FEM budget — answered by the M1 stress test.
2. Is walking core to the identity (prototype earlier) or a later locomotion option?
3. Arena-hit damage factor and brace multiplier values — playtest.
4. Does mouse-directed melee swing direction stay, or fixed arcs only?
5. Glam version alignment between Rapier 0.36.1 and the chosen Bevy release.
