//! Mech-V frame, built from standard parts so that separate parts only meet at a joint:
//!
//! * **Joint nodes**: `ball` (sphere: shoulders, hips, ankles) and `hinge` (cylinder knuckle: waist, knees,
//!   hocks, elbows). Every limb segment ends 1 cm short of the joint node it attaches to.
//! * **Segments**: tapered boxes (`v_*`), sized between two joint nodes.
//! * **Plates**: pauldron cap/side plates, groin plate, loads.
//! * **Chest**: V-tapered, with its front face rising to a point (pyramid on the front-face corners).
//!
//! Torso has three connections: two ball shoulders on side sockets and a yaw-only waist hinge at the
//! bottom. The pelvis carries two ball hips on its sides and a groin plate between them. Head is fixed.
//!
//! Weight classes change the *structure*: every part's cross-section is scaled (Light 0.8×, Medium 1×,
//! Heavy 1.25×; joint nodes 0.9× / 1× / 1.12×) with mass following cross-section area, and hips and
//! shoulders spread to keep clearance. Medium/Heavy also carry weapons and back units.
//! Joint positions (heights, limb lengths) are identical across classes.

use crate::robot::*;
use rapier3d::prelude::*;

const GAP: f32 = 0.01;

fn span(top: f32, bottom: f32) -> (f32, f32) {
    (0.5 * (top + bottom), 0.5 * (top - bottom))
}

pub fn build_v(w: &mut PhysicsWorld, o: Vector, spec: RobotSpec) -> Robot {
    let mut links = Vec::new();
    let (_, hip_y, pelvis_y) = spec.hip();
    let ([l1, l2, l3], ankle_y) = spec.segments();
    let (k, kn) = match spec.load {
        Load::Light => (0.8f32, 0.9f32),
        Load::Medium => (1.0, 1.0),
        Load::Heavy => (1.25, 1.12),
    };
    let km = k * k; // mass follows cross-section area
    let z = Vector::ZERO;
    let p = |half: [f32; 3], offset: [f32; 3], mass: f32, tag: &'static str| Part { half, offset, mass, tag };
    let at = |x: f32, y: f32, zz: f32| o + Vector::new(x, y, zz);
    // Thickened half-extents: x/z scaled, y (length) kept.
    let th = |h: [f32; 3]| [h[0] * k, h[1], h[2] * k];

    // ---------------- pelvis: hip balls on the sides, groin plate below ----------------
    let pelvis_pos = at(0.0, pelvis_y, 0.0);
    let pelvis_half = th([0.26, 0.14, 0.22]);
    let mut pelvis_parts = vec![
        p(pelvis_half, [0.0; 3], 430.0 * km, "v_pelvis"),
        p([0.20 * k, 0.07, 0.03], [0.0, 0.0, pelvis_half[2] + 0.03], 150.0, "reactor"),
        p(th([0.12, 0.13, 0.14]), [0.0, -pelvis_half[1] - 0.13, 0.02], 20.0 * km, "v_groin"),
    ];
    // Rear hip-skirt thrusters: they sit just behind the top of each thigh but are fixed to the
    // pelvis, so firing them pushes the body rather than yanking the leg out of its stride.
    let hip_x0 = pelvis_half[0] + GAP + 0.15 * kn;
    let skirt_z = -(0.15 * kn + GAP + 0.09);
    let skirt_y = hip_y - pelvis_y + 0.04;
    if spec.thrusters {
        for side in [1.0f32, -1.0] {
            pelvis_parts.push(p([0.07, 0.05, 0.05], [side * hip_x0, skirt_y + 0.07, skirt_z + 0.03], 15.0, "v_skirt"));
            pelvis_parts.push(p([0.06, 0.08, 0.07], [side * hip_x0, skirt_y - 0.05, skirt_z], 30.0, "thruster"));
        }
    }
    let (pelvis, _) = add_body(w, &mut links, "pelvis".into(), pelvis_pos, &pelvis_parts, true);

    // ---------------- upper body on a yaw-only waist hinge ----------------
    let upper_y = 2.72;
    let upper_pos = at(0.0, upper_y, 0.0);
    let waist_y = pelvis_y + pelvis_half[1] + GAP + 0.07;
    let chest_half = th([0.46, 0.30, 0.30]);
    let shoulder_y = 2.92;
    let ball_r_sh = 0.14 * kn;
    let socket_half_x = 0.07;
    // Chest half-width at the shoulder height (V taper 1.0 at top, 0.55 at bottom).
    let t_sh = 0.55 + 0.45 * ((shoulder_y - upper_y) / chest_half[1] + 1.0) * 0.5;
    let socket_inner = chest_half[0] * t_sh - 0.02;
    let shoulder_x = socket_inner + 2.0 * socket_half_x + GAP + ball_r_sh;
    let mut upper_parts = vec![
        p([0.14 * k, 0.07, 0.14 * k], [0.0, waist_y - upper_y, 0.0], 40.0 * km, "hinge_y"),
        p(chest_half, [0.0, 0.0, 0.0], 210.0 * km, "v_chest"),
        p([socket_half_x, 0.10 * kn, 0.14 * k], [socket_inner + socket_half_x, shoulder_y - upper_y, 0.0], 10.0 * km, "torso"),
        p([socket_half_x, 0.10 * kn, 0.14 * k], [-(socket_inner + socket_half_x), shoulder_y - upper_y, 0.0], 10.0 * km, "torso"),
        p([0.07 * k, 0.04, 0.07 * k], [0.0, chest_half[1] + 0.04, 0.02], 5.0, "torso"),
        p([0.10, 0.11, 0.13], [0.0, chest_half[1] + 0.08 + 0.11, 0.05], 30.0, "head"),
    ];
    let pack_depth = match spec.load {
        Load::Light => 0.0,
        Load::Medium => 0.28,
        Load::Heavy => 0.40,
    };
    match spec.load {
        Load::Light => {}
        Load::Medium => upper_parts.push(p([0.30, 0.28, 0.14], [0.0, 0.04, -chest_half[2] - 0.15], 140.0, "pack")),
        Load::Heavy => upper_parts.push(p([0.44, 0.34, 0.20], [0.0, 0.06, -chest_half[2] - 0.21], 260.0, "pack")),
    }
    // Thruster mounts on the upper body: (name, local position, nominal force direction, half-size, mass, max force at reference weight).
    let apex_z = chest_half[2] * 0.5 * (1.0 + 0.55) + 0.55 * chest_half[2];
    let socket_c = socket_inner + 0.04;
    let upper_thr: Vec<(&'static str, [f32; 3], [f32; 3], [f32; 3], f32, f32)> = if spec.thrusters {
        vec![
            ("shoulder_L", [socket_c, shoulder_y - upper_y + 0.10 * kn + 0.08, 0.0], [0.0, 1.0, 0.0], [0.06, 0.07, 0.08], 30.0, 9_000.0),
            ("shoulder_R", [-socket_c, shoulder_y - upper_y + 0.10 * kn + 0.08, 0.0], [0.0, 1.0, 0.0], [0.06, 0.07, 0.08], 30.0, 9_000.0),
            ("back_L", [0.20 * k, 0.10, -chest_half[2] - pack_depth - 0.09], [0.0, 0.0, 1.0], [0.07, 0.09, 0.08], 30.0, 6_000.0),
            ("back_R", [-0.20 * k, 0.10, -chest_half[2] - pack_depth - 0.09], [0.0, 0.0, 1.0], [0.07, 0.09, 0.08], 30.0, 6_000.0),
            ("chest", [0.0, 0.0, apex_z + 0.08], [0.0, 0.0, -1.0], [0.11, 0.11, 0.10], 60.0, 10_000.0),
        ]
    } else {
        vec![]
    };
    for (_, pos, _, half, m, _) in &upper_thr {
        upper_parts.push(p(*half, *pos, *m, "thruster"));
    }
    let (upper, _) = add_body(w, &mut links, "upper_body".into(), upper_pos, &upper_parts, true);
    let wj = revolute(w, pelvis, upper, Vector::Y, Vector::new(0.0, waist_y - pelvis_y, 0.0), Vector::new(0.0, waist_y - upper_y, 0.0), Some([-0.8, 0.8]), MAX_TORQUE[WAIST]);

    // ---------------- legs ----------------
    let (r_hip, r_knee, r_hock, r_ankle) = (0.15 * kn, 0.10 * kn, 0.09 * kn, 0.09 * kn);
    let hip_x = pelvis_half[0] + GAP + r_hip;
    let knee_y = hip_y - l1;
    let hock_y = knee_y - l2;
    let foot_half = [0.26 * k * spec.foot_scale, 0.07, 0.42 * spec.foot_scale * (0.5 + 0.5 * k)];
    let mk_leg = |w: &mut PhysicsWorld, links: &mut Vec<LinkInfo>, side: f32| -> Leg {
        let n = if side > 0.0 { "L" } else { "R" };
        let x = side * hip_x;
        let hip = at(x, hip_y, 0.0);
        let (yg, _) = add_body(w, links, format!("{n}_hip_ball"), hip, &[p([r_hip; 3], [0.0; 3], 40.0, "ball")], false);
        let (rg, _) = add_body(w, links, format!("{n}_hip_roll"), hip, &[p([0.04; 3], [0.0; 3], 40.0, "gimbal")], false);
        let j0 = revolute(w, pelvis, yg, Vector::Y, hip - pelvis_pos, z, Some([-0.5, 0.5]), MAX_TORQUE[HIP_YAW]);
        let j1 = revolute(w, yg, rg, Vector::Z, z, z, Some([-0.6, 0.6]), MAX_TORQUE[HIP_ROLL]);
        let (cy, hh) = span(hip_y - r_hip - GAP, knee_y + r_knee + GAP);
        let thigh_half = th([0.11, hh, 0.14]);
        let mut thigh_parts = vec![p(thigh_half, [0.0; 3], 240.0 * km, "v_thigh")];
        let (thb, _) = add_body(w, links, format!("{n}_thigh"), at(x, cy, 0.0), &thigh_parts, true);
        let j2 = revolute(w, rg, thb, Vector::X, z, Vector::new(0.0, hip_y - cy, 0.0), Some([-1.8, 1.0]), MAX_TORQUE[HIP_PITCH]);
        let (cy2, hh2) = span(knee_y - r_knee - GAP, hock_y + r_hock + GAP);
        let shin_half = th([0.16, hh2, 0.19]);
        let knee_thr = [0.0, hh2 - 0.10, shin_half[2] * 1.0 + GAP + 0.06];
        let mut shin_parts = vec![
            p(shin_half, [0.0; 3], 180.0 * km, "v_shin"),
            p([r_knee, 0.15 * k, r_knee], [0.0, knee_y - cy2, 0.0], 10.0, "hinge_x"),
        ];
        if spec.thrusters {
            shin_parts.push(p([0.07, 0.07, 0.06], knee_thr, 30.0, "thruster"));
        }
        let (sh, _) = add_body(w, links, format!("{n}_shin"), at(x, cy2, 0.0), &shin_parts, true);
        let j3 = revolute(w, thb, sh, Vector::X, Vector::new(0.0, knee_y - cy, 0.0), Vector::new(0.0, knee_y - cy2, 0.0), Some([-0.02, 2.4]), MAX_TORQUE[KNEE]);
        let (cy3, hh3) = span(hock_y - r_hock - GAP, ankle_y + r_ankle + GAP);
        let (mt, _) = add_body(w, links, format!("{n}_meta"), at(x, cy3, 0.0), &[
            p(th([0.16, hh3, 0.19]), [0.0; 3], 117.0 * km, "v_meta"),
            p([r_hock, 0.14 * k, r_hock], [0.0, hock_y - cy3, 0.0], 8.0, "hinge_x"),
        ], true);
        let jm = revolute(w, sh, mt, Vector::X, Vector::new(0.0, hock_y - cy2, 0.0), Vector::new(0.0, hock_y - cy3, 0.0), Some([-2.4, 0.3]), MAX_TORQUE[META]);
        let apos = at(x, ankle_y, 0.0);
        let (ag, _) = add_body(w, links, format!("{n}_ankle_ball"), apos, &[p([r_ankle; 3], [0.0; 3], 30.0, "ball")], false);
        let j4 = revolute(w, mt, ag, Vector::X, Vector::new(0.0, ankle_y - cy3, 0.0), z, Some([-1.2, 1.2]), MAX_TORQUE[ANKLE_PITCH]);
        let foot_c = at(x, ankle_y - r_ankle - GAP - foot_half[1], 0.12);
        let (ft, fc) = add_body(w, links, format!("{n}_foot"), foot_c, &[p(foot_half, [0.0; 3], 110.0 * km, "foot")], true);
        let j5 = revolute(w, ag, ft, Vector::Z, z, apos - foot_c, Some([-0.5, 0.5]), MAX_TORQUE[ANKLE_ROLL]);
        Leg { side, joints: [j0, j1, j2, j3, j4, j5], bodies: [yg, rg, thb, sh, ag, ft], meta: Some((jm, mt)), foot_collider: fc[0] }
    };
    let left = mk_leg(w, &mut links, 1.0);
    let right = mk_leg(w, &mut links, -1.0);

    // ---------------- arms: ball shoulder (roll, pitch, twist) + elbow hinge ----------------
    let elbow_y = shoulder_y - 0.62;
    let r_elbow = 0.09 * kn;
    let mk_arm = |w: &mut PhysicsWorld, links: &mut Vec<LinkInfo>, side: f32| -> Arm {
        let n = if side > 0.0 { "L" } else { "R" };
        let sh_local = Vector::new(side * shoulder_x, shoulder_y - upper_y, 0.0);
        let shoulder = upper_pos + sh_local;
        let (g1, _) = add_body(w, links, format!("{n}_shoulder_ball"), shoulder, &[p([ball_r_sh; 3], [0.0; 3], 15.0, "ball")], false);
        let jr = revolute(w, upper, g1, Vector::Z, sh_local, z, Some([-0.1, 1.0]), MAX_TORQUE[ARM]);
        let (g2, _) = add_body(w, links, format!("{n}_shoulder_pitch"), shoulder, &[p([0.03; 3], [0.0; 3], 5.0, "gimbal")], false);
        let jp = revolute(w, g1, g2, Vector::X, z, z, Some([-1.4, 1.2]), MAX_TORQUE[ARM]);
        let (cy, hh) = span(shoulder_y - ball_r_sh - GAP, elbow_y + r_elbow + GAP);
        let cap_half = [0.17 * k, 0.06 * kn, 0.24 * k];
        let side_half = [0.05 * k, 0.20 * kn, 0.22 * k];
        let mut ua_parts = vec![
            p(th([0.12, hh, 0.14]), [0.0; 3], 30.0 * km, "arm"),
            p(cap_half, [side * 0.06 * k, shoulder_y + ball_r_sh + GAP + cap_half[1] - cy, 0.0], 8.0 * km, "v_pauldron"),
            p(side_half, [side * (ball_r_sh + GAP + side_half[0]), shoulder_y - cy, 0.0], 6.0 * km, "v_pauldron"),
        ];
        if spec.load == Load::Heavy {
            let a = [0.08, 0.22, 0.24];
            ua_parts.push(p(a, [side * (ball_r_sh + GAP + 2.0 * side_half[0] + GAP + a[0]), shoulder_y - 0.08 - cy, 0.0], 100.0, "armor"));
        }
        let (ua, _) = add_body(w, links, format!("{n}_upper_arm"), at(side * shoulder_x, cy, 0.0), &ua_parts, true);
        let jy = revolute(w, g2, ua, Vector::Y, z, Vector::new(0.0, shoulder_y - cy, 0.0), Some([-0.8, 0.8]), MAX_TORQUE[ARM]);
        let (cy2, hh2) = span(elbow_y - r_elbow - GAP, elbow_y - 0.70);
        let fa_half = th([0.14, hh2, 0.16]);
        let mut fa_parts = vec![
            p(fa_half, [0.0; 3], 40.0 * km, "v_forearm"),
            p([r_elbow, 0.13 * k, r_elbow], [0.0, elbow_y - cy2, 0.0], 5.0, "hinge_x"),
        ];
        let fa_outer = fa_half[0] * 1.2 + GAP;
        match spec.load {
            Load::Light => {}
            Load::Medium => fa_parts.push(p([0.09, 0.32, 0.11], [side * (fa_outer + 0.09), -0.04, 0.08], 80.0, "weapon")),
            Load::Heavy => fa_parts.push(p([0.12, 0.40, 0.15], [side * (fa_outer + 0.12), -0.08, 0.10], 120.0, "weapon")),
        }
        let (fa, _) = add_body(w, links, format!("{n}_forearm"), at(side * shoulder_x, cy2, 0.0), &fa_parts, true);
        let je = revolute(w, ua, fa, Vector::X, Vector::new(0.0, elbow_y - cy, 0.0), Vector::new(0.0, elbow_y - cy2, 0.0), Some([-2.2, 0.05]), MAX_TORQUE[ARM]);
        Arm { pitch: (jp, ua), roll: Some(jr), yaw: Some(jy), elbow: Some((je, fa)) }
    };
    let arms = Some([mk_arm(w, &mut links, 1.0), mk_arm(w, &mut links, -1.0)]);

    // ---------------- reaction wheels behind the pelvis ----------------
    let mut wheels = Vec::new();
    if spec.reaction_wheels {
        for (axis, half, off, name) in [
            (Vector::Z, [0.20, 0.20, 0.05], [0.0, 0.0, -pelvis_half[2] - GAP - 0.05], "roll_wheel"),
            (Vector::X, [0.05, 0.20, 0.20], [0.0, 0.0, -pelvis_half[2] - GAP - 0.10 - GAP - 0.20], "pitch_wheel"),
        ] {
            let pos = pelvis_pos + Vector::new(off[0], off[1], off[2]);
            let (b, _) = add_body(w, &mut links, name.into(), pos, &[p(half, [0.0; 3], 100.0, "wheel")], false);
            let j = RevoluteJointBuilder::new(axis)
                .local_anchor1(Vector::new(off[0], off[1], off[2]))
                .local_anchor2(z)
                .contacts_enabled(false)
                .motor_model(MotorModel::ForceBased)
                .motor_velocity(0.0, 0.0)
                .motor_max_force(25_000.0);
            let jh = w.insert_multibody_joint(pelvis, b, j).expect("tree");
            wheels.push(Wheel { joint: jh, body: b, axis });
        }
    }

    let total_mass: f32 = links.iter().map(|l| w.bodies[l.body].mass()).sum();
    let motor_scale = spec.strength * total_mass / 3560.0;
    // Thrust scales with weight (forces listed for a 2.1 t frame), like the motors.
    let fscale = total_mass / 2087.0;
    let mut thrusters = Vec::new();
    if spec.thrusters {
        let v = |a: [f32; 3]| Vector::new(a[0], a[1], a[2]);
        for (name, pos, dir, _, _, f) in &upper_thr {
            thrusters.push(Thruster { name, body: upper, local_pos: v(*pos), local_dir: v(*dir), cone: std::f32::consts::FRAC_PI_2, max_force: f * fscale, dash_force: 2.5 * f * fscale });
        }
        for (i, leg) in [&left, &right].iter().enumerate() {
            let (tn, kn_) = if i == 0 { ("thigh_L", "knee_L") } else { ("thigh_R", "knee_R") };
            let side = if i == 0 { 1.0 } else { -1.0 };
            thrusters.push(Thruster { name: tn, body: pelvis, local_pos: Vector::new(side * hip_x0, skirt_y - 0.05, skirt_z - 0.07), local_dir: Vector::Z, cone: std::f32::consts::FRAC_PI_2, max_force: 4_000.0 * fscale, dash_force: 10_000.0 * fscale });
            let (_, hh2) = span(knee_y - r_knee - GAP, hock_y + r_hock + GAP);
            thrusters.push(Thruster { name: kn_, body: leg.bodies[B_SHIN], local_pos: Vector::new(0.0, hh2 - 0.10, 0.19 * k + GAP + 0.06), local_dir: -Vector::Z, cone: std::f32::consts::FRAC_PI_2, max_force: 4_000.0 * fscale, dash_force: 10_000.0 * fscale });
        }
    }
    Robot { spec, thrusters, trunk: pelvis, legs: [left, right], arms, wheels, links, total_mass, waist: Some((wj, upper)), motor_scale }
}
