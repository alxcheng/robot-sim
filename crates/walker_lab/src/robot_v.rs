//! Mech-V frame, built from standard parts so that separate parts only meet at a joint:
//!
//! * **Joint nodes**: `ball` (sphere: shoulders, hips, ankles) and `hinge` (cylinder knuckle: waist, knees,
//!   hocks, elbows). Every limb segment ends 1 cm short of the joint node it attaches to.
//! * **Segments**: tapered boxes (`v_*`), sized between two joint nodes.
//! * **Plates**: pauldron cap/side plates, groin plate, pyramid chest point, loads.
//!
//! Torso has three connections: two ball shoulders on side sockets and a yaw-only waist hinge at the
//! bottom. The pelvis carries two ball hips on its sides and a groin plate between them. Head is fixed.
//!
//! Loads (swap tests): Light = bare; Medium = forearm weapons + back pack (+300 kg);
//! Heavy = bigger weapons + back unit + outer arm armor (+700 kg). Load mass sits on the upper body.

use crate::robot::*;
use rapier3d::prelude::*;

const GAP: f32 = 0.01;

/// Centre and half-height of a segment spanning [bottom, top].
fn span(top: f32, bottom: f32) -> (f32, f32) {
    (0.5 * (top + bottom), 0.5 * (top - bottom))
}

pub fn build_v(w: &mut PhysicsWorld, o: Vector, spec: RobotSpec) -> Robot {
    let mut links = Vec::new();
    let (hip_x, hip_y, pelvis_y) = spec.hip();
    let ([l1, l2, l3], ankle_y) = spec.segments();
    let z = Vector::ZERO;
    let p = |half: [f32; 3], offset: [f32; 3], mass: f32, tag: &'static str| Part { half, offset, mass, tag };
    let at = |x: f32, y: f32, zz: f32| o + Vector::new(x, y, zz);

    // ---------------- pelvis: hip balls on the sides, groin plate below ----------------
    let pelvis_pos = at(0.0, pelvis_y, 0.0);
    let pelvis_half = [0.26, 0.14, 0.22];
    let (pelvis, _) = add_body(w, &mut links, "pelvis".into(), pelvis_pos, &[
        p(pelvis_half, [0.0; 3], 430.0, "v_pelvis"),
        p([0.20, 0.07, 0.03], [0.0, 0.0, pelvis_half[2] + 0.03], 150.0, "reactor"),
        // Groin plate: hangs from the pelvis between the thighs.
        p([0.12, 0.13, 0.14], [0.0, -pelvis_half[1] - 0.13, 0.02], 20.0, "v_groin"),
    ], true);

    // ---------------- upper body on a yaw-only waist hinge ----------------
    let upper_y = 2.72;
    let upper_pos = at(0.0, upper_y, 0.0);
    let waist_y = pelvis_y + pelvis_half[1] + GAP + 0.07; // hinge knuckle sits on the pelvis top
    let chest_half = [0.46, 0.30, 0.30];
    let shoulder_y = 2.92;
    let ball_r_sh = 0.14;
    let shoulder_x = 0.74;
    let socket_outer = shoulder_x - ball_r_sh - GAP;
    let socket_half_x = 0.07;
    let mut upper_parts = vec![
        p([0.14, 0.07, 0.14], [0.0, waist_y - upper_y, 0.0], 40.0, "hinge_y"),
        p(chest_half, [0.0, 0.0, 0.0], 190.0, "v_chest"),
        // Pyramid point on the chest front.
        p([0.30, 0.22, 0.24], [0.0, 0.04, 0.20], 20.0, "pyramid"),
        // Shoulder sockets: part of the torso, ending 1 cm short of the shoulder balls.
        p([socket_half_x, 0.10, 0.14], [socket_outer - socket_half_x, shoulder_y - upper_y, 0.0], 10.0, "torso"),
        p([socket_half_x, 0.10, 0.14], [-(socket_outer - socket_half_x), shoulder_y - upper_y, 0.0], 10.0, "torso"),
        p([0.07, 0.04, 0.07], [0.0, chest_half[1] + 0.04, 0.02], 5.0, "torso"),
        p([0.10, 0.11, 0.13], [0.0, chest_half[1] + 0.08 + 0.11, 0.05], 30.0, "head"),
    ];
    match spec.load {
        Load::Light => {}
        Load::Medium => upper_parts.push(p([0.30, 0.28, 0.14], [0.0, 0.04, -chest_half[2] - 0.15], 140.0, "pack")),
        Load::Heavy => upper_parts.push(p([0.40, 0.34, 0.20], [0.0, 0.06, -chest_half[2] - 0.21], 260.0, "pack")),
    }
    let (upper, _) = add_body(w, &mut links, "upper_body".into(), upper_pos, &upper_parts, true);
    let wj = revolute(w, pelvis, upper, Vector::Y, Vector::new(0.0, waist_y - pelvis_y, 0.0), Vector::new(0.0, waist_y - upper_y, 0.0), Some([-0.8, 0.8]), MAX_TORQUE[WAIST]);

    // ---------------- legs ----------------
    let (r_hip, r_knee, r_hock, r_ankle) = (0.15, 0.10, 0.09, 0.09);
    let knee_y = hip_y - l1;
    let hock_y = knee_y - l2;
    debug_assert!((hock_y - l3 - ankle_y).abs() < 1e-3);
    let foot_half = [0.26 * spec.foot_scale, 0.07, 0.42 * spec.foot_scale];
    let mk_leg = |w: &mut PhysicsWorld, links: &mut Vec<LinkInfo>, side: f32| -> Leg {
        let n = if side > 0.0 { "L" } else { "R" };
        let x = side * hip_x;
        let hip = at(x, hip_y, 0.0);
        let (yg, _) = add_body(w, links, format!("{n}_hip_ball"), hip, &[p([r_hip; 3], [0.0; 3], 40.0, "ball")], false);
        let (rg, _) = add_body(w, links, format!("{n}_hip_roll"), hip, &[p([0.04; 3], [0.0; 3], 40.0, "gimbal")], false);
        let j0 = revolute(w, pelvis, yg, Vector::Y, hip - pelvis_pos, z, Some([-0.5, 0.5]), MAX_TORQUE[HIP_YAW]);
        let j1 = revolute(w, yg, rg, Vector::Z, z, z, Some([-0.6, 0.6]), MAX_TORQUE[HIP_ROLL]);
        // Thigh: from below the hip ball to above the knee knuckle.
        let (cy, hh) = span(hip_y - r_hip - GAP, knee_y + r_knee + GAP);
        let (th, _) = add_body(w, links, format!("{n}_thigh"), at(x, cy, 0.0), &[p([0.11, hh, 0.14], [0.0; 3], 240.0, "v_thigh")], true);
        let j2 = revolute(w, rg, th, Vector::X, z, Vector::new(0.0, hip_y - cy, 0.0), Some([-1.8, 1.0]), MAX_TORQUE[HIP_PITCH]);
        // Shin carries the knee knuckle.
        let (cy2, hh2) = span(knee_y - r_knee - GAP, hock_y + r_hock + GAP);
        let (sh, _) = add_body(w, links, format!("{n}_shin"), at(x, cy2, 0.0), &[
            p([0.16, hh2, 0.19], [0.0; 3], 180.0, "v_shin"),
            p([r_knee, 0.15, r_knee], [0.0, knee_y - cy2, 0.0], 10.0, "hinge_x"),
        ], true);
        let j3 = revolute(w, th, sh, Vector::X, Vector::new(0.0, knee_y - cy, 0.0), Vector::new(0.0, knee_y - cy2, 0.0), Some([-0.02, 2.4]), MAX_TORQUE[KNEE]);
        // Cannon bone (metatarsal) carries the hock knuckle.
        let (cy3, hh3) = span(hock_y - r_hock - GAP, ankle_y + r_ankle + GAP);
        let (mt, _) = add_body(w, links, format!("{n}_meta"), at(x, cy3, 0.0), &[
            p([0.16, hh3, 0.19], [0.0; 3], 117.0, "v_meta"),
            p([r_hock, 0.14, r_hock], [0.0, hock_y - cy3, 0.0], 8.0, "hinge_x"),
        ], true);
        let jm = revolute(w, sh, mt, Vector::X, Vector::new(0.0, hock_y - cy2, 0.0), Vector::new(0.0, hock_y - cy3, 0.0), Some([-2.4, 0.3]), MAX_TORQUE[META]);
        let apos = at(x, ankle_y, 0.0);
        let (ag, _) = add_body(w, links, format!("{n}_ankle_ball"), apos, &[p([r_ankle; 3], [0.0; 3], 30.0, "ball")], false);
        let j4 = revolute(w, mt, ag, Vector::X, Vector::new(0.0, ankle_y - cy3, 0.0), z, Some([-1.2, 1.2]), MAX_TORQUE[ANKLE_PITCH]);
        // Foot top sits 1 cm under the ankle ball.
        let foot_c = at(x, ankle_y - r_ankle - GAP - foot_half[1], 0.12);
        let (ft, fc) = add_body(w, links, format!("{n}_foot"), foot_c, &[p(foot_half, [0.0; 3], 110.0, "foot")], true);
        let j5 = revolute(w, ag, ft, Vector::Z, z, apos - foot_c, Some([-0.5, 0.5]), MAX_TORQUE[ANKLE_ROLL]);
        Leg { side, joints: [j0, j1, j2, j3, j4, j5], bodies: [yg, rg, th, sh, ag, ft], meta: Some((jm, mt)), foot_collider: fc[0] }
    };
    let left = mk_leg(w, &mut links, 1.0);
    let right = mk_leg(w, &mut links, -1.0);

    // ---------------- arms: ball shoulder (roll, pitch, twist) + elbow hinge ----------------
    let elbow_y = shoulder_y - 0.62;
    let r_elbow = 0.09;
    let mk_arm = |w: &mut PhysicsWorld, links: &mut Vec<LinkInfo>, side: f32| -> Arm {
        let n = if side > 0.0 { "L" } else { "R" };
        let sh_local = Vector::new(side * shoulder_x, shoulder_y - upper_y, 0.0);
        let shoulder = upper_pos + sh_local;
        let (g1, _) = add_body(w, links, format!("{n}_shoulder_ball"), shoulder, &[p([ball_r_sh; 3], [0.0; 3], 15.0, "ball")], false);
        let jr = revolute(w, upper, g1, Vector::Z, sh_local, z, Some([-0.1, 1.0]), MAX_TORQUE[ARM]);
        let (g2, _) = add_body(w, links, format!("{n}_shoulder_pitch"), shoulder, &[p([0.03; 3], [0.0; 3], 5.0, "gimbal")], false);
        let jp = revolute(w, g1, g2, Vector::X, z, z, Some([-1.4, 1.2]), MAX_TORQUE[ARM]);
        let (cy, hh) = span(shoulder_y - ball_r_sh - GAP, elbow_y + r_elbow + GAP);
        let mut ua_parts = vec![
            p([0.12, hh, 0.14], [0.0; 3], 30.0, "arm"),
            // Slim shoulder armor: a cap above the ball (clear of the chest top) and a plate outside it.
            p([0.17, 0.06, 0.24], [0.0, shoulder_y + ball_r_sh + GAP + 0.06 - cy, 0.0], 8.0, "v_pauldron"),
            p([0.05, 0.20, 0.22], [side * (ball_r_sh + GAP + 0.05), shoulder_y - cy, 0.0], 6.0, "v_pauldron"),
        ];
        if spec.load == Load::Heavy {
            ua_parts.push(p([0.08, 0.22, 0.24], [side * (ball_r_sh + GAP + 0.10 + 0.08 + GAP), shoulder_y - 0.08 - cy, 0.0], 100.0, "armor"));
        }
        let (ua, _) = add_body(w, links, format!("{n}_upper_arm"), at(side * shoulder_x, cy, 0.0), &ua_parts, true);
        let jy = revolute(w, g2, ua, Vector::Y, z, Vector::new(0.0, shoulder_y - cy, 0.0), Some([-0.8, 0.8]), MAX_TORQUE[ARM]);
        let (cy2, hh2) = span(elbow_y - r_elbow - GAP, elbow_y - 0.70);
        let mut fa_parts = vec![
            p([0.14, hh2, 0.16], [0.0; 3], 40.0, "v_forearm"),
            p([r_elbow, 0.13, r_elbow], [0.0, elbow_y - cy2, 0.0], 5.0, "hinge_x"),
        ];
        match spec.load {
            Load::Light => {}
            Load::Medium => fa_parts.push(p([0.09, 0.32, 0.11], [side * (0.14 * 1.2 + GAP + 0.09), -0.04, 0.08], 80.0, "weapon")),
            Load::Heavy => fa_parts.push(p([0.12, 0.40, 0.15], [side * (0.14 * 1.2 + GAP + 0.12), -0.08, 0.10], 120.0, "weapon")),
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
            // Roll wheel: thin disc right behind the pelvis; pitch wheel: edge-on disc behind that.
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
    Robot { spec, trunk: pelvis, legs: [left, right], arms, wheels, links, total_mass, waist: Some((wj, upper)), motor_scale }
}
