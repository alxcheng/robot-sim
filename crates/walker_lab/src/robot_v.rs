//! Mech-V frame: slim yaw-only waist, V-shaped torso, ball shoulders and hips, legs that widen
//! toward large feet (legs ≈ 59% of standing height), fixed head. Deer-style leg chain
//! (hip yaw/roll/pitch, knee, metatarsal, foot pitch/roll). Light top, strong legs.
//!
//! Loads (swap tests): Light = bare frame; Medium = forearm weapons + back pack (+300 kg);
//! Heavy = bigger weapons + back unit + shoulder armor (+700 kg). All load mass sits on the upper body.

use crate::robot::*;
use rapier3d::prelude::*;

pub fn build_v(w: &mut PhysicsWorld, o: Vector, spec: RobotSpec) -> Robot {
    let mut links = Vec::new();
    let (hip_x, hip_y, pelvis_y) = spec.hip();
    let pelvis_pos = o + Vector::new(0.0, pelvis_y, 0.0);
    let z = Vector::ZERO;
    let p = |half: [f32; 3], offset: [f32; 3], mass: f32, tag: &'static str| Part { half, offset, mass, tag };

    // ---- pelvis (root: hips attach here; reactor lives here) ----
    let (pelvis, _) = add_body(w, &mut links, "pelvis".into(), pelvis_pos, &[
        p([0.30, 0.16, 0.26], [0.0, 0.0, 0.0], 450.0, "v_pelvis"),
        p([0.20, 0.08, 0.04], [0.0, 0.0, 0.25], 150.0, "reactor"),
    ], true);

    // ---- upper body on a yaw-only waist ----
    let upper_pos = o + Vector::new(0.0, 2.75, 0.0);
    let mut upper_parts = vec![
        p([0.15, 0.10, 0.17], [0.0, -0.32, 0.0], 40.0, "waist"),
        p([0.48, 0.32, 0.32], [0.0, 0.05, 0.0], 220.0, "v_chest"),
        p([0.30, 0.06, 0.24], [0.0, 0.42, -0.02], 20.0, "torso"),
        p([0.10, 0.11, 0.13], [0.0, 0.60, 0.05], 30.0, "head"),
    ];
    match spec.load {
        Load::Light => {}
        Load::Medium => upper_parts.push(p([0.30, 0.30, 0.16], [0.0, 0.05, -0.48], 140.0, "pack")),
        Load::Heavy => upper_parts.push(p([0.42, 0.36, 0.22], [0.0, 0.08, -0.54], 260.0, "pack")),
    }
    let (upper, _) = add_body(w, &mut links, "upper_body".into(), upper_pos, &upper_parts, true);
    let waist_y = 2.33;
    let wj = revolute(w, pelvis, upper, Vector::Y, Vector::new(0.0, waist_y - pelvis_y, 0.0), Vector::new(0.0, waist_y - 2.75, 0.0), Some([-0.8, 0.8]), MAX_TORQUE[WAIST]);

    // ---- legs ----
    let ([l1, l2, l3], gimbal_y) = spec.segments();
    let mk_leg = |w: &mut PhysicsWorld, links: &mut Vec<LinkInfo>, side: f32| -> Leg {
        let n = if side > 0.0 { "L" } else { "R" };
        let x = side * hip_x;
        let hip = o + Vector::new(x, hip_y, 0.0);
        let (yg, _) = add_body(w, links, format!("{n}_hip_ball"), hip, &[p([0.17, 0.17, 0.17], [0.0; 3], 40.0, "ball")], false);
        let (rg, _) = add_body(w, links, format!("{n}_hip_roll"), hip, &[p([0.06, 0.06, 0.06], [0.0; 3], 40.0, "gimbal")], false);
        let j0 = revolute(w, pelvis, yg, Vector::Y, hip - pelvis_pos, z, Some([-0.5, 0.5]), MAX_TORQUE[HIP_YAW]);
        let j1 = revolute(w, yg, rg, Vector::Z, z, z, Some([-0.6, 0.6]), MAX_TORQUE[HIP_ROLL]);
        let (th, _) = add_body(w, links, format!("{n}_thigh"), hip - Vector::new(0.0, l1 * 0.5, 0.0),
            &[p([0.14, l1 * 0.5, 0.17], [0.0; 3], 240.0, "v_thigh")], true);
        let j2 = revolute(w, rg, th, Vector::X, z, Vector::new(0.0, l1 * 0.5, 0.0), Some([-1.8, 1.0]), MAX_TORQUE[HIP_PITCH]);
        let (sh, _) = add_body(w, links, format!("{n}_shin"), hip - Vector::new(0.0, l1 + l2 * 0.5, 0.0),
            &[p([0.19, l2 * 0.5, 0.22], [0.0; 3], 200.0, "v_shin")], true);
        let j3 = revolute(w, th, sh, Vector::X, Vector::new(0.0, -l1 * 0.5, 0.0), Vector::new(0.0, l2 * 0.5, 0.0), Some([-0.02, 2.4]), MAX_TORQUE[KNEE]);
        let (mt, _) = add_body(w, links, format!("{n}_meta"), hip - Vector::new(0.0, l1 + l2 + l3 * 0.5, 0.0),
            &[p([0.18, l3 * 0.5, 0.21], [0.0; 3], 130.0, "v_meta")], true);
        let jm = revolute(w, sh, mt, Vector::X, Vector::new(0.0, -l2 * 0.5, 0.0), Vector::new(0.0, l3 * 0.5, 0.0), Some([-2.4, 0.3]), MAX_TORQUE[META]);
        let gpos = o + Vector::new(x, gimbal_y, 0.0);
        let (ag, _) = add_body(w, links, format!("{n}_ankle"), gpos, &[p([0.08, 0.08, 0.08], [0.0; 3], 30.0, "gimbal")], false);
        let j4 = revolute(w, mt, ag, Vector::X, Vector::new(0.0, -l3 * 0.5, 0.0), z, Some([-1.2, 1.2]), MAX_TORQUE[ANKLE_PITCH]);
        let fh = [0.26 * spec.foot_scale, 0.08, 0.42 * spec.foot_scale];
        let foot_c = o + Vector::new(x, fh[1], 0.12);
        let (ft, fc) = add_body(w, links, format!("{n}_foot"), foot_c, &[p(fh, [0.0; 3], 110.0, "foot")], true);
        let j5 = revolute(w, ag, ft, Vector::Z, z, gpos - foot_c, Some([-0.5, 0.5]), MAX_TORQUE[ANKLE_ROLL]);
        Leg { side, joints: [j0, j1, j2, j3, j4, j5], bodies: [yg, rg, th, sh, ag, ft], meta: Some((jm, mt)), foot_collider: fc[0] }
    };
    let left = mk_leg(w, &mut links, 1.0);
    let right = mk_leg(w, &mut links, -1.0);

    // ---- arms: ball shoulder (roll, pitch, twist) + elbow ----
    let mk_arm = |w: &mut PhysicsWorld, links: &mut Vec<LinkInfo>, side: f32| -> Arm {
        let n = if side > 0.0 { "L" } else { "R" };
        let sh_local = Vector::new(side * 0.58, 0.28, 0.0);
        let shoulder = upper_pos + sh_local;
        let (hu, hf) = (0.30, 0.30);
        let (g1, _) = add_body(w, links, format!("{n}_shoulder_ball"), shoulder, &[p([0.16, 0.16, 0.16], [0.0; 3], 15.0, "ball")], false);
        let jr = revolute(w, upper, g1, Vector::Z, sh_local, z, Some([-0.6, 1.0]), MAX_TORQUE[ARM]);
        let (g2, _) = add_body(w, links, format!("{n}_shoulder_pitch"), shoulder, &[p([0.05, 0.05, 0.05], [0.0; 3], 5.0, "gimbal")], false);
        let jp = revolute(w, g1, g2, Vector::X, z, z, Some([-1.4, 1.2]), MAX_TORQUE[ARM]);
        let mut ua_parts = vec![
            p([0.12, hu, 0.14], [0.0; 3], 35.0, "arm"),
            p([0.30, 0.22, 0.32], [side * 0.10, hu, 0.0], 15.0, "v_pauldron"),
        ];
        if spec.load == Load::Heavy {
            ua_parts.push(p([0.18, 0.24, 0.30], [side * 0.32, hu - 0.06, 0.0], 100.0, "armor"));
        }
        let (ua, _) = add_body(w, links, format!("{n}_upper_arm"), shoulder - Vector::new(0.0, hu, 0.0), &ua_parts, true);
        let jy = revolute(w, g2, ua, Vector::Y, z, Vector::new(0.0, hu, 0.0), Some([-0.8, 0.8]), MAX_TORQUE[ARM]);
        let mut fa_parts = vec![p([0.16, hf, 0.18], [0.0; 3], 45.0, "v_forearm")];
        match spec.load {
            Load::Light => {}
            Load::Medium => fa_parts.push(p([0.10, 0.34, 0.12], [side * 0.24, -0.05, 0.10], 80.0, "weapon")),
            Load::Heavy => fa_parts.push(p([0.13, 0.42, 0.16], [side * 0.27, -0.08, 0.12], 120.0, "weapon")),
        }
        let (fa, _) = add_body(w, links, format!("{n}_forearm"), shoulder - Vector::new(0.0, 2.0 * hu + hf, 0.0), &fa_parts, true);
        let je = revolute(w, ua, fa, Vector::X, Vector::new(0.0, -hu, 0.0), Vector::new(0.0, hf, 0.0), Some([-2.2, 0.05]), MAX_TORQUE[ARM]);
        Arm { pitch: (jp, ua), roll: Some(jr), yaw: Some(jy), elbow: Some((je, fa)) }
    };
    let arms = Some([mk_arm(w, &mut links, 1.0), mk_arm(w, &mut links, -1.0)]);

    // ---- reaction wheels behind the pelvis ----
    let mut wheels = Vec::new();
    if spec.reaction_wheels {
        for (axis, half, off, name) in [
            (Vector::X, [0.05, 0.26, 0.26], [0.0, 0.05, -0.36], "pitch_wheel"),
            (Vector::Z, [0.26, 0.26, 0.05], [0.0, 0.05, -0.50], "roll_wheel"),
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
    // Motors sized to this body weight (reference 3,560 kg), times the strength factor.
    let motor_scale = spec.strength * total_mass / 3560.0;
    Robot { spec, trunk: pelvis, legs: [left, right], arms, wheels, links, total_mass, waist: Some((wj, upper)), motor_scale }
}
