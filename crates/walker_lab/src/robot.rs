//! Biped robot model built as one Rapier multibody, sized to the concept-art frame (~3.8 m, ~3.6 t).
//!
//! Axes: +Y up, +Z forward, +X left (right-handed). Every joint is a revolute joint, so each
//! motor angle has one meaning. Leg chains:
//!   Biped:        hip yaw (Y) → hip roll (Z) → hip pitch (X) → knee (X) → ankle pitch (X) → ankle roll (Z) → foot
//!   Digitigrade:  hip yaw → hip roll → hip pitch → knee → **metatarsal (X)** → toe pitch (X) → toe roll (Z) → foot pad
//!
//! Optional parts (RobotSpec): bigger feet, motor-driven swinging arms, a reaction-wheel back unit.

use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};

pub const HIP_YAW: usize = 0;
pub const HIP_ROLL: usize = 1;
pub const HIP_PITCH: usize = 2;
pub const KNEE: usize = 3;
pub const ANKLE_PITCH: usize = 4; // foot pitch joint (ankle on biped, toe on digitigrade)
pub const ANKLE_ROLL: usize = 5;
pub const META: usize = 6; // digitigrade metatarsal joint
pub const ARM: usize = 7;

// Body slots within a leg.
pub const B_YAW_GIMBAL: usize = 0;
pub const B_ROLL_GIMBAL: usize = 1;
pub const B_THIGH: usize = 2;
pub const B_SHIN: usize = 3;
pub const B_ANKLE_GIMBAL: usize = 4; // gimbal above the foot
pub const B_FOOT: usize = 5;

pub const HIP_Y: f32 = 2.0;
pub const HIP_X: f32 = 0.38;
pub const TRUNK_Y: f32 = 2.75;

/// Base position-motor stiffness (N·m/rad) and torque caps, indexed by the joint constants.
pub const BASE_KP: [f32; 8] = [40_000.0, 150_000.0, 150_000.0, 180_000.0, 120_000.0, 90_000.0, 180_000.0, 40_000.0];
pub const MAX_TORQUE: [f32; 8] = [20_000.0, 60_000.0, 80_000.0, 80_000.0, 40_000.0, 30_000.0, 80_000.0, 20_000.0];

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum LegKind {
    Biped,
    Digitigrade,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RobotSpec {
    pub legs: LegKind,
    pub foot_scale: f32,
    pub swing_arms: bool,
    pub reaction_wheels: bool,
}

impl Default for RobotSpec {
    fn default() -> Self {
        RobotSpec { legs: LegKind::Biped, foot_scale: 1.0, swing_arms: false, reaction_wheels: false }
    }
}

impl RobotSpec {
    /// Segment lengths from hip to the foot gimbal (virtual leg), and gimbal height above the ground.
    pub fn segments(&self) -> ([f32; 3], f32) {
        match self.legs {
            LegKind::Biped => ([0.9, 0.88, 0.0], 0.22),
            LegKind::Digitigrade => ([0.72, 0.72, 0.42], 0.14),
        }
    }
}

pub fn robot_groups() -> InteractionGroups {
    InteractionGroups::new(Group::GROUP_2, Group::GROUP_1, InteractionTestMode::And)
}
pub fn world_groups() -> InteractionGroups {
    InteractionGroups::new(Group::GROUP_1, Group::ALL, InteractionTestMode::And)
}
fn no_contact_groups() -> InteractionGroups {
    InteractionGroups::new(Group::NONE, Group::NONE, InteractionTestMode::And)
}

#[derive(Clone, Serialize)]
pub struct ShapeInfo {
    pub half: [f32; 3],
    pub offset: [f32; 3],
    pub tag: &'static str,
}

#[derive(Clone, Serialize)]
pub struct LinkInfo {
    pub name: String,
    #[serde(skip)]
    pub body: RigidBodyHandle,
    pub shapes: Vec<ShapeInfo>,
}

pub struct Leg {
    pub side: f32, // +1 left (+X), -1 right
    /// [hip yaw, hip roll, hip pitch, knee, foot pitch, foot roll]
    pub joints: [MultibodyJointHandle; 6],
    /// [yaw gimbal, roll gimbal, thigh, shin, foot gimbal, foot]
    pub bodies: [RigidBodyHandle; 6],
    /// Digitigrade metatarsal (joint, body).
    pub meta: Option<(MultibodyJointHandle, RigidBodyHandle)>,
    pub foot_collider: ColliderHandle,
}

pub struct Wheel {
    pub joint: MultibodyJointHandle,
    pub body: RigidBodyHandle,
    pub axis: Vector, // trunk-local spin axis
}

pub struct Robot {
    pub spec: RobotSpec,
    pub trunk: RigidBodyHandle,
    pub legs: [Leg; 2],
    pub arms: Option<[(MultibodyJointHandle, RigidBodyHandle); 2]>,
    pub wheels: Vec<Wheel>,
    pub links: Vec<LinkInfo>,
    pub total_mass: f32,
}

struct Part {
    half: [f32; 3],
    offset: [f32; 3],
    mass: f32,
    tag: &'static str,
}

fn add_body(
    w: &mut PhysicsWorld,
    links: &mut Vec<LinkInfo>,
    name: String,
    pos: Vector,
    parts: &[Part],
    collide: bool,
) -> (RigidBodyHandle, Vec<ColliderHandle>) {
    let body = w.insert_body(RigidBodyBuilder::dynamic().translation(pos).angular_damping(0.05).can_sleep(false));
    let mut cols = Vec::new();
    let mut shapes = Vec::new();
    for p in parts {
        let groups = if collide { robot_groups() } else { no_contact_groups() };
        let c = ColliderBuilder::cuboid(p.half[0], p.half[1], p.half[2])
            .translation(Vector::new(p.offset[0], p.offset[1], p.offset[2]))
            .mass(p.mass)
            .friction(1.0)
            .collision_groups(groups);
        cols.push(w.insert_collider(c, Some(body)));
        shapes.push(ShapeInfo { half: p.half, offset: p.offset, tag: p.tag });
    }
    links.push(LinkInfo { name, body, shapes });
    (body, cols)
}

pub struct JointGains {
    pub kp: f32,
    pub kd: f32,
}

#[allow(clippy::too_many_arguments)]
fn revolute(
    w: &mut PhysicsWorld,
    parent: RigidBodyHandle,
    child: RigidBodyHandle,
    axis: Vector,
    a1: Vector,
    a2: Vector,
    limits: Option<[f32; 2]>,
    max_torque: f32,
) -> MultibodyJointHandle {
    let mut j = RevoluteJointBuilder::new(axis)
        .local_anchor1(a1)
        .local_anchor2(a2)
        .contacts_enabled(false)
        .motor_model(MotorModel::ForceBased)
        .motor_position(0.0, 50_000.0, 6_000.0)
        .motor_max_force(max_torque);
    if let Some(l) = limits {
        j = j.limits(l);
    }
    w.insert_multibody_joint(parent, child, j).expect("robot must stay a kinematic tree")
}

pub fn build(w: &mut PhysicsWorld, origin: Vector, spec: RobotSpec) -> Robot {
    let mut links = Vec::new();
    let o = origin;
    let trunk_pos = o + Vector::new(0.0, TRUNK_Y, 0.0);

    // Trunk. Swinging arms and reaction wheels become their own bodies; reactor mass is reduced to keep the total.
    let wheel_mass = if spec.reaction_wheels { 2.0 * 100.0 } else { 0.0 };
    let mut trunk_parts = vec![
        Part { half: [0.60, 0.62, 0.40], offset: [0.0, 0.38, 0.0], mass: 700.0, tag: "torso" },
        Part { half: [0.46, 0.28, 0.40], offset: [0.0, -0.48, 0.0], mass: 1000.0 - wheel_mass, tag: "reactor" },
        Part { half: [0.18, 0.16, 0.20], offset: [0.0, 1.18, 0.02], mass: 60.0, tag: "head" },
    ];
    if !spec.swing_arms {
        trunk_parts.push(Part { half: [0.15, 0.58, 0.18], offset: [0.82, 0.12, 0.0], mass: 150.0, tag: "arm" });
        trunk_parts.push(Part { half: [0.15, 0.58, 0.18], offset: [-0.82, 0.12, 0.0], mass: 150.0, tag: "arm" });
    }
    let (trunk, _) = add_body(w, &mut links, "trunk".into(), trunk_pos, &trunk_parts, true);

    let fs = spec.foot_scale;
    let mk_leg = |w: &mut PhysicsWorld, links: &mut Vec<LinkInfo>, side: f32| -> Leg {
        let n = if side > 0.0 { "L" } else { "R" };
        let x = side * HIP_X;
        let hip = o + Vector::new(x, HIP_Y, 0.0);
        let z = Vector::ZERO;
        let gimbal = |m: f32| [Part { half: [0.10, 0.10, 0.10], offset: [0.0; 3], mass: m, tag: "gimbal" }];
        let (yg, _) = add_body(w, links, format!("{n}_hip_yaw"), hip, &gimbal(40.0), false);
        let (rg, _) = add_body(w, links, format!("{n}_hip_roll"), hip, &gimbal(40.0), false);
        let j0 = revolute(w, trunk, yg, Vector::Y, hip - trunk_pos, z, Some([-0.5, 0.5]), MAX_TORQUE[0]);
        let j1 = revolute(w, yg, rg, Vector::Z, z, z, Some([-0.6, 0.6]), MAX_TORQUE[1]);

        let ([l1, l2, l3], gimbal_y) = spec.segments();
        let (m_th, m_sh) = if spec.legs == LegKind::Biped { (300.0, 220.0) } else { (280.0, 200.0) };
        let (th, _) = add_body(
            w, links, format!("{n}_thigh"), hip - Vector::new(0.0, l1 * 0.5, 0.0),
            &[Part { half: [0.17, l1 * 0.5, 0.20], offset: [0.0; 3], mass: m_th, tag: "thigh" }], true,
        );
        let j2 = revolute(w, rg, th, Vector::X, z, Vector::new(0.0, l1 * 0.5, 0.0), Some([-1.8, 1.0]), MAX_TORQUE[2]);
        let (sh, _) = add_body(
            w, links, format!("{n}_shin"), hip - Vector::new(0.0, l1 + l2 * 0.5, 0.0),
            &[Part { half: [0.15, l2 * 0.5, 0.18], offset: [0.0; 3], mass: m_sh, tag: "shin" }], true,
        );
        let j3 = revolute(
            w, th, sh, Vector::X, Vector::new(0.0, -l1 * 0.5, 0.0), Vector::new(0.0, l2 * 0.5, 0.0),
            Some([-0.02, 2.4]), MAX_TORQUE[3],
        );
        // Optional metatarsal segment (bends the opposite way to the knee).
        let (above_gimbal, above_half, meta) = if l3 > 0.0 {
            let (mt, _) = add_body(
                w, links, format!("{n}_meta"), hip - Vector::new(0.0, l1 + l2 + l3 * 0.5, 0.0),
                &[Part { half: [0.12, l3 * 0.5, 0.14], offset: [0.0; 3], mass: 110.0, tag: "meta" }], true,
            );
            let jm = revolute(
                w, sh, mt, Vector::X, Vector::new(0.0, -l2 * 0.5, 0.0), Vector::new(0.0, l3 * 0.5, 0.0),
                Some([-2.4, 0.3]), MAX_TORQUE[META],
            );
            (mt, l3 * 0.5, Some((jm, mt)))
        } else {
            (sh, l2 * 0.5, None)
        };
        let gpos = o + Vector::new(x, gimbal_y, 0.0);
        let (ag, _) = add_body(
            w, links, format!("{n}_ankle"), gpos,
            &[Part { half: [0.08, 0.08, 0.08], offset: [0.0; 3], mass: 30.0, tag: "gimbal" }], false,
        );
        let j4 = revolute(w, above_gimbal, ag, Vector::X, Vector::new(0.0, -above_half, 0.0), z, Some([-1.2, 1.2]), MAX_TORQUE[4]);

        let (fh, fwd, fm) = match spec.legs {
            LegKind::Biped => ([0.24 * fs, 0.08, 0.42 * fs], 0.10 * fs, 120.0),
            LegKind::Digitigrade => ([0.22 * fs, 0.07, 0.32 * fs], 0.12 * fs, 90.0),
        };
        let foot_c = o + Vector::new(x, fh[1], fwd);
        let (ft, fc) = add_body(w, links, format!("{n}_foot"), foot_c, &[Part { half: fh, offset: [0.0; 3], mass: fm, tag: "foot" }], true);
        let j5 = revolute(w, ag, ft, Vector::Z, z, gpos - foot_c, Some([-0.5, 0.5]), MAX_TORQUE[5]);
        Leg { side, joints: [j0, j1, j2, j3, j4, j5], bodies: [yg, rg, th, sh, ag, ft], meta, foot_collider: fc[0] }
    };
    let left = mk_leg(w, &mut links, 1.0);
    let right = mk_leg(w, &mut links, -1.0);

    let arms = spec.swing_arms.then(|| {
        let mk_arm = |w: &mut PhysicsWorld, links: &mut Vec<LinkInfo>, side: f32| {
            let n = if side > 0.0 { "L" } else { "R" };
            let c = trunk_pos + Vector::new(side * 0.82, 0.12, 0.0);
            let (a, _) = add_body(w, links, format!("{n}_arm"), c,
                &[Part { half: [0.15, 0.58, 0.18], offset: [0.0; 3], mass: 150.0, tag: "arm" }], true);
            let j = revolute(w, trunk, a, Vector::X, Vector::new(side * 0.82, 0.62, 0.0), Vector::new(0.0, 0.5, 0.0),
                Some([-1.2, 1.2]), MAX_TORQUE[ARM]);
            (j, a)
        };
        [mk_arm(w, &mut links, 1.0), mk_arm(w, &mut links, -1.0)]
    });

    let mut wheels = Vec::new();
    if spec.reaction_wheels {
        for (axis, half, off, name) in [
            (Vector::X, [0.06, 0.42, 0.42], [0.0, 0.55, -0.52], "pitch_wheel"),
            (Vector::Z, [0.42, 0.42, 0.06], [0.0, -0.05, -0.52], "roll_wheel"),
        ] {
            let p = trunk_pos + Vector::new(off[0], off[1], off[2]);
            let (b, _) = add_body(w, &mut links, name.into(), p,
                &[Part { half, offset: [0.0; 3], mass: 100.0, tag: "wheel" }], false);
            let j = RevoluteJointBuilder::new(axis)
                .local_anchor1(Vector::new(off[0], off[1], off[2]))
                .local_anchor2(Vector::ZERO)
                .contacts_enabled(false)
                .motor_model(MotorModel::ForceBased)
                .motor_velocity(0.0, 0.0)
                .motor_max_force(25_000.0);
            let jh = w.insert_multibody_joint(trunk, b, j).expect("tree");
            wheels.push(Wheel { joint: jh, body: b, axis });
        }
    }

    let total_mass = links.iter().map(|l| w.bodies[l.body].mass()).sum();
    Robot { spec, trunk, legs: [left, right], arms, wheels, links, total_mass }
}

/// Update one joint's position motor. `g.kp` scales the joint's base stiffness; `g.kd` is damping
/// as a fraction of stiffness (s).
pub fn set_motor(w: &mut PhysicsWorld, j: MultibodyJointHandle, k: usize, target: f32, g: &JointGains) {
    let kp = BASE_KP[k] * g.kp;
    let kd = kp * g.kd;
    if let Some((mb, id)) = w.multibody_joints.get_mut(j) {
        if let Some(link) = mb.link_mut(id) {
            link.joint.data.set_motor_model(JointAxis::AngX, MotorModel::ForceBased);
            link.joint.data.set_motor(JointAxis::AngX, target, 0.0, kp, kd);
            link.joint.data.set_motor_max_force(JointAxis::AngX, MAX_TORQUE[k]);
        }
    }
}

/// Velocity motor (used by reaction wheels): torque = factor·(target − ω), capped.
pub fn set_velocity_motor(w: &mut PhysicsWorld, j: MultibodyJointHandle, target: f32, factor: f32, max_torque: f32) {
    if let Some((mb, id)) = w.multibody_joints.get_mut(j) {
        if let Some(link) = mb.link_mut(id) {
            link.joint.data.set_motor(JointAxis::AngX, 0.0, target, 0.0, factor);
            link.joint.data.set_motor_max_force(JointAxis::AngX, max_torque);
        }
    }
}
