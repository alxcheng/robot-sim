//! Biped robot model: a heavy trunk (torso + pelvis reactor + head + arms) and two 6-joint legs,
//! built as one Rapier multibody. Sized to the concept-art frame: ~3.8 m tall, ~3.6 t.
//!
//! Axes: +Y up, +Z forward, +X left (right-handed). Every leg joint is a revolute joint so each
//! motor angle has a single, unambiguous meaning:
//!   hip yaw (Y) -> hip roll (Z) -> hip pitch (X) -> knee (X) -> ankle pitch (X) -> ankle roll (Z)

use rapier3d::prelude::*;
use serde::Serialize;

pub const HIP_YAW: usize = 0;
pub const HIP_ROLL: usize = 1;
pub const HIP_PITCH: usize = 2;
pub const KNEE: usize = 3;
pub const ANKLE_PITCH: usize = 4;
pub const ANKLE_ROLL: usize = 5;

// Body slots within a leg (child body of the joint with the same index).
pub const B_YAW_GIMBAL: usize = 0;
pub const B_ROLL_GIMBAL: usize = 1;
pub const B_THIGH: usize = 2;
pub const B_SHIN: usize = 3;
pub const B_ANKLE_GIMBAL: usize = 4;
pub const B_FOOT: usize = 5;

pub const HIP_Y: f32 = 2.0;
pub const HIP_X: f32 = 0.38;
pub const THIGH_LEN: f32 = 0.9;
pub const SHIN_LEN: f32 = 0.88;
pub const ANKLE_Y: f32 = HIP_Y - THIGH_LEN - SHIN_LEN; // 0.22
pub const FOOT_HALF: [f32; 3] = [0.24, 0.08, 0.42];
pub const FOOT_FWD: f32 = 0.10;
pub const TRUNK_Y: f32 = 2.75;

/// Robot colliders touch the world only (no self-collision), per spec §4.6.
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
    pub joints: [MultibodyJointHandle; 6],
    pub bodies: [RigidBodyHandle; 6],
    pub foot_collider: ColliderHandle,
}

pub struct Robot {
    pub trunk: RigidBodyHandle,
    pub legs: [Leg; 2],
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
    let body = w.insert_body(
        RigidBodyBuilder::dynamic()
            .translation(pos)
            .angular_damping(0.05)
            .can_sleep(false),
    );
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
    w.insert_multibody_joint(parent, child, j)
        .expect("leg chain must stay a tree")
}

/// Base position-motor stiffness (N·m/rad) per joint, sized for a ~3.6 t robot.
pub const BASE_KP: [f32; 6] = [40_000.0, 150_000.0, 150_000.0, 180_000.0, 120_000.0, 90_000.0];

/// Torque caps (N·m) per joint, indexed like the joint constants.
pub const MAX_TORQUE: [f32; 6] = [20_000.0, 60_000.0, 80_000.0, 80_000.0, 40_000.0, 30_000.0];

pub fn build(w: &mut PhysicsWorld, origin: Vector) -> Robot {
    let mut links = Vec::new();
    let o = origin;

    // Trunk: pelvis reactor carries most of the mass so the centre of mass stays low (spec §5.5).
    let trunk_parts = [
        Part { half: [0.60, 0.62, 0.40], offset: [0.0, 0.38, 0.0], mass: 700.0, tag: "torso" },
        Part { half: [0.46, 0.28, 0.40], offset: [0.0, -0.48, 0.0], mass: 1000.0, tag: "reactor" },
        Part { half: [0.18, 0.16, 0.20], offset: [0.0, 1.18, 0.02], mass: 60.0, tag: "head" },
        Part { half: [0.15, 0.58, 0.18], offset: [0.82, 0.12, 0.0], mass: 150.0, tag: "arm" },
        Part { half: [0.15, 0.58, 0.18], offset: [-0.82, 0.12, 0.0], mass: 150.0, tag: "arm" },
    ];
    let (trunk, _) = add_body(w, &mut links, "trunk".into(), o + Vector::new(0.0, TRUNK_Y, 0.0), &trunk_parts, true);

    let mk_leg = |w: &mut PhysicsWorld, links: &mut Vec<LinkInfo>, side: f32| -> Leg {
        let n = if side > 0.0 { "L" } else { "R" };
        let hip = o + Vector::new(side * HIP_X, HIP_Y, 0.0);
        let gimbal = |tag| [Part { half: [0.10, 0.10, 0.10], offset: [0.0; 3], mass: 40.0, tag }];
        let (yg, _) = add_body(w, links, format!("{n}_hip_yaw"), hip, &gimbal("gimbal"), false);
        let (rg, _) = add_body(w, links, format!("{n}_hip_roll"), hip, &gimbal("gimbal"), false);
        let thigh_c = hip - Vector::new(0.0, THIGH_LEN * 0.5, 0.0);
        let (th, _) = add_body(
            w, links, format!("{n}_thigh"), thigh_c,
            &[Part { half: [0.17, THIGH_LEN * 0.5, 0.20], offset: [0.0; 3], mass: 300.0, tag: "thigh" }],
            true,
        );
        let shin_c = hip - Vector::new(0.0, THIGH_LEN + SHIN_LEN * 0.5, 0.0);
        let (sh, _) = add_body(
            w, links, format!("{n}_shin"), shin_c,
            &[Part { half: [0.15, SHIN_LEN * 0.5, 0.18], offset: [0.0; 3], mass: 220.0, tag: "shin" }],
            true,
        );
        let ankle = o + Vector::new(side * HIP_X, ANKLE_Y, 0.0);
        let (ag, _) = add_body(
            w, links, format!("{n}_ankle"), ankle,
            &[Part { half: [0.08, 0.08, 0.08], offset: [0.0; 3], mass: 30.0, tag: "gimbal" }],
            false,
        );
        let foot_c = o + Vector::new(side * HIP_X, FOOT_HALF[1], FOOT_FWD);
        let (ft, fc) = add_body(
            w, links, format!("{n}_foot"), foot_c,
            &[Part { half: FOOT_HALF, offset: [0.0; 3], mass: 120.0, tag: "foot" }],
            true,
        );

        let z = Vector::ZERO;
        let j0 = revolute(w, trunk, yg, Vector::Y, hip - (o + Vector::new(0.0, TRUNK_Y, 0.0)), z, Some([-0.5, 0.5]), MAX_TORQUE[0]);
        let j1 = revolute(w, yg, rg, Vector::Z, z, z, Some([-0.6, 0.6]), MAX_TORQUE[1]);
        let j2 = revolute(w, rg, th, Vector::X, z, Vector::new(0.0, THIGH_LEN * 0.5, 0.0), Some([-1.8, 1.0]), MAX_TORQUE[2]);
        let j3 = revolute(
            w, th, sh, Vector::X,
            Vector::new(0.0, -THIGH_LEN * 0.5, 0.0), Vector::new(0.0, SHIN_LEN * 0.5, 0.0),
            Some([-0.02, 2.3]), MAX_TORQUE[3],
        );
        let j4 = revolute(w, sh, ag, Vector::X, Vector::new(0.0, -SHIN_LEN * 0.5, 0.0), z, Some([-0.9, 0.9]), MAX_TORQUE[4]);
        let j5 = revolute(
            w, ag, ft, Vector::Z, z,
            Vector::new(0.0, ANKLE_Y - FOOT_HALF[1], -FOOT_FWD),
            Some([-0.5, 0.5]), MAX_TORQUE[5],
        );
        Leg { side, joints: [j0, j1, j2, j3, j4, j5], bodies: [yg, rg, th, sh, ag, ft], foot_collider: fc[0] }
    };
    let left = mk_leg(w, &mut links, 1.0);
    let right = mk_leg(w, &mut links, -1.0);

    let total_mass = links.iter().map(|l| w.bodies[l.body].mass()).sum();
    Robot { trunk, legs: [left, right], links, total_mass }
}

/// Update one joint's position motor.
/// `g.kp` scales the joint's base stiffness; `g.kd` is damping as a fraction of stiffness (s).
pub fn set_motor(w: &mut PhysicsWorld, j: MultibodyJointHandle, k: usize, target: f32, g: &JointGains) {
    let kp = BASE_KP[k] * g.kp;
    let kd = kp * g.kd;
    let max_torque = MAX_TORQUE[k];
    if let Some((mb, id)) = w.multibody_joints.get_mut(j) {
        if let Some(link) = mb.link_mut(id) {
            link.joint.data.set_motor_model(JointAxis::AngX, MotorModel::ForceBased);
            link.joint.data.set_motor(JointAxis::AngX, target, 0.0, kp, kd);
            link.joint.data.set_motor_max_force(JointAxis::AngX, max_torque);
        }
    }
}
