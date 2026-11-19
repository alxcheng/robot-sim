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
    /// Deer-like hind leg: short heavy thigh, long light shin and cannon bone (metatarsal),
    /// standing on a small hoof (unguligrade). Mass sits near the hip so the leg swings fast.
    Deer,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RobotSpec {
    pub legs: LegKind,
    pub foot_scale: f32,
    pub swing_arms: bool,
    pub reaction_wheels: bool,
    /// 1 = single shoulder pitch joint; 3 = shoulder roll + shoulder pitch + elbow.
    #[serde(default = "one")]
    pub arm_dof: u8,
    /// Body proportions.
    #[serde(default)]
    pub frame: Frame,
}

/// `Classic`: the original boxy core. `Mech`: broad shoulders with large shoulder armor, compact chest over a narrow
/// waist, long legs (~54% of height), long arms reaching mid-thigh, small head; reactor stays in the pelvis.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum Frame {
    #[default]
    Classic,
    Mech,
}

fn one() -> u8 {
    1
}

impl Default for RobotSpec {
    fn default() -> Self {
        RobotSpec { legs: LegKind::Biped, foot_scale: 1.0, swing_arms: false, reaction_wheels: false, arm_dof: 1, frame: Frame::Classic }
    }
}

impl RobotSpec {
    /// (hip half-spacing, hip height, trunk body height)
    pub fn hip(&self) -> (f32, f32, f32) {
        match self.frame {
            Frame::Classic => (HIP_X, HIP_Y, TRUNK_Y),
            Frame::Mech => (0.34, 2.05, 2.70),
        }
    }
    /// Segment lengths from hip to the foot gimbal (virtual leg), and gimbal height above the ground.
    pub fn segments(&self) -> ([f32; 3], f32) {
        if self.frame == Frame::Mech && self.legs == LegKind::Deer {
            return ([0.52, 0.74, 0.67], 0.12);
        }
        match self.legs {
            LegKind::Biped => ([0.9, 0.88, 0.0], 0.22),
            LegKind::Digitigrade => ([0.72, 0.72, 0.42], 0.14),
            LegKind::Deer => ([0.50, 0.72, 0.66], 0.12),
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

pub struct Arm {
    /// Shoulder pitch joint and the upper-arm body (the whole arm when arm_dof = 1).
    pub pitch: (MultibodyJointHandle, RigidBodyHandle),
    pub roll: Option<MultibodyJointHandle>,
    pub elbow: Option<(MultibodyJointHandle, RigidBodyHandle)>,
}

pub struct Robot {
    pub spec: RobotSpec,
    pub trunk: RigidBodyHandle,
    pub legs: [Leg; 2],
    pub arms: Option<[Arm; 2]>,
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
    let mech = spec.frame == Frame::Mech;
    let (hip_x, hip_y, trunk_y) = spec.hip();
    let trunk_pos = o + Vector::new(0.0, trunk_y, 0.0);

    // Trunk. Swinging arms and reaction wheels become their own bodies; reactor mass is reduced to keep the total.
    let wheel_mass = if spec.reaction_wheels { 2.0 * 100.0 } else { 0.0 };
    let mut trunk_parts = if mech {
        vec![
            Part { half: [0.30, 0.20, 0.32], offset: [0.0, -0.48, 0.0], mass: 900.0 - wheel_mass, tag: "reactor" },
            Part { half: [0.22, 0.14, 0.24], offset: [0.0, -0.28, 0.0], mass: 120.0, tag: "waist" },
            Part { half: [0.46, 0.34, 0.38], offset: [0.0, 0.18, 0.0], mass: 480.0, tag: "torso" },
            Part { half: [0.36, 0.20, 0.06], offset: [0.0, 0.26, 0.42], mass: 20.0, tag: "plate" },
            Part { half: [0.30, 0.08, 0.30], offset: [0.0, 0.56, 0.0], mass: 20.0, tag: "torso" },
            Part { half: [0.11, 0.12, 0.15], offset: [0.0, 0.74, 0.04], mass: 40.0, tag: "head" },
        ]
    } else {
        vec![
            Part { half: [0.60, 0.62, 0.40], offset: [0.0, 0.38, 0.0], mass: 700.0, tag: "torso" },
            Part { half: [0.46, 0.28, 0.40], offset: [0.0, -0.48, 0.0], mass: 1000.0 - wheel_mass, tag: "reactor" },
            Part { half: [0.18, 0.16, 0.20], offset: [0.0, 1.18, 0.02], mass: 60.0, tag: "head" },
        ]
    };
    if !spec.swing_arms {
        trunk_parts.push(Part { half: [0.15, 0.58, 0.18], offset: [0.82, 0.12, 0.0], mass: 150.0, tag: "arm" });
        trunk_parts.push(Part { half: [0.15, 0.58, 0.18], offset: [-0.82, 0.12, 0.0], mass: 150.0, tag: "arm" });
    }
    let (trunk, _) = add_body(w, &mut links, "trunk".into(), trunk_pos, &trunk_parts, true);

    let fs = spec.foot_scale;
    let mk_leg = |w: &mut PhysicsWorld, links: &mut Vec<LinkInfo>, side: f32| -> Leg {
        let n = if side > 0.0 { "L" } else { "R" };
        let x = side * hip_x;
        let hip = o + Vector::new(x, hip_y, 0.0);
        let z = Vector::ZERO;
        let gimbal = |m: f32| [Part { half: [0.10, 0.10, 0.10], offset: [0.0; 3], mass: m, tag: "gimbal" }];
        let (yg, _) = add_body(w, links, format!("{n}_hip_yaw"), hip, &gimbal(40.0), false);
        let (rg, _) = add_body(w, links, format!("{n}_hip_roll"), hip, &gimbal(40.0), false);
        let j0 = revolute(w, trunk, yg, Vector::Y, hip - trunk_pos, z, Some([-0.5, 0.5]), MAX_TORQUE[0]);
        let j1 = revolute(w, yg, rg, Vector::Z, z, z, Some([-0.6, 0.6]), MAX_TORQUE[1]);

        let ([l1, l2, l3], gimbal_y) = spec.segments();
        let (m_th, m_sh, m_mt) = match spec.legs {
            LegKind::Biped => (300.0, 220.0, 0.0),
            LegKind::Digitigrade => (280.0, 200.0, 110.0),
            LegKind::Deer => if mech { (300.0, 190.0, 100.0) } else { (340.0, 170.0, 80.0) },
        };
        // Mech frame: slim thigh, chunky armored shin and cannon bone.
        let (th_w, sh_w, mt_w) = if mech { ([0.15, 0.18], [0.21, 0.25], [0.18, 0.22]) } else { ([0.17, 0.20], [0.15, 0.18], [0.11, 0.12]) };
        let (th, _) = add_body(
            w, links, format!("{n}_thigh"), hip - Vector::new(0.0, l1 * 0.5, 0.0),
            &[Part { half: [th_w[0], l1 * 0.5, th_w[1]], offset: [0.0; 3], mass: m_th, tag: "thigh" }], true,
        );
        let j2 = revolute(w, rg, th, Vector::X, z, Vector::new(0.0, l1 * 0.5, 0.0), Some([-1.8, 1.0]), MAX_TORQUE[2]);
        let (sh, _) = add_body(
            w, links, format!("{n}_shin"), hip - Vector::new(0.0, l1 + l2 * 0.5, 0.0),
            &[Part { half: [sh_w[0], l2 * 0.5, sh_w[1]], offset: [0.0; 3], mass: m_sh, tag: "shin" }], true,
        );
        let j3 = revolute(
            w, th, sh, Vector::X, Vector::new(0.0, -l1 * 0.5, 0.0), Vector::new(0.0, l2 * 0.5, 0.0),
            Some([-0.02, 2.4]), MAX_TORQUE[3],
        );
        // Optional metatarsal segment (bends the opposite way to the knee).
        let (above_gimbal, above_half, meta) = if l3 > 0.0 {
            let (mt, _) = add_body(
                w, links, format!("{n}_meta"), hip - Vector::new(0.0, l1 + l2 + l3 * 0.5, 0.0),
                &[Part { half: match (spec.legs, mech) { (LegKind::Deer, true) => [mt_w[0], l3 * 0.5, mt_w[1]], (LegKind::Deer, false) => [0.11, l3 * 0.5, 0.12], _ => [0.12, l3 * 0.5, 0.14] }, offset: [0.0; 3], mass: m_mt, tag: "meta" }], true,
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
            LegKind::Deer => if mech { ([0.23 * fs, 0.08, 0.36 * fs], 0.10 * fs, 70.0) } else { ([0.17 * fs, 0.06, 0.24 * fs], 0.08 * fs, 50.0) },
        };
        let tag = if spec.legs == LegKind::Deer { "hoof" } else { "foot" };
        let foot_c = o + Vector::new(x, fh[1], fwd);
        let (ft, fc) = add_body(w, links, format!("{n}_foot"), foot_c, &[Part { half: fh, offset: [0.0; 3], mass: fm, tag }], true);
        let j5 = revolute(w, ag, ft, Vector::Z, z, gpos - foot_c, Some([-0.5, 0.5]), MAX_TORQUE[5]);
        Leg { side, joints: [j0, j1, j2, j3, j4, j5], bodies: [yg, rg, th, sh, ag, ft], meta, foot_collider: fc[0] }
    };
    let left = mk_leg(w, &mut links, 1.0);
    let right = mk_leg(w, &mut links, -1.0);

    let arms = spec.swing_arms.then(|| {
        let mk_arm = |w: &mut PhysicsWorld, links: &mut Vec<LinkInfo>, side: f32| -> Arm {
            let n = if side > 0.0 { "L" } else { "R" };
            let shoulder_local = if mech { Vector::new(side * 0.66, 0.40, 0.0) } else { Vector::new(side * 0.82, 0.62, 0.0) };
            let shoulder = trunk_pos + shoulder_local;
            if spec.arm_dof < 3 {
                let (a, _) = add_body(w, links, format!("{n}_arm"), trunk_pos + Vector::new(side * 0.82, 0.12, 0.0),
                    &[Part { half: [0.15, 0.58, 0.18], offset: [0.0; 3], mass: 150.0, tag: "arm" }], true);
                let j = revolute(w, trunk, a, Vector::X, shoulder_local, Vector::new(0.0, 0.5, 0.0), Some([-1.2, 1.2]), MAX_TORQUE[ARM]);
                return Arm { pitch: (j, a), roll: None, elbow: None };
            }
            // Shoulder roll (Z) -> shoulder pitch (X) -> elbow (X).
            let (g, _) = add_body(w, links, format!("{n}_shoulder"), shoulder,
                &[Part { half: [0.08, 0.08, 0.08], offset: [0.0; 3], mass: 15.0, tag: "gimbal" }], false);
            let jr = revolute(w, trunk, g, Vector::Z, shoulder_local, Vector::ZERO, Some([-0.6, 0.6]), MAX_TORQUE[ARM]);
            // (upper-arm half length, forearm half length)
            let (hu, hf) = if mech { (0.31, 0.31) } else { (0.30, 0.28) };
            let upper_parts: Vec<Part> = if mech {
                vec![
                    Part { half: [0.13, hu, 0.15], offset: [0.0; 3], mass: 45.0, tag: "arm" },
                    // Large, light shoulder armor (hollow plate).
                    Part { half: [0.30, 0.28, 0.33], offset: [side * 0.08, hu + 0.02, 0.0], mass: 25.0, tag: "pauldron" },
                ]
            } else {
                vec![Part { half: [0.14, hu, 0.16], offset: [0.0; 3], mass: 75.0, tag: "arm" }]
            };
            let (ua, _) = add_body(w, links, format!("{n}_upper_arm"), shoulder - Vector::new(0.0, hu, 0.0), &upper_parts, true);
            let jp = revolute(w, g, ua, Vector::X, Vector::ZERO, Vector::new(0.0, hu, 0.0), Some([-1.4, 1.2]), MAX_TORQUE[ARM]);
            let fore = if mech { Part { half: [0.19, hf, 0.21], offset: [0.0; 3], mass: 70.0, tag: "forearm" } } else { Part { half: [0.13, hf, 0.15], offset: [0.0; 3], mass: 60.0, tag: "forearm" } };
            let (fa, _) = add_body(w, links, format!("{n}_forearm"), shoulder - Vector::new(0.0, 2.0 * hu + hf, 0.0), &[fore], true);
            let je = revolute(w, ua, fa, Vector::X, Vector::new(0.0, -hu, 0.0), Vector::new(0.0, hf, 0.0), Some([-2.2, 0.05]), MAX_TORQUE[ARM]);
            Arm { pitch: (jp, ua), roll: Some(jr), elbow: Some((je, fa)) }
        };
        [mk_arm(w, &mut links, 1.0), mk_arm(w, &mut links, -1.0)]
    });

    let mut wheels = Vec::new();
    if spec.reaction_wheels {
        let wheel_set = if mech {
            [(Vector::X, [0.06, 0.32, 0.32], [0.0, 0.22, -0.48], "pitch_wheel"), (Vector::Z, [0.32, 0.32, 0.06], [0.0, -0.22, -0.48], "roll_wheel")]
        } else {
            [(Vector::X, [0.06, 0.42, 0.42], [0.0, 0.55, -0.52], "pitch_wheel"), (Vector::Z, [0.42, 0.42, 0.06], [0.0, -0.05, -0.52], "roll_wheel")]
        };
        for (axis, half, off, name) in wheel_set {
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
            link.joint.data.set_motor_max_force(JointAxis::AngX, MAX_TORQUE[k] * torque_scale());
        }
    }
}

/// Experiment knob: WTORQUE scales every position-motor torque cap (stronger actuators).
pub fn torque_scale() -> f32 {
    static S: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("WTORQUE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0))
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
