//! Running controller ("virtual leg" + Raibert), usable on biped and digitigrade legs.
//!
//! Each leg is commanded as a virtual leg (hip → foot gimbal) with a *length* and a *pitch*.
//! Inverse kinematics maps that onto knee (+ metatarsal) and hip angles, so the same controller
//! drives both leg kinds.
//!
//! State machine:  Stance(s) --stance foot lifts--> Flight(s) --other foot lands--> Stance(o)
//!                 Stance(s) --other foot lands first (no flight)--> Stance(o)
//! * Stance: leg compresses (spring), then extends + toe push-off after `push_start` of the stance.
//!   The stance hip servos the trunk pitch/roll (as in SIMBICON).
//! * Swing/flight: the free leg retracts, then reaches for the Raibert landing angle
//!   φ = asin((v·T_st/2 + k_r·(v − v_cmd)) / ℓ0), which regulates speed.
//! * Arms counter-swing against the same-side leg. Reaction wheels (if fitted) supply internal
//!   trunk torque. `assist` is the external upright torque ("Normal law" helper).

use crate::controller::*;
use crate::robot::*;
use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RunGait {
    pub t_st: f32,
    pub l0: f32,
    pub l_comp: f32,
    pub l_push: f32,
    pub push_start: f32,
    pub l_swing: f32,
    pub k_r: f32,
    pub k_rl: f32,
    pub width: f32,
    pub lean: f32,
    pub lean_v: f32,
    pub kp: f32,
    pub kd: f32,
    pub leg_k: f32,
    pub toe_push: f32,
    pub arm_k: f32,
    pub wheel_k: f32,
    pub couple: f32,
    /// Fraction of the stance at which the swing leg starts reaching for the ground (later = longer swing).
    #[serde(default = "reach_default")]
    pub reach_at: f32,
    /// Extra forward landing angle (rad) on top of Raibert placement: longer strides.
    #[serde(default)]
    pub stride_k: f32,
    /// Minimum stance time before the other foot may take over (s).
    #[serde(default = "min_stance_default")]
    pub min_stance: f32,
}

fn reach_default() -> f32 {
    0.55
}
fn min_stance_default() -> f32 {
    0.15
}

impl Default for RunGait {
    fn default() -> Self {
        RunGait {
            t_st: 0.32,
            l0: 0.90,
            l_comp: 0.86,
            l_push: 0.98,
            push_start: 0.45,
            l_swing: 0.72,
            k_r: 0.10,
            k_rl: 0.10,
            width: 0.04,
            lean: 0.06,
            lean_v: 0.03,
            kp: 1.0,
            kd: 0.10,
            leg_k: 0.8,
            toe_push: 0.25,
            arm_k: 0.8,
            wheel_k: 0.6,
            couple: 1.0,
            reach_at: 0.55,
            stride_k: 0.0,
            min_stance: 0.15,
        }
    }
}

pub const RUN_BOUNDS: [(f32, f32, f32); 21] = [
    // (min, max, perturbation scale)
    (0.18, 0.6, 0.03),  // t_st
    (0.80, 0.99, 0.02), // l0
    (0.70, 0.97, 0.02), // l_comp
    (0.88, 1.0, 0.015), // l_push
    (0.2, 0.8, 0.05),   // push_start
    (0.5, 0.92, 0.03),  // l_swing
    (-0.1, 0.5, 0.03),  // k_r
    (-0.1, 0.5, 0.03),  // k_rl
    (-0.05, 0.15, 0.015), // width
    (-0.1, 0.35, 0.02), // lean
    (-0.05, 0.12, 0.01), // lean_v
    (0.3, 3.0, 0.1),    // kp
    (0.03, 0.4, 0.015), // kd
    (0.2, 2.5, 0.1),    // leg_k
    (0.0, 0.9, 0.05),   // toe_push (toe-down angle at push-off)
    (0.0, 2.0, 0.1),    // arm_k
    (0.0, 2.0, 0.1),    // wheel_k
    (0.3, 1.4, 0.05),   // couple
    (0.3, 1.6, 0.06),   // reach_at
    (0.0, 0.45, 0.03),  // stride_k
    (0.08, 0.45, 0.02), // min_stance
];

impl RunGait {
    pub fn to_vec(&self) -> [f32; 21] {
        [self.t_st, self.l0, self.l_comp, self.l_push, self.push_start, self.l_swing, self.k_r, self.k_rl, self.width,
         self.lean, self.lean_v, self.kp, self.kd, self.leg_k, self.toe_push, self.arm_k, self.wheel_k, self.couple, self.reach_at, self.stride_k, self.min_stance]
    }
    pub fn from_vec(v: &[f32; 21]) -> Self {
        let c = |i: usize| v[i].clamp(RUN_BOUNDS[i].0, RUN_BOUNDS[i].1);
        RunGait {
            t_st: c(0), l0: c(1), l_comp: c(2), l_push: c(3), push_start: c(4), l_swing: c(5), k_r: c(6), k_rl: c(7),
            width: c(8), lean: c(9), lean_v: c(10), kp: c(11), kd: c(12), leg_k: c(13), toe_push: c(14), arm_k: c(15),
            wheel_k: c(16), couple: c(17), reach_at: c(18), stride_k: c(19), min_stance: c(20),
        }
    }
}

/// Virtual-leg vector (forward, down) with thigh pitch 0, for knee angle `qk`.
fn chain(seg: [f32; 3], qk: f32, couple: f32) -> (f32, f32) {
    let qm = -couple * qk;
    let p = [0.0, qk, qk + qm];
    let mut f = 0.0;
    let mut d = 0.0;
    for i in 0..3 {
        f += -seg[i] * p[i].sin();
        d += seg[i] * p[i].cos();
    }
    (f, d)
}

/// IK for a virtual-leg length: returns (knee, metatarsal, pitch of the leg vector at thigh pitch 0).
pub fn leg_ik(spec: &RobotSpec, len: f32, couple: f32) -> (f32, f32, f32) {
    let (seg, _) = spec.segments();
    let couple = if spec.legs == LegKind::Biped { 0.0 } else { couple };
    let (mut lo, mut hi) = (0.0f32, 2.3f32);
    for _ in 0..30 {
        let mid = 0.5 * (lo + hi);
        let (f, d) = chain(seg, mid, couple);
        if (f * f + d * d).sqrt() > len { lo = mid } else { hi = mid }
    }
    let qk = 0.5 * (lo + hi);
    let (f, d) = chain(seg, qk, couple);
    (qk, -couple * qk, (-f).atan2(d))
}

pub fn max_leg_len(spec: &RobotSpec) -> f32 {
    let (s, _) = spec.segments();
    s[0] + s[1] + s[2]
}

#[derive(Clone, Copy, Debug)]
pub enum Phase {
    Stance(usize),
    Flight(usize),
}

pub struct RunController {
    pub g: RunGait,
    pub phase: Phase,
    pub t: f32,
    pub assist: f32,
    pub steps: u32,
    pub t_st_est: f32,
}

impl RunController {
    pub fn new(g: RunGait, assist: f32) -> Self {
        RunController { g, phase: Phase::Stance(0), t: 0.0, assist, steps: 0, t_st_est: g.t_st }
    }
}

impl Ctl for RunController {
    fn steps(&self) -> u32 {
        self.steps
    }

    fn tick(&mut self, w: &mut PhysicsWorld, r: &Robot, cmd: Command, dt: f32) -> Telemetry {
        let (act, stance_leg) = self.compute(w, r, cmd, dt);
        apply_action(w, r, &act, self.g.kd);
        let lean_des = self.g.lean + self.g.lean_v * cmd.speed;
        reflexes(w, r, self.g.wheel_k, self.assist, lean_des);
        telemetry(w, r, stance_leg)
    }
}

impl RunController {
    /// Compute the teacher's joint targets without touching the world (also used to label
    /// states visited by a student policy).
    pub fn compute(&mut self, w: &PhysicsWorld, r: &Robot, cmd: Command, dt: f32) -> (Action, Option<usize>) {
        let g = self.g;
        let spec = r.spec;
        let lmax = max_leg_len(&spec);
        let trunk_rot = *w.bodies[r.trunk].rotation();
        let yaw = yaw_of(&trunk_rot);
        let inv_h = Rotation::from_rotation_y(-yaw);
        let (com, com_vel) = centre_of_mass(w, r);
        let v = inv_h * com_vel;
        let (v_f, v_l) = (v.z, v.x);
        let c = [foot_in_contact(w, &r.legs[0]), foot_in_contact(w, &r.legs[1])];

        // ---- state machine ----
        self.t += dt;
        self.phase = match self.phase {
            Phase::Stance(s) => {
                let o = 1 - s;
                if !c[s] && !c[o] && self.t > 0.06 {
                    self.t_st_est = 0.8 * self.t_st_est + 0.2 * self.t.clamp(0.1, 0.8);
                    self.t = 0.0;
                    Phase::Flight(s)
                } else if c[o] && self.t > g.min_stance {
                    self.t_st_est = 0.8 * self.t_st_est + 0.2 * self.t.clamp(0.1, 0.8);
                    self.t = 0.0;
                    self.steps += 1;
                    Phase::Stance(o)
                } else if self.t > 2.5 * g.t_st {
                    self.t = 0.0;
                    self.steps += 1;
                    Phase::Stance(o)
                } else {
                    Phase::Stance(s)
                }
            }
            Phase::Flight(s) => {
                let o = 1 - s;
                if c[o] {
                    self.t = 0.0;
                    self.steps += 1;
                    Phase::Stance(o)
                } else if c[s] && self.t > 0.05 {
                    self.t = 0.0;
                    Phase::Stance(s)
                } else if self.t > 0.8 {
                    self.t = 0.0;
                    Phase::Stance(o)
                } else {
                    Phase::Flight(s)
                }
            }
        };

        let (tr_pitch, tr_roll) = trunk_angles(&trunk_rot, &inv_h);
        let ang = |b: RigidBodyHandle| limb_angles(w.bodies[b].rotation(), &inv_h);
        let lean_des = g.lean + g.lean_v * cmd.speed;
        let heading_err = wrap(cmd.heading - yaw);
        let gains = JointGains { kp: g.kp, kd: g.kd };
        let stiff = JointGains { kp: g.kp * 1.6, kd: g.kd };
        let spring = JointGains { kp: g.kp * g.leg_k, kd: g.kd };

        // Raibert landing angles.
        let len0 = g.l0 * lmax;
        let t_st = self.t_st_est;
        let phi_f = ((v_f * t_st * 0.5 + g.k_r * (v_f - cmd.speed)) / len0).clamp(-0.7, 0.7).asin() + g.stride_k * (cmd.speed / 3.0).clamp(0.0, 1.0);
        let phi_l_base = ((v_l * t_st * 0.5 + g.k_rl * v_l) / len0).clamp(-0.5, 0.5).asin();

        let mut act = Action::default();
        let foot_parent = |leg: &Leg| leg.meta.map(|m| m.1).unwrap_or(leg.bodies[B_SHIN]);

        let (stance_leg, free_mode): (Option<usize>, [u8; 2]) = match self.phase {
            // free_mode: 0 = stance, 1 = swing toward landing (retracted), 2 = reach for landing, 3 = trailing retract
            Phase::Stance(s) => {
                let frac = self.t / t_st;
                let mut m = [0u8; 2];
                m[1 - s] = if frac < g.reach_at { 1 } else { 2 };
                (Some(s), m)
            }
            Phase::Flight(s) => {
                let mut m = [0u8; 2];
                m[s] = 3;
                m[1 - s] = 2;
                (None, m)
            }
        };

        for (i, leg) in r.legs.iter().enumerate() {
            let (th_p, th_r) = ang(leg.bodies[B_THIGH]);
            let (fp_p, fp_r) = ang(foot_parent(leg));
            let mode = if stance_leg == Some(i) { 0 } else { free_mode[i].max(1) };
            if mode == 0 {
                let frac = self.t / t_st;
                let pushing = frac >= g.push_start;
                let len = if pushing { g.l_push } else { g.l_comp } * lmax;
                let (qk, qm, _) = leg_ik(&spec, len, g.couple);
                let foot_yaw = yaw_of(w.bodies[leg.bodies[B_FOOT]].rotation());
                // Toe push-off only once the centre of mass is ahead of the stance foot (late stance);
                // pushing earlier drives the body backwards.
                let com_ahead = (inv_h * (com - w.bodies[leg.bodies[B_FOOT]].translation())).z > 0.05;
                act.set(i, S_YAW, wrap(foot_yaw - cmd.heading).clamp(-0.35, 0.35), gains.kp);
                act.set(i, S_ROLL, th_r, stiff.kp);
                act.set(i, S_PITCH, th_p - lean_des, stiff.kp);
                act.set(i, S_KNEE, qk, spring.kp);
                act.set(i, S_META, qm, spring.kp);
                act.set(i, S_FPITCH, -fp_p + if pushing && com_ahead { g.toe_push } else { 0.0 }, gains.kp);
                act.set(i, S_FROLL, -fp_r, gains.kp);
            } else {
                let (len_frac, pitch) = match mode {
                    1 => (g.l_swing, -phi_f),
                    2 => (g.l0, -phi_f),
                    _ => (g.l_swing, th_p), // trailing: keep angle, retract
                };
                let (qk, qm, p0) = leg_ik(&spec, len_frac * lmax, g.couple);
                let roll = leg.side * g.width + phi_l_base;
                let hip_pitch = if mode == 3 { th_p - tr_pitch } else { pitch - p0 - tr_pitch };
                act.set(i, S_YAW, heading_err.clamp(-0.35, 0.35), gains.kp);
                act.set(i, S_ROLL, (roll - tr_roll).clamp(-0.6, 0.6), gains.kp);
                act.set(i, S_PITCH, hip_pitch, gains.kp);
                act.set(i, S_KNEE, qk, gains.kp);
                act.set(i, S_META, qm, gains.kp);
                act.set(i, S_FPITCH, -fp_p, gains.kp);
                act.set(i, S_FROLL, -fp_r, gains.kp);
            }
        }

        // Arms counter-swing against the same-side leg; with elbows, the forearm folds up on the
        // forward swing (like a runner), and the shoulders sit slightly out from the body.
        if let Some(arms) = &r.arms {
            for i in 0..2 {
                let (th_p, _) = ang(r.legs[i].bodies[B_THIGH]);
                let pitch = (-g.arm_k * (th_p - tr_pitch)).clamp(-1.0, 1.0);
                act.arms[i][0] = pitch;
                if arms[i].roll.is_some() {
                    act.arms[i][1] = r.legs[i].side * 0.10;
                }
                if arms[i].elbow.is_some() {
                    act.arms[i][2] = -(0.55 + 0.8 * (-pitch).max(0.0)).min(2.0);
                }
            }
        }
        (act, stance_leg)
    }
}

pub const S_YAW: usize = 0;
pub const S_ROLL: usize = 1;
pub const S_PITCH: usize = 2;
pub const S_KNEE: usize = 3;
pub const S_META: usize = 4;
pub const S_FPITCH: usize = 5;
pub const S_FROLL: usize = 6;
const SLOT_JOINT: [usize; 7] = [HIP_YAW, HIP_ROLL, HIP_PITCH, KNEE, META, ANKLE_PITCH, ANKLE_ROLL];

/// Everything a leg/arm controller decides each tick: per-joint position targets and stiffness
/// scales (×BASE_KP), plus arm targets. This is the interface a learned policy imitates.
#[derive(Clone, Copy, Default, Debug)]
pub struct Action {
    pub legs: [[f32; 7]; 2],
    pub kp: [[f32; 7]; 2],
    /// Per arm: [shoulder pitch, shoulder roll, elbow].
    pub arms: [[f32; 3]; 2],
}

pub const ACTION_DIM: usize = 34;

impl Action {
    pub fn set(&mut self, leg: usize, slot: usize, target: f32, kp: f32) {
        self.legs[leg][slot] = target;
        self.kp[leg][slot] = kp;
    }
    pub fn to_vec(&self) -> [f32; ACTION_DIM] {
        let mut v = [0.0; ACTION_DIM];
        for l in 0..2 {
            for k in 0..7 {
                v[l * 7 + k] = self.legs[l][k];
                v[14 + l * 7 + k] = self.kp[l][k];
            }
        }
        for i in 0..2 {
            for k in 0..3 {
                v[28 + i * 3 + k] = self.arms[i][k];
            }
        }
        v
    }
    pub fn from_slice(v: &[f32]) -> Self {
        let mut a = Action::default();
        for l in 0..2 {
            for k in 0..7 {
                a.legs[l][k] = v[l * 7 + k];
                a.kp[l][k] = v[14 + l * 7 + k].clamp(0.1, 4.0);
            }
        }
        for i in 0..2 {
            for k in 0..3 {
                a.arms[i][k] = v[28 + i * 3 + k];
            }
        }
        a
    }
}

pub fn apply_action(w: &mut PhysicsWorld, r: &Robot, a: &Action, kd: f32) {
    for (i, leg) in r.legs.iter().enumerate() {
        for k in 0..7 {
            let jh = if k == S_META {
                match leg.meta {
                    Some((jm, _)) => jm,
                    None => continue,
                }
            } else {
                leg.joints[SLOT_JOINT[k]]
            };
            set_motor(w, jh, SLOT_JOINT[k], a.legs[i][k], &JointGains { kp: a.kp[i][k], kd });
        }
    }
    if let Some(arms) = &r.arms {
        let g = JointGains { kp: 1.0, kd: 0.08 };
        for (i, arm) in arms.iter().enumerate() {
            set_motor(w, arm.pitch.0, ARM, a.arms[i][0], &g);
            if let Some(jr) = arm.roll {
                set_motor(w, jr, ARM, a.arms[i][1], &g);
            }
            if let Some((je, _)) = arm.elbow {
                set_motor(w, je, ARM, a.arms[i][2], &JointGains { kp: 0.6, kd: 0.08 });
            }
        }
    }
}

/// Low-level reflexes kept outside any learned policy: reaction wheels (internal torque) and the
/// optional external upright assist.
pub fn reflexes(w: &mut PhysicsWorld, r: &Robot, wheel_k: f32, assist: f32, lean_des: f32) {
    let trunk_rot = *w.bodies[r.trunk].rotation();
    let yaw = yaw_of(&trunk_rot);
    let inv_h = Rotation::from_rotation_y(-yaw);
    let (tr_pitch, tr_roll) = trunk_angles(&trunk_rot, &inv_h);
    let om = inv_h * w.bodies[r.trunk].angvel();
    let tau_pd = Vector::new(-(90_000.0 * (tr_pitch - lean_des) + 20_000.0 * om.x), 0.0, -(90_000.0 * tr_roll + 20_000.0 * om.z));
    let tau_world = Rotation::from_rotation_y(yaw) * tau_pd;
    for wh in &r.wheels {
        let axis_w = trunk_rot * wh.axis;
        let rel = (w.bodies[wh.body].angvel() - w.bodies[r.trunk].angvel()).dot(axis_w);
        // Motor torque on the wheel reacts on the trunk with the opposite sign.
        let tau_m = -wheel_k * tau_world.dot(axis_w);
        let factor = 2_000.0;
        let mut target = rel + tau_m / factor;
        target -= 0.02 * rel; // slow desaturation
        target = target.clamp(-250.0, 250.0);
        set_velocity_motor(w, wh.joint, target, factor, 25_000.0);
    }
    let trunk = &mut w.bodies[r.trunk];
    trunk.reset_torques(true);
    if assist > 0.0 {
        trunk.add_torque(tau_world * assist, true);
    }
}

pub fn telemetry(w: &PhysicsWorld, r: &Robot, stance_leg: Option<usize>) -> Telemetry {
    let trunk_rot = *w.bodies[r.trunk].rotation();
    let yaw = yaw_of(&trunk_rot);
    let inv_h = Rotation::from_rotation_y(-yaw);
    let (com, com_vel) = centre_of_mass(w, r);
    let c = [foot_in_contact(w, &r.legs[0]), foot_in_contact(w, &r.legs[1])];
    let stance_leg = stance_leg.or(if c[0] { Some(0) } else if c[1] { Some(1) } else { None });
        let st_foot_v = match stance_leg {
            Some(s) => w.bodies[r.legs[s].bodies[B_FOOT]].linvel(),
            None => Vector::ZERO,
        };
    let up = trunk_rot * Vector::Y;
    Telemetry {
            com: com.to_array(),
            com_vel: com_vel.to_array(),
            fwd_speed: (inv_h * com_vel).z,
            trunk_tilt: up.y.clamp(-1.0, 1.0).acos(),
            yaw,
            stance_foot_slip: (st_foot_v.x * st_foot_v.x + st_foot_v.z * st_foot_v.z).sqrt(),
            feet_in_contact: c[0] as u8 + c[1] as u8,
        }
}
