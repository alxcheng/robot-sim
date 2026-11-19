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
        }
    }
}

pub const RUN_BOUNDS: [(f32, f32, f32); 18] = [
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
    (-0.2, 0.8, 0.05),  // toe_push
    (0.0, 2.0, 0.1),    // arm_k
    (0.0, 2.0, 0.1),    // wheel_k
    (0.3, 1.4, 0.05),   // couple
];

impl RunGait {
    pub fn to_vec(&self) -> [f32; 18] {
        [self.t_st, self.l0, self.l_comp, self.l_push, self.push_start, self.l_swing, self.k_r, self.k_rl, self.width,
         self.lean, self.lean_v, self.kp, self.kd, self.leg_k, self.toe_push, self.arm_k, self.wheel_k, self.couple]
    }
    pub fn from_vec(v: &[f32; 18]) -> Self {
        let c = |i: usize| v[i].clamp(RUN_BOUNDS[i].0, RUN_BOUNDS[i].1);
        RunGait {
            t_st: c(0), l0: c(1), l_comp: c(2), l_push: c(3), push_start: c(4), l_swing: c(5), k_r: c(6), k_rl: c(7),
            width: c(8), lean: c(9), lean_v: c(10), kp: c(11), kd: c(12), leg_k: c(13), toe_push: c(14), arm_k: c(15),
            wheel_k: c(16), couple: c(17),
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
                } else if c[o] && self.t > 0.15 {
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
        let phi_f = ((v_f * t_st * 0.5 + g.k_r * (v_f - cmd.speed)) / len0).clamp(-0.7, 0.7).asin();
        let phi_l_base = ((v_l * t_st * 0.5 + g.k_rl * v_l) / len0).clamp(-0.5, 0.5).asin();

        let mut targets: Vec<(MultibodyJointHandle, usize, f32, &JointGains)> = Vec::new();
        let foot_parent = |leg: &Leg| leg.meta.map(|m| m.1).unwrap_or(leg.bodies[B_SHIN]);

        let (stance_leg, free_mode): (Option<usize>, [u8; 2]) = match self.phase {
            // free_mode: 0 = stance, 1 = swing toward landing (retracted), 2 = reach for landing, 3 = trailing retract
            Phase::Stance(s) => {
                let frac = self.t / t_st;
                let mut m = [0u8; 2];
                m[1 - s] = if frac < 0.55 { 1 } else { 2 };
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
            let j = leg.joints;
            let mode = if stance_leg == Some(i) { 0 } else { free_mode[i].max(1) };
            if mode == 0 {
                let frac = self.t / t_st;
                let pushing = frac >= g.push_start;
                let len = if pushing { g.l_push } else { g.l_comp } * lmax;
                let (qk, qm, _) = leg_ik(&spec, len, g.couple);
                let foot_yaw = yaw_of(w.bodies[leg.bodies[B_FOOT]].rotation());
                targets.push((j[HIP_YAW], HIP_YAW, wrap(foot_yaw - cmd.heading).clamp(-0.35, 0.35), &gains));
                targets.push((j[HIP_ROLL], HIP_ROLL, th_r, &stiff));
                targets.push((j[HIP_PITCH], HIP_PITCH, th_p - lean_des, &stiff));
                targets.push((j[KNEE], KNEE, qk, &spring));
                if let Some((jm, _)) = leg.meta {
                    targets.push((jm, META, qm, &spring));
                }
                targets.push((j[ANKLE_PITCH], ANKLE_PITCH, -fp_p - if pushing { g.toe_push } else { 0.0 }, &gains));
                targets.push((j[ANKLE_ROLL], ANKLE_ROLL, -fp_r, &gains));
            } else {
                let (len_frac, pitch) = match mode {
                    1 => (g.l_swing, -phi_f),
                    2 => (g.l0, -phi_f),
                    _ => (g.l_swing, th_p), // trailing: keep angle, retract
                };
                let (qk, qm, p0) = leg_ik(&spec, len_frac * lmax, g.couple);
                let roll = leg.side * g.width + phi_l_base;
                let hip_pitch = if mode == 3 { th_p - tr_pitch } else { pitch - p0 - tr_pitch };
                targets.push((j[HIP_YAW], HIP_YAW, heading_err.clamp(-0.35, 0.35), &gains));
                targets.push((j[HIP_ROLL], HIP_ROLL, (roll - tr_roll).clamp(-0.6, 0.6), &gains));
                targets.push((j[HIP_PITCH], HIP_PITCH, hip_pitch, &gains));
                targets.push((j[KNEE], KNEE, qk, &gains));
                if let Some((jm, _)) = leg.meta {
                    targets.push((jm, META, qm, &gains));
                }
                targets.push((j[ANKLE_PITCH], ANKLE_PITCH, -fp_p, &gains));
                targets.push((j[ANKLE_ROLL], ANKLE_ROLL, -fp_r, &gains));
            }
        }

        // Arms counter-swing against the same-side leg.
        let mut arm_targets = Vec::new();
        if let Some(arms) = r.arms {
            for (i, (ja, _)) in arms.iter().enumerate() {
                let (th_p, _) = ang(r.legs[i].bodies[B_THIGH]);
                arm_targets.push((*ja, (-g.arm_k * (th_p - tr_pitch)).clamp(-1.0, 1.0)));
            }
        }
        for (jh, k, t, gg) in targets {
            set_motor(w, jh, k, t, gg);
        }
        let arm_g = JointGains { kp: 1.0, kd: 0.08 };
        for (ja, t) in arm_targets {
            set_motor(w, ja, ARM, t, &arm_g);
        }

        // Trunk stabilising torque wanted (heading frame): used by reaction wheels (internal) and assist (external).
        let om = inv_h * w.bodies[r.trunk].angvel();
        let tau_pd = Vector::new(-(90_000.0 * (tr_pitch - lean_des) + 20_000.0 * om.x), 0.0, -(90_000.0 * tr_roll + 20_000.0 * om.z));
        let tau_world = Rotation::from_rotation_y(yaw) * tau_pd;
        for wh in &r.wheels {
            let axis_w = trunk_rot * wh.axis;
            let rel = (w.bodies[wh.body].angvel() - w.bodies[r.trunk].angvel()).dot(axis_w);
            // Motor torque on the wheel reacts on the trunk with the opposite sign.
            let tau_m = -g.wheel_k * tau_world.dot(axis_w);
            let factor = 2_000.0;
            let mut target = rel + tau_m / factor;
            // Saturation + slow desaturation so the wheel can't spin up forever.
            target -= 0.02 * rel;
            target = target.clamp(-250.0, 250.0);
            set_velocity_motor(w, wh.joint, target, factor, 25_000.0);
        }
        let trunk = &mut w.bodies[r.trunk];
        trunk.reset_torques(true);
        if self.assist > 0.0 {
            trunk.add_torque(tau_world * self.assist, true);
        }

        let st_foot_v = match stance_leg {
            Some(s) => w.bodies[r.legs[s].bodies[B_FOOT]].linvel(),
            None => Vector::ZERO,
        };
        let up = trunk_rot * Vector::Y;
        Telemetry {
            com: com.to_array(),
            com_vel: com_vel.to_array(),
            fwd_speed: v_f,
            trunk_tilt: up.y.clamp(-1.0, 1.0).acos(),
            yaw,
            stance_foot_slip: (st_foot_v.x * st_foot_v.x + st_foot_v.z * st_foot_v.z).sqrt(),
            feet_in_contact: c[0] as u8 + c[1] as u8,
        }
    }
}
