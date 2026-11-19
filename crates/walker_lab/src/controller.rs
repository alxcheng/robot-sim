//! SIMBICON-style balance controller (Yin, Loken & van de Panne 2007), adapted to implicit
//! position motors:
//! * a two-state gait (left stance / right stance) with per-state target poses;
//! * the swing hip target is set in the *heading frame* and corrected by the centre-of-mass
//!   offset `d` and velocity `v` from the stance ankle (the Raibert/SIMBICON foot-placement rule);
//! * the stance hip is servoed so that the *trunk* reaches its world-frame target;
//! * ankles hold the feet level; hip yaw steers toward the commanded heading.
//!
//! Optional "Normal law" assist (spec §8.4): an upright torque on the trunk, scaled by `assist`.

use crate::robot::*;
use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Gait {
    pub t_step: f32,
    pub sw_hip0: f32,
    pub cd: f32,
    pub cv: f32,
    pub kv: f32,
    pub cdl: f32,
    pub cvl: f32,
    pub width: f32,
    pub sw_knee1: f32,
    pub sw_knee2: f32,
    pub st_knee: f32,
    pub lean: f32,
    pub lean_v: f32,
    pub ca_d: f32,
    pub ca_v: f32,
    pub ca_walk: f32,
    pub touchdown: f32,
    pub kp: f32,
    pub kd: f32,
}

impl Default for Gait {
    fn default() -> Self {
        Gait {
            t_step: 0.42,
            sw_hip0: -0.35,
            cd: 0.6,
            cv: 0.25,
            kv: 0.10,
            cdl: 0.6,
            cvl: 0.25,
            width: 0.05,
            sw_knee1: 0.9,
            sw_knee2: 0.08,
            st_knee: 0.08,
            lean: 0.03,
            lean_v: 0.02,
            ca_d: 1.0,
            ca_v: 0.35,
            ca_walk: 0.0,
            touchdown: 0.7,
            kp: 1.0,
            kd: 0.12,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct Command {
    pub speed: f32,
    pub heading: f32,
}

pub struct Controller {
    pub gait: Gait,
    pub stance: usize, // index into robot.legs
    pub phase_t: f32,
    pub assist: f32,
    pub steps: u32,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Telemetry {
    pub com: [f32; 3],
    pub com_vel: [f32; 3],
    pub fwd_speed: f32,
    pub trunk_tilt: f32,
    pub yaw: f32,
    pub stance_foot_slip: f32,
    pub feet_in_contact: u8,
}

fn wrap(a: f32) -> f32 {
    let mut a = a;
    while a > std::f32::consts::PI {
        a -= 2.0 * std::f32::consts::PI;
    }
    while a < -std::f32::consts::PI {
        a += 2.0 * std::f32::consts::PI;
    }
    a
}

pub fn yaw_of(q: &Rotation) -> f32 {
    let f = *q * Vector::Z;
    f.x.atan2(f.z)
}

/// (pitch, roll) of a limb's -Y axis expressed in the heading frame.
fn limb_angles(q: &Rotation, inv_heading: &Rotation) -> (f32, f32) {
    let d = *inv_heading * (*q * -Vector::Y);
    ((-d.z).atan2(-d.y), d.x.atan2(-d.y))
}

/// (pitch, roll) of the trunk's +Y axis in the heading frame.
fn trunk_angles(q: &Rotation, inv_heading: &Rotation) -> (f32, f32) {
    let u = *inv_heading * (*q * Vector::Y);
    (u.z.atan2(u.y), (-u.x).atan2(u.y))
}

pub fn foot_in_contact(w: &PhysicsWorld, leg: &Leg) -> bool {
    w.narrow_phase
        .contact_pairs_with(leg.foot_collider)
        .any(|p| p.has_any_active_contact())
}

pub fn centre_of_mass(w: &PhysicsWorld, r: &Robot) -> (Vector, Vector) {
    let (mut m, mut p, mut v) = (0.0, Vector::ZERO, Vector::ZERO);
    for l in &r.links {
        let b = &w.bodies[l.body];
        m += b.mass();
        p += b.center_of_mass() * b.mass();
        v += b.linvel() * b.mass();
    }
    (p / m, v / m)
}

impl Controller {
    pub fn new(gait: Gait, assist: f32) -> Self {
        Controller { gait, stance: 0, phase_t: 0.0, assist, steps: 0 }
    }

    pub fn tick(&mut self, w: &mut PhysicsWorld, r: &Robot, cmd: Command, dt: f32) -> Telemetry {
        let g = self.gait;
        let trunk_rot = *w.bodies[r.trunk].rotation();
        let yaw = yaw_of(&trunk_rot);
        let inv_h = Rotation::from_rotation_y(-yaw);
        let (com, com_vel) = centre_of_mass(w, r);

        let st = self.stance;
        let sw = 1 - st;
        // State switch: swing foot touches down after half a step, or the step times out.
        self.phase_t += dt;
        if cmd.speed < 0.0 {
            // Stand mode: both legs act as stance legs.
            self.phase_t = 0.0;
        }
        let sw_contact = foot_in_contact(w, &r.legs[sw]);
        if cmd.speed >= 0.0 && (self.phase_t > g.touchdown * g.t_step && sw_contact) || self.phase_t > 1.6 * g.t_step {
            self.stance = sw;
            self.phase_t = 0.0;
            self.steps += 1;
        }
        let st = self.stance;
        let sw = 1 - st;
        let phase = (self.phase_t / g.t_step).min(1.0);

        let st_leg = &r.legs[st];
        let sw_leg = &r.legs[sw];
        let st_ankle = w.bodies[st_leg.bodies[B_ANKLE_GIMBAL]].translation();
        let d = inv_h * (com - st_ankle);
        let v = inv_h * com_vel;
        // In the heading frame: +z forward, +x left.
        let (d_f, d_l, v_f, v_l) = (d.z, d.x, v.z, v.x);

        // Ankle strategy: centre of mass relative to the support centre (stance foot, or both feet when standing).
        let foot_c = |l: &Leg| w.bodies[l.bodies[B_FOOT]].translation();
        let support = if cmd.speed < 0.0 { (foot_c(st_leg) + foot_c(sw_leg)) * 0.5 } else { foot_c(st_leg) };
        let ds = inv_h * (com - support);
        // Walking: only a light ankle term, so the stance foot doesn't fight forward progress.
        let ca = if cmd.speed < 0.0 { 1.0 } else { g.ca_walk };
        let ank_p = (ca * (g.ca_d * ds.z + g.ca_v * v_f)).clamp(-0.4, 0.4);
        let ank_r = -(ca * (g.ca_d * ds.x + g.ca_v * v_l)).clamp(-0.3, 0.3);
        let (tr_pitch, tr_roll) = trunk_angles(&trunk_rot, &inv_h);
        let ang = |b: RigidBodyHandle| limb_angles(w.bodies[b].rotation(), &inv_h);
        let lean_des = g.lean + g.lean_v * cmd.speed;

        let gains = JointGains { kp: g.kp, kd: g.kd };
        let stiff = JointGains { kp: g.kp * 1.6, kd: g.kd };

        // ---- swing leg ----
        let sw_pitch = g.sw_hip0 - g.cd * d_f - g.cv * v_f + g.kv * cmd.speed;
        let sw_roll = sw_leg.side * g.width + g.cdl * d_l + g.cvl * v_l;
        let heading_err = wrap(cmd.heading - yaw);
        let sw_targets = {
            let (sh_p, sh_r) = ang(sw_leg.bodies[B_SHIN]);
            let knee = if phase < 0.45 { g.sw_knee1 } else { g.sw_knee2 };
            [
                heading_err.clamp(-0.35, 0.35),
                (sw_roll - tr_roll).clamp(-0.6, 0.6),
                sw_pitch - tr_pitch,
                knee,
                -sh_p,
                -sh_r,
            ]
        };
        // ---- stance leg ----
        let st_targets = {
            let (th_p, th_r) = ang(st_leg.bodies[B_THIGH]);
            let (sh_p, sh_r) = ang(st_leg.bodies[B_SHIN]);
            let foot_yaw = yaw_of(w.bodies[st_leg.bodies[B_FOOT]].rotation());
            [
                wrap(foot_yaw - cmd.heading).clamp(-0.35, 0.35),
                th_r - 0.0,
                th_p - lean_des,
                g.st_knee,
                -sh_p + ank_p,
                -sh_r + ank_r,
            ]
        };

        let sw_targets = if cmd.speed < 0.0 {
            let (th_p, th_r) = ang(sw_leg.bodies[B_THIGH]);
            let (sh_p, sh_r) = ang(sw_leg.bodies[B_SHIN]);
            [0.0, th_r - 0.0, th_p - lean_des, g.st_knee, -sh_p + ank_p, -sh_r + ank_r]
        } else {
            sw_targets
        };
        let swj = sw_leg.joints;
        let stj = st_leg.joints;
        for k in 0..6 {
            let gs = if k == HIP_PITCH || k == HIP_ROLL { &stiff } else { &gains };
            set_motor(w, swj[k], k, sw_targets[k], &gains);
            set_motor(w, stj[k], k, st_targets[k], gs);
        }

        // ---- assist: upright torque on the trunk ("Normal law") ----
        let trunk = &mut w.bodies[r.trunk];
        trunk.reset_torques(true);
        if self.assist > 0.0 {
            let om = inv_h * trunk.angvel();
            let kp = 90_000.0 * self.assist;
            let kd = 20_000.0 * self.assist;
            let tau_h = Vector::new(-(kp * (tr_pitch - lean_des) + kd * om.x), 0.0, -(kp * tr_roll + kd * om.z));
            let tau_w = Rotation::from_rotation_y(yaw) * tau_h;
            trunk.add_torque(tau_w, true);
        }

        let st_foot_v = w.bodies[st_leg.bodies[B_FOOT]].linvel();
        let up = trunk_rot * Vector::Y;
        Telemetry {
            com: com.to_array(),
            com_vel: com_vel.to_array(),
            fwd_speed: v_f,
            trunk_tilt: up.y.clamp(-1.0, 1.0).acos(),
            yaw,
            stance_foot_slip: (st_foot_v.x * st_foot_v.x + st_foot_v.z * st_foot_v.z).sqrt(),
            feet_in_contact: foot_in_contact(w, &r.legs[0]) as u8 + foot_in_contact(w, &r.legs[1]) as u8,
        }
    }
}
