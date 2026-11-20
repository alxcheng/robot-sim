//! Thrust allocation for the 9-thruster set.
//!
//! Given a desired total force (from the player's up / left / right / forward / back input, or a
//! dash) and a desired torque about the centre of mass (attitude hold, so thrust doesn't tip the
//! robot over), find each thruster's force vector:
//!
//!   minimise |Σ fᵢ − F|² + w_τ·|Σ rᵢ × fᵢ − τ|² + λ·Σ|fᵢ|²
//!   subject to  fᵢ · nᵢ ≥ 0  (each nozzle swings within a 180° hemisphere around its nominal direction)
//!               |fᵢ| ≤ fmaxᵢ (sustained, or the higher dash limit)
//!
//! solved by projected gradient descent (9 thrusters × 3 = 27 variables; ~60 iterations per tick).

use crate::DT;
use crate::controller::*;
use crate::robot::*;
use rapier3d::prelude::*;

pub struct ThrustOut {
    /// World position and world force of each thruster.
    pub forces: Vec<(Vector, Vector)>,
}

/// Fastest descent the landing brake allows (m/s).
pub const SAFE_DESCENT: f32 = 3.0;

/// Desired force (N, world frame) for a heading-frame thrust command.
/// `input`: [left, up, forward] in −1..1; `dash` is the dash level 0..1 (it spools like the thrust);
/// `vy` is the centre of mass's vertical velocity.
/// Horizontal = 1 is 4 m/s² (dash 15). Up is a climb-rate command, 2.5 m/s at full input (dash 5),
/// limited to 1.3 g of lift (dash 2 g), so holding up rises steadily instead of rocketing.
pub fn desired_force(r: &Robot, yaw: f32, input: [f32; 3], dash: f32, vy: f32) -> Vector {
    let m = r.total_mass;
    let g = 9.81;
    let horiz = 4.0 + 11.0 * dash;
    let up_max = (1.3 + 0.7 * dash) * m * g;
    let h = Rotation::from_rotation_y(yaw) * Vector::new(input[0] * horiz * m, 0.0, input[2] * horiz * m);
    // Horizontal boosting also carries part of the weight (skimming), so the feet aren't dragged
    // sideways under full load; lift scales with how hard the horizontal input is.
    let hmag = (input[0] * input[0] + input[2] * input[2]).sqrt().min(1.0);
    let skim = 0.3f32;
    let mut v = skim * hmag * m * g;
    if input[1] > 0.0 {
        let climb = input[1] * (2.5 + 2.5 * dash);
        v = v.max((m * (g + 4.0 * (climb - vy))).clamp(0.0, up_max));
    }
    h + Vector::new(0.0, v, 0.0)
}

/// Landing brake: lift that caps the descent speed at SAFE_DESCENT while airborne.
pub fn brake_force(r: &Robot, vy: f32) -> f32 {
    let m = r.total_mass;
    (m * (9.81 + 4.0 * (-SAFE_DESCENT - vy))).clamp(0.0, 1.3 * m * 9.81)
}

pub fn allocate(w: &PhysicsWorld, r: &Robot, f_des: Vector, tau_des: Vector, dash: f32) -> ThrustOut {
    let (com, _) = centre_of_mass(w, r);
    let n = r.thrusters.len();
    let mut pos = Vec::with_capacity(n);
    let mut dir = Vec::with_capacity(n);
    let mut fmax = Vec::with_capacity(n);
    // Leg-mounted thrusters (thighs, knees) yank the legs out of their stride, so they cost more:
    // the torso set does the work and the leg set tops it up when more force is needed.
    // The thigh-skirt pair is fixed to the pelvis; the knee pair is held in reserve for dashes.
    let leg_cost: f32 = if dash > 0.05 { 0.3 } else { 1000.0 };
    let cost: Vec<f32> = r.thrusters.iter().map(|t| if t.name.contains("knee") { leg_cost } else { 1e-3 }).collect();
    for t in &r.thrusters {
        let b = &w.bodies[t.body];
        let rot = *b.rotation();
        pos.push(b.translation() + rot * t.local_pos);
        dir.push((rot * t.local_dir).normalize());
        fmax.push(t.max_force + (t.dash_force - t.max_force) * dash);
    }
    let arm: Vec<Vector> = pos.iter().map(|p| *p - com).collect();
    let w_tau = 1.0f32;
    let mut f = vec![Vector::ZERO; n];
    // Step size from the problem's curvature (n + w_τ·Σ|r|²).
    let curv = n as f32 + w_tau * arm.iter().map(|a| a.length_squared()).sum::<f32>();
    // Per-thruster step so a costly (reserve) thruster doesn't slow the others' convergence.
    let alpha: Vec<f32> = cost.iter().map(|c| 0.5 / (curv + 2.0 * c)).collect();
    for _ in 0..60 {
        let sum_f: Vector = f.iter().copied().sum();
        let sum_t: Vector = arm.iter().zip(&f).map(|(a, fi)| a.cross(*fi)).sum();
        let gf = 2.0 * (sum_f - f_des);
        let gt = 2.0 * w_tau * (sum_t - tau_des);
        for i in 0..n {
            let grad = gf + gt.cross(arm[i]) + 2.0 * cost[i] * f[i];
            let mut fi = f[i] - alpha[i] * grad;
            // Hemisphere around the nominal direction.
            let along = fi.dot(dir[i]);
            if along < 0.0 {
                fi -= along * dir[i];
            }
            let len = fi.length();
            if len > fmax[i] {
                fi *= fmax[i] / len;
            }
            f[i] = fi;
        }
    }
    ThrustOut { forces: pos.into_iter().zip(f).collect() }
}

/// Attitude-hold torque (world) that keeps the pelvis upright and on the commanded heading while
/// thrusting (a sideways push with a foot planted otherwise spins the robot about that foot).
pub fn attitude_torque(w: &PhysicsWorld, r: &Robot, strength: f32, heading: f32) -> Vector {
    let rot = *w.bodies[r.trunk].rotation();
    let yaw = yaw_of(&rot);
    let inv_h = Rotation::from_rotation_y(-yaw);
    let (p, rl) = trunk_angles(&rot, &inv_h);
    let om = inv_h * w.bodies[r.trunk].angvel();
    let s = strength * r.total_mass / 2087.0;
    let yaw_err = wrap(heading - yaw);
    let tau_y = 30_000.0 * s * yaw_err.clamp(-0.5, 0.5) - 8_000.0 * s * om.y;
    let tau_h = Vector::new(-(60_000.0 * s * p + 15_000.0 * s * om.x), tau_y, -(60_000.0 * s * rl + 15_000.0 * s * om.z));
    Rotation::from_rotation_y(yaw) * tau_h
}

/// Engine spool state: thrust ramps up in ~0.1 s and down in ~0.35 s instead of switching instantly
/// (an instant cut dropped the boost's lift in one tick and buckled the legs).
#[derive(Clone, Copy, Default)]
pub struct Spool {
    pub thr: [f32; 3],
    pub dash: f32,
}

impl Spool {
    /// `k` scales the dash level (thrust curriculum); the thrust input arrives already scaled.
    pub fn update(&mut self, cmd: &Command, k: f32) {
        let toward = |x: &mut f32, target: f32, up_s: f32, down_s: f32| {
            let rate = if target.abs() > x.abs() { DT / up_s } else { DT / down_s };
            *x += (target - *x).clamp(-rate, rate);
        };
        for i in 0..3 {
            toward(&mut self.thr[i], cmd.thrust[i], 0.1, 0.35);
        }
        toward(&mut self.dash, if cmd.dash { k } else { 0.0 }, 0.05, 0.25);
    }
    pub fn active(&self) -> bool {
        self.thr.iter().any(|v| v.abs() > 1e-3) || self.dash > 1e-3
    }
}

/// Apply thrusters for a command. Returns the applied forces for recording.
pub fn apply(w: &mut PhysicsWorld, r: &Robot, cmd: &Command, spool: &mut Spool, k: f32) -> Vec<(Vector, Vector)> {
    if r.thrusters.is_empty() {
        return Vec::new();
    }
    spool.update(cmd, k);
    let (_, com_v) = centre_of_mass(w, r);
    let airborne = !foot_in_contact(w, &r.legs[0]) && !foot_in_contact(w, &r.legs[1]);
    // Automatic soft landing: falling faster than SAFE_DESCENT with no foot down fires the thrusters.
    let braking = airborne && com_v.y < -SAFE_DESCENT + 0.5;
    let active = spool.active() || braking;
    // Clear last tick's thruster forces on their mount bodies.
    let mut bodies: Vec<RigidBodyHandle> = r.thrusters.iter().map(|t| t.body).collect();
    bodies.sort_by_key(|h| h.into_raw_parts());
    bodies.dedup();
    for b in &bodies {
        w.bodies[*b].reset_forces(true);
        // add_force_at_point also accumulates torque, which reset_forces does not clear.
        w.bodies[*b].reset_torques(true);
    }
    // Thrusters only fire on input or to brake a fall (an always-on attitude hold was tested and fought the gait's lean).
    if !active {
        return Vec::new();
    }
    let yaw = yaw_of(w.bodies[r.trunk].rotation());
    let mut f_des = desired_force(r, yaw, spool.thr, spool.dash, com_v.y);
    if braking {
        f_des.y = f_des.y.max(brake_force(r, com_v.y));
    }
    let tau = attitude_torque(w, r, 1.0, cmd.heading);
    let out = allocate(w, r, f_des, tau, spool.dash);
    for (t, (p, f)) in r.thrusters.iter().zip(&out.forces) {
        w.bodies[t.body].add_force_at_point(*f, *p, true);
    }
    out.forces
}
