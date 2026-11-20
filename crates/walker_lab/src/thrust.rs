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

use crate::controller::*;
use crate::robot::*;
use rapier3d::prelude::*;

pub struct ThrustOut {
    /// World position and world force of each thruster.
    pub forces: Vec<(Vector, Vector)>,
}

/// Desired force (N, world frame) for a heading-frame thrust command and dash flag.
/// `input`: [left, up, forward] in −1..1. Up = 1 lifts with 1.3 g; horizontal = 1 is 4 m/s².
pub fn desired_force(r: &Robot, yaw: f32, input: [f32; 3], dash: bool) -> Vector {
    let m = r.total_mass;
    let g = 9.81;
    let horiz = if dash { 15.0 } else { 4.0 };
    let up = if dash { 2.0 } else { 1.3 };
    let h = Rotation::from_rotation_y(yaw) * Vector::new(input[0] * horiz * m, 0.0, input[2] * horiz * m);
    let v = if input[1] > 0.0 { input[1] * up * m * g } else { 0.0 };
    h + Vector::new(0.0, v, 0.0)
}

pub fn allocate(w: &PhysicsWorld, r: &Robot, f_des: Vector, tau_des: Vector, dash: bool) -> ThrustOut {
    let (com, _) = centre_of_mass(w, r);
    let n = r.thrusters.len();
    let mut pos = Vec::with_capacity(n);
    let mut dir = Vec::with_capacity(n);
    let mut fmax = Vec::with_capacity(n);
    for t in &r.thrusters {
        let b = &w.bodies[t.body];
        let rot = *b.rotation();
        pos.push(b.translation() + rot * t.local_pos);
        dir.push((rot * t.local_dir).normalize());
        fmax.push(if dash { t.dash_force } else { t.max_force });
    }
    let arm: Vec<Vector> = pos.iter().map(|p| *p - com).collect();
    let w_tau = 1.0f32;
    let lambda = 1e-3f32;
    let mut f = vec![Vector::ZERO; n];
    // Step size from the problem's curvature (n + w_τ·Σ|r|²).
    let curv = n as f32 + w_tau * arm.iter().map(|a| a.length_squared()).sum::<f32>() + lambda;
    let alpha = 0.5 / curv;
    for _ in 0..60 {
        let sum_f: Vector = f.iter().copied().sum();
        let sum_t: Vector = arm.iter().zip(&f).map(|(a, fi)| a.cross(*fi)).sum();
        let gf = 2.0 * (sum_f - f_des);
        let gt = 2.0 * w_tau * (sum_t - tau_des);
        for i in 0..n {
            let grad = gf + gt.cross(arm[i]) + 2.0 * lambda * f[i];
            let mut fi = f[i] - alpha * grad;
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

/// Attitude-hold torque (world) that keeps the pelvis upright while thrusting.
pub fn attitude_torque(w: &PhysicsWorld, r: &Robot, strength: f32) -> Vector {
    let rot = *w.bodies[r.trunk].rotation();
    let yaw = yaw_of(&rot);
    let inv_h = Rotation::from_rotation_y(-yaw);
    let (p, rl) = trunk_angles(&rot, &inv_h);
    let om = inv_h * w.bodies[r.trunk].angvel();
    let s = strength * r.total_mass / 2087.0;
    let tau_h = Vector::new(-(60_000.0 * s * p + 15_000.0 * s * om.x), 0.0, -(60_000.0 * s * rl + 15_000.0 * s * om.z));
    Rotation::from_rotation_y(yaw) * tau_h
}

/// Apply thrusters for a command. Returns the applied forces for recording.
pub fn apply(w: &mut PhysicsWorld, r: &Robot, cmd: &Command) -> Vec<(Vector, Vector)> {
    if r.thrusters.is_empty() {
        return Vec::new();
    }
    let active = cmd.thrust.iter().any(|v| v.abs() > 1e-3) || cmd.dash;
    // Clear last tick's thruster forces on their mount bodies.
    let mut bodies: Vec<RigidBodyHandle> = r.thrusters.iter().map(|t| t.body).collect();
    bodies.sort_by_key(|h| h.into_raw_parts());
    bodies.dedup();
    for b in &bodies {
        w.bodies[*b].reset_forces(true);
    }
    // Thrusters only fire on input (an always-on attitude hold was tested and fought the gait's lean).
    if !active {
        return Vec::new();
    }
    let yaw = yaw_of(w.bodies[r.trunk].rotation());
    let f_des = desired_force(r, yaw, cmd.thrust, cmd.dash);
    let tau = attitude_torque(w, r, 1.0);
    let out = allocate(w, r, f_des, tau, cmd.dash);
    for (t, (p, f)) in r.thrusters.iter().zip(&out.forces) {
        w.bodies[t.body].add_force_at_point(*f, *p, true);
    }
    out.forces
}
