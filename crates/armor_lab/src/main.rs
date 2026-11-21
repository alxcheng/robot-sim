//! Metal deformation tests with Rapier's FEM soft bodies (the "tier A hero plate" of the spec).
//!
//! `armor_lab dent [E] [yield]`: a bolted armor plate (rim pinned) is hit by a rigid ball at several
//! speeds; reports peak and permanent dent depth, tears, and ms/step.

use rapier3d::prelude::*;
use std::time::Instant;

const DT: f32 = 1.0 / 60.0;

struct Plate {
    half: Vector,
    n: [usize; 3],
    mass: f32,
}

fn env_or(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn steel(e: f32, yield_: f32) -> SoftBodyMaterial {
    SoftBodyMaterial {
        young_modulus: e,
        poisson_ratio: 0.3,
        elastic_damping_ratio: 1.0,
        plastic_yield: yield_,
        plastic_creep: 60.0,
        plastic_max: 0.6,
        deformation_damping: 4.0,
        tear_strain: Some(env_or("TEAR", 0.35)).filter(|t| *t > 0.0),
        ..Default::default()
    }
}

struct Outcome {
    peak: f32,
    permanent: f32,
    pieces: usize,
    ms: f64,
    frames: Vec<serde_json::Value>,
}

/// Largest downward displacement of a centre-region particle from its starting position (m).
fn dent(w: &PhysicsWorld, h: SoftBodyHandle, rest: &[Vector]) -> f32 {
    let sb = &w.soft_bodies[h];
    let mut d = 0.0f32;
    for (i, p) in sb.particle_positions().enumerate() {
        if let Some(r) = rest.get(i) {
            if r.x.abs() < 0.3 && r.z.abs() < 0.3 {
                d = d.max(r.y - p.y);
            }
        }
    }
    d
}

fn snapshot(w: &PhysicsWorld, ball: RigidBodyHandle, t: f32) -> serde_json::Value {
    let mut meshes = Vec::new();
    for (_, sb) in w.soft_bodies.iter() {
        let pos: Vec<f32> = sb.particle_positions().flat_map(|p| [p.x, p.y, p.z]).collect();
        let tri: Vec<u32> = sb.boundary().iter().flat_map(|t| t.iter().copied()).collect();
        meshes.push(serde_json::json!({"p": pos, "t": tri}));
    }
    let b = w.bodies[ball].translation();
    serde_json::json!({"t": t, "meshes": meshes, "ball": [b.x, b.y, b.z]})
}

fn dent_test(plate: &Plate, mat: SoftBodyMaterial, ball_m: f32, ball_r: f32, speed: f32, record: bool) -> Outcome {
    let mut w = PhysicsWorld::new();
    w.integration_parameters.dt = DT;
    w.integration_parameters.num_solver_iterations = env_or("SUBSTEPS", 4.0) as usize;
    let half = plate.half;
    let builder = SoftBodyBuilder::cuboid(Vector::ZERO, half, plate.n[0], plate.n[1], plate.n[2])
        .cell_model(SoftBodyCellModel::Corotational)
        .material(mat)
        .solver(SoftBodySolver::Fem)
        .mass(plate.mass)
        .can_sleep(false);
    // Bolted rim: every particle within 4 cm of the plate's edge is pinned.
    let pinned: Vec<u32> = builder
        .particle_positions()
        .iter()
        .enumerate()
        .filter(|(_, p)| p.x.abs() > half.x - 0.04 || p.z.abs() > half.z - 0.04)
        .map(|(i, _)| i as u32)
        .collect();
    let rest: Vec<Vector> = builder.particle_positions().to_vec();
    let h = w.insert_soft_body(builder.pinned_particles(pinned));
    w.soft_bodies[h].set_additional_pgs_iterations(env_or("PGS", 0.0) as usize);
    let top_y = half.y;
    let (ball, _) = w.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::new(0.0, if speed > 0.0 { top_y + ball_r + 0.05 } else { 50.0 }, 0.0))
            .linvel(Vector::new(0.0, -speed, 0.0))
            .ccd_enabled(true)
            .gravity_scale(if speed > 0.0 { 1.0 } else { 0.0 })
            .can_sleep(false),
        ColliderBuilder::ball(ball_r).mass(ball_m).restitution(0.1),
    );
    let mut frames = Vec::new();
    let (mut peak, mut times) = (0.0f32, Vec::new());
    let steps = (2.0 / DT) as usize;
    for s in 0..steps {
        let t0 = Instant::now();
        w.step();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
        peak = peak.max(dent(&w, h, &rest));
        // Take the ball away once it has bounced off, so the plate can settle on its own.
        let bv = w.bodies[ball].linvel();
        if s > 10 && bv.y > 0.0 {
            w.bodies[ball].set_translation(Vector::new(0.0, 50.0, 0.0), true);
            w.bodies[ball].set_linvel(Vector::ZERO, true);
            w.bodies[ball].set_gravity_scale(0.0, true);
        }
        if record {
            frames.push(snapshot(&w, ball, s as f32 * DT));
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Outcome { peak, permanent: dent(&w, h, &rest), pieces: w.soft_bodies.len(), ms: times[times.len() / 2], frames }
}

/// A robot from the walker lab, with an FEM chest plate bolted on, launched into a concrete wall.
/// Returns (recording JSON, summary line).
fn wall_crash(build: &str, speed: f32, mat: SoftBodyMaterial) -> (serde_json::Value, String) {
    use walker_lab::{robot, sim};
    let mut sc = sim::scenario("run_mid", 0.0);
    // Same frame as the walker lab's GLT/GMT/GHT builds (Mech-V, deer legs, thrusters).
    sc.spec = robot::RobotSpec {
        legs: robot::LegKind::Deer, foot_scale: 1.0, swing_arms: true, reaction_wheels: true, arm_dof: 3,
        frame: robot::Frame::MechV, strength: 1.3,
        load: match build { "GL" => robot::Load::Light, "GM" => robot::Load::Medium, _ => robot::Load::Heavy },
        thrusters: true,
    };
    let (mut w, r) = sim::build_world(&sc);
    w.integration_parameters.num_solver_iterations = env_or("SUBSTEPS", 4.0) as usize;
    let upper = r.waist.expect("MechV has a waist").1;
    // Front of the chest (largest z of the upper body's colliders).
    let mut front = f32::MIN;
    let mut chest_y = 0.0;
    for c in w.bodies[upper].colliders().to_vec() {
        let aabb = w.colliders[c].compute_aabb();
        front = front.max(aabb.maxs.z);
        chest_y = w.bodies[upper].translation().y;
    }
    // The plate must also clear every other robot part in its footprint (arms, pauldrons, head).
    let plate_half_xy = (0.45f32, 0.30f32);
    for l in &r.links {
        for c in w.bodies[l.body].colliders().to_vec() {
            let bb = w.colliders[c].compute_aabb();
            let overlaps_xy = bb.maxs.x > -plate_half_xy.0 && bb.mins.x < plate_half_xy.0
                && bb.maxs.y > chest_y - plate_half_xy.1 && bb.mins.y < chest_y + plate_half_xy.1;
            if overlaps_xy && bb.maxs.z > front {
                if std::env::var("DEBUG").is_ok() {
                    eprintln!("  {} reaches z {:.2} (chest front {:.2})", l.name, bb.maxs.z, front);
                }
                front = bb.maxs.z;
            }
        }
    }
    // Concrete wall 2.5 m ahead of the chest plate.
    let wall_z = front + 2.5;
    let wall_half = Vector::new(3.0, 3.0, 0.4);
    let wall_c = Vector::new(0.0, wall_half.y, wall_z + wall_half.z);
    w.insert(
        RigidBodyBuilder::fixed().translation(wall_c),
        ColliderBuilder::cuboid(wall_half.x, wall_half.y, wall_half.z).friction(0.8).collision_groups(robot::world_groups()),
    );
    // FEM chest plate: 0.9 × 0.6 × 0.06 m, 2 cm in front of the chest, bolted at 8 points.
    let half = Vector::new(0.45, 0.30, 0.03);
    let centre = Vector::new(0.0, chest_y, front + env_or("GAPZ", 0.06) + half.z);
    let builder = SoftBodyBuilder::cuboid(centre, half, 19, 13, 3)
        .cell_model(SoftBodyCellModel::Corotational)
        .material(mat)
        .solver(SoftBodySolver::Fem)
        .mass(250.0)
        .can_sleep(false);
    let rest: Vec<Vector> = builder.particle_positions().to_vec();
    let h = w.insert_soft_body(builder);
    // Armor mount: a plain rigid body (plate attachments don't hold on multibody links), fixed to
    // the chest by an impulse joint, which is also where a bolt-shear threshold would go.
    let up_pos = *w.bodies[upper].position();
    let mount_pos = Vector::new(0.0, chest_y, centre.z - half.z - 0.01);
    let (mount, _) = w.insert(
        RigidBodyBuilder::dynamic().translation(mount_pos).can_sleep(false),
        ColliderBuilder::cuboid(0.4, 0.28, 0.01).mass(30.0).collision_groups(InteractionGroups::none()),
    );
    let fixed = FixedJointBuilder::new()
        .local_anchor1(up_pos.inverse_transform_point(mount_pos))
        .local_anchor2(Vector::ZERO)
        .contacts_enabled(false);
    if std::env::var("FREEMOUNT").is_err() {
        w.insert_impulse_joint(upper, mount, fixed);
    }
    let bolts = [(-0.4, -0.25), (0.0, -0.25), (0.4, -0.25), (-0.4, 0.0), (0.4, 0.0), (-0.4, 0.25), (0.0, 0.25), (0.4, 0.25)];
    let mut n_att = 0;
    for (i, p) in rest.iter().enumerate() {
        let back = p.z < centre.z - half.z + 1e-3;
        if std::env::var("NOBOLT").is_err() && back && bolts.iter().any(|(bx, by)| (p.x - bx).abs() < 0.06 && (p.y - centre.y - by).abs() < 0.06) {
            let sb = &mut w.soft_bodies[h];
            sb.attach_particle(i, if std::env::var("DIRECT").is_ok() { upper } else { mount }, &w.bodies);
            n_att += 1;
        }
    }
    // Rest positions in the upper body's frame, to measure how far the plate is pushed in.
    let up0 = *w.bodies[upper].position();
    let rest_local: Vec<Vector> = rest.iter().map(|p| up0.inverse_transform_point(*p)).collect();

    // Launch: a 0.15 s push on every link brings the whole robot to `speed` toward the wall
    // (a thruster dash), then it coasts into the wall.
    let push_t = 0.15;
    let total_mass: f32 = r.links.iter().map(|l| w.bodies[l.body].mass()).sum();
    let mut frames = Vec::new();
    let (mut peak_dent, mut peak_decel, mut prev_v) = (0.0f32, 0.0f32, 0.0f32);
    let mut times = Vec::new();
    let steps = (2.5 / DT) as usize;
    for s in 0..steps {
        let t = s as f32 * DT;
        for l in &r.links {
            let b = &mut w.bodies[l.body];
            b.reset_forces(true);
            if t < push_t {
                let m = b.mass();
                b.add_force(Vector::new(0.0, 0.0, m * speed / push_t), true);
            }
        }
        if t < push_t {
            let sb = &mut w.soft_bodies[h];
            sb.reset_forces(true);
            // add_force applies to every particle: give each its own share.
            let m = sb.mass() / sb.num_particles() as f32;
            sb.add_force(Vector::new(0.0, 0.0, m * speed / push_t), true);
        } else if s == (push_t / DT) as usize + 1 {
            w.soft_bodies[h].reset_forces(true);
        }
        let t0 = Instant::now();
        w.step();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
        // Dent: deepest push of a plate particle toward the chest, in the chest's frame.
        let up = *w.bodies[upper].position();
        let sb = &w.soft_bodies[h];
        let mut dent = 0.0f32;
        let mut disp = Vec::with_capacity(sb.num_particles());
        for (i, p) in sb.particle_positions().enumerate() {
            let d = rest_local.get(i).map(|r0| (r0.z - up.inverse_transform_point(p).z).max(0.0)).unwrap_or(0.0);
            dent = dent.max(d);
            disp.push(d);
        }
        peak_dent = peak_dent.max(dent);
        if std::env::var("DEBUG").is_ok() && s % 5 == 0 {
            let tz = w.bodies[r.trunk].translation().z;
            let mz = w.bodies[mount].translation().z;
            let pc = w.soft_bodies[h].center_of_mass();
            eprintln!("t {:.2} trunk z {:.2} vz {:.1} mount z {:.2} v {:?} plate com {:.2} {:.2} {:.2} pieces {} dent {:.2}", t, tz, w.bodies[r.trunk].linvel().z, mz, w.bodies[mount].linvel(), pc.x, pc.y, pc.z, w.soft_bodies.len(), dent);
        }
        let v = w.bodies[r.trunk].linvel().z;
        if t > push_t + DT {
            peak_decel = peak_decel.max((prev_v - v) / DT / 9.81);
        }
        prev_v = v;
        let mut meshes = Vec::new();
        for (sh, sbx) in w.soft_bodies.iter() {
            let pos: Vec<f32> = sbx.particle_positions().flat_map(|p| [p.x, p.y, p.z]).collect();
            let tri: Vec<u32> = sbx.boundary().iter().flat_map(|t| t.iter().copied()).collect();
            let d: Vec<f32> = if sh == h { disp.clone() } else { vec![0.1; sbx.num_particles()] };
            meshes.push(serde_json::json!({"p": pos, "t": tri, "d": d}));
        }
        let b: Vec<[f32; 7]> = r.links.iter().map(|l| {
            let bb = &w.bodies[l.body];
            let (p, q) = (bb.translation(), bb.rotation());
            [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        }).collect();
        let label = if t < push_t { "DASH".to_string() } else { String::new() };
        frames.push(serde_json::json!({"t": t, "b": b, "speed": v, "cmd_speed": speed, "heading": 0.0, "cmd_heading": 0.0,
            "label": label, "feet": 0, "toe": [0.0, 0.0], "contact": [false, false], "soft": meshes, "dent": dent}));
    }
    let sb = &w.soft_bodies[h];
    let mut final_dent = 0.0f32;
    let up = *w.bodies[upper].position();
    for (i, p) in sb.particle_positions().enumerate() {
        if let Some(r0) = rest_local.get(i) {
            final_dent = final_dent.max(r0.z - up.inverse_transform_point(p).z);
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let summary = format!(
        "{build} {:.0} kg at {speed} m/s ({:.0} kJ): plate peak {:.1} cm, permanent {:.1} cm, pieces {}, bolts {} particles, peak decel {:.0} g, median {:.1} ms/step",
        total_mass, 0.5 * total_mass * speed * speed / 1000.0, peak_dent * 100.0, final_dent * 100.0, w.soft_bodies.len(), n_att, peak_decel, times[times.len() / 2]
    );
    let rec = serde_json::json!({
        "scenario": format!("{} smashed into a wall · {speed} m/s", match build { "GL" => "light robot", "GM" => "medium robot", _ => "heavy robot" }),
        "assist": 0.0, "dt": DT, "links": r.links, "platform": null, "pushes": [], "frames": frames,
        "wall": [wall_c.x, wall_c.y, wall_c.z, wall_half.x, wall_half.y, wall_half.z],
        "summary": summary,
    });
    (rec, summary)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let a = |i: usize| args.get(i).cloned();
    match a(1).as_deref() {
        Some("dent") => {
            let e: f32 = a(2).map(|s| s.parse().unwrap()).unwrap_or(2.0e8);
            let y: f32 = a(3).map(|s| s.parse().unwrap()).unwrap_or(0.02);
            let plate = Plate { half: Vector::new(0.6, 0.03, 0.4), n: [25, 3, 17], mass: 450.0 };
            println!("plate 1.2 × 0.8 × 0.06 m, 450 kg, rim bolted | E {e:.1e} yield {y} | ball 80 kg r 0.12 m");
            println!("{:>8} {:>10} {:>10} {:>14} {:>7} {:>9}", "speed", "energy kJ", "peak cm", "permanent cm", "pieces", "ms/step");
            for v in std::env::var("SPEEDS").unwrap_or("0,5,20,80".into()).split(',').map(|x| x.parse::<f32>().unwrap()) {
                let o = dent_test(&plate, steel(e, y), 80.0, 0.12, v, false);
                println!("{:>8.0} {:>10.1} {:>10.1} {:>14.1} {:>7} {:>9.2}", v, 0.5 * 80.0 * v * v / 1000.0, o.peak * 100.0, o.permanent * 100.0, o.pieces, o.ms);
            }
        }
        Some("record") => {
            // record <speed> <out.json> [E] [yield]
            let v: f32 = a(2).unwrap().parse().unwrap();
            let out = a(3).unwrap();
            let e: f32 = a(4).map(|s| s.parse().unwrap()).unwrap_or(2.0e8);
            let y: f32 = a(5).map(|s| s.parse().unwrap()).unwrap_or(0.02);
            let plate = Plate { half: Vector::new(0.6, 0.03, 0.4), n: [25, 3, 17], mass: 450.0 };
            let o = dent_test(&plate, steel(e, y), 80.0, 0.12, v, true);
            println!("{v} m/s: peak {:.1} cm, permanent {:.1} cm, pieces {}", o.peak * 100.0, o.permanent * 100.0, o.pieces);
            let label = std::env::var("LABEL").unwrap_or_default();
            let rec = serde_json::json!({"speed": v, "ball_r": 0.12, "ball_m": 80.0, "peak": o.peak, "permanent": o.permanent, "pieces": o.pieces, "dt": DT, "label": label, "frames": o.frames});
            std::fs::write(out, serde_json::to_string(&rec).unwrap()).unwrap();
        }
        Some("crash") => {
            // crash <build GL|GM|GH> <speed> <out.json> [E] [yield]
            let build = a(2).unwrap();
            let v: f32 = a(3).unwrap().parse().unwrap();
            let out = a(4).unwrap();
            let e: f32 = a(5).map(|s| s.parse().unwrap()).unwrap_or(2.0e8);
            let y: f32 = a(6).map(|s| s.parse().unwrap()).unwrap_or(0.02);
            let (rec, summary) = wall_crash(&build, v, steel(e, y));
            println!("{summary}");
            std::fs::write(out, serde_json::to_string(&rec).unwrap()).unwrap();
        }
        _ => eprintln!("usage: armor_lab dent [E] [yield] | record <speed> <out.json> [E] [yield]"),
    }
}
