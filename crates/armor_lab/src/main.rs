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
        tear_strain: Some(env_or("TEAR", 0.35)),
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
        _ => eprintln!("usage: armor_lab dent [E] [yield] | record <speed> <out.json> [E] [yield]"),
    }
}
