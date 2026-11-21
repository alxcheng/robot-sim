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

/// Crumpling metal for the all-metal robot: no tearing (parts bend and stay bent), fast plastic
/// flow with a high cap so crushed shells keep their new shape.
fn crumple_metal(e: f32, yield_: f32) -> SoftBodyMaterial {
    SoftBodyMaterial {
        young_modulus: e,
        poisson_ratio: 0.3,
        elastic_damping_ratio: env_or("EDAMP", 1.0),
        plastic_yield: yield_,
        plastic_creep: env_or("CREEP", 200.0),
        plastic_max: env_or("PMAX", 1.5),
        deformation_damping: env_or("DDAMP", 4.0),
        tear_strain: Some(env_or("TEAR", 0.0)).filter(|t| *t > 0.0),
        ..Default::default()
    }
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

/// One metal shell: an FEM body standing in for a robot part, bolted to a mount on its link.
struct Shell {
    handle: SoftBodyHandle,
    /// The body whose frame the shell's rest shape is expressed in (the link itself when the
    /// bolts are driven, else the welded mount).
    mount: RigidBodyHandle,
    rest_local: Vec<Vector>,
    link: usize,
    /// Driven bolts: pinned particles moved with the link every step.
    driven: Vec<usize>,
    /// Inner frame box in the link frame (centre, half extents): free skin points are kept outside.
    frame: (Vector, Vector),
}

/// Every armored part of the robot as an FEM metal shell, smashed into a wall.
/// Returns (recording JSON, summary line).
fn metal_crash(build: &str, speed: f32, mat: SoftBodyMaterial) -> (serde_json::Value, String) {
    use walker_lab::{robot, sim};
    let mut sc = sim::scenario("run_mid", 0.0);
    sc.spec = robot::RobotSpec {
        legs: robot::LegKind::Deer, foot_scale: 1.0, swing_arms: true, reaction_wheels: true, arm_dof: 3,
        frame: robot::Frame::MechV, strength: 1.3,
        load: match build { "GL" => robot::Load::Light, "GM" => robot::Load::Medium, _ => robot::Load::Heavy },
        thrusters: true,
    };
    let (mut w, r) = sim::build_world(&sc);
    w.integration_parameters.num_solver_iterations = env_or("SUBSTEPS", 4.0) as usize;
    // Optional: drive the joints with the trained thruster policy (runs, then dashes into the wall).
    let pilot: Option<(walker_lab::policy::Mlp, walker_lab::run_ctl::RunGait)> = std::env::var("POLICY").ok().map(|path| {
        let txt = std::fs::read_to_string(&path).expect("policy file");
        let mut net = match serde_json::from_str::<walker_lab::ppo::GaussPolicy>(&txt) {
            Ok(p) => p.net,
            Err(_) => serde_json::from_str::<walker_lab::policy::Mlp>(&txt).expect("policy json"),
        };
        net.expand_inputs(walker_lab::policy::OBS_DIM);
        let base = build.trim_end_matches('T');
        let gtxt = std::fs::read_to_string(format!("out/run_{base}_a0.json")).expect("gait file");
        (net, serde_json::from_str(&gtxt).expect("gait json"))
    });
    let spacing = env_or("SPACING", 0.07);
    // Shells collide with the world; the two legs' shells also collide with each other.
    let world_only = InteractionGroups::new(Group::GROUP_5, Group::GROUP_1 | Group::GROUP_8, InteractionTestMode::And);
    let leg_groups = |side: usize| {
        if side == 0 {
            InteractionGroups::new(Group::GROUP_5 | Group::GROUP_6, Group::GROUP_1 | Group::GROUP_7 | Group::GROUP_8, InteractionTestMode::And)
        } else {
            InteractionGroups::new(Group::GROUP_5 | Group::GROUP_7, Group::GROUP_1 | Group::GROUP_6 | Group::GROUP_8, InteractionTestMode::And)
        }
    };
    let mut shells: Vec<Shell> = Vec::new();
    let mut shell_info: Vec<serde_json::Value> = Vec::new();
    let mut hidden: Vec<usize> = Vec::new();
    let mut front = f32::MIN;
    for (li, l) in r.links.iter().enumerate() {
        // The part's main shape: the largest solid (skip joint balls, gimbals, wheels).
        let main_idx = l
            .shapes
            .iter()
            .enumerate()
            .filter(|(_, s)| !matches!(s.kind, "ball") && !matches!(s.tag, "gimbal" | "wheel"))
            .max_by(|(_, a), (_, b)| (a.half[0] * a.half[1] * a.half[2]).partial_cmp(&(b.half[0] * b.half[1] * b.half[2])).unwrap())
            .map(|(i, _)| i);
        let Some(main_idx) = main_idx else { continue };
        let main = &l.shapes[main_idx];
        // Feet stay rigid soles: they carry the whole robot, more than the bolt joints can hold.
        if l.name.ends_with("_foot") {
            continue;
        }
        if main.half.iter().any(|h| *h < 0.025) {
            continue;
        }
        let body_pos = *w.bodies[l.body].position();
        let link_mass = w.bodies[l.body].mass();
        // Its rigid colliders stop colliding: the shell is the part's surface now.
        for c in w.bodies[l.body].colliders().to_vec() {
            w.colliders[c].set_collision_groups(InteractionGroups::none());
        }
        let half = Vector::new(main.half[0], main.half[1], main.half[2]);
        let n = |h: f32| ((2.0 * h / spacing).round() as usize + 1).clamp(3, env_or("NMAX", 10.0) as usize);
        let (nx, ny, nz) = (n(half.x), n(half.y), n(half.z));
        let local_c = Vector::new(main.offset[0], main.offset[1], main.offset[2]);
        let proto = SoftBodyBuilder::cuboid(Vector::ZERO, half, nx, ny, nz);
        let local_pts: Vec<Vector> = proto.particle_positions().to_vec();
        let world_pts: Vec<Vector> = local_pts.iter().map(|p| body_pos.transform_point(local_c + *p)).collect();
        // Each part has its own bit (GROUP_9 upward): its skin collides with its own frame only.
        let own = Group::from_bits_truncate(1u32 << (8 + shells.len() as u32));
        let base = match robot::leg_side(&l.name) { Some(k) => leg_groups(k), None => world_only };
        let groups = InteractionGroups::new(base.memberships | own, (base.filter - Group::GROUP_8) | own, InteractionTestMode::And);
        let builder = proto
            .positions(world_pts.clone())
            .cell_model(if std::env::var("NEO").is_ok() { SoftBodyCellModel::NeoHookean } else { SoftBodyCellModel::Corotational })
            .material(mat)
            .solver(SoftBodySolver::Fem)
            .mass((env_or("SHELLM", 0.1) * link_mass).max(5.0))
            .particle_radius(env_or("PRAD", 0.01))
            .surface_collider(ColliderBuilder::ball(0.01).friction(env_or("SKINF", 0.2)).collision_groups(groups))
            .can_sleep(false);
        let h = w.insert_soft_body(builder);
        // Bolts: the inner core, plus the face centres of the two largest axes (the part's
        // mounting points); the outer skin is free to dent.
        let mut axes = [(half.x, 0usize), (half.y, 1), (half.z, 2)];
        axes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let mut bolted: Vec<usize> = Vec::new();
        for (i, p) in local_pts.iter().enumerate() {
            let u = [p.x / half.x, p.y / half.y, p.z / half.z];
            // Thin parts (feet, plates) are also bolted across the inner part of their middle layer.
            let thin = axes[2].1;
            let mid_plane = u[thin].abs() < 0.05 && (0..3).filter(|k| *k != thin).all(|k| u[k].abs() < 0.7);
            let core = u.iter().all(|v| v.abs() < 0.65) || mid_plane;
            let face_centre = axes[..2].iter().any(|(_, ax)| {
                (0..3).all(|k| if k == *ax { u[k].abs() > 0.99 } else { u[k].abs() < 0.05 })
            });
            // Joint ends: the whole end faces where the part meets its joints are bolted, so the
            // shell stays fixed to the actual connections (limbs: both ends of the long axis;
            // chest: shoulders and waist; pelvis: hips and waist).
            let long = axes[0].1;
            // Only the part of an end face inside the inner frame's outline is bolted: a bolted point
            // in the crumple zone would be driven into the wall past the frame.
            let core_k0 = env_or("CORE", 0.7);
            let full: [bool; 3] = match l.name.as_str() {
                "upper_body" | "pelvis" => [true, true, false],
                _ => [long == 0, long == 1, long == 2],
            };
            let inside_frame = (0..3).all(|k| full[k] || u[k].abs() <= core_k0 - 0.02);
            let end_face = inside_frame && match l.name.as_str() {
                "upper_body" => u[0].abs() > 0.99 || u[1] < -0.99,
                "pelvis" => u[0].abs() > 0.99 || u[1] > 0.99,
                _ => u[long].abs() > 0.99,
            };
            let in_frame_volume = (0..3).all(|k| full[k] || u[k].abs() <= core_k0 + 0.02);
            let _ = (core, face_centre);
            if in_frame_volume || (env_or("ENDBOLT", 1.0) > 0.0 && end_face) {
                bolted.push(i);
            }
        }
        // Rigid inner frame: covers everything that is pinned (full length along the axes whose end
        // faces bolt to joints, CORE elsewhere), so the skin outside it is a clean crumple zone that
        // the frame stops before any pinned point reaches the wall.
        let core_k = env_or("CORE", 0.7);
        let mut frame_box = (local_c, Vector::ZERO);
        if core_k > 0.0 {
            let long = axes[0].1;
            let full: [bool; 3] = match l.name.as_str() {
                "upper_body" => [true, true, false],
                "pelvis" => [true, true, false],
                _ => [long == 0, long == 1, long == 2],
            };
            let k = |ax: usize, h: f32| if full[ax] { h } else { h * core_k };
            frame_box = (local_c, Vector::new(k(0, half.x), k(1, half.y), k(2, half.z)));
            let core = ColliderBuilder::cuboid(k(0, half.x), k(1, half.y), k(2, half.z))
                .translation(local_c)
                .density(0.0)
                .friction(0.7)
                // Frames hit the world and the metal skins (their own and other parts').
                .collision_groups(InteractionGroups::new(Group::GROUP_2 | own, if env_or("SKINFRAME", 0.0) > 0.0 { Group::GROUP_1 | own } else { Group::GROUP_1 }, InteractionTestMode::And));
            w.insert_collider(core, Some(l.body));
        }
        let drive = env_or("DRIVE", 1.0) > 0.0;
        let (mount, rest_local, driven) = if drive {
            // Driven bolts: pinned particles that follow the link exactly (set every step), so the
            // shell can't jiggle or slide at its connections.
            let sb = &mut w.soft_bodies[h];
            for &i in &bolted {
                sb.set_particle_pinned(i, true);
            }
            (l.body, local_pts.iter().map(|p| local_c + *p).collect::<Vec<_>>(), bolted.clone())
        } else {
            // Mount: a plain rigid body welded to the link (attachments need a non-multibody body).
            let centre_w = body_pos.transform_point(local_c);
            let (mount, _) = w.insert(
                RigidBodyBuilder::dynamic().position(Pose::from_parts(centre_w, body_pos.rotation)).can_sleep(false),
                ColliderBuilder::cuboid(0.05, 0.05, 0.05).mass(5.0).collision_groups(InteractionGroups::none()),
            );
            w.insert_impulse_joint(
                l.body,
                mount,
                FixedJointBuilder::new().local_anchor1(local_c).local_anchor2(Vector::ZERO).contacts_enabled(false),
            );
            for &i in &bolted {
                w.soft_bodies[h].attach_particle(i, mount, &w.bodies);
            }
            (mount, local_pts.clone(), Vec::new())
        };
        for p in &world_pts {
            front = front.max(p.z);
        }
        shell_info.push(serde_json::json!({"link": li, "shape": main_idx,
            "rest": rest_local.iter().flat_map(|p| [p.x, p.y, p.z]).collect::<Vec<f32>>()}));
        shells.push(Shell { handle: h, mount, rest_local, link: li, driven, frame: frame_box });
        hidden.push(li);
    }
    // Concrete wall 2.5 m ahead (12 m when the robot runs up to it first).
    let wall_z = front + if pilot.is_some() { env_or("RUNUP", 12.0) } else { 2.5 };
    let wall_half = Vector::new(4.0, 3.5, 0.4);
    let wall_c = Vector::new(0.0, wall_half.y, wall_z + wall_half.z);
    w.insert(
        RigidBodyBuilder::fixed().translation(wall_c),
        ColliderBuilder::cuboid(wall_half.x, wall_half.y, wall_half.z).friction(0.8).collision_groups(robot::world_groups()),
    );
    let total_mass: f32 = r.links.iter().map(|l| w.bodies[l.body].mass()).sum::<f32>()
        + shells.iter().map(|s| w.soft_bodies[s.handle].mass() + if s.driven.is_empty() { w.bodies[s.mount].mass() } else { 0.0 }).sum::<f32>();
    let n_particles: usize = shells.iter().map(|s| w.soft_bodies[s.handle].num_particles()).sum();

    // Launch: a 0.15 s push on every rigid body and shell particle (a thruster dash).
    let push_t = 0.15;
    let mut frames = Vec::new();
    let (mut peak_dent, mut peak_decel, mut prev_v, mut times) = (0.0f32, 0.0f32, 0.0f32, Vec::new());
    let mut failed_at = None;
    let mut max_stretch = 1.0f32;
    let (mut jiggle_sum, mut jiggle_n, mut rest_jiggle_sum, mut rest_jiggle_n) = (0.0f64, 0usize, 0.0f64, 0usize);
    let mut all_bodies: Vec<RigidBodyHandle> = r.links.iter().map(|l| l.body).chain(shells.iter().map(|s| s.mount)).collect();
    all_bodies.sort_by_key(|h| h.into_raw_parts());
    all_bodies.dedup();
    let steps = (env_or("DUR", 2.5) / DT) as usize;
    let mut tracker = walker_lab::policy::ContactTracker::default();
    let mut spool = walker_lab::thrust::Spool::default();
    // Piloted: the launch starts when the robot is LAUNCH_AT m from the wall, else at t = 0.
    let launch_at = env_or("LAUNCH_AT", 3.5);
    let mut launch: Option<(f32, f32)> = if pilot.is_some() { None } else { Some((0.0, speed)) };
    for s in 0..steps {
        let t = s as f32 * DT;
        for b in &all_bodies {
            w.bodies[*b].reset_forces(true);
        }
        for sh in &shells {
            w.soft_bodies[sh.handle].reset_forces(true);
        }
        let tz = w.bodies[r.trunk].translation().z;
        if launch.is_none() && pilot.is_some() && tz > wall_z - launch_at {
            launch = Some((t, speed));
        }
        let launching = launch.map(|(t0, _)| t >= t0 && t < t0 + push_t).unwrap_or(false);
        let mut thr_rec: Vec<[f32; 6]> = Vec::new();
        if let Some((net, gait)) = &pilot {
            // Run toward the wall, then a forward dash on the thrusters for the launch.
            let run_v = (1.5 + 2.0 * t).min(6.0);
            let mut cmd = walker_lab::controller::Command { speed: run_v, heading: 0.0, ..Default::default() };
            if launch.map(|(t0, _)| t >= t0 && t < t0 + 0.6).unwrap_or(false) {
                cmd.thrust = [0.0, 0.0, 1.0];
                cmd.dash = true;
            }
            let o = walker_lab::policy::observe(&w, &r, cmd, &tracker);
            let a = net.forward_norm(&o);
            let real: Vec<f32> = a.iter().enumerate().map(|(i, v)| v * net.out_std[i] + net.out_mean[i]).collect();
            let act = walker_lab::run_ctl::Action::from_slice(&real);
            walker_lab::run_ctl::apply_action(&mut w, &r, &act, gait.kd);
            walker_lab::run_ctl::reflexes(&mut w, &r, gait.wheel_k, 0.0, gait.lean + gait.lean_v * cmd.speed.max(0.0));
            let forces = walker_lab::thrust::apply(&mut w, &r, &cmd, &mut spool, 1.0);
            thr_rec = forces.iter().filter(|(_, f)| f.length() > 50.0).map(|(p, f)| [p.x, p.y, p.z, f.x / 1000.0, f.y / 1000.0, f.z / 1000.0]).collect();
        }
        if launching {
            // Extra launch force (beyond the thrusters) so the robot hits the wall at `speed`.
            let vz = w.bodies[r.trunk].linvel().z;
            let need = ((speed - vz) / (push_t * 0.5)).max(0.0);
            for b in &all_bodies {
                let rb = &mut w.bodies[*b];
                let m = rb.mass();
                rb.add_force(Vector::new(0.0, 0.0, m * need), true);
            }
            for sh in &shells {
                let sb = &mut w.soft_bodies[sh.handle];
                let m = sb.mass() / sb.num_particles() as f32;
                sb.add_force(Vector::new(0.0, 0.0, m * need), true);
            }
        }
        // Driven bolts: move each pinned particle to where its link will be after this step.
        for sh in &shells {
            if sh.driven.is_empty() {
                continue;
            }
            let lb = &w.bodies[sh.mount];
            let (pos, v, om) = (*lb.position(), lb.linvel(), lb.angvel());
            let next_rot = Rotation::from_scaled_axis(om * DT) * pos.rotation;
            let next = Pose::from_parts(pos.translation + v * DT, next_rot);
            let sb = &mut w.soft_bodies[sh.handle];
            for &i in &sh.driven {
                sb.set_particle_kinematic_target(i, next.transform_point(sh.rest_local[i]));
            }
        }
        let t0 = Instant::now();
        let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| w.step())).is_ok();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
        if !ok {
            failed_at = Some(t);
            break;
        }
        tracker.update([walker_lab::controller::foot_in_contact(&w, &r.legs[0]), walker_lab::controller::foot_in_contact(&w, &r.legs[1])], DT);
        // Stability guardrail: clamp runaway velocities (the spec's post-step safety pass).
        let mut bad = false;
        for b in &all_bodies {
            let rb = &mut w.bodies[*b];
            let v = rb.linvel();
            if !v.is_finite() {
                bad = true;
            } else if v.length() > 60.0 {
                rb.set_linvel(v * (60.0 / v.length()), true);
            }
        }
        if bad {
            failed_at = Some(t);
            break;
        }
        // Skin against its own frame (done by hand: a frame inside a solid skin can't be a contact):
        // a free skin point pushed into the frame goes back to the frame's surface, and no skin
        // point may end up more than MAXD from where it belongs on the part.
        let maxd = env_or("MAXD", 0.25);
        for sh in &shells {
            if sh.driven.is_empty() || sh.frame.1 == Vector::ZERO {
                continue;
            }
            let lb = &w.bodies[sh.mount];
            let (lp, lv, lw) = (*lb.position(), lb.linvel(), lb.angvel());
            let (fc, fh) = sh.frame;
            let sb = &mut w.soft_bodies[sh.handle];
            for i in 0..sb.num_particles() {
                if sh.driven.contains(&i) {
                    continue;
                }
                let p = sb.particle_position(i);
                let mut q = lp.inverse_transform_point(p);
                let mut moved = false;
                let d = q - fc;
                if d.x.abs() < fh.x && d.y.abs() < fh.y && d.z.abs() < fh.z {
                    // Out through the nearest face.
                    let gaps = [fh.x - d.x.abs(), fh.y - d.y.abs(), fh.z - d.z.abs()];
                    let ax = (0..3).min_by(|a, b| gaps[*a].partial_cmp(&gaps[*b]).unwrap()).unwrap();
                    let mut dd = d;
                    dd[ax] = fh[ax].copysign(if d[ax] == 0.0 { 1.0 } else { d[ax] });
                    q = fc + dd;
                    moved = true;
                }
                let off = q - sh.rest_local[i];
                if off.length() > maxd {
                    q = sh.rest_local[i] + off * (maxd / off.length());
                    moved = true;
                }
                if moved {
                    let np = lp.transform_point(q);
                    sb.set_particle_position(i, np);
                    sb.set_particle_velocity(i, lv + lw.cross(np - lp.translation));
                }
            }
        }
        // Soft guardrail: a particle may not move faster than 30 m/s relative to its mount.
        let vclamp = env_or("PVCLAMP", 12.0);
        for sh in &shells {
            let mv = w.bodies[sh.mount].linvel();
            let sb = &mut w.soft_bodies[sh.handle];
            for i in 0..sb.num_particles() {
                let rel = sb.particle_velocity(i) - mv;
                if rel.length() > vclamp {
                    sb.set_particle_velocity(i, mv + rel * (vclamp / rel.length()));
                }
            }
        }
        // Jiggle: RMS speed of the metal relative to its part's own rigid motion, before the launch
        // (standing) and over the last 0.8 s (after the crash has settled).
        let settle = t > env_or("DUR", 2.5) - 0.8;
        if settle || t < push_t {
            for sh in &shells {
                let mb = &w.bodies[sh.mount];
                let (mp, mv, mw) = (mb.translation(), mb.linvel(), mb.angvel());
                let sb = &w.soft_bodies[sh.handle];
                for i in 0..sb.num_particles() {
                    let rel = sb.particle_velocity(i) - (mv + mw.cross(sb.particle_position(i) - mp));
                    if settle {
                        jiggle_sum += rel.length_squared() as f64;
                        jiggle_n += 1;
                    } else {
                        rest_jiggle_sum += rel.length_squared() as f64;
                        rest_jiggle_n += 1;
                    }
                }
            }
        }
        // Glitch metric: the most-stretched structural edge (current / rest length).
        for sh in &shells {
            let sb = &w.soft_bodies[sh.handle];
            for c in sb.cells() {
                let v = c.vertices;
                for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
                    let (ia, ib) = (v[a] as usize, v[b] as usize);
                    let rest = (sh.rest_local[ia] - sh.rest_local[ib]).length();
                    let len = (sb.particle_position(ia) - sb.particle_position(ib)).length();
                    if rest > 1e-4 {
                        if len / rest > max_stretch.max(2.5) && std::env::var("DEBUG").is_ok() {
                            let pa = sh.driven.contains(&ia);
                            let pb = sh.driven.contains(&ib);
                            eprintln!("  stretch {:.1}x t {:.2} {} pinned {}/{} rest a {:?} b {:?} pos a {:?}", len / rest, t, r.links[sh.link].name, pa, pb, sh.rest_local[ia], sh.rest_local[ib], sb.particle_position(ia));
                        }
                        max_stretch = max_stretch.max(len / rest);
                    }
                }
            }
        }
        // Dent per particle: how far it moved from its bolted rest position, in its mount's frame.
        let mut meshes = Vec::new();
        let mut dent_now = 0.0f32;
        for sh in &shells {
            let mp = *w.bodies[sh.mount].position();
            for (ph, sb) in w.soft_bodies.iter() {
                let mine = ph == sh.handle || sb.origin() == Some(sh.handle);
                if !mine {
                    continue;
                }
                // Colour value per particle: the metal's strain there (largest |length/rest - 1| over
                // its cell edges), so bolt slip and rigid motion don't light up; only real bending.
                let mut d = vec![0.0f32; sb.num_particles()];
                if ph == sh.handle {
                    for c in sb.cells() {
                        let v = c.vertices;
                        for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
                            let (ia, ib) = (v[a] as usize, v[b] as usize);
                            let rest = (sh.rest_local[ia] - sh.rest_local[ib]).length();
                            if rest > 1e-4 {
                                let e = ((sb.particle_position(ia) - sb.particle_position(ib)).length() / rest - 1.0).abs();
                                d[ia] = d[ia].max(e);
                                d[ib] = d[ib].max(e);
                            }
                        }
                    }
                }
                if ph == sh.handle {
                    dent_now = dent_now.max(d.iter().cloned().fold(0.0, f32::max));
                    let _ = mp;
                }
                let pos: Vec<f32> = sb.particle_positions().flat_map(|p| [p.x, p.y, p.z]).collect();
                let tri: Vec<u32> = sb.boundary().iter().flat_map(|t| t.iter().copied()).collect();
                meshes.push(serde_json::json!({"p": pos, "t": tri, "d": d}));
            }
        }
        peak_dent = peak_dent.max(dent_now);
        let v = w.bodies[r.trunk].linvel().z;
        if launch.map(|(t0, _)| t > t0 + push_t + DT).unwrap_or(false) {
            peak_decel = peak_decel.max((prev_v - v) / DT / 9.81);
        }
        prev_v = v;
        let b: Vec<[f32; 7]> = r.links.iter().map(|l| {
            let bb = &w.bodies[l.body];
            let (p, q) = (bb.translation(), bb.rotation());
            [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        }).collect();
        if std::env::var("DEBUG").is_ok() && s % 5 == 0 {
            eprintln!("t {:.2} trunk z {:.2} vz {:.1} pieces {} dent {:.2}", t, w.bodies[r.trunk].translation().z, v, w.soft_bodies.len(), dent_now);
        }
        frames.push(serde_json::json!({"t": t, "b": b, "speed": v, "cmd_speed": speed, "heading": 0.0, "cmd_heading": 0.0,
            "label": if launching { "DASH" } else { "" }, "feet": 0, "toe": [0.0, 0.0], "contact": [false, false], "soft": meshes, "dent": dent_now, "thr": thr_rec}));
    }
    if std::env::var("DEBUG").is_ok() {
        for sh in &shells {
            let mp = *w.bodies[sh.mount].position();
            let sb = &w.soft_bodies[sh.handle];
            let att: Vec<u32> = sb.particle_attachments().iter().map(|a| a.particle).collect();
            let (mut dmax, mut amax) = (0.0f32, 0.0f32);
            for (i, p) in sb.particle_positions().enumerate() {
                let d = (mp.inverse_transform_point(p) - sh.rest_local[i]).length();
                dmax = dmax.max(d);
                if att.contains(&(i as u32)) {
                    amax = amax.max(d);
                }
            }
            eprintln!("  {:16} particles {:3} bolted {:3} max disp {:.3} bolted disp {:.3}", r.links[sh.link].name, sb.num_particles(), att.len(), dmax, amax);
        }
    }
    let mut final_def = Vec::new();
    for sh in &shells {
        let mp = *w.bodies[sh.mount].position();
        let sb = &w.soft_bodies[sh.handle];
        let d = sb.particle_positions().enumerate().map(|(i, p)| (mp.inverse_transform_point(p) - sh.rest_local[i]).length()).fold(0.0f32, f32::max);
        let mut bend = 0.0f32;
        for c in sb.cells() {
            let v = c.vertices;
            for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
                let (ia, ib) = (v[a] as usize, v[b] as usize);
                let rest = (sh.rest_local[ia] - sh.rest_local[ib]).length();
                let len = (sb.particle_position(ia) - sb.particle_position(ib)).length();
                bend = bend.max((len - rest).abs());
            }
        }
        let _ = d;
        let d = bend;
        let bolt_ids: Vec<usize> = if sh.driven.is_empty() { sb.particle_attachments().iter().map(|a| a.particle as usize).collect() } else { sh.driven.clone() };
        let slip = bolt_ids.iter().map(|&i| (mp.inverse_transform_point(sb.particle_position(i)) - sh.rest_local[i]).length()).fold(0.0f32, f32::max);
        final_def.push((format!("{} (bolt slip {:.0})", r.links[sh.link].name, slip * 100.0), d));
    }
    final_def.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top: Vec<String> = final_def.iter().take(4).map(|(n, d)| format!("{n} {:.0} cm", d * 100.0)).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pieces = w.soft_bodies.len();
    let summary = format!(
        "jiggle after crash {:.2} m/s rms | max edge stretch {max_stretch:.2}× | most bent (largest permanent change in a metal edge): {} | {build} all-metal ({} shells, {} particles) {:.0} kg at {speed} m/s ({:.0} kJ): peak strain {:.0}%, {} metal pieces, peak decel {:.0} g, median {:.0} ms/step{}",
        (jiggle_sum / jiggle_n.max(1) as f64).sqrt(),
        top.join(", "),
        shells.len(), n_particles, total_mass, 0.5 * total_mass * speed * speed / 1000.0, peak_dent * 100.0, pieces, peak_decel,
        times.get(times.len() / 2).copied().unwrap_or(0.0),
        failed_at.map(|t| format!(" — SIMULATION FAILED at t {t:.2} s (kept frames up to there)")).unwrap_or_default()
    );
    let rec = serde_json::json!({
        "scenario": format!("all-metal {} robot smashed into a wall · {speed} m/s", match build { "GL" => "light", "GM" => "medium", _ => "heavy" }),
        "assist": 0.0, "dt": DT, "links": r.links, "platform": null, "pushes": [], "frames": frames,
        "wall": [wall_c.x, wall_c.y, wall_c.z, wall_half.x, wall_half.y, wall_half.z],
        "hide": hidden, "shells": shell_info, "summary": summary,
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
        Some("metal") => {
            // metal <build GL|GM|GH> <speed> <out.json> [E] [yield]
            let build = a(2).unwrap();
            let v: f32 = a(3).unwrap().parse().unwrap();
            let out = a(4).unwrap();
            let e: f32 = a(5).map(|s| s.parse().unwrap()).unwrap_or(5.0e7);
            let y: f32 = a(6).map(|s| s.parse().unwrap()).unwrap_or(0.01);
            let (rec, summary) = metal_crash(&build, v, crumple_metal(e, y));
            println!("{summary}");
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
