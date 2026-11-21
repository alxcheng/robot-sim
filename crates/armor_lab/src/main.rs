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

/// One metal shell: an FEM body standing in for a robot part, bolted to a mount on its link.
struct Shell {
    handle: SoftBodyHandle,
    mount: RigidBodyHandle,
    rest_local: Vec<Vector>,
    link: usize,
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
    let spacing = env_or("SPACING", 0.12);
    // Shells collide with the world; the two legs' shells also collide with each other.
    let world_only = InteractionGroups::new(Group::GROUP_5, Group::GROUP_1, InteractionTestMode::And);
    let leg_groups = |side: usize| {
        if side == 0 {
            InteractionGroups::new(Group::GROUP_5 | Group::GROUP_6, Group::GROUP_1 | Group::GROUP_7, InteractionTestMode::And)
        } else {
            InteractionGroups::new(Group::GROUP_5 | Group::GROUP_7, Group::GROUP_1 | Group::GROUP_6, InteractionTestMode::And)
        }
    };
    let mut shells: Vec<Shell> = Vec::new();
    let mut hidden: Vec<usize> = Vec::new();
    let mut front = f32::MIN;
    for (li, l) in r.links.iter().enumerate() {
        // The part's main shape: the largest solid (skip joint balls, gimbals, wheels).
        let main = l
            .shapes
            .iter()
            .filter(|s| !matches!(s.kind, "ball") && !matches!(s.tag, "gimbal" | "wheel"))
            .max_by(|a, b| (a.half[0] * a.half[1] * a.half[2]).partial_cmp(&(b.half[0] * b.half[1] * b.half[2])).unwrap());
        let Some(main) = main else { continue };
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
        let n = |h: f32| ((2.0 * h / spacing).round() as usize + 1).clamp(3, 6);
        let (nx, ny, nz) = (n(half.x), n(half.y), n(half.z));
        let local_c = Vector::new(main.offset[0], main.offset[1], main.offset[2]);
        let proto = SoftBodyBuilder::cuboid(Vector::ZERO, half, nx, ny, nz);
        let local_pts: Vec<Vector> = proto.particle_positions().to_vec();
        let world_pts: Vec<Vector> = local_pts.iter().map(|p| body_pos.transform_point(local_c + *p)).collect();
        let groups = match robot::leg_side(&l.name) { Some(k) => leg_groups(k), None => world_only };
        let builder = proto
            .positions(world_pts.clone())
            .cell_model(SoftBodyCellModel::Corotational)
            .material(mat)
            .solver(SoftBodySolver::Fem)
            .mass((0.3 * link_mass).max(10.0))
            .surface_collider(ColliderBuilder::ball(0.01).friction(0.7).collision_groups(groups))
            .can_sleep(false);
        let h = w.insert_soft_body(builder);
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
        // Bolts: the inner core, plus the face centres of the two largest axes (the part's
        // mounting points); the outer skin is free to dent.
        let mut axes = [(half.x, 0usize), (half.y, 1), (half.z, 2)];
        axes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (i, p) in local_pts.iter().enumerate() {
            let u = [p.x / half.x, p.y / half.y, p.z / half.z];
            // Thin parts (feet, plates) are also bolted across the inner part of their middle layer.
            let thin = axes[2].1;
            let mid_plane = u[thin].abs() < 0.05 && (0..3).filter(|k| *k != thin).all(|k| u[k].abs() < 0.7);
            let core = u.iter().all(|v| v.abs() < 0.65) || mid_plane;
            let face_centre = axes[..2].iter().any(|(_, ax)| {
                (0..3).all(|k| if k == *ax { u[k].abs() > 0.99 } else { u[k].abs() < 0.05 })
            });
            if core || face_centre {
                w.soft_bodies[h].attach_particle(i, mount, &w.bodies);
            }
        }
        let rest_local: Vec<Vector> = local_pts.clone();
        for p in &world_pts {
            front = front.max(p.z);
        }
        shells.push(Shell { handle: h, mount, rest_local, link: li });
        hidden.push(li);
    }
    // Concrete wall 2.5 m ahead.
    let wall_z = front + 2.5;
    let wall_half = Vector::new(4.0, 3.5, 0.4);
    let wall_c = Vector::new(0.0, wall_half.y, wall_z + wall_half.z);
    w.insert(
        RigidBodyBuilder::fixed().translation(wall_c),
        ColliderBuilder::cuboid(wall_half.x, wall_half.y, wall_half.z).friction(0.8).collision_groups(robot::world_groups()),
    );
    let total_mass: f32 = r.links.iter().map(|l| w.bodies[l.body].mass()).sum::<f32>()
        + shells.iter().map(|s| w.soft_bodies[s.handle].mass() + w.bodies[s.mount].mass()).sum::<f32>();
    let n_particles: usize = shells.iter().map(|s| w.soft_bodies[s.handle].num_particles()).sum();

    // Launch: a 0.15 s push on every rigid body and shell particle (a thruster dash).
    let push_t = 0.15;
    let mut frames = Vec::new();
    let (mut peak_dent, mut peak_decel, mut prev_v, mut times) = (0.0f32, 0.0f32, 0.0f32, Vec::new());
    let mut failed_at = None;
    let all_bodies: Vec<RigidBodyHandle> = r.links.iter().map(|l| l.body).chain(shells.iter().map(|s| s.mount)).collect();
    let steps = (env_or("DUR", 2.5) / DT) as usize;
    for s in 0..steps {
        let t = s as f32 * DT;
        for b in &all_bodies {
            let rb = &mut w.bodies[*b];
            rb.reset_forces(true);
            if t < push_t {
                let m = rb.mass();
                rb.add_force(Vector::new(0.0, 0.0, m * speed / push_t), true);
            }
        }
        for sh in &shells {
            let sb = &mut w.soft_bodies[sh.handle];
            sb.reset_forces(true);
            if t < push_t {
                let m = sb.mass() / sb.num_particles() as f32;
                sb.add_force(Vector::new(0.0, 0.0, m * speed / push_t), true);
            }
        }
        let t0 = Instant::now();
        let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| w.step())).is_ok();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
        if !ok {
            failed_at = Some(t);
            break;
        }
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
                let d: Vec<f32> = sb
                    .particle_positions()
                    .enumerate()
                    .map(|(i, p)| {
                        if ph == sh.handle {
                            sh.rest_local.get(i).map(|r0| (mp.inverse_transform_point(p) - *r0).length()).unwrap_or(0.1)
                        } else {
                            0.1
                        }
                    })
                    .collect();
                if ph == sh.handle {
                    dent_now = dent_now.max(d.iter().cloned().fold(0.0, f32::max));
                }
                let pos: Vec<f32> = sb.particle_positions().flat_map(|p| [p.x, p.y, p.z]).collect();
                let tri: Vec<u32> = sb.boundary().iter().flat_map(|t| t.iter().copied()).collect();
                meshes.push(serde_json::json!({"p": pos, "t": tri, "d": d}));
            }
        }
        peak_dent = peak_dent.max(dent_now);
        let v = w.bodies[r.trunk].linvel().z;
        if t > push_t + DT {
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
            "label": if t < push_t { "DASH" } else { "" }, "feet": 0, "toe": [0.0, 0.0], "contact": [false, false], "soft": meshes, "dent": dent_now}));
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
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pieces = w.soft_bodies.len();
    let summary = format!(
        "{build} all-metal ({} shells, {} particles) {:.0} kg at {speed} m/s ({:.0} kJ): deepest dent {:.1} cm, {} metal pieces, peak decel {:.0} g, median {:.0} ms/step{}",
        shells.len(), n_particles, total_mass, 0.5 * total_mass * speed * speed / 1000.0, peak_dent * 100.0, pieces, peak_decel,
        times.get(times.len() / 2).copied().unwrap_or(0.0),
        failed_at.map(|t| format!(" — SIMULATION FAILED at t {t:.2} s (kept frames up to there)")).unwrap_or_default()
    );
    let rec = serde_json::json!({
        "scenario": format!("all-metal {} robot smashed into a wall · {speed} m/s", match build { "GL" => "light", "GM" => "medium", _ => "heavy" }),
        "assist": 0.0, "dt": DT, "links": r.links, "platform": null, "pushes": [], "frames": frames,
        "wall": [wall_c.x, wall_c.y, wall_c.z, wall_half.x, wall_half.y, wall_half.z],
        "hide": hidden, "summary": summary,
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
            let e: f32 = a(5).map(|s| s.parse().unwrap()).unwrap_or(2.0e8);
            let y: f32 = a(6).map(|s| s.parse().unwrap()).unwrap_or(0.02);
            let (rec, summary) = metal_crash(&build, v, steel(e, y));
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
