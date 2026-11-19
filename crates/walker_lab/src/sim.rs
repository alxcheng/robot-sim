//! Scenario runner: builds a world, drives the controller with a command schedule, applies pushes,
//! records frames for rendering, and reports metrics.

use crate::controller::*;
use crate::robot::{self, Robot, RobotSpec};
use rapier3d::prelude::*;
use serde::Serialize;

pub const DT: f32 = 1.0 / 60.0;
pub const SUBSTEPS: usize = 4;

#[derive(Clone, Copy, Serialize)]
pub struct Push {
    pub t: f32,
    /// Push size expressed as the velocity change it would give the whole robot (m/s), heading frame.
    pub dv_fwd: f32,
    pub dv_left: f32,
}

#[derive(Clone)]
pub struct Scenario {
    pub name: String,
    pub duration: f32,
    /// (start time, command); piecewise constant, speed ramps linearly over 1.5 s.
    pub commands: Vec<(f32, Command)>,
    pub pushes: Vec<Push>,
    /// Raised platform (height m, ends at z m) the robot starts on.
    pub platform: Option<(f32, f32)>,
    pub assist: f32,
    pub record: bool,
    /// Stop at the first fall (tuning) or reset and continue (soak).
    pub reset_on_fall: bool,
    pub spec: RobotSpec,
}

#[derive(Serialize, Default, Clone)]
pub struct Metrics {
    pub scenario: String,
    pub assist: f32,
    pub sim_seconds: f32,
    pub falls: u32,
    pub first_fall_t: Option<f32>,
    pub steps_taken: u32,
    pub mean_speed: f32,
    pub mean_cmd_speed: f32,
    pub speed_err: f32,
    pub heading_err_end: f32,
    pub mean_stance_slip: f32,
    pub flight_fraction: f32,
    /// Mean duration of one airborne phase (ms) and how many there were.
    pub mean_flight_ms: f32,
    pub flight_phases: u32,
    /// Std-dev of centre-of-mass height (m): bounce.
    pub com_y_std: f32,
    pub non_finite: bool,
    pub max_tilt_deg: f32,
    pub wall_ms: f64,
}

#[derive(Serialize)]
pub struct Frame {
    pub t: f32,
    pub b: Vec<[f32; 7]>,
    pub speed: f32,
    pub cmd_speed: f32,
    pub heading: f32,
    pub cmd_heading: f32,
    pub label: String,
    pub feet: u8,
}

#[derive(Serialize)]
pub struct Recording {
    pub scenario: String,
    pub assist: f32,
    pub dt: f32,
    pub links: Vec<robot::LinkInfo>,
    pub platform: Option<(f32, f32)>,
    pub pushes: Vec<Push>,
    pub frames: Vec<Frame>,
}

fn build_world(sc: &Scenario) -> (PhysicsWorld, Robot) {
    let mut w = PhysicsWorld::new();
    w.integration_parameters.dt = DT;
    w.integration_parameters.num_solver_iterations = SUBSTEPS;
    let ground = w.insert_body(RigidBodyBuilder::fixed());
    w.insert_collider(
        ColliderBuilder::cuboid(400.0, 0.5, 400.0)
            .translation(Vector::new(0.0, -0.5, 0.0))
            .friction(1.0)
            .collision_groups(robot::world_groups()),
        Some(ground),
    );
    let mut origin = Vector::ZERO;
    if let Some((h, z_end)) = sc.platform {
        let z0 = -8.0;
        w.insert_collider(
            ColliderBuilder::cuboid(4.0, h * 0.5, (z_end - z0) * 0.5)
                .translation(Vector::new(0.0, h * 0.5, (z_end + z0) * 0.5))
                .friction(1.0)
                .collision_groups(robot::world_groups()),
            Some(ground),
        );
        origin = Vector::new(0.0, h, z0 + 2.0);
    }
    let r = robot::build(&mut w, origin, sc.spec);
    (w, r)
}

fn command_at(sc: &Scenario, t: f32) -> Command {
    let mut cur = Command::default();
    let mut prev = Command::default();
    let mut t0 = 0.0;
    for (ts, c) in &sc.commands {
        if *ts <= t {
            prev = cur;
            cur = *c;
            t0 = *ts;
        }
    }
    let a = ((t - t0) / 1.5).clamp(0.0, 1.0);
    Command { speed: prev.speed + (cur.speed - prev.speed) * a, heading: cur.heading }
}

pub fn run(sc: &Scenario, gait: Gait) -> (Metrics, Option<Recording>) {
    let a = sc.assist;
    run_with(sc, &move || Box::new(Controller::new(gait, a)) as Box<dyn Ctl>)
}

pub fn run_with(sc: &Scenario, mk: &dyn Fn() -> Box<dyn Ctl>) -> (Metrics, Option<Recording>) {
    let start = std::time::Instant::now();
    let (mut w, mut r) = build_world(sc);
    let mut ctl = mk();
    let mut m = Metrics { scenario: sc.name.clone(), assist: sc.assist, ..Default::default() };
    let mut rec = (sc.record || std::env::var("WREC").is_ok()).then(|| Recording {
        scenario: sc.name.clone(),
        assist: sc.assist,
        dt: DT,
        links: r.links.clone(),
        platform: sc.platform,
        pushes: sc.pushes.clone(),
        frames: Vec::new(),
    });

    let n = (sc.duration / DT).round() as usize;
    let (mut speed_sum, mut cmd_sum, mut err_sum, mut slip_sum, mut samples) = (0.0, 0.0, 0.0, 0.0, 0usize);
    let mut flight = 0usize;
    let (mut in_air, mut phases, mut air_frames) = (false, 0u32, 0usize);
    let (mut ysum, mut y2sum) = (0.0f64, 0.0f64);
    let mut push_idx = 0;
    // Pushes are 0.1 s force pulses: instantaneous impulses on a multibody link get overwritten
    // when Rapier syncs link velocities from the multibody's generalized velocities.
    const PUSH_DUR: f32 = 0.1;
    let mut push_force = Vector::ZERO;
    let mut push_until = -1.0f32;
    let mut label_until = -1.0f32;
    let mut label = String::new();

    for i in 0..n {
        let t = i as f32 * DT;
        let cmd = command_at(sc, t);
        let tel = ctl.tick(&mut w, &r, cmd, DT);

        while push_idx < sc.pushes.len() && sc.pushes[push_idx].t <= t {
            let p = sc.pushes[push_idx];
            let h = Rotation::from_rotation_y(tel.yaw);
            let dv = h * Vector::new(p.dv_left, 0.0, p.dv_fwd);
            push_force = dv * r.total_mass / PUSH_DUR;
            push_until = t + PUSH_DUR;
            label = format!("PUSH {:.2} m/s", (p.dv_fwd * p.dv_fwd + p.dv_left * p.dv_left).sqrt());
            label_until = t + 1.2;
            push_idx += 1;
        }

        {
            let trunk = &mut w.bodies[r.trunk];
            trunk.reset_forces(true);
            if t < push_until {
                trunk.add_force(push_force, true);
            }
        }
        w.step();

        // Post-step safety pass (spec §4.2): finite check + velocity clamp.
        let mut finite = true;
        for l in &r.links {
            let b = &mut w.bodies[l.body];
            let v = b.linvel();
            if !v.is_finite() || !b.translation().is_finite() {
                finite = false;
            } else if v.length() > 40.0 {
                b.set_linvel(v.normalize() * 40.0, true);
            }
        }
        if !finite {
            m.non_finite = true;
            break;
        }

        let fallen = tel.trunk_tilt > 0.9 || tel.com[1] - sc.platform.map_or(0.0, |p| p.0).min(tel.com[1]) < 1.2;
        let fallen = fallen || tel.com[1] < 1.2;
        if std::env::var("WDEBUG").is_ok() && i % 6 == 0 {
            eprintln!("t {t:5.2} st {:?} comY {:.2} tilt {:.2} yaw {:+.2} vF {:+.2} feet {} slip {:.2}", ctl.steps(), tel.com[1], tel.trunk_tilt, tel.yaw, tel.fwd_speed, tel.feet_in_contact, tel.stance_foot_slip);
        }
        m.max_tilt_deg = m.max_tilt_deg.max(tel.trunk_tilt.to_degrees());
        if t > 2.0 {
            speed_sum += tel.fwd_speed;
            cmd_sum += cmd.speed;
            err_sum += (tel.fwd_speed - cmd.speed).abs();
            slip_sum += tel.stance_foot_slip;
            if tel.feet_in_contact == 0 {
                flight += 1;
                air_frames += 1;
                if !in_air {
                    phases += 1;
                    in_air = true;
                }
            } else {
                in_air = false;
            }
            ysum += tel.com[1] as f64;
            y2sum += (tel.com[1] as f64).powi(2);
            samples += 1;
        }
        m.heading_err_end = (cmd.heading - tel.yaw).abs();

        if let Some(rec) = rec.as_mut() {
            let b = r
                .links
                .iter()
                .map(|l| {
                    let body = &w.bodies[l.body];
                    let p = body.translation();
                    let q = body.rotation();
                    [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
                })
                .collect();
            rec.frames.push(Frame {
                t,
                b,
                speed: tel.fwd_speed,
                cmd_speed: cmd.speed,
                heading: tel.yaw,
                cmd_heading: cmd.heading,
                feet: tel.feet_in_contact,
                label: if t < label_until { label.clone() } else if fallen { "FALL".into() } else { String::new() },
            });
        }

        if fallen && (sc.reset_on_fall || m.first_fall_t.is_none()) {
            m.falls += 1;
        }
        if fallen {
            m.first_fall_t.get_or_insert(t);
            if sc.reset_on_fall {
                let (nw, nr) = build_world(sc);
                w = nw;
                r = nr;
                ctl = mk();
            } else if rec.is_none() || t > m.first_fall_t.unwrap() + 2.0 {
                m.sim_seconds = t;
                break;
            }
        }
        m.sim_seconds = t + DT;
    }
    m.steps_taken = ctl.steps();
    if samples > 0 {
        let s = samples as f32;
        m.mean_speed = speed_sum / s;
        m.mean_cmd_speed = cmd_sum / s;
        m.speed_err = err_sum / s;
        m.mean_stance_slip = slip_sum / s;
        m.flight_fraction = flight as f32 / s;
        m.flight_phases = phases;
        m.mean_flight_ms = if phases > 0 { air_frames as f32 * DT * 1000.0 / phases as f32 } else { 0.0 };
        let mean = ysum / samples as f64;
        m.com_y_std = ((y2sum / samples as f64 - mean * mean).max(0.0)).sqrt() as f32;
    }
    m.wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    (m, rec)
}

// ---------------- scenario library ----------------

pub fn cmd(t: f32, speed: f32, heading: f32) -> (f32, Command) {
    (t, Command { speed, heading })
}

pub fn scenario(name: &str, assist: f32) -> Scenario {
    let base = Scenario {
        name: name.to_string(),
        duration: 20.0,
        commands: vec![],
        pushes: vec![],
        platform: None,
        assist,
        record: false,
        reset_on_fall: false,
        spec: RobotSpec::default(),
    };
    match name {
        "stand" => Scenario { duration: 10.0, commands: vec![cmd(0.0, -1.0, 0.0)], ..base },
        "walk" => Scenario { duration: 14.0, commands: vec![cmd(0.0, 1.2, 0.0)], ..base },
        "walk_slow" => Scenario { duration: 12.0, commands: vec![cmd(0.0, 0.7, 0.0)], ..base },
        "walk_fast" => Scenario { duration: 12.0, commands: vec![cmd(0.0, 1.0, 0.0), cmd(3.0, 2.0, 0.0)], ..base },
        "turn" => Scenario {
            duration: 16.0,
            commands: vec![cmd(0.0, 1.0, 0.0), cmd(4.0, 1.0, 0.8), cmd(9.0, 1.0, -0.2)],
            ..base
        },
        "push_tune" => Scenario {
            duration: 14.0,
            commands: vec![cmd(0.0, 1.2, 0.0)],
            pushes: vec![
                Push { t: 5.0, dv_fwd: 0.0, dv_left: 0.25 },
                Push { t: 9.0, dv_fwd: 0.0, dv_left: -0.25 },
            ],
            ..base
        },
        "showcase" => Scenario {
            duration: 32.0,
            commands: vec![
                cmd(0.0, 0.8, 0.0),
                cmd(4.0, 1.4, 0.0),
                cmd(9.0, 1.2, 0.9),
                cmd(15.0, 1.2, 0.2),
                cmd(19.0, 2.0, 0.2),
                cmd(25.0, 0.8, 0.2),
            ],
            pushes: vec![Push { t: 27.0, dv_fwd: 0.0, dv_left: 0.3 }],
            ..base
        },
        "stepdown" => Scenario {
            duration: 14.0,
            commands: vec![cmd(0.0, 1.0, 0.0)],
            platform: Some((0.10, 4.0)),
            ..base
        },
        "pushes" => Scenario {
            duration: 22.0,
            commands: vec![cmd(0.0, 1.2, 0.0)],
            pushes: vec![
                Push { t: 5.0, dv_fwd: 0.0, dv_left: 0.3 },
                Push { t: 10.0, dv_fwd: 0.0, dv_left: -0.3 },
                Push { t: 14.0, dv_fwd: 0.4, dv_left: 0.0 },
                Push { t: 18.0, dv_fwd: -0.4, dv_left: 0.0 },
            ],
            ..base
        },
        "bigpush" => Scenario {
            duration: 12.0,
            commands: vec![cmd(0.0, 1.2, 0.0)],
            pushes: vec![Push { t: 5.0, dv_fwd: 0.0, dv_left: 2.0 }],
            ..base
        },
        "run_mid" => Scenario { duration: 12.0, commands: vec![cmd(0.0, 1.0, 0.0), cmd(2.0, 2.2, 0.0)], ..base },
        "run_fast" => Scenario { duration: 14.0, commands: vec![cmd(0.0, 1.5, 0.0), cmd(2.0, 3.0, 0.0)], ..base },
        "run_show" => Scenario {
            duration: 22.0,
            commands: vec![cmd(0.0, 1.0, 0.0), cmd(3.0, 2.2, 0.0), cmd(8.0, 3.0, 0.0), cmd(13.0, 3.0, 0.6), cmd(18.0, 1.5, 0.6)],
            pushes: vec![Push { t: 16.0, dv_fwd: 0.0, dv_left: 0.3 }],
            ..base
        },
        "run" => Scenario {
            duration: 16.0,
            commands: vec![cmd(0.0, 1.2, 0.0), cmd(3.0, 2.2, 0.0), cmd(7.0, 3.0, 0.0), cmd(11.0, 3.6, 0.0)],
            ..base
        },
        _ => panic!("unknown scenario {name}"),
    }
}
