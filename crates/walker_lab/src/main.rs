//! walker_lab — headless biped walker prototype.
//!
//! Usage:
//!   walker_lab run <scenario> [assist] [out.json]   run one scenario, print metrics, optionally record
//!   walker_lab tune <iters> [assist]                 random-search gait parameters -> out/gait[_a{assist}].json
//!   walker_lab pushtest [assist]                     largest survivable side/forward push
//!   walker_lab soak <seconds> [assist]               long run with random commands and pushes
//!   walker_lab report                                all of the above into out/report.json

mod controller;
mod policy;
mod ppo;
mod robot;
mod robot_v;
mod thrust;
mod run_ctl;
mod sim;

use controller::Gait;
use sim::*;
use std::fs;

fn gait_path(assist: f32) -> String {
    format!("out/gait_a{assist}.json")
}

fn load_gait(assist: f32) -> Gait {
    fs::read_to_string(gait_path(assist))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn n(&mut self) -> f32 {
        // approx normal
        (0..4).map(|_| self.f()).sum::<f32>() - 2.0
    }
}

const TUNE_SET: [&str; 5] = ["walk_slow", "walk", "walk_fast", "turn", "push_tune"];

fn score(g: Gait, assist: f32) -> (f32, Vec<Metrics>) {
    let mut total = 0.0;
    let mut ms = Vec::new();
    for name in TUNE_SET {
        let sc = scenario(name, assist);
        let (m, _) = run(&sc, g);
        let survive = m.sim_seconds / sc.duration;
        let mut s = 100.0 * survive;
        if m.falls == 0 {
            s += 50.0 - 30.0 * m.speed_err - 10.0 * m.mean_stance_slip;
            if name == "turn" {
                s -= 20.0 * m.heading_err_end;
            }
        }
        if m.non_finite {
            s -= 200.0;
        }
        total += s;
        ms.push(m);
    }
    (total, ms)
}

fn perturb(g: &Gait, rng: &mut Rng, sigma: f32) -> Gait {
    let mut v: [f32; 19] = [
        g.t_step, g.sw_hip0, g.cd, g.cv, g.kv, g.cdl, g.cvl, g.width, g.sw_knee1, g.sw_knee2, g.st_knee, g.lean, g.lean_v,
        g.kp, g.kd, g.ca_d, g.ca_v, g.ca_walk, g.touchdown,
    ];
    let scale: [f32; 19] = [0.05, 0.08, 0.1, 0.06, 0.04, 0.1, 0.06, 0.03, 0.12, 0.04, 0.04, 0.03, 0.02, 0.12, 0.015, 0.08, 0.04, 0.05, 0.05];
    for i in 0..v.len() {
        v[i] += rng.n() * scale[i] * sigma;
    }
    Gait {
        t_step: v[0].clamp(0.25, 0.8),
        sw_hip0: v[1].clamp(-1.0, 0.3),
        cd: v[2].clamp(0.0, 2.0),
        cv: v[3].clamp(0.0, 1.0),
        kv: v[4].clamp(-0.3, 0.5),
        cdl: v[5].clamp(0.0, 2.0),
        cvl: v[6].clamp(0.0, 1.0),
        width: v[7].clamp(-0.05, 0.2),
        sw_knee1: v[8].clamp(0.2, 1.8),
        sw_knee2: v[9].clamp(0.0, 0.6),
        st_knee: v[10].clamp(0.0, 0.5),
        lean: v[11].clamp(-0.15, 0.25),
        lean_v: v[12].clamp(-0.05, 0.15),
        kp: v[13].clamp(0.3, 3.0),
        kd: v[14].clamp(0.03, 0.4),
        ca_d: v[15].clamp(0.0, 1.5),
        ca_v: v[16].clamp(0.0, 0.6),
        ca_walk: v[17].clamp(0.0, 0.6),
        touchdown: v[18].clamp(0.35, 0.95),
    }
}

/// (1+λ) random search with step-size adaptation, evaluated on 4 threads.
fn tune(iters: usize, assist: f32) {
    fs::create_dir_all("out").ok();
    let mut best = load_gait(assist);
    let (mut best_s, _) = score(best, assist);
    println!("start score {best_s:.1}");
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut sigma = 1.0;
    for it in 0..iters {
        let cands: Vec<Gait> = (0..4).map(|_| perturb(&best, &mut rng, sigma)).collect();
        let results: Vec<(f32, Gait)> = std::thread::scope(|s| {
            let hs: Vec<_> = cands.iter().map(|g| s.spawn(move || (score(*g, assist).0, *g))).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let (s, g) = results.into_iter().fold((f32::MIN, best), |a, b| if b.0 > a.0 { b } else { a });
        if s > best_s {
            best_s = s;
            best = g;
            sigma = (sigma * 1.15).min(2.0);
            fs::write(gait_path(assist), serde_json::to_string_pretty(&best).unwrap()).unwrap();
            println!("iter {it}: score {best_s:.1}  {:?}", best);
        } else {
            sigma = (sigma * 0.97).max(0.15);
        }
    }
    let (_, ms) = score(best, assist);
    for m in ms {
        println!("{}", serde_json::to_string(&m).unwrap());
    }
}

/// Survival must hold for pushes landing at four different points of the gait cycle,
/// so the reported limit is the one the robot handles regardless of step phase.
fn survives(name: &str, dv_fwd: f32, dv_left: f32, g: Gait, assist: f32) -> bool {
    [5.0f32, 5.2, 5.4, 5.6].iter().all(|&t| {
        let mut sc = scenario("walk", assist);
        sc.name = name.into();
        sc.duration = 10.0;
        sc.pushes = vec![Push { t, dv_fwd, dv_left }];
        run(&sc, g).0.falls == 0
    })
}

fn pushtest(assist: f32) -> serde_json::Value {
    let g = load_gait(assist);
    let search = |f: &dyn Fn(f32) -> bool| {
        if !f(0.0) {
            return 0.0;
        }
        let (mut lo, mut hi) = (0.0f32, 5.0f32);
        for _ in 0..10 {
            let mid = 0.5 * (lo + hi);
            if f(mid) { lo = mid } else { hi = mid }
        }
        lo
    };
    let mass = {
        let sc = scenario("walk", assist);
        let mut w = rapier3d::prelude::PhysicsWorld::new();
        let _ = &sc;
        robot::build(&mut w, rapier3d::prelude::Vector::ZERO, robot::RobotSpec::default()).total_mass
    };
    let side = search(&|dv| survives("push_side", 0.0, dv, g, assist));
    let fwd = search(&|dv| survives("push_fwd", dv, 0.0, g, assist));
    let back = search(&|dv| survives("push_back", -dv, 0.0, g, assist));
    serde_json::json!({
        "assist": assist,
        "robot_mass_kg": mass,
        "max_side_push_dv_mps": side, "max_side_push_Ns": side * mass,
        "max_forward_push_dv_mps": fwd, "max_forward_push_Ns": fwd * mass,
        "max_backward_push_dv_mps": back, "max_backward_push_Ns": back * mass,
        "note": "worst case over 4 push timings across the step cycle; push applied to the trunk while walking at 1.2 m/s; dv is the velocity change it would give the whole robot"
    })
}

fn soak(seconds: f32, assist: f32) -> Metrics {
    let g = load_gait(assist);
    let mut rng = Rng(42);
    let mut sc = scenario("walk", assist);
    sc.name = "soak".into();
    sc.duration = seconds;
    sc.reset_on_fall = true;
    sc.commands.clear();
    let mut t = 0.0;
    let mut heading = 0.0;
    while t < seconds {
        heading += (rng.f() - 0.5) * 1.2;
        sc.commands.push(cmd(t, 0.6 + rng.f() * 1.4, heading));
        if rng.f() < 0.5 {
            let a = rng.f() * std::f32::consts::TAU;
            sc.pushes.push(Push { t: t + 3.0, dv_fwd: 0.15 * a.cos(), dv_left: 0.15 * a.sin() });
        }
        t += 6.0;
    }
    run(&sc, g).0
}

// ---------------- running experiments ----------------

fn build_spec(name: &str) -> robot::RobotSpec {
    use robot::{LegKind, RobotSpec};
    match name {
        "A" => RobotSpec::default(),
        "B" => RobotSpec { foot_scale: 1.3, swing_arms: true, ..Default::default() },
        "C" => RobotSpec { legs: LegKind::Digitigrade, foot_scale: 1.3, swing_arms: true, reaction_wheels: false, ..Default::default() },
        "D" => RobotSpec { legs: LegKind::Digitigrade, foot_scale: 1.3, swing_arms: true, reaction_wheels: true, ..Default::default() },
        // Deer legs on hooves, 3-joint arms, reaction wheels.
        "E" => RobotSpec { legs: LegKind::Deer, foot_scale: 1.0, swing_arms: true, reaction_wheels: true, arm_dof: 3, ..Default::default() },
        // E on the mech-proportioned frame.
        "F" => RobotSpec { legs: LegKind::Deer, foot_scale: 1.0, swing_arms: true, reaction_wheels: true, arm_dof: 3, frame: robot::Frame::Mech, ..Default::default() },
        // Mech-V frame with swappable loads; strong legs (1.3× motors for its weight), light top.
        "GL" | "GM" | "GH" => RobotSpec {
            legs: LegKind::Deer, foot_scale: 1.0, swing_arms: true, reaction_wheels: true, arm_dof: 3,
            frame: robot::Frame::MechV, strength: 1.3,
            load: match name { "GL" => robot::Load::Light, "GM" => robot::Load::Medium, _ => robot::Load::Heavy },
            thrusters: false,
        },
        // Same builds with the 9-thruster set.
        "GLT" | "GMT" | "GHT" => RobotSpec {
            legs: LegKind::Deer, foot_scale: 1.0, swing_arms: true, reaction_wheels: true, arm_dof: 3,
            frame: robot::Frame::MechV, strength: 1.3,
            load: match name { "GLT" => robot::Load::Light, "GMT" => robot::Load::Medium, _ => robot::Load::Heavy },
            thrusters: true,
        },
        _ => panic!("unknown build {name}"),
    }
}

fn run_gait_path(build: &str, assist: f32) -> String {
    format!("out/run_{build}_a{assist}.json")
}

fn load_run_gait(build: &str, assist: f32) -> run_ctl::RunGait {
    // Thruster builds (GLT/GMT/GHT) share their base build's gait.
    let base = build.strip_suffix('T').filter(|b| b.starts_with('G')).unwrap_or(build);
    fs::read_to_string(run_gait_path(build, assist))
        .or_else(|_| fs::read_to_string(run_gait_path(base, assist)))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn run_scenario(build: &str, name: &str, assist: f32, g: run_ctl::RunGait) -> (Metrics, Option<Recording>) {
    let mut sc = scenario(name, assist);
    sc.spec = build_spec(build);
    sc.name = format!("{name} · build {build}");
    run_with(&sc, &move || Box::new(run_ctl::RunController::new(g, assist)) as Box<dyn controller::Ctl>)
}

const RUN_SET: [&str; 3] = ["run_mid", "run_fast", "run_show"];

fn run_score(build: &str, g: run_ctl::RunGait, assist: f32) -> (f32, Vec<Metrics>) {
    let mut total = 0.0;
    let mut ms = Vec::new();
    for name in RUN_SET {
        let (m, _) = run_scenario(build, name, assist, g);
        let dur = scenario(name, assist).duration;
        let survive = m.sim_seconds / dur;
        let mut s = 100.0 * survive + 100.0 * m.flight_fraction.min(0.35) * survive;
        if m.falls == 0 {
            s += 150.0 * m.flight_fraction.min(0.35) + 50.0 - 25.0 * m.speed_err - 5.0 * m.mean_stance_slip;
            if name == "run_show" {
                s -= 30.0 * m.heading_err_end;
            }
            if build == "E" || build == "F" || build.starts_with('G') {
                // Style: real push-off, deer-like high foot lift, upright torso. Only counts when the run
                // actually tracks the command, so the tuner can't trade speed for style.
                let on_task = (1.0 - m.speed_err / 0.5).clamp(0.0, 1.0) * (1.0 - m.heading_err_end / 0.5).clamp(0.0, 1.0);
                s += on_task * (40.0 * (m.pushoff_deg / 20.0).clamp(0.0, 1.0) + 40.0 * (m.swing_clearance_m / 0.30).clamp(0.0, 1.0));
                // Bigger strides: reward step length up to 1.8 m (≈0.47 × robot height).
                s += on_task * 50.0 * (m.step_len_m / 1.8).clamp(0.0, 1.0);
                s -= 2.0 * (m.mean_tilt_deg - 10.0).max(0.0);
                s -= 40.0 * m.speed_err + 20.0 * m.heading_err_end;
            }
        }
        if m.non_finite {
            s -= 200.0;
        }
        total += s;
        ms.push(m);
    }
    // G builds: also score robustness on two random-command runs (fixed seeds).
    if build.starts_with('G') {
        for seed in [11u64, 12] {
            let sc = random_run_scenario(seed, build, assist, 15.0);
            let (m, _) = run_with(&sc, &move || Box::new(run_ctl::RunController::new(g, assist)) as Box<dyn controller::Ctl>);
            let survive = m.sim_seconds / sc.duration;
            let mut s = 100.0 * survive;
            if m.falls == 0 {
                s += 50.0 - 25.0 * m.speed_err;
            }
            total += s;
            ms.push(m);
        }
    }
    (total, ms)
}

fn run_tune(build: &str, iters: usize, assist: f32) {
    fs::create_dir_all("out").ok();
    let mut best = load_run_gait(build, assist);
    let (mut best_s, _) = run_score(build, best, assist);
    println!("[{build}] start score {best_s:.1}");
    let mut rng = Rng(0xA5A5_1234_9E37_79B9 ^ build.as_bytes()[0] as u64);
    let mut sigma = 1.5;
    for it in 0..iters {
        let cands: Vec<run_ctl::RunGait> = (0..4)
            .map(|_| {
                let mut v = best.to_vec();
                for (i, x) in v.iter_mut().enumerate() {
                    *x += rng.n() * run_ctl::RUN_BOUNDS[i].2 * sigma;
                }
                run_ctl::RunGait::from_vec(&v)
            })
            .collect();
        let results: Vec<(f32, run_ctl::RunGait)> = std::thread::scope(|s| {
            let hs: Vec<_> = cands.iter().map(|g| s.spawn(move || (run_score(build, *g, assist).0, *g))).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let (s, g) = results.into_iter().fold((f32::MIN, best), |a, b| if b.0 > a.0 { b } else { a });
        if s > best_s {
            best_s = s;
            best = g;
            sigma = (sigma * 1.15).min(2.5);
            fs::write(run_gait_path(build, assist), serde_json::to_string_pretty(&best).unwrap()).unwrap();
            println!("[{build}] iter {it}: score {best_s:.1}");
        } else {
            sigma = (sigma * 0.97).max(0.2);
        }
    }
    for m in run_score(build, best, assist).1 {
        println!("{}", serde_json::to_string(&m).unwrap());
    }
}

fn run_survives(build: &str, g: run_ctl::RunGait, assist: f32, dv_left: f32) -> bool {
    [6.0f32, 6.1, 6.2, 6.3].iter().all(|&t| {
        let mut sc = scenario("run_mid", assist);
        sc.spec = build_spec(build);
        sc.pushes = vec![Push { t, dv_fwd: 0.0, dv_left }];
        run_with(&sc, &move || Box::new(run_ctl::RunController::new(g, assist)) as Box<dyn controller::Ctl>).0.falls == 0
    })
}

fn run_report() {
    let mut out = Vec::new();
    for (b, a) in [("A", 0.35f32), ("B", 0.35), ("C", 0.35), ("D", 0.35), ("C", 0.0), ("D", 0.0)] {
        let g = load_run_gait(b, a);
        let mut runs = Vec::new();
        for name in ["run_mid", "run_fast", "run_show"] {
            runs.push(serde_json::to_value(run_scenario(b, name, a, g).0).unwrap());
        }
        let side = {
            let (mut lo, mut hi) = (0.0f32, 3.0f32);
            if run_survives(b, g, a, 0.0) {
                for _ in 0..8 {
                    let mid = 0.5 * (lo + hi);
                    if run_survives(b, g, a, mid) { lo = mid } else { hi = mid }
                }
                lo
            } else {
                0.0
            }
        };
        // Soak: 300 s of random speeds 1.5–3.2 m/s, heading changes and pushes; auto-reset on falls.
        let mut rng = Rng(7);
        let mut sc = scenario("run_mid", a);
        sc.spec = build_spec(b);
        sc.name = "soak".into();
        sc.duration = 300.0;
        sc.reset_on_fall = true;
        sc.commands.clear();
        let (mut t, mut heading) = (0.0, 0.0);
        while t < 300.0 {
            heading += (rng.f() - 0.5) * 1.0;
            sc.commands.push(cmd(t, 1.5 + rng.f() * 1.7, heading));
            if rng.f() < 0.5 {
                let ang = rng.f() * std::f32::consts::TAU;
                sc.pushes.push(Push { t: t + 3.0, dv_fwd: 0.15 * ang.cos(), dv_left: 0.15 * ang.sin() });
            }
            t += 6.0;
        }
        let soak = run_with(&sc, &move || Box::new(run_ctl::RunController::new(g, a)) as Box<dyn controller::Ctl>).0;
        let mass = robot::build(&mut rapier3d::prelude::PhysicsWorld::new(), rapier3d::prelude::Vector::ZERO, build_spec(b)).total_mass;
        out.push(serde_json::json!({ "build": b, "spec": build_spec(b), "assist": a, "mass_kg": mass, "gait": g,
            "scenarios": runs, "max_side_push_dv_mps": side, "soak_300s": soak }));
        eprintln!("done {b} a{a}");
    }
    let s = serde_json::to_string_pretty(&out).unwrap();
    fs::write("out/run_report.json", &s).unwrap();
}

// ---------------- distillation (behaviour cloning + DAgger) ----------------

fn random_run_scenario(seed: u64, build: &str, assist: f32, dur: f32) -> Scenario {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut sc = scenario("run_mid", assist);
    sc.spec = build_spec(build);
    sc.duration = dur;
    sc.commands.clear();
    let (mut t, mut heading) = (0.0, 0.0);
    while t < dur {
        heading += (rng.f() - 0.5) * 0.8;
        sc.commands.push(cmd(t, 1.2 + rng.f() * 2.0, heading));
        if rng.f() < 0.4 {
            let a = rng.f() * std::f32::consts::TAU;
            sc.pushes.push(Push { t: t + 2.5, dv_fwd: 0.15 * a.cos(), dv_left: 0.15 * a.sin() });
        }
        t += 5.0;
    }
    sc
}

type Data = std::sync::Arc<std::sync::Mutex<Vec<(Vec<f32>, Vec<f32>)>>>;

fn collect(build: &str, assist: f32, g: run_ctl::RunGait, student: Option<std::sync::Arc<policy::Mlp>>, episodes: usize, seed0: u64) -> (Vec<(Vec<f32>, Vec<f32>)>, f32) {
    let out: Data = Default::default();
    let survived = std::sync::Mutex::new(0.0f32);
    std::thread::scope(|s| {
        for t in 0..4 {
            let (out, student, survived) = (out.clone(), student.clone(), &survived);
            s.spawn(move || {
                for e in (t..episodes).step_by(4) {
                    let sc = random_run_scenario(seed0 + e as u64, build, assist, 20.0);
                    let (o, st) = (out.clone(), student.clone());
                    let (m, _) = run_with(&sc, &move || {
                        Box::new(policy::Recorder {
                            teacher: run_ctl::RunController::new(g, assist),
                            student: st.clone(),
                            tracker: Default::default(),
                            data: o.clone(),
                        }) as Box<dyn controller::Ctl>
                    });
                    *survived.lock().unwrap() += m.sim_seconds;
                }
            });
        }
    });
    let d = std::mem::take(&mut *out.lock().unwrap());
    (d, survived.into_inner().unwrap() / episodes as f32)
}

fn eval_student(build: &str, assist: f32, g: run_ctl::RunGait, net: std::sync::Arc<policy::Mlp>, name: &str, clip: Option<(&str, f32)>) -> Metrics {
    let mut sc = scenario("run_show", assist);
    sc.spec = build_spec(build);
    sc.name = name.into();
    sc.record = clip.is_some();
    let (m, rec) = run_with(&sc, &move || {
        Box::new(policy::StudentController { net: net.clone(), tracker: Default::default(), gait: g, assist, steps: 0 }) as Box<dyn controller::Ctl>
    });
    if let (Some((path, len)), Some(mut rec)) = (clip, rec) {
        // Short clip: the 3 m/s stretch if it survives, else the seconds leading into the fall.
        let end = match m.first_fall_t { Some(t) => t + 1.5, None => 13.0 };
        let start = (end - len).max(0.0);
        rec.frames.retain(|f| f.t >= start && f.t <= end);
        fs::write(path, serde_json::to_string(&rec).unwrap()).unwrap();
    }
    m
}

fn distill(build: &str, assist: f32, iters: usize) {
    let g = load_run_gait(build, assist);
    fs::create_dir_all(format!("out/distill_{build}")).ok();
    // Teacher clip for reference.
    {
        let mut sc = scenario("run_show", assist);
        sc.spec = build_spec(build);
        sc.name = "TEACHER · hand-built controller".into();
        sc.record = true;
        let (_, rec) = run_with(&sc, &move || Box::new(run_ctl::RunController::new(g, assist)) as Box<dyn controller::Ctl>);
        let mut rec = rec.unwrap();
        rec.frames.retain(|f| f.t >= 7.0 && f.t <= 13.0);
        fs::write(format!("out/distill_{build}/clip_teacher.json"), serde_json::to_string(&rec).unwrap()).unwrap();
    }
    let (mut data, teacher_surv) = collect(build, assist, g, None, 24, 1000);
    println!("teacher data: {} samples, mean survival {:.1}/20 s", data.len(), teacher_surv);
    let mut net = policy::Mlp::new(&[policy::OBS_DIM, 192, 192, run_ctl::ACTION_DIM], 7);
    {
        let xs: Vec<Vec<f32>> = data.iter().map(|d| d.0.clone()).collect();
        let ys: Vec<Vec<f32>> = data.iter().map(|d| d.1.clone()).collect();
        net.fit_normalisation(&xs, &ys);
    }
    let mut opt = net.adam();
    let mut log = Vec::new();
    for it in 0..=iters {
        let xs: Vec<Vec<f32>> = data.iter().map(|d| d.0.clone()).collect();
        let ys: Vec<Vec<f32>> = data.iter().map(|d| d.1.clone()).collect();
        let epochs = if it == 0 { 12 } else { 6 };
        let mut loss = 0.0;
        let t0 = std::time::Instant::now();
        for e in 0..epochs {
            loss = net.train_epoch(&mut opt, &xs, &ys, 512, 1e-3, (it * 100 + e) as u64 + 1);
        }
        let train_s = t0.elapsed().as_secs_f32();
        let arc = std::sync::Arc::new(net.clone());
        let m = eval_student(build, assist, g, arc.clone(), &format!("STUDENT · DAgger round {it}"),
            Some((&format!("out/distill_{build}/clip_round{it}.json"), 6.0)));
        println!("round {it}: {} samples, loss {:.4}, train {:.0}s | run_show: survived {:.1}/22 s, speed {:.2}, airborne {:.2}",
            data.len(), loss, train_s, m.sim_seconds, m.mean_speed, m.flight_fraction);
        log.push(serde_json::json!({"round": it, "samples": data.len(), "loss": loss, "eval": m}));
        fs::write(format!("out/distill_{build}/net_round{it}.json"), serde_json::to_string(&net).unwrap()).unwrap();
        fs::write(format!("out/distill_{build}/log.json"), serde_json::to_string_pretty(&log).unwrap()).unwrap();
        if it < iters {
            let (more, surv) = collect(build, assist, g, Some(arc), 24, 5000 + it as u64 * 100);
            println!("   student rollouts: +{} samples, mean survival {:.1}/20 s", more.len(), surv);
            data.extend(more);
        }
    }
}

/// Fixed evaluation set: the turn-and-push run plus 9 random 20 s runs (seeds disjoint from training).
fn eval_set(build: &str, assist: f32, mk: &(dyn Fn() -> Box<dyn controller::Ctl> + Sync)) -> serde_json::Value {
    let mut scs = vec![{ let mut s = scenario("run_show", assist); s.spec = build_spec(build); s }];
    for k in 0..9 {
        scs.push(random_run_scenario(90_000 + k, build, assist, 20.0));
    }
    let ms: Vec<Metrics> = std::thread::scope(|s| {
        let hs: Vec<_> = scs.iter().map(|sc| s.spawn(move || run_with(sc, mk).0)).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let total: f32 = scs.iter().map(|s| s.duration).sum();
    let surv: f32 = ms.iter().map(|m| m.sim_seconds).sum();
    let falls = ms.iter().filter(|m| m.falls > 0).count();
    let speed_err: f32 = ms.iter().map(|m| m.speed_err).sum::<f32>() / ms.len() as f32;
    let air: f32 = ms.iter().map(|m| m.flight_fraction).sum::<f32>() / ms.len() as f32;
    let stride: f32 = ms.iter().map(|m| m.step_len_m).sum::<f32>() / ms.len() as f32;
    let push: f32 = ms.iter().map(|m| m.pushoff_deg).sum::<f32>() / ms.len() as f32;
    let tilt: f32 = ms.iter().map(|m| m.mean_tilt_deg).sum::<f32>() / ms.len() as f32;
    serde_json::json!({"runs": ms.len(), "falls": falls, "survival_frac": surv / total, "mean_speed_err": speed_err, "mean_airborne": air,
        "mean_step_m": stride, "mean_pushoff_deg": push, "mean_tilt_deg": tilt})
}

fn eval_distill(build: &str, assist: f32) {
    let g = load_run_gait(build, assist);
    let mut out = serde_json::Map::new();
    out.insert("teacher".into(), eval_set(build, assist, &move || Box::new(run_ctl::RunController::new(g, assist)) as Box<dyn controller::Ctl>));
    for r in 0.. {
        let Ok(s) = fs::read_to_string(format!("out/distill_{build}/net_round{r}.json")) else { break };
        let net = std::sync::Arc::new(serde_json::from_str::<policy::Mlp>(&s).unwrap());
        out.insert(format!("round{r}"), eval_set(build, assist, &move || {
            Box::new(policy::StudentController { net: net.clone(), tracker: Default::default(), gait: g, assist, steps: 0 }) as Box<dyn controller::Ctl>
        }));
    }
    let s = serde_json::to_string_pretty(&out).unwrap();
    fs::write(format!("out/distill_{build}/eval.json"), &s).unwrap();
    for (k, v) in &out {
        println!("{k:8} falls {}/{}  survival {:.0}%  speed err {:.2}  airborne {:.0}%", v["falls"], v["runs"],
            v["survival_frac"].as_f64().unwrap() * 100.0, v["mean_speed_err"].as_f64().unwrap(), v["mean_airborne"].as_f64().unwrap() * 100.0);
    }
}

/// Fixed-speed test: hold each commanded speed for 12 s (after a short ramp) with the student policy.
fn eval_speeds(build: &str, assist: f32, gait: run_ctl::RunGait, net: std::sync::Arc<policy::Mlp>) -> Vec<serde_json::Value> {
    let speeds = [3.0f32, 4.0, 5.0, 6.0, 7.0];
    std::thread::scope(|s| {
        let hs: Vec<_> = speeds
            .iter()
            .map(|&v| {
                let net = net.clone();
                s.spawn(move || {
                    let mut sc = scenario("run_mid", assist);
                    sc.spec = build_spec(build);
                    sc.duration = 12.0;
                    sc.commands = vec![cmd(0.0, 1.5, 0.0), cmd(1.0, 0.5 * (1.5 + v), 0.0), cmd(2.5, v, 0.0)];
                    let (m, _) = run_with(&sc, &move || {
                        Box::new(policy::StudentController { net: net.clone(), tracker: Default::default(), gait, assist, steps: 0 }) as Box<dyn controller::Ctl>
                    });
                    serde_json::json!({"cmd": v, "fell": m.falls > 0, "survived_s": m.sim_seconds, "mean_speed": m.mean_speed,
                        "step_m": m.step_len_m, "cadence_hz": m.cadence_hz, "pushoff_deg": m.pushoff_deg, "airborne": m.flight_fraction})
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

fn ppo_train(build: &str, iters: usize, init: &str) {
    ppo_train_mode(build, iters, init, false, None, 0, false)
}

/// Deterministic evaluation episode through the PPO environment (supports knockdowns), optionally recorded.
fn run_env(build: &str, gait: run_ctl::RunGait, net: &policy::Mlp, sc: Scenario, mode: ppo::EpMode, knock_dir: f32, label: &str, rec: bool)
    -> (ppo::Env, Vec<(f32, f32, f32)>, bool) {
    let cfg = ppo::RewardCfg { stride_target: 1.8, thrust_k: 1.0, vmax: 7.0, speed_mode: true, omni: true, knock_dv: 3.0 };
    let dur = sc.duration;
    let mut env = ppo::Env::with(build_spec(build), gait, sc, mode, knock_dir);
    env.label = label.to_string();
    if rec {
        env.frames = Some(Vec::new());
    }
    let mut vel = Vec::new();
    let mut fell = false;
    while env.t() < dur {
        let o = env.obs();
        let a = net.forward_norm(&o);
        let (_, term, trunc) = env.step(net, &a, cfg);
        vel.push((env.t(), env.velocity().0, env.velocity().1));
        if term {
            fell = true;
            break;
        }
        if trunc {
            break;
        }
    }
    (env, vel, fell)
}

/// Omni + get-up evaluation: backward jog, sidesteps, forward speed, and 4 knockdowns.
fn eval_omni(build: &str, gait: run_ctl::RunGait, net: &policy::Mlp, dir: &str, it: usize, clips: bool) -> serde_json::Value {
    let mk = |cmds: Vec<(f32, controller::Command)>, dur: f32| {
        let mut sc = scenario("run_mid", 0.0);
        sc.spec = build_spec(build);
        sc.duration = dur;
        sc.commands = cmds;
        sc
    };
    let tests: Vec<(&str, f32, f32, Scenario)> = vec![
        ("back", -1.5, 0.0, mk(vec![cmd(0.0, 1.0, 0.0), cmd_l(1.5, -1.5, 0.0, 0.0)], 8.0)),
        ("left", 0.0, 1.0, mk(vec![cmd(0.0, 1.0, 0.0), cmd_l(1.5, 0.0, 1.0, 0.0)], 8.0)),
        ("right", 0.0, -1.0, mk(vec![cmd(0.0, 1.0, 0.0), cmd_l(1.5, 0.0, -1.0, 0.0)], 8.0)),
        ("fwd6", 6.0, 0.0, mk(vec![cmd(0.0, 1.5, 0.0), cmd(1.0, 3.75, 0.0), cmd(2.5, 6.0, 0.0)], 10.0)),
    ];
    let mut out = serde_json::Map::new();
    for (name, vf, vl, sc) in tests {
        let (_, vel, fell) = run_env(build, gait, net, sc, ppo::EpMode::Omni, 0.0, "", false);
        let tail: Vec<_> = vel.iter().filter(|v| v.0 > 4.0).collect();
        let n = tail.len().max(1) as f32;
        let (mf, ml) = (tail.iter().map(|v| v.1).sum::<f32>() / n, tail.iter().map(|v| v.2).sum::<f32>() / n);
        out.insert(name.into(), serde_json::json!({"cmd_fwd": vf, "cmd_left": vl, "fwd": mf, "left": ml, "fell": fell}));
    }
    let mut ups = Vec::new();
    for k in 0..4 {
        let mut sc = scenario("run_mid", 0.0);
        sc.spec = build_spec(build);
        sc.duration = ppo::GETUP_DUR;
        sc.commands = vec![cmd(0.0, 0.0, 0.0)];
        let (env, _, _) = run_env(build, gait, net, sc, ppo::EpMode::GetUp, k as f32 * 1.5708, "", false);
        ups.push(serde_json::json!({"stood_at_end": env.standing(), "first_stand_s": env.first_stand_t.map(|t| t - ppo::KNOCK_T)}));
    }
    out.insert("getup".into(), serde_json::Value::Array(ups));
    if build_spec(build).thrusters {
        // Thrust 2 s in each direction from a slow jog, then 3 s to recover; did it stay up, and
        // what was the peak speed along the thrust (height gained, for up)?
        let mut th = serde_json::Map::new();
        for (name, dir, dash) in [("L", [1.0, 0.0, 0.0], false), ("R", [-1.0, 0.0, 0.0], false), ("F", [0.0, 0.0, 1.0], false),
            ("B", [0.0, 0.0, -1.0], false), ("U", [0.0, 1.0, 0.0], false), ("dashF", [0.0, 0.0, 1.0], true), ("dashL", [1.0, 0.0, 0.0], true)] {
            let on = if dash { 0.4 } else { 2.0 };
            let sc = mk(vec![cmd(0.0, 1.0, 0.0), cmd_t(1.5, 1.0, 0.0, 0.0, dir, dash), cmd(1.5 + on, 1.0, 0.0)], 6.5);
            let (env, vel, fell) = run_env(build, gait, net, sc, ppo::EpMode::Thrust, 0.0, "", false);
            let peak = if dir[1] > 0.0 {
                env.max_height()
            } else {
                vel.iter().map(|v| v.1 * dir[2] + v.2 * dir[0]).fold(0.0f32, f32::max)
            };
            th.insert(name.into(), serde_json::json!({"peak": peak, "fell": fell}));
        }
        out.insert("thrust".into(), serde_json::Value::Object(th));
    }
    if clips {
        if build_spec(build).thrusters {
            let (cmds, labels) = thrust_script();
            let (mut env, _, _) = run_env(build, gait, net, mk(cmds, 13.0), ppo::EpMode::Thrust, 0.0, "", true);
            env.set_labels(&labels);
            save_clip(&env, &format!("{dir}/clip_thrust_iter{it:04}.json"), &format!("THRUSTERS · iter {it}"));
        }
        // Demo: forward → backwards → sidestep left → sidestep right, then a knockdown.
        let demo = mk(vec![cmd(0.0, 1.5, 0.0), cmd_l(3.0, -1.5, 0.0, 0.0), cmd_l(6.0, 0.0, 1.0, 0.0), cmd_l(9.0, 0.0, -1.0, 0.0)], 12.0);
        let (env, _, _) = run_env(build, gait, net, demo, ppo::EpMode::Omni, 0.0, "", true);
        save_clip(&env, &format!("{dir}/clip_omni_iter{it:04}.json"), &format!("OMNI · iter {it} · fwd → back → left → right"));
        let mut sc = scenario("run_mid", 0.0);
        sc.spec = build_spec(build);
        sc.duration = ppo::GETUP_DUR;
        sc.commands = vec![cmd(0.0, 0.0, 0.0)];
        let (env, _, _) = run_env(build, gait, net, sc, ppo::EpMode::GetUp, 0.0, "", true);
        save_clip(&env, &format!("{dir}/clip_getup_iter{it:04}.json"), &format!("GET UP · iter {it}"));
    }
    serde_json::Value::Object(out)
}

fn save_clip(env: &ppo::Env, path: &str, name: &str) {
    let rec = serde_json::json!({ "scenario": name, "assist": 0.0, "dt": sim::DT, "links": env.links(), "platform": null,
        "pushes": [], "frames": env.frames.as_ref().unwrap() });
    fs::write(path, serde_json::to_string(&rec).unwrap()).unwrap();
}

/// `vmax0` / `iter0`: resume a speed run at a given top speed and iteration number.
fn ppo_train_mode(build: &str, iters: usize, init: &str, speed_mode: bool, vmax0: Option<f32>, iter0: usize, omni: bool) {
    let assist = 0.0;
    let gait = load_run_gait(build, assist);
    let spec = build_spec(build);
    let dir = if omni { format!("out/ppoomni_{build}") } else if speed_mode { format!("out/ppospeed_{build}") } else { format!("out/ppo_{build}") };
    fs::create_dir_all(&dir).ok();
    let txt = fs::read_to_string(init).expect("init net");
    // Accept either a saved PPO policy (net + log-std) or a distilled network.
    let mut pol = match serde_json::from_str::<ppo::GaussPolicy>(&txt) {
        Ok(p) => p,
        Err(_) => {
            let net: policy::Mlp = serde_json::from_str(&txt).unwrap();
            ppo::GaussPolicy { log_std: vec![(0.08f32).ln(); run_ctl::ACTION_DIM], net }
        }
    };
    // Older networks have fewer inputs (no sideways command): widen with zero weights.
    pol.net.expand_inputs(policy::OBS_DIM);
    if omni {
        // New skills need some exploration again.
        for l in pol.log_std.iter_mut() {
            *l = l.max((0.12f32).ln());
        }
    }
    let mut value = policy::Mlp::new(&[policy::OBS_DIM, 128, 128, 1], 99);
    value.in_mean = pol.net.in_mean.clone();
    value.in_std = pol.net.in_std.clone();
    let mut popt = pol.net.adam();
    let mut vopt = value.adam();
    let mut lsopt = (vec![0.0; run_ctl::ACTION_DIM], vec![0.0; run_ctl::ACTION_DIM], 0);
    let mut cfg = ppo::RewardCfg { stride_target: 1.8, thrust_k: 1.0, vmax: 3.1, speed_mode, omni, knock_dv: 1.0 };
    if speed_mode {
        cfg.vmax = vmax0.unwrap_or(3.0);
    }
    if spec.thrusters {
        cfg.thrust_k = 0.3;
    }
    // Resuming: restore the curricula saved in the lightweight state file.
    if iter0 > 0 {
        if let Some(st) = fs::read_to_string(format!("{dir}/state.json")).ok().and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok()) {
            if let Some(k) = st["thrust_k"].as_f64() {
                cfg.thrust_k = k as f32;
            }
            if let Some(k) = st["knock_dv"].as_f64() {
                cfg.knock_dv = k as f32;
            }
        }
    }
    let (mut streak, mut vmax_log) = (0usize, vec![(iter0, cfg.vmax)]);
    let mut log = Vec::new();
    let checkpoint = |it: usize, pol: &ppo::GaussPolicy, log: &mut Vec<serde_json::Value>, extra: serde_json::Value| {
        let arc = std::sync::Arc::new(pol.net.clone());
        if omni {
            let ev = eval_omni(build, gait, &pol.net, &dir, it, it % 100 == 0);
            let g = &ev["getup"];
            let stood = (0..4).filter(|k| g[*k]["stood_at_end"].as_bool().unwrap_or(false)).count();
            let f = |k: &str| format!("{}{:+.1}/{:+.1}{}", k, ev[k]["fwd"].as_f64().unwrap(), ev[k]["left"].as_f64().unwrap(), if ev[k]["fell"].as_bool().unwrap() { "✗" } else { "" });
            println!("  omni {it}: {} | {} | {} | {} | get-up {stood}/4", f("back"), f("left"), f("right"), f("fwd6"));
            if let Some(th) = ev["thrust"].as_object() {
                let s: Vec<String> = th.iter().map(|(k, v)| format!("{k} {:.1}{}", v["peak"].as_f64().unwrap_or(0.0), if v["fell"].as_bool().unwrap_or(false) { "✗" } else { "✓" })).collect();
                println!("  thrust {it}: {}", s.join(" | "));
            }
            fs::write(format!("{dir}/policy_iter{it:04}.json"), serde_json::to_string(pol).unwrap()).unwrap();
            log.push(serde_json::json!({"iter": it, "omni": ev, "train": extra}));
            fs::write(format!("{dir}/log.json"), serde_json::to_string_pretty(&*log).unwrap()).unwrap();
            return;
        }
        if speed_mode {
            let sp = eval_speeds(build, assist, gait, arc.clone());
            let line: Vec<String> = sp.iter().map(|e| format!("{}→{:.1}{} step {:.2}m@{:.1}Hz push {:.0}°",
                e["cmd"], e["mean_speed"].as_f64().unwrap(), if e["fell"].as_bool().unwrap() { "✗" } else { "" },
                e["step_m"].as_f64().unwrap(), e["cadence_hz"].as_f64().unwrap(), e["pushoff_deg"].as_f64().unwrap())).collect();
            println!("  speeds {it}: {}", line.join(" | "));
            // Clip at the top trained speed.
            let vtop = extra["vmax"].as_f64().unwrap_or(3.0) as f32;
            let mut sc = scenario("run_mid", assist);
            sc.spec = build_spec(build);
            sc.duration = 12.0;
            sc.name = format!("SPEED RL · iteration {it} · top cmd {vtop:.1} m/s");
            sc.commands = vec![cmd(0.0, 1.5, 0.0), cmd(1.0, 0.5 * (1.5 + vtop), 0.0), cmd(2.5, vtop, 0.0)];
            sc.record = true;
            let a2 = arc.clone();
            let (m, rec) = run_with(&sc, &move || Box::new(policy::StudentController { net: a2.clone(), tracker: Default::default(), gait, assist, steps: 0 }) as Box<dyn controller::Ctl>);
            if let Some(mut rec) = rec {
                let end = m.first_fall_t.map(|t| t + 1.5).unwrap_or(12.0);
                rec.frames.retain(|f| f.t >= (end - 6.0).max(0.0) && f.t <= end);
                fs::write(format!("{dir}/clip_iter{it:04}.json"), serde_json::to_string(&rec).unwrap()).unwrap();
            }
            fs::write(format!("{dir}/policy_iter{it:04}.json"), serde_json::to_string(pol).unwrap()).unwrap();
            log.push(serde_json::json!({"iter": it, "speeds": sp, "train": extra}));
            fs::write(format!("{dir}/log.json"), serde_json::to_string_pretty(&*log).unwrap()).unwrap();
            return;
        }
        let show = eval_student(build, assist, gait, arc.clone(), &format!("PPO · iteration {it}"), Some((&format!("{dir}/clip_iter{it:03}.json"), 6.0)));
        let ev = eval_set(build, assist, &move || {
            Box::new(policy::StudentController { net: arc.clone(), tracker: Default::default(), gait, assist, steps: 0 }) as Box<dyn controller::Ctl>
        });
        println!("  ckpt {it}: eval falls {}/10 surv {:.0}% speed_err {:.2} stride {:.2} m pushoff {:.0}° tilt {:.0}° | run_show stride {:.2} speed {:.2}",
            ev["falls"], ev["survival_frac"].as_f64().unwrap() * 100.0, ev["mean_speed_err"].as_f64().unwrap(),
            ev["mean_step_m"].as_f64().unwrap(), ev["mean_pushoff_deg"].as_f64().unwrap(), ev["mean_tilt_deg"].as_f64().unwrap(),
            show.step_len_m, show.mean_speed);
        fs::write(format!("{dir}/policy_iter{it:03}.json"), serde_json::to_string(pol).unwrap()).unwrap();
        log.push(serde_json::json!({"iter": it, "eval": ev, "run_show": show, "train": extra}));
        fs::write(format!("{dir}/log.json"), serde_json::to_string_pretty(&*log).unwrap()).unwrap();
    };
    checkpoint(iter0, &pol, &mut log, serde_json::json!({"vmax": cfg.vmax}));
    let warmup = iter0 + 5;
    for it in iter0 + 1..=iter0 + iters {
        let t0 = std::time::Instant::now();
        let batches: Vec<ppo::Batch> = std::thread::scope(|s| {
            let (p, v) = (&pol, &value);
            let hs: Vec<_> = (0..4).map(|k| s.spawn(move || ppo::rollout(p, v, spec, gait, 6000, (it * 17 + k) as u64 * 1_000_003, cfg))).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut b = ppo::Batch { obs: vec![], act: vec![], logp: vec![], adv: vec![], ret: vec![], ep_returns: vec![], ep_lengths: vec![], ep_track: vec![], ep_cmd: vec![], ep_getup: vec![], ep_thrust: vec![] };
        for x in batches {
            b.obs.extend(x.obs);
            b.act.extend(x.act);
            b.logp.extend(x.logp);
            b.adv.extend(x.adv);
            b.ret.extend(x.ret);
            b.ep_returns.extend(x.ep_returns);
            b.ep_lengths.extend(x.ep_lengths);
            b.ep_track.extend(x.ep_track);
            b.ep_cmd.extend(x.ep_cmd);
            b.ep_getup.extend(x.ep_getup);
            b.ep_thrust.extend(x.ep_thrust);
        }
        let roll_s = t0.elapsed().as_secs_f32();
        let lr_pi = if it <= warmup { 0.0 } else { 1e-5 };
        let st = ppo::update(&mut pol, &mut value, &mut popt, &mut vopt, &mut lsopt, &b, lr_pi, 1e-3);
        let ne = b.ep_returns.len().max(1) as f32;
        let ret = b.ep_returns.iter().sum::<f32>() / ne;
        let len = b.ep_lengths.iter().sum::<f32>() / ne;
        let std_mean = pol.log_std.iter().map(|l| l.exp()).sum::<f32>() / pol.log_std.len() as f32;
        // Speed curriculum: raise the top commanded speed after 5 consecutive good iterations
        // (≥85% of episodes survive the full 10 s and episodes near the top speed track it well).
        let survive = b.ep_lengths.iter().filter(|l| **l >= 9.9).count() as f32 / ne;
        let top: Vec<f32> = b.ep_track.iter().zip(&b.ep_cmd).filter(|(_, c)| **c >= cfg.vmax - 1.0).map(|(t, _)| *t).collect();
        let top_track = if top.is_empty() { 0.0 } else { top.iter().sum::<f32>() / top.len() as f32 };
        // Knockdown curriculum: stronger shoves once ≥60% of get-up episodes end standing.
        let gu: Vec<f32> = b.ep_getup.iter().copied().filter(|v| *v >= 0.0).collect();
        let gu_rate = if gu.is_empty() { 0.0 } else { gu.iter().sum::<f32>() / gu.len() as f32 };
        if omni && it > warmup && gu.len() >= 4 && gu_rate >= 0.6 && cfg.knock_dv < 3.0 {
            cfg.knock_dv = (cfg.knock_dv + 0.5).min(3.0);
            println!("  >>> knockdown shove raised to {:.1} m/s at iteration {it}", cfg.knock_dv);
        }
        if speed_mode && !omni && it > warmup {
            if survive >= 0.85 && top_track >= 0.55 { streak += 1 } else { streak = 0 }
            if streak >= 5 && cfg.vmax < 7.5 {
                cfg.vmax = (cfg.vmax + 0.5).min(7.5);
                streak = 0;
                vmax_log.push((it, cfg.vmax));
                println!("  >>> top speed raised to {:.1} m/s at iteration {it}", cfg.vmax);
            }
        }
        // Thrust curriculum: stronger thrust once ≥60% of thrust episodes stay up.
        let th: Vec<f32> = b.ep_thrust.iter().copied().filter(|v| *v >= 0.0).collect();
        let th_rate = if th.is_empty() { 0.0 } else { th.iter().sum::<f32>() / th.len() as f32 };
        if omni && it > warmup && th.len() >= 4 && th_rate >= 0.6 && cfg.thrust_k < 1.0 {
            cfg.thrust_k = (cfg.thrust_k + 0.1).min(1.0);
            println!("  >>> thrust strength raised to {:.0}% at iteration {it}", cfg.thrust_k * 100.0);
        }
        if omni {
            print!("[knock {:.1} getup {:.2}] ", cfg.knock_dv, gu_rate);
            if !th.is_empty() {
                print!("[thr {:.0}% up {:.2}] ", cfg.thrust_k * 100.0, th_rate);
            }
        }
        println!("iter {it:4}: ep_return {ret:7.1} ep_len {len:4.1}s eps {:3} surv {:.2} top_track {:.2} vmax {:.1} | kl {:.4} clip {:.2} vloss {:.3} std {:.3} | {:.0}s",
            b.ep_returns.len(), survive, top_track, cfg.vmax, st.kl, st.clip_frac, st.v_loss, std_mean, t0.elapsed().as_secs_f32());
        // Lightweight save every 10 iterations so a restarted machine loses little work.
        if (speed_mode || omni) && it % 10 == 0 {
            fs::write(format!("{dir}/latest_policy.json"), serde_json::to_string(&pol).unwrap()).unwrap();
            fs::write(format!("{dir}/state.json"), serde_json::json!({"iter": it, "vmax": cfg.vmax, "thrust_k": cfg.thrust_k, "knock_dv": cfg.knock_dv}).to_string()).unwrap();
        }
        let ck_every = if speed_mode { 50 } else { 25 };
        if it % ck_every == 0 {
            checkpoint(it, &pol, &mut log, serde_json::json!({"ep_return": ret, "ep_len": len, "kl": st.kl, "vmax": cfg.vmax, "vmax_history": vmax_log}));
        }
    }
}

/// Render a standing pose for look approval: pelvis pinned, joints driven to a stand pose,
/// optional waist twist; the frame is shifted so the feet sit on the ground.
fn pose(build: &str, out: &str, twist: f32) {
    use rapier3d::prelude::*;
    let spec = build_spec(build);
    let mut w = PhysicsWorld::new();
    w.integration_parameters.dt = sim::DT;
    w.integration_parameters.num_solver_iterations = sim::SUBSTEPS;
    w.gravity = Vector::ZERO;
    let r = robot::build(&mut w, Vector::ZERO, spec);
    w.bodies[r.trunk].set_body_type(RigidBodyType::Fixed, true);
    let lmax = run_ctl::max_leg_len(&spec);
    let (qk, qm, p0) = run_ctl::leg_ik(&spec, 0.90 * lmax, 1.0);
    let g = robot::JointGains { kp: 1.0, kd: 0.2 };
    for _ in 0..240 {
        for leg in &r.legs {
            robot::set_motor(&mut w, leg.joints[robot::HIP_PITCH], robot::HIP_PITCH, -p0, &g);
            robot::set_motor(&mut w, leg.joints[robot::KNEE], robot::KNEE, qk, &g);
            if let Some((jm, _)) = leg.meta {
                robot::set_motor(&mut w, jm, robot::META, qm, &g);
            }
            robot::set_motor(&mut w, leg.joints[robot::ANKLE_PITCH], robot::ANKLE_PITCH, -(-p0 + qk + qm), &g);
            robot::set_motor(&mut w, leg.joints[robot::HIP_ROLL], robot::HIP_ROLL, leg.side * 0.04, &g);
        }
        if let Some(arms) = &r.arms {
            for (i, arm) in arms.iter().enumerate() {
                let side = r.legs[i].side;
                robot::set_motor(&mut w, arm.pitch.0, robot::ARM, -0.15, &g);
                if let Some(jr) = arm.roll { robot::set_motor(&mut w, jr, robot::ARM, side * 0.18, &g); }
                if let Some((je, _)) = arm.elbow { robot::set_motor(&mut w, je, robot::ARM, -0.45, &g); }
            }
        }
        if let Some((wj, _)) = r.waist {
            robot::set_motor(&mut w, wj, robot::WAIST, twist, &g);
        }
        w.step();
    }
    // Clipping check: every pair of colliders on different bodies must not intersect.
    let names: std::collections::HashMap<_, _> = r.links.iter().map(|l| (l.body, l.name.clone())).collect();
    let cols: Vec<_> = w.colliders.iter().filter_map(|(h, c)| c.parent().map(|b| (h, b))).collect();
    let mut overlaps = Vec::new();
    for i in 0..cols.len() {
        for j in i + 1..cols.len() {
            if cols[i].1 == cols[j].1 {
                continue;
            }
            // Inner axes of a ball joint live inside the ball by design.
            let internal = |b| names[&b].ends_with("_hip_roll") || names[&b].ends_with("_shoulder_pitch");
            if internal(cols[i].1) || internal(cols[j].1) {
                continue;
            }
            let (a, b) = (&w.colliders[cols[i].0], &w.colliders[cols[j].0]);
            if rapier3d::parry::query::intersection_test(a.position(), a.shape(), b.position(), b.shape()).map(|r| r.intersecting).unwrap_or(false) {
                overlaps.push(format!("{} × {}", names[&cols[i].1], names[&cols[j].1]));
            }
        }
    }
    println!("{build}: clipping check — {} overlapping part pairs{}", overlaps.len(), if overlaps.is_empty() { String::new() } else { format!(": {}", overlaps.join(", ")) });
    // Shift so the lowest foot touches y = 0.
    let min_y = r.legs.iter().map(|l| w.bodies[l.bodies[robot::B_FOOT]].translation().y - 0.07).fold(f32::MAX, f32::min);
    let b: Vec<[f32; 7]> = r.links.iter().map(|l| {
        let body = &w.bodies[l.body];
        let (p, q) = (body.translation(), body.rotation());
        [p.x, p.y - min_y, p.z, q.x, q.y, q.z, q.w]
    }).collect();
    let top = r.links.iter().flat_map(|l| { let body = &w.bodies[l.body]; l.shapes.iter().map(move |s| body.translation().y + s.offset[1] + s.half[1]) }).fold(f32::MIN, f32::max) - min_y;
    let hip = w.bodies[r.legs[0].bodies[robot::B_YAW_GIMBAL]].translation().y - min_y;
    println!("{build}: mass {:.0} kg, motor scale {:.2}, standing height ~{:.2} m, hip {:.2} m ({:.0}% of height)", r.total_mass, r.motor_scale, top, hip, 100.0 * hip / top);
    let rec = serde_json::json!({ "scenario": format!("BUILD {build} · look check"), "assist": 0.0, "dt": sim::DT, "links": r.links, "platform": null,
        "pushes": [], "frames": [{ "t": 0.0, "b": b, "speed": 0.0, "cmd_speed": 0.0, "heading": 0.0, "cmd_heading": 0.0, "label": "", "feet": 2, "toe": [0.0, 0.0], "contact": [true, true] }] });
    fs::write(out, serde_json::to_string(&rec).unwrap()).unwrap();
}

/// Thruster demo: legs run the given policy while the thrusters follow a scripted input sequence.
/// Demo: jog, thrust left, dash forward, thrust up, thrust back, dash right.
fn thrust_script() -> (Vec<(f32, controller::Command)>, Vec<(f32, String)>) {
    let cmds = vec![
        cmd(0.0, 2.0, 0.0),
        cmd_t(2.0, 1.0, 0.0, 0.0, [1.0, 0.0, 0.0], false),   // strafe left on thrust
        cmd_t(3.5, 1.0, 0.0, 0.0, [0.0, 0.0, 0.0], false),
        cmd_t(4.0, 2.5, 0.0, 0.0, [0.0, 0.0, 1.0], true),    // dash forward
        cmd_t(4.35, 3.0, 0.0, 0.0, [0.0, 0.0, 0.0], false),
        cmd_t(6.0, 1.5, 0.0, 0.0, [0.0, 1.0, 0.0], false),   // thrust up
        cmd_t(7.0, 1.0, 0.0, 0.0, [0.0, 0.0, 0.0], false),
        cmd_t(8.0, 0.0, 0.0, 0.0, [0.0, 0.0, -1.0], false),  // thrust backwards
        cmd_t(9.5, 0.5, 0.0, 0.0, [0.0, 0.0, 0.0], false),
        cmd_t(10.0, 0.5, 0.0, 0.0, [-1.0, 0.0, 0.0], true),  // dash right
        cmd_t(10.35, 1.0, 0.0, 0.0, [0.0, 0.0, 0.0], false),
    ];
    let labels = vec![(2.0, "THRUST LEFT".to_string()), (4.0, "DASH FORWARD".to_string()), (6.0, "THRUST UP".to_string()), (8.0, "THRUST BACK".to_string()), (10.0, "DASH RIGHT".to_string())];
    (cmds, labels)
}

/// Allocation residuals at the standing pose for each thrust direction.
/// Run each thrust test and print how it ends: fall time and the pose just before.
fn thrust_diag(build: &str, policy_path: &str) {
    let txt = fs::read_to_string(policy_path).expect("policy");
    let mut net = match serde_json::from_str::<ppo::GaussPolicy>(&txt) {
        Ok(p) => p.net,
        Err(_) => serde_json::from_str::<policy::Mlp>(&txt).unwrap(),
    };
    net.expand_inputs(policy::OBS_DIM);
    let gait = load_run_gait(build, 0.0);
    let cfg = ppo::RewardCfg { stride_target: 1.8, thrust_k: 1.0, vmax: 7.0, speed_mode: true, omni: true, knock_dv: 3.0 };
    let base: f32 = std::env::var("DIAG_V").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    for (name, dir, dash) in [("none", [0.0, 0.0, 0.0], false), ("L", [1.0, 0.0, 0.0], false), ("R", [-1.0, 0.0, 0.0], false), ("F", [0.0, 0.0, 1.0], false),
        ("B", [0.0, 0.0, -1.0], false), ("U", [0.0, 1.0, 0.0], false), ("U.5", [0.0, 0.5, 0.0], false), ("L.5", [0.5, 0.0, 0.0], false), ("L.3", [0.3, 0.0, 0.0], false), ("F.3", [0.0, 0.0, 0.3], false), ("B.3", [0.0, 0.0, -0.3], false), ("L.1", [0.1, 0.0, 0.0], false),
        ("dashF", [0.0, 0.0, 1.0], true), ("dashL", [1.0, 0.0, 0.0], true)] {
        let on = if dash { 0.4 } else { 2.0 };
        let mut sc = scenario("run_mid", 0.0);
        sc.spec = build_spec(build);
        sc.duration = 6.5;
        sc.commands = vec![cmd(0.0, 1.0, 0.0), cmd_t(1.5, base, 0.0, 0.0, dir, dash), cmd(1.5 + on, base, 0.0)];
        let mut env = ppo::Env::with(sc.spec, gait, sc.clone(), ppo::EpMode::Thrust, 0.0);
        let mut hist = Vec::new();
        let mut end = String::from("stayed up");
        while env.t() < sc.duration {
            let o = env.obs();
            let a = net.forward_norm(&o);
            let (_, term, _) = env.step(&net, &a, cfg);
            let (h, tilt, feet) = env.pose_stats();
            let (vf, vl, yaw) = env.velocity();
            hist.push((env.t(), h, tilt.to_degrees(), feet, vf, vl, yaw.to_degrees()));
            if term {
                end = format!("FELL at {:.2}s ({:.2}s after thrust end)", env.t(), env.t() - 1.5 - on);
                break;
            }
        }
        println!("{name:6} {end}");
        for hh in hist.iter().rev().step_by(15).take(5).collect::<Vec<_>>().iter().rev() {
            println!("        t {:.2} h {:.2} tilt {:>5.1} feet {} v_f {:+.1} v_l {:+.1} yaw {:+.0}", hh.0, hh.1, hh.2, hh.3, hh.4, hh.5, hh.6);
        }
    }
}

fn thrust_check(build: &str) {
    use rapier3d::prelude::Vector;
    let mut sc = scenario("run_mid", 0.0);
    sc.spec = build_spec(build);
    let (w, r) = sim::build_world(&sc);
    for (name, dir, dash) in [("L", [1.0, 0.0, 0.0], false), ("F", [0.0, 0.0, 1.0], false), ("B", [0.0, 0.0, -1.0], false),
        ("U", [0.0, 1.0, 0.0], false), ("dashF", [0.0, 0.0, 1.0], true), ("dashL", [1.0, 0.0, 0.0], true)] {
        let fd = thrust::desired_force(&r, 0.0, dir, if dash { 1.0 } else { 0.0 }, 0.0);
        let out = thrust::allocate(&w, &r, fd, Vector::ZERO, if dash { 1.0 } else { 0.0 });
        let (com, _) = controller::centre_of_mass(&w, &r);
        let sf: Vector = out.forces.iter().map(|(_, f)| *f).sum();
        let st: Vector = out.forces.iter().map(|(p, f)| (*p - com).cross(*f)).sum();
        if std::env::var("THR_DUMP").is_ok() {
            for (t, (p, f)) in r.thrusters.iter().zip(&out.forces) {
                println!("    {:8} at {:+.2} {:+.2} {:+.2}  f {:>7.0} {:>7.0} {:>7.0}", t.name, p.x, p.y, p.z, f.x, f.y, f.z);
            }
        }
        println!("{name:6} want {:>7.0} {:>7.0} {:>7.0} | got {:>7.0} {:>7.0} {:>7.0} | torque {:>7.0} {:>7.0} {:>7.0} N·m",
            fd.x, fd.y, fd.z, sf.x, sf.y, sf.z, st.x, st.y, st.z);
    }
}

fn thrust_demo(build: &str, policy_path: &str, out: &str) {
    let txt = fs::read_to_string(policy_path).expect("policy");
    let mut net = match serde_json::from_str::<ppo::GaussPolicy>(&txt) {
        Ok(p) => p.net,
        Err(_) => serde_json::from_str::<policy::Mlp>(&txt).unwrap(),
    };
    net.expand_inputs(policy::OBS_DIM);
    let gait = load_run_gait("GL", 0.0);
    let mut sc = scenario("run_mid", 0.0);
    sc.spec = build_spec(build);
    sc.duration = 13.0;
    sc.commands = thrust_script().0;
    let labels = thrust_script().1;
    let cfg = ppo::RewardCfg { stride_target: 1.8, thrust_k: 1.0, vmax: 7.0, speed_mode: true, omni: true, knock_dv: 3.0 };
    let mut env = ppo::Env::with(sc.spec, gait, sc.clone(), ppo::EpMode::Omni, 0.0);
    env.frames = Some(Vec::new());
    let mut fell_at = None;
    let mut max_h: f32 = 0.0;
    while env.t() < sc.duration {
        env.label = labels.iter().rev().find(|(t, _)| env.t() >= *t && env.t() < *t + 1.5).map(|l| l.1.to_string()).unwrap_or_default();
        let o = env.obs();
        let a = net.forward_norm(&o);
        let (_, term, _) = env.step(&net, &a, cfg);
        let (vf, vl, _) = env.velocity();
        if (env.t() * 60.0).round() as i32 % 30 == 0 {
            println!("t {:5.2} fwd {:+.2} left {:+.2} {}", env.t(), vf, vl, env.label);
        }
        max_h = max_h.max(env.frames.as_ref().unwrap().last().map(|f| f.b[0][1]).unwrap_or(0.0));
        if term && fell_at.is_none() {
            fell_at = Some(env.t());
            println!("fell at {:.2}", env.t());
            break;
        }
    }
    println!("max pelvis height {max_h:.2} m");
    save_clip(&env, out, &format!("THRUSTERS · build {build}"));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let a = |i: usize| args.get(i).cloned();
    match a(1).as_deref() {
        Some("run") => {
            let name = a(2).expect("scenario");
            let assist: f32 = a(3).map(|s| s.parse().unwrap()).unwrap_or(1.0);
            let mut sc = scenario(&name, assist);
            sc.record = a(4).is_some();
            let (m, rec) = run(&sc, load_gait(assist));
            println!("{}", serde_json::to_string_pretty(&m).unwrap());
            if let (Some(path), Some(rec)) = (a(4), rec) {
                fs::write(path, serde_json::to_string(&rec).unwrap()).unwrap();
            }
        }
        Some("tune") => tune(a(2).map(|s| s.parse().unwrap()).unwrap_or(100), a(3).map(|s| s.parse().unwrap()).unwrap_or(1.0)),
        Some("pushtest") => println!("{}", pushtest(a(2).map(|s| s.parse().unwrap()).unwrap_or(1.0))),
        Some("soak") => {
            let m = soak(a(2).map(|s| s.parse().unwrap()).unwrap_or(600.0), a(3).map(|s| s.parse().unwrap()).unwrap_or(1.0));
            println!("{}", serde_json::to_string_pretty(&m).unwrap());
        }
        Some("diag") => diag(),
        Some("runreport") => run_report(),
        Some("evaldistill") => eval_distill(&a(2).unwrap(), a(3).map(|s| s.parse().unwrap()).unwrap_or(0.0)),
        Some("thrustdiag") => thrust_diag(&a(2).unwrap(), &a(3).unwrap()),
        Some("thrustcheck") => thrust_check(&a(2).unwrap()),
        Some("thrustdemo") => thrust_demo(&a(2).unwrap(), &a(3).unwrap(), &a(4).unwrap()),
        Some("pose") => pose(&a(2).unwrap(), &a(3).unwrap(), a(4).map(|s| s.parse().unwrap()).unwrap_or(0.0)),
        Some("ppospeed") => ppo_train_mode(&a(2).unwrap(), a(3).map(|s| s.parse().unwrap()).unwrap_or(1000), &a(4).unwrap(), true,
            a(5).map(|s| s.parse().unwrap()), a(6).map(|s| s.parse().unwrap()).unwrap_or(0), false),
        // Omnidirectional jogging + get-up (speed rewards kept for forward running episodes).
        Some("ppoomni") => ppo_train_mode(&a(2).unwrap(), a(3).map(|s| s.parse().unwrap()).unwrap_or(1000), &a(4).unwrap(), true,
            Some(7.0), a(5).map(|s| s.parse().unwrap()).unwrap_or(0), true),
        Some("ppo") => ppo_train(&a(2).unwrap(), a(3).map(|s| s.parse().unwrap()).unwrap_or(100), &a(4).unwrap()),
        Some("distill") => distill(&a(2).unwrap(), a(3).map(|s| s.parse().unwrap()).unwrap_or(0.0), a(4).map(|s| s.parse().unwrap()).unwrap_or(5)),
        Some("runtune") => run_tune(&a(2).unwrap(), a(3).map(|s| s.parse().unwrap()).unwrap_or(100), a(4).map(|s| s.parse().unwrap()).unwrap_or(0.35)),
        Some("runrun") => {
            let (b, name) = (a(2).unwrap(), a(3).unwrap());
            let assist: f32 = a(4).map(|s| s.parse().unwrap()).unwrap_or(0.35);
            let (m, rec) = run_scenario(&b, &name, assist, load_run_gait(&b, assist));
            println!("{}", serde_json::to_string_pretty(&m).unwrap());
            if let (Some(path), Some(rec)) = (a(5), rec) {
                fs::write(path, serde_json::to_string(&rec).unwrap()).unwrap();
            }
        }
        Some("report") => {
            let mut out = serde_json::Map::new();
            for assist in [1.0f32, 0.35, 0.0] {
                let g = load_gait(assist);
                let mut runs = Vec::new();
                for name in ["walk_slow", "walk", "walk_fast", "turn", "stepdown", "pushes", "bigpush", "run", "showcase"] {
                    runs.push(serde_json::to_value(run(&scenario(name, assist), g).0).unwrap());
                }
                out.insert(format!("assist_{assist}"), serde_json::json!({
                    "gait": g,
                    "scenarios": runs,
                    "pushtest": pushtest(assist),
                    "soak_600s": soak(600.0, assist),
                }));
            }
            fs::create_dir_all("out").ok();
            let s = serde_json::to_string_pretty(&out).unwrap();
            fs::write("out/report.json", &s).unwrap();
            println!("{s}");
        }
        _ => eprintln!("see module docs for usage"),
    }
}

/// Sign-convention check: pin the trunk, drive a few joints, print measured limb angles.
#[allow(dead_code)]
pub fn diag() {
    use rapier3d::prelude::*;
    let mut w = PhysicsWorld::new();
    w.integration_parameters.dt = sim::DT;
    w.integration_parameters.num_solver_iterations = sim::SUBSTEPS;
    let r = robot::build(&mut w, Vector::new(0.0, 3.0, 0.0), robot::RobotSpec::default());
    w.bodies[r.trunk].set_body_type(RigidBodyType::Fixed, true);
    let g = robot::JointGains { kp: 1.0, kd: 0.12 };
    let l = &r.legs[0];
    let targets = [(robot::HIP_PITCH, -0.4), (robot::KNEE, 0.6), (robot::HIP_ROLL, 0.2), (robot::HIP_YAW, 0.3), (robot::ANKLE_PITCH, 0.3)];
    for _ in 0..180 {
        for (k, t) in targets {
            robot::set_motor(&mut w, l.joints[k], k, t, &g);
        }
        w.step();
    }
    let ang = |b: RigidBodyHandle| {
        let q = w.bodies[b].rotation();
        let d = *q * -Vector::Y;
        let f = *q * Vector::Z;
        (((-d.z).atan2(-d.y)), d.x.atan2(-d.y), f.x.atan2(f.z))
    };
    for (n, b) in [("thigh", robot::B_THIGH), ("shin", robot::B_SHIN), ("ankleG", robot::B_ANKLE_GIMBAL), ("foot", robot::B_FOOT), ("yawG", robot::B_YAW_GIMBAL)] {
        let (p, ro, y) = ang(l.bodies[b]);
        println!("{n:8} pitch {p:+.3} roll {ro:+.3} yaw {y:+.3}");
    }
}
