//! walker_lab — headless biped walker prototype.
//!
//! Usage:
//!   walker_lab run <scenario> [assist] [out.json]   run one scenario, print metrics, optionally record
//!   walker_lab tune <iters> [assist]                 random-search gait parameters -> out/gait[_a{assist}].json
//!   walker_lab pushtest [assist]                     largest survivable side/forward push
//!   walker_lab soak <seconds> [assist]               long run with random commands and pushes
//!   walker_lab report                                all of the above into out/report.json

mod controller;
mod robot;
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
        robot::build(&mut w, rapier3d::prelude::Vector::ZERO).total_mass
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
    let r = robot::build(&mut w, Vector::new(0.0, 3.0, 0.0));
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
