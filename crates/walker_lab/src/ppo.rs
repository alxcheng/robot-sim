//! PPO fine-tuning of a distilled leg policy inside Rapier.
//!
//! * Policy: Gaussian with mean = MLP(obs) in the network's normalised action space, learnable
//!   per-dimension log-std. Initialised from the behaviour-cloned network.
//! * Value: separate MLP (same input normalisation), predicting return / RET_SCALE.
//! * Reward (per 1/60 s tick): alive + speed tracking + heading + upright, plus event rewards for
//!   long strides (at each footfall, gated on speed) and toe push-off (at each lift-off).

use crate::controller::*;
use crate::policy::*;
use crate::robot::*;
use crate::run_ctl::*;
use crate::sim::*;
use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};

pub const RET_SCALE: f32 = 50.0;

#[derive(Clone, Serialize, Deserialize)]
pub struct GaussPolicy {
    pub net: Mlp,
    pub log_std: Vec<f32>,
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        let u1 = self.f().max(1e-7);
        let u2 = self.f();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

#[derive(Clone, Copy)]
pub struct RewardCfg {
    pub stride_target: f32,
}

pub struct Env {
    w: PhysicsWorld,
    r: Robot,
    sc: Scenario,
    t: f32,
    tracker: ContactTracker,
    gait: RunGait,
    prev_a: Vec<f32>,
    last_td: Option<[f32; 2]>,
    foot_prev: [bool; 2],
    toe_peak: [f32; 2],
    pub ep_return: f32,
}

fn random_scenario(spec: RobotSpec, rng: &mut Rng, dur: f32) -> Scenario {
    let mut sc = scenario("run_mid", 0.0);
    sc.spec = spec;
    sc.duration = dur;
    sc.commands.clear();
    let (mut t, mut heading) = (0.0, 0.0);
    while t < dur {
        heading += (rng.f() - 0.5) * 0.6;
        sc.commands.push(cmd(t, 1.5 + rng.f() * 1.6, heading));
        if rng.f() < 0.3 {
            let a = rng.f() * std::f32::consts::TAU;
            sc.pushes.push(Push { t: t + 2.0, dv_fwd: 0.12 * a.cos(), dv_left: 0.12 * a.sin() });
        }
        t += 4.0;
    }
    sc
}

impl Env {
    pub fn new(spec: RobotSpec, gait: RunGait, seed: u64) -> Self {
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let sc = random_scenario(spec, &mut rng, 10.0);
        let (w, r) = build_world(&sc);
        Env {
            w, r, sc, t: 0.0, tracker: Default::default(), gait, prev_a: vec![0.0; ACTION_DIM],
            last_td: None, foot_prev: [true; 2], toe_peak: [0.0; 2], ep_return: 0.0,
        }
    }

    pub fn obs(&self) -> [f32; OBS_DIM] {
        observe(&self.w, &self.r, command_at(&self.sc, self.t), &self.tracker)
    }

    /// Advance one control tick with a normalised action. Returns (reward, terminal, truncated).
    pub fn step(&mut self, net: &Mlp, a_norm: &[f32], cfg: RewardCfg) -> (f32, bool, bool) {
        let cmdv = command_at(&self.sc, self.t);
        let real: Vec<f32> = a_norm.iter().enumerate().map(|(i, v)| v * net.out_std[i] + net.out_mean[i]).collect();
        let act = Action::from_slice(&real);
        apply_action(&mut self.w, &self.r, &act, self.gait.kd);
        reflexes(&mut self.w, &self.r, self.gait.wheel_k, 0.0, self.gait.lean + self.gait.lean_v * cmdv.speed);
        // Pushes as short force pulses (see sim.rs).
        {
            let trunk = &mut self.w.bodies[self.r.trunk];
            trunk.reset_forces(true);
            for p in &self.sc.pushes {
                if self.t >= p.t && self.t < p.t + 0.1 {
                    let yaw = yaw_of(trunk.rotation());
                    let dv = Rotation::from_rotation_y(yaw) * Vector::new(p.dv_left, 0.0, p.dv_fwd);
                    trunk.add_force(dv * self.r.total_mass / 0.1, true);
                }
            }
        }
        self.w.step();
        self.t += DT;
        let c = [foot_in_contact(&self.w, &self.r.legs[0]), foot_in_contact(&self.w, &self.r.legs[1])];
        self.tracker.update(c, DT);
        let tel = telemetry(&self.w, &self.r, None);

        let finite = tel.com.iter().all(|v| v.is_finite());
        let fallen = !finite || tel.trunk_tilt > 0.9 || tel.com[1] < 1.2;
        if fallen {
            return (-10.0, true, false);
        }
        let v_err = tel.fwd_speed - cmdv.speed;
        let head_err = wrap(cmdv.heading - tel.yaw);
        let tilt_deg = tel.trunk_tilt.to_degrees();
        let smooth: f32 = a_norm.iter().zip(&self.prev_a).map(|(a, b)| (a - b) * (a - b)).sum();
        self.prev_a = a_norm.to_vec();
        let mut rew = 0.5 + 1.0 * (-v_err * v_err / 0.3).exp() + 0.3 * (-head_err * head_err / 0.05).exp()
            - 0.03 * (tilt_deg - 10.0).max(0.0)
            - 0.002 * smooth;
        let speed_gate = (-v_err * v_err / 0.5).exp();
        for k in 0..2 {
            let fb = &self.w.bodies[self.r.legs[k].bodies[B_FOOT]];
            let f = *fb.rotation() * Vector::Z;
            let toe = (-f.y).atan2((f.x * f.x + f.z * f.z).sqrt()).to_degrees();
            if c[k] {
                self.toe_peak[k] = self.toe_peak[k].max(toe);
            }
            if c[k] && !self.foot_prev[k] {
                // Footfall: stride reward, gated on holding the commanded speed.
                let p = [tel.com[0], tel.com[2]];
                if let Some(q) = self.last_td {
                    let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
                    rew += 3.0 * (d / cfg.stride_target).clamp(0.0, 1.2) * speed_gate;
                }
                self.last_td = Some(p);
            }
            if !c[k] && self.foot_prev[k] {
                // Lift-off: push-off reward.
                rew += 1.0 * (self.toe_peak[k] / 20.0).clamp(0.0, 1.0) * speed_gate;
                self.toe_peak[k] = 0.0;
            }
            self.foot_prev[k] = c[k];
        }
        self.ep_return += rew;
        (rew, false, self.t >= self.sc.duration)
    }
}

pub struct Batch {
    pub obs: Vec<[f32; OBS_DIM]>,
    pub act: Vec<Vec<f32>>,
    pub logp: Vec<f32>,
    pub adv: Vec<f32>,
    pub ret: Vec<f32>,
    pub ep_returns: Vec<f32>,
    pub ep_lengths: Vec<f32>,
}

pub fn logp(mu: &[f32], log_std: &[f32], a: &[f32]) -> f32 {
    let mut s = 0.0;
    for i in 0..mu.len() {
        let z = (a[i] - mu[i]) / log_std[i].exp();
        s += -0.5 * z * z - log_std[i] - 0.918_938_5;
    }
    s
}

/// Collect `steps` transitions with the stochastic policy (one thread's share), with GAE.
pub fn rollout(pol: &GaussPolicy, value: &Mlp, spec: RobotSpec, gait: RunGait, steps: usize, seed: u64, cfg: RewardCfg) -> Batch {
    let (gamma, lam) = (0.99f32, 0.95f32);
    let mut rng = Rng(seed | 1);
    let mut env = Env::new(spec, gait, seed);
    let mut episode = 0u64;
    let (mut obs, mut act, mut lp, mut rews, mut vals, mut dones, mut boots) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    let (mut ep_returns, mut ep_lengths) = (vec![], vec![]);
    for _ in 0..steps {
        let o = env.obs();
        let mu = pol.net.forward_norm(&o);
        let a: Vec<f32> = mu.iter().enumerate().map(|(i, m)| m + pol.log_std[i].exp() * rng.normal()).collect();
        let v = value.forward_norm(&o)[0] * RET_SCALE;
        let (r, term, trunc) = env.step(&pol.net, &a, cfg);
        obs.push(o);
        lp.push(logp(&mu, &pol.log_std, &a));
        act.push(a);
        rews.push(r);
        vals.push(v);
        let end = term || trunc;
        dones.push(end);
        // Bootstrap value after a truncation (not after a fall).
        boots.push(if trunc && !term { value.forward_norm(&env.obs())[0] * RET_SCALE } else { 0.0 });
        if end {
            ep_returns.push(env.ep_return);
            ep_lengths.push(env.t);
            episode += 1;
            env = Env::new(spec, gait, seed.wrapping_add(episode * 7919));
        }
    }
    // Last state bootstrap for an unfinished episode.
    let last_v = value.forward_norm(&env.obs())[0] * RET_SCALE;
    let n = rews.len();
    let mut adv = vec![0.0; n];
    let mut gae = 0.0;
    for t in (0..n).rev() {
        let next_v = if dones[t] { boots[t] } else if t + 1 < n { vals[t + 1] } else { last_v };
        let delta = rews[t] + gamma * next_v - vals[t];
        gae = delta + if dones[t] { 0.0 } else { gamma * lam * gae };
        adv[t] = gae;
    }
    let ret: Vec<f32> = adv.iter().zip(&vals).map(|(a, v)| a + v).collect();
    Batch { obs, act, logp: lp, adv, ret, ep_returns, ep_lengths }
}

pub struct UpdateStats {
    pub kl: f32,
    pub clip_frac: f32,
    pub v_loss: f32,
    pub epochs_run: usize,
}

/// PPO-clip update on a merged batch, gradients on 4 threads.
#[allow(clippy::too_many_arguments)]
pub fn update(
    pol: &mut GaussPolicy,
    value: &mut Mlp,
    popt: &mut Adam,
    vopt: &mut Adam,
    lsopt: &mut (Vec<f32>, Vec<f32>, i32),
    b: &Batch,
    lr_pi: f32,
    lr_v: f32,
) -> UpdateStats {
    let (clip, ent_coef, epochs, mb) = (0.2f32, 0.0005f32, 4usize, 4096usize);
    let n = b.obs.len();
    let mean = b.adv.iter().sum::<f32>() / n as f32;
    let std = (b.adv.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / n as f32).sqrt().max(1e-6);
    let adv: Vec<f32> = b.adv.iter().map(|a| (a - mean) / std).collect();
    let mut rng = Rng(n as u64 * 31 + 7);
    let mut stats = UpdateStats { kl: 0.0, clip_frac: 0.0, v_loss: 0.0, epochs_run: 0 };
    for _ in 0..epochs {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (rng.f() * (i + 1) as f32) as usize % (i + 1);
            idx.swap(i, j);
        }
        let (mut kl_sum, mut clip_sum, mut vl_sum) = (0.0f32, 0.0f32, 0.0f32);
        for chunk in idx.chunks(mb) {
            let parts: Vec<&[usize]> = chunk.chunks(chunk.len().div_ceil(4)).collect();
            let polr = &*pol;
            let valr = &*value;
            let adv = &adv;
            let results: Vec<_> = std::thread::scope(|s| {
                let hs: Vec<_> = parts
                    .iter()
                    .map(|p| {
                        s.spawn(move || {
                            let mut gp = polr.net.zero_grads();
                            let mut gv = valr.zero_grads();
                            let mut gls = vec![0.0f32; polr.log_std.len()];
                            let (mut kl, mut clipped, mut vl) = (0.0f32, 0.0f32, 0.0f32);
                            for &i in p.iter() {
                                let o = &b.obs[i];
                                let a = &b.act[i];
                                let mu = polr.net.forward_norm(o);
                                let lpn = logp(&mu, &polr.log_std, a);
                                let ratio = (lpn - b.logp[i]).exp();
                                kl += b.logp[i] - lpn;
                                let av = adv[i];
                                let active = !((av >= 0.0 && ratio > 1.0 + clip) || (av < 0.0 && ratio < 1.0 - clip));
                                if !active {
                                    clipped += 1.0;
                                } else {
                                    let coef = -av * ratio;
                                    let dmu: Vec<f32> = (0..mu.len())
                                        .map(|j| {
                                            let s2 = (2.0 * polr.log_std[j]).exp();
                                            coef * (a[j] - mu[j]) / s2
                                        })
                                        .collect();
                                    for j in 0..mu.len() {
                                        let s2 = (2.0 * polr.log_std[j]).exp();
                                        gls[j] += coef * ((a[j] - mu[j]).powi(2) / s2 - 1.0);
                                    }
                                    polr.net.backprop(o, &dmu, &mut gp);
                                }
                                for g in gls.iter_mut() {
                                    *g -= ent_coef;
                                }
                                let v = valr.forward_norm(o)[0];
                                let target = b.ret[i] / RET_SCALE;
                                vl += (v - target).powi(2);
                                valr.backprop(o, &[2.0 * (v - target)], &mut gv);
                            }
                            (gp, gv, gls, kl, clipped, vl)
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let m = chunk.len() as f32;
            let mut gp = pol.net.zero_grads();
            let mut gv = value.zero_grads();
            let mut gls = vec![0.0f32; pol.log_std.len()];
            for (rp, rv, rl, kl, cl, vl) in results {
                for (a, bb) in gp.iter_mut().zip(rp) {
                    for (x, y) in a.iter_mut().zip(bb) {
                        *x += y;
                    }
                }
                for (a, bb) in gv.iter_mut().zip(rv) {
                    for (x, y) in a.iter_mut().zip(bb) {
                        *x += y;
                    }
                }
                for (x, y) in gls.iter_mut().zip(rl) {
                    *x += y;
                }
                kl_sum += kl;
                clip_sum += cl;
                vl_sum += vl;
            }
            if lr_pi > 0.0 {
                pol.net.apply_adam(popt, &gp, m, lr_pi);
                // Adam on log-std.
                lsopt.2 += 1;
                let (b1, b2) = (0.9f32, 0.999f32);
                let c1 = 1.0 - b1.powi(lsopt.2);
                let c2 = 1.0 - b2.powi(lsopt.2);
                for j in 0..pol.log_std.len() {
                    let g = gls[j] / m;
                    lsopt.0[j] = b1 * lsopt.0[j] + (1.0 - b1) * g;
                    lsopt.1[j] = b2 * lsopt.1[j] + (1.0 - b2) * g * g;
                    pol.log_std[j] -= 1e-3 * (lsopt.0[j] / c1) / ((lsopt.1[j] / c2).sqrt() + 1e-8);
                    pol.log_std[j] = pol.log_std[j].clamp((0.02f32).ln(), (0.6f32).ln());
                }
            }
            value.apply_adam(vopt, &gv, m, lr_v);
        }
        stats.epochs_run += 1;
        stats.kl = kl_sum / n as f32;
        stats.clip_frac = clip_sum / n as f32;
        stats.v_loss = vl_sum / n as f32;
        if stats.kl > 0.02 {
            break;
        }
    }
    stats
}
