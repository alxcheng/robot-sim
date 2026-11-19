//! Learned leg policy: observation builder, a small MLP (tanh) with Adam, and a controller that
//! runs the network in place of the hand-built running controller.

use crate::controller::*;
use crate::robot::*;
use crate::run_ctl::*;
use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};

pub const OBS_DIM: usize = 58;

/// Contact history the policy is allowed to see (derived from sensors, not from the teacher).
#[derive(Clone, Copy, Default)]
pub struct ContactTracker {
    pub last_stance: usize,
    pub t_since_change: f32,
    pub prev: [bool; 2],
}

impl ContactTracker {
    pub fn update(&mut self, c: [bool; 2], dt: f32) {
        if c != self.prev {
            self.t_since_change = 0.0;
            for i in 0..2 {
                if c[i] && !self.prev[i] {
                    self.last_stance = i;
                }
            }
            self.prev = c;
        }
        self.t_since_change += dt;
    }
}

pub fn observe(w: &PhysicsWorld, r: &Robot, cmd: Command, tr: &ContactTracker) -> [f32; OBS_DIM] {
    let mut o = [0.0f32; OBS_DIM];
    let mut k = 0;
    let mut push = |v: f32| {
        o[k] = if v.is_finite() { v } else { 0.0 };
        k += 1;
    };
    let trunk_rot = *w.bodies[r.trunk].rotation();
    let yaw = yaw_of(&trunk_rot);
    let inv_h = Rotation::from_rotation_y(-yaw);
    let (com, com_vel) = centre_of_mass(w, r);
    push(cmd.speed / 3.0);
    push(wrap(cmd.heading - yaw));
    let (tp, trl) = trunk_angles(&trunk_rot, &inv_h);
    push(tp);
    push(trl);
    let om = inv_h * w.bodies[r.trunk].angvel();
    push(om.x);
    push(om.y);
    push(om.z);
    let v = inv_h * com_vel;
    push(v.x);
    push(v.y);
    push(v.z);
    push(com.y - 1.8);
    for leg in &r.legs {
        let parts = [leg.bodies[B_THIGH], leg.bodies[B_SHIN], leg.meta.map(|m| m.1).unwrap_or(leg.bodies[B_SHIN]), leg.bodies[B_FOOT]];
        for b in parts {
            let (p, rl) = limb_angles(w.bodies[b].rotation(), &inv_h);
            push(p);
            push(rl);
        }
        for b in parts {
            let wv = inv_h * w.bodies[b].angvel();
            push(wv.x * 0.2);
            push(wv.z * 0.2);
        }
        let fp = inv_h * (w.bodies[leg.bodies[B_FOOT]].translation() - com);
        push(fp.x);
        push(fp.y);
        push(fp.z);
    }
    let c = [foot_in_contact(w, &r.legs[0]), foot_in_contact(w, &r.legs[1])];
    push(c[0] as u8 as f32);
    push(c[1] as u8 as f32);
    push((tr.last_stance == 0) as u8 as f32);
    push((tr.last_stance == 1) as u8 as f32);
    push(tr.t_since_change.min(1.0));
    for i in 0..2 {
        let (a, e) = match &r.arms {
            Some(arms) => {
                let up = limb_angles(w.bodies[arms[i].pitch.1].rotation(), &inv_h).0;
                let el = arms[i].elbow.map(|(_, fb)| limb_angles(w.bodies[fb].rotation(), &inv_h).0 - up).unwrap_or(0.0);
                (up - tp, el)
            }
            None => (0.0, 0.0),
        };
        push(a);
        push(e);
    }
    debug_assert_eq!(k, OBS_DIM);
    o
}

// ---------------- MLP ----------------

#[derive(Clone, Serialize, Deserialize)]
pub struct Layer {
    pub n_in: usize,
    pub n_out: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Mlp {
    pub layers: Vec<Layer>,
    pub in_mean: Vec<f32>,
    pub in_std: Vec<f32>,
    pub out_mean: Vec<f32>,
    pub out_std: Vec<f32>,
}

pub struct Adam {
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: i32,
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Chunked so the compiler vectorises it.
    let mut acc = [0.0f32; 8];
    let n = a.len() / 8 * 8;
    for i in (0..n).step_by(8) {
        for j in 0..8 {
            acc[j] += a[i + j] * b[i + j];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in n..a.len() {
        s += a[i] * b[i];
    }
    s
}

impl Mlp {
    pub fn new(sizes: &[usize], seed: u64) -> Self {
        let mut rng = Rng(seed | 1);
        let layers = sizes
            .windows(2)
            .map(|p| {
                let (n_in, n_out) = (p[0], p[1]);
                let s = (6.0 / (n_in + n_out) as f32).sqrt();
                Layer { n_in, n_out, w: (0..n_in * n_out).map(|_| (rng.f() * 2.0 - 1.0) * s).collect(), b: vec![0.0; n_out] }
            })
            .collect();
        Mlp {
            layers,
            in_mean: vec![0.0; sizes[0]],
            in_std: vec![1.0; sizes[0]],
            out_mean: vec![0.0; *sizes.last().unwrap()],
            out_std: vec![1.0; *sizes.last().unwrap()],
        }
    }

    pub fn fit_normalisation(&mut self, xs: &[Vec<f32>], ys: &[Vec<f32>]) {
        let stats = |d: &[Vec<f32>]| {
            let n = d.len() as f32;
            let dim = d[0].len();
            let mut mean = vec![0.0; dim];
            let mut var = vec![0.0; dim];
            for r in d {
                for i in 0..dim {
                    mean[i] += r[i] / n;
                }
            }
            for r in d {
                for i in 0..dim {
                    var[i] += (r[i] - mean[i]).powi(2) / n;
                }
            }
            (mean, var.into_iter().map(|v: f32| v.sqrt().max(1e-3)).collect::<Vec<_>>())
        };
        (self.in_mean, self.in_std) = stats(xs);
        (self.out_mean, self.out_std) = stats(ys);
    }

    /// Forward pass on normalised input; returns activations of every layer (last = normalised output).
    fn forward_acts(&self, x: &[f32]) -> Vec<Vec<f32>> {
        let mut acts = vec![x.iter().enumerate().map(|(i, v)| (v - self.in_mean[i]) / self.in_std[i]).collect::<Vec<f32>>()];
        for (li, l) in self.layers.iter().enumerate() {
            let a = acts.last().unwrap();
            let last = li + 1 == self.layers.len();
            let z: Vec<f32> = (0..l.n_out)
                .map(|o| {
                    let v = l.b[o] + dot(&l.w[o * l.n_in..(o + 1) * l.n_in], a);
                    if last { v } else { v.tanh() }
                })
                .collect();
            acts.push(z);
        }
        acts
    }

    pub fn predict(&self, x: &[f32]) -> Vec<f32> {
        let acts = self.forward_acts(x);
        acts.last().unwrap().iter().enumerate().map(|(i, v)| v * self.out_std[i] + self.out_mean[i]).collect()
    }

    fn grads(&self, xs: &[&[f32]], ys: &[&[f32]]) -> (Vec<Vec<f32>>, f32) {
        let mut g: Vec<Vec<f32>> = self.layers.iter().flat_map(|l| [vec![0.0; l.w.len()], vec![0.0; l.b.len()]]).collect();
        let mut loss = 0.0;
        for (x, y) in xs.iter().zip(ys) {
            let acts = self.forward_acts(x);
            let out = acts.last().unwrap();
            let n_out = out.len() as f32;
            let mut delta: Vec<f32> = out
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let t = (y[i] - self.out_mean[i]) / self.out_std[i];
                    loss += (v - t).powi(2) / n_out;
                    2.0 * (v - t) / n_out
                })
                .collect();
            for li in (0..self.layers.len()).rev() {
                let l = &self.layers[li];
                let a = &acts[li];
                let (gw, rest) = g.split_at_mut(2 * li + 1);
                let gw = &mut gw[2 * li];
                let gb = &mut rest[0];
                let mut prev = vec![0.0f32; l.n_in];
                for o in 0..l.n_out {
                    let d = delta[o];
                    gb[o] += d;
                    let row = o * l.n_in;
                    for i in 0..l.n_in {
                        gw[row + i] += d * a[i];
                        prev[i] += l.w[row + i] * d;
                    }
                }
                if li > 0 {
                    for i in 0..l.n_in {
                        prev[i] *= 1.0 - a[i] * a[i]; // tanh'
                    }
                }
                delta = prev;
            }
        }
        (g, loss)
    }

    pub fn zero_grads(&self) -> Vec<Vec<f32>> {
        self.layers.iter().flat_map(|l| [vec![0.0; l.w.len()], vec![0.0; l.b.len()]]).collect()
    }

    /// Normalised network output (before de-normalisation).
    pub fn forward_norm(&self, x: &[f32]) -> Vec<f32> {
        self.forward_acts(x).pop().unwrap()
    }

    /// Accumulate parameter gradients for an arbitrary loss given dL/d(normalised output).
    pub fn backprop(&self, x: &[f32], dout: &[f32], g: &mut [Vec<f32>]) {
        let acts = self.forward_acts(x);
        let mut delta = dout.to_vec();
        for li in (0..self.layers.len()).rev() {
            let l = &self.layers[li];
            let a = &acts[li];
            let (gw, rest) = g.split_at_mut(2 * li + 1);
            let gw = &mut gw[2 * li];
            let gb = &mut rest[0];
            let mut prev = vec![0.0f32; l.n_in];
            for o in 0..l.n_out {
                let d = delta[o];
                if d == 0.0 {
                    continue;
                }
                gb[o] += d;
                let row = o * l.n_in;
                for i in 0..l.n_in {
                    gw[row + i] += d * a[i];
                    prev[i] += l.w[row + i] * d;
                }
            }
            if li > 0 {
                for i in 0..l.n_in {
                    prev[i] *= 1.0 - a[i] * a[i];
                }
            }
            delta = prev;
        }
    }

    /// Adam step with pre-summed gradients divided by `n`.
    pub fn apply_adam(&mut self, opt: &mut Adam, g: &[Vec<f32>], n: f32, lr: f32) {
        opt.t += 1;
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let c1 = 1.0 - b1.powi(opt.t);
        let c2 = 1.0 - b2.powi(opt.t);
        for (pi, li) in (0..self.layers.len()).flat_map(|l| [(2 * l, l), (2 * l + 1, l)]) {
            let params = if pi % 2 == 0 { &mut self.layers[li].w } else { &mut self.layers[li].b };
            for k in 0..params.len() {
                let gk = g[pi][k] / n;
                let m = &mut opt.m[pi][k];
                let v = &mut opt.v[pi][k];
                *m = b1 * *m + (1.0 - b1) * gk;
                *v = b2 * *v + (1.0 - b2) * gk * gk;
                params[k] -= lr * (*m / c1) / ((*v / c2).sqrt() + eps);
            }
        }
    }

    pub fn adam(&self) -> Adam {
        let z: Vec<Vec<f32>> = self.layers.iter().flat_map(|l| [vec![0.0; l.w.len()], vec![0.0; l.b.len()]]).collect();
        Adam { m: z.clone(), v: z, t: 0 }
    }

    /// One epoch of minibatch Adam over the dataset, gradients computed on 4 threads. Returns mean loss.
    pub fn train_epoch(&mut self, opt: &mut Adam, xs: &[Vec<f32>], ys: &[Vec<f32>], batch: usize, lr: f32, seed: u64) -> f32 {
        let mut idx: Vec<usize> = (0..xs.len()).collect();
        let mut rng = Rng(seed | 1);
        for i in (1..idx.len()).rev() {
            let j = (rng.f() * (i + 1) as f32) as usize % (i + 1);
            idx.swap(i, j);
        }
        let mut total = 0.0;
        for chunk in idx.chunks(batch) {
            let parts: Vec<&[usize]> = chunk.chunks(chunk.len().div_ceil(4)).collect();
            let results: Vec<(Vec<Vec<f32>>, f32)> = std::thread::scope(|s| {
                let me = &*self;
                let hs: Vec<_> = parts
                    .iter()
                    .map(|p| {
                        s.spawn(move || {
                            let bx: Vec<&[f32]> = p.iter().map(|&i| xs[i].as_slice()).collect();
                            let by: Vec<&[f32]> = p.iter().map(|&i| ys[i].as_slice()).collect();
                            me.grads(&bx, &by)
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let n = chunk.len() as f32;
            opt.t += 1;
            let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
            let c1 = 1.0 - b1.powi(opt.t);
            let c2 = 1.0 - b2.powi(opt.t);
            for (pi, li) in (0..self.layers.len()).flat_map(|l| [(2 * l, l), (2 * l + 1, l)]) {
                let params = if pi % 2 == 0 { &mut self.layers[li].w } else { &mut self.layers[li].b };
                for k in 0..params.len() {
                    let gk: f32 = results.iter().map(|r| r.0[pi][k]).sum::<f32>() / n;
                    let m = &mut opt.m[pi][k];
                    let v = &mut opt.v[pi][k];
                    *m = b1 * *m + (1.0 - b1) * gk;
                    *v = b2 * *v + (1.0 - b2) * gk * gk;
                    params[k] -= lr * (*m / c1) / ((*v / c2).sqrt() + eps);
                }
            }
            total += results.iter().map(|r| r.1).sum::<f32>();
        }
        total / xs.len() as f32
    }
}

// ---------------- controllers ----------------

/// Runs the learned network on the legs and arms; reaction wheels and assist stay as reflexes.
pub struct StudentController {
    pub net: std::sync::Arc<Mlp>,
    pub tracker: ContactTracker,
    pub gait: RunGait,
    pub assist: f32,
    pub steps: u32,
}

impl Ctl for StudentController {
    fn steps(&self) -> u32 {
        self.steps
    }
    fn tick(&mut self, w: &mut PhysicsWorld, r: &Robot, cmd: Command, dt: f32) -> Telemetry {
        let c = [foot_in_contact(w, &r.legs[0]), foot_in_contact(w, &r.legs[1])];
        let before = self.tracker.last_stance;
        self.tracker.update(c, dt);
        if self.tracker.last_stance != before {
            self.steps += 1;
        }
        let obs = observe(w, r, cmd, &self.tracker);
        let act = Action::from_slice(&self.net.predict(&obs));
        apply_action(w, r, &act, self.gait.kd);
        reflexes(w, r, self.gait.wheel_k, self.assist, self.gait.lean + self.gait.lean_v * cmd.speed);
        telemetry(w, r, None)
    }
}

/// DAgger data collection: the teacher labels every visited state; either it or the student drives.
pub struct Recorder {
    pub teacher: RunController,
    pub student: Option<std::sync::Arc<Mlp>>,
    pub tracker: ContactTracker,
    pub data: std::sync::Arc<std::sync::Mutex<Vec<(Vec<f32>, Vec<f32>)>>>,
}

impl Ctl for Recorder {
    fn steps(&self) -> u32 {
        self.teacher.steps
    }
    fn tick(&mut self, w: &mut PhysicsWorld, r: &Robot, cmd: Command, dt: f32) -> Telemetry {
        let c = [foot_in_contact(w, &r.legs[0]), foot_in_contact(w, &r.legs[1])];
        self.tracker.update(c, dt);
        let obs = observe(w, r, cmd, &self.tracker);
        let (label, stance) = self.teacher.compute(w, r, cmd, dt);
        self.data.lock().unwrap().push((obs.to_vec(), label.to_vec().to_vec()));
        let act = match &self.student {
            Some(net) => Action::from_slice(&net.predict(&obs)),
            None => label,
        };
        apply_action(w, r, &act, self.teacher.g.kd);
        let g = self.teacher.g;
        reflexes(w, r, g.wheel_k, self.teacher.assist, g.lean + g.lean_v * cmd.speed);
        telemetry(w, r, stance)
    }
}
