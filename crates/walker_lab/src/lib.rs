//! Library view of the walker lab (robot builds, controllers, simulation), shared with armor_lab.
#![allow(dead_code)]
pub mod controller;
pub mod policy;
pub mod ppo;
pub mod robot;
pub mod robot_v;
pub mod run_ctl;
pub mod sim;
pub mod thrust;
pub use sim::{DT, SUBSTEPS};
