//! Print seeded patch draws as JSON lines, for comparison with 1.x.
//!
//! `cargo run --example sampling_parity -- FILE STRATEGY SEED N PATCH [WEIGHTS]`

use std::path::Path;

use medh5::rng::SeededRng;
use medh5::sampling::{grid_patches, ClassWeights, PatchSampler, PatchSize, TimepointPairSampler};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sample = medh5::sample::open_sample(Path::new(&args[0])).expect("open");
    let strategy = &args[1];
    let seed: u64 = args[2].parse().unwrap();
    let n: usize = args[3].parse().unwrap();
    let patch: i64 = args[4].parse().unwrap();
    let weights = args.get(5).cloned().unwrap_or_else(|| "uniform".into());
    let sampler = PatchSampler::new(PatchSize::Scalar(patch), strategy, 0.5, None, ClassWeights::Named(weights))
        .expect("sampler");
    let mut rng = SeededRng::new(seed);
    for _ in 0..n {
        match sampler.draw(&sample, None, &mut rng, None) {
            Ok(p) => println!("{}", medh5::json::canonical(&p.to_json())),
            Err(e) => {
                println!("ERROR {}", e.python_line());
                break;
            }
        }
    }
    let shape: Vec<i64> = sample.reference_grid().unwrap().spatial_shape().iter().map(|v| *v as i64).collect();
    let grid = grid_patches(&shape, &PatchSize::Scalar(patch), 2.min(patch - 1), Some("g")).unwrap();
    println!("grid {}", grid.len());
    for p in grid.iter().take(3) {
        println!("{}", medh5::json::canonical(&p.to_json()));
    }
    for mode in ["consecutive", "baseline_vs_all", "all_pairs"] {
        let pairs = TimepointPairSampler::new(mode).unwrap().pairs(&sample).unwrap();
        println!("{mode} {}", pairs.iter().map(|p| p.repr()).collect::<Vec<_>>().join(";"));
    }
}
