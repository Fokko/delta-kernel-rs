// Copyright 2025 The Delta Kernel Rust Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Add action generation for benchmarking.
//!
//! Combines stats generation and deletion vector generation to create
//! complete Add action metadata.

use crate::deletion_vector::DeletionVectorGenerator;
use crate::stats::{GeneratedStats, StatsGenerator};
use delta_kernel::actions::deletion_vector::DeletionVectorDescriptor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use uuid::Uuid;

/// Metadata for a single Add action
#[derive(Debug, Clone)]
pub struct AddActionMetadata {
    pub path: String,
    pub size: i64,
    pub modification_time: i64,
    pub stats: GeneratedStats,
    pub deletion_vector: Option<DeletionVectorDescriptor>,
}

/// Generate a vector of Add action metadata
///
/// # Arguments
/// * `num_actions` - Number of Add actions to generate
/// * `dv_probability` - Probability of generating deletion vectors (0.0-1.0)
/// * `deterministic_start` - Starting value for the deterministic stats column
/// * `seed` - Optional random seed for reproducibility
pub fn generate_add_actions(
    num_actions: usize,
    dv_probability: f64,
    deterministic_start: i64,
    seed: Option<u64>,
) -> Vec<AddActionMetadata> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_os_rng(),
    };

    let stats_gen = StatsGenerator::new(deterministic_start);
    let dv_gen = DeletionVectorGenerator::new();

    (0..num_actions)
        .map(|i| {
            // Generate UUID-based file path
            let uuid = Uuid::new_v4();
            let path = format!("part-{:05}-{}.snappy.parquet", i, uuid);

            // Generate size in bytes [4MB, 8MB)
            let size = rng.random_range(4_000_000..8_000_000);

            // Current timestamp in milliseconds
            let modification_time = chrono::Utc::now().timestamp_millis();

            // Generate stats
            let stats = stats_gen.generate(&mut rng);

            // Generate deletion vector based on probability
            let deletion_vector = if rng.random::<f64>() < dv_probability {
                Some(dv_gen.generate(&mut rng, stats.num_records))
            } else {
                None
            };

            AddActionMetadata {
                path,
                size,
                modification_time,
                stats,
                deletion_vector,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_add_actions_reproducible() {
        let actions1 = generate_add_actions(10, 0.3, 0, Some(42));
        let actions2 = generate_add_actions(10, 0.3, 0, Some(42));

        assert_eq!(actions1.len(), actions2.len());
        for (a1, a2) in actions1.iter().zip(actions2.iter()) {
            assert_eq!(a1.size, a2.size);
            assert_eq!(a1.stats.num_records, a2.stats.num_records);
            assert_eq!(a1.stats.phonetic_min, a2.stats.phonetic_min);
        }
    }

    #[test]
    fn test_generate_correct_count() {
        let actions = generate_add_actions(50, 0.0, 0, Some(42));
        assert_eq!(actions.len(), 50);
    }

    #[test]
    fn test_size_in_range() {
        let actions = generate_add_actions(100, 0.0, 0, Some(42));
        for action in actions {
            assert!(action.size >= 4_000_000);
            assert!(action.size < 8_000_000);
        }
    }

    #[test]
    fn test_dv_probability() {
        let actions = generate_add_actions(1000, 0.3, 0, Some(42));
        let dv_count = actions
            .iter()
            .filter(|a| a.deletion_vector.is_some())
            .count();
        let ratio = dv_count as f64 / 1000.0;

        // Should be roughly 30% with some variance
        assert!(ratio > 0.25 && ratio < 0.35, "DV ratio: {}", ratio);
    }

    #[test]
    fn test_no_dvs_when_probability_zero() {
        let actions = generate_add_actions(100, 0.0, 0, Some(42));
        assert!(actions.iter().all(|a| a.deletion_vector.is_none()));
    }

    #[test]
    fn test_all_dvs_when_probability_one() {
        let actions = generate_add_actions(100, 1.0, 0, Some(42));
        assert!(actions.iter().all(|a| a.deletion_vector.is_some()));
    }

    #[test]
    fn test_deterministic_counter_increments() {
        let actions = generate_add_actions(5, 0.0, 100, Some(42));

        assert_eq!(actions[0].stats.id_value, 100);
        assert_eq!(actions[1].stats.id_value, 101);
        assert_eq!(actions[2].stats.id_value, 102);
        assert_eq!(actions[3].stats.id_value, 103);
        assert_eq!(actions[4].stats.id_value, 104);
    }

    #[test]
    fn test_path_format() {
        let actions = generate_add_actions(10, 0.0, 0, Some(42));
        for (i, action) in actions.iter().enumerate() {
            assert!(action.path.starts_with(&format!("part-{:05}-", i)));
            assert!(action.path.ends_with(".snappy.parquet"));
        }
    }
}
