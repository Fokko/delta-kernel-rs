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

//! Stats generation with fixed schema for benchmarking.
//!
//! This module generates statistics with a fixed 10-column schema:
//! - 3 string columns with low cardinality (phonetic alphabet, cities, states)
//! - 6 int64 columns with random values
//! - 1 int64 column with deterministic values (min=max, incrementing)

use rand::rngs::StdRng;
use rand::Rng;
use std::cell::Cell;

/// NATO phonetic alphabet (26 values)
const PHONETIC_ALPHABET: &[&str] = &[
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango",
    "uniform", "victor", "whiskey", "xray", "yankee", "zulu",
];

/// Major US cities (20 values)
const US_CITIES: &[&str] = &[
    "New York",
    "Los Angeles",
    "Chicago",
    "Houston",
    "Phoenix",
    "Philadelphia",
    "San Antonio",
    "San Diego",
    "Dallas",
    "San Jose",
    "Austin",
    "Jacksonville",
    "Fort Worth",
    "Columbus",
    "Charlotte",
    "San Francisco",
    "Indianapolis",
    "Seattle",
    "Denver",
    "Boston",
];

/// US state codes (50 values)
const US_STATES: &[&str] = &[
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS",
    "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY",
];

/// Generated statistics for a single Add action
#[derive(Debug, Clone)]
pub struct GeneratedStats {
    pub num_records: i64,

    // String columns (low cardinality)
    pub phonetic_min: String,
    pub phonetic_max: String,
    pub city_min: String,
    pub city_max: String,
    pub state_min: String,
    pub state_max: String,

    // Random int64 columns (num1-num5)
    pub num1_min: i64,
    pub num1_max: i64,
    pub num2_min: i64,
    pub num2_max: i64,
    pub num3_min: i64,
    pub num3_max: i64,
    pub num4_min: i64,
    pub num4_max: i64,
    pub num5_min: i64,
    pub num5_max: i64,

    // Dollar values (num6) - float64 between 1000.00 and 10000.00
    pub num6_min: f64,
    pub num6_max: f64,

    // Additional random int64 columns (num7-num16)
    pub num7_min: i64,
    pub num7_max: i64,
    pub num8_min: i64,
    pub num8_max: i64,
    pub num9_min: i64,
    pub num9_max: i64,
    pub num10_min: i64,
    pub num10_max: i64,
    pub num11_min: i64,
    pub num11_max: i64,
    pub num12_min: i64,
    pub num12_max: i64,
    pub num13_min: i64,
    pub num13_max: i64,
    pub num14_min: i64,
    pub num14_max: i64,
    pub num15_min: i64,
    pub num15_max: i64,
    pub num16_min: i64,
    pub num16_max: i64,

    // Deterministic column (min == max)
    pub id_value: i64,
}

/// Generator for statistics with fixed schema
pub struct StatsGenerator {
    deterministic_counter: Cell<i64>,
}

impl StatsGenerator {
    /// Create a new StatsGenerator with a starting value for the deterministic column
    pub fn new(deterministic_start: i64) -> Self {
        Self {
            deterministic_counter: Cell::new(deterministic_start),
        }
    }

    /// Generate statistics for a single Add action
    pub fn generate(&self, rng: &mut StdRng) -> GeneratedStats {
        // Generate num_records in range [10000, 20000)
        let num_records = rng.random_range(10_000..20_000);

        // String columns - select random values from predefined arrays
        // Ensure min <= max by comparing actual string values
        let phonetic_val1 = PHONETIC_ALPHABET[rng.random_range(0..PHONETIC_ALPHABET.len())];
        let phonetic_val2 = PHONETIC_ALPHABET[rng.random_range(0..PHONETIC_ALPHABET.len())];
        let (phonetic_min, phonetic_max) = if phonetic_val1 <= phonetic_val2 {
            (phonetic_val1.to_string(), phonetic_val2.to_string())
        } else {
            (phonetic_val2.to_string(), phonetic_val1.to_string())
        };

        let city_val1 = US_CITIES[rng.random_range(0..US_CITIES.len())];
        let city_val2 = US_CITIES[rng.random_range(0..US_CITIES.len())];
        let (city_min, city_max) = if city_val1 <= city_val2 {
            (city_val1.to_string(), city_val2.to_string())
        } else {
            (city_val2.to_string(), city_val1.to_string())
        };

        let state_val1 = US_STATES[rng.random_range(0..US_STATES.len())];
        let state_val2 = US_STATES[rng.random_range(0..US_STATES.len())];
        let (state_min, state_max) = if state_val1 <= state_val2 {
            (state_val1.to_string(), state_val2.to_string())
        } else {
            (state_val2.to_string(), state_val1.to_string())
        };

        // Random int64 columns (num1-num5)
        // For each column, generate min in [-10000, 10000] and max in [min, 100000]
        let num1_min = rng.random_range(-10_000..=10_000);
        let num1_max = rng.random_range(num1_min..=100_000);

        let num2_min = rng.random_range(-10_000..=10_000);
        let num2_max = rng.random_range(num2_min..=100_000);

        let num3_min = rng.random_range(-10_000..=10_000);
        let num3_max = rng.random_range(num3_min..=100_000);

        let num4_min = rng.random_range(-10_000..=10_000);
        let num4_max = rng.random_range(num4_min..=100_000);

        let num5_min = rng.random_range(-10_000..=10_000);
        let num5_max = rng.random_range(num5_min..=100_000);

        // Dollar values (num6) - float64 between 1000.00 and 10000.00
        // Round to 2 decimal places
        let num6_min = (rng.random_range(1000.0_f64..10000.0_f64) * 100.0).round() / 100.0;
        let num6_max = (rng.random_range(num6_min..=10000.0_f64) * 100.0).round() / 100.0;

        // Additional random int64 columns (num7-num16)
        let num7_min = rng.random_range(-10_000..=10_000);
        let num7_max = rng.random_range(num7_min..=100_000);

        let num8_min = rng.random_range(-10_000..=10_000);
        let num8_max = rng.random_range(num8_min..=100_000);

        let num9_min = rng.random_range(-10_000..=10_000);
        let num9_max = rng.random_range(num9_min..=100_000);

        let num10_min = rng.random_range(-10_000..=10_000);
        let num10_max = rng.random_range(num10_min..=100_000);

        let num11_min = rng.random_range(-10_000..=10_000);
        let num11_max = rng.random_range(num11_min..=100_000);

        let num12_min = rng.random_range(-10_000..=10_000);
        let num12_max = rng.random_range(num12_min..=100_000);

        let num13_min = rng.random_range(-10_000..=10_000);
        let num13_max = rng.random_range(num13_min..=100_000);

        let num14_min = rng.random_range(-10_000..=10_000);
        let num14_max = rng.random_range(num14_min..=100_000);

        let num15_min = rng.random_range(-10_000..=10_000);
        let num15_max = rng.random_range(num15_min..=100_000);

        let num16_min = rng.random_range(-10_000..=10_000);
        let num16_max = rng.random_range(num16_min..=100_000);

        // Deterministic column - min == max, incrementing
        let id_value = self.deterministic_counter.get();
        self.deterministic_counter.set(id_value + 1);

        GeneratedStats {
            num_records,
            phonetic_min,
            phonetic_max,
            city_min,
            city_max,
            state_min,
            state_max,
            num1_min,
            num1_max,
            num2_min,
            num2_max,
            num3_min,
            num3_max,
            num4_min,
            num4_max,
            num5_min,
            num5_max,
            num6_min,
            num6_max,
            num7_min,
            num7_max,
            num8_min,
            num8_max,
            num9_min,
            num9_max,
            num10_min,
            num10_max,
            num11_min,
            num11_max,
            num12_min,
            num12_max,
            num13_min,
            num13_max,
            num14_min,
            num14_max,
            num15_min,
            num15_max,
            num16_min,
            num16_max,
            id_value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_reproducible_with_seed() {
        let gen = StatsGenerator::new(0);
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);

        let stats1 = gen.generate(&mut rng1);

        let gen2 = StatsGenerator::new(0);
        let stats2 = gen2.generate(&mut rng2);

        assert_eq!(stats1.num_records, stats2.num_records);
        assert_eq!(stats1.phonetic_min, stats2.phonetic_min);
        assert_eq!(stats1.city_min, stats2.city_min);
        assert_eq!(stats1.state_min, stats2.state_min);
        assert_eq!(stats1.num1_min, stats2.num1_min);
    }

    #[test]
    fn test_deterministic_column_increments() {
        let gen = StatsGenerator::new(100);
        let mut rng = StdRng::seed_from_u64(42);

        let stats1 = gen.generate(&mut rng);
        let stats2 = gen.generate(&mut rng);
        let stats3 = gen.generate(&mut rng);

        assert_eq!(stats1.id_value, 100);
        assert_eq!(stats2.id_value, 101);
        assert_eq!(stats3.id_value, 102);
    }

    #[test]
    fn test_num_records_in_range() {
        let gen = StatsGenerator::new(0);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let stats = gen.generate(&mut rng);
            assert!(stats.num_records >= 10_000);
            assert!(stats.num_records < 20_000);
        }
    }

    #[test]
    fn test_min_max_relationship() {
        let gen = StatsGenerator::new(0);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let stats = gen.generate(&mut rng);
            // String columns should be lexicographically ordered
            assert!(stats.phonetic_min <= stats.phonetic_max);
            assert!(stats.city_min <= stats.city_max);
            assert!(stats.state_min <= stats.state_max);
            // Numeric columns
            assert!(stats.num1_min <= stats.num1_max);
            assert!(stats.num2_min <= stats.num2_max);
            assert!(stats.num3_min <= stats.num3_max);
            assert!(stats.num4_min <= stats.num4_max);
            assert!(stats.num5_min <= stats.num5_max);
            assert!(stats.num6_min <= stats.num6_max);
        }
    }

    #[test]
    fn test_string_values_in_arrays() {
        let gen = StatsGenerator::new(0);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let stats = gen.generate(&mut rng);
            assert!(PHONETIC_ALPHABET.contains(&stats.phonetic_min.as_str()));
            assert!(PHONETIC_ALPHABET.contains(&stats.phonetic_max.as_str()));
            assert!(US_CITIES.contains(&stats.city_min.as_str()));
            assert!(US_CITIES.contains(&stats.city_max.as_str()));
            assert!(US_STATES.contains(&stats.state_min.as_str()));
            assert!(US_STATES.contains(&stats.state_max.as_str()));
        }
    }
}
