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

//! Deletion vector generation for benchmarking.
//!
//! Generates three types of deletion vectors:
//! - Inline: RoaringTreemap serialized and z85 encoded
//! - Relative: UUID-based relative paths
//! - Absolute: Fake S3 paths

use delta_kernel::actions::deletion_vector::{DeletionVectorDescriptor, DeletionVectorStorageType};
use rand::rngs::StdRng;
use rand::Rng;
use roaring::RoaringTreemap;
use uuid::Uuid;

/// Magic number for portable RoaringBitmap serialization format
#[allow(dead_code)]
const ROARING_BITMAP_PORTABLE_MAGIC: u32 = 1681511377;

/// Generator for deletion vectors
pub struct DeletionVectorGenerator;

impl DeletionVectorGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate a deletion vector with random storage type
    /// Always uses relative path DVs (most common in production)
    pub fn generate(&self, rng: &mut StdRng, num_records: i64) -> DeletionVectorDescriptor {
        // Cardinality: 5-15% of total records, minimum 10
        let cardinality = (num_records as f64 * rng.random_range(0.05..0.15)) as i64;
        let cardinality = cardinality.max(10);

        // Always use relative path DVs (most common and efficient in production)
        self.generate_relative_dv(rng, cardinality)
    }

    /// Generate with explicit storage type preference (for testing)
    #[allow(dead_code)]
    pub fn generate_with_type_distribution(
        &self,
        rng: &mut StdRng,
        num_records: i64,
        inline_prob: f64,
        relative_prob: f64,
    ) -> DeletionVectorDescriptor {
        let cardinality = (num_records as f64 * rng.random_range(0.05..0.15)) as i64;
        let cardinality = cardinality.max(10);

        let rand_val: f64 = rng.random();
        if rand_val < inline_prob {
            self.generate_inline_dv(rng, cardinality, num_records)
        } else if rand_val < inline_prob + relative_prob {
            self.generate_relative_dv(rng, cardinality)
        } else {
            self.generate_absolute_dv(rng, cardinality)
        }
    }

    /// Generate an inline deletion vector with actual RoaringTreemap serialization
    #[allow(dead_code)]
    fn generate_inline_dv(
        &self,
        rng: &mut StdRng,
        cardinality: i64,
        num_records: i64,
    ) -> DeletionVectorDescriptor {
        // Create a RoaringTreemap with random row IDs
        let mut bitmap = RoaringTreemap::new();
        let mut added = 0;

        // Insert unique random row IDs
        while added < cardinality {
            let row_id = rng.random_range(0..num_records as u64);
            if bitmap.insert(row_id) {
                added += 1;
            }
        }

        // Serialize with portable magic number
        let mut buffer = Vec::new();
        buffer.extend_from_slice(&ROARING_BITMAP_PORTABLE_MAGIC.to_le_bytes());
        bitmap.serialize_into(&mut buffer).unwrap();

        // Encode with z85
        let encoded = z85::encode(&buffer);
        let size = buffer.len();

        DeletionVectorDescriptor {
            storage_type: DeletionVectorStorageType::Inline,
            path_or_inline_dv: encoded,
            offset: None,
            size_in_bytes: size as i32,
            cardinality,
        }
    }

    /// Generate a relative deletion vector with UUID-based path
    fn generate_relative_dv(&self, rng: &mut StdRng, cardinality: i64) -> DeletionVectorDescriptor {
        // Generate a random prefix (0-2 hex characters)
        let prefix = format!("{:02x}", rng.random::<u8>());

        // Generate UUID and encode in base85
        let uuid = Uuid::new_v4();
        let uuid_bytes = uuid.as_bytes();
        let uuid_encoded = z85::encode(uuid_bytes);

        // The encoded path is prefix + base85 encoded UUID
        // Note: The actual UUID decoding happens in kernel code
        let path_or_inline_dv = format!("{}{}", prefix, uuid_encoded);

        DeletionVectorDescriptor {
            storage_type: DeletionVectorStorageType::PersistedRelative,
            path_or_inline_dv,
            offset: Some(rng.random_range(0..1000)),
            size_in_bytes: rng.random_range(100..1000),
            cardinality,
        }
    }

    /// Generate an absolute deletion vector with fake S3 path
    #[allow(dead_code)]
    fn generate_absolute_dv(&self, rng: &mut StdRng, cardinality: i64) -> DeletionVectorDescriptor {
        let uuid = Uuid::new_v4();

        DeletionVectorDescriptor {
            storage_type: DeletionVectorStorageType::PersistedAbsolute,
            path_or_inline_dv: format!("s3://benchmark-bucket/dvs/{}.bin", uuid),
            offset: Some(0),
            size_in_bytes: rng.random_range(100..5000),
            cardinality,
        }
    }
}

impl Default for DeletionVectorGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_inline_dv_format() {
        let gen = DeletionVectorGenerator::new();
        let mut rng = StdRng::seed_from_u64(42);
        let dv = gen.generate_inline_dv(&mut rng, 100, 10000);

        assert_eq!(dv.storage_type, DeletionVectorStorageType::Inline);
        assert!(dv.offset.is_none());
        assert!(dv.cardinality > 0);
        assert!(dv.size_in_bytes > 0);
        // Inline DVs should be z85 encoded
        assert!(!dv.path_or_inline_dv.is_empty());
    }

    #[test]
    fn test_relative_dv_format() {
        let gen = DeletionVectorGenerator::new();
        let mut rng = StdRng::seed_from_u64(42);
        let dv = gen.generate_relative_dv(&mut rng, 100);

        assert_eq!(
            dv.storage_type,
            DeletionVectorStorageType::PersistedRelative
        );
        assert!(dv.offset.is_some());
        assert_eq!(dv.cardinality, 100);
        assert!(dv.size_in_bytes >= 100 && dv.size_in_bytes < 1000);
        // Should have prefix + base85 encoded UUID
        assert!(dv.path_or_inline_dv.len() > 20);
    }

    #[test]
    fn test_absolute_dv_format() {
        let gen = DeletionVectorGenerator::new();
        let mut rng = StdRng::seed_from_u64(42);
        let dv = gen.generate_absolute_dv(&mut rng, 100);

        assert_eq!(
            dv.storage_type,
            DeletionVectorStorageType::PersistedAbsolute
        );
        assert_eq!(dv.offset, Some(0));
        assert_eq!(dv.cardinality, 100);
        assert!(dv
            .path_or_inline_dv
            .starts_with("s3://benchmark-bucket/dvs/"));
        assert!(dv.path_or_inline_dv.ends_with(".bin"));
    }

    #[test]
    fn test_cardinality_range() {
        let gen = DeletionVectorGenerator::new();
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let dv = gen.generate(&mut rng, 10000);
            // Cardinality should be 5-15% of 10000 = 500-1500
            assert!(dv.cardinality >= 10); // Minimum
            assert!(dv.cardinality <= 1500); // Maximum
        }
    }

    #[test]
    fn test_storage_type_distribution() {
        let gen = DeletionVectorGenerator::new();
        let mut rng = StdRng::seed_from_u64(42);

        // Test that generate() always uses relative path DVs (most common in production)
        for _ in 0..100 {
            let dv = gen.generate(&mut rng, 10000);
            assert_eq!(
                dv.storage_type,
                DeletionVectorStorageType::PersistedRelative
            );
        }

        // Test generate_with_type_distribution for varied types
        let mut inline_count = 0;
        let mut relative_count = 0;
        let mut absolute_count = 0;

        for _ in 0..1000 {
            let dv = gen.generate_with_type_distribution(&mut rng, 10000, 0.4, 0.4);
            match dv.storage_type {
                DeletionVectorStorageType::Inline => inline_count += 1,
                DeletionVectorStorageType::PersistedRelative => relative_count += 1,
                DeletionVectorStorageType::PersistedAbsolute => absolute_count += 1,
            }
        }

        // Should be roughly 40%/40%/20% distribution
        // Allow some variance
        assert!(inline_count > 300 && inline_count < 500);
        assert!(relative_count > 300 && relative_count < 500);
        assert!(absolute_count > 100 && absolute_count < 300);
    }
}
