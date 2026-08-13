use crate::case::AuthoredCase;
use crate::hash::sha256_bytes;
use base64::Engine as _;
use serde::{Deserialize, Serialize};

pub const TRANSLATOR_FINGERPRINT: &str = env!("METAL2VULKAN_PRODUCT_FINGERPRINT");

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetalObservation {
    pub case_id: String,
    pub air_sha256: String,
    pub input_sha256: String,
    pub metal_output_sha256: String,
    pub output_b64: String,
    pub environment_id: String,
    pub environment: serde_json::Value,
    pub oracle_abi: String,
    pub status: MetalStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetalStatus {
    Qualified,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateObservation {
    pub case_id: String,
    pub air_sha256: String,
    pub input_sha256: String,
    pub golden_output_sha256: String,
    pub spv_sha256: String,
    #[serde(default)]
    pub translator_fingerprint: String,
    pub candidate_output_sha256: String,
    pub output_b64: String,
    pub backend: Backend,
    pub environment_id: String,
    pub environment: serde_json::Value,
    pub executor_abi: String,
    pub comparison: ComparisonResult,
    pub status: CandidateStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Moltenvk,
    Vulkan,
}

impl Backend {
    pub fn directory(self) -> &'static str {
        match self {
            Self::Moltenvk => "moltenvk",
            Self::Vulkan => "vulkan",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonResult {
    Exact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateStatus {
    Match,
    Mismatch,
}

impl MetalObservation {
    pub fn validate_content(&self) -> Result<(), String> {
        validate_hash("case_id", &self.case_id)?;
        validate_hash("air_sha256", &self.air_sha256)?;
        validate_hash("input_sha256", &self.input_sha256)?;
        validate_hash("metal_output_sha256", &self.metal_output_sha256)?;
        validate_slot_fields(&self.environment_id, &self.oracle_abi)?;
        if !self.environment.is_object() {
            return Err("Metal environment must be a JSON object".into());
        }
        let output = base64::engine::general_purpose::STANDARD
            .decode(&self.output_b64)
            .map_err(|error| format!("invalid Metal output_b64: {error}"))?;
        if output.is_empty() {
            return Err("Metal output_b64 must not be empty".into());
        }
        let computed = sha256_bytes(&output);
        if computed != self.metal_output_sha256 {
            return Err(format!(
                "Metal output hash mismatch: row={} computed={computed}",
                self.metal_output_sha256
            ));
        }
        Ok(())
    }

    pub fn dependency_matches(&self, case: &AuthoredCase, oracle_abi: &str) -> bool {
        self.validate_content().is_ok()
            && self.case_id == case.case_id
            && self.air_sha256 == case.air_sha256
            && case
                .computed_input_sha256()
                .is_ok_and(|digest| digest == self.input_sha256)
            && self.oracle_abi == oracle_abi
            && self.status == MetalStatus::Qualified
    }
}

pub struct CandidateDependencies<'a> {
    pub case: &'a AuthoredCase,
    pub metal: &'a MetalObservation,
    pub spv_sha256: &'a str,
    pub translator_fingerprint: &'a str,
    pub backend: Backend,
    pub environment_id: &'a str,
    pub executor_abi: &'a str,
}

impl CandidateObservation {
    pub fn validate_content(&self) -> Result<(), String> {
        for (field, value) in [
            ("case_id", self.case_id.as_str()),
            ("air_sha256", self.air_sha256.as_str()),
            ("input_sha256", self.input_sha256.as_str()),
            ("golden_output_sha256", self.golden_output_sha256.as_str()),
            ("spv_sha256", self.spv_sha256.as_str()),
            (
                "candidate_output_sha256",
                self.candidate_output_sha256.as_str(),
            ),
        ] {
            validate_hash(field, value)?;
        }
        if !self.translator_fingerprint.is_empty() {
            validate_hash("translator_fingerprint", &self.translator_fingerprint)?;
        }
        validate_slot_fields(&self.environment_id, &self.executor_abi)?;
        if !self.environment.is_object() {
            return Err("candidate environment must be a JSON object".into());
        }
        let output = base64::engine::general_purpose::STANDARD
            .decode(&self.output_b64)
            .map_err(|error| format!("invalid candidate output_b64: {error}"))?;
        if output.is_empty() {
            return Err("candidate output_b64 must not be empty".into());
        }
        let computed = sha256_bytes(&output);
        if computed != self.candidate_output_sha256 {
            return Err(format!(
                "candidate output hash mismatch: row={} computed={computed}",
                self.candidate_output_sha256
            ));
        }
        let hashes_match = self.candidate_output_sha256 == self.golden_output_sha256;
        if (self.status == CandidateStatus::Match) != hashes_match {
            return Err(format!(
                "candidate status {:?} contradicts candidate/golden output hashes",
                self.status
            ));
        }
        Ok(())
    }

    pub fn dependency_matches(&self, dependencies: &CandidateDependencies<'_>) -> bool {
        self.validate_content().is_ok()
            && dependencies.metal.validate_content().is_ok()
            && self.case_id == dependencies.case.case_id
            && self.air_sha256 == dependencies.case.air_sha256
            && self.input_sha256 == dependencies.metal.input_sha256
            && self.golden_output_sha256 == dependencies.metal.metal_output_sha256
            && self.spv_sha256 == dependencies.spv_sha256
            && self.translator_fingerprint == dependencies.translator_fingerprint
            && self.backend == dependencies.backend
            && self.environment_id == dependencies.environment_id
            && self.executor_abi == dependencies.executor_abi
    }
}

fn validate_hash(field: &str, value: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{field} must be 64 hexadecimal characters"));
    }
    if value.bytes().any(|byte| byte.is_ascii_uppercase()) {
        return Err(format!("{field} must use lowercase hexadecimal"));
    }
    Ok(())
}

fn validate_slot_fields(environment_id: &str, abi: &str) -> Result<(), String> {
    if environment_id.trim().is_empty() {
        return Err("environment_id must not be empty".into());
    }
    if abi.trim().is_empty() {
        return Err("executor/oracle ABI must not be empty".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{
        AuthoredCase, BufferResource, Comparison, Dispatch, ExecutionSafety, OutputSelection,
        ResourceRole, Stage,
    };

    fn case() -> AuthoredCase {
        let mut case = AuthoredCase {
            air_sha256: "11".repeat(32),
            case_id: String::new(),
            name: "case".into(),
            entry: "main".into(),
            stage: Stage::Kernel,
            buffers: vec![BufferResource {
                binding: 0,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some("AAAAAA==".into()),
            }],
            argument_buffer_buffers: vec![],
            threadgroup_memory: vec![],
            imageblock: None,
            fragment_imageblock: None,
            acceleration_structures: vec![],
            visible_function_references: vec![],
            visible_function_tables: vec![],
            intersection_function_tables: vec![],
            argument_buffer_intersection_function_tables: vec![],
            textures: vec![],
            texture_arrays: vec![],
            argument_buffer_textures: vec![],
            samplers: vec![],
            render_targets: vec![],
            depth_stencil: None,
            vertex_inputs: vec![],
            vertex_observation: None,
            kernel_stage_inputs: vec![],
            function_constants: vec![],
            dispatch: Some(Dispatch {
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            }),
            draw: None,
            tessellation: None,
            output: OutputSelection::Buffer {
                binding: 0,
                offset: 0,
                length: 4,
            },
            compare: Comparison::Exact,
            execution_safety: ExecutionSafety::LoopFree,
            rationale: None,
            authored_by: None,
        };
        case.case_id = case.computed_case_id().unwrap();
        case
    }

    #[test]
    fn every_candidate_dependency_participates_in_reuse() {
        let case = case();
        let input_sha256 = case.computed_input_sha256().unwrap();
        let output_sha256 = sha256_bytes(&[1, 0, 0, 0]);
        let metal = MetalObservation {
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: input_sha256.clone(),
            metal_output_sha256: output_sha256.clone(),
            output_b64: "AQAAAA==".into(),
            environment_id: "metal-env".into(),
            environment: serde_json::json!({"device": "test"}),
            oracle_abi: "oracle-v1".into(),
            status: MetalStatus::Qualified,
        };
        assert!(metal.dependency_matches(&case, "oracle-v1"));
        assert!(!metal.dependency_matches(&case, "oracle-v2"));

        let candidate = CandidateObservation {
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256,
            golden_output_sha256: metal.metal_output_sha256.clone(),
            spv_sha256: "33".repeat(32),
            translator_fingerprint: "55".repeat(32),
            candidate_output_sha256: output_sha256,
            output_b64: metal.output_b64.clone(),
            backend: Backend::Vulkan,
            environment_id: "vk-env".into(),
            environment: serde_json::json!({"driver": "test"}),
            executor_abi: "executor-v1".into(),
            comparison: ComparisonResult::Exact,
            status: CandidateStatus::Match,
        };
        let spv_sha256 = "33".repeat(32);
        let dependencies = CandidateDependencies {
            case: &case,
            metal: &metal,
            spv_sha256: &spv_sha256,
            translator_fingerprint: &candidate.translator_fingerprint,
            backend: Backend::Vulkan,
            environment_id: "vk-env",
            executor_abi: "executor-v1",
        };
        assert!(candidate.dependency_matches(&dependencies));

        let changed_spv = "44".repeat(32);
        let changed = CandidateDependencies {
            spv_sha256: &changed_spv,
            ..dependencies
        };
        assert!(!candidate.dependency_matches(&changed));
        let changed_fingerprint = "66".repeat(32);
        let changed = CandidateDependencies {
            translator_fingerprint: &changed_fingerprint,
            ..dependencies
        };
        assert!(!candidate.dependency_matches(&changed));
        let changed = CandidateDependencies {
            backend: Backend::Moltenvk,
            ..dependencies
        };
        assert!(!candidate.dependency_matches(&changed));
        let changed = CandidateDependencies {
            environment_id: "new-driver",
            ..dependencies
        };
        assert!(!candidate.dependency_matches(&changed));
        let changed = CandidateDependencies {
            executor_abi: "executor-v2",
            ..dependencies
        };
        assert!(!candidate.dependency_matches(&changed));

        let mut changed_metal = metal.clone();
        changed_metal.metal_output_sha256 = sha256_bytes(&[2, 0, 0, 0]);
        changed_metal.output_b64 = "AgAAAA==".into();
        let changed = CandidateDependencies {
            metal: &changed_metal,
            ..dependencies
        };
        assert!(!candidate.dependency_matches(&changed));

        let mut changed_case = case.clone();
        changed_case.buffers[0].initial_bytes_b64 = Some("AQAAAA==".into());
        changed_case.case_id = changed_case.computed_case_id().unwrap();
        let changed = CandidateDependencies {
            case: &changed_case,
            ..dependencies
        };
        assert!(!candidate.dependency_matches(&changed));
    }

    #[test]
    fn observation_payload_hash_and_status_are_checked() {
        let case = case();
        let output_sha256 = sha256_bytes(&[1, 0, 0, 0]);
        let mut row = CandidateObservation {
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: case.computed_input_sha256().unwrap(),
            golden_output_sha256: output_sha256.clone(),
            spv_sha256: "33".repeat(32),
            translator_fingerprint: "55".repeat(32),
            candidate_output_sha256: output_sha256,
            output_b64: "AQAAAA==".into(),
            backend: Backend::Vulkan,
            environment_id: "vk-env".into(),
            environment: serde_json::json!({}),
            executor_abi: "executor-v1".into(),
            comparison: ComparisonResult::Exact,
            status: CandidateStatus::Match,
        };
        assert!(row.validate_content().is_ok());
        row.output_b64 = "AgAAAA==".into();
        assert!(row
            .validate_content()
            .unwrap_err()
            .contains("hash mismatch"));
    }
}
