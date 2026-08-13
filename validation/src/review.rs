use crate::source::shard_index_for_hash;
use serde::{Deserialize, Serialize};

/// A durable human/agent queue annotation, never semantic or execution evidence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewNote {
    pub air_sha256: String,
    pub reason: String,
    pub reviewed_by: String,
}

impl ReviewNote {
    pub fn validate(&self) -> Result<(), String> {
        shard_index_for_hash(&self.air_sha256)?;
        if self.reason.trim().is_empty() {
            return Err("review reason must not be empty".into());
        }
        if self.reviewed_by.trim().is_empty() {
            return Err("reviewed_by must not be empty".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn review_notes_require_explicit_content() {
        let mut note = ReviewNote {
            air_sha256: "11".repeat(32),
            reason: "requires an unsupported texture shape".into(),
            reviewed_by: "test".into(),
        };
        assert!(note.validate().is_ok());
        note.reason.clear();
        assert!(note.validate().is_err());
    }
}
