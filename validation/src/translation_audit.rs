//! Resumable exact-source translation census stored in the disposable corpus index.
//!
//! Discovery remembers a successful source across product edits while retrying prior failures with
//! the current translator. A final current-fingerprint sweep remains available to prove that later
//! fixes did not regress earlier successes.

use rusqlite::{params, Connection, Transaction};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

const TRANSLATION_AUDIT_FINGERPRINT: &str = env!("METAL2VULKAN_TRANSLATION_AUDIT_FINGERPRINT");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionMode {
    Discovery,
    CurrentFingerprint,
    RetryCurrentFailures,
    RetryHistoricalLinkage,
    MissingTierCensus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranslationAuditStatus {
    Translated,
    AuthoredLinkageRequired,
    Failed,
}

impl TranslationAuditStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Translated => "translated",
            Self::AuthoredLinkageRequired => "authored_linkage_required",
            Self::Failed => "failed",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranslationAuditResult {
    pub air_sha256: String,
    pub status: TranslationAuditStatus,
    pub failure_shape: Option<String>,
    pub detail: Option<String>,
    #[serde(default)]
    pub adopted_tier: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TranslationAuditSummary {
    pub total_sources: usize,
    pub discovery_covered: usize,
    pub discovery_remaining: usize,
    pub current_attempted: usize,
    pub current_translated: usize,
    pub current_authored_linkage_required: usize,
    pub current_failed: usize,
    pub current_remaining: usize,
}

pub fn select_translation_audit_batch(
    index: &Path,
    mode: SelectionMode,
    after: Option<&str>,
    limit: usize,
) -> Result<Vec<String>, String> {
    let connection = open(index)?;
    ensure_table(&connection)?;
    let prioritize_failures = after.is_none() && mode != SelectionMode::RetryCurrentFailures;
    let sql = selection_sql(mode, prioritize_failures);
    let mut statement = connection
        .prepare(&sql)
        .map_err(|error| format!("prepare translation-audit selection: {error}"))?;
    let hashes = statement
        .query_map(
            params![TRANSLATION_AUDIT_FINGERPRINT, after, limit as i64],
            |row| row.get::<_, String>(0),
        )
        .map_err(|error| format!("query translation-audit selection: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read translation-audit selection: {error}"))?;
    Ok(hashes)
}

fn selection_sql(mode: SelectionMode, prioritize_failures: bool) -> String {
    let predicate = match mode {
        SelectionMode::Discovery => {
            "NOT EXISTS (
               SELECT 1 FROM translation_audit a
               WHERE a.air_sha256=s.air_sha256 AND (
                 a.status IN ('translated', 'authored_linkage_required')
                 OR a.translator_fingerprint=?1
               )
             )"
        }
        SelectionMode::CurrentFingerprint => {
            "NOT EXISTS (
               SELECT 1 FROM translation_audit a
               WHERE a.air_sha256=s.air_sha256 AND a.translator_fingerprint=?1
             )"
        }
        SelectionMode::RetryCurrentFailures => {
            "EXISTS (
               SELECT 1 FROM translation_audit a
               WHERE a.air_sha256=s.air_sha256
                 AND a.translator_fingerprint=?1
                 AND a.status='failed'
             )"
        }
        SelectionMode::RetryHistoricalLinkage => {
            "NOT EXISTS (
               SELECT 1 FROM translation_audit current
               WHERE current.air_sha256=s.air_sha256
                 AND current.translator_fingerprint=?1
             ) AND EXISTS (
               SELECT 1 FROM translation_audit old
               WHERE old.air_sha256=s.air_sha256
                 AND old.status='authored_linkage_required'
             )"
        }
        SelectionMode::MissingTierCensus => {
            "NOT EXISTS (
               SELECT 1 FROM translation_audit a
               WHERE a.air_sha256=s.air_sha256
                 AND a.translator_fingerprint=?1
                 AND a.adopted_tier IS NOT NULL
             )"
        }
    };
    let order = if prioritize_failures {
        "CASE WHEN s.air_sha256 IN (
           SELECT old.air_sha256 FROM translation_audit old WHERE old.status='failed'
         ) THEN 0 ELSE 1 END,
         s.air_sha256"
    } else {
        "s.air_sha256"
    };
    format!(
        "SELECT s.air_sha256 FROM sources s WHERE {predicate}
           AND (?2 IS NULL OR s.air_sha256 > ?2)
         ORDER BY {order}
         LIMIT ?3"
    )
}

pub fn write_translation_audit_results(
    index: &Path,
    results: &[TranslationAuditResult],
) -> Result<(), String> {
    let mut connection = open(index)?;
    ensure_table(&connection)?;
    let transaction = connection
        .transaction()
        .map_err(|error| format!("begin translation-audit update: {error}"))?;
    write_results(&transaction, results)?;
    transaction
        .commit()
        .map_err(|error| format!("commit translation-audit update: {error}"))
}

pub fn translation_audit_summary(index: &Path) -> Result<TranslationAuditSummary, String> {
    let connection = open(index)?;
    ensure_table(&connection)?;
    let count = |sql: &str| -> Result<usize, String> {
        connection
            .query_row(sql, [TRANSLATION_AUDIT_FINGERPRINT], |row| {
                row.get::<_, usize>(0)
            })
            .map_err(|error| format!("query translation-audit summary: {error}"))
    };
    let total_sources = count("SELECT count(*) FROM sources WHERE ?1 IS NOT NULL")?;
    let discovery_covered = count(
        "SELECT count(*) FROM sources s WHERE EXISTS (
           SELECT 1 FROM translation_audit a WHERE a.air_sha256=s.air_sha256 AND (
             a.status IN ('translated', 'authored_linkage_required')
             OR a.translator_fingerprint=?1
           )
         )",
    )?;
    let current_attempted = count(
        "SELECT count(*) FROM sources s WHERE EXISTS (
           SELECT 1 FROM translation_audit a
           WHERE a.air_sha256=s.air_sha256 AND a.translator_fingerprint=?1
         )",
    )?;
    let current_status = |status: &str| -> Result<usize, String> {
        connection
            .query_row(
                "SELECT count(*) FROM sources s JOIN translation_audit a USING (air_sha256)
                 WHERE a.translator_fingerprint=?1 AND a.status=?2",
                params![TRANSLATION_AUDIT_FINGERPRINT, status],
                |row| row.get::<_, usize>(0),
            )
            .map_err(|error| format!("query translation-audit status {status}: {error}"))
    };
    Ok(TranslationAuditSummary {
        total_sources,
        discovery_covered,
        discovery_remaining: total_sources - discovery_covered,
        current_attempted,
        current_translated: current_status("translated")?,
        current_authored_linkage_required: current_status("authored_linkage_required")?,
        current_failed: current_status("failed")?,
        current_remaining: total_sources - current_attempted,
    })
}

pub fn translation_tier_summary(index: &Path) -> Result<BTreeMap<String, usize>, String> {
    let connection = open(index)?;
    ensure_table(&connection)?;
    let mut statement = connection
        .prepare(
            "SELECT adopted_tier, count(*) FROM translation_audit
             WHERE translator_fingerprint=?1 AND adopted_tier IS NOT NULL
             GROUP BY adopted_tier ORDER BY adopted_tier",
        )
        .map_err(|error| format!("prepare translation tier summary: {error}"))?;
    let summary = statement
        .query_map([TRANSLATION_AUDIT_FINGERPRINT], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
        })
        .map_err(|error| format!("query translation tier summary: {error}"))?
        .collect::<Result<BTreeMap<_, _>, _>>()
        .map_err(|error| format!("read translation tier summary: {error}"))?;
    Ok(summary)
}

fn open(index: &Path) -> Result<Connection, String> {
    Connection::open(index).map_err(|error| format!("open index {}: {error}", index.display()))
}

fn ensure_table(connection: &Connection) -> Result<(), String> {
    connection
        .execute_batch(
            "CREATE TABLE IF NOT EXISTS translation_audit (
               air_sha256 TEXT NOT NULL,
               translator_fingerprint TEXT NOT NULL,
               status TEXT NOT NULL CHECK(status IN (
                 'translated', 'authored_linkage_required', 'failed'
               )),
               failure_shape TEXT,
               detail TEXT,
               adopted_tier TEXT,
               PRIMARY KEY(air_sha256, translator_fingerprint)
             );
             CREATE INDEX IF NOT EXISTS translation_audit_current_status
               ON translation_audit(translator_fingerprint, status);",
        )
        .map_err(|error| format!("create translation-audit cache: {error}"))?;
    let has_adopted_tier = connection
        .prepare("PRAGMA table_info(translation_audit)")
        .and_then(|mut statement| {
            statement
                .query_map([], |row| row.get::<_, String>(1))?
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(|error| format!("inspect translation-audit schema: {error}"))?
        .iter()
        .any(|column| column == "adopted_tier");
    if !has_adopted_tier {
        connection
            .execute(
                "ALTER TABLE translation_audit ADD COLUMN adopted_tier TEXT",
                [],
            )
            .map_err(|error| format!("add translation-audit adopted tier: {error}"))?;
    }
    Ok(())
}

fn write_results(
    transaction: &Transaction<'_>,
    results: &[TranslationAuditResult],
) -> Result<(), String> {
    let mut insert = transaction
        .prepare(
            "INSERT INTO translation_audit
               (air_sha256, translator_fingerprint, status, failure_shape, detail, adopted_tier)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(air_sha256, translator_fingerprint) DO UPDATE SET
               status=excluded.status,
               failure_shape=excluded.failure_shape,
               detail=excluded.detail,
               adopted_tier=CASE
                 WHEN excluded.adopted_tier IS NOT NULL THEN excluded.adopted_tier
                 WHEN excluded.status=translation_audit.status THEN translation_audit.adopted_tier
                 ELSE NULL
               END",
        )
        .map_err(|error| format!("prepare translation-audit update: {error}"))?;
    for result in results {
        insert
            .execute(params![
                result.air_sha256,
                TRANSLATION_AUDIT_FINGERPRINT,
                result.status.as_str(),
                result.failure_shape,
                result.detail,
                result.adopted_tier,
            ])
            .map_err(|error| {
                format!(
                    "record translation audit for {}: {error}",
                    result.air_sha256
                )
            })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScratchDir;

    fn index() -> (ScratchDir, std::path::PathBuf) {
        let scratch = ScratchDir::new("translation-audit-cache").unwrap();
        let index = scratch.path().join("index.sqlite");
        let connection = Connection::open(&index).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY);
                 INSERT INTO sources VALUES ('a'), ('b'), ('c');",
            )
            .unwrap();
        (scratch, index)
    }

    #[test]
    fn discovery_skips_any_success_and_current_failures() {
        let (_scratch, index) = index();
        write_translation_audit_results(
            &index,
            &[
                TranslationAuditResult {
                    air_sha256: "a".into(),
                    status: TranslationAuditStatus::Translated,
                    failure_shape: None,
                    detail: None,
                    adopted_tier: Some("default".into()),
                },
                TranslationAuditResult {
                    air_sha256: "b".into(),
                    status: TranslationAuditStatus::Failed,
                    failure_shape: Some("shape".into()),
                    detail: Some("detail".into()),
                    adopted_tier: Some("fallback".into()),
                },
            ],
        )
        .unwrap();
        assert_eq!(
            select_translation_audit_batch(&index, SelectionMode::Discovery, None, 10).unwrap(),
            ["c"]
        );
        let summary = translation_audit_summary(&index).unwrap();
        assert_eq!(summary.total_sources, 3);
        assert_eq!(summary.discovery_covered, 2);
        assert_eq!(summary.current_attempted, 2);
        assert_eq!(summary.current_failed, 1);
    }

    #[test]
    fn after_is_a_strict_hash_order_cursor_even_with_old_failures() {
        let (_scratch, index) = index();
        let connection = Connection::open(&index).unwrap();
        ensure_table(&connection).unwrap();
        connection
            .execute(
                "INSERT INTO translation_audit
                   (air_sha256, translator_fingerprint, status, failure_shape, detail)
                 VALUES ('c', 'old-fingerprint', 'failed', 'old-shape', NULL)",
                [],
            )
            .unwrap();

        // Without a cursor, retrying a prior failure remains the discovery priority.
        assert_eq!(
            select_translation_audit_batch(&index, SelectionMode::Discovery, None, 3).unwrap(),
            ["c", "a", "b"]
        );
        // With a cursor, ordering must be monotonic or page N can skip hashes that sort before an
        // old failure returned on page N-1.
        assert_eq!(
            select_translation_audit_batch(&index, SelectionMode::Discovery, Some("a"), 2).unwrap(),
            ["b", "c"]
        );
    }

    #[test]
    fn historical_failure_priority_is_materialized_once_per_selection() {
        let (_scratch, index) = index();
        let connection = Connection::open(&index).unwrap();
        ensure_table(&connection).unwrap();
        let explain = format!(
            "EXPLAIN QUERY PLAN {}",
            selection_sql(SelectionMode::CurrentFingerprint, true)
        );
        let details = connection
            .prepare(&explain)
            .unwrap()
            .query_map(
                params![TRANSLATION_AUDIT_FINGERPRINT, None::<&str>, 10],
                |row| row.get::<_, String>(3),
            )
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert!(details
            .iter()
            .any(|detail| detail.contains("LIST SUBQUERY")));
        assert_eq!(
            details
                .iter()
                .filter(|detail| detail.contains("CORRELATED SCALAR SUBQUERY"))
                .count(),
            1,
            "{details:?}"
        );
    }

    #[test]
    fn retry_current_failures_selects_only_failed_current_rows() {
        let (_scratch, index) = index();
        write_translation_audit_results(
            &index,
            &[
                TranslationAuditResult {
                    air_sha256: "a".into(),
                    status: TranslationAuditStatus::Translated,
                    failure_shape: None,
                    detail: None,
                    adopted_tier: Some("default".into()),
                },
                TranslationAuditResult {
                    air_sha256: "b".into(),
                    status: TranslationAuditStatus::Failed,
                    failure_shape: Some("shape".into()),
                    detail: Some("detail".into()),
                    adopted_tier: Some("fallback".into()),
                },
            ],
        )
        .unwrap();

        assert_eq!(
            select_translation_audit_batch(&index, SelectionMode::RetryCurrentFailures, None, 10,)
                .unwrap(),
            ["b"]
        );
    }

    #[test]
    fn retry_historical_linkage_is_resumable_under_the_current_fingerprint() {
        let (_scratch, index) = index();
        let connection = Connection::open(&index).unwrap();
        ensure_table(&connection).unwrap();
        connection
            .execute_batch(
                "INSERT INTO translation_audit
                   (air_sha256, translator_fingerprint, status, failure_shape, detail)
                 VALUES
                   ('a', 'old-fingerprint', 'authored_linkage_required', NULL, NULL),
                   ('b', 'old-fingerprint', 'translated', NULL, NULL);",
            )
            .unwrap();

        assert_eq!(
            select_translation_audit_batch(
                &index,
                SelectionMode::RetryHistoricalLinkage,
                None,
                10,
            )
            .unwrap(),
            ["a"]
        );
        write_translation_audit_results(
            &index,
            &[TranslationAuditResult {
                air_sha256: "a".into(),
                status: TranslationAuditStatus::AuthoredLinkageRequired,
                failure_shape: None,
                detail: None,
                adopted_tier: None,
            }],
        )
        .unwrap();
        assert!(select_translation_audit_batch(
            &index,
            SelectionMode::RetryHistoricalLinkage,
            None,
            10,
        )
        .unwrap()
        .is_empty());
    }

    #[test]
    fn tier_census_selects_only_current_rows_without_a_tier_and_summarizes_them() {
        let (_scratch, index) = index();
        write_translation_audit_results(
            &index,
            &[
                TranslationAuditResult {
                    air_sha256: "a".into(),
                    status: TranslationAuditStatus::Translated,
                    failure_shape: None,
                    detail: None,
                    adopted_tier: Some("default".into()),
                },
                TranslationAuditResult {
                    air_sha256: "b".into(),
                    status: TranslationAuditStatus::Translated,
                    failure_shape: None,
                    detail: None,
                    adopted_tier: None,
                },
            ],
        )
        .unwrap();

        assert_eq!(
            select_translation_audit_batch(&index, SelectionMode::MissingTierCensus, None, 10)
                .unwrap(),
            ["b", "c"]
        );
        assert_eq!(
            translation_tier_summary(&index).unwrap(),
            BTreeMap::from([("default".to_string(), 1)])
        );
    }

    #[test]
    fn existing_translation_audit_table_migrates_adopted_tier_in_place() {
        let scratch = ScratchDir::new("translation-audit-tier-migration").unwrap();
        let index = scratch.path().join("index.sqlite");
        let connection = Connection::open(&index).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE translation_audit (
                   air_sha256 TEXT NOT NULL,
                   translator_fingerprint TEXT NOT NULL,
                   status TEXT NOT NULL,
                   failure_shape TEXT,
                   detail TEXT,
                   PRIMARY KEY(air_sha256, translator_fingerprint)
                 );",
            )
            .unwrap();

        ensure_table(&connection).unwrap();

        let columns = connection
            .prepare("PRAGMA table_info(translation_audit)")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(columns.iter().any(|column| column == "adopted_tier"));
    }

    #[test]
    fn unchanged_status_preserves_measured_tier_but_changed_status_clears_it() {
        let (_scratch, index) = index();
        let result = |status, adopted_tier| TranslationAuditResult {
            air_sha256: "a".into(),
            status,
            failure_shape: None,
            detail: None,
            adopted_tier,
        };
        write_translation_audit_results(
            &index,
            &[result(
                TranslationAuditStatus::Translated,
                Some("default".into()),
            )],
        )
        .unwrap();
        write_translation_audit_results(
            &index,
            &[result(TranslationAuditStatus::Translated, None)],
        )
        .unwrap();
        assert_eq!(
            translation_tier_summary(&index).unwrap(),
            BTreeMap::from([("default".to_string(), 1)])
        );

        write_translation_audit_results(&index, &[result(TranslationAuditStatus::Failed, None)])
            .unwrap();
        assert!(translation_tier_summary(&index).unwrap().is_empty());
        assert_eq!(
            select_translation_audit_batch(&index, SelectionMode::MissingTierCensus, None, 10)
                .unwrap(),
            ["a", "b", "c"]
        );
    }
}
