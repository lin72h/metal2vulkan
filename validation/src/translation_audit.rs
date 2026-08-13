//! Resumable exact-source translation census stored in the disposable corpus index.
//!
//! Discovery remembers a successful source across product edits while retrying prior failures with
//! the current translator. A final current-fingerprint sweep remains available to prove that later
//! fixes did not regress earlier successes.

use crate::observation::TRANSLATOR_FINGERPRINT;
use rusqlite::{params, Connection, Transaction};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionMode {
    Discovery,
    CurrentFingerprint,
    RetryCurrentFailures,
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
    };
    let sql = format!(
        "SELECT s.air_sha256 FROM sources s WHERE {predicate}
           AND (?2 IS NULL OR s.air_sha256 > ?2)
         ORDER BY
           CASE WHEN ?2 IS NULL AND EXISTS (
             SELECT 1 FROM translation_audit old
             WHERE old.air_sha256=s.air_sha256 AND old.status='failed'
           ) THEN 0 ELSE 1 END,
           s.air_sha256
         LIMIT ?3"
    );
    let mut statement = connection
        .prepare(&sql)
        .map_err(|error| format!("prepare translation-audit selection: {error}"))?;
    let hashes = statement
        .query_map(
            params![TRANSLATOR_FINGERPRINT, after, limit as i64],
            |row| row.get::<_, String>(0),
        )
        .map_err(|error| format!("query translation-audit selection: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read translation-audit selection: {error}"))?;
    Ok(hashes)
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
            .query_row(sql, [TRANSLATOR_FINGERPRINT], |row| row.get::<_, usize>(0))
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
                params![TRANSLATOR_FINGERPRINT, status],
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
               PRIMARY KEY(air_sha256, translator_fingerprint)
             );
             CREATE INDEX IF NOT EXISTS translation_audit_current_status
               ON translation_audit(translator_fingerprint, status);",
        )
        .map_err(|error| format!("create translation-audit cache: {error}"))
}

fn write_results(
    transaction: &Transaction<'_>,
    results: &[TranslationAuditResult],
) -> Result<(), String> {
    let mut insert = transaction
        .prepare(
            "INSERT INTO translation_audit
               (air_sha256, translator_fingerprint, status, failure_shape, detail)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(air_sha256, translator_fingerprint) DO UPDATE SET
               status=excluded.status,
               failure_shape=excluded.failure_shape,
               detail=excluded.detail",
        )
        .map_err(|error| format!("prepare translation-audit update: {error}"))?;
    for result in results {
        insert
            .execute(params![
                result.air_sha256,
                TRANSLATOR_FINGERPRINT,
                result.status.as_str(),
                result.failure_shape,
                result.detail,
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
                },
                TranslationAuditResult {
                    air_sha256: "b".into(),
                    status: TranslationAuditStatus::Failed,
                    failure_shape: Some("shape".into()),
                    detail: Some("detail".into()),
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
                },
                TranslationAuditResult {
                    air_sha256: "b".into(),
                    status: TranslationAuditStatus::Failed,
                    failure_shape: Some("shape".into()),
                    detail: Some("detail".into()),
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
}
