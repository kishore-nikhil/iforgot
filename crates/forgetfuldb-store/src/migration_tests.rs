use super::*;
use forgetfuldb_core::ids::new_id;
use std::path::PathBuf;

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(new_id("iforgot-migration", "fixture"));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn table_exists(conn: &Connection, name: &str) -> bool {
    conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1)",
        [name],
        |r| r.get(0),
    )
    .unwrap()
}

#[test]
fn failed_migration_rolls_back_schema_data_and_marker() {
    let store = Store::open_in_memory().unwrap();
    let before = store.stats().unwrap().total_memories;
    let failure = store.apply_migration(
        "test_failure",
        "CREATE TABLE migration_probe (value TEXT);
         INSERT INTO migration_probe VALUES ('partial');
         INSERT INTO missing_table VALUES ('fails');",
    );
    assert!(failure.is_err());
    assert!(!table_exists(&store.conn, "migration_probe"));
    assert_eq!(store.stats().unwrap().total_memories, before);
    let markers: i64 = store
        .conn
        .query_row(
            "SELECT COUNT(*) FROM schema_migrations WHERE name = 'test_failure'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(markers, 0);

    // Retrying the same migration after fixing it succeeds; reopening does
    // not run it twice (the CREATE has deliberately no IF NOT EXISTS).
    store
        .apply_migration("test_failure", "CREATE TABLE migration_probe (value TEXT)")
        .unwrap();
    store
        .apply_migration("test_failure", "CREATE TABLE migration_probe (value TEXT)")
        .unwrap();
    assert!(table_exists(&store.conn, "migration_probe"));
}

#[test]
fn marker_failure_also_rolls_back_schema_changes() {
    let store = Store::open_in_memory().unwrap();
    store
        .conn
        .execute_batch(
            "CREATE TRIGGER reject_marker BEFORE INSERT ON schema_migrations
         WHEN NEW.name = 'test_marker'
         BEGIN SELECT RAISE(ABORT, 'simulated marker failure'); END;",
        )
        .unwrap();
    assert!(store
        .apply_migration("test_marker", "CREATE TABLE marker_probe (value TEXT)")
        .is_err());
    assert!(!table_exists(&store.conn, "marker_probe"));
}

#[test]
fn backed_up_pre_v2_store_upgrades_and_reopens_without_losing_records() {
    let fixture = Fixture::new();
    let path = fixture.0.join("legacy.sqlite3");
    let backup = fixture.0.join("legacy-backup.sqlite3");
    let old = Connection::open(&path).unwrap();
    old.execute_batch(
        "CREATE TABLE schema_migrations (name TEXT PRIMARY KEY, applied_at INTEGER NOT NULL);",
    )
    .unwrap();
    // A reproducible, populated schema-0008 fixture. This is synthetic; it
    // never reads or upgrades the user's global store.
    for (name, sql) in MIGRATIONS.iter().take(8) {
        old.execute_batch(sql).unwrap();
        old.execute("INSERT INTO schema_migrations VALUES (?1, 100)", [name])
            .unwrap();
    }
    old.execute_batch(
        "INSERT INTO memory_items (id, content, memory_type, created_at, updated_at, content_hash, salience)
         VALUES ('old-goal', 'Reduce calls using self-service', 'semantic', 100, 100, 'hash-goal', 0.7),
                ('old-failure', 'Locked accounts cannot finish verification', 'episodic', 110, 110, 'hash-failure', 0.8);
         INSERT INTO memory_links VALUES ('old-failure', 'old-goal', 'supports');
         INSERT INTO memory_edges VALUES ('old-goal', 'old-failure', 'sequence', 0.5, 1, 110, 110);
         INSERT INTO raw_events VALUES ('message-1', 'session-1', 'user', 'Reduce calls using self-service', 100, 'hash-goal');",
    ).unwrap();
    old.execute("VACUUM INTO ?1", [backup.to_str().unwrap()])
        .unwrap();
    drop(old);

    for _ in 0..2 {
        let store = Store::open(&path).unwrap();
        assert_eq!(store.stats().unwrap().total_memories, 2);
        assert_eq!(store.get_memory("old-goal").unwrap().unwrap().salience, 0.7);
        assert_eq!(
            store.get_memory("old-failure").unwrap().unwrap().content,
            "Locked accounts cannot finish verification"
        );
        assert_eq!(store.links_for("old-goal").unwrap().len(), 1);
        assert_eq!(store.list_edges().unwrap().len(), 1);
        assert_eq!(store.raw_events_older_than(200).unwrap().len(), 1);
        assert!(store.all_evidence().unwrap().is_empty());
        let count: usize = store
            .conn
            .query_row("SELECT COUNT(*) FROM schema_migrations", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, MIGRATIONS.len());
        let integrity: String = store
            .conn
            .query_row("PRAGMA integrity_check", [], |r| r.get(0))
            .unwrap();
        assert_eq!(integrity, "ok");
    }
    let untouched =
        Connection::open_with_flags(&backup, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    assert!(!table_exists(&untouched, "memory_evidence"));
    let count: i64 = untouched
        .query_row("SELECT COUNT(*) FROM memory_items", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 2);
}

#[test]
fn memory_and_disk_stores_both_enforce_foreign_keys() {
    let fixture = Fixture::new();
    for store in [
        Store::open_in_memory().unwrap(),
        Store::open(&fixture.0.join("fk.sqlite3")).unwrap(),
    ] {
        store
            .conn
            .execute_batch(
                "CREATE TABLE parent_probe (id TEXT PRIMARY KEY);
             CREATE TABLE child_probe (parent_id TEXT REFERENCES parent_probe(id));",
            )
            .unwrap();
        assert!(store
            .conn
            .execute("INSERT INTO child_probe VALUES ('missing')", [])
            .is_err());
    }
}
