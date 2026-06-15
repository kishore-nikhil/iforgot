//! Background memory writer.
//!
//! Memory updates (ingesting messages, recording metrics) are taken off
//! the conversation's critical path: the chat thread queues jobs on a
//! channel and returns to the user immediately; a dedicated writer thread
//! owns its own SQLite connection (WAL mode allows readers alongside one
//! writer) plus the Bloom filter and embedding provider, and applies jobs
//! in order.
//!
//! Only retrieval stays synchronous — the prompt needs it. Everything
//! else is eventually consistent within milliseconds, which is invisible
//! at conversation timescales.

use anyhow::Result;
use forgetfuldb_core::config::Config;
use forgetfuldb_store::pipeline::{ingest, warm_bloom, IngestRequest};
use forgetfuldb_store::{ChatTurn, Store};
use std::sync::mpsc;
use std::thread::JoinHandle;

pub enum WriteJob {
    Ingest(IngestRequest),
    RecordTurn(ChatTurn),
    /// Ack when every job queued before this one has been applied.
    /// Used for graceful shutdown and deterministic tests.
    Flush(mpsc::Sender<()>),
}

pub struct MemoryWriter {
    tx: Option<mpsc::Sender<WriteJob>>,
    handle: Option<JoinHandle<()>>,
}

impl MemoryWriter {
    /// Spawn the writer with its own store connection, Bloom filter and
    /// embedding provider.
    pub fn spawn(cfg: &Config) -> Result<MemoryWriter> {
        let cfg = cfg.clone();
        let store = Store::open(std::path::Path::new(&cfg.sqlite_path))?;
        let mut bloom = warm_bloom(&store)?;
        let provider = forgetfuldb_embed::create_provider_from_config(&cfg)?;
        let (tx, rx) = mpsc::channel::<WriteJob>();

        let handle = std::thread::Builder::new()
            .name("forgetfuldb-writer".to_string())
            .spawn(move || {
                for job in rx {
                    let outcome = match job {
                        WriteJob::Ingest(req) => {
                            ingest(&store, &mut bloom, provider.as_ref(), &cfg, req).map(|_| ())
                        }
                        WriteJob::RecordTurn(turn) => store.insert_chat_turn(&turn),
                        WriteJob::Flush(ack) => {
                            let _ = ack.send(());
                            Ok(())
                        }
                    };
                    if let Err(e) = outcome {
                        eprintln!("forgetfuldb-writer: memory update failed: {e:#}");
                    }
                }
            })?;

        Ok(MemoryWriter { tx: Some(tx), handle: Some(handle) })
    }

    pub fn submit(&self, job: WriteJob) {
        if let Some(tx) = &self.tx {
            if tx.send(job).is_err() {
                eprintln!("forgetfuldb-writer: writer thread is gone; memory update dropped");
            }
        }
    }

    /// Block until all previously queued jobs have been applied.
    pub fn flush(&self) {
        if let Some(tx) = &self.tx {
            let (ack_tx, ack_rx) = mpsc::channel();
            if tx.send(WriteJob::Flush(ack_tx)).is_ok() {
                let _ = ack_rx.recv();
            }
        }
    }
}

impl Drop for MemoryWriter {
    /// Closing the channel lets the thread drain remaining jobs and exit;
    /// joining guarantees no memory update is lost on shutdown.
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_cfg(name: &str) -> (Config, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!(
            "forgetfuldb-writer-{name}-{}-{}.sqlite3",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let cfg = Config { sqlite_path: path.display().to_string(), ..Config::default() };
        (cfg, path)
    }

    #[test]
    fn writes_land_after_flush_and_drop_drains_queue() {
        let (cfg, path) = temp_cfg("flush");
        let reader = Store::open(&path).unwrap();

        let writer = MemoryWriter::spawn(&cfg).unwrap();
        writer.submit(WriteJob::Ingest(IngestRequest {
            text: "queued in the background".into(),
            source: Some("chat".into()),
            tags: vec![],
            memory_type: None,
            session_id: None,
            role: Some("user".into()),
        }));
        writer.flush();
        assert_eq!(reader.stats().unwrap().total_memories, 1);

        writer.submit(WriteJob::Ingest(IngestRequest {
            text: "second message before shutdown".into(),
            source: Some("chat".into()),
            tags: vec![],
            memory_type: None,
            session_id: None,
            role: Some("user".into()),
        }));
        drop(writer); // must drain, not lose, the queued job
        assert_eq!(reader.stats().unwrap().total_memories, 2);

        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{}", path.display(), suffix));
        }
    }
}
