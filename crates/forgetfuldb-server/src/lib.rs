//! forgetfuldb-server
//!
//! Optional local HTTP API. Binds to 127.0.0.1 only when
//! `local_only = true` (the default) — ForgetfulDB is a private memory,
//! not a network service.
//!
//! Routes:
//! - `POST /ingest`       {"text", "source?", "tags?", "memory_type?", "session_id?", "role?"}
//! - `POST /retrieve`     {"query", "top_k?", "include_stale?"}
//! - `POST /consolidate`  {}
//! - `GET  /memory/:id`
//! - `GET  /stats`

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use forgetfuldb_consolidate::ExtractiveSummarizer;
use forgetfuldb_core::config::Config;
use forgetfuldb_core::types::MemoryType;
use forgetfuldb_embed::EmbeddingProvider;
use forgetfuldb_retrieve::RetrieveOptions;
use forgetfuldb_store::pipeline::{ingest, warm_bloom, IngestRequest};
use forgetfuldb_store::Store;
use forgetfuldb_prob::BloomFilter;
use serde::Deserialize;
use serde_json::json;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

/// Shared, mutex-guarded state. SQLite connections are not Sync, and a
/// single-writer model is exactly what a personal memory store needs.
struct AppState {
    store: Store,
    bloom: BloomFilter,
    provider: Box<dyn EmbeddingProvider>,
    cfg: Config,
}

type SharedState = Arc<Mutex<AppState>>;

/// Anyhow-friendly error type that renders as JSON.
struct ApiError(anyhow::Error);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, Json(json!({ "error": self.0.to_string() }))).into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(e: E) -> Self {
        ApiError(e.into())
    }
}

#[derive(Deserialize)]
struct IngestBody {
    text: String,
    source: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    memory_type: Option<String>,
    session_id: Option<String>,
    role: Option<String>,
}

#[derive(Deserialize)]
struct RetrieveBody {
    query: String,
    top_k: Option<usize>,
    #[serde(default)]
    include_stale: bool,
}

async fn ingest_handler(
    State(state): State<SharedState>,
    Json(body): Json<IngestBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mut app = state.lock().expect("state mutex poisoned");
    let memory_type = body
        .memory_type
        .as_deref()
        .map(MemoryType::from_str)
        .transpose()
        .map_err(|e| anyhow::anyhow!(e))?;
    let req = IngestRequest {
        text: body.text,
        source: body.source,
        tags: body.tags,
        memory_type,
        session_id: body.session_id,
        role: body.role,
    };
    let AppState { store, bloom, provider, cfg } = &mut *app;
    let outcome = ingest(store, bloom, provider.as_ref(), cfg, req)?;
    Ok(Json(json!({
        "duplicate": outcome.is_duplicate(),
        "memory": outcome.memory(),
    })))
}

async fn retrieve_handler(
    State(state): State<SharedState>,
    Json(body): Json<RetrieveBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let app = state.lock().expect("state mutex poisoned");
    let opts = RetrieveOptions {
        top_k: body.top_k.unwrap_or(10),
        include_stale: body.include_stale,
        include_archived: false,
    };
    let pack = forgetfuldb_retrieve::retrieve(&app.store, app.provider.as_ref(), &app.cfg, &body.query, &opts)?;
    Ok(Json(serde_json::to_value(pack)?))
}

async fn consolidate_handler(State(state): State<SharedState>) -> Result<Json<serde_json::Value>, ApiError> {
    let app = state.lock().expect("state mutex poisoned");
    let report = forgetfuldb_consolidate::consolidate(&app.store, &ExtractiveSummarizer::default(), &app.cfg)?;
    Ok(Json(serde_json::to_value(report)?))
}

async fn memory_handler(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let app = state.lock().expect("state mutex poisoned");
    match app.store.get_memory(&id)? {
        Some(item) => {
            let links = app.store.links_for(&id)?;
            Ok(Json(json!({ "memory": item, "links": links })))
        }
        None => Err(anyhow::anyhow!("memory not found: {id}").into()),
    }
}

async fn stats_handler(State(state): State<SharedState>) -> Result<Json<serde_json::Value>, ApiError> {
    let app = state.lock().expect("state mutex poisoned");
    let stats = app.store.stats()?;
    Ok(Json(json!({
        "total_memories": stats.total_memories,
        "by_type": stats.by_type.iter().cloned().collect::<std::collections::BTreeMap<String, i64>>(),
        "stale": stats.stale,
        "pinned": stats.pinned,
        "raw_events": stats.raw_events,
        "links": stats.links,
        "sessions": stats.sessions,
    })))
}

fn build_router(state: SharedState) -> Router {
    Router::new()
        .route("/ingest", post(ingest_handler))
        .route("/retrieve", post(retrieve_handler))
        .route("/consolidate", post(consolidate_handler))
        .route("/memory/:id", get(memory_handler))
        .route("/stats", get(stats_handler))
        .with_state(state)
}

/// Open the store and serve the API. Blocks until shutdown (ctrl-c).
pub async fn serve(cfg: Config, port: u16) -> anyhow::Result<()> {
    let store = Store::open(std::path::Path::new(&cfg.sqlite_path))?;
    let bloom = warm_bloom(&store)?;
    let provider = forgetfuldb_embed::create_provider(&cfg.embedding_backend, cfg.embedding_dim)?;
    let host = if cfg.local_only { "127.0.0.1" } else { "0.0.0.0" };
    let state: SharedState = Arc::new(Mutex::new(AppState { store, bloom, provider, cfg }));

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("forgetfuldb-server listening on http://{addr}");
    axum::serve(listener, build_router(state)).await?;
    Ok(())
}
