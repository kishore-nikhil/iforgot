//! forgetfuldb-server
//!
//! Optional local HTTP API. Binds to 127.0.0.1 only when
//! `local_only = true` (the default) — ForgetfulDB is a private memory,
//! not a network service.
//!
//! Routes:
//! - `POST /ingest`       {"text", "source?", "tags?", "memory_type?", "session_id?", "role?"}
//! - `POST /retrieve`     {"query", "top_k?", "include_stale?", "debug?",
//!   "min_score?", "memory_types?", "since?", "until?"}
//! - `POST /consolidate`  {}
//! - `GET  /graph`        ?since&until&types=csv&limit — nodes+edges for the UI
//! - `GET  /uiconfig`     decay lambdas, retrieval weights, chat knobs
//! - `GET  /turns`        ?limit — recent chat_turns rows (oldest first)
//! - `GET  /consolidations` ?limit — logged consolidation runs
//! - `GET  /memory/:id`
//! - `POST /memory/:id/pin`     {"pinned": bool}
//! - `POST /memory/:id/archive`
//! - `GET  /stats`
//! - `GET  /metrics`      aggregate chat-turn token/context metrics
//! - `GET  /ui`           the observability SPA (when built/mounted)
//! - `POST /v1/chat/completions`  OpenAI-compatible memory proxy (see
//!   [`proxy`]): point any OpenAI-shaped chat UI here and it gains
//!   automatic long-term memory.

pub mod proxy;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::{Stream, StreamExt};
use forgetfuldb_consolidate::ExtractiveSummarizer;
use forgetfuldb_core::config::Config;
use forgetfuldb_core::types::MemoryType;
use forgetfuldb_core::{age_days, decay, now_unix};
use forgetfuldb_embed::EmbeddingProvider;
use forgetfuldb_retrieve::RetrieveOptions;
use forgetfuldb_store::pipeline::{ingest, warm_bloom, IngestRequest};
use forgetfuldb_store::Store;
use forgetfuldb_prob::BloomFilter;
use serde::Deserialize;
use serde_json::json;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use tower_http::services::{ServeDir, ServeFile};

/// Shared, mutex-guarded state. SQLite connections are not Sync, and a
/// single-writer model is exactly what a personal memory store needs.
/// Handlers must not hold the lock across an await (std MutexGuard is
/// !Send); the proxy releases it during LLM generation.
pub(crate) struct AppState {
    pub(crate) store: Store,
    pub(crate) bloom: BloomFilter,
    pub(crate) provider: Box<dyn EmbeddingProvider>,
    pub(crate) cfg: Config,
    /// Plain-HTTP client for forwarding to the local LLM (cheap to clone).
    pub(crate) http_client: reqwest::Client,
    /// Tools the server knows about (listed by `/tools`, executed by
    /// `/tools/execute` only when `tools.allow_server_execute` is set).
    pub(crate) tools: forgetfuldb_tools::ToolRegistry,
    /// Broadcast channel for Server-Sent Events: a "change" is pushed
    /// whenever the store is modified, so the UI updates instantly instead
    /// of blind-polling.
    pub(crate) events: tokio::sync::broadcast::Sender<String>,
}

impl AppState {
    /// Notify SSE subscribers that the store changed (best-effort).
    pub(crate) fn notify_change(&self) {
        let _ = self.events.send("change".to_string());
    }
}

pub(crate) type SharedState = Arc<Mutex<AppState>>;

/// Anyhow-friendly error type that renders as JSON.
pub(crate) struct ApiError(anyhow::Error);

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
    /// Debug mode mirrors the chat path (config gate + damping) and also
    /// returns near-misses with score breakdowns.
    #[serde(default)]
    debug: bool,
    /// Override the relevance gate (debug defaults it from config).
    min_score: Option<f64>,
    /// Restrict to these memory types (names as in the schema).
    memory_types: Option<Vec<String>>,
    since: Option<i64>,
    until: Option<i64>,
}

fn parse_types(names: &[String]) -> Result<Option<Vec<MemoryType>>, ApiError> {
    if names.is_empty() {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(names.len());
    for n in names {
        out.push(MemoryType::from_str(n.trim()).map_err(|e| anyhow::anyhow!(e))?);
    }
    Ok(Some(out))
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
    let events = app.events.clone();
    let AppState { store, bloom, provider, cfg, .. } = &mut *app;
    let outcome = ingest(store, bloom, provider.as_ref(), cfg, req)?;
    let body = json!({
        "duplicate": outcome.is_duplicate(),
        "memory": outcome.memory(),
    });
    let _ = events.send("change".to_string());
    Ok(Json(body))
}

async fn retrieve_handler(
    State(state): State<SharedState>,
    Json(body): Json<RetrieveBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let app = state.lock().expect("state mutex poisoned");
    // Debug mode mirrors what the chat path would actually inject (config
    // gate + conversational damping); plain mode keeps the historical
    // "return everything" behavior for scripts.
    let (default_min, damping) = if body.debug {
        (app.cfg.chat.min_retrieval_score, app.cfg.chat.conversational_damping)
    } else {
        (0.0, 1.0)
    };
    let opts = RetrieveOptions {
        top_k: body.top_k.unwrap_or(app.cfg.chat.top_k.max(10)),
        include_stale: body.include_stale,
        min_score: body.min_score.unwrap_or(default_min),
        conversational_damping: damping,
        memory_types: parse_types(&body.memory_types.unwrap_or_default())?,
        since: body.since,
        until: body.until,
        debug: body.debug,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let pack = forgetfuldb_retrieve::retrieve(&app.store, app.provider.as_ref(), &app.cfg, &body.query, &opts)?;
    let mut value = serde_json::to_value(pack)?;
    value["retrieve_ms"] = json!(t0.elapsed().as_millis() as i64);
    Ok(Json(value))
}

async fn consolidate_handler(State(state): State<SharedState>) -> Result<Json<serde_json::Value>, ApiError> {
    let app = state.lock().expect("state mutex poisoned");
    let report = forgetfuldb_consolidate::consolidate(&app.store, &ExtractiveSummarizer::default(), &app.cfg)?;
    app.notify_change();
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

async fn metrics_handler(State(state): State<SharedState>) -> Result<Json<serde_json::Value>, ApiError> {
    let app = state.lock().expect("state mutex poisoned");
    let m = app.store.chat_metrics_summary()?;
    Ok(Json(json!({
        "turns": m.turns,
        "avg_prompt_tokens": m.avg_prompt_tokens,
        "avg_completion_tokens": m.avg_completion_tokens,
        "total_prompt_tokens": m.total_prompt_tokens,
        "total_completion_tokens": m.total_completion_tokens,
        "avg_context_chars": m.avg_context_chars,
        "avg_context_memories": m.avg_context_memories,
        "avg_retrieve_ms": m.avg_retrieve_ms,
        "avg_llm_ms": m.avg_llm_ms,
    })))
}

async fn tools_handler(State(state): State<SharedState>) -> Result<Json<serde_json::Value>, ApiError> {
    let app = state.lock().expect("state mutex poisoned");
    Ok(Json(json!({
        "tools": app.tools.list(),
        // Tell clients up front whether the server will actually run a tool.
        "execution_enabled": app.cfg.tools.allow_server_execute,
    })))
}

#[derive(Deserialize)]
struct ToolExecuteBody {
    tool: String,
    #[serde(default)]
    args: serde_json::Value,
}

async fn tools_execute_handler(
    State(state): State<SharedState>,
    Json(body): Json<ToolExecuteBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let app = state.lock().expect("state mutex poisoned");
    // An HTTP endpoint can't ask a human to confirm, so executing tools
    // here is a remote shell. Refuse unless the operator explicitly
    // opted in on a trusted local machine.
    if !app.cfg.tools.allow_server_execute {
        return Err(anyhow::anyhow!(
            "tool execution over HTTP is disabled. Set tools.allow_server_execute = true to enable \
             it (this lets clients run shell commands on this machine without confirmation)."
        )
        .into());
    }
    let call = forgetfuldb_tools::ToolCall { tool: body.tool, args: body.args };
    let output = app.tools.execute(&call)?;
    Ok(Json(json!({ "output": output })))
}

/// Hard cap on graph nodes per response: force layout degrades beyond
/// this, so the server keeps the strongest (highest live decay) memories.
const GRAPH_NODE_CAP: usize = 500;

#[derive(Deserialize)]
struct GraphQuery {
    since: Option<i64>,
    until: Option<i64>,
    /// CSV of memory type names.
    types: Option<String>,
    limit: Option<usize>,
}

async fn graph_handler(
    State(state): State<SharedState>,
    Query(q): Query<GraphQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let app = state.lock().expect("state mutex poisoned");
    let now = now_unix();
    // Default window: the last 30 days.
    let since = q.since.unwrap_or(now - 30 * 86_400);
    let until = q.until.unwrap_or(i64::MAX);
    let types: Option<Vec<MemoryType>> = match &q.types {
        None => None,
        Some(csv) => {
            let mut parsed = Vec::new();
            for part in csv.split(',').filter(|s| !s.trim().is_empty()) {
                parsed.push(MemoryType::from_str(part.trim()).map_err(|e| anyhow::anyhow!(e))?);
            }
            if parsed.is_empty() { None } else { Some(parsed) }
        }
    };
    let cap = q.limit.unwrap_or(GRAPH_NODE_CAP).min(GRAPH_NODE_CAP);
    let lambdas = app.cfg.decay_lambdas();

    let mut nodes: Vec<(f64, serde_json::Value)> = Vec::new();
    let mut total_count = 0usize;
    for item in app.store.list_memories(None)? {
        if item.created_at < since || item.created_at > until {
            continue;
        }
        if types.as_ref().is_some_and(|t| !t.contains(&item.memory_type)) {
            continue;
        }
        total_count += 1;
        // Decay as of *now*, not the stale stored column.
        let live_decay = decay::decay_score(
            item.importance_score,
            lambdas.for_type(item.memory_type),
            age_days(item.created_at, now),
            item.pinned,
        );
        let preview: String = item.content.chars().take(120).collect();
        nodes.push((
            live_decay,
            json!({
                "id": item.id,
                "content_preview": preview,
                "memory_type": item.memory_type.as_str(),
                "importance_score": item.importance_score,
                "decay_score": live_decay,
                "recurrence_score": item.recurrence_score,
                "pinned": item.pinned,
                "stale": item.stale,
                "salience": item.salience,
                "created_at": item.created_at,
                "last_accessed_at": item.last_accessed_at,
                "tags": item.tags,
                "topic": item.topic,
            }),
        ));
    }
    nodes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    nodes.truncate(cap);
    let kept: std::collections::HashSet<String> =
        nodes.iter().map(|(_, n)| n["id"].as_str().unwrap_or_default().to_string()).collect();

    // Two edge sources: typed provenance/dedup links (memory_links, weight
    // 1.0) and weighted association edges (memory_edges, e.g. co_occurred).
    let mut edges: Vec<serde_json::Value> = app
        .store
        .all_links()?
        .into_iter()
        .filter(|l| kept.contains(&l.source_id) && kept.contains(&l.target_id))
        .map(|l| {
            json!({
                "src_id": l.source_id,
                "dst_id": l.target_id,
                "edge_type": l.relation.as_str(),
                "weight": 1.0,
            })
        })
        .collect();
    edges.extend(
        app.store
            .list_edges()?
            .into_iter()
            .filter(|e| kept.contains(&e.src_id) && kept.contains(&e.dst_id))
            .map(|e| {
                json!({
                    "src_id": e.src_id,
                    "dst_id": e.dst_id,
                    "edge_type": e.edge_type,
                    "weight": e.weight,
                    "co_count": e.co_count,
                })
            }),
    );

    Ok(Json(json!({
        "nodes": nodes.into_iter().map(|(_, n)| n).collect::<Vec<_>>(),
        "edges": edges,
        "total_count": total_count,
        "window": { "since": since, "until": if until == i64::MAX { None } else { Some(until) } },
        "generated_at": now,
    })))
}

/// Read-only config slice the UI needs: decay lambdas for client-side
/// scrubbing, plus the retrieval knobs the inspector mirrors.
async fn uiconfig_handler(State(state): State<SharedState>) -> Result<Json<serde_json::Value>, ApiError> {
    let app = state.lock().expect("state mutex poisoned");
    let lambdas = app.cfg.decay_lambdas();
    let w = &app.cfg.retrieval_weights;
    let db_size = std::fs::metadata(&app.cfg.sqlite_path).map(|m| m.len()).unwrap_or(0);
    Ok(Json(json!({
        "name": app.cfg.name,
        "db_path": app.cfg.sqlite_path,
        "db_size_bytes": db_size,
        "decay_lambdas": {
            "raw_event": lambdas.raw_event,
            "episodic": lambdas.episodic,
            "semantic": lambdas.semantic,
            "procedural": lambdas.procedural,
            "preference": lambdas.preference,
            "archive": lambdas.archive,
        },
        "retrieval_weights": {
            "semantic": w.semantic,
            "importance": w.importance,
            "recurrence": w.recurrence,
            "recency": w.recency,
            "pinned_boost": w.pinned_boost,
            "staleness_penalty": w.staleness_penalty,
        },
        "chat": {
            "top_k": app.cfg.chat.top_k,
            "min_retrieval_score": app.cfg.chat.min_retrieval_score,
            "conversational_damping": app.cfg.chat.conversational_damping,
        },
        // The embedding identity the retrieval inspector is actually using
        // (the transparency the inspector was missing), plus the salience knobs.
        "embedding": {
            "backend": app.cfg.embedding_backend,
            "model": app.cfg.embedding_model,
            "dim": app.provider.dim(),
        },
        "salience": {
            "resist": app.cfg.salience_resist,
            "keep_threshold": app.cfg.salience_keep_threshold,
            "spreading_activation": app.cfg.spreading_activation,
        },
    })))
}

#[derive(Deserialize)]
struct LimitQuery {
    limit: Option<usize>,
}

async fn turns_handler(
    State(state): State<SharedState>,
    Query(q): Query<LimitQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let app = state.lock().expect("state mutex poisoned");
    let turns = app.store.list_chat_turns(q.limit.unwrap_or(200).min(2000))?;
    Ok(Json(json!({ "turns": turns })))
}

async fn consolidations_handler(
    State(state): State<SharedState>,
    Query(q): Query<LimitQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let app = state.lock().expect("state mutex poisoned");
    let runs = app.store.list_consolidation_runs(q.limit.unwrap_or(20).min(200))?;
    Ok(Json(json!({ "runs": runs })))
}

#[derive(Deserialize)]
struct PinBody {
    pinned: bool,
}

async fn pin_handler(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Json(body): Json<PinBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let app = state.lock().expect("state mutex poisoned");
    if !app.store.set_pinned(&id, body.pinned)? {
        return Err(anyhow::anyhow!("memory not found: {id}").into());
    }
    app.notify_change();
    Ok(Json(json!({ "id": id, "pinned": body.pinned })))
}

async fn archive_handler(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let app = state.lock().expect("state mutex poisoned");
    if !app.store.set_memory_type(&id, MemoryType::Archive)? {
        return Err(anyhow::anyhow!("memory not found: {id}").into());
    }
    app.notify_change();
    Ok(Json(json!({ "id": id, "memory_type": "archive" })))
}

/// The observability UI embedded into the binary at build time (see
/// build.rs). Lets `forgetfuldb server` serve `/ui` from any directory
/// with no `--ui` path. Compiled out entirely when the `embed-ui` feature
/// is off or `ui/dist` wasn't built.
#[cfg(embed_ui)]
mod embedded_ui {
    use axum::http::{header, HeaderValue, StatusCode, Uri};
    use axum::response::{IntoResponse, Response};
    use include_dir::{include_dir, Dir};

    static UI: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/../../ui/dist");

    pub(crate) async fn handler(uri: Uri) -> Response {
        let rel = uri.path().trim_start_matches("/ui").trim_start_matches('/');
        // SPA fallback: unknown sub-paths serve index.html.
        let file = if rel.is_empty() {
            UI.get_file("index.html")
        } else {
            UI.get_file(rel).or_else(|| UI.get_file("index.html"))
        };
        match file {
            Some(f) => {
                let ext = f.path().extension().and_then(|e| e.to_str()).unwrap_or("");
                (
                    [(header::CONTENT_TYPE, HeaderValue::from_static(content_type(ext)))],
                    f.contents().to_vec(),
                )
                    .into_response()
            }
            None => (StatusCode::NOT_FOUND, "embedded UI missing").into_response(),
        }
    }

    fn content_type(ext: &str) -> &'static str {
        match ext {
            "html" => "text/html; charset=utf-8",
            "js" | "mjs" => "text/javascript; charset=utf-8",
            "css" => "text/css; charset=utf-8",
            "json" | "map" => "application/json",
            "svg" => "image/svg+xml",
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "ico" => "image/x-icon",
            "woff2" => "font/woff2",
            "woff" => "font/woff",
            "ttf" => "font/ttf",
            _ => "application/octet-stream",
        }
    }
}

/// Is an embedded UI compiled into this binary?
pub fn ui_is_embedded() -> bool {
    cfg!(embed_ui)
}

/// `GET /events` — Server-Sent Events stream of `change` notifications, so
/// the UI refreshes the instant the store changes (from this server or from
/// a separate `iforgot` process) instead of blind-polling.
async fn events_handler(
    State(state): State<SharedState>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let rx = state.lock().expect("state mutex poisoned").events.subscribe();
    let stream = BroadcastStream::new(rx)
        .map(|msg| Ok(Event::default().data(msg.unwrap_or_else(|_| "change".to_string()))));
    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn build_router(state: SharedState, ui_dir: Option<&std::path::Path>) -> Router {
    let mut router = Router::new()
        .route("/ingest", post(ingest_handler))
        .route("/retrieve", post(retrieve_handler))
        .route("/consolidate", post(consolidate_handler))
        .route("/graph", get(graph_handler))
        .route("/uiconfig", get(uiconfig_handler))
        .route("/turns", get(turns_handler))
        .route("/consolidations", get(consolidations_handler))
        .route("/memory/:id", get(memory_handler))
        .route("/memory/:id/pin", post(pin_handler))
        .route("/memory/:id/archive", post(archive_handler))
        .route("/stats", get(stats_handler))
        .route("/metrics", get(metrics_handler))
        .route("/events", get(events_handler))
        .route("/tools", get(tools_handler))
        .route("/tools/execute", post(tools_execute_handler))
        .route("/v1/chat/completions", post(proxy::chat_completions));
    if let Some(dir) = ui_dir {
        // Explicit/disk UI (dev or --ui override): served from the
        // filesystem so rebuilds show up without recompiling the binary.
        // SPA fallback: unknown paths under /ui get index.html.
        let serve = ServeDir::new(dir).fallback(ServeFile::new(dir.join("index.html")));
        router = router.nest_service("/ui", serve);
    } else {
        // No path given: serve the UI baked into the binary, if any.
        // Three routes so "/ui", "/ui/" (trailing slash) and "/ui/<asset>"
        // all reach the handler — `/ui/*rest` alone misses the bare "/ui/".
        #[cfg(embed_ui)]
        {
            router = router
                .route("/ui", get(embedded_ui::handler))
                .route("/ui/", get(embedded_ui::handler))
                .route("/ui/*rest", get(embedded_ui::handler));
        }
    }
    router.with_state(state)
}

/// Open the store and serve the API. Blocks until shutdown (ctrl-c).
/// `ui_dir`, when set, mounts the built observability SPA at `/ui`.
pub async fn serve(cfg: Config, port: u16, ui_dir: Option<std::path::PathBuf>) -> anyhow::Result<()> {
    forgetfuldb_agent::backend::ensure_local_url(&cfg)?;
    let store = Store::open(std::path::Path::new(&cfg.sqlite_path))?;
    let bloom = warm_bloom(&store)?;
    let provider = forgetfuldb_embed::create_provider_from_config(&cfg)?;
    let host = if cfg.local_only { "127.0.0.1" } else { "0.0.0.0" };
    let tools = forgetfuldb_tools::ToolRegistry::from_config(&cfg.tools);
    let (events, _) = tokio::sync::broadcast::channel::<String>(64);
    let state: SharedState = Arc::new(Mutex::new(AppState {
        store,
        bloom,
        provider,
        cfg,
        http_client: reqwest::Client::new(),
        tools,
        events: events.clone(),
    }));

    // Watch for writes made by *other* connections (a separate iforgot
    // process chatting against the same store) and push an SSE event so the
    // UI updates live. Server-originated writes notify explicitly from their
    // handlers; this catches everything else. PRAGMA data_version is a
    // microsecond read, so a 500ms poll is cheap.
    {
        let poll_state = state.clone();
        let poll_tx = events.clone();
        tokio::spawn(async move {
            let mut last: i64 = -1;
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                let ver = match poll_state.lock() {
                    Ok(app) => app.store.data_version().unwrap_or(last),
                    Err(_) => last,
                };
                if ver != last {
                    if last != -1 {
                        let _ = poll_tx.send("change".to_string());
                    }
                    last = ver;
                }
            }
        });
    }

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("forgetfuldb-server listening on http://{addr}");
    match &ui_dir {
        Some(dir) => eprintln!("observability UI at http://{addr}/ui (serving {})", dir.display()),
        None if ui_is_embedded() => eprintln!("observability UI at http://{addr}/ui (embedded)"),
        None => eprintln!("(no UI: build ui/dist and pass --ui, or install with the embed-ui feature)"),
    }
    axum::serve(listener, build_router(state, ui_dir.as_deref())).await?;
    Ok(())
}
