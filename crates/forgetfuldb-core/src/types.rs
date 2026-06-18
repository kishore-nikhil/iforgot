//! The memory schema: strongly typed enums and records that mirror the
//! SQLite tables in `forgetfuldb-store`.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// What kind of memory an item is. Stored as a lowercase snake_case string
/// in SQLite so rows stay human-readable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    /// Verbatim event (chat turn, log line). Decays fast.
    RawEvent,
    /// "What happened" — an experience tied to a moment in time.
    Episodic,
    /// "What is true" — distilled facts, decays slowly.
    Semantic,
    /// "How to do things" — workflows and commands, decays slowly.
    Procedural,
    /// User preferences ("I like dark mode").
    Preference,
    /// A decay-exempt trait *concluded* by consolidation from accumulated
    /// habit evidence ("initiated tic-tac-toe 4× over 3 months → likes
    /// games"). The identity layer: like a pin, it never decays and is never
    /// pruned, but it is reached automatically, not set by hand.
    Foundation,
    /// Compressed/retired memory kept for the record, excluded from
    /// normal retrieval.
    Archive,
}

impl MemoryType {
    pub const ALL: [MemoryType; 7] = [
        MemoryType::RawEvent,
        MemoryType::Episodic,
        MemoryType::Semantic,
        MemoryType::Procedural,
        MemoryType::Preference,
        MemoryType::Foundation,
        MemoryType::Archive,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryType::RawEvent => "raw_event",
            MemoryType::Episodic => "episodic",
            MemoryType::Semantic => "semantic",
            MemoryType::Procedural => "procedural",
            MemoryType::Preference => "preference",
            MemoryType::Foundation => "foundation",
            MemoryType::Archive => "archive",
        }
    }

    /// Decay-exempt by type: a Foundation trait never fades, mirroring a pin
    /// but reached by consolidation rather than set by hand. Used everywhere
    /// decay or pruning would otherwise erode a memory.
    pub fn is_decay_exempt(&self) -> bool {
        matches!(self, MemoryType::Foundation)
    }
}

impl FromStr for MemoryType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "raw_event" => Ok(MemoryType::RawEvent),
            "episodic" => Ok(MemoryType::Episodic),
            "semantic" => Ok(MemoryType::Semantic),
            "procedural" => Ok(MemoryType::Procedural),
            "preference" => Ok(MemoryType::Preference),
            "foundation" => Ok(MemoryType::Foundation),
            "archive" => Ok(MemoryType::Archive),
            other => Err(format!("unknown memory type: {other}")),
        }
    }
}

impl fmt::Display for MemoryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Typed relation between two memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkRelation {
    Supports,
    Contradicts,
    Updates,
    Duplicates,
    DerivedFrom,
    BelongsToProject,
}

impl LinkRelation {
    pub fn as_str(&self) -> &'static str {
        match self {
            LinkRelation::Supports => "supports",
            LinkRelation::Contradicts => "contradicts",
            LinkRelation::Updates => "updates",
            LinkRelation::Duplicates => "duplicates",
            LinkRelation::DerivedFrom => "derived_from",
            LinkRelation::BelongsToProject => "belongs_to_project",
        }
    }
}

impl FromStr for LinkRelation {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "supports" => Ok(LinkRelation::Supports),
            "contradicts" => Ok(LinkRelation::Contradicts),
            "updates" => Ok(LinkRelation::Updates),
            "duplicates" => Ok(LinkRelation::Duplicates),
            "derived_from" => Ok(LinkRelation::DerivedFrom),
            "belongs_to_project" => Ok(LinkRelation::BelongsToProject),
            other => Err(format!("unknown link relation: {other}")),
        }
    }
}

impl fmt::Display for LinkRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Lifecycle state for a memory candidate/trace. These states are mostly
/// explanatory today; consolidation can use them later to drive promotion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryLifecycleState {
    RawInput,
    CandidateEvidence,
    WeakMemory,
    ReinforcedMemory,
    ConsolidatedDurable,
    Decayed,
    Archived,
    Forgotten,
}

impl MemoryLifecycleState {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryLifecycleState::RawInput => "raw_input",
            MemoryLifecycleState::CandidateEvidence => "candidate_evidence",
            MemoryLifecycleState::WeakMemory => "weak_memory",
            MemoryLifecycleState::ReinforcedMemory => "reinforced_memory",
            MemoryLifecycleState::ConsolidatedDurable => "consolidated_durable",
            MemoryLifecycleState::Decayed => "decayed",
            MemoryLifecycleState::Archived => "archived",
            MemoryLifecycleState::Forgotten => "forgotten",
        }
    }
}

/// Evidence that can change retention/consolidation. Plain retrieval is
/// intentionally absent as a reinforcement signal; only accepted/successful
/// reuse should strengthen a memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceType {
    ExplicitRememberRequest,
    UserConfirmation,
    UserCorrection,
    TopicRepeated,
    CrossSessionRecurrence,
    CrossDayRecurrence,
    RetrievalSuccess,
    RetrievalFailure,
    SessionThemeSupport,
    GraphClusterSupport,
    ConnectedMemoryCreated,
    LlmAuditorSupport,
}

impl EvidenceType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EvidenceType::ExplicitRememberRequest => "explicit_remember_request",
            EvidenceType::UserConfirmation => "user_confirmation",
            EvidenceType::UserCorrection => "user_correction",
            EvidenceType::TopicRepeated => "topic_repeated",
            EvidenceType::CrossSessionRecurrence => "cross_session_recurrence",
            EvidenceType::CrossDayRecurrence => "cross_day_recurrence",
            EvidenceType::RetrievalSuccess => "retrieval_success",
            EvidenceType::RetrievalFailure => "retrieval_failure",
            EvidenceType::SessionThemeSupport => "session_theme_support",
            EvidenceType::GraphClusterSupport => "graph_cluster_support",
            EvidenceType::ConnectedMemoryCreated => "connected_memory_created",
            EvidenceType::LlmAuditorSupport => "llm_auditor_support",
        }
    }
}

impl FromStr for EvidenceType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "explicit_remember_request" => Ok(EvidenceType::ExplicitRememberRequest),
            "user_confirmation" => Ok(EvidenceType::UserConfirmation),
            "user_correction" => Ok(EvidenceType::UserCorrection),
            "topic_repeated" => Ok(EvidenceType::TopicRepeated),
            "cross_session_recurrence" => Ok(EvidenceType::CrossSessionRecurrence),
            "cross_day_recurrence" => Ok(EvidenceType::CrossDayRecurrence),
            "retrieval_success" => Ok(EvidenceType::RetrievalSuccess),
            "retrieval_failure" => Ok(EvidenceType::RetrievalFailure),
            "session_theme_support" => Ok(EvidenceType::SessionThemeSupport),
            "graph_cluster_support" => Ok(EvidenceType::GraphClusterSupport),
            "connected_memory_created" => Ok(EvidenceType::ConnectedMemoryCreated),
            "llm_auditor_support" => Ok(EvidenceType::LlmAuditorSupport),
            other => Err(format!("unknown evidence type: {other}")),
        }
    }
}

impl fmt::Display for EvidenceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceSource {
    DeterministicExtractor,
    User,
    RetrievalFeedback,
    Consolidation,
    LlmAuditor,
}

impl EvidenceSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            EvidenceSource::DeterministicExtractor => "deterministic_extractor",
            EvidenceSource::User => "user",
            EvidenceSource::RetrievalFeedback => "retrieval_feedback",
            EvidenceSource::Consolidation => "consolidation",
            EvidenceSource::LlmAuditor => "llm_auditor",
        }
    }
}

impl FromStr for EvidenceSource {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "deterministic_extractor" => Ok(EvidenceSource::DeterministicExtractor),
            "user" => Ok(EvidenceSource::User),
            "retrieval_feedback" => Ok(EvidenceSource::RetrievalFeedback),
            "consolidation" => Ok(EvidenceSource::Consolidation),
            "llm_auditor" => Ok(EvidenceSource::LlmAuditor),
            other => Err(format!("unknown evidence source: {other}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvidence {
    pub id: String,
    pub memory_id: String,
    pub evidence_type: EvidenceType,
    pub strength: f64,
    pub source: EvidenceSource,
    pub session_id: Option<String>,
    pub created_at: i64,
}

/// Coarse input class used before memory extraction. Long or code/log-heavy
/// messages become source documents instead of a burst of durable memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputMode {
    ConversationMessage,
    MixedMessage,
    PastedDocument,
    CodeBlock,
    LogDump,
    ArticleDraft,
    ReferenceMaterial,
}

impl InputMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            InputMode::ConversationMessage => "conversation_message",
            InputMode::MixedMessage => "mixed_message",
            InputMode::PastedDocument => "pasted_document",
            InputMode::CodeBlock => "code_block",
            InputMode::LogDump => "log_dump",
            InputMode::ArticleDraft => "article_draft",
            InputMode::ReferenceMaterial => "reference_material",
        }
    }

    pub fn is_long_source(&self) -> bool {
        !matches!(
            self,
            InputMode::ConversationMessage | InputMode::MixedMessage
        )
    }
}

impl FromStr for InputMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "conversation_message" => Ok(InputMode::ConversationMessage),
            "mixed_message" => Ok(InputMode::MixedMessage),
            "pasted_document" => Ok(InputMode::PastedDocument),
            "code_block" => Ok(InputMode::CodeBlock),
            "log_dump" => Ok(InputMode::LogDump),
            "article_draft" => Ok(InputMode::ArticleDraft),
            "reference_material" => Ok(InputMode::ReferenceMaterial),
            other => Err(format!("unknown input mode: {other}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceDocument {
    pub id: String,
    pub raw_text_hash: String,
    pub source_type: InputMode,
    pub session_id: Option<String>,
    pub summary: Option<String>,
    pub entities: Vec<String>,
    pub topics: Vec<String>,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceChunk {
    pub id: String,
    pub source_id: String,
    pub chunk_index: usize,
    pub text: String,
    pub summary: Option<String>,
    pub entities: Vec<String>,
    pub topics: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryCandidateType {
    PreferenceStrong,
    PreferenceWeak,
    GoalCandidate,
    IdentityCandidate,
    ProjectFactCandidate,
    EpisodicCandidate,
    SentimentObservation,
    EntityMention,
    ContextLink,
    CorrectionSignal,
    ThemeCandidate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyReason {
    TooManyTypos,
    MissingSubject,
    MissingObject,
    AmbiguousPronoun,
    DeicticLocation,
    WeakCueOnly,
    NoKnownEntity,
    FragmentSentence,
    ConflictingSignals,
    LongInput,
    PastedDocument,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConversationFrame {
    pub active_topics: Vec<String>,
    pub active_entities: Vec<String>,
    pub active_location: Option<String>,
    pub active_project: Option<String>,
    pub session_intent: Option<String>,
    pub sentiment_direction: f64,
    pub expires_after_turns: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCandidate {
    pub candidate_type: MemoryCandidateType,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub scope: Option<String>,
    pub confidence: f64,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseResult {
    pub candidates: Vec<MemoryCandidate>,
    pub confidence: f64,
    pub uncertainty_reasons: Vec<UncertaintyReason>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTheme {
    pub id: String,
    pub label: String,
    pub supporting_nodes: Vec<String>,
    pub confidence: f64,
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionSnapshot {
    pub id: String,
    pub memory_id: String,
    pub predicted_importance: f64,
    pub predicted_lifetime_days: f64,
    pub predicted_consolidation_probability: f64,
    pub model_version: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeSnapshot {
    pub id: String,
    pub memory_id: String,
    pub actual_importance: f64,
    pub evidence_count: u32,
    pub survived_days: u32,
    pub correction_count: u32,
    pub created_at: i64,
}

/// One durable memory. Mirrors the `memory_items` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: String,
    pub content: String,
    pub summary: Option<String>,
    pub memory_type: MemoryType,
    pub source: Option<String>,
    pub topic: Option<String>,
    pub entities: Vec<String>,
    pub tags: Vec<String>,
    pub created_at: i64,
    pub updated_at: i64,
    pub last_accessed_at: Option<i64>,
    pub access_count: i64,
    pub importance_score: f64,
    pub recurrence_score: f64,
    pub recency_score: f64,
    pub decay_score: f64,
    /// How strongly this memory resists forgetting — U-shaped over novelty
    /// (high for both surprising and habitual memories). Distinct axis from
    /// decay: a salient memory can be old and untouched yet survive. Set
    /// provisionally at ingest, recomputed authoritatively at consolidation.
    #[serde(default)]
    pub salience: f64,
    pub confidence: f64,
    pub stale: bool,
    pub pinned: bool,
    /// Embedding vector, stored as a little-endian f32 BLOB in SQLite.
    /// Skipped in JSON output to keep context packs compact.
    #[serde(skip_serializing)]
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    pub content_hash: String,
}

impl MemoryItem {
    /// A fresh item with sane defaults; callers fill in scores.
    pub fn new(
        id: String,
        content: String,
        memory_type: MemoryType,
        content_hash: String,
        now: i64,
    ) -> Self {
        MemoryItem {
            id,
            content,
            summary: None,
            memory_type,
            source: None,
            topic: None,
            entities: Vec::new(),
            tags: Vec::new(),
            created_at: now,
            updated_at: now,
            last_accessed_at: None,
            access_count: 0,
            importance_score: 0.5,
            recurrence_score: 0.0,
            recency_score: 1.0,
            decay_score: 0.5,
            salience: 0.0,
            confidence: 1.0,
            stale: false,
            pinned: false,
            embedding: None,
            content_hash,
        }
    }
}

/// Mirrors the `memory_links` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLink {
    pub source_id: String,
    pub target_id: String,
    pub relation: LinkRelation,
}

/// Mirrors the `raw_events` table: the untouched input log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawEvent {
    pub id: String,
    pub session_id: Option<String>,
    pub role: Option<String>,
    pub content: String,
    pub created_at: i64,
    pub content_hash: String,
}

/// Mirrors the `sessions` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub title: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}
