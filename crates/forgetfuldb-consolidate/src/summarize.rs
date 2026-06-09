//! Pluggable summarization.
//!
//! [`Summarizer`] is the seam where a local LLM (Ollama, llama.cpp,
//! MLX, ...) plugs in later: implement the trait, construct it in the
//! CLI/server wiring, and consolidation picks it up unchanged. v1 ships a
//! dependency-free extractive summarizer.

/// Turn a cluster of related memory texts into one compact summary.
pub trait Summarizer: Send + Sync {
    fn name(&self) -> &'static str;
    fn summarize(&self, texts: &[&str]) -> String;
}

/// Frequency-based extractive summarizer: scores each input text by how
/// many of the cluster's most common keywords it contains, then keeps the
/// top few texts verbatim. Crude, but deterministic and model-free.
#[derive(Default)]
pub struct ExtractiveSummarizer {
    /// Maximum number of source texts quoted in the summary.
    pub max_sentences: usize,
}

impl ExtractiveSummarizer {
    pub fn new(max_sentences: usize) -> Self {
        ExtractiveSummarizer { max_sentences }
    }

    fn max_sentences(&self) -> usize {
        if self.max_sentences == 0 {
            3
        } else {
            self.max_sentences
        }
    }
}

impl Summarizer for ExtractiveSummarizer {
    fn name(&self) -> &'static str {
        "extractive"
    }

    fn summarize(&self, texts: &[&str]) -> String {
        if texts.is_empty() {
            return String::new();
        }
        // Cluster-wide keyword frequencies.
        let mut freq = std::collections::HashMap::new();
        for text in texts {
            for token in forgetfuldb_core::ingest::tokenize(text) {
                *freq.entry(token).or_insert(0usize) += 1;
            }
        }
        // Score each text by the total frequency of its tokens, normalized
        // by length so long texts don't win automatically.
        let mut scored: Vec<(usize, f64)> = texts
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let tokens = forgetfuldb_core::ingest::tokenize(text);
                if tokens.is_empty() {
                    return (i, 0.0);
                }
                let total: usize = tokens.iter().map(|t| freq.get(t).copied().unwrap_or(0)).sum();
                (i, total as f64 / tokens.len() as f64)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));

        let mut picked: Vec<usize> = scored.into_iter().take(self.max_sentences()).map(|(i, _)| i).collect();
        picked.sort(); // restore chronological order
        let lines: Vec<&str> = picked.into_iter().map(|i| texts[i]).collect();
        format!("Summary of {} related memories: {}", texts.len(), lines.join(" / "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_is_nonempty_and_mentions_count() {
        let s = ExtractiveSummarizer::default();
        let out = s.summarize(&[
            "billing uses stripe for invoices",
            "stripe invoices are sent monthly",
            "the billing dashboard shows stripe payouts",
            "lunch was a sandwich",
        ]);
        assert!(out.contains("4 related memories"));
        assert!(out.contains("stripe"));
    }

    #[test]
    fn empty_input_gives_empty_summary() {
        let s = ExtractiveSummarizer::default();
        assert!(s.summarize(&[]).is_empty());
    }

    #[test]
    fn picks_central_texts_over_outliers() {
        let s = ExtractiveSummarizer::new(2);
        let out = s.summarize(&[
            "billing invoices stripe payments",
            "stripe billing payments invoices monthly",
            "completely unrelated gardening note about tulips",
        ]);
        assert!(!out.contains("tulips"));
    }
}
