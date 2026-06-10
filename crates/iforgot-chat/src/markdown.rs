//! Lightweight streaming Markdown renderer for the terminal.
//!
//! Models emit Markdown; iTerm/Terminal don't render it. A full Markdown
//! parser needs the whole document, which would defeat token streaming.
//! Instead this applies inline ANSI styling token-by-token as text arrives,
//! using a tiny line-buffered state machine:
//!
//! - lines are styled once complete (headings, bullets, numbered lists,
//!   blockquotes, horizontal rules)
//! - fenced ``` code blocks are detected and dimmed, with no inline
//!   styling applied inside them
//! - inline `code`, **bold**, *italic* are styled within a finished line
//!
//! It is deliberately approximate: the goal is readable output while
//! preserving streaming, not spec compliance. Color can be disabled (piped
//! output) in which case it passes text through unchanged.

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const ITALIC: &str = "\x1b[3m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";

/// Streaming renderer. Feed it tokens; it prints styled output and buffers
/// the current partial line until a newline lets it decide the line style.
pub struct MarkdownStream {
    color: bool,
    line: String,
    in_code_block: bool,
    /// Inside a ```tool block: suppress output entirely (it's machine
    /// plumbing the user shouldn't see), but keep streaming everything
    /// else normally.
    hidden: bool,
}

impl MarkdownStream {
    pub fn new(color: bool) -> Self {
        MarkdownStream { color, line: String::new(), in_code_block: false, hidden: false }
    }

    /// Push a streamed token, returning the styled text to print now.
    /// Complete lines are styled and emitted; the trailing partial line is
    /// held back until its newline arrives.
    pub fn push(&mut self, token: &str) -> String {
        let mut out = String::new();
        for ch in token.chars() {
            if ch == '\n' {
                if let Some(rendered) = self.render_line(&self.line.clone()) {
                    out.push_str(&rendered);
                    out.push('\n');
                }
                self.line.clear();
            } else {
                self.line.push(ch);
            }
        }
        out
    }

    /// Flush any buffered partial line at end of stream.
    pub fn finish(&mut self) -> String {
        if self.line.is_empty() {
            return String::new();
        }
        let rendered = self.render_line(&self.line.clone()).unwrap_or_default();
        self.line.clear();
        rendered
    }

    /// Render one completed line, or `None` to suppress it (tool blocks).
    fn render_line(&mut self, line: &str) -> Option<String> {
        let trimmed = line.trim_start();

        // A ```tool fence starts a suppressed block; everything up to its
        // closing fence is hidden from the user (the agent parses it).
        if self.hidden {
            if trimmed.starts_with("```") {
                self.hidden = false;
            }
            return None;
        }
        if trimmed.starts_with("```tool") {
            self.hidden = true;
            return None;
        }

        // Plain mode (no TTY / color off): pass the line through unstyled,
        // but tool-block suppression above still applies.
        if !self.color {
            return Some(line.to_string());
        }

        // Fenced code block borders toggle the mode; the fence line itself
        // is dimmed.
        if trimmed.starts_with("```") {
            self.in_code_block = !self.in_code_block;
            return Some(format!("{DIM}{line}{RESET}"));
        }
        if self.in_code_block {
            return Some(format!("{DIM}{line}{RESET}"));
        }

        // Horizontal rule.
        if trimmed == "---" || trimmed == "***" || trimmed == "___" {
            return Some(format!("{DIM}────────────────────{RESET}"));
        }

        // Headings: # .. ######
        if trimmed.starts_with('#') {
            let text = trimmed.trim_start_matches('#').trim();
            return Some(format!("{BOLD}{CYAN}{}{RESET}", inline(text)));
        }

        // Blockquote.
        if let Some(rest) = trimmed.strip_prefix("> ") {
            return Some(format!("{DIM}▎ {}{RESET}", inline(rest)));
        }

        // Bullet list: -, *, +
        for marker in ["- ", "* ", "+ "] {
            if let Some(rest) = trimmed.strip_prefix(marker) {
                let indent = &line[..line.len() - trimmed.len()];
                return Some(format!("{indent}{GREEN}•{RESET} {}", inline(rest)));
            }
        }

        // Numbered list: "1. ", "2) " ...
        if let Some(styled) = numbered_list(line, trimmed) {
            return Some(styled);
        }

        Some(inline(line))
    }
}

/// Style "N. text" / "N) text" list items, preserving the number.
fn numbered_list(line: &str, trimmed: &str) -> Option<String> {
    let digits: String = trimmed.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let after = &trimmed[digits.len()..];
    let rest = after.strip_prefix(". ").or_else(|| after.strip_prefix(") "))?;
    let indent = &line[..line.len() - trimmed.len()];
    Some(format!("{indent}{GREEN}{digits}.{RESET} {}", inline(rest)))
}

/// Apply inline styling for `code`, **bold**, and *italic* within a line.
/// Backtick spans win and suppress emphasis inside them.
fn inline(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '`' => {
                if let Some(end) = find_close(&chars, i + 1, '`') {
                    let span: String = chars[i + 1..end].iter().collect();
                    out.push_str(&format!("{YELLOW}{span}{RESET}"));
                    i = end + 1;
                    continue;
                }
                out.push(c);
            }
            '*' if i + 1 < chars.len() && chars[i + 1] == '*' => {
                if let Some(end) = find_close_str(&chars, i + 2, "**") {
                    let span: String = chars[i + 2..end].iter().collect();
                    out.push_str(&format!("{BOLD}{span}{RESET}"));
                    i = end + 2;
                    continue;
                }
                out.push(c);
            }
            '*' => {
                if let Some(end) = find_close(&chars, i + 1, '*') {
                    let span: String = chars[i + 1..end].iter().collect();
                    out.push_str(&format!("{ITALIC}{span}{RESET}"));
                    i = end + 1;
                    continue;
                }
                out.push(c);
            }
            _ => out.push(c),
        }
        i += 1;
    }
    out
}

fn find_close(chars: &[char], from: usize, target: char) -> Option<usize> {
    (from..chars.len()).find(|&j| chars[j] == target)
}

fn find_close_str(chars: &[char], from: usize, target: &str) -> Option<usize> {
    let t: Vec<char> = target.chars().collect();
    let mut j = from;
    while j + t.len() <= chars.len() {
        if chars[j..j + t.len()] == t[..] {
            return Some(j);
        }
        j += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_when_color_disabled() {
        let mut md = MarkdownStream::new(false);
        assert_eq!(md.push("**bold** text\n"), "**bold** text\n");
        assert_eq!(md.finish(), "");
    }

    #[test]
    fn buffers_partial_line_until_newline() {
        let mut md = MarkdownStream::new(true);
        // No newline yet -> nothing emitted.
        assert_eq!(md.push("- hello"), "");
        let out = md.push(" world\n");
        assert!(out.contains("hello world"));
        assert!(out.contains('•'));
    }

    #[test]
    fn styles_headings_and_inline_code() {
        let mut md = MarkdownStream::new(true);
        let out = md.push("# Title with `code`\n");
        assert!(out.contains("\x1b[1m")); // bold
        assert!(out.contains("\x1b[33m")); // yellow code
        assert!(!out.contains('#'));
    }

    #[test]
    fn code_block_suppresses_inline_styling() {
        let mut md = MarkdownStream::new(true);
        md.push("```\n");
        let out = md.push("let x = **not bold**;\n");
        assert!(out.contains("**not bold**"), "inline markdown left literal inside code");
        md.push("```\n");
        // Back outside the block, emphasis applies again.
        let out2 = md.push("**bold**\n");
        assert!(out2.contains("\x1b[1m"));
    }

    #[test]
    fn tool_block_is_hidden_from_output() {
        let mut md = MarkdownStream::new(true);
        let mut out = String::new();
        out.push_str(&md.push("Let me check.\n"));
        out.push_str(&md.push("```tool\n"));
        out.push_str(&md.push("{\"tool\": \"shell\", \"args\": {\"command\": \"ls\"}}\n"));
        out.push_str(&md.push("```\n"));
        out.push_str(&md.push("done\n"));
        assert!(out.contains("Let me check"));
        assert!(out.contains("done"));
        // The machine-readable tool JSON must not reach the user.
        assert!(!out.contains("\"tool\""));
        assert!(!out.contains("shell"));
        assert!(!out.contains("```"));
    }

    #[test]
    fn finish_flushes_partial_line() {
        let mut md = MarkdownStream::new(true);
        md.push("trailing without newline");
        let out = md.finish();
        assert!(out.contains("trailing"));
    }
}
