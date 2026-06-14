//! A tiny single-line spinner shown while waiting for the model's first
//! token — the silent stretch that covers retrieval, model load (cold
//! starts can take many seconds) and prompt evaluation.
//!
//! The animation runs on its own thread because the chat turn blocks the
//! main thread until tokens stream. Stopping is synchronous: it joins the
//! thread and erases the frame first, so streamed tokens never interleave
//! with a half-drawn spinner.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const TICK: Duration = Duration::from_millis(80);
/// Sleep in short slices so stop() never waits a full tick.
const POLL: Duration = Duration::from_millis(10);

pub struct Spinner {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Spinner {
    /// Start animating at the current cursor position. When `animate` is
    /// false (stdout isn't a terminal) nothing is drawn at all, so piped
    /// output stays clean.
    pub fn start(animate: bool, dim: &'static str, reset: &'static str) -> Spinner {
        let stop = Arc::new(AtomicBool::new(false));
        if !animate {
            return Spinner { stop, handle: None };
        }
        let flag = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            let mut frame = 0;
            while !flag.load(Ordering::Relaxed) {
                print!("{dim}{}{reset}", FRAMES[frame % FRAMES.len()]);
                let _ = std::io::stdout().flush();
                frame += 1;
                let mut waited = Duration::ZERO;
                while waited < TICK && !flag.load(Ordering::Relaxed) {
                    std::thread::sleep(POLL);
                    waited += POLL;
                }
                // Erase the frame (backspace, blank, backspace): the next
                // frame redraws in place, and on exit the line is clean.
                print!("\u{8} \u{8}");
                let _ = std::io::stdout().flush();
            }
        });
        Spinner { stop, handle: Some(handle) }
    }

    /// Stop and erase the spinner. Idempotent; blocks (max ~one poll
    /// slice) until the line is clean.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_is_idempotent_and_drop_safe() {
        let mut s = Spinner::start(true, "", "");
        std::thread::sleep(Duration::from_millis(30));
        s.stop();
        s.stop(); // second stop must not panic or hang
        drop(s);
    }

    #[test]
    fn disabled_spinner_spawns_no_thread() {
        let mut s = Spinner::start(false, "", "");
        assert!(s.handle.is_none());
        s.stop();
    }
}
