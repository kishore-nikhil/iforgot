//! Nightly consolidation timer.
//!
//! Consolidation is the "sleep cycle" — salience revision, edge rebuilds,
//! promotion, archiving — and nothing runs it on its own. This installs a
//! per-user **launchd** agent (macOS) that runs `forgetfuldb consolidate`
//! once a night against a specific store, so a memory left alone keeps
//! decaying, consolidating, and forming traits the way the model expects.
//!
//! It is opt-in: the user runs `forgetfuldb schedule install`. We only ever
//! touch the current user's `~/Library/LaunchAgents` and `launchctl` domain.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Reverse-DNS launchd label; also the plist file stem.
pub const LABEL: &str = "com.forgetfuldb.consolidate";

/// `~/Library/LaunchAgents/<LABEL>.plist` — the per-user agent directory.
fn plist_path() -> Result<PathBuf> {
    let home = forgetfuldb_core::config::home_dir()
        .context("cannot determine home directory (HOME unset)")?;
    Ok(home.join("Library/LaunchAgents").join(format!("{LABEL}.plist")))
}

/// Build the launchd property list. Pure so it can be unit-tested without
/// touching the filesystem. Runs `<exe> consolidate --config <config>` at
/// `hour:minute` every day, appending stdout/stderr to `log_path`.
pub fn consolidate_plist(exe: &Path, config: &Path, hour: u8, minute: u8, log_path: &Path) -> String {
    // launchd reads these paths verbatim; XML-escape so a path with `&` or
    // quotes can't corrupt the plist.
    let esc = |p: &Path| {
        p.display()
            .to_string()
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    };
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exe}</string>
        <string>consolidate</string>
        <string>--config</string>
        <string>{config}</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>{minute}</integer>
    </dict>
    <key>RunAtLoad</key>
    <false/>
    <key>StandardOutPath</key>
    <string>{log}</string>
    <key>StandardErrorPath</key>
    <string>{log}</string>
</dict>
</plist>
"#,
        exe = esc(exe),
        config = esc(config),
        log = esc(log_path),
    )
}

/// Where consolidation output is logged — next to the store's config so it
/// travels with the memory it maintains.
fn log_path(config: &Path) -> PathBuf {
    config
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join("consolidate.log")
}

/// Install (or refresh) the nightly agent for the store at `config_path`.
pub fn install(config_path: &Path, hour: u8, minute: u8) -> Result<()> {
    if !cfg!(target_os = "macos") {
        anyhow::bail!(
            "`schedule` uses launchd and is only supported on macOS. On Linux, add a cron entry:\n  \
             {minute} {hour} * * *  forgetfuldb consolidate --config {}",
            config_path.display()
        );
    }
    anyhow::ensure!(hour < 24, "hour must be 0–23 (got {hour})");
    anyhow::ensure!(minute < 60, "minute must be 0–59 (got {minute})");

    let exe = std::env::current_exe().context("cannot locate the forgetfuldb binary")?;
    let config_path = config_path
        .canonicalize()
        .unwrap_or_else(|_| config_path.to_path_buf());
    let log = log_path(&config_path);
    let plist = plist_path()?;

    if let Some(dir) = plist.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    std::fs::write(&plist, consolidate_plist(&exe, &config_path, hour, minute, &log))
        .with_context(|| format!("writing {}", plist.display()))?;

    // Reload so an edited schedule takes effect. `unload` is best-effort:
    // it fails harmlessly when nothing is loaded yet.
    let _ = launchctl(&["unload", "-w"], &plist);
    launchctl(&["load", "-w"], &plist).context("launchctl load failed")?;

    println!("installed nightly consolidation at {hour:02}:{minute:02}");
    println!("  agent : {}", plist.display());
    println!("  store : {}", config_path.display());
    println!("  log   : {}", log.display());
    println!("run `forgetfuldb schedule status` to check it, `… uninstall` to remove it.");
    Ok(())
}

/// Stop and remove the nightly agent.
pub fn uninstall() -> Result<()> {
    if !cfg!(target_os = "macos") {
        anyhow::bail!("`schedule` is only supported on macOS.");
    }
    let plist = plist_path()?;
    if !plist.exists() {
        println!("no nightly consolidation agent installed.");
        return Ok(());
    }
    let _ = launchctl(&["unload", "-w"], &plist);
    std::fs::remove_file(&plist).with_context(|| format!("removing {}", plist.display()))?;
    println!("removed nightly consolidation agent ({})", plist.display());
    Ok(())
}

/// Report whether the agent is installed and currently loaded.
pub fn status() -> Result<()> {
    if !cfg!(target_os = "macos") {
        println!("scheduling via launchd is only supported on macOS.");
        return Ok(());
    }
    let plist = plist_path()?;
    if !plist.exists() {
        println!("not installed. run `forgetfuldb schedule install` to enable nightly consolidation.");
        return Ok(());
    }
    println!("agent file : {}", plist.display());
    let loaded = std::process::Command::new("launchctl")
        .args(["list", LABEL])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    println!("loaded     : {}", if loaded { "yes" } else { "no (run `schedule install` to (re)load)" });
    Ok(())
}

fn launchctl(args: &[&str], plist: &Path) -> Result<()> {
    let status = std::process::Command::new("launchctl")
        .args(args)
        .arg(plist)
        .status()
        .context("failed to run launchctl")?;
    anyhow::ensure!(status.success(), "launchctl {} exited with {status}", args.join(" "));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plist_contains_schedule_and_command() {
        let xml = consolidate_plist(
            Path::new("/Users/x/.cargo/bin/forgetfuldb"),
            Path::new("/Users/x/.forgetfuldb/forgetfuldb.toml"),
            3,
            30,
            Path::new("/Users/x/.forgetfuldb/consolidate.log"),
        );
        assert!(xml.contains("<string>com.forgetfuldb.consolidate</string>"));
        assert!(xml.contains("<string>consolidate</string>"));
        assert!(xml.contains("<string>/Users/x/.forgetfuldb/forgetfuldb.toml</string>"));
        // Schedule fields land as integers under StartCalendarInterval.
        assert!(xml.contains("<key>Hour</key>\n        <integer>3</integer>"));
        assert!(xml.contains("<key>Minute</key>\n        <integer>30</integer>"));
        // Don't fire on load — only on the calendar interval.
        assert!(xml.contains("<key>RunAtLoad</key>\n    <false/>"));
    }

    #[test]
    fn plist_escapes_special_chars_in_paths() {
        let xml = consolidate_plist(
            Path::new("/bin/fdb"),
            Path::new("/tmp/a & b/cfg.toml"),
            0,
            0,
            Path::new("/tmp/log"),
        );
        assert!(xml.contains("/tmp/a &amp; b/cfg.toml"));
        assert!(!xml.contains("a & b"));
    }

    #[test]
    fn log_sits_next_to_config() {
        assert_eq!(
            log_path(Path::new("/Users/x/.forgetfuldb/forgetfuldb.toml")),
            PathBuf::from("/Users/x/.forgetfuldb/consolidate.log")
        );
    }
}
