//! Decide at build time whether the observability UI can be embedded.
//!
//! Embedding requires the built `ui/dist`. We only set the `embed_ui` cfg
//! when the `embed-ui` feature is on AND `ui/dist/index.html` exists — so
//! a fresh checkout that hasn't run `npm run build` still compiles (it just
//! won't have an embedded UI; `--ui <path>` or `./ui/dist` still work).

use std::path::Path;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(embed_ui)");

    let feature_on = std::env::var_os("CARGO_FEATURE_EMBED_UI").is_some();
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let index = Path::new(&manifest).join("../../ui/dist/index.html");

    if feature_on && index.exists() {
        println!("cargo::rustc-cfg=embed_ui");
    }
    // Rebuild if the UI is (re)built or removed.
    println!("cargo::rerun-if-changed=../../ui/dist/index.html");
}
