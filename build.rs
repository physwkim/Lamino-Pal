//! Embeds the tomoxide submodule commit as `TOMOXIDE_REV` for the op log.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .args(["-C", "lib/tomoxide"])
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

fn main() {
    // The reflog moves on every commit, checkout and reset (a soft reset
    // leaves HEAD and the index alone); the sources make `-dirty` follow edits.
    if let Some(dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        println!("cargo:rerun-if-changed={dir}/logs/HEAD");
        println!("cargo:rerun-if-changed={dir}/index");
    }
    println!("cargo:rerun-if-changed=lib/tomoxide/crates/tomoxide");
    println!("cargo:rerun-if-changed=lib/tomoxide/Cargo.toml");
    let rev = git(&["describe", "--always", "--dirty", "--abbrev=12"])
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=TOMOXIDE_REV={rev}");
}
