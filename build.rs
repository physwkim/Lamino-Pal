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
    if let Some(dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        println!("cargo:rerun-if-changed={dir}/HEAD");
        println!("cargo:rerun-if-changed={dir}/index");
    }
    let rev = git(&["describe", "--always", "--dirty", "--abbrev=12"])
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=TOMOXIDE_REV={rev}");
}
