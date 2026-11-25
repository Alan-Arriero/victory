use std::process::Command;

// The inputs live in a temporary directory: the asset policy refuses paths
// inside the source tree.

#[test]
fn census_fails_when_a_requested_path_is_missing() {
    let dir = tempfile::tempdir().unwrap();
    let valid = dir.path().join("not-an-image.txt");
    std::fs::write(&valid, "no PE image here").unwrap();
    let missing = dir.path().join("missing-census-input");
    assert!(!missing.exists());

    let output = Command::new(env!("CARGO_BIN_EXE_victory"))
        .arg("census")
        .args([&valid, &missing])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "census succeeded: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(stderr.contains("missing-census-input"), "{stderr}");
}

#[test]
fn census_accepts_a_valid_path_with_no_pe_images() {
    let dir = tempfile::tempdir().unwrap();
    let valid = dir.path().join("not-an-image.txt");
    std::fs::write(&valid, "no PE image here").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_victory"))
        .arg("census")
        .arg(&valid)
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stdout).contains("0 images scanned, 0 unreadable"));
}
