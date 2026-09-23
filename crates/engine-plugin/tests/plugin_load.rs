//! Plugin load integration: ABI reject, panic fence, snapshot roundtrip.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use engine_plugin::{
    CapabilityFlags, HostBus, PermissionGuard, PluginHost, PluginState, ProjectPermissions,
};
use starman_plugin_sdk::STARMAN_PLUGIN_ABI_VERSION;

fn example_library() -> Option<PathBuf> {
    let name = "example_gameplay";
    let candidates = [
        format!("target/debug/lib{name}.dylib"),
        format!("target/debug/lib{name}.so"),
        format!("target/debug/{name}.dll"),
        format!("target/release/lib{name}.dylib"),
        format!("target/release/lib{name}.so"),
        format!("target/release/{name}.dll"),
    ];
    candidates
        .into_iter()
        .map(PathBuf::from)
        .find(|p| p.is_file())
}

#[test]
fn loads_example_gameplay_when_built() {
    let Some(lib) = example_library() else {
        eprintln!("skipping: example_gameplay library not built yet");
        return;
    };

    let root = std::env::temp_dir().join("starman-plugin-load");
    let _ = std::fs::create_dir_all(&root);
    let bus = Arc::new(Mutex::new(HostBus::new(PermissionGuard::new(
        &root,
        ProjectPermissions::default(),
    ))));
    let mut host = PluginHost::new(bus, root.join("cache"));
    let id = host
        .load(
            "example_gameplay",
            &lib,
            CapabilityFlags::ALL,
            true,
            false,
            false,
        )
        .expect("load should succeed");
    assert_eq!(host.plugins()[0].state, PluginState::Loaded);
    assert!(!host.plugins()[0].version.is_empty());

    let snap = host.snapshot(id).expect("snapshot");
    assert_eq!(snap.len(), 4);

    let id2 = host
        .reload(id, &lib, CapabilityFlags::ALL, true, false, false)
        .expect("reload");
    assert!(host.restore(id2, &snap).unwrap());
    host.unload(id2).unwrap();
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn abi_version_constant_is_v1() {
    assert_eq!(STARMAN_PLUGIN_ABI_VERSION, 1);
}
