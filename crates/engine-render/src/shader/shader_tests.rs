use super::*;
use std::time::{SystemTime, UNIX_EPOCH};

struct Root(PathBuf);

impl Root {
    fn new(tag: &str) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("starman-shader-{tag}-{nanos}"));
        std::fs::create_dir_all(root.join("common")).unwrap();
        Self(root)
    }

    fn write(&self, relative: &str, text: &str) {
        std::fs::write(self.0.join(relative), text).unwrap();
    }

    fn library(&self) -> ShaderLibrary {
        ShaderLibrary::new(&self.0, self.0.join("cache"))
    }
}

impl Drop for Root {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn expands_includes_once_and_detects_cycles() {
    let root = Root::new("include");
    root.write("common/math.wgsl", "const PI: f32 = 3.14;\n");
    root.write(
        "mesh.wgsl",
        "#include \"common/math.wgsl\"\n#include \"common/math.wgsl\"\nfn f() {}\n",
    );
    let mut lib = root.library();
    let expanded = lib.expand_source("mesh.wgsl").unwrap();
    assert_eq!(expanded.matches("PI").count(), 1);
    assert!(expanded.contains("fn f()"));

    root.write("a.wgsl", "#include \"b.wgsl\"\n");
    root.write("b.wgsl", "#include \"a.wgsl\"\n");
    let error = lib.expand_source("a.wgsl").unwrap_err().to_string();
    assert!(error.contains("cycle"), "{error}");
}

#[test]
fn conditionals_select_variants() {
    let root = Root::new("ifdef");
    root.write(
        "v.wgsl",
        "a\n#ifdef SKINNED\nskinned\n#ifndef MASK\nopaque\n#else\nmasked\n#endif\n#else\nstatic\n#endif\n#if defined(SKINNED) && !defined(MASK) || defined(FORCE)\nboth\n#endif\n#define LATE\n#ifdef LATE\nlate\n#endif\n",
    );
    let mut lib = root.library();
    let plain = lib.expand_variant("v.wgsl", &[]).unwrap().source;
    assert_eq!(plain, "a\nstatic\nlate\n");
    let skinned = lib.expand_variant("v.wgsl", &["SKINNED"]).unwrap().source;
    assert_eq!(skinned, "a\nskinned\nopaque\nboth\nlate\n");
    let masked = lib
        .expand_variant("v.wgsl", &["MASK", "SKINNED"])
        .unwrap()
        .source;
    assert_eq!(masked, "a\nskinned\nmasked\nlate\n");
    let forced = lib.expand_variant("v.wgsl", &["FORCE"]).unwrap().source;
    assert!(forced.contains("both"));
}

#[test]
fn unbalanced_conditionals_are_rejected() {
    let root = Root::new("unbalanced");
    root.write("bad.wgsl", "#ifdef X\nfoo\n");
    root.write("bad2.wgsl", "#endif\n");
    let mut lib = root.library();
    assert!(lib.expand_source("bad.wgsl").is_err());
    let error = lib.expand_source("bad2.wgsl").unwrap_err().to_string();
    assert!(error.contains("bad2.wgsl:1"), "{error}");
}

#[test]
fn line_map_points_back_to_origin() {
    let root = Root::new("linemap");
    root.write("common/lib.wgsl", "fn helper() {}\n");
    root.write(
        "main.wgsl",
        "// top\n#include \"common/lib.wgsl\"\nfn main2() {}\n",
    );
    let mut lib = root.library();
    let expanded = lib.expand_variant("main.wgsl", &[]).unwrap();
    assert_eq!(
        expanded.origin_of(2),
        Some(&SourceLocation {
            file: "common/lib.wgsl".to_owned(),
            line: 1
        })
    );
    assert_eq!(expanded.origin_of(3).unwrap().file, "main.wgsl");
    assert_eq!(expanded.origin_of(3).unwrap().line, 3);
}

#[test]
fn wgsl_errors_are_mapped_to_the_authored_file() {
    let root = Root::new("errors");
    root.write("common/broken.wgsl", "fn ok() {}\nfn broken( {}\n");
    root.write("main.wgsl", "#include \"common/broken.wgsl\"\n");
    let mut lib = root.library();
    let expanded = lib.expand_variant("main.wgsl", &[]).unwrap();
    let error = validate_wgsl("main.wgsl", &[], &expanded)
        .unwrap_err()
        .to_string();
    assert!(error.contains("common/broken.wgsl:2"), "{error}");
}

#[test]
fn every_builtin_shader_variant_parses_and_validates() {
    let mut lib = ShaderLibrary::builtin();
    for (path, defines) in crate::passes::builtin_shader_variants() {
        let expanded = lib
            .expand_variant(path, defines)
            .unwrap_or_else(|error| panic!("{path} {defines:?}: {error}"));
        validate_wgsl(path, defines, &expanded)
            .unwrap_or_else(|error| panic!("{path} {defines:?}: {error}"));
    }
}

#[test]
fn virtual_sources_include_builtin_modules_and_invalidate_on_change() {
    let mut library = ShaderLibrary::builtin();
    library.add_source(
        "ext/effect.wgsl",
        "#include \"common/view.wgsl\"\n#ifdef GLOW\nconst GLOW: f32 = 1.0;\n#endif\n",
    );
    assert!(library.has_source("ext/effect.wgsl"));
    let expanded = library
        .expand_variant("ext/effect.wgsl", &["GLOW"])
        .unwrap();
    assert!(expanded.source.contains("struct ViewUniform"));
    assert!(expanded.source.contains("const GLOW"));
    library.add_source("ext/effect.wgsl", "const CHANGED: u32 = 1u;\n");
    let expanded = library
        .expand_variant("ext/effect.wgsl", &["GLOW"])
        .unwrap();
    assert!(expanded.source.contains("CHANGED"));
    assert!(!expanded.source.contains("ViewUniform"));
}
