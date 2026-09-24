//! Built-in render passes.

pub mod debug_lines;
pub mod environment;
pub mod mesh;
pub mod post;
pub mod probes;
pub(crate) mod sprites;

/// Every built-in `(shader, defines)` combination the renderer compiles;
/// the shader test validates all of them with naga so a broken variant
/// fails CI instead of the first frame that needs it.
pub fn builtin_shader_variants() -> Vec<(&'static str, &'static [&'static str])> {
    let mut variants: Vec<(&'static str, &'static [&'static str])> = vec![
        ("sprite2d.wgsl", &[]),
        ("sky/skybox.wgsl", &[]),
        ("sky/environment_bake.wgsl", &[]),
        ("ibl/irradiance.wgsl", &[]),
        ("ibl/prefilter.wgsl", &[]),
        ("ibl/brdf_lut.wgsl", &[]),
        ("ibl/cube_downsample.wgsl", &[]),
        ("post/ssao.wgsl", &[]),
        ("post/ssao_blur.wgsl", &[]),
        ("post/bloom.wgsl", &[]),
        ("post/taa.wgsl", &[]),
        ("post/tonemap_aces.wgsl", &[]),
        ("post/tonemap_aces.wgsl", &["OUTPUT_SRGB_ENCODE"]),
        ("post/debug_view.wgsl", &[]),
        ("post/debug_view.wgsl", &["OUTPUT_SRGB_ENCODE"]),
        ("debug/lines.wgsl", &[]),
        ("debug/lines.wgsl", &["OUTPUT_SRGB_ENCODE"]),
    ];
    const PASSES: [&str; 5] = ["PREPASS", "SHADOW", "LIT", "PICKING", "OVERDRAW"];
    const MESH: [[&[&str]; 4]; 5] = [
        [
            &["PREPASS"],
            &["PREPASS", "SKINNED"],
            &["PREPASS", "ALPHA_MASK"],
            &["PREPASS", "SKINNED", "ALPHA_MASK"],
        ],
        [
            &["SHADOW"],
            &["SHADOW", "SKINNED"],
            &["SHADOW", "ALPHA_MASK"],
            &["SHADOW", "SKINNED", "ALPHA_MASK"],
        ],
        [
            &["LIT"],
            &["LIT", "SKINNED"],
            &["LIT", "ALPHA_MASK"],
            &["LIT", "SKINNED", "ALPHA_MASK"],
        ],
        [
            &["PICKING"],
            &["PICKING", "SKINNED"],
            &["PICKING", "ALPHA_MASK"],
            &["PICKING", "SKINNED", "ALPHA_MASK"],
        ],
        [
            &["OVERDRAW"],
            &["OVERDRAW", "SKINNED"],
            &["OVERDRAW", "ALPHA_MASK"],
            &["OVERDRAW", "SKINNED", "ALPHA_MASK"],
        ],
    ];
    debug_assert_eq!(PASSES.len(), MESH.len());
    for pass in MESH.iter() {
        for defines in pass.iter() {
            variants.push(("mesh.wgsl", defines));
        }
    }
    variants
}
