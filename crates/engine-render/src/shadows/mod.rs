//! Shadow math (ADR 0010): practical-split cascades fitted as stable
//! bounding spheres with texel snapping (no shimmering while the camera
//! moves), and spot/point local shadow projections packed in an atlas.

use engine_math::{Mat4, Vec3, Vec4};

pub const MAX_CASCADES: usize = 4;
/// Atlas matrices: 2 local slots x up to 6 faces.
pub const MAX_LOCAL_SHADOW_MATRICES: usize = 12;
/// The local shadow atlas is a 4x4 grid of tiles.
pub const LOCAL_ATLAS_TILES_PER_ROW: u32 = 4;

#[derive(Clone, Debug)]
pub struct CascadeSplit {
    pub near: f32,
    pub far: f32,
    pub view_proj: Mat4,
}

#[derive(Clone, Debug)]
pub struct CsmData {
    pub cascades: Vec<CascadeSplit>,
    pub light_direction: Vec3,
}

/// Practical split scheme (log/uniform blend by `lambda`).
pub fn cascade_split_depths(
    camera_near: f32,
    camera_far: f32,
    cascade_count: u32,
    lambda: f32,
) -> Vec<f32> {
    let n = cascade_count.max(1);
    let mut splits = Vec::with_capacity(n as usize);
    let ratio = camera_far / camera_near.max(1e-3);
    for i in 1..=n {
        let p = i as f32 / n as f32;
        let log_d = camera_near * ratio.powf(p);
        let uni_d = camera_near + (camera_far - camera_near) * p;
        splits.push(lambda * log_d + (1.0 - lambda) * uni_d);
    }
    splits
}

/// World-space corners of the camera frustum slice `[near, far]`.
pub fn frustum_slice_corners(
    camera_world: Mat4,
    fov_y: f32,
    aspect: f32,
    near: f32,
    far: f32,
) -> [Vec3; 8] {
    let tan_y = (fov_y * 0.5).tan();
    let tan_x = tan_y * aspect;
    let mut corners = [Vec3::ZERO; 8];
    for (i, depth) in [near, far].into_iter().enumerate() {
        let (x, y) = (tan_x * depth, tan_y * depth);
        for (j, (sx, sy)) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
            .into_iter()
            .enumerate()
        {
            corners[i * 4 + j] = camera_world.transform_point3(Vec3::new(x * sx, y * sy, -depth));
        }
    }
    corners
}

/// Fits cascades for a directional light. `caster_extension` pushes the
/// near plane back so casters outside the view still cast into it.
#[allow(clippy::too_many_arguments)]
pub fn fit_cascades(
    light_dir: Vec3,
    camera_world: Mat4,
    fov_y: f32,
    aspect: f32,
    near: f32,
    shadow_distance: f32,
    cascade_count: u32,
    resolution: u32,
    caster_extension: f32,
) -> CsmData {
    let light_dir = light_dir.normalize_or(Vec3::NEG_Y);
    let far = shadow_distance.max(near * 2.0);
    let splits = cascade_split_depths(near, far, cascade_count.clamp(1, MAX_CASCADES as u32), 0.75);
    let up = if light_dir.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let light_view = Mat4::look_at_rh(Vec3::ZERO, light_dir, up);

    let mut cascades = Vec::with_capacity(splits.len());
    let mut previous = near;
    for split in splits {
        let corners = frustum_slice_corners(camera_world, fov_y, aspect, previous, split);
        // Bounding sphere of the slice: stable under camera rotation.
        let center = corners.iter().copied().sum::<Vec3>() / 8.0;
        let radius = corners
            .iter()
            .map(|corner| corner.distance(center))
            .fold(0.0f32, f32::max);
        // Quantize radius to reduce size jitter.
        let radius = (radius * 16.0).ceil() / 16.0;

        // Snap the center to shadow texels in light space.
        let texel = (radius * 2.0) / resolution.max(1) as f32;
        let light_center = light_view.transform_point3(center);
        let snapped = Vec3::new(
            (light_center.x / texel).floor() * texel,
            (light_center.y / texel).floor() * texel,
            light_center.z,
        );
        let snapped_world = light_view.inverse().transform_point3(snapped);

        let back = radius + caster_extension;
        let eye = snapped_world - light_dir * back;
        let view = Mat4::look_at_rh(eye, snapped_world, up);
        let proj = Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.0, back + radius);
        cascades.push(CascadeSplit {
            near: previous,
            far: split,
            view_proj: proj * view,
        });
        previous = split;
    }
    CsmData {
        cascades,
        light_direction: light_dir,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalShadowKind {
    Spot,
    Point,
}

impl LocalShadowKind {
    pub fn face_count(self) -> usize {
        match self {
            Self::Spot => 1,
            Self::Point => 6,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LocalShadowSlot {
    pub light_index: u32,
    pub kind: LocalShadowKind,
    pub atlas_index: u32,
}

/// Picks the `budget` most important local casters (by score).
pub fn select_local_shadow_casters(
    lights: &[(u32, f32, LocalShadowKind)],
    budget: u32,
) -> Vec<LocalShadowSlot> {
    let mut ranked = lights.to_vec();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
        .into_iter()
        .take(budget as usize)
        .enumerate()
        .map(|(i, (light_index, _, kind))| LocalShadowSlot {
            light_index,
            kind,
            atlas_index: i as u32,
        })
        .collect()
}

/// Cube-face view matrices in the order the shader selects them
/// (+X, -X, +Y, -Y, +Z, -Z).
pub fn point_face_views(position: Vec3) -> [Mat4; 6] {
    let faces = [
        (Vec3::X, Vec3::NEG_Y),
        (Vec3::NEG_X, Vec3::NEG_Y),
        (Vec3::Y, Vec3::Z),
        (Vec3::NEG_Y, Vec3::NEG_Z),
        (Vec3::Z, Vec3::NEG_Y),
        (Vec3::NEG_Z, Vec3::NEG_Y),
    ];
    faces.map(|(dir, up)| Mat4::look_at_rh(position, position + dir, up))
}

/// Projection for point-light faces: 90° plus a small guard band so PCF
/// at face seams stays inside the tile.
pub fn point_face_projection(range: f32, guard: f32) -> Mat4 {
    Mat4::perspective_rh(
        std::f32::consts::FRAC_PI_2 + guard,
        1.0,
        0.05,
        range.max(0.1),
    )
}

pub fn spot_view_proj(position: Vec3, direction: Vec3, outer_cone: f32, range: f32) -> Mat4 {
    let dir = direction.normalize_or(Vec3::NEG_Y);
    let up = if dir.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
    let view = Mat4::look_at_rh(position, position + dir, up);
    let fov = (outer_cone * 2.0 + 0.1).min(std::f32::consts::PI - 0.1);
    Mat4::perspective_rh(fov, 1.0, 0.05, range.max(0.1)) * view
}

/// Atlas tile rect (uv offset, uv scale) of matrix slot `index`.
pub fn atlas_tile(index: u32) -> [f32; 4] {
    let scale = 1.0 / LOCAL_ATLAS_TILES_PER_ROW as f32;
    let x = index % LOCAL_ATLAS_TILES_PER_ROW;
    let y = index / LOCAL_ATLAS_TILES_PER_ROW;
    [x as f32 * scale, y as f32 * scale, scale, scale]
}

/// Viewport (pixels) of matrix slot `index` in an atlas of `atlas_size`.
pub fn atlas_viewport(index: u32, atlas_size: u32) -> (u32, u32, u32) {
    let tile = atlas_size / LOCAL_ATLAS_TILES_PER_ROW;
    let x = index % LOCAL_ATLAS_TILES_PER_ROW;
    let y = index / LOCAL_ATLAS_TILES_PER_ROW;
    (x * tile, y * tile, tile)
}

/// Whether a world AABB may cast into the ortho/perspective `view_proj`.
pub fn aabb_in_clip_volume(view_proj: Mat4, min: Vec3, max: Vec3) -> bool {
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, max.y, max.z),
        Vec3::new(max.x, max.y, max.z),
    ];
    let clips: Vec<Vec4> = corners
        .iter()
        .map(|c| view_proj * Vec4::new(c.x, c.y, c.z, 1.0))
        .collect();
    // Reject when every corner is outside the same clip plane.
    let outside = |f: &dyn Fn(&Vec4) -> bool| clips.iter().all(f);
    !(outside(&|c| c.x < -c.w)
        || outside(&|c| c.x > c.w)
        || outside(&|c| c.y < -c.w)
        || outside(&|c| c.y > c.w)
        || outside(&|c| c.z > c.w))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_are_monotonic() {
        let splits = cascade_split_depths(0.1, 100.0, 4, 0.5);
        assert_eq!(splits.len(), 4);
        for w in splits.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn local_budget_respected() {
        let lights = vec![
            (0, 10.0, LocalShadowKind::Spot),
            (1, 5.0, LocalShadowKind::Point),
            (2, 1.0, LocalShadowKind::Spot),
        ];
        let slots = select_local_shadow_casters(&lights, 2);
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[0].light_index, 0);
    }

    #[test]
    fn cascades_cover_their_slice_and_are_stable_under_small_moves() {
        let camera = Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0));
        let a = fit_cascades(
            Vec3::new(-0.3, -1.0, -0.2),
            camera,
            1.0,
            1.6,
            0.1,
            60.0,
            4,
            1024,
            20.0,
        );
        assert_eq!(a.cascades.len(), 4);
        for cascade in &a.cascades {
            let corners = frustum_slice_corners(camera, 1.0, 1.6, cascade.near, cascade.far);
            for corner in corners {
                let clip = cascade.view_proj * Vec4::new(corner.x, corner.y, corner.z, 1.0);
                let ndc = clip.truncate() / clip.w;
                assert!(ndc.x.abs() <= 1.001 && ndc.y.abs() <= 1.001, "{ndc:?}");
                assert!((0.0..=1.0).contains(&ndc.z), "{ndc:?}");
            }
        }
        // A sub-texel camera move keeps the projection identical (snapping).
        let moved = Mat4::from_translation(Vec3::new(0.0001, 2.0, 0.0));
        let b = fit_cascades(
            Vec3::new(-0.3, -1.0, -0.2),
            moved,
            1.0,
            1.6,
            0.1,
            60.0,
            4,
            1024,
            20.0,
        );
        let diff = (a.cascades[0].view_proj - b.cascades[0].view_proj)
            .to_cols_array()
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-4, "cascade matrix changed by {diff}");
    }

    #[test]
    fn point_faces_cover_all_axes() {
        let views = point_face_views(Vec3::ZERO);
        let proj = point_face_projection(10.0, 0.0);
        for (face, target) in [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ]
        .iter()
        .enumerate()
        {
            let clip =
                proj * views[face] * Vec4::new(target.x * 5.0, target.y * 5.0, target.z * 5.0, 1.0);
            let ndc = clip.truncate() / clip.w;
            assert!(
                ndc.x.abs() < 1e-4 && ndc.y.abs() < 1e-4,
                "face {face}: {ndc:?}"
            );
        }
    }

    #[test]
    fn atlas_tiles_do_not_overlap() {
        assert_eq!(atlas_tile(0), [0.0, 0.0, 0.25, 0.25]);
        assert_eq!(atlas_tile(5), [0.25, 0.25, 0.25, 0.25]);
        assert_eq!(atlas_viewport(5, 2048), (512, 512, 512));
    }

    #[test]
    fn clip_volume_rejects_boxes_outside() {
        let vp = Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.0, 10.0);
        assert!(aabb_in_clip_volume(
            vp,
            Vec3::splat(-0.5),
            Vec3::splat(0.5) - Vec3::Z
        ));
        assert!(!aabb_in_clip_volume(
            vp,
            Vec3::new(5.0, 5.0, -2.0),
            Vec3::new(6.0, 6.0, -1.0)
        ));
    }
}
