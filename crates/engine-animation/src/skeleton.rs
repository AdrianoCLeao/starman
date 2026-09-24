//! Skeletons (bone hierarchy + bind data) and poses.

use std::collections::HashMap;

use engine_assets::{Asset, AssetLoader, LoadContext};
use engine_core::Result;
use engine_math::{Affine3A, Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::gltf_import::{gltf_document, node_globals, node_parents};

/// Local transform of one bone.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoneTransform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for BoneTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl BoneTransform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    pub fn to_affine(&self) -> Affine3A {
        Affine3A::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    pub fn from_affine(affine: &Affine3A) -> Self {
        let (scale, rotation, translation) = affine.to_scale_rotation_translation();
        Self {
            translation,
            rotation: rotation.normalize(),
            scale,
        }
    }

    /// Weighted blend towards `other`.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            translation: self.translation.lerp(other.translation, t),
            rotation: self.rotation.slerp(other.rotation, t),
            scale: self.scale.lerp(other.scale, t),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bone {
    pub name: String,
    /// Parent bone index (always lower than this bone's index).
    pub parent: Option<usize>,
    /// Rest (bind-time local) transform.
    pub rest: BoneTransform,
}

/// A bone hierarchy in parent-before-child order, with the skinning joint
/// table of the mesh it deforms.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Skeleton {
    pub name: String,
    pub bones: Vec<Bone>,
    /// Transform of the root bones' parent space relative to the model
    /// (armature nodes above the skeleton in the source file).
    pub root_transform: Mat4,
    /// Skinning joint `j` → bone index.
    pub joints: Vec<usize>,
    /// Inverse bind matrix per joint.
    pub inverse_bind: Vec<Mat4>,
    #[serde(skip)]
    by_name: HashMap<String, usize>,
}

impl Asset for Skeleton {
    const TYPE_NAME: &'static str = "Skeleton";
}

impl Skeleton {
    pub fn new(
        name: impl Into<String>,
        bones: Vec<Bone>,
        root_transform: Mat4,
        joints: Vec<usize>,
        inverse_bind: Vec<Mat4>,
    ) -> Self {
        let mut skeleton = Self {
            name: name.into(),
            bones,
            root_transform,
            joints,
            inverse_bind,
            by_name: HashMap::new(),
        };
        skeleton.rebuild_index();
        skeleton
    }

    fn rebuild_index(&mut self) {
        self.by_name = self
            .bones
            .iter()
            .enumerate()
            .map(|(index, bone)| (bone.name.clone(), index))
            .collect();
    }

    pub fn bone_index(&self, name: &str) -> Option<usize> {
        if self.by_name.len() == self.bones.len() {
            return self.by_name.get(name).copied();
        }
        self.bones.iter().position(|bone| bone.name == name)
    }

    pub fn len(&self) -> usize {
        self.bones.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bones.is_empty()
    }

    pub fn rest_pose(&self) -> Pose {
        Pose {
            locals: self.bones.iter().map(|bone| bone.rest).collect(),
        }
    }

    /// Whether `bone` is `ancestor` or one of its descendants.
    pub fn is_descendant(&self, bone: usize, ancestor: usize) -> bool {
        let mut current = Some(bone);
        while let Some(index) = current {
            if index == ancestor {
                return true;
            }
            current = self.bones.get(index).and_then(|bone| bone.parent);
        }
        false
    }

    /// Checks ordering and table sizes.
    pub fn validate(&self) -> std::result::Result<(), String> {
        for (index, bone) in self.bones.iter().enumerate() {
            if bone.parent.is_some_and(|parent| parent >= index) {
                return Err(format!("bone '{}' is listed before its parent", bone.name));
            }
        }
        if self.joints.len() != self.inverse_bind.len() {
            return Err("joint and inverse bind tables differ in length".to_owned());
        }
        if self.joints.iter().any(|bone| *bone >= self.bones.len()) {
            return Err("a joint references a missing bone".to_owned());
        }
        Ok(())
    }
}

/// Local bone transforms of one skeleton instance.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pose {
    pub locals: Vec<BoneTransform>,
}

impl Pose {
    /// Model-space bone transforms (parents are always earlier).
    pub fn model_space(&self, skeleton: &Skeleton, out: &mut Vec<Affine3A>) {
        out.clear();
        let root = Affine3A::from_mat4(skeleton.root_transform);
        for (index, bone) in skeleton.bones.iter().enumerate() {
            let local = self
                .locals
                .get(index)
                .copied()
                .unwrap_or(bone.rest)
                .to_affine();
            let parent = match bone.parent {
                Some(parent) => out[parent],
                None => root,
            };
            out.push(parent * local);
        }
    }
}

/// Loads `file.glb#skin:<i>` as a [`Skeleton`].
pub struct SkeletonLoader;

impl AssetLoader for SkeletonLoader {
    type Asset = Skeleton;

    fn extensions(&self) -> &'static [&'static str] {
        &["glb", "gltf"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<Skeleton> {
        let index = match ctx.sub_key {
            None => 0,
            Some(key) => key
                .strip_prefix("skin:")
                .and_then(|index| index.parse::<usize>().ok())
                .ok_or_else(|| ctx.error(format!("'{key}' is not a skin sub-asset key")))?,
        };
        let (document, buffers) = gltf_document(bytes, ctx)?;
        let skin = document
            .skins()
            .nth(index)
            .ok_or_else(|| ctx.error(format!("the file has no skin {index}")))?;
        skeleton_from_skin(&document, &buffers, &skin).map_err(|reason| ctx.error(reason))
    }
}

/// Builds a skeleton from a glTF skin: every joint plus the non-joint
/// nodes between joints, in parent-before-child order.
pub fn skeleton_from_skin(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    skin: &gltf::Skin<'_>,
) -> std::result::Result<Skeleton, String> {
    let parents = node_parents(document);
    let globals = node_globals(document, &parents);
    let joint_nodes: Vec<usize> = skin.joints().map(|node| node.index()).collect();
    if joint_nodes.is_empty() {
        return Err("skin has no joints".to_owned());
    }

    // Include intermediate ancestors that lie between two joints.
    let is_joint = |node: usize| joint_nodes.contains(&node);
    let mut included: Vec<usize> = Vec::new();
    for &joint in &joint_nodes {
        let mut chain = vec![joint];
        let mut current = parents[joint];
        while let Some(node) = current {
            if is_joint(node) {
                break;
            }
            chain.push(node);
            current = parents[node];
        }
        if current.is_some() {
            for node in chain {
                if !included.contains(&node) {
                    included.push(node);
                }
            }
        } else if !included.contains(&joint) {
            included.push(joint);
        }
    }

    // Parent-before-child order: sort by depth, stable by first appearance.
    let depth = |node: usize| {
        let mut depth = 0;
        let mut current = parents[node];
        while let Some(parent) = current {
            depth += 1;
            current = parents[parent];
        }
        depth
    };
    included.sort_by_key(|node| depth(*node));

    let nodes: Vec<gltf::Node<'_>> = document.nodes().collect();
    let mut bones = Vec::with_capacity(included.len());
    let mut root_parent_global: Option<Mat4> = None;
    for &node in &included {
        let parent = parents[node].and_then(|p| included.iter().position(|n| *n == p));
        if parent.is_none() {
            let parent_global = parents[node].map_or(Mat4::IDENTITY, |p| globals[p]);
            root_parent_global.get_or_insert(parent_global);
        }
        let (t, r, s) = nodes[node].transform().decomposed();
        bones.push(Bone {
            name: nodes[node]
                .name()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("node_{node}")),
            parent,
            rest: BoneTransform {
                translation: Vec3::from(t),
                rotation: Quat::from_array(r).normalize(),
                scale: Vec3::from(s),
            },
        });
    }

    let reader = skin.reader(|buffer| buffers.get(buffer.index()).map(|data| &data.0[..]));
    let inverse_bind: Vec<Mat4> = match reader.read_inverse_bind_matrices() {
        Some(matrices) => matrices.map(|m| Mat4::from_cols_array_2d(&m)).collect(),
        None => vec![Mat4::IDENTITY; joint_nodes.len()],
    };
    if inverse_bind.len() != joint_nodes.len() {
        return Err("inverse bind matrix count does not match the joints".to_owned());
    }
    let joints = joint_nodes
        .iter()
        .map(|node| {
            included
                .iter()
                .position(|n| n == node)
                .expect("joint included")
        })
        .collect();

    let skeleton = Skeleton::new(
        skin.name().unwrap_or("skeleton"),
        bones,
        root_parent_global.unwrap_or(Mat4::IDENTITY),
        joints,
        inverse_bind,
    );
    skeleton.validate()?;
    Ok(skeleton)
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn chain(len: usize) -> Skeleton {
        let bones = (0..len)
            .map(|i| Bone {
                name: format!("b{i}"),
                parent: i.checked_sub(1),
                rest: BoneTransform {
                    translation: if i == 0 { Vec3::ZERO } else { Vec3::Y },
                    ..BoneTransform::IDENTITY
                },
            })
            .collect();
        Skeleton::new(
            "chain",
            bones,
            Mat4::IDENTITY,
            (0..len).collect(),
            vec![Mat4::IDENTITY; len],
        )
    }

    #[test]
    fn model_space_composes_parents() {
        let skeleton = chain(3);
        let mut pose = skeleton.rest_pose();
        pose.locals[1].rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let mut model = Vec::new();
        pose.model_space(&skeleton, &mut model);
        let tip = model[2].translation;
        assert!(
            (Vec3::from(tip) - Vec3::new(-1.0, 1.0, 0.0)).length() < 1e-5,
            "{tip}"
        );
        assert_eq!(skeleton.bone_index("b2"), Some(2));
        assert!(skeleton.is_descendant(2, 1));
        assert!(!skeleton.is_descendant(0, 1));
        skeleton.validate().unwrap();
    }
}
