//! A small, deterministic glTF 2.0 binary (`.glb`) writer: meshes with
//! optional skinning attributes, node hierarchies, skins and TRS
//! animations. Output is byte-identical for identical input.

use serde_json::{json, Value};

#[derive(Clone, Debug, Default)]
pub struct MeshPrimitive {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    /// Four joint indices per vertex (skinned meshes).
    pub joints: Vec<[u16; 4]>,
    /// Four weights per vertex, summing to one.
    pub weights: Vec<[f32; 4]>,
    pub material: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct Material {
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub name: String,
    pub primitives: Vec<MeshPrimitive>,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    pub translation: [f32; 3],
    /// Quaternion `[x, y, z, w]`.
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub children: Vec<usize>,
    pub mesh: Option<usize>,
    pub skin: Option<usize>,
}

impl Node {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
            children: Vec::new(),
            mesh: None,
            skin: None,
        }
    }

    pub fn at(mut self, translation: [f32; 3]) -> Self {
        self.translation = translation;
        self
    }
}

#[derive(Clone, Debug)]
pub struct Skin {
    pub name: String,
    pub joints: Vec<usize>,
    /// Column-major 4×4 per joint.
    pub inverse_bind_matrices: Vec<[f32; 16]>,
    pub skeleton: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelPath {
    Translation,
    Rotation,
    Scale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    Step,
    Linear,
}

#[derive(Clone, Debug)]
pub struct Channel {
    pub node: usize,
    pub path: ChannelPath,
    pub interpolation: Interpolation,
    pub times: Vec<f32>,
    /// 3 floats per key (translation/scale) or 4 (rotation xyzw).
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct Animation {
    pub name: String,
    pub channels: Vec<Channel>,
}

#[derive(Clone, Debug, Default)]
pub struct Document {
    pub nodes: Vec<Node>,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub skins: Vec<Skin>,
    pub animations: Vec<Animation>,
    /// Root nodes of the default scene.
    pub roots: Vec<usize>,
}

struct Builder {
    bin: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
}

const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;
const FLOAT: u32 = 5126;
const UNSIGNED_SHORT: u32 = 5123;
const UNSIGNED_INT: u32 = 5125;

impl Builder {
    fn align(&mut self) {
        while !self.bin.len().is_multiple_of(4) {
            self.bin.push(0);
        }
    }

    fn view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        self.align();
        let offset = self.bin.len();
        self.bin.extend_from_slice(bytes);
        let mut view = json!({"buffer": 0, "byteOffset": offset, "byteLength": bytes.len()});
        if let Some(target) = target {
            view["target"] = json!(target);
        }
        self.views.push(view);
        self.views.len() - 1
    }

    fn floats<const N: usize>(
        &mut self,
        data: &[[f32; N]],
        kind: &str,
        target: Option<u32>,
        bounds: bool,
    ) -> usize {
        let bytes: Vec<u8> = data
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let view = self.view(&bytes, target);
        let mut accessor = json!({
            "bufferView": view,
            "componentType": FLOAT,
            "count": data.len(),
            "type": kind,
        });
        if bounds && !data.is_empty() {
            let mut min = [f32::MAX; N];
            let mut max = [f32::MIN; N];
            for item in data {
                for i in 0..N {
                    min[i] = min[i].min(item[i]);
                    max[i] = max[i].max(item[i]);
                }
            }
            accessor["min"] = json!(min.to_vec());
            accessor["max"] = json!(max.to_vec());
        }
        self.accessors.push(accessor);
        self.accessors.len() - 1
    }

    fn scalars(&mut self, data: &[f32]) -> usize {
        let wrapped: Vec<[f32; 1]> = data.iter().map(|v| [*v]).collect();
        self.floats(&wrapped, "SCALAR", None, true)
    }

    fn indices(&mut self, data: &[u32]) -> usize {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = self.view(&bytes, Some(ELEMENT_ARRAY_BUFFER));
        self.accessors.push(json!({
            "bufferView": view,
            "componentType": UNSIGNED_INT,
            "count": data.len(),
            "type": "SCALAR",
        }));
        self.accessors.len() - 1
    }

    fn joints(&mut self, data: &[[u16; 4]]) -> usize {
        let bytes: Vec<u8> = data
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let view = self.view(&bytes, Some(ARRAY_BUFFER));
        self.accessors.push(json!({
            "bufferView": view,
            "componentType": UNSIGNED_SHORT,
            "count": data.len(),
            "type": "VEC4",
        }));
        self.accessors.len() - 1
    }
}

/// Serializes `document` as a `.glb`.
pub fn write_glb(document: &Document) -> Vec<u8> {
    let mut builder = Builder {
        bin: Vec::new(),
        views: Vec::new(),
        accessors: Vec::new(),
    };

    let meshes: Vec<Value> = document
        .meshes
        .iter()
        .map(|mesh| {
            let primitives: Vec<Value> = mesh
                .primitives
                .iter()
                .map(|p| {
                    let mut attributes = serde_json::Map::new();
                    attributes.insert(
                        "POSITION".into(),
                        json!(builder.floats(&p.positions, "VEC3", Some(ARRAY_BUFFER), true)),
                    );
                    if !p.normals.is_empty() {
                        attributes.insert(
                            "NORMAL".into(),
                            json!(builder.floats(&p.normals, "VEC3", Some(ARRAY_BUFFER), false)),
                        );
                    }
                    if !p.uvs.is_empty() {
                        attributes.insert(
                            "TEXCOORD_0".into(),
                            json!(builder.floats(&p.uvs, "VEC2", Some(ARRAY_BUFFER), false)),
                        );
                    }
                    if !p.joints.is_empty() {
                        attributes.insert("JOINTS_0".into(), json!(builder.joints(&p.joints)));
                        attributes.insert(
                            "WEIGHTS_0".into(),
                            json!(builder.floats(&p.weights, "VEC4", Some(ARRAY_BUFFER), false)),
                        );
                    }
                    let mut primitive = json!({
                        "attributes": attributes,
                        "indices": builder.indices(&p.indices),
                        "mode": 4,
                    });
                    if let Some(material) = p.material {
                        primitive["material"] = json!(material);
                    }
                    primitive
                })
                .collect();
            json!({"name": mesh.name, "primitives": primitives})
        })
        .collect();

    let skins: Vec<Value> = document
        .skins
        .iter()
        .map(|skin| {
            let matrices: Vec<[f32; 16]> = skin.inverse_bind_matrices.clone();
            let accessor = builder.floats(&matrices, "MAT4", None, false);
            let mut value = json!({
                "name": skin.name,
                "joints": skin.joints,
                "inverseBindMatrices": accessor,
            });
            if let Some(skeleton) = skin.skeleton {
                value["skeleton"] = json!(skeleton);
            }
            value
        })
        .collect();

    let animations: Vec<Value> = document
        .animations
        .iter()
        .map(|animation| {
            let mut samplers = Vec::new();
            let mut channels = Vec::new();
            for channel in &animation.channels {
                let input = builder.scalars(&channel.times);
                let output = match channel.path {
                    ChannelPath::Rotation => {
                        let data: Vec<[f32; 4]> = channel
                            .values
                            .chunks_exact(4)
                            .map(|c| [c[0], c[1], c[2], c[3]])
                            .collect();
                        builder.floats(&data, "VEC4", None, false)
                    }
                    _ => {
                        let data: Vec<[f32; 3]> = channel
                            .values
                            .chunks_exact(3)
                            .map(|c| [c[0], c[1], c[2]])
                            .collect();
                        builder.floats(&data, "VEC3", None, false)
                    }
                };
                samplers.push(json!({
                    "input": input,
                    "output": output,
                    "interpolation": match channel.interpolation {
                        Interpolation::Step => "STEP",
                        Interpolation::Linear => "LINEAR",
                    },
                }));
                channels.push(json!({
                    "sampler": samplers.len() - 1,
                    "target": {
                        "node": channel.node,
                        "path": match channel.path {
                            ChannelPath::Translation => "translation",
                            ChannelPath::Rotation => "rotation",
                            ChannelPath::Scale => "scale",
                        },
                    },
                }));
            }
            json!({"name": animation.name, "samplers": samplers, "channels": channels})
        })
        .collect();

    let nodes: Vec<Value> = document
        .nodes
        .iter()
        .map(|node| {
            let mut value = json!({
                "name": node.name,
                "translation": node.translation,
                "rotation": node.rotation,
                "scale": node.scale,
            });
            if !node.children.is_empty() {
                value["children"] = json!(node.children);
            }
            if let Some(mesh) = node.mesh {
                value["mesh"] = json!(mesh);
            }
            if let Some(skin) = node.skin {
                value["skin"] = json!(skin);
            }
            value
        })
        .collect();

    let materials: Vec<Value> = document
        .materials
        .iter()
        .map(|m| {
            json!({
                "name": m.name,
                "pbrMetallicRoughness": {
                    "baseColorFactor": m.base_color,
                    "metallicFactor": m.metallic,
                    "roughnessFactor": m.roughness,
                },
                "emissiveFactor": m.emissive,
            })
        })
        .collect();

    builder.align();
    let mut root = json!({
        "asset": {"version": "2.0", "generator": "starman content-gen"},
        "scene": 0,
        "scenes": [{"nodes": document.roots}],
        "nodes": nodes,
        "buffers": [{"byteLength": builder.bin.len()}],
        "bufferViews": builder.views,
        "accessors": builder.accessors,
    });
    for (key, value) in [
        ("meshes", meshes),
        ("materials", materials),
        ("skins", skins),
        ("animations", animations),
    ] {
        if !value.is_empty() {
            root[key] = Value::Array(value);
        }
    }

    let mut json_bytes = serde_json::to_vec(&root).expect("JSON serializes");
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }
    let total = 12 + 8 + json_bytes.len() + 8 + builder.bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json_bytes);
    out.extend_from_slice(&(builder.bin.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&builder.bin);
    out
}

/// Column-major translation matrix (for inverse bind matrices).
pub fn translation_matrix(t: [f32; 3]) -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, t[0], t[1], t[2], 1.0,
    ]
}
