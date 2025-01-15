use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::io::Result as IoResult;
use std::iter::repeat;
use std::iter::Filter;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::str::Split;
use std::sync::{Arc, RwLock};

use nalgebra::{Point2, Point3, Vector3};
use ncollide3d::procedural::{IndexBuffer, TriMesh};
use num_traits::{Bounded, Zero};

use crate::loader::mtl;
use crate::loader::mtl::MtlMaterial;
use crate::resource::vertex_index::VertexIndex;
use crate::resource::gpu_vector::{AllocationType, BufferType, GPUVec};
use crate::resource::mesh::{self, Mesh};

pub type Coord = Point3<f32>;
pub type Normal = Vector3<f32>;
pub type UV = Point2<f32>;
pub type Words<'a> = Filter<Split<'a, fn(char) -> bool>, fn(&&str) -> bool>;

pub fn split_words(s: &str) -> Words {
    fn is_not_empty(s: &&str) -> bool {
        !s.is_empty()
    }
    let is_not_empty: fn(&&str) -> bool = is_not_empty; 

    fn is_whitespace(c: char) -> bool {
        c.is_whitespace()
    }
    let is_whitespace: fn(char) -> bool = is_whitespace;

    s.split(is_whitespace).filter(is_not_empty)
}

fn error(line: usize, err: &str) -> ! {
    panic!("At line {}: {}", line, err)
}

fn warn(line: usize, err: &str) {
    println!("At line {}: {}", line, err)
}

fn parse_usemtl<'a>(
    l: usize,
    ws: Words<'a>,
    curr_group: usize,
    mtllib: &HashMap<String, MtlMaterial>,
    group2mtl: &mut HashMap<usize, MtlMaterial>,
    groups: &mut HashMap<String, usize>,
    groups_ids: &mut Vec<Vec<Point3<VertexIndex>>>,
    curr_mtl: &mut Option<MtlMaterial>,
) -> usize {
    let mname: Vec<&'a str> = ws.collect();
    let mname = mname.join(" ");
    let none = "None";
    if mname[..] != none[..] {
        match mtllib.get(&mname) {
            None => {
                *curr_mtl = None;
                warn(l, &format!("could not find the material {}", mname)[..]);

                curr_group
            }
            Some(m) => {
                if !group2mtl.contains_key(&curr_group) {
                    let _ = group2mtl.insert(curr_group, m.clone());
                    *curr_mtl = Some(m.clone());
                    curr_group
                } else {
                    let mut g = curr_group.to_string();
                    g.push_str(&mname[..]);

                    let new_group = parse_g(
                        l,
                        split_words(&g[..]),
                        "auto_generated_group_",
                        groups,
                        groups_ids,
                    );

                    let _ = group2mtl.insert(new_group, m.clone());
                    *curr_mtl = Some(m.clone());

                    new_group
                }
            }
        }
    } else {
        *curr_mtl = None;
        curr_group
    }
}

fn parse_mtllib<'a>(
    l: usize,
    ws: Words<'a>,
    mtl_base_dir: &Path,
    mtllib: &mut HashMap<String, MtlMaterial>,
) {
    let filename: Vec<&'a str> = ws.collect();
    let filename = filename.join(" ");

    let mut path = PathBuf::new();
    path.push(mtl_base_dir);
    path.push(filename);

    let ms = mtl::parse_file(&path);

    match ms {
        Ok(ms) => {
            for m in ms.into_iter() {
                let _ = mtllib.insert(m.name.to_string(), m);
            }
        }
        Err(err) => warn(l, &format!("{}", err)[..]),
    }
}

fn parse_v_or_vn(l: usize, mut ws: Words) -> Vector3<f32> {
    let sx = ws
        .next()
        .unwrap_or_else(|| error(l, "3 components were expected, found 0."));
    let sy = ws
        .next()
        .unwrap_or_else(|| error(l, "3 components were expected, found 1."));
    let sz = ws
        .next()
        .unwrap_or_else(|| error(l, "3 components were expected, found 2."));

    let x: Result<f32, _> = FromStr::from_str(sx);
    let y: Result<f32, _> = FromStr::from_str(sy);
    let z: Result<f32, _> = FromStr::from_str(sz);

    let x =
        x.unwrap_or_else(|e| error(l, &format!("failed to parse `{}' as a f32: {}", sx, e)[..]));
    let y =
        y.unwrap_or_else(|e| error(l, &format!("failed to parse `{}' as a f32: {}", sy, e)[..]));
    let z =
        z.unwrap_or_else(|e| error(l, &format!("failed to parse `{}' as a f32: {}", sz, e)[..]));

    Vector3::new(x, y, z)
}

fn parse_f<'a>(
    l: usize,
    ws: Words<'a>,
    coords: &[Point3<f32>],
    uvs: &[Point2<f32>],
    normals: &[Vector3<f32>],
    ignore_uvs: &mut bool,
    ignore_normals: &mut bool,
    groups_ids: &mut Vec<Vec<Point3<VertexIndex>>>,
    curr_group: usize,
) {
    let mut i = 0;
    for word in ws {
        let mut curr_ids: Vector3<i32> = Bounded::max_value();

        for (i, w) in word.split('/').enumerate() {
            if i == 0 || !w.is_empty() {
                let idx: Result<i32, _> = FromStr::from_str(w);
                match idx {
                    Ok(id) => curr_ids[i] = id - 1,
                    Err(e) => error(l, &format!("failed to parse `{}' as a i32: {}", w, e)[..]),
                }
            }
        }

        if i > 2 {
            let g = &mut groups_ids[curr_group];
            let p1 = (*g)[g.len() - i];
            let p2 = (*g)[g.len() - 1];
            g.push(p1);
            g.push(p2);
        }

        if curr_ids.y == i32::max_value() as i32 {
            *ignore_uvs = true;
        }

        if curr_ids.z == i32::max_value() as i32 {
            *ignore_normals = true;
        }

        let x;
        let y;
        let z;

        if curr_ids.x < 0 {
            x = coords.len() as i32 + curr_ids.x + 1;
        } else {
            x = curr_ids.x;
        }

        if curr_ids.y < 0 {
            y = uvs.len() as i32 + curr_ids.y + 1;
        } else {
            y = curr_ids.y;
        }

        if curr_ids.z < 0 {
            z = normals.len() as i32 + curr_ids.z + 1;
        } else {
            z = curr_ids.z;
        }

        assert!(x >= 0 && y >= 0 && z >= 0);
        groups_ids[curr_group].push(Point3::new(
            x as VertexIndex,
            y as VertexIndex,
            z as VertexIndex,
        ));

        i += 1;
    }

    if i < 2 {
        for _ in 0usize..3 - i {
            let last = *(*groups_ids)[curr_group].last().unwrap();
            groups_ids[curr_group].push(last);
        }
    }
}

fn parse_vt(l: usize, mut ws: Words) -> UV {
    let sx = ws
        .next()
        .unwrap_or_else(|| error(l, "at least 2 components were expected, found 0."));
    let sy = ws
        .next()
        .unwrap_or_else(|| error(l, "at least 2 components were expected, found 1."));

    let x: Result<f32, _> = FromStr::from_str(sx);
    let y: Result<f32, _> = FromStr::from_str(sy);

    let x =
        x.unwrap_or_else(|e| error(l, &format!("failed to parse `{}' as a f32: {}", sx, e)[..]));
    let y =
        y.unwrap_or_else(|e| error(l, &format!("failed to parse `{}' as a f32: {}", sy, e)[..]));

    Point2::new(x, y)
}

fn parse_g<'a>(
    _: usize,
    ws: Words<'a>,
    prefix: &str,
    groups: &mut HashMap<String, usize>,
    groups_ids: &mut Vec<Vec<Point3<VertexIndex>>>,
) -> usize {
    let suffix: Vec<&'a str> = ws.collect();
    let suffix = suffix.join(" ");
    let name = if suffix.is_empty() {
        prefix.to_string()
    } else {
        format!("{}/{}", prefix, suffix)
    };

    match groups.entry(name) {
        Entry::Occupied(entry) => *entry.into_mut(),
        Entry::Vacant(entry) => {
            groups_ids.push(Vec::new());

            let val = groups_ids.len() - 1;
            *entry.insert(val)
        }
    }
}

fn reformat(
    coords: Vec<Coord>,
    normals: Option<Vec<Normal>>,
    uvs: Option<Vec<UV>>,
    groups_ids: Vec<Vec<Point3<VertexIndex>>>,
    groups: HashMap<String, usize>,
    group2mtl: HashMap<usize, MtlMaterial>,
) -> Vec<(String, Mesh, Option<MtlMaterial>)> {
    let mut vt2id: HashMap<Point3<VertexIndex>, VertexIndex> = HashMap::new();
    let mut vertex_ids: Vec<VertexIndex> = Vec::new();
    let mut resc: Vec<Coord> = Vec::new();
    let mut resn: Option<Vec<Normal>> = normals.as_ref().map(|_| Vec::new());
    let mut resu: Option<Vec<UV>> = uvs.as_ref().map(|_| Vec::new());
    let mut resfs: Vec<Vec<Point3<VertexIndex>>> = Vec::new();
    let mut allfs: Vec<Point3<VertexIndex>> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut mtls: Vec<Option<MtlMaterial>> = Vec::new();

    for (name, i) in groups.into_iter() {
        names.push(name);
        mtls.push(group2mtl.get(&i).cloned());

        for point in groups_ids[i].iter() {
            let idx = match vt2id.get(point) {
                Some(i) => {
                    vertex_ids.push(*i);
                    None
                }
                None => {
                    let idx = resc.len() as VertexIndex;

                    resc.push(coords[point.x as usize]);

                    let _ = resu
                        .as_mut()
                        .map(|l| l.push((*uvs.as_ref().unwrap())[point.y as usize]));
                    let _ = resn
                        .as_mut()
                        .map(|l| l.push((*normals.as_ref().unwrap())[point.z as usize]));

                    vertex_ids.push(idx);

                    Some(idx)
                }
            };

            let _ = idx.map(|i| vt2id.insert(*point, i));
        }

        let mut resf = Vec::with_capacity(vertex_ids.len() / 3);

        assert!(vertex_ids.len() % 3 == 0);

        for f in vertex_ids[..].chunks(3) {
            resf.push(Point3::new(f[0], f[1], f[2]));
            allfs.push(Point3::new(f[0], f[1], f[2]));
        }

        resfs.push(resf);
        vertex_ids.clear();
    }

    let resn = resn.unwrap_or_else(|| Mesh::compute_normals_array(&resc[..], &allfs[..]));
    let resn = Arc::new(RwLock::new(GPUVec::new(
        resn,
        BufferType::Array,
        AllocationType::StaticDraw,
    )));
    let resu = resu.unwrap_or_else(|| repeat(Point2::origin()).take(resc.len()).collect());
    let resu = Arc::new(RwLock::new(GPUVec::new(
        resu,
        BufferType::Array,
        AllocationType::StaticDraw,
    )));
    let resc = Arc::new(RwLock::new(GPUVec::new(
        resc,
        BufferType::Array,
        AllocationType::StaticDraw,
    )));

    let mut meshes = Vec::new();
    for ((fs, name), mtl) in resfs
        .into_iter()
        .zip(names.into_iter())
        .zip(mtls.into_iter())
    {
        if !fs.is_empty() {
            let fs = Arc::new(RwLock::new(GPUVec::new(
                fs,
                BufferType::ElementArray,
                AllocationType::StaticDraw,
            )));
            let mesh = Mesh::new_with_gpu_vectors(resc.clone(), fs, resn.clone(), resu.clone());
            meshes.push((name, mesh, mtl))
        }
    }

    meshes
}

