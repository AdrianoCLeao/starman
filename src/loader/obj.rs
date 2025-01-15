use std::fs::File;
use std::path::PathBuf;
use std::{iter::Filter, path::Path};
use std::str::Split;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::str::FromStr;
use std::io::Result as IoResult;
use nalgebra::{Point2, Point3, Vector3};
use ncollide3d::procedural::{IndexBuffer, TriMesh};
use num_traits::{Bounded, Zero};

use crate::loader::mtl;
use crate::loader::mtl::MtlMaterial;
use crate::resource::vertex_index::VertexIndex;
use crate::resource::gpu_vector::GPUVec;
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

