use std::iter::Filter;
use std::str::Split;
use nalgebra::{Point2, Point3, Vector3};


pub type Coord = Point3<f32>;
pub type Normal = Vector3<f32>;
pub type UV = Point2<f32>;

pub type Words<'a> = Filter<Split<'a, fn(char) -> bool>, fn(&&str) -> bool>;

pub fn split_words(s: &str) -> impl Iterator<Item = &str> {
    s.split_whitespace()
}