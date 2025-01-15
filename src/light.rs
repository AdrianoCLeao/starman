use nalgebra::Point3;

#[derive(Clone)]
pub enum Light {
    Absolute(Point3<f32>),
    StickToCamera,
}
