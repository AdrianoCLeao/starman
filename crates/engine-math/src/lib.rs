pub use glam;

pub type Vec2 = glam::Vec2;
pub type Vec3 = glam::Vec3;
pub type Vec4 = glam::Vec4;
pub type Mat3 = glam::Mat3;
pub type Affine3A = glam::Affine3A;
pub type Quat = glam::Quat;
pub type Mat4 = glam::Mat4;

pub fn module_name() -> &'static str {
    "engine-math"
}

pub fn identity() -> Mat4 {
    Mat4::IDENTITY
}
