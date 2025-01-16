//! Structures that a gpu buffer may contain.

use crate::{
    context::context::{Context, UniformLocation},
    verify,
};
use std::{mem, slice};

use nalgebra::{
    Matrix2, Matrix3, Matrix4, Point2, Point3, Point4, Rotation2, Rotation3, Vector2, Vector3,
    Vector4,
};

pub unsafe trait GLPrimitive: Copy {
    type Element;
    const GLTYPE: u32;

    fn size() -> u32 {
        (mem::size_of::<Self>() / mem::size_of::<Self::Element>()) as u32
    }

    fn flatten(array: &[Self]) -> &[Self::Element] {
        unsafe { slice::from_raw_parts(array.as_ptr() as *const Self::Element, array.len()) }
    }

    fn upload(&self, _: &UniformLocation) {
        unimplemented!()
    }
}

unsafe impl GLPrimitive for f32 {
    type Element = f32;
    const GLTYPE: u32 = Context::FLOAT;

    #[inline]
    fn upload(&self, location: &UniformLocation) {
        verify!(Context::get().uniform1f(Some(location), *self));
    }
}

unsafe impl GLPrimitive for i32 {
    type Element = i32;
    const GLTYPE: u32 = Context::INT;

    #[inline]
    fn upload(&self, location: &UniformLocation) {
        verify!(Context::get().uniform1i(Some(location), *self));
    }
}

unsafe impl GLPrimitive for Matrix2<f32> {
    type Element = f32;
    const GLTYPE: u32 = Context::FLOAT;

    #[inline]
    fn upload(&self, location: &UniformLocation) {
        verify!(Context::get().uniform_matrix2fv(Some(location), false, self));
    }
}

unsafe impl GLPrimitive for Rotation2<f32> {
    type Element = f32;
    const GLTYPE: u32 = Context::FLOAT;

    #[inline]
    fn upload(&self, location: &UniformLocation) {
        verify!(Context::get().uniform_matrix2fv(Some(location), false, self.matrix()));
    }
}

unsafe impl GLPrimitive for Matrix3<f32> {
    type Element = f32;
    const GLTYPE: u32 = Context::FLOAT;

    #[inline]
    fn upload(&self, location: &UniformLocation) {
        verify!(Context::get().uniform_matrix3fv(Some(location), false, self));
    }
}

unsafe impl GLPrimitive for Rotation3<f32> {
    type Element = f32;
    const GLTYPE: u32 = Context::FLOAT;

    #[inline]
    fn upload(&self, location: &UniformLocation) {
        unsafe {
            verify!(Context::get().uniform_matrix3fv(Some(location), false, self.matrix()));
        }
    }
}

unsafe impl GLPrimitive for Matrix4<f32> {
    type Element = f32;
    const GLTYPE: u32 = Context::FLOAT;

    #[inline]
    fn upload(&self, location: &UniformLocation) {
        unsafe {
            verify!(Context::get().uniform_matrix4fv(Some(location), false, self));
        }
    }
}

unsafe impl GLPrimitive for Vector4<f32> {
    type Element = f32;
    const GLTYPE: u32 = Context::FLOAT;

    #[inline]
    fn upload(&self, location: &UniformLocation) {
        verify!(Context::get().uniform4f(Some(location), self.x, self.y, self.z, self.w));
    }
}

unsafe impl GLPrimitive for Vector3<f32> {
    type Element = f32;
    const GLTYPE: u32 = Context::FLOAT;

    #[inline]
    fn upload(&self, location: &UniformLocation) {
        verify!(Context::get().uniform3f(Some(location), self.x, self.y, self.z));
    }
}

unsafe impl GLPrimitive for Vector2<f32> {
    type Element = f32;
    const GLTYPE: u32 = Context::FLOAT;

    #[inline]
    fn upload(&self, location: &UniformLocation) {
        verify!(Context::get().uniform2f(Some(location), self.x, self.y));
    }
}

unsafe impl GLPrimitive for Point4<f32> {
    type Element = f32;
    const GLTYPE: u32 = Context::FLOAT;

    #[inline]
    fn upload(&self, location: &UniformLocation) {
        verify!(Context::get().uniform4f(Some(location), self.x, self.y, self.z, self.w));
    }
}

unsafe impl GLPrimitive for Point3<f32> {
    type Element = f32;
    const GLTYPE: u32 = Context::FLOAT;

    #[inline]
    fn upload(&self, location: &UniformLocation) {
        verify!(Context::get().uniform3f(Some(location), self.x, self.y, self.z));
    }
}

unsafe impl GLPrimitive for Point2<f32> {
    type Element = f32;
    const GLTYPE: u32 = Context::FLOAT;

    #[inline]
    fn upload(&self, location: &UniformLocation) {
        verify!(Context::get().uniform2f(Some(location), self.x, self.y));
    }
}

unsafe impl GLPrimitive for Point3<i32> {
    type Element = i32;
    const GLTYPE: u32 = Context::INT;

    #[inline]
    fn upload(&self, location: &UniformLocation) {
        verify!(Context::get().uniform3i(Some(location), self.x, self.y, self.z));
    }
}

unsafe impl GLPrimitive for Point2<i32> {
    type Element = i32;
    const GLTYPE: u32 = Context::INT;

    #[inline]
    fn upload(&self, location: &UniformLocation) {
        verify!(Context::get().uniform2i(Some(location), self.x, self.y));
    }
}

unsafe impl GLPrimitive for Point2<u16> {
    type Element = u16;
    const GLTYPE: u32 = Context::UNSIGNED_SHORT;
}

unsafe impl GLPrimitive for Point3<u16> {
    type Element = u16;
    const GLTYPE: u32 = Context::UNSIGNED_SHORT;
}

unsafe impl GLPrimitive for Point3<u32> {
    type Element = u32;
    const GLTYPE: u32 = Context::UNSIGNED_INT;
}

unsafe impl GLPrimitive for Point2<u32> {
    type Element = u32;
    const GLTYPE: u32 = Context::UNSIGNED_INT;
}

unsafe impl GLPrimitive for (f32, f32, f32) {
    type Element = f32;
    const GLTYPE: u32 = Context::FLOAT;

    #[inline]
    fn upload(&self, location: &UniformLocation) {
        verify!(Context::get().uniform3f(Some(location), self.0, self.1, self.2));
    }
}

unsafe impl GLPrimitive for (f32, f32) {
    type Element = f32;
    const GLTYPE: u32 = Context::FLOAT;

    #[inline]
    fn upload(&self, location: &UniformLocation) {
        verify!(Context::get().uniform2f(Some(location), self.0, self.1));
    }
}
