use crate::vector::Vector3;

pub type Trivector3 = f64;

pub trait Wedge {
    fn wedge_vector3(&self, a: Vector3) -> Vector3;
}

impl Wedge for f64 {
    fn wedge_vector3(&self, a: Vector3) -> Vector3 {
        *self * a
    }
}
