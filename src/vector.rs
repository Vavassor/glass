use crate::bivector::Bivector3;
use crate::point::Point3;
use crate::rotor::Rotor3;
use crate::trivector::Trivector3;
use std::{
    convert::From,
    fmt::Debug,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[derive(Clone, Copy, Debug, Default)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Add for Vector3 {
    type Output = Vector3;

    fn add(self, addend: Vector3) -> Self::Output {
        Vector3::new(self.x + addend.x, self.y + addend.y, self.z + addend.z)
    }
}

impl AddAssign for Vector3 {
    fn add_assign(&mut self, addend: Self) {
        *self = Vector3::new(self.x + addend.x, self.y + addend.y, self.z + addend.z);
    }
}

impl Add<Point3> for Vector3 {
    type Output = Point3;

    fn add(self, addend: Point3) -> Self::Output {
        Point3::new(self.x + addend.x, self.y + addend.y, self.z + addend.z)
    }
}

impl Div for Vector3 {
    type Output = Vector3;

    fn div(self, divisor: Vector3) -> Self::Output {
        Vector3::new(self.x / divisor.x, self.y / divisor.y, self.z / divisor.z)
    }
}

impl Div<f64> for Vector3 {
    type Output = Vector3;

    fn div(self, divisor: f64) -> Self::Output {
        Vector3::new(self.x / divisor, self.y / divisor, self.z / divisor)
    }
}

impl DivAssign for Vector3 {
    fn div_assign(&mut self, divisor: Vector3) {
        *self = Vector3::new(self.x / divisor.x, self.y / divisor.y, self.z / divisor.z);
    }
}

impl DivAssign<f64> for Vector3 {
    fn div_assign(&mut self, divisor: f64) {
        *self = Vector3::new(self.x / divisor, self.y / divisor, self.z / divisor);
    }
}

impl From<[f64; 3]> for Vector3 {
    fn from(a: [f64; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }
}

impl From<Point3> for Vector3 {
    fn from(a: Point3) -> Self {
        Self::new(a.x, a.y, a.z)
    }
}

impl Index<usize> for Vector3 {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!(),
        }
    }
}

impl IndexMut<usize> for Vector3 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!(),
        }
    }
}

impl Mul for Vector3 {
    type Output = Vector3;

    fn mul(self, multiplicand: Vector3) -> Self::Output {
        Vector3::new(
            self.x * multiplicand.x,
            self.y * multiplicand.y,
            self.z * multiplicand.z,
        )
    }
}

impl Mul<f64> for Vector3 {
    type Output = Vector3;

    fn mul(self, multiplicand: f64) -> Self::Output {
        Vector3::new(
            self.x * multiplicand,
            self.y * multiplicand,
            self.z * multiplicand,
        )
    }
}

impl Mul<Vector3> for f64 {
    type Output = Vector3;

    fn mul(self, multiplicand: Vector3) -> Self::Output {
        Vector3::new(
            self * multiplicand.x,
            self * multiplicand.y,
            self * multiplicand.z,
        )
    }
}

impl MulAssign for Vector3 {
    fn mul_assign(&mut self, multiplicand: Vector3) {
        *self = Vector3::new(
            self.x * multiplicand.x,
            self.y * multiplicand.y,
            self.z * multiplicand.z,
        );
    }
}

impl MulAssign<f64> for Vector3 {
    fn mul_assign(&mut self, multiplicand: f64) {
        *self = Vector3::new(
            self.x * multiplicand,
            self.y * multiplicand,
            self.z * multiplicand,
        );
    }
}

impl Neg for Vector3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Vector3::new(-self.x, -self.y, -self.z)
    }
}

impl Sub for Vector3 {
    type Output = Self;

    fn sub(self, subtrahend: Self) -> Self::Output {
        Vector3::new(
            self.x - subtrahend.x,
            self.y - subtrahend.y,
            self.z - subtrahend.z,
        )
    }
}

impl SubAssign for Vector3 {
    fn sub_assign(&mut self, subtrahend: Self) {
        *self = Vector3::new(
            self.x - subtrahend.x,
            self.y - subtrahend.y,
            self.z - subtrahend.z,
        );
    }
}

impl Vector3 {
    pub const ONE: Vector3 = Vector3 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };

    pub const UNIT_X: Vector3 = Vector3 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };

    pub const UNIT_Y: Vector3 = Vector3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    pub const UNIT_Z: Vector3 = Vector3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    pub const ZERO: Vector3 = Vector3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(x: f64, y: f64, z: f64) -> Vector3 {
        Vector3 { x, y, z }
    }

    pub fn scalar_triple(a: Vector3, b: Vector3, c: Vector3) -> f64 {
        a.wedge_vector3(b).wedge_vector3(c)
    }

    // This uses Lagrange's Formula: b(a⋅c) - c(a⋅b).
    pub fn vector_triple(a: Vector3, b: Vector3, c: Vector3) -> Vector3 {
        a.dot(c) * b - a.dot(b) * c
    }

    pub fn dot(&self, a: Vector3) -> f64 {
        self.x * a.x + self.y * a.y + self.z * a.z
    }

    pub fn from_rotation_between(from: Vector3, to: Vector3) -> Rotor3 {
        Rotor3::new(1.0 + to.dot(from), to.wedge_vector3(from)).normalize()
    }

    pub fn geometric_product(&self, a: Vector3) -> Rotor3 {
        Rotor3::new(self.dot(a), self.wedge_vector3(a))
    }

    pub fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalize(&self) -> Vector3 {
        let length = self.magnitude();
        assert_ne!(length, 0.0);
        *self / length
    }

    pub fn reflect(&self, normal: Vector3) -> Vector3 {
        *self - (2.0 * self.dot(normal)) * normal
    }

    pub fn scalar_project(&self, a: Vector3) -> f64 {
        let m = a.magnitude();
        assert_ne!(m, 0.0);
        self.dot(a) / m
    }

    pub fn vector_project(&self, a: Vector3) -> Vector3 {
        let m = a.magnitude_squared();
        assert_ne!(m, 0.0);
        (self.dot(a) / m) * a
    }

    pub fn vector_reject(&self, a: Vector3) -> Vector3 {
        *self - self.vector_project(a)
    }

    pub fn wedge_bivector3(&self, a: Bivector3) -> Trivector3 {
        self.x * a.yz + self.y * a.zx + self.z * a.xy
    }

    pub fn wedge_f64(&self, a: f64) -> Vector3 {
        *self * a
    }

    pub fn wedge_vector3(&self, a: Vector3) -> Bivector3 {
        Bivector3::new(
            self.x * a.y - self.y * a.x,
            self.y * a.z - self.z * a.y,
            self.z * a.x - self.x * a.z,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn almost_equal(a: f64, b: f64) -> bool {
        (a - b).abs() < 1.0e-6
    }

    fn vectors_almost_equal(a: Vector3, b: Vector3) -> bool {
        almost_equal(a.x, b.x) && almost_equal(a.y, b.y) && almost_equal(a.z, b.z)
    }

    fn vectors_exactly_equal(a: Vector3, b: Vector3) -> bool {
        a.x == b.x && a.y == b.y && a.z == b.z
    }

    #[test]
    fn test_addition_associativity() {
        let a = Vector3::new(1392.1, 0.0, 0.041);
        let b = Vector3::new(2.0, -0.7, -5203.3);
        let c = Vector3::new(30.4, 2.0, -2.1);
        assert!(vectors_almost_equal((a + b) + c, a + (b + c)));
    }

    #[test]
    fn test_addition_communtativity() {
        let a = Vector3::new(2.1, 1.0, 2.03);
        let b = Vector3::new(2.0, 0.7, 2.33);
        assert!(vectors_exactly_equal(a + b, b + a));
    }

    #[test]
    fn test_addition_identity() {
        let a = Vector3::new(100.0, 300.0, 200.0);
        assert!(vectors_exactly_equal(a + Vector3::ZERO, a));
    }

    #[test]
    fn test_addition_inverse() {
        let a = Vector3::new(3e-4, 4003.0, -0.01);
        assert!(vectors_exactly_equal(-a + a, Vector3::ZERO));
    }

    #[test]
    fn test_multiplicative_identity() {
        let a = Vector3::new(-3e12, -3.0, -0.004);
        assert!(vectors_exactly_equal(1.0 * a, a));
    }

    #[test]
    fn test_scalar_distributive() {
        let a = Vector3::new(2e-2, 3004.0, 3.01);
        let b = Vector3::new(-3e-4, -4.4e-6, 141.0);
        const C: f64 = 10203.0;
        assert!(vectors_almost_equal(C * (a + b), C * a + C * b));
    }
}
