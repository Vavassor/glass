use crate::trivector::Trivector3;
use crate::vector::Vector3;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default)]
pub struct Bivector3 {
    pub xy: f64,
    pub yz: f64,
    pub zx: f64,
}

impl Add for Bivector3 {
    type Output = Bivector3;

    fn add(self, addend: Bivector3) -> Self::Output {
        Bivector3::new(
            self.xy + addend.xy,
            self.yz + addend.yz,
            self.zx + addend.zx,
        )
    }
}

impl AddAssign for Bivector3 {
    fn add_assign(&mut self, addend: Self) {
        *self = Bivector3::new(
            self.xy + addend.xy,
            self.yz + addend.yz,
            self.zx + addend.zx,
        );
    }
}

impl Div<f64> for Bivector3 {
    type Output = Bivector3;

    fn div(self, divisor: f64) -> Self::Output {
        Bivector3::new(self.xy / divisor, self.yz / divisor, self.zx / divisor)
    }
}

impl DivAssign<f64> for Bivector3 {
    fn div_assign(&mut self, divisor: f64) {
        *self = Bivector3::new(self.xy / divisor, self.yz / divisor, self.zx / divisor);
    }
}

impl Mul<f64> for Bivector3 {
    type Output = Bivector3;

    fn mul(self, multiplicand: f64) -> Self::Output {
        Bivector3::new(
            self.xy * multiplicand,
            self.yz * multiplicand,
            self.zx * multiplicand,
        )
    }
}

impl Mul<Bivector3> for f64 {
    type Output = Bivector3;

    fn mul(self, multiplicand: Bivector3) -> Self::Output {
        Bivector3::new(
            self * multiplicand.xy,
            self * multiplicand.yz,
            self * multiplicand.zx,
        )
    }
}

impl MulAssign<f64> for Bivector3 {
    fn mul_assign(&mut self, multiplicand: f64) {
        *self = Bivector3::new(
            self.xy * multiplicand,
            self.yz * multiplicand,
            self.zx * multiplicand,
        );
    }
}

impl Neg for Bivector3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Bivector3::new(-self.xy, -self.yz, -self.zx)
    }
}

impl Sub for Bivector3 {
    type Output = Self;

    fn sub(self, subtrahend: Self) -> Self::Output {
        Bivector3::new(
            self.xy - subtrahend.xy,
            self.yz - subtrahend.yz,
            self.zx - subtrahend.zx,
        )
    }
}

impl SubAssign for Bivector3 {
    fn sub_assign(&mut self, subtrahend: Self) {
        *self = Bivector3::new(
            self.xy - subtrahend.xy,
            self.yz - subtrahend.yz,
            self.zx - subtrahend.zx,
        );
    }
}

impl Bivector3 {
    pub const UNIT_XY: Bivector3 = Bivector3 {
        xy: 1.0,
        yz: 0.0,
        zx: 0.0,
    };

    pub const UNIT_YZ: Bivector3 = Bivector3 {
        xy: 0.0,
        yz: 1.0,
        zx: 0.0,
    };

    pub const UNIT_ZX: Bivector3 = Bivector3 {
        xy: 0.0,
        yz: 0.0,
        zx: 1.0,
    };

    pub const ZERO: Bivector3 = Bivector3 {
        xy: 0.0,
        yz: 0.0,
        zx: 0.0,
    };

    pub fn new(xy: f64, yz: f64, zx: f64) -> Bivector3 {
        Bivector3 { xy, yz, zx }
    }

    pub fn dot(&self, a: Bivector3) -> f64 {
        -self.xy * a.xy - self.yz * a.yz - self.zx * a.zx
    }

    pub fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.xy * self.xy + self.yz * self.yz + self.zx * self.zx
    }

    pub fn normalize(&self) -> Bivector3 {
        let m = self.magnitude();
        assert_ne!(m, 0.0);
        *self / m
    }

    pub fn wedge_vector3(&self, a: Vector3) -> Trivector3 {
        self.xy * a.z + self.yz * a.x + self.zx * a.y
    }
}
