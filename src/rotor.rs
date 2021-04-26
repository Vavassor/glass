use crate::bivector::Bivector3;

#[derive(Clone, Copy, Debug)]
pub struct Rotor3 {
    a: f64,
    b: Bivector3,
}

impl Default for Rotor3 {
    fn default() -> Self {
        Rotor3::IDENTITY
    }
}

impl Rotor3 {
    pub const IDENTITY: Rotor3 = Rotor3 {
        a: 1.0,
        b: Bivector3::ZERO,
    };

    pub fn new(scalar: f64, bivector: Bivector3) -> Rotor3 {
        Rotor3 {
            a: scalar,
            b: bivector,
        }
    }

    pub fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.a * self.a + self.b.magnitude()
    }

    pub fn normalize(&self) -> Rotor3 {
        let m = self.magnitude();
        assert_ne!(m, 0.0);
        Rotor3::new(self.a / m, self.b / m)
    }

    pub fn reverse(&self) -> Rotor3 {
        Rotor3::new(self.a, -self.b)
    }
}
