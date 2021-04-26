use crate::bivector::Bivector3;
use crate::vector::Vector3;

fn almost_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < 1.0e-6
}

fn bivectors_almost_equal(a: Bivector3, b: Bivector3) -> bool {
    almost_equal(a.xy, b.xy) && almost_equal(a.yz, b.yz) && almost_equal(a.zx, b.zx)
}

// Test the following property.
// a ∧ (b + c) = a ∧ b + a ∧ c
#[test]
fn test_addition_left_distributive() {
    let a = Vector3::new(1.0, 0.007, 0.3);
    let b = Vector3::new(-2.5, -0.7, 5.0);
    let c = Vector3::new(-3.0, 12.0, -0.5);
    let r0 = a.wedge_vector3(b + c);
    let r1 = a.wedge_vector3(b) + a.wedge_vector3(c);
    assert!(bivectors_almost_equal(r0, r1));
}

// Test the following property.
// (a + b) ∧ c = a ∧ c + b ∧ c
#[test]
fn test_addition_right_distributive() {
    let a = Vector3::new(1.0, 0.007, 0.3);
    let b = Vector3::new(-2.5, -0.7, 5.0);
    let c = Vector3::new(-3.0, 12.0, -0.5);
    let r0 = (a + b).wedge_vector3(c);
    let r1 = a.wedge_vector3(c) + b.wedge_vector3(c);
    assert!(bivectors_almost_equal(r0, r1));
}

// Test the following property.
// a ∧ b = -b ∧ a
#[test]
fn test_anti_symmetric() {
    let a = Vector3::new(-0.03, 50.0, -0.005);
    let b = Vector3::new(0.23404, 121.4, -2.5);
    let r0 = a.wedge_vector3(b);
    let r1 = (-b).wedge_vector3(a);
    assert!(bivectors_almost_equal(r0, r1));
}

// Test the following property.
// (a ∧ b) ∧ c = a ∧ (b ∧ c)
#[test]
fn test_associativity() {
    let a = Vector3::new(101.0, 10.0, -0.3);
    let b = Vector3::new(-2.4343, 109.2, 0.0);
    let c = Vector3::new(399.0, -65.0, 0.1);
    let r0 = a.wedge_vector3(b).wedge_vector3(c);
    let r1 = a.wedge_bivector3(b.wedge_vector3(c));
    assert!(almost_equal(r0, r1));
}

// Test the following property.
// (a ∧ b) ⋅ (c ∧ d) = (a ⋅ d)(b ⋅ c) - (a ⋅ c)(b ⋅ d)
#[test]
fn test_dot_product_of_bivectors() {
    let a = Vector3::new(0.0, 1.0, -1.0);
    let b = Vector3::new(-1.0, -1.0, 1.0);
    let c = Vector3::new(2.5, -1.0, 5.0);
    let d = Vector3::new(3.0, 3.0, 0.5);
    let r0 = a.wedge_vector3(b).dot(c.wedge_vector3(d));
    let r1 = a.dot(d) * b.dot(c) - a.dot(c) * b.dot(d);
    assert!(almost_equal(r0, r1));
}

// Test the following property.
// s(a ∧ b) = sa ∧ b = a ∧ sb
#[test]
fn test_scalar_associativity() {
    let a = Vector3::new(2.5, -1.0, 5.0);
    let b = Vector3::new(3.0, 3.0, 0.5);
    let ab = a.wedge_vector3(b);
    const S: f64 = 3.4;
    let r0 = S * ab;
    let r1 = (S * a).wedge_vector3(b);
    let r2 = a.wedge_vector3(S * b);
    assert!(bivectors_almost_equal(r0, r1));
    assert!(bivectors_almost_equal(r0, r2));
}
