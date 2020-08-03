use num_traits::{One, Zero};
/// Trait that incorporates other traits that are needed for proper matrix operations.
use std::ops::{Add, Div, Mul, Sub};
pub trait Element:
    Default
    + Copy
    + Clone
    + Add<Output = Self>
    + Mul<Output = Self>
    + One
    + Zero
    + PartialOrd
    + Sub<Output = Self>
  // Div trait not used due to integer division.
{
}
macro_rules! implementElement(
    ($name:ty) => (
        impl Element for $name {}
    );
);

pub trait FloatElement:
Element
+ Div<Output = Self>
{
}
macro_rules! implementFloatElement(
    ($name:ty) => (
        impl FloatElement for $name {}
    );
);

implementFloatElement!(f32);
implementFloatElement!(f64);

implementElement!(u8);
implementElement!(u16);
implementElement!(u32);
implementElement!(u64);

implementElement!(i8);
implementElement!(i16);
implementElement!(i32);
implementElement!(i64);

implementElement!(f32);
implementElement!(f64);

implementElement!(isize);
implementElement!(usize);


