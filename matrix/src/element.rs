/// Trait that incorporates other traits that are needed for proper matrix operations.
use std::ops::{Add, Mul};
pub trait Element: Default + Copy + Clone + Add<Output = Self> + Mul<Output = Self> {}
macro_rules! implement(
    ($name:ty) => (
        impl Element for $name {}
    );
);
implement!(u8);
implement!(u16);
implement!(u32);
implement!(u64);

implement!(i8);
implement!(i16);
implement!(i32);
implement!(i64);

implement!(f32);
implement!(f64);

implement!(isize);
implement!(usize);