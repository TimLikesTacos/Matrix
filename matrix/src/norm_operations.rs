use crate::element::Element;
use crate::Matrix;

use std::error::Error;

pub trait Augment<T: Element, M: Matrix<T>>: Matrix<T> {
    fn augment(&self, other: M) -> Self;
}

pub trait Transpose<T: Element>: Matrix<T> {
    // Provides a transpose of the provided matrix
    fn transpose(&self) -> Self;
}

pub trait GaussianEliminate<T: Element>: GaussianEliminateSteps<T> {
    /// Performs gaussian elimination on the matrix this called with, with the result being the identity matrix.
    /// Return value is a vector containing every element in the last column.
    fn gaussian_eliminate(&self) -> Vec<T>;
}

/// Functions needed to perform Gaussian Elimination.
pub trait GaussianEliminateSteps<T: Element> {
    fn pivot(&mut self, row: usize);
    fn scale(&mut self, row: usize);
    fn eliminate(&mut self, row: usize);
    fn back_solve(&mut self, row: usize);
}

/// Overall actions for Gaussian Elimination, does not depend on matrix type.
/// Each matrix type will define how to do the substeps.
/// Does not mutate the matrix that calls this function
impl<M: Matrix<T> + GaussianEliminateSteps<T> + Clone, T: Element> GaussianEliminate<T> for M {
    fn gaussian_eliminate(&self) -> Vec<T> {
        let mut a: M = self.clone();
        let mut b: Vec<T> = Vec::new();
        for i in 0..a.rows() {
            a.pivot(i);
            a.scale(i);
            a.eliminate(i);
            a.back_solve(i);
        }
        for i in 0..a.rows() {
            b.push(a.get(i, a.columns() - 1).unwrap());
        }
        b
    }
}

#[cfg(test)]
mod gaus_unit_test {
    use super::*;
    use crate::norm_operations::{Augment, GaussianEliminate, Transpose};
    use crate::{Matrix, NormMatrix};

    #[test]
    fn gaus_emin_test() {
        let x: NormMatrix<f64> = NormMatrix::new(3, 3)
            .set_slice(0, 0, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0])
            .unwrap();
        let y: NormMatrix<f64> = NormMatrix::new(3, 1).set_slice(0, 0, &[0.0, 1.0, 4.0]).unwrap();

        let xt = x.transpose();
        let xtx = &xt * &x;
        let xty = xt * y;

        let mut xtx_xty = xtx.augment(xty);
        let result = xtx_xty.gaussian_eliminate();
        let expected: Vec<f64> = vec![0.0, 0.0, 1.0];
        for (res, exp) in result.iter().zip(expected){
            assert!((res - exp).abs() < 1e-10);
        }
    }
}
