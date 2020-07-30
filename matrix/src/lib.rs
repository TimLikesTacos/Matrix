use std::ops::{Index, IndexMut};

mod element;
mod norm_matrix;
use element::Element;
use norm_matrix::NormMatrix;

/// The Mat trait is used for Matrix types.  It contains basic set and get functions to manipulate data,
/// along with a set_slice function to set a certain length, or all, of the matrix using an array or vector.
pub trait Matrix<T: Element>: Index<usize> + IndexMut<usize> {
    type Output;

    /// Gets value at coordinate
    fn get(&self, row: usize, column: usize) -> Option<T>;
    /// Sets value at coordinate.  Returns Option with value set or None if invalid coordinate.
    fn set(&mut self, row: usize, column: usize, value: T) -> Option<T>;

    /// sets matrix starting at the start position by filling in each column of the row then wrapping to the beginning
    /// of the next row.
    /// returns Option with self if setting is successful, None if setting matrix will go out of bounds.
    fn set_slice(&self, start_row: usize, start_col: usize, slice: &[T]) -> Option<NormMatrix<T>>;

    /// returns the number of rows in the matrix
    fn rows(&self) -> usize;

    /// returns the number of columns in the matrix
    fn columns(&self) -> usize;

    /// Creates a new matrix with default values
    fn new(rows: usize, columns: usize) -> Self;

}



#[cfg(test)]
mod tests {

    use super::*;
    fn setup() -> NormMatrix<f64> {
        NormMatrix::new(3, 2).set_slice(0, 0, &[0.0, 3.0, 1.0, -2.0, 2.0, 1.0]).unwrap()
        
    }

    #[test]
    fn set_test() {
        let mut v = setup();
        assert_eq!(v.get(0, 0), Some(0.0));
        assert_eq!(v.set(0, 0, 2.0), Some(2.0));
        assert_eq!(v.get(0, 0), Some(2.0));
        assert_eq!(v.rows(), 3);
        assert_eq!(v.columns(), 2);
    }

    #[test]
    fn new_test() {
        let v: NormMatrix<i32> = NormMatrix::new(3, 5);
        assert_eq!(v.rows(), 3);
        assert_eq!(v.columns(), 5);

        assert_eq!(v.get(0, 0), Some(0));
        assert_eq!(v.get(2, 4), Some(0));
        assert!(v.get(3, 5).is_none());
    }

    #[test]
    fn set_slice_test() {
        let v: NormMatrix<i32> = NormMatrix::new(5, 3).set_slice(0, 0, &[1, 2, 3, 4]).unwrap();
    
        assert_eq!(v.get(0, 0), Some(1));
        assert_eq!(v.get(0, 1), Some(2));
        assert_eq!(v.get(0, 2), Some(3));
        assert_eq!(v.get(0, 3), None);
        assert_eq!(v.get(1, 0), Some(4));
        assert_eq!(v.get(1, 1), Some(0));
        assert_eq!(v.rows(), 5);
        assert_eq!(v.columns(), 3);

        let v = v.set_slice(3, 2, &[1, 2, 3, 4]).unwrap();

        assert_eq!(v.get(v.rows() - 1, v.columns() - 1), Some(4));
        assert!(v.set_slice(3, 3, &[1, 2, 3, 4]).is_none()); // should return false, as length is larger than matrix allows
    }

    #[test]
    fn brackets() {
        let v = setup();
        assert_eq!(v[1], 3.0);
    }

    #[test]
    #[should_panic]
    fn bracket_panic() {
        let v = setup();
        assert!(v[10] == 0.0); // should panic!
    }

    #[test]
    fn bracket_mut() {
        let mut v = setup();
        v[0] = 5.0;
        assert_eq!(v.get(0, 0), Some(5.0));
        assert_eq!(v.get(0, 1), Some(3.0));

        let mut v: NormMatrix<i32> = NormMatrix::new(2, 2);

        v[1] = 3;
        assert_eq!(v.get(0, 1), Some(3));
        assert_eq!(v.get(1, 1), Some(0));
    }

    #[test]
    fn iter_test() {
        let v = setup();
        let mut it = v.iter();
        assert_eq!(it.next(), Some(0.0));
        assert_eq!(it.next(), Some(3.0));
        assert_eq!(it.next(), Some(1.0));
        assert_eq!(it.next(), Some(-2.0));
        assert_eq!(it.next(), Some(2.0));
        assert_eq!(it.next(), Some(1.0));
        assert!(it.next().is_none());
    }
    #[test]
    fn add_test() {
        let v1: NormMatrix<i32> = NormMatrix::new(3, 2).set_slice(0, 0, &[1, 2, 3, 4, 5, 6]).unwrap();
        let v2: NormMatrix<i32> = NormMatrix::new(3, 2).set_slice(0, 0, &[0, 2, 4, 6, 8, 10]).unwrap();

        let v3 = &v1 + &v2;
        assert_eq!(v3.get(0, 0), Some(1));
        assert_eq!(v3.get(0, 1), Some(4));
        assert_eq!(v3.get(1, 0), Some(7));
        assert_eq!(v3.get(1, 1), Some(10));
        assert_eq!(v3.get(2, 0), Some(13));
        assert_eq!(v3.get(2, 1), Some(16));
        assert_eq!(v1.get(0, 0), Some(1));
        assert_eq!(v2.get(0, 0), Some(0));

        let v3 = v1 + v2;
        assert_eq!(v3.get(0, 0), Some(1));
        assert_eq!(v3.get(0, 1), Some(4));
        assert_eq!(v3.get(1, 0), Some(7));
        assert_eq!(v3.get(1, 1), Some(10));
        assert_eq!(v3.get(2, 0), Some(13));
        assert_eq!(v3.get(2, 1), Some(16));
    }

    #[test]
    #[should_panic]
    fn add_test_panic() {
        let v1: NormMatrix<i32> = NormMatrix::new(2, 2);
        let v2: NormMatrix<i32> = NormMatrix::new(3, 2);
        let _ = v1 + v2; // should panic here due to different size
        assert!(false); // fails test if this is reached.
    }

    #[test]
    fn multiply_test() {
        let a: NormMatrix<i32> = NormMatrix::new(3,3).set_slice(0,0,&[1, 1, 1, 1, 1, 1, 1, 1, 1]).unwrap();
        let b: NormMatrix<i32> = NormMatrix::new (3,1).set_slice(0,0,&[2,2,2]).unwrap();
        
        let result = &a * &b;

        assert_eq!(result.rows(), 3);
        assert_eq!(result.columns(), 1);

        assert_eq!(result.get(0,0), Some(6));
        assert_eq!(result.get(1,0), Some(6));
        assert_eq!(result.get(2,0), Some(6));

        let result = a * b;

        assert_eq!(result.rows(), 3);
        assert_eq!(result.columns(), 1);

        assert_eq!(result.get(0,0), Some(6));
        assert_eq!(result.get(1,0), Some(6));
        assert_eq!(result.get(2,0), Some(6));
    }

    #[test]
    fn iter_test2 () {
        let a: NormMatrix<i32> = NormMatrix::new(3,3).set_slice(0,0,&[1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let mut iter = a.iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.next(), Some(6));
        assert_eq!(iter.next(), Some(7));
        assert_eq!(iter.next(), Some(8));
        assert_eq!(iter.next(), Some(9));
        assert_eq!(iter.next(), None);
    }

}
