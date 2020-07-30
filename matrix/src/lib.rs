use std::default::Default;
use std::ops::{Add, Index, IndexMut, Mul};
/// Trait that incorporates other traits that are needed for proper matrix operations.
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

/// Matrix struct.  Contains the data stored in a one-dimension vector, and values for rows and columns.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct NormMatrix<T: Element> {
    cells: Vec<T>,
    m_rows: usize,
    m_columns: usize,
}


pub struct NormMatrixIterator<'a, T: Element> {
    matrix_ref: &'a NormMatrix<T>,
    index: usize,
}

impl <T: Element> NormMatrix<T> {
    fn iter(&self) -> NormMatrixIterator<T> {
        NormMatrixIterator {
            matrix_ref: self,
            index: 0,
        }
    }
}

// pub trait MatrixIterator: Iterator {}
// impl <T: Element> MatrixIterator for  NormMatrixIterator<'_, T> {}

impl <T: Element> Iterator for NormMatrixIterator<'_, T> {
    type Item = T;
    fn next(&mut self) -> Option<T>{
        self.index += 1;
        match self.matrix_ref.cells.get(self.index - 1) {
            None => None,
            Some(v) => Some(*v),
        }
    }
    
}

impl<T: Element> Index<usize> for NormMatrix<T> {
    type Output = T;

    /// Returns value at index of matrix vector.  Panics! if out of bounds
    fn index(&self, ind: usize) -> &Self::Output {
        &self.cells[ind]
    }
}

impl<T: Element> IndexMut<usize> for NormMatrix<T> {
    /// Returns mutable value at index of matrix vector.  Panics! if out of bounds.
    fn index_mut(&mut self, ind: usize) -> &mut Self::Output {
        &mut self.cells[ind]
    }
}

impl<T: Element> NormMatrix<T> {
    // Private converison function
    fn get_cell(&self, row: usize, col: usize) -> usize {
        row * self.m_columns + col
    }
}

impl<T: Element + 'static> Matrix<T> for NormMatrix<T> {
    type Output = T;

    fn get(&self, row: usize, column: usize) -> Option<T> {
        if row >= self.rows() || column >= self.columns() {
            None
        } else {
            let v = self.cells.get(NormMatrix::get_cell(self, row, column)).unwrap();
            Some(*v)
        }
    }

    fn set<'a>(&mut self, row: usize, column: usize, value: T) -> Option<T> {
        let cell = NormMatrix::get_cell(&self, row, column);
        if let Some(change) = self.cells.get_mut(cell) {
            *change = value;
            return Some(value);
        }
        None
    }

    fn set_slice(& self, start_row: usize, start_col: usize, slice: &[T]) -> Option<NormMatrix<T>> {
        let start_cell = NormMatrix::get_cell(self, start_row, start_col);
        // Returns None if out of bounds.
        if start_cell + slice.len() > self.cells.len() {
            return None;
        }

        let mut new_cells: Vec<T> = self.cells.clone();
        


        new_cells.splice(
            start_cell..(start_cell + slice.len()),
            slice.iter().cloned(),
        );
        Some(NormMatrix {
            m_rows: self.rows(),
            m_columns: self.columns(),
            cells: new_cells,
        })
    }

    fn rows(&self) -> usize {
        self.m_rows
    }

    fn columns(&self) -> usize {
        self.m_columns
    }

    fn new(rows: usize, columns: usize) -> Self {
        NormMatrix {
            cells: vec![T::default(); rows * columns],
            m_rows: rows,
            m_columns: columns,
        }
    }
}

/// Implements add for Matrix.  
/// Returns the sum of the two matrices
/// Panics if the matrices are not the same size
impl<T: Element + 'static> Add for NormMatrix<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        if (self.rows() != rhs.rows()) || (self.columns() != rhs.columns()) {
            panic!("Cannot add different sizes of `Matrix` together");
        }

        let mut result = Self::new(self.rows(), self.columns());
        for (index, lhs) in self.cells.iter().enumerate() {
            result[index] = *lhs + rhs.cells[index];
        }
        result
    }
}

/// Implements add for Matrix.  
/// Returns the sum of the two matrices
/// Panics if the matrices are not the same size
impl<T: Element + 'static> Add for &NormMatrix<T> {
    type Output = NormMatrix<T>;

    fn add(self, rhs: Self) -> NormMatrix<T> {
        if (self.rows() != rhs.rows()) || (self.columns() != rhs.columns()) {
            panic!("Cannot add different sizes of `Matrix` together");
        }
        let mut result = NormMatrix::new(self.rows(), self.columns());
        for (index, lhs) in self.cells.iter().enumerate() {
            result[index] = *lhs + rhs.cells[index];
        }
        result
    }
}

/// Multiplies two matrices together
/// returns product of the two matrices.
/// # Example
/// ```no_run
/// /*             | 1 1 |
/// * Assume lhs = | 2 1 |
/// *              | 3 2 |
/// *
/// *              | 0 1 2 |
/// *        rhs = | 1 3 1 |
/// *
/// */
/// let mut a: Matrix<i32> = Matrix::new (3,2);
/// a = 
/// let result = multiply(&lhs, &rhs);
///
/// assert_eq!(big_x[0][0], 1);
/// assert_eq!(big(x[1][1], 5);
///
/// /*              |1  4  3|
/// * with result = |1  5  5|
/// *               |2  9  8|
/// */
/// ```
/// With the LHS matrix being an `a x b` sized matrix, and RHS being a `b x c` matrix, then:  
/// This function has `O(a * b * c)` time complexity.  
/// This function has `O(a * c)` spatial complexity.
/// Panics if the lhs matrix columns are not equal to the rhs matrix rows.
impl<T: Element + 'static> Mul for NormMatrix<T> {
    type Output = NormMatrix<T>;

    fn mul(self, rhs: NormMatrix<T>) -> NormMatrix<T> {
        // Number of lhs columns need to equal number of rhs rows
        if self.columns() != rhs.rows() {
            panic!("Invalid sizes. Cannot multiply a `Matrix` with {} columns by a `Matrix` with {} rows", self.columns(), rhs.rows());
        }
        let row = self.rows();
        let col = rhs.columns();

        // Creates empty matrix of appropiate size, allows for direct changing elements.
        // Size is lhs.rows by rhs.columns
        let mut result: NormMatrix<T>= NormMatrix::new(row, col);

        for i in 0..row {
            for j in 0..col {
                for k in 0..self.columns() {
                    let val = result.get(i, j).unwrap()
                        + self.get(i, k).unwrap() * rhs.get(k, j).unwrap();
                    result.set(i, j, val);
                }
            }
        }
        //println!("{:?}", result);
        result
    }
}

impl<T: Element + 'static> Mul for &NormMatrix<T> {
    type Output = NormMatrix<T>;

    fn mul(self, rhs: &NormMatrix<T>) -> NormMatrix<T> {
        // Number of lhs columns need to equal number of rhs rows
        if self.columns() != rhs.rows() {
            panic!("Invalid sizes. Cannot multiply a `Matrix` with {} columns by a `Matrix` with {} rows", self.columns(), rhs.rows());
        }
        let row = self.rows();
        let col = rhs.columns();

        // Creates empty matrix of appropiate size, allows for direct changing elements.
        // Size is lhs.rows by rhs.columns
        let mut result: NormMatrix<T>= NormMatrix::new(row, col);

        for i in 0..row {
            for j in 0..col {
                for k in 0..self.columns() {
                    let val = result.get(i, j).unwrap()
                        + self.get(i, k).unwrap() * rhs.get(k, j).unwrap();
                    result.set(i, j, val);
                }
            }
        }
        //println!("{:?}", result);
        result
    }
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
