use std::ops::{Index, IndexMut, Add, Mul};
use crate::Matrix;
use crate::element::Element;

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
    pub fn iter(&self) -> NormMatrixIterator<T> {
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