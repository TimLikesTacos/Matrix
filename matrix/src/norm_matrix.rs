use crate::element::Element;
use crate::norm_operations::{Augment, GaussianEliminateSteps, Transpose};
use crate::Matrix;
use std::ops::{Add, Index, IndexMut, Mul};


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

impl<T: Element> NormMatrix<T> {
    pub fn iter(&self) -> NormMatrixIterator<T> {
        NormMatrixIterator {
            matrix_ref: self,
            index: 0,
        }
    }
}

// pub trait MatrixIterator: Iterator {}
// impl <T: Element> MatrixIterator for  NormMatrixIterator<'_, T> {}

impl<T: Element> Iterator for NormMatrixIterator<'_, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
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
            let v = self
                .cells
                .get(NormMatrix::get_cell(self, row, column))
                .unwrap();
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

    //todo! fn get_slice (&self, ))

    fn set_slice(&self, start_row: usize, start_col: usize, slice: &[T]) -> Option<NormMatrix<T>> {
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
/// ```rust
/// /*             | 1 1 |
/// * Assume lhs = | 2 1 |
/// *              | 3 2 |
/// *
/// *              | 0 1 2 |
/// *        rhs = | 1 3 1 |
/// *
/// */
/// use matrix::{NormMatrix, Matrix};
/// let a: NormMatrix<i32> = NormMatrix::new (3,2).set_slice(0,0,&[1,1,2,1,3,2]).unwrap();
/// let b: NormMatrix<i32> = NormMatrix::new (2,3).set_slice(0,0,&[0,1,2,1,3,1]).unwrap();
/// let result = a * b;
///
/// assert_eq!(result.get(0,0).unwrap(), 1);
/// assert_eq!(result.get(1,1).unwrap(), 5);
/// assert_eq!(result.get(2,1).unwrap(), 9);
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
        let mut result: NormMatrix<T> = NormMatrix::new(row, col);

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
        let mut result: NormMatrix<T> = NormMatrix::new(row, col);

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

impl<T: Element + 'static, M: Matrix<T>> Augment<T, M> for NormMatrix<T> {
    fn augment(&self, other: M) -> Self {
        let mut to_return: NormMatrix<T> = NormMatrix::new(self.rows(), self.columns() + 1);

        // todo!("Implement get_slice, improve readability / efficiency here")

        for (i, cell) in self.iter().enumerate() {
            // Copy self to to_return
            to_return.set(i / self.columns(), i % self.columns(), cell);
            // If at end of a row, add the element from the 'other' vector
            if i % self.columns() == self.columns() - 1 {
                let val = other.get(i / self.columns(), 0).unwrap();
                to_return.set(i / self.columns(), to_return.columns() - 1, val);
            }
        }
        to_return
    }
}

impl<T: Element + 'static> Transpose<T> for NormMatrix<T> {
    fn transpose(&self) -> Self {
        let mut to_return = NormMatrix::new(self.columns(), self.rows());

        for i in 0..to_return.rows() {
            for j in 0..to_return.columns() {
                to_return.set(i, j, self.get(j, i).unwrap());
            }
        }
        to_return
    }
}

impl<T: Element + 'static> GaussianEliminateSteps<T> for NormMatrix<T> {
    fn pivot(&mut self, row: usize) {

        fn swap<T2: Element>(
            m: &mut NormMatrix<T2>,
            (a, b): (usize, usize),
            (j, k): (usize, usize),
        ) {
            // How to swap two elements from a vector code found here on Stackoverflow:
            // https://stackoverflow.com/questions/25531963/how-can-i-swap-items-in-a-vector-slice-or-array-in-rust
            // Can't take two mutable loans from one vector, so instead just cast
            // them to their raw pointers to do the swap
            let pa: *mut T2 = m.index_mut(m.get_cell(a, b));
            let pb: *mut T2 = m.index_mut(m.get_cell(j, k));

            unsafe {
                std::ptr::swap(pa, pb);
            }

        }
        let largest = find_largest_row_by_col(self, row);

        for col in 0..self.columns() {
            swap(self, (row, col), (largest, col));
        }

        /// Returns the row number that has the highest absolute value
        /// in that column.  Helps minimize rounding error.
        /// This function has a `O(n)` time complexity where `n` is the number of rows of the matrix.
        /// This function has a `O(1)` spatial complexity.
        fn find_largest_row_by_col<M: Matrix<T>, T: Element>(m: &M, i: usize) -> usize {
            let mut largest = m.get(i, i).unwrap_or_else(|| T::zero());
            let mut largest_row = i;
            for j in (i + 1)..m.rows() {
                let mut val: T = match m.get(j, i) {
                    Some(v) => v,
                    None => return largest_row,
                };
                if val < T::zero() {
                    val = T::zero() - val;
                }

                if val > largest {
                    largest = val;
                    largest_row = j;
                }
            }
            largest_row
        }
    }

    fn scale(&mut self, row: usize) {
        let factor: T;
        match self.get(row, row) {
            Some(v) => {
                factor = v;
            }
            None => panic!("Unable to perform elimination due to matrix size error"),
        }
        // factor is 1 / get(i,i)
        let factor: T = T::one() / factor;
        for j in 0..self.columns() {
            self.set(
                row,
                j,
                match self.get(row, j) {
                    Some(s) => (s * factor),
                    None => break,
                },
            );
        }
    }

    fn eliminate(&mut self, row: usize) {
        // for each row below row i
        for r in row + 1..self.rows() {
            // factor is m[r][i] / m[i][i], however m[i][i] == 1 from scaling function
            let factor = self.get(r, row).unwrap();
            for k in 0..self.columns() {
                self.set(
                    r,
                    k,
                    match self.get(r, k) {
                        Some(s) => s - factor * self.get(row, k).unwrap(),
                        None => break,
                    },
                );
            }
        }
    }

    fn back_solve(&mut self, row: usize) {
        for r in (0..row).rev() {
            let factor = self.get(r, row).unwrap();
            for k in 0..self.columns() {
                self.set(
                    r,
                    k,
                    match self.get(r, k) {
                        Some(s) => s - factor * self.get(row, k).unwrap(),
                        None => break,
                    },
                );
            }
        }
    }
}
