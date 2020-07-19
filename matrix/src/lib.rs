use std::default::Default;
use std::error::Error;

use std::iter::FromIterator;
use std::ops::{Index, IndexMut};

pub trait Mat<T>:
    Clone + MutVecRow<T> + VecRow<T> + Index<usize> + IndexMut<usize> + FromIterator<Vec<Option<T>>>
{
    type Output;

    fn get(&self, row: usize, column: usize) -> Result<Option<T>, MatrixError>;
    fn at(&mut self, row: usize, column: usize) -> Result<&mut Option<T>, MatrixError>;

    //Option<&Vec<<Self as Mat<T>>::Output>>;
    fn iter(&self) -> MatrixIterator<T>;
    fn rows(&self) -> usize;
    fn columns(&self) -> usize;

    fn push(&mut self, inp: Vec<Option<T>>);
    fn new(n: usize, m: usize) -> Self;
}

pub trait VecRow<T> {
    fn get_row(&self, row: usize) -> Result<&Vec<Option<T>>, MatrixError>;
}
pub trait MutVecRow<T> {
    fn get_row_mut<'a>(&mut self, row: usize) -> Result<&mut Vec<Option<T>>, MatrixError>;
}

impl<'a, T> MutVecRow<T> for Matrix<T> {
    fn get_row_mut(&mut self, row: usize) -> Result<&mut Vec<Option<T>>, MatrixError> {
        if self.0.len() <= row {
            Err(MatrixError::LocationError)
        } else {
            Ok(&mut self.0[row])
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Matrix<T>(pub Vec<Vec<Option<T>>>);

pub struct MatrixIterator<'a, T> {
    matrix_ref: &'a Matrix<T>,
    index: usize,
}

impl<'a, I> Iterator for MatrixIterator<'a, I> {
    type Item = &'a Vec<Option<I>>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.matrix_ref.0.len() {
            let r = &self.matrix_ref[self.index];
            self.index += 1;

            Some(r)
        } else {
            None
        }
    }
}

impl<T> Index<usize> for Matrix<T> {
    type Output = Vec<Option<T>>;

    fn index(&self, ind: usize) -> &Self::Output {
        &self.0[ind]
    }
}

impl<T> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, ind: usize) -> &mut Self::Output {
        &mut self.0[ind]
    }
}

impl<T> VecRow<T> for Matrix<T> {
    fn get_row(&self, row: usize) -> Result<&Vec<Option<T>>, MatrixError> {
        if self.0.len() <= row {
            Err(MatrixError::LocationError)
        } else {
            Ok(&self.0[row])
        }
    }
}

impl<T: Copy + Clone + Default> Mat<T> for Matrix<T> {
    type Output = T;
    fn get(&self, row: usize, column: usize) -> Result<Option<T>, MatrixError> {
        if self.0.len() <= row {
            Err(MatrixError::LocationError)
        } else if self.0[0].len() <= column {
            Err(MatrixError::LocationError)
        } else {
            Ok(self.0[row][column])
        }
    }

    fn at(&mut self, row: usize, column: usize) -> Result<&mut Option<T>, MatrixError> {
        if self.0.len() <= row {
            Err(MatrixError::LocationError)
        } else if self.0[0].len() <= column {
            Err(MatrixError::LocationError)
        } else {
            Ok(&mut self.0[row][column])
        }
    }

    fn iter(&self) -> MatrixIterator<T> {
        MatrixIterator {
            matrix_ref: self,
            index: 0,
        }
    }

    fn rows(&self) -> usize {
        self.0.len()
    }

    fn columns(&self) -> usize {
        if self.0.len() == 0 {
            0
        } else {
            self.0[0].len()
        }
    }

    fn push(&mut self, inp: Vec<Option<T>>) {
        self.0.push(inp);
    }

    fn new(n: usize, m: usize) -> Self {
        Matrix(vec![vec![None; m]; n])
    }
}

impl<T: Default + Copy + Clone> FromIterator<Vec<Option<T>>> for Matrix<T> {
    fn from_iter<I: IntoIterator<Item = Vec<Option<T>>>>(iter: I) -> Self {
        //let mut m = Matrix::new(0, 0);
        let mut m: Vec<Vec<Option<T>>> = Vec::new();
        for i in iter {
            m.push(i);
        }
        Matrix(m)
    }
}


#[derive(Debug)]
pub enum MatrixError {
    LocationError,
    BuildError,
    TestError,
}
impl Error for MatrixError {}
impl std::fmt::Display for MatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "invalid matrix coordinate")
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    fn setup() -> Matrix<f64> {
        Matrix(vec![
            vec![Some(0.0), Some(3.0)],
            vec![Some(1.0), Some(-2.0)],
            vec![Some(2.0), Some(1.0)],
        ])
    }


    #[test]
    fn modify_matrix() -> Result<(), MatrixError> {
        let mut v = setup();
        let v2 = Matrix(vec![
            vec![Some(1.5), Some(4.5)],
            vec![Some(2.0), Some(-3.4)],
            vec![Some(2.4), Some(1.7)],
        ]);
        *v.at(0, 0)? = v2.get(0, 0)?;
        assert_eq!(v.get(0, 0)?.unwrap(), 1.5);
        assert_eq!(v.get(0, 0)?.unwrap(), v2.get(0, 0)?.unwrap());
        Ok(())
    }

    #[test]
    fn new_test() -> Result<(), MatrixError> {
        let v: Matrix<i32> = Matrix::new(3, 5);
        assert_eq!(v.rows(), 3);
        assert_eq!(v.columns(), 5);

        assert_eq!(v.get(0, 0)?, None);
        assert_eq!(v.get(2, 4)?, None);
        assert!(v.get(3, 5).is_err());
        Ok(())
    }

    #[test]
    fn basic() {
        let v = setup();
        assert_eq!(*v.0.get(0).unwrap().get(0).unwrap(), Some(0.0));
        assert_eq!(v.0.get(10), None);
        assert_eq!(v.get(0,0).unwrap(), Some(0.0));
        match v.get_row(10) {
            Err(MatrixError::LocationError) => assert!(true),
            _ => assert!(false),
        };
    
    }

    #[test]
    fn brackets() {
        let v = setup();
        assert_eq!(v.0[0][1], Some(3.0));
    }

    #[test]
    #[should_panic]
    fn bracket_panic() {
        let v = setup();
        assert!(v.0[10][10].unwrap() == 0.0);
    }

    #[test]
    fn braket_improved() {
        let v = setup();
        assert_eq!(v.get(0, 1).unwrap(), Some(3.0));
    }

    #[test]
    #[should_panic]
    fn bracket_panic_improved() {
        let v = setup();
        assert!(v[10][10].unwrap() == 0.0f64);
    }

    #[test]
    fn bracket_mut() -> Result<(), MatrixError> {
        let mut v = setup();
        v[0][0] = Some(5.0);
        assert_eq!(v.get(0, 0).unwrap(), Some(5.0));
        assert_eq!(v.get(0, 1).unwrap(), Some(3.0));

        let mut v: Matrix<i32> = Matrix::new(2, 2);
        assert!(v.get(1, 1)?.is_none());
        v[1][1] = Some(2);
        assert_eq!(v.get(1, 1)?.unwrap(), 2);
        assert!(v.get(1, 1)?.is_some());
        Ok(())
    }

    #[test]
    #[should_panic]
    fn bracket_mute_panic_improved() {
        let mut v = setup();
        *v.at(10, 10).unwrap() = Some(4.4); // should fail here, passing the test
        assert!(false); // fail test if not failed by here.
    }

    #[test]
    fn get_row_test() -> Result<(), MatrixError> {
        let v = setup();
        let vec = v.get_row(0)?;

        assert_eq!(*vec, vec![Some(0.0), Some(3.0)]);

        // Test none is returned
        match v.get_row(11) {
            Ok(_) => Err(MatrixError::TestError),
            _ => Ok(()),
        }
    }

    #[test]
    fn iter_test() -> Result<(), &'static str> {
        let v = setup();
        let mut it = v.iter();
        assert_eq!(it.next().unwrap()[1], Some(3.0));
        assert_eq!(it.next().unwrap()[0], Some(1.0));
        assert_eq!(it.next().unwrap()[1], Some(1.0));
        assert!(it.next().is_none());
        Ok(())
    }

    #[test]
    fn intermediate_iter_test() -> Result<(), &'static str> {
        let v = setup();
        let iv: Vec<usize> = v.iter().map(|v| v.len()).collect();
        assert_eq!(iv[0], 2);
        assert_eq!(iv[1], 2);
        assert_eq!(iv[2], 2);
        assert!(iv.get(3).is_none());
        assert_eq!(v.get(2, 1).unwrap(), Some(1.0));
        Ok(())
    }

    #[test]
    fn from_iterator_test() -> Result<(), MatrixError> {
        // Matrix(vec![vec![0.0, 3.0], vec![1.0, -2.0], vec![2.0, 1.0]])
        let v = setup();
        let v2: Matrix<f64> = v
            .iter()
            .map(|x| vec![Some(x[0].unwrap() * 2.0), Some(x[1].unwrap() * 3.0)])
            .collect();
        assert!(v2.get(0, 0)?.unwrap().abs() < 1e-10);
        assert!((v2.get(0, 1)?.unwrap() - 9.0).abs() < 1e-10);
        assert!((v2.get(2, 0)?.unwrap() - 4.0).abs() < 1e-10);
        assert!((v2.get(1, 1)?.unwrap() + 6.0).abs() < 1e-10);
        assert_eq!(v2.rows(), 3);
        assert_eq!(v2.columns(), 2);
        Ok(())
    }
}
/***
// #[cfg(test)]
// mod more_test {


// }





    // #[ignore]
    // fn cubic_test() {
    //     let input: Matrix<f64> = vec![vec![0.0, 3.0], vec![1.0, -2.0], vec![2.0, 1.0]];
    //     let result = cubic_spline(&input).unwrap();
    //     assert_eq!(input.len() - 1, result.len());
    //     println!("{:#?}", result);
    //     assert!((result[0].coeff[0] - 3.0).abs() < 1e-10);
    //     assert!((result[0].coeff[1] + 7.0).abs() < 1e-10);
    //     assert!((result[0].coeff[2] + 0.0).abs() < 1e-10);
    //     assert!((result[0].coeff[3] - 2.0).abs() < 1e-10);
    //     assert!((result[1].coeff[0] + 2.0).abs() < 1e-10);
    //     assert!((result[1].coeff[1] + 1.0).abs() < 1e-10);
    //     assert!((result[1].coeff[2] - 6.0).abs() < 1e-10);
    //     assert!((result[1].coeff[3] + 2.0).abs() < 1e-10);
    // }

**/
