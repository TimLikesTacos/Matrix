
use matrix::{Mat, MatrixError, MutVecRow, Matrix};
use std::ops::{Add, Sub, Mul, Div};
use num_traits::{FromPrimitive, Zero};
extern crate matrix;


pub fn solve<M: std::fmt::Debug + Mat<T> + Clone, T>(input: &M, poly: usize) -> Result<Vec<T>, MatrixError>
where
    T: Add<Output = T>
        + Div<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Zero
        + Copy
        + Clone 
        + Default
        + FromPrimitive
        + PartialOrd,
{
    // Build the matrices from the inputs to calculate the XTX|XTY
    let x = build_x::<M, T>(input, poly);
    let y = build_y::<M, T>(input)?;
    let xt = transpose::<M, T>(&x);
    let xtx = multiply::<M, T>(&xt, &x)?;
    let xty = multiply::<M, T>(&xt, &y)?;

    // Creates augmented matrix from XTX and XTY
    let mut xtx_xty = augment::<M, T>(xtx, xty)?;
    gaussian_elim(&mut xtx_xty)
    
}

/// Performs Gausian elimination and returns a vector of coefficients
///
pub fn gaussian_elim<
    M: Mat<T>, T: Zero + Default + Copy + Mul<Output = T> + Div<Output = T> + Sub<Output = T> + PartialOrd + FromPrimitive,
>(
    inp: &mut M,
) -> Result<Vec<T>, MatrixError> {
    // Reduces the XTX portion of the marix to an identity matrix.
    for i in 0..inp.rows() {
        pivot(inp, i);
        scale(inp, i)?;
        eliminate(inp, i)?;
        back_solve(inp, i)?;
    }
    // Returns a vector of the coefficients
    Ok(inp.iter().map(|x| x[x.len() - 1].unwrap()).collect())
}

/// Returns the X matrix that is same size as the input matrix, dependent on values of x.  
/// Second paramter is for the amount of terms for the polynomial.
/// # Definition of values:
/// for each value of x:  
/// first column: y = x^0;  
/// second column y = x^1;  
/// third column: y = x^2;  
/// ...
/// with number of rows determined by the number of x values.
///
///
/// # Example
/// ```rust
/// /*           | 1 2 |
/// * Assume x = | 4 5 |
/// *            | 7 8 |
/// * where the first column is x and the second column is f(x)
/// */
///
/// let big_x = build_x(&x, 2_i32);
///
/// assert_eq!(big_x[0], 2);
/// assert_eq!(big(x[1], 5);
///
/// /*             |1 1  1|
/// * with big_x = |1 4 16|
/// *              |1 7 49|
/// */
/// ```
/// This function has `O(x * n)` time and spatial complexity, where `x` is the number of points and `n` is number of polynomial terms.  
fn build_x<M: Mat<T> + std::fmt::Debug, T: FromPrimitive + Default + Copy + Mul<Output = T>>(x: &M, poly: usize) -> M {
    
    let pow = |x, n| {
        let mut start: T = x;
        for _ in 2..=n {
            start = start * x;

        }
        start
    };
    let mut to_return: M = M::new(0,0);
    for the_row in x.iter() {
        let mut row_vec: Vec<Option<T>> = Vec::new();
        
        for j in 0..=poly {
            let mut val: T = the_row[0].unwrap_or_else(||T::default());
            if j == 0 {
                row_vec.push(Some(T::from_usize(1).unwrap()));
                continue;
            }
            
            val = pow(val, j);
            row_vec.push(Some(val));
        }
        to_return.push(row_vec);
    }

    to_return
}

/// Returns the Y matrix from a matrix containing x and f(x).
/// # Example
/// ```rust
/// /*           | 1 2 |
/// * Assume x = | 4 5 |
/// *            | 7 8 |
/// * where the first column is x and the second column is f(x)
/// */
///
/// let y = build_y(&x);
///
/// assert_eq!(y[0][0], 2);
/// assert_eq!(y[1][0], 5);
///
/// /*         |2|
/// * with y = |5|
/// *          |8|
/// */
/// ```
/// 
///This function has `O(x)` spatial and time complexity, where `x` is the number of points.
fn build_y<M: Mat<T>, T: Default + Copy>(x: &M) -> Result<M, MatrixError> {
    let mut y =  M::new(x.rows(), 1);
    for i in 0..x.rows() {
        *y.at(i,0)? =   Some(x.get(i,1)?.unwrap_or_else(||T::default()));
    }
    Ok(y)
}

/// Returns transpose of the input matrix
/// # Example
/// ```rust
/// /*           | 1 2 |
/// * Assume x = | 4 5 |
/// *            | 7 8 |
/// */
///
/// let x_t = transpose(&x);
///
/// assert_eq!(xt[0][2], 7);
/// assert_eq!(xt[1][1], 5);
///
/// /*           |1 4 7|
/// * with x_t = |2 5 8|
/// *           
/// */
/// ```
/// This function has `O(x * n)` time and spatial complexity, where `x` is the number of points and `n` is number of polynomial terms.
fn transpose<M: Mat<T>, T: Default + Copy>(x: &M) -> M {
    let mut to_return: M = M::new(x.columns(), x.rows());

    for i in 0..to_return.rows() {
        for j in 0..to_return.columns() {
            *to_return.at(i,j).unwrap() = Some(x.get(j,i).unwrap().unwrap_or_else(||T::default()));
        }
    }
    to_return
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
///
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
pub fn multiply<M: Mat<T> + std::fmt::Debug, T: Default + Copy + Add<Output = T> + Mul<Output = T>>(
    lhs: &M,
    rhs: &M,
) -> Result<M, MatrixError> {
    let row = lhs.rows();
    let col = rhs.columns();

    // Creates empty matrix of appropiate size, allows for direct changing elements.
    // Size is lhs.rows by rhs.columns
    let mut result: M = M::new(row, col);

    for i in 0..row {
        for j in 0..col {
            for k in 0..lhs.columns() {
                if result.get(i,j)?.is_none() && 
                    lhs.get(i,k)?.is_none()&&
                    rhs.get(k,j)?.is_none() {
                        continue;
                    } 
                *result.at(i,j)? = Some(result.get(i,j)?.unwrap_or_else(||T::default()) + lhs.get(i,k)?.unwrap_or_else(||T::default()) * rhs.get(k,j)?.unwrap_or_else(||T::default()));
            }
        }
    }
    println!("{:?}", result);
    Ok(result)
}

/// Finds the largest value in a column, and swaps rows.
fn pivot<M: Mat<T>, T: Sub<Output = T> + Zero + Copy + PartialOrd + FromPrimitive + Mul<Output = T>>(m: &mut M, i: usize) {
    let largest = match find_largest_row_by_col(m, i) {
        Ok(l) => l,
        Err(_) => 0,
    };
    //m.0.swap::<Option<T>>(i, largest);
    // let temp = m.get_row(i);
    // let tempa = &m.get_row(i);
    // tempa = m.get_row(largest)?;
    // tempb = &m.get_row(largest);
    // tempb = temp;

    // Can't take two mutable loans from one vector, so instead just cast
    // them to their raw pointers to do the swap
    let pa: *mut Vec<Option<T>> = &mut (*m.get_row_mut(i).unwrap());
    let pb: *mut Vec<Option<T>> = &mut( *m.get_row_mut(largest).unwrap());
    unsafe {std::ptr::swap(pa, pb);}

}

/// Returns the row number that has the highest absolute value
/// in that column.  Helps minimize rounding error.
/// This function has a `O(n)` time complexity where `n` is the number of rows of the matrix.
/// This function has a `O(1)` spatial complexity.
fn find_largest_row_by_col<M: Mat<T>, T: Sub<Output = T>+ Zero + Copy + PartialOrd + FromPrimitive + Mul<Output = T>>(
    m: &mut M,
    i: usize,
) -> Result<usize, MatrixError> {
    let mut largest = m.get(i,i)?.unwrap();
    let mut largest_row = i;
    for j in i + 1..m.rows() {
        let mut val: T = match m.get(j,i)? {
            Some(v) => v,
            None => return Ok(largest_row),
        };
        if val < T::from_usize(0).unwrap() {
            val = T::zero() - val;
        }
       
        if val > largest {
            largest = val;
            largest_row = j;
        }
    }
    Ok(largest_row)
}

/// Scales a row by factoring each column by the inverse of the index value passed in.
/// # Example
/// ```
/// /*              |2  4  8|
/// * with input  = |1  5  5|
/// *               |2  9  8| */
///
/// scale(&input, 1)
/// ```
/// results in
/// ```
/// /*              |1  2  4|
/// * with result = |1  5  5|
/// *               |2  9  8| */
/// ```
/// This function has a `O(m)` time complexity where `m` is the number of columns of the matrix.
/// This function has a `O(1)` spatial complexity
fn scale<M: Mat<T>, T: Mul<Output = T> + Div<Output = T> + FromPrimitive + Copy + Clone>(
    m: &mut M,
    i: usize,
) -> Result<(), MatrixError>{
    let factor: T = T::from_usize(1).unwrap() / m.get(i,i)?.unwrap();
    for j in 0..m.columns() {
        
        *m.at(i,j)? = match m.get(i,j)?{
            Some(s) => Some(s * factor),
            None => break,
        };
    }
    Ok(())
}

/// Makes cells below the previously scaled value zeros.
///
/// # Example
/// ```
/// /*              |1  2  4|
/// * with input  = |1  5  5|
/// *               |2  9  8| */
///
/// eliminate(&input, 0)
/// ```
/// results in
/// ```
/// /*              |1  2  4|
/// * with result = |0  3  1|
/// *               |0  5  0| */
/// ```
/// This function has a `O(n*m)` time complexity, where `n` is the number of rows of the matrix and `m` is the number of columns.  
/// This function has a `O(1)` spatial complexity.
fn eliminate<M: Mat<T>, T: Copy + Sub<Output = T> + Mul<Output = T>>(m: &mut M, i: usize) -> Result<(), MatrixError> {
    // for each row below row i
    for r in i + 1..m.rows() {
        // factor is m[r][i] / m[i][i], however m[i][i] == 1 from scaling function
        let factor = m.get(r,i)?.unwrap();
        for k in 0..m.columns() {
            *m.at(r,k)? = match m.get(r,k)?{
                Some (s) => Some(s - factor * m.get(i,k)?.unwrap()),
                None => break,
            } 
        }
    }
    Ok(())
}

/// Makes cells above the called index zeros.
///
/// # Example
/// ```
/// /*              |1  2  4|
/// * with input  = |0  1  1|
/// *               |0  0  8| */
///
/// eliminate(&input, 1)
/// ```
/// results in
/// ```
/// /*              |1  0  2|
/// * with result = |0  1  1|
/// *               |0  0  8| */
/// ```
/// This function has a `O(n*m)` time complexity, where `n` is the number of rows of the matrix and `m` is the number of columns.
/// This function has a `O(1)` spatial complexity.
fn back_solve<M: Mat<T>, T: Default + Copy + Sub<Output = T> + Mul<Output = T>>(m: &mut M, i: usize) -> Result<(), MatrixError>{
    for r in (0..i).rev() {
        let factor = m.get(r,i)?.unwrap();
        for k in 0..m.columns() {
            *m.at(r,k)? = match m.get(r,k)? {
                Some(s) => Some(s - factor * m.get(i,k)?.unwrap()),
                None => Some(T::default() - factor * m.get(i,k)?.unwrap()),
            };
        }

    }
    Ok(())
}

/// Create augmented matrix
/// Adds the `b` matrix to the end of the `a` matrix
/// # Example
/// ```no_run
/// let a: Matrix<f64> = vec![vec![1.0, 1.0], vec![2.2, 3.3]];
/// let b: Matrix<f64> = vec![vec![8.0], vec![9.1]];
/// let ab = augment(a, b);
/// let expected_r0 = vec![1.0, 1.0, 8.0];
/// assert_eq!(ab[0], expected_r0);
/// ```
/// This function has a `O(n*m)` time complexity, where `n` is the number of rows of the lhs matrix and `m` is the number of columns of the rhs matrix.  This value for `m` typically will be 1.
/// For limited situations where that is true, this can be simplified to `O(n)`.  
/// This function has a `O(m*w + m*n)` spatial complexity, where `w` is the number of columns of the LHS.
pub fn  augment<'a, M: Clone + Mat<T> + MutVecRow<T>, T: Clone + Copy>(x:  M, b: M) -> Result<M, MatrixError> {
    
    let mut a = x.clone();
    for r in 0..b.rows() {
        for c in 0..b.columns() {
            // let mut item: &Option<T> = match a.get_row(r)?{
            //     Some(s) => s,
            //     None => 
            // };
            // (*row).push(b.get(r,c)?);
            let res: &mut Vec<Option<T>> = match a.get_row_mut(r) {
                Err(e) => return Err(e),
                Ok(s) => s,
            };
            
            res.push(b.get(r,c)?);
        
        }
    }
    Ok(a)
}

#[cfg(test)]
mod operations_tests {

    use super::*;
    // fn setup() -> Matrix<f64> {
    //     Matrix(vec![vec![Some(0.0), Some(3.0)], vec![Some(1.0), Some(-2.0)], vec![Some(2.0), Some(1.0)]])
    // }

     #[test]
    fn solve_test() -> Result<(), MatrixError>{

        let test: Matrix<f64> = create_test_matrix();
        let x = build_x(&test, 1);
        let y = build_y(&test)?;
        let xt = transpose(&x);
        let xtx = multiply(&xt, &x)?;
        let xty = multiply(&xt, &y)?;
        let mut xtxxty = augment(xtx, xty)?;
        //println!("before elim: {:?}", xtxxty);
        gaussian_elim(&mut xtxxty)?;

        //println!("after elim: {:?}", xtxxty);

        let result = solve(&test, 1).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result[0] + 15.0 < 1e-10);
        assert!(result[1] - 10.0 < 1e-10);

        Ok(())
    }

    /// The function for mapping values of x to y
    /// Where `f(x) = x ^ 2`
    /// Returns the value of f(x)
    /// # Example
    /// ```
    /// let result = f_x(3);
    /// assert_eq!(result, 9);
    /// ```

    fn f_x<T: Copy + Mul<Output = T> + FromPrimitive>(x: T) -> T {
        x * x
    }

    // / Returns a matrix with two columns, one for x and the other for f(x),
    // / with rows determined by the amount of points which is currently set for \[0, 10\].
    // / # Example
    // / ```rust
    // / //f(x) = x ^ 2
    // / let x: Matrix = create_test_matrix ();
    // /
    // / assert_eq!(x[0][2], 2);
    // / assert_eq!(x[1][2], 4);
    // / ```

    fn create_test_matrix<M: Mat<T>, T: FromPrimitive>() -> M {
        let mut matrix: M = M::new(0,0);
        for i in 0..=10 {
            matrix.push(vec![
                Some(T::from_usize(i).unwrap()),
                Some(T::from_usize(f_x(i)).unwrap()),
            ]);
        }
        matrix
    }

    #[test]
    fn build_x_test () -> Result<(), MatrixError> {
        let m: Matrix<i32> = create_test_matrix();
        let x = build_x(&m, 1);
        assert_eq!(x.rows(), 11);
        assert_eq!(x.columns(), 2);
        assert_eq!(x.get(0,0)?.unwrap(), 1);
        assert_eq!(x.get(0,1)?.unwrap(), 0);
        assert_eq!(x.get(9,1)?.unwrap(), 9);
        match x.get(11,0) {
            Err(MatrixError::LocationError) => (),
            _ => assert!(false),
            
        }
        // 2 degree polynomial line
        let x = build_x(&m,2);
        assert_eq!(x.rows(), 11);
        assert_eq!(x.columns(), 3);
        assert_eq!(x.get(0,2)?.unwrap(), 0);
        assert_eq!(x.get(3,2)?.unwrap(), 9);
        assert_eq!(x.get(9,2)?.unwrap(), 81);
        match x.get(11,0) {
            Err(MatrixError::LocationError) => (),
            _ => assert!(false),
            
        }


        Ok(())
    }

    #[test]
    fn transpose_test () -> Result<(), MatrixError> {
        let m: Matrix<i32> = create_test_matrix();
        let x = build_x(&m, 2);
        let xtx = transpose(&x);
        
        // println!("{:?}",x);
        // println!("{:?}", xtx);
        assert_eq!(x.rows(), xtx.columns());
        assert_eq!(x.columns(), xtx.rows());
        assert_eq!(xtx.get(0,9)?.unwrap(), 1);
        assert_eq!(xtx.get(2,2)?.unwrap(), 4);
        Ok(())
    }

    #[test]
    fn buld_y_test() -> Result<(), MatrixError> {
        
        let m: Matrix<i32> = create_test_matrix();
        let y = build_y(&m)?;
        assert_eq!(y.rows(), 11);
        assert_eq!(y.columns(), 1);
        assert_eq!(y.get(0,0)? , Some(0));
        assert_eq!(y.get(9,0)?, Some(81));
        match y.get(11,0) {
            Err(MatrixError::LocationError) => (),
            _ => assert!(false),
        } 
        Ok(())
    }

    #[test]
    fn multiply_test() -> Result<(), MatrixError> {
        let m: Matrix<i32> = create_test_matrix();
        let x = build_x(&m, 1);
        let xt = transpose(&x);
        let xtx = match multiply(&xt, &x) {
            Ok(m) => m,
            Err(e) => {println!("Error in multiplicaiton function: {}", e); return Err(e);}
        };

        //println!("Test: {:?}", xtx);

        assert_eq!(xtx.columns(), x.columns());
        assert_eq!(xtx.rows(), xt.rows());
        assert_eq!(xtx.get(0,0)?, Some(11));
        
        let x : Matrix<i32> = Matrix::new(2,2);
        let y : Matrix<i32> = Matrix::new(2,1);
        let xy = multiply(&x, &y)?;
        assert_eq!(xy.rows(), 2);
        assert_eq!(xy.columns(), 1);
        assert_eq!(xy.get(0,0)?, None);
        assert_eq!(xy.get(1,0)?, None);

        let y: Matrix<i32> = Matrix::new(1,1);
        match multiply(&x, &y) {
            Err(MatrixError::LocationError) => (),
            Err(_) => { println!("Incorrect error returned"); return Err(MatrixError::TestError);},
            Ok(_) => { println!("Should have produced error"); return Err(MatrixError::TestError);},
        }

        Ok(())
    }

    #[test]
    fn pivot_test () -> Result<(), MatrixError> {
        let mut m = Matrix(vec![vec![Some(2), Some(4)], vec![Some(1),Some(7)], vec![Some(1), Some(9)]]);
        pivot(&mut m,0);
        assert_eq!(m.get(0,0)?, Some(2));
        assert_eq!(m.get(1,1)?, Some (7));

        pivot (&mut m, 1);
        assert_eq!(m.get(1,0)?, Some(1));
        assert_eq!(m.get(1,1)?, Some(9));
        assert_eq!(m.get(2,0)?, Some(1));
        assert_eq!(m.get(2,1)?, Some(7));

        let mut m = Matrix(vec![vec![Some(1), None], vec![Some(2),Some(7)]]);
        pivot (&mut m, 0);
        assert_eq!(m.get(0,0)?, Some(2));
        assert_eq!(m.get(1,1)?, None);
        Ok(())

    }

    #[test]
    fn scale_test () -> Result<(), MatrixError> {
        let mut m:Matrix<f64> = Matrix(vec![vec![Some(2.0), Some(4.0), Some(2.0)], vec![Some(1.0),Some(7.0), Some(15.0)],
             vec![Some(1.0), None, Some(11.0)]]);
        scale(&mut m, 0)?;
        assert_eq!(m.get(0,0)?.unwrap(), 1.0);
        assert_eq!(m.get(0,1)?.unwrap(), 2.0);
        assert_eq!(m.get(0,2)?.unwrap(), 1.0);
        assert_eq!(m.get(1,1)?.unwrap(), 7.0);
        assert_eq!(m.get(2,2)?.unwrap(), 11.0);

        scale (&mut m, 1)?;
        assert_eq!(m.get(0,0)?.unwrap(), 1.0);
        assert_eq!(m.get(0,1)?.unwrap(), 2.0);
        assert_eq!(m.get(0,2)?.unwrap(), 1.0);
        assert_eq!(m.get(1,1)?.unwrap(), 1.0);
        assert_eq!(m.get(1,2)?.unwrap(), 15.0/7.0);
        assert_eq!(m.get(2,2)?.unwrap(), 11.0);

        //println!("{:?}", m);
        scale (&mut m, 2)?;
        //println!("{:?}", m);
        assert_eq!(m.get(0,0)?.unwrap(), 1.0);
        assert_eq!(m.get(0,1)?.unwrap(), 2.0);
        assert_eq!(m.get(0,2)?.unwrap(), 1.0);
        assert_eq!(m.get(1,1)?.unwrap(), 1.0);
        assert_eq!(m.get(1,2)?.unwrap(), 15.0/7.0);
        assert_eq!(m.get(2,0)?.unwrap(), 1.0/11.0);
        assert!(m.get(2,1)?.is_none());
        assert_eq!(m.get(2,2)?.unwrap(), 11.0);

        Ok(())
    }

    #[test]
    fn eliminate_test () -> Result<(), MatrixError> {
        let mut m:Matrix<f64> = Matrix(vec![vec![Some(1.0), Some(4.0), Some(2.0)], vec![Some(2.0),Some(7.0), Some(15.0)],
        vec![Some(3.0), None, Some(11.0)]]);

        eliminate (&mut m, 0)?;

        //println!("{:?}", m);
        assert_eq!(m.get(0,1)?.unwrap(), 4.0);
        assert!((m.get(1,0)?.unwrap()).abs() < 1e-10);
        assert!((m.get(1,1)?.unwrap() + 1.0).abs() < 1e-10);
        assert!((m.get(1,2)?.unwrap() -11.0).abs() < 1e-10);
        assert!((m.get(2,0)?.unwrap()).abs() < 1e-10);
        assert!((m.get(2,2)?.unwrap() - 11.0).abs() < 1e-10);
        assert!(m.get(2,1)?.is_none());

        Ok(())
    }
}