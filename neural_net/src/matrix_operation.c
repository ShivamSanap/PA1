#include "matrix_operation.h"
#include <immintrin.h>

Matrix MatrixOperation::NaiveMatMul(const Matrix &A, const Matrix &B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}
	
	
	Matrix C(n,m);
	
	for(int i = 0; i < n ; i++) {
		for (int j = 0 ; j< m ; j++) {
			for(int l = 0; l < k; l++) {
				C(i,j) += A(i,l) * B(l,j);
			}
		}
	}
	
	return C;
}

// Loop reordered matrix multiplication (ikj order for better cache locality)
Matrix MatrixOperation::ReorderedMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}
	
	
	Matrix C(n,m);
	
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    for (size_t i = 0; i < n; ++i) {
		for (size_t l = 0; l < k; ++l) {
			element_t a_il = A(i, l);
			for (size_t j = 0; j < m; ++j) {
				C(i, j) += a_il * B(l, j);
			}
		}
	}

//-------------------------------------------------------------------------------------------------------------------------------------------


	return C;
}

// Loop unrolled matrix multiplication
Matrix MatrixOperation::UnrolledMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);

    const int UNROLL = 4;
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    for (size_t i = 0; i < n; ++i) {
		for (size_t l = 0; l < k; ++l) {
			element_t a_il = A(i, l);
			size_t j = 0;
			for (; j + UNROLL - 1 < m; j += UNROLL) {
				C(i, j)     += a_il * B(l, j);
				C(i, j + 1) += a_il * B(l, j + 1);
				C(i, j + 2) += a_il * B(l, j + 2);
				C(i, j + 3) += a_il * B(l, j + 3);
			}
			// tail: handles m not being a multiple of UNROLL
			for (; j < m; ++j) {
				C(i, j) += a_il * B(l, j);
			}
		}
	}

//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// Tiled (blocked) matrix multiplication for cache efficiency
Matrix MatrixOperation::TiledMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);
    const int T = 128;   // tile size
	int i_max = 0;
	int k_max = 0;
	int j_max = 0;
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    for (size_t ii = 0; ii < n; ii += T) {
		i_max = std::min(ii + T, n);
		for (size_t kk = 0; kk < k; kk += T) {
			k_max = std::min(kk + T, k);
			for (size_t jj = 0; jj < m; jj += T) {
				j_max = std::min(jj + T, m);
 
				for (size_t i = ii; i < (size_t)i_max; ++i) {
					for (size_t l = kk; l < (size_t)k_max; ++l) {
						element_t a_il = A(i, l);
						for (size_t j = jj; j < (size_t)j_max; ++j) {
							C(i, j) += a_il * B(l, j);
						}
					}
				}
			}
		}
	}

//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// SIMD vectorized matrix multiplication (using AVX2)
Matrix MatrixOperation::VectorizedMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    
for (size_t i = 0; i < n; ++i) {
		for (size_t l = 0; l < k; ++l) {
			__m256d a_il = _mm256_set1_pd(A(i, l));
			size_t j = 0;
			for (; j + 3 < m; j += 4) {
				__m256d b_vec = _mm256_loadu_pd(&B(l, j));
				__m256d c_vec = _mm256_loadu_pd(&C(i, j));
				c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_il, b_vec));
				_mm256_storeu_pd(&C(i, j), c_vec);
			}
			// tail: handles m not being a multiple of 4
			double a_il_s = A(i, l);
			for (; j < m; ++j) {
				C(i, j) += a_il_s * B(l, j);
			}
		}
	}
//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// Optimized matrix transpose
Matrix MatrixOperation::Transpose(const Matrix& A) {
	size_t rows = A.getRows();
	size_t cols = A.getCols();
	Matrix result(cols, rows);

	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			result(j, i) = A(i, j);
		}
	}

	// Optimized transpose using blocking for better cache performance
	// This is a simple implementation, more advanced techniques can be applied
	// Write your code here and commnent the above code
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    const size_t T = 32;
	for (size_t ii = 0; ii < rows; ii += T) {
		size_t i_max = std::min(ii + T, rows);
		for (size_t jj = 0; jj < cols; jj += T) {
			size_t j_max = std::min(jj + T, cols);
			for (size_t i = ii; i < i_max; ++i) {
				for (size_t j = jj; j < j_max; ++j) {
					result(j, i) = A(i, j);
				}
			}
		}
	}

//-------------------------------------------------------------------------------------------------------------------------------------------

	
	return result;
}
