/*******************************************************************
 * Author: <Name1>, <Name2>
 * Date: <Date>
 * File: mat_mul.c
 * Description: This file contains implementations of matrix multiplication
 *			    algorithms using various optimization techniques.
 *******************************************************************/

// PA 1: Matrix Multiplication

// includes
#include <stdio.h>
#include <stdlib.h>         // for malloc, free, atoi
#include <time.h>           // for time()
#include <chrono>	        // for timing
#include <xmmintrin.h> 		// for SSE
#include <immintrin.h>		// for AVX

#include "helper.h"			// for helper functions

// defines
// NOTE: you can change this value as per your requirement
#define TILE_SIZE	100		// size of the tile for blocking

/**
 * @brief 		Performs matrix multiplication of two matrices.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 */
void naive_mat_mul(double *A, double *B, double *C, int size) {

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				C[i * size + j] += A[i * size + k] * B[k * size + j];
			}
		}
	}
}

/**
 * @brief 		Task 1A: Performs matrix multiplication of two matrices using loop optimization.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 */
void loop_opt_mat_mul(double *A, double *B, double *C, int size){
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
	for (int i = 0; i < size; ++i) {
		int j = 0;
		// JU = 2: process two columns (j and j+1)
		for (; j + 1 < size; j += 2) {
			double r0 = A[i * size + j];
			double r1 = A[i * size + j + 1];
			for (int k = 0; k < size; k += 32) {
				C[j * size + k + 0]      += r0 * B[i * size + k + 0];
				C[(j + 1) * size + k + 0] += r1 * B[i * size + k + 0];
				C[j * size + k + 1]      += r0 * B[i * size + k + 1];
				C[(j + 1) * size + k + 1] += r1 * B[i * size + k + 1];
				C[j * size + k + 2]      += r0 * B[i * size + k + 2];
				C[(j + 1) * size + k + 2] += r1 * B[i * size + k + 2];
				C[j * size + k + 3]      += r0 * B[i * size + k + 3];
				C[(j + 1) * size + k + 3] += r1 * B[i * size + k + 3];
				C[j * size + k + 4]      += r0 * B[i * size + k + 4];
				C[(j + 1) * size + k + 4] += r1 * B[i * size + k + 4];
				C[j * size + k + 5]      += r0 * B[i * size + k + 5];
				C[(j + 1) * size + k + 5] += r1 * B[i * size + k + 5];
				C[j * size + k + 6]      += r0 * B[i * size + k + 6];
				C[(j + 1) * size + k + 6] += r1 * B[i * size + k + 6];
				C[j * size + k + 7]      += r0 * B[i * size + k + 7];
				C[(j + 1) * size + k + 7] += r1 * B[i * size + k + 7];
				C[j * size + k + 8]      += r0 * B[i * size + k + 8];
				C[(j + 1) * size + k + 8] += r1 * B[i * size + k + 8];
				C[j * size + k + 9]      += r0 * B[i * size + k + 9];
				C[(j + 1) * size + k + 9] += r1 * B[i * size + k + 9];
				C[j * size + k + 10]     += r0 * B[i * size + k + 10];
				C[(j + 1) * size + k + 10] += r1 * B[i * size + k + 10];
				C[j * size + k + 11]     += r0 * B[i * size + k + 11];
				C[(j + 1) * size + k + 11] += r1 * B[i * size + k + 11];
				C[j * size + k + 12]     += r0 * B[i * size + k + 12];
				C[(j + 1) * size + k + 12] += r1 * B[i * size + k + 12];
				C[j * size + k + 13]     += r0 * B[i * size + k + 13];
				C[(j + 1) * size + k + 13] += r1 * B[i * size + k + 13];
				C[j * size + k + 14]     += r0 * B[i * size + k + 14];
				C[(j + 1) * size + k + 14] += r1 * B[i * size + k + 14];
				C[j * size + k + 15]     += r0 * B[i * size + k + 15];
				C[(j + 1) * size + k + 15] += r1 * B[i * size + k + 15];
				C[j * size + k + 16]     += r0 * B[i * size + k + 16];
				C[(j + 1) * size + k + 16] += r1 * B[i * size + k + 16];
				C[j * size + k + 17]     += r0 * B[i * size + k + 17];
				C[(j + 1) * size + k + 17] += r1 * B[i * size + k + 17];
				C[j * size + k + 18]     += r0 * B[i * size + k + 18];
				C[(j + 1) * size + k + 18] += r1 * B[i * size + k + 18];
				C[j * size + k + 19]     += r0 * B[i * size + k + 19];
				C[(j + 1) * size + k + 19] += r1 * B[i * size + k + 19];
				C[j * size + k + 20]     += r0 * B[i * size + k + 20];
				C[(j + 1) * size + k + 20] += r1 * B[i * size + k + 20];
				C[j * size + k + 21]     += r0 * B[i * size + k + 21];
				C[(j + 1) * size + k + 21] += r1 * B[i * size + k + 21];
				C[j * size + k + 22]     += r0 * B[i * size + k + 22];
				C[(j + 1) * size + k + 22] += r1 * B[i * size + k + 22];
				C[j * size + k + 23]     += r0 * B[i * size + k + 23];
				C[(j + 1) * size + k + 23] += r1 * B[i * size + k + 23];
				C[j * size + k + 24]     += r0 * B[i * size + k + 24];
				C[(j + 1) * size + k + 24] += r1 * B[i * size + k + 24];
				C[j * size + k + 25]     += r0 * B[i * size + k + 25];
				C[(j + 1) * size + k + 25] += r1 * B[i * size + k + 25];
				C[j * size + k + 26]     += r0 * B[i * size + k + 26];
				C[(j + 1) * size + k + 26] += r1 * B[i * size + k + 26];
				C[j * size + k + 27]     += r0 * B[i * size + k + 27];
				C[(j + 1) * size + k + 27] += r1 * B[i * size + k + 27];
				C[j * size + k + 28]     += r0 * B[i * size + k + 28];
				C[(j + 1) * size + k + 28] += r1 * B[i * size + k + 28];
				C[j * size + k + 29]     += r0 * B[i * size + k + 29];
				C[(j + 1) * size + k + 29] += r1 * B[i * size + k + 29];
				C[j * size + k + 30]     += r0 * B[i * size + k + 30];
				C[(j + 1) * size + k + 30] += r1 * B[i * size + k + 30];
				C[j * size + k + 31]     += r0 * B[i * size + k + 31];
				C[(j + 1) * size + k + 31] += r1 * B[i * size + k + 31];
			}
		}
		// Tail for odd j
		for (; j < size; ++j) {
			double r = A[i * size + j];
			for (int k = 0; k < size; k += 32) {
				C[j * size + k + 0]  += r * B[i * size + k + 0];
				C[j * size + k + 1]  += r * B[i * size + k + 1];
				C[j * size + k + 2]  += r * B[i * size + k + 2];
				C[j * size + k + 3]  += r * B[i * size + k + 3];
				C[j * size + k + 4]  += r * B[i * size + k + 4];
				C[j * size + k + 5]  += r * B[i * size + k + 5];
				C[j * size + k + 6]  += r * B[i * size + k + 6];
				C[j * size + k + 7]  += r * B[i * size + k + 7];
				C[j * size + k + 8]  += r * B[i * size + k + 8];
				C[j * size + k + 9]  += r * B[i * size + k + 9];
				C[j * size + k + 10] += r * B[i * size + k + 10];
				C[j * size + k + 11] += r * B[i * size + k + 11];
				C[j * size + k + 12] += r * B[i * size + k + 12];
				C[j * size + k + 13] += r * B[i * size + k + 13];
				C[j * size + k + 14] += r * B[i * size + k + 14];
				C[j * size + k + 15] += r * B[i * size + k + 15];
				C[j * size + k + 16] += r * B[i * size + k + 16];
				C[j * size + k + 17] += r * B[i * size + k + 17];
				C[j * size + k + 18] += r * B[i * size + k + 18];
				C[j * size + k + 19] += r * B[i * size + k + 19];
				C[j * size + k + 20] += r * B[i * size + k + 20];
				C[j * size + k + 21] += r * B[i * size + k + 21];
				C[j * size + k + 22] += r * B[i * size + k + 22];
				C[j * size + k + 23] += r * B[i * size + k + 23];
				C[j * size + k + 24] += r * B[i * size + k + 24];
				C[j * size + k + 25] += r * B[i * size + k + 25];
				C[j * size + k + 26] += r * B[i * size + k + 26];
				C[j * size + k + 27] += r * B[i * size + k + 27];
				C[j * size + k + 28] += r * B[i * size + k + 28];
				C[j * size + k + 29] += r * B[i * size + k + 29];
				C[j * size + k + 30] += r * B[i * size + k + 30];
				C[j * size + k + 31] += r * B[i * size + k + 31];
			}
		}
	}
//-------------------------------------------------------------------------------------------------------------------------------------------

}


/**
 * @brief 		Task 1B: Performs matrix multiplication of two matrices using tiling.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @param 		tile_size 	size of the tile
 * @note 		The tile size should be a multiple of the dimension of the matrices.
 * 				For example, if the dimension is 1024, then the tile size can be 32, 64, 128, etc.
 * 				You can assume that the matrices are square matrices.
*/
void tile_mat_mul(double *A, double *B, double *C, int size, int tile_size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    

//-------------------------------------------------------------------------------------------------------------------------------------------
    
}

/**
 * @brief 		Task 1C: Performs matrix multiplication of two matrices using SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/
void simd_mat_mul(double *A, double *B, double *C, int size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    

//-------------------------------------------------------------------------------------------------------------------------------------------
    
}

/**
 * @brief 		Task 1D: Performs matrix multiplication of two matrices using combination of tiling/SIMD/loop optimization.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @param 		tile_size 	size of the tile
 * @note 		The tile size should be a multiple of the dimension of the matrices.
 * @note 		You can assume that the matrices are square matrices.
*/
void combination_mat_mul(double *A, double *B, double *C, int size, int tile_size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    
    
//-------------------------------------------------------------------------------------------------------------------------------------------
    
}

// NOTE: DO NOT CHANGE ANYTHING BELOW THIS LINE
/**
 * @brief 		Main function
 * @param 		argc 		number of command line arguments
 * @param 		argv 		array of command line arguments
 * @return 		0 on success
 * @note 		DO NOT CHANGE THIS FUNCTION
 * 				DO NOT ADD OR REMOVE ANY COMMAND LINE ARGUMENTS
*/
int main(int argc, char **argv) {

	if ( argc <= 1 ) {
		printf("Usage: %s <matrix_dimension>\n", argv[0]);
		return 0;
	}

	else {
		int size = atoi(argv[1]);

		double *A = (double *)malloc(size * size * sizeof(double));
		double *B = (double *)malloc(size * size * sizeof(double));
		double *C = (double *)calloc(size * size, sizeof(double));

		// initialize random seed
		srand(time(NULL));

		// initialize matrices A and B with random values
		initialize_matrix(A, size, size);
		initialize_matrix(B, size, size);

		// perform normal matrix multiplication
		auto start = std::chrono::high_resolution_clock::now();
		naive_mat_mul(A, B, C, size);
		auto end = std::chrono::high_resolution_clock::now();
		auto time_naive_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Normal matrix multiplication took %ld ms to execute \n\n", time_naive_mat_mul);

	#ifdef OPTIMIZE_LOOP_OPT
		// Task 1a: perform matrix multiplication with loop optimization

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		loop_opt_mat_mul(A, B, C, size);
		end = std::chrono::high_resolution_clock::now();
		auto time_loop_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Loop optimized matrix multiplication took %ld ms to execute \n", time_loop_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_loop_mat_mul);
	#endif

	#ifdef OPTIMIZE_TILING
		// Task 1b: perform matrix multiplication with tiling

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		tile_mat_mul(A, B, C, size, TILE_SIZE);
		end = std::chrono::high_resolution_clock::now();
		auto time_tiling_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Tiling matrix multiplication took %ld ms to execute \n", time_tiling_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_tiling_mat_mul);
	#endif

	#ifdef OPTIMIZE_SIMD
		// Task 1c: perform matrix multiplication with SIMD instructions 

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		simd_mat_mul(A, B, C, size);
		end = std::chrono::high_resolution_clock::now();
		auto time_simd_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		printf("SIMD matrix multiplication took %ld ms to execute \n", time_simd_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_simd_mat_mul);
	#endif

	#ifdef OPTIMIZE_COMBINED
		// Task 1d: perform matrix multiplication with combination of tiling, SIMD and loop optimization

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		combination_mat_mul(A, B, C, size, TILE_SIZE);
		end = std::chrono::high_resolution_clock::now();
		auto time_combination = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Combined optimization matrix multiplication took %ld ms to execute \n", time_combination);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_combination);
	#endif

		// free allocated memory
		free(A);
		free(B);
		free(C);

		return 0;
	}
}
