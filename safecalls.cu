// ~ Error checks in CUDA code can help catch CUDA errors at their source. There are 2 sources of errors in CUDA source code:
// ~ 
	// ~ Errors from CUDA API calls. For example, a call to cudaMalloc() might fail.
	// ~ Errors from CUDA kernel calls. For example, there might be invalid memory access inside a kernel.
// ~ 
// ~ To use this functions, just include this file und use it like this:
// ~ CudaFunctions:
// ~ CudaSafeCall( cudaMalloc( &fooPtr, fooSize ) );
 // ~ 
// ~ Kernel call
// ~ fooKernel<<< x, y >>>(); 
// ~ CudaCheckError();
//
// taken from https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/


#include <stdio.h>
#include <stdlib.h>

// Define this to turn on error checking
#define CUDA_ERROR_CHECK
 
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
 
inline void __cudaSafeCall( cudaError err, const char *file, const int line ){
	#ifdef CUDA_ERROR_CHECK
	if ( cudaSuccess != err ){
		fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
		file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
	#endif
	 
	return;
}
 
inline void __cudaCheckError( const char *file, const int line ){
	#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err ){
		fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
		file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
	 
	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if( cudaSuccess != err ){
		fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
		file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
	#endif
	 
	return;
}
