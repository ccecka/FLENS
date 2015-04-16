// $ nvcc -std=c++11 -ccbin=g++-4.7 -I. -o test_cusolver test_cusolver.cu -lcublas -lcusolver

#include <iostream>
#include <complex>

#include <thrust/device_malloc_allocator.h>

#define WITH_CUBLAS
#define WITH_CUSOLVER
#define HAVE_CUBLAS
#define HAVE_CUSOLVER
#define CXXBLAS_DEBUG
#define CXXLAPACK_DEBUG

#define ALWAYS_USE_CXXLAPACK

// XXX: Figure out where to put this -- needed by both blas and lapack...
#include "cxxblas/cxxblas.h"
#include "cxxlapack/cxxlapack.h"
#include "flens/auxiliary/cuda.h"
#include "flens/auxiliary/cuda.tcc"

#include "flens/flens.cxx"

using namespace flens;
using namespace std;

template <typename T>
using CPUArray = Array<T>;

template <typename T>
using CPUFull = FullStorage<T,ColMajor>;


template <typename T, typename I = IndexOptions<> >
using GPUArray = Array<T,I,thrust::device_malloc_allocator<T> >;

template <typename T, typename I = IndexOptions<> >
using GPUFull  = FullStorage<T,ColMajor,I,thrust::device_malloc_allocator<T> >;


int main() {
  //using T = std::complex<double>;
  using T = double;

  typedef DenseVector<GPUArray<T> >   Vector;
  typedef GeMatrix<GPUFull<T> >       Matrix;

  typedef typename Vector::IndexType        IndexType;

  flens::CudaEnv::init(); // XXX: revisit

  std::cout << CudaEnv::getInfo() << std::endl;

  Matrix A(5,5);

  A = 1;
  A.diag(0) = 2;

  cout << "A = " << A << endl;

  DenseVector<GPUArray<IndexType> > ipiv;

  flens::lapack::trf(A, ipiv);

  cout << "A = " << A << endl;

  return 0;
}
