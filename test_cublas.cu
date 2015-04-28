// $ nvcc -std=c++11 -ccbin=g++-4.7 -I. -o test_cublas test_cublas.cu -lcublas

#include <iostream>
#include <complex>

#include <thrust/device_malloc_allocator.h>

#define WITH_CUBLAS
#define CXXBLAS_DEBUG
#define CXXLAPACK_DEBUG

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
  using T = std::complex<double>;
  //using T = double;

  typedef DenseVector<GPUArray<T> >   GPUVector;
  typedef GeMatrix<GPUFull<T> >       GPUMatrix;

  typedef DenseVector<CPUArray<T> >   CPUVector;
  typedef GeMatrix<CPUFull<T> >       CPUMatrix;

  typedef typename GPUVector::IndexType        IndexType;

  cxxblas::CublasEnv::init(); // XXX: revisit

  std::cout << cxxblas::CudaEnv::getInfo() << std::endl;

  GPUVector x(5);
  x = 1, 2, 3, 4, 5;

  cout << "x.range() = " << x.range() << endl;
  cout << "x.length() = " << x.length() << endl;

  cout << "x = " << x << endl;

  for (IndexType i=x.firstIndex(); i<=x.lastIndex(); ++i) {
    x(i) = i*i;
  }

  cout << "x = " << x << endl;

  CPUVector cpux = x;
  x = 1, 2, 3, 4, 5;

  cout << "cpux.range() = " << cpux.range() << endl;
  cout << "cpux.length() = " << cpux.length() << endl;

  cout << "cpux = " << cpux << endl;

  const Underscore<IndexType> _;

  GPUVector::View y = x(_(2,3));
  y = 666;

  GPUVector::NoView z = x(_(2,3));
  z = 42;

  cout << "x = " << x << endl;
  cout << "y = " << y << endl;
  cout << "z = " << z << endl;

  GPUVector z2 = 2.0*x(_(1,2,5));

  cout << "z2 = " << z2 << endl;

  GPUMatrix A(5,5);

  A = 0;
  A.diag(1) = -1;

  cout << "A = " << A << endl;

  CPUMatrix CPUA = A;

  cout << "CPUA = " << CPUA << endl;

  GPUVector a = A*x;

  cout << "a = " << a << endl;

  cxxblas::CublasEnv::release(); // XXX: revisit

  return 0;
}
