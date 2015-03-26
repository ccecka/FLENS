// $ nvcc -std=c++11 -ccbin=g++-4.7 -I. -DWITH_CUBLAS -DCXXBLAS_DEBUG -o test_cublas test_cublas.cu -lcublas

#include <iostream>

#include <thrust/device_malloc_allocator.h>

#include "flens/flens.cxx"

using namespace flens;
using namespace std;

template <typename T, typename I = IndexOptions<> >
using ThrustArray = Array<T,I,thrust::device_malloc_allocator<T> >;

template <typename T, typename I = IndexOptions<> >
using ThrustFull  = FullStorage<T,ColMajor,I,thrust::device_malloc_allocator<T> >;

int main() {
  typedef DenseVector<ThrustArray<double> >   Vector;
  typedef GeMatrix<ThrustFull<double> >       Matrix;

  typedef typename Vector::IndexType        IndexType;

  flens::CudaEnv::init(); // XXX: revisit

  Vector x(8);
  x = 1, 2, 3, 4, 5, 6, 7, 8;

  cout << "x.range() = " << x.range() << endl;
  cout << "x.length() = " << x.length() << endl;

  cout << "x = " << x << endl;

  for (IndexType i=x.firstIndex(); i<=x.lastIndex(); ++i) {
    x(i) = i*i;
  }

  cout << "x = " << x << endl;


  const Underscore<IndexType> _;

  Vector::View y = x(_(2,4));
  y = 666;

  Vector::NoView z = x(_(1,2));
  z = 42;

  cout << "x = " << x << endl;
  cout << "y = " << y << endl;
  cout << "z = " << z << endl;

  Vector z2 = 2*x;

  Matrix A(8,8);

  A = 0;
  A.diag(1) = -1;

  Vector a = A*x;

  cout << "a = " << a << endl;

  return 0;
}
