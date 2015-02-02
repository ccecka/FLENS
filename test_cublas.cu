// $ nvcc -std=c++11 -ccbin=g++-4.7 -I. -DWITH_CUBLAS -DCXXBLAS_DEBUG -o test_cublas test_cublas.cu -lcublas

#include <iostream>

#include <thrust/device_malloc_allocator.h>

#include "flens/flens.cxx"

using namespace flens;
using namespace std;

template <typename T, typename I = IndexOptions<> >
using ThrustArray = Array<T,I,thrust::device_malloc_allocator<T> >;


int main() {
  typedef DenseVector<ThrustArray<double> >   DenseVector;

  typedef DenseVector::IndexType        IndexType;

  DenseVector x(4);

  x = 1, 2, 3, 4;

  cout << "x.range() = " << x.range() << endl;
  cout << "x.length() = " << x.length() << endl;

  cout << "x = " << x << endl;

  for (IndexType i=x.firstIndex(); i<=x.lastIndex(); ++i) {
    x(i) = i*i;
  }

  const Underscore<IndexType> _;

  DenseVector::View y = x(_(2,4));
  y = 666;

  cout << "x = " << x << endl;
  cout << "y = " << y << endl;

  DenseVector::NoView z = x(_(1,2));
  z = 42;

  cout << "z = " << z << endl;

  return 0;
}
