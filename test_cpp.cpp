// $ g++ -std=c++11 -I. -DCXXBLAS_DEBUG -DWITH_OPENBLAS -o test_cpp test_cpp.cpp -L/usr/local/opt/openblas/lib -lopenblas

#include <iostream>

#include "flens/flens.cxx"

using namespace flens;
using namespace std;

int main() {
  typedef DenseVector<Array<double,IndexOptions<int,1>> >   FDenseVector;
  typedef DenseVector<Array<double,IndexOptions<int,0>> >   CDenseVector;


  typedef FDenseVector::IndexType        IndexType;

  FDenseVector x(4);

  x = 1, 2, 3, 4;

  FDenseVector a = x;

  cout << "x.range() = " << x.range() << endl;
  cout << "x.length() = " << x.length() << endl;

  cout << "x = " << x << endl;

  for (IndexType i=x.firstIndex(); i<=x.lastIndex(); ++i) {
    x(i) = i*i;
  }

  const Underscore<IndexType> _;

  //DenseVector::View y = x(_(2,4));
  //y = 666;

  FDenseVector::NoView z = x(_(1,2));
  //z = 42;

  cout << "x = " << x << endl;
  //cout << "y = " << y << endl;
  cout << "z = " << z << endl;

  CDenseVector z2 = x(_(1,2));

  return 0;
}
