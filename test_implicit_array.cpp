// $ g++ -std=c++11 -I. -o test_implicit_array test_implicit_array.cpp

#include <iostream>

#include "flens/flens.cxx"

#include "flens/storage/implicit/implicitarray.h"
#include "flens/storage/implicit/implicitarray.tcc"
#include "flens/storage/implicit/implicitarrayview.h"
#include "flens/storage/implicit/implicitarrayview.tcc"
#include "flens/storage/implicit/constimplicitarrayview.h"
#include "flens/storage/implicit/constimplicitarrayview.tcc"

using namespace flens;
using namespace std;

struct MyF {
  double operator()(int i) const { return i+1; }
};



int main() {

  //typedef const DenseVector<Array<double> > DenseVector;
  //DenseVector xx(4);
  //const DenseVector::ConstView x = xx;

  typedef DenseVector<ImplicitArray<MyF>> ImDenseVector;

  ImDenseVector x(8);

  cout << "x.range() = " << x.range() << endl;
  cout << "x.length() = " << x.length() << endl;

  cout << "x = " << x << endl;

  typedef ImDenseVector::IndexType        IndexType;
  const Underscore<IndexType> _;

  auto y = x(_(5,8));

  cout << "y.range() = " << y.range() << endl;
  cout << "y.length() = " << y.length() << endl;

  cout << "y = " << y << endl;

  auto z = y(_(3,4));

  cout << "z.range() = " << z.range() << endl;
  cout << "z.length() = " << z.length() << endl;

  cout << "z = " << z << endl;

/*
  DenseVector<Array<double>> z = x;

  cout << "z = " << z << endl;
*/

  return 0;
}
