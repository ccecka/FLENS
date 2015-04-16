#ifndef CXXSTD_ALGORITHM_H
#define CXXSTD_ALGORITHM_H 1

#include <algorithm>

#if defined(__CUDACC__)   // Compiling with nvcc -- thrust available

#  include <thrust/fill.h>
#  include <thrust/uninitialized_fill.h>
#  include <thrust/device_ptr.h>

namespace flens {
namespace alg {
  using thrust::fill;
  using thrust::fill_n;
  using thrust::uninitialized_fill;
  using thrust::uninitialized_fill_n;
}
}

#else

namespace flens {
namespace alg {
  using std::fill;
  using std::fill_n;
  using std::uninitialized_fill;
  using std::uninitialized_fill_n;
}
}

#endif


#endif // CXXSTD_ALGORITHM_H
