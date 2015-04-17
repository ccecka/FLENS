#ifndef CXXSTD_COMPLEX_H
#define CXXSTD_COMPLEX_H 1

#include <complex>

#if defined(WITH_CUBLAS) || defined(WITH_CUSOLVER)
#include <thrust/complex.h>
#endif

#endif // CXXSTD_COMPLEX_H
