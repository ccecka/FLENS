#ifndef CXXSTD_MEMORY_H
#define CXXSTD_MEMORY_H 1

#include <memory>

#if defined(__CUDACC__)   // Compiling with nvcc -- thrust available

#  include <thrust/device_ptr.h>
#  include <thrust/device_reference.h>

#endif

#endif // CXXSTD_MEMORY_H
