#ifndef CXXBLAS_AUXILIARY_CUDA_H
#define CXXBLAS_AUXILIARY_CUDA_H 1

#ifdef WITH_CUBLAS

#include <thrust/device_ptr.h>
#include <string>
#include <map>

namespace flens {

enum StorageType {
    CUDA
};

namespace detail {

template <typename T, StorageType Storage>
struct device_ptr {};

template <typename T>
struct device_ptr<T, StorageType::CUDA> {
    using type = thrust::device_ptr<T>;
};

} // end namespace detail

template <typename T, StorageType Storage>
using device_ptr = typename detail::device_ptr<T,Storage>::type;



class CudaEnv {
 public:
    static void
    init();

    static void
    release();

    static void
    destroyStream(int _streamID);

    static cublasHandle_t &
    getHandle();

    static cudaStream_t &
    getStream();

    static int
    getStreamID();

    static void
    setStream(int _streamID);

    static void
    syncStream(int _streamID);

    static void
    enableSyncCopy();

    static void
    disableSyncCopy();

    static bool
    isSyncCopyEnabled();

    static void
    eventRecord(int _eventID);

    static void
    eventSynchronize(int _eventID);

    static std::string
    getInfo();

private:
    static int                          NCalls;
    static cublasHandle_t               handle;
    static std::map<int, cudaStream_t>  streams;
    static int                          streamID;
    static bool                         syncCopyEnabled;
    static std::map<int, cudaEvent_t >  events;
};


// XXX XXX
int                         CudaEnv::NCalls          = 0;
cublasHandle_t              CudaEnv::handle          = 0;
std::map<int, cudaStream_t> CudaEnv::streams         = std::map<int, cudaStream_t>();
int                         CudaEnv::streamID        = 0;
bool                        CudaEnv::syncCopyEnabled = true;
std::map<int, cudaEvent_t>  CudaEnv::events          = std::map<int, cudaEvent_t>();




void
checkStatus(cublasStatus_t status);

void
checkStatus(cudaError_t error);

void
setStream(int streamID);

int
getStreamID();

void
destroyStream(int streamID);

template <typename... Args>
void
destroyStream(int streamID, Args... args);

void
syncStream();

void
syncStream(int streamID);

template <typename... Args>
void
syncStream(int streamID, Args... args);

} // end namespace flens

#endif // WITH_CUBLAS

#endif
