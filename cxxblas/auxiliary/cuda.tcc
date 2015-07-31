#ifndef CXXBLAS_AUXILIARY_CUDA_TCC
#define CXXBLAS_AUXILIARY_CUDA_TCC 1

#if defined(HAVE_CUBLAS) || defined(HAVE_CUSOLVER)

// XXX
#include <utility>
#include <sstream>
#include <iomanip>

namespace cxxblas {

void
CudaEnv::release()
{
  // destroy streams
  for (auto& s : streams)
    checkStatus(cudaStreamDestroy(s));
  streams.clear();

  // destroy events
  for (auto& e : events)
    checkStatus(cudaEventDestroy(e));
  events.clear();
}

void
CudaEnv::destroyStream(int _streamID)
{
    if (_streamID < int(streams.size())) {
        checkStatus(cudaStreamDestroy(streams[_streamID]));
        streams[_streamID] = cudaStream_t(); // XXX: Needed?
    }
}

cudaStream_t &
CudaEnv::getStream(int _streamID)
{
    // Expand the stream map
    while (_streamID >= int(streams.size()))
        streams.push_back(cudaStream_t());
    // Create new stream if not inited
    if (streams[_streamID] == cudaStream_t())
        checkStatus(cudaStreamCreate(&streams[_streamID]));

    return streams[_streamID];
}

void
CudaEnv::syncStream(int _streamID)
{
    checkStatus(cudaStreamSynchronize(streams[_streamID]));
}

void
CudaEnv::eventRecord(int _eventID, int _streamID)
{
    // Expand the event map
    while (_eventID >= int(events.size()))
        events.push_back(cudaEvent_t());
    // Create new event if not inited
    if (events.at(_eventID) == cudaEvent_t())
        checkStatus(cudaEventCreate(&events.at(_eventID)));

    // Create Event
    checkStatus(cudaEventRecord(events[_eventID], getStream(_streamID)));
}

void
CudaEnv::eventSynchronize(int _eventID)
{
    checkStatus(cudaEventSynchronize(events[_eventID]));
}

std::string
CudaEnv::getInfo()
{
    std::stringstream sstream;
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);

    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);

        sstream << "CUDA Device " << devProp.name << std::endl;
        sstream << "==============================================" << std::endl;
        sstream << "Major revision number:         " << std::setw(15) << devProp.major << std::endl;
        sstream << "Minor revision number:         " << std::setw(15) << devProp.minor << std::endl;
        sstream << "Number of multiprocessors:     " << std::setw(15) << devProp.multiProcessorCount << std::endl;
        sstream << "Clock rate:                    " << std::setw(11) << devProp.clockRate / (1024.f*1024.f) << " GHz" << std::endl;
        sstream << "Total global memory:           " << std::setw(9) << devProp.totalGlobalMem / (1024*1024) << " MByte" << std::endl;
        sstream << "Total constant memory:         " << std::setw(9) << devProp.totalConstMem  / (1024) << " KByte"<< std::endl;
        sstream << "Maximum memory:                " << std::setw(9) << devProp.memPitch  / (1024*1024) << " MByte"<< std::endl;
        sstream << "Total shared memory per block: " << std::setw(9) << devProp.sharedMemPerBlock / (1024) << " KByte" << std::endl;
        sstream << "Total registers per block:     " << std::setw(15) << devProp.regsPerBlock << std::endl;
        sstream << "Warp size:                     " << std::setw(15) << devProp.warpSize << std::endl;
        sstream << "Maximum threads per block:     " << std::setw(15) << devProp.maxThreadsPerBlock << std::endl;
        for (int j = 0; j < 3; ++j) {
            sstream << "Maximum dimension of block " << j << ":  " << std::setw(15) << devProp.maxThreadsDim[j] << std::endl;
        }
        for (int j = 0; j < 3; ++j) {
            sstream << "Maximum dimension of grid " << j << ":   "  << std::setw(15) << devProp.maxGridSize[j] << std::endl;
        }
        sstream << "Texture alignment:             " << std::setw(15) << devProp.textureAlignment << std::endl;
        sstream << "Concurrent copy and execution: " << std::setw(15) << std::boolalpha << (devProp.deviceOverlap>0) << std::endl;
        sstream << "Concurrent kernel execution:   " << std::setw(15) << std::boolalpha << (devProp.concurrentKernels>0) << std::endl;
        sstream << "Kernel execution timeout:      " << std::setw(15) << std::boolalpha << (devProp.kernelExecTimeoutEnabled>0) << std::endl;
        sstream << "ECC support enabled:           " << std::setw(15) << std::boolalpha << (devProp.ECCEnabled>0) << std::endl;
    }
    return sstream.str();
}

void
checkStatus(cudaError_t status)
{
    if(status==cudaSuccess)
        return;
    std::cerr << cudaGetErrorString(status) << std::endl;
}

void
destroyStream(int streamID)
{
    CudaEnv::destroyStream(streamID);
}

template <typename... Args>
void
destroyStream(int streamID, Args... args)
{
    destroyStream(streamID);
    destroyStream(args...);
}

void
syncStream(int streamID)
{
    CudaEnv::syncStream(streamID);
}

template <typename... Args>
void
syncStream(int streamID, Args... args)
{
    syncStream(streamID);
    syncStream(args...);
}


#ifdef HAVE_CUBLAS

void
checkStatus(cublasStatus_t status)
{
    if (status==CUBLAS_STATUS_SUCCESS) {
        return;
    }

    if (status==CUBLAS_STATUS_NOT_INITIALIZED) {
        std::cerr << "CUBLAS: Library was not initialized!" << std::endl;
    } else if  (status==CUBLAS_STATUS_INVALID_VALUE) {
        std::cerr << "CUBLAS: Parameter had illegal value!" << std::endl;
    } else if  (status==CUBLAS_STATUS_MAPPING_ERROR) {
        std::cerr << "CUBLAS: Error accessing GPU memory!" << std::endl;
    } else if  (status==CUBLAS_STATUS_ALLOC_FAILED) {
        std::cerr << "CUBLAS: allocation failed!" << std::endl;
    } else if  (status==CUBLAS_STATUS_ARCH_MISMATCH) {
        std::cerr << "CUBLAS: Device does not support double precision!" << std::endl;
    } else if  (status==CUBLAS_STATUS_EXECUTION_FAILED) {
        std::cerr << "CUBLAS: Failed to launch function on the GPU" << std::endl;
    } else if  (status==CUBLAS_STATUS_INTERNAL_ERROR) {
        std::cerr << "CUBLAS: An internal operation failed" << std::endl;
    } else {
        std::cerr << "CUBLAS: Unkown error" << std::endl;
    }

    ASSERT(status==CUBLAS_STATUS_SUCCESS); // false
}

void
CublasEnv::init()
{
    checkStatus(cublasCreate(&handle_));
}

void
CublasEnv::release()
{
    checkStatus(cublasDestroy(handle_));
    CudaEnv::release();  // XXX race
}

cublasHandle_t &
CublasEnv::handle()
{
    // TODO: Safety checks? Error msgs?
    return handle_;
}

cudaStream_t
CublasEnv::stream()
{
    cudaStream_t s;
    cublasGetStream(handle_, &s);
    return s;
}

int
CublasEnv::streamID()
{
    return streamID_;
}

void
CublasEnv::setStream(int _streamID)
{
    streamID_ = _streamID;
    checkStatus(cublasSetStream(handle_, CudaEnv::getStream(streamID_)));
}

void
CublasEnv::enableSyncCopy()
{
    syncCopyEnabled = true;
}

void
CublasEnv::disableSyncCopy()
{
    syncCopyEnabled = false;
}

bool
CublasEnv::isSyncCopyEnabled()
{
    return syncCopyEnabled;
}

void
CublasEnv::syncCopy()
{
    if (syncCopyEnabled && streamID_ >= 0)
        CudaEnv::syncStream(streamID_);
}

#endif // HAVE_CUBLAS

} // end namespace flens

#endif // WITH_CUDA

#endif
