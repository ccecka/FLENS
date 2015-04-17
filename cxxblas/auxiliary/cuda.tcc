#ifndef CXXBLAS_AUXILIARY_CUDA_TCC
#define CXXBLAS_AUXILIARY_CUDA_TCC 1

#if defined(HAVE_CUBLAS) || defined(HAVE_CUSOLVER)

// XXX
#include <utility>
#include <sstream>
#include <iomanip>

namespace cxxblas {

void
CudaEnv::init()
{
    if(NCalls==0) {
        // Create stream with index 0
        streamID   = 0;
        streams.insert(std::make_pair(streamID, cudaStream_t()));
        checkStatus(cudaStreamCreate(&streams.at(0)));
    }
    NCalls++;
}


void
CudaEnv::release()
{
    ASSERT(NCalls>0);
    if (NCalls==1) {

        // destroy events
        for (std::map<int, cudaEvent_t >::iterator it=events.begin(); it!=events.end(); ++it) {
            checkStatus(cudaEventDestroy(it->second));
        }
        events.clear();

        // destroy stream
        for (std::map<int, cudaStream_t>::iterator it=streams.begin(); it!=streams.end(); ++it) {
            checkStatus(cudaStreamDestroy(it->second));
        }
        streams.clear();

    }
    NCalls--;
}


void
CudaEnv::destroyStream(int _streamID)
{
    if(NCalls==0) {
        std::cerr << "Error: Cuda not initialized!" << std::endl;
        ASSERT(0);
    }

    ASSERT(_streamID!=streamID);
    checkStatus(cudaStreamDestroy(streams.at(_streamID)));
    streams.erase(_streamID);
}

cudaStream_t &
CudaEnv::getStream()
{
    if(NCalls==0) {
        std::cerr << "Error: Cuda not initialized!" << std::endl;
        ASSERT(0);
    }

    return streams.at(streamID);
}

int
CudaEnv::getStreamID()
{
    if(NCalls==0) {
        std::cerr << "Error: Cuda not initialized!" << std::endl;
        ASSERT(0);
    }

    return streamID;
}


void
CudaEnv::setStream(int _streamID)
{
    if(NCalls==0) {
        std::cerr << "Error: Cuda not initialized!" << std::endl;
        ASSERT(0);
    }

    streamID = _streamID;
    // Create new stream, object not found
    if (streams.find(streamID) == streams.end()) {
        streams.insert(std::make_pair(streamID, cudaStream_t()));
        checkStatus(cudaStreamCreate(&streams.at(streamID)));
    }
}

void
CudaEnv::syncStream(int _streamID)
{
    if(NCalls==0) {
        std::cerr << "Error: Cuda not initialized!" << std::endl;
        ASSERT(0);
    }

    checkStatus(cudaStreamSynchronize(streams.at(_streamID)));
}

void
CudaEnv::enableSyncCopy()
{
    syncCopyEnabled = true;
}

void
CudaEnv::disableSyncCopy()
{
    syncCopyEnabled = false;
}

bool
CudaEnv::isSyncCopyEnabled()
{
    return syncCopyEnabled;
}


void
CudaEnv::eventRecord(int _eventID)
{
    ///
    /// Creates event on current stream
    ///

    // Insert new event
    if (events.find(_eventID) == events.end()) {
        events.insert(std::make_pair(_eventID, cudaEvent_t ()));
        checkStatus(cudaEventCreate(&events.at(_eventID)));
    }

    // Create Event
    checkStatus(cudaEventRecord(events.at(_eventID), streams.at(streamID)));
}

void
CudaEnv::eventSynchronize(int _eventID)
{
    ///
    /// cudaEventSynchronize: Host waits until -eventID is completeted
    ///
    ///
    checkStatus(cudaEventSynchronize(events.at(_eventID)));
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
setStream(int streamID)
{
    CudaEnv::setStream(streamID);
}

int
getStreamID()
{
    return CudaEnv::getStreamID();
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
syncStream()
{
    syncStream(getStreamID());
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
        std::cerr << "CUBLAS: Failed to launch function of the GPU" << std::endl;
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
    CudaEnv::init();

    // create BLAS handle
    checkStatus(cublasCreate(&handle_));
}

void
CublasEnv::release()
{
    // destroy BLAS handle
    checkStatus(cublasDestroy(handle_));

    CudaEnv::release();
}

cublasHandle_t &
CublasEnv::handle()
{
    // TODO: Safety checks? Error msgs?

    return handle_;
}

#endif // HAVE_CUBLAS

} // end namespace flens

#endif // WITH_CUDA

#endif
