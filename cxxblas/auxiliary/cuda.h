#ifndef CXXBLAS_AUXILIARY_CUDA_H
#define CXXBLAS_AUXILIARY_CUDA_H 1

#if defined(HAVE_CUBLAS) || defined(HAVE_CUSOLVER)

#include <string> // XXX
#include <vector>

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

// Implement a strided range/iterator
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>


namespace cxxblas {

class CudaEnv {
 public:
    static void
    release();

    static cudaStream_t &
    getStream(int streamID);

    static void
    destroyStream(int streamID);

    static void
    syncStream(int streamID);

    static void
    eventRecord(int _eventID, int streamID);

    static void
    eventSynchronize(int _eventID);

    static std::string
    getInfo();

  //private:
    static std::vector<cudaStream_t>    streams_;
    static std::vector<cudaEvent_t>     events_;
};

// XXX?
std::vector<cudaStream_t>   CudaEnv::streams_    = {};
std::vector<cudaEvent_t>    CudaEnv::events_     = {};


void
checkStatus(cudaError_t error);

void
destroyStream(int streamID);

template <typename... Args>
void
destroyStream(int streamID, Args... args);

void
syncStream(int streamID);

template <typename... Args>
void
syncStream(int streamID, Args... args);


#ifdef HAVE_CUBLAS

class CublasEnv {
 public:

    static void
    init();

    static void
    release();

    static cublasHandle_t &
    handle();

    static cudaStream_t
    stream();

    static int
    streamID();

    static void
    setStream(int streamID);

    static void
    enableSyncCopy();

    static void
    disableSyncCopy();

    static bool
    isSyncCopyEnabled();

    static void
    syncCopy();

  //private:
    static cublasHandle_t               handle_;
#pragma omp threadprivate(handle_)
    static int                          streamID_;
#pragma omp threadprivate(streamID_)
    static bool                         syncCopyEnabled_;
#pragma omp threadprivate(syncCopyEnabled_)
};

// XXX?
cublasHandle_t              CublasEnv::handle_          =  0;
int                         CublasEnv::streamID_        = -1;
bool                        CublasEnv::syncCopyEnabled_ = true;


void
checkStatus(cublasStatus_t status);

#endif // HAVE_CUBLAS




/*! \brief RandomAccessIterator for strided access to array entries.
 *
 * \tparam RandomAccessIterator The iterator type used to encapsulate the underlying data.
 *
 * \par Overview
 * \p strided_iterator is an iterator which represents a pointer into
 *  a strided range entries in a underlying array. This iterator is useful
 *  for creating a strided sublist of entries from a larger iterator.
 */
template <typename RandomAccessIterator>
class StridedRange
{
public:

    /*! \cond */
    typedef typename thrust::iterator_value<RandomAccessIterator>::type                       value_type;
    typedef typename thrust::iterator_system<RandomAccessIterator>::type                      memory_space;
    typedef typename thrust::iterator_pointer<RandomAccessIterator>::type                     pointer;
    typedef typename thrust::iterator_reference<RandomAccessIterator>::type                   reference;
    typedef typename thrust::iterator_difference<RandomAccessIterator>::type                  difference_type;
    typedef typename thrust::iterator_difference<RandomAccessIterator>::type                  size_type;

    struct Strider : thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        Strider(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        {
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                               CountingIterator;
    typedef typename thrust::transform_iterator<Strider, CountingIterator>              TransformIterator;
    typedef typename thrust::permutation_iterator<RandomAccessIterator,TransformIterator>     PermutationIterator;

    // type of the StridedRange iterator
    typedef PermutationIterator iterator;
    /*! \endcond */

    /*! \brief Null constructor initializes this \p strided_iterator's stride to zero.
     */
    StridedRange(void)
        : stride(0) {}

    /*! \brief This constructor builds a \p StridedRange from a range.
     *  \param begin The beginning of the range.
     *  \param end The end of the range.
     *  \param stride The stride between consecutive entries in the iterator.
     */
    StridedRange(RandomAccessIterator first, RandomAccessIterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}

    /*! \brief This method returns an iterator pointing to the beginning of
     *  this strided sequence of entries.
     *  \return mStart
     */
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), Strider(stride)));
    }

    /*! \brief This method returns an iterator pointing to one element past
     *  the last of this strided sequence of entries.
     *  \return mEnd
     */
    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }

    /*! \brief Subscript access to the data contained in this iterator.
     *  \param n The index of the element for which data should be accessed.
     *  \return Read/write reference to data.
     *
     *  This operator allows for easy, array-style, data access.
     *  Note that data access with this operator is unchecked and
     *  out_of_range lookups are not defined.
     */
    reference operator[](size_type n) const
    {
        return *(begin() + n);
    }

protected:
    RandomAccessIterator first;
    RandomAccessIterator last;
    difference_type stride;
}; // end StridedRange



} // end namespace flens

#endif // HAVE_CUBLAS || HAVE_CUSOLVER

#endif
