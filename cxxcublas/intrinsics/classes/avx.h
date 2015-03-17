/*
 *   Copyright (c) 2012, Klaus Pototzky
 *
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *   1) Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2) Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in
 *      the documentation and/or other materials provided with the
 *      distribution.
 *   3) Neither the name of the FLENS development group nor the names of
 *      its contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PLAYGROUND_CXXBLAS_INTRINSICS_CLASSES_AVX_H
#define PLAYGROUND_CXXBLAS_INTRINSICS_CLASSES_AVX_H 1

#include <playground/cxxblas/intrinsics/includes.h>
#include <playground/cxxblas/intrinsics/classes/functions/functions.h>

#ifdef HAVE_AVX

template <>
class Intrinsics<float, IntrinsicsLevel::AVX>
{
public:
    typedef float                                     DataType;
    typedef float                                     PrimitiveDataType;
    typedef __m256                                    IntrinsicsDataType;
    typedef Intrinsics<float, IntrinsicsLevel::AVX>  IntrinsicsType;
    static  const int                                 numElements = 8;


    Intrinsics()                                      {}
    Intrinsics(const __m256 &val)                     { v = val; }
    Intrinsics(float *a)                              { this->load(a); }
    Intrinsics(const float a)                         { this->fill(a); }

    void operator=(const IntrinsicsDataType &a)       {v = a; }
    void operator=(const IntrinsicsType &a)           {v = a.get(); }

    IntrinsicsDataType get() const                    {return v;}
    __m128             get_high() const               {return _mm256_extractf128_ps(v,1);}

    void fill(float a)                                { v = _mm256_load1_ps(&a); }
    void setZero()                                    { v = _mm256_setzero_ps(); }
    void load(const float *a)                         { v = _mm256_loadu_ps(a);  }
    void loadu(const float *a)                        { v = _mm256_loadu_ps(a);  }
    void load_aligned(const float *a)                 { v = _mm256_load_ps(a);  }
    void load_partial(const float *a, const int length)  
    {
        ASSERT(length>=0 && length<=numElements);
        if (length==1) {    
            v = _mm256_insertf128_ps(v, _mm_load_ss(a), 0);   
        } else if (length==2) {     
            __m128d _tmp128d = _mm_load_sd(reinterpret_cast<const double *>(a));
            v =  _mm256_insertf128_ps(v, _mm_castpd_ps(_tmp128d), 0);
        } else if (length==3) {
            __m128 _tmp128 = _mm_setr_ps(a[0], a[1], a[2], float(0));
            v = _mm256_insertf128_ps(v,_tmp128, 0);
        } else if (length==4) { 
            v = _mm256_insertf128_ps(v, _mm_loadu_ps(a), 0);
        } else if (length==5) {    
            v = _mm256_insertf128_ps(v, _mm_loadu_ps(a), 0);
            v = _mm256_insertf128_ps(v, _mm_load_ss(a+4),  1);     
        } else if (length==6) {     
            v = _mm256_insertf128_ps(v, _mm_loadu_ps(a), 0);
            __m128d _tmp128d = _mm_load_sd(reinterpret_cast<const double *>(a+4));
            v =  _mm256_insertf128_ps(v, _mm_castpd_ps(_tmp128d), 1);     
        } else if (length==7) {   
            v = _mm256_insertf128_ps(v, _mm_loadu_ps(a), 0);
            __m128 _tmp128 = _mm_setr_ps(a[4],a[5], a[6], float(0));
            v = _mm256_insertf128_ps(v,_tmp128, 1);  
        } else if (length==8) { 
            v = _mm256_loadu_ps(a);
        }     
    }

    void store(float *a)                              { _mm256_storeu_ps(a, v);  }
    void storeu(float *a)                             { _mm256_storeu_ps(a, v);  }
    void store_aligned(float *a)                      { _mm256_store_ps(a, v);  }   
    void store_partial(float *a, const int length)  
    {
        ASSERT(length>=0 && length<=numElements);
        if (length==1) {  
            _mm_store_ss(a, _mm256_extractf128_ps(v, 0));   
        } else if (length==2) { 
            _mm_store_sd(reinterpret_cast<double *>(a), _mm_castps_pd(_mm256_extractf128_ps(v, 0)));  
        } else if (length==3) {    
            _mm_store_sd(reinterpret_cast<double *>(a), _mm_castps_pd(_mm256_extractf128_ps(v, 0)));
            _mm_store_ss(a+2, _mm_movehl_ps(_mm256_extractf128_ps(v, 0), _mm256_extractf128_ps(v, 0)));   
        } else if (length==4) { 
            _mm_storeu_ps(a, _mm256_extractf128_ps(v, 0));    
        } else if (length==5) {     
            _mm_storeu_ps(a, _mm256_extractf128_ps(v, 0));
            _mm_store_ss(a+4,_mm256_extractf128_ps(v, 1));    
        } else if (length==6) {
            _mm_storeu_ps(a, _mm256_extractf128_ps(v, 0));
            _mm_store_sd(reinterpret_cast<double *>(a+4), _mm_castps_pd(_mm256_extractf128_ps(v, 1)));     
        } else if (length==7) {
            _mm_storeu_ps(a, _mm256_extractf128_ps(v, 0));
            _mm_store_sd(reinterpret_cast<double *>(a+4), _mm_castps_pd(_mm256_extractf128_ps(v, 1)));
            _mm_store_ss(a+6, _mm_movehl_ps(_mm256_extractf128_ps(v, 0), _mm256_extractf128_ps(v, 1)));
        } else if (length==8) {
            _mm256_storeu_ps(a, v);
        }        
    }

    bool operator< (const IntrinsicsType & b) const 
    {
        __m256 _c = _mm256_cmp_ps(v, b.get(), _CMP_LT_OS);     
        return (_mm256_movemask_ps(_c)==255);

    }

    bool operator<= (const IntrinsicsType & b) const 
    {
        __m256 _c = _mm256_cmp_ps(v, b.get(), _CMP_LE_OS);      
        return (_mm256_movemask_ps(_c)==255);

    }

    bool operator> (const IntrinsicsType & b) const 
    {
        __m256 _c = _mm256_cmp_ps(v, b.get(), _CMP_GT_OS);   
        return (_mm256_movemask_ps(_c)==255);

    }

    bool operator>= (const IntrinsicsType & b) const 
    {
        __m256 _c = _mm256_cmp_ps(v, b.get(), _CMP_GE_OS);       
        return (_mm256_movemask_ps(_c)==255);

    }

    IntrinsicsType max () const 
    {
        typedef Intrinsics<float, IntrinsicsLevel::SSE>  IntrinsicsTypeSSE;
        IntrinsicsTypeSSE _v_0 (_mm256_extractf128_ps(v,0));
        IntrinsicsTypeSSE _v_1 (_mm256_extractf128_ps(v,1));
        __m128 _v_max = _mm_max_ps(_v_0.max().get(), _v_1.max().get());
        __m256 _result;
        _result = _mm256_insertf128_ps(_result, _v_max, 0);
        _result = _mm256_insertf128_ps(_result, _v_max, 1);
        return IntrinsicsType(_result);
    }
    
private:
    __m256                                   v;

};

template <>
class Intrinsics<double, IntrinsicsLevel::AVX> {

public:
    typedef double                                    DataType;
    typedef double                                    PrimitiveDataType;
    typedef __m256d                                   IntrinsicsDataType;
    typedef Intrinsics<double, IntrinsicsLevel::AVX>  IntrinsicsType;
    static  const int                                 numElements = 4;


    Intrinsics()                                      {}
    Intrinsics(const IntrinsicsDataType &val)         {v = val;}
    Intrinsics(double *a)                             {this->load(a);}
    Intrinsics(const double a)                        {this->fill(a);}

    void operator=(const IntrinsicsDataType &a)       {v = a; }
    void operator=(const IntrinsicsType &a)           {v = a.get(); }

    IntrinsicsDataType get() const                    {return v;}
    __m128d            get_high() const               {return _mm256_extractf128_pd(v,1);}

    void fill(double a)                               { v = _mm256_load1_pd(&a);}
    void setZero()                                    { v = _mm256_setzero_pd();}
    void load(const double *a)                        { v = _mm256_loadu_pd(a);}
    void loadu(const double *a)                        { v = _mm256_loadu_pd(a);}
    void load_aligned(const double *a)                { v = _mm256_load_pd(a);}
    void load_partial(const double *a, const int length)  
    {
        ASSERT(length>=0 && length<=numElements);
        if (length==1) {
            
            v = _mm256_insertf128_pd(v, _mm_load_sd(a), 0);
            
        } else if (length==2) {
            
            v =  _mm256_insertf128_pd(v, _mm_loadu_pd(a), 0);
            
        } else if (length==3) {
            
            v = _mm256_insertf128_pd(v, _mm_loadu_pd(a) , 0);
            v = _mm256_insertf128_pd(v, _mm_load_sd(a+2), 1);
            
        } else if (length==4) {
            
            v = _mm256_loadu_pd(a);
            
        }
    }
    void store(double *a)                             { _mm256_storeu_pd(a, v); }
    void storeu(double *a)                             { _mm256_storeu_pd(a, v); }
    void store_aligned(double *a)                     { _mm256_store_pd(a, v); }
    void store_partial(double *a, const int length)
    {
        
        ASSERT(length>=0 && length<=numElements);
        if (length==1) {
            _mm_store_sd(a, _mm256_extractf128_pd(v,0));
            
        } else if (length==2) {
            _mm_storeu_pd(a, _mm256_extractf128_pd(v,0));
            
        } else if (length==3) {
            _mm_storeu_pd(a  , _mm256_extractf128_pd(v,0));
            _mm_store_sd(a+2, _mm256_extractf128_pd(v,1));
            
        } else if (length==4) {
            
             _mm256_storeu_pd(a, v);
            
        }

    }

    bool operator< (const IntrinsicsType & b) const 
    {
        __m256d _c = _mm256_cmp_pd(v, b.get(), _CMP_LT_OS);      
        return (_mm256_movemask_pd(_c)==15);

    }

    bool operator<= (const IntrinsicsType & b) const 
    {
        __m256d _c = _mm256_cmp_pd(v, b.get(), _CMP_LE_OS);        
        return (_mm256_movemask_pd(_c)==15);

    }

    bool operator> (const IntrinsicsType & b) const 
    {
        __m256d _c = _mm256_cmp_pd(v, b.get(), _CMP_GT_OS);          
        return (_mm256_movemask_pd(_c)==15);

    }

    bool operator>= (const IntrinsicsType & b) const 
    {
        __m256d _c = _mm256_cmp_pd(v, b.get(), _CMP_GE_OS);         
        return (_mm256_movemask_pd(_c)==15);

    }

    IntrinsicsType max () const 
    {
        __m256d _tmp1 = _mm256_max_pd(v, _mm256_permute_pd(v,5));
        __m256d _tmp2 = _mm256_permute2f128_pd( _tmp1 , _tmp1 , 1);
        return IntrinsicsType(_mm256_max_pd(_tmp1, _tmp2));
    }


private:
    __m256d                                  v;

};

template <>
class Intrinsics<std::complex<float>, IntrinsicsLevel::AVX> {

public:
    typedef std::complex<float>              DataType;
    typedef float                            PrimitiveDataType;
    typedef __m256                           IntrinsicsDataType;
    static  const int                        numElements = 4;

    Intrinsics(void)                         {}
    Intrinsics(__m256 val)                   {v = val;}
    Intrinsics(std::complex<float> *a)       {this->load(a);}
    Intrinsics(float a)                      {this->fill(a);}

    void operator=(std::complex<float> *a)   {this->load(a);}
    void operator=(__m256 a)                 {v = a; }

    __m256 get(void) const                   {return v;}

    void fill(float a)                       { v = _mm256_broadcast_ss(&a); }
    void load(const std::complex<float> *a)  { v = _mm256_loadu_ps(reinterpret_cast<const float* >(a));}
    void loadu(const std::complex<float> *a) { v = _mm256_loadu_ps(reinterpret_cast<const float* >(a));}
    void load_partial(const std::complex<float> *a, const int length)
    {
        ASSERT(length>=0 && length<=numElements);
        if (length==1) {
            __m128d _tmp128d = _mm_load_sd(reinterpret_cast<const double *>(a));
            v =  _mm256_insertf128_ps(v, _mm_castpd_ps(_tmp128d), 0);
        } else if (length==2) {
            v = _mm256_insertf128_ps(v, _mm_loadu_ps(reinterpret_cast<const float*>(a)), 0);
        } else if (length==3) {
            v = _mm256_insertf128_ps(v, _mm_loadu_ps(reinterpret_cast<const float*>(a)), 0);
            __m128d _tmp128d = _mm_load_sd(reinterpret_cast<const double *>(a+2));
            v =  _mm256_insertf128_ps(v, _mm_castpd_ps(_tmp128d), 1);
        } else if (length==4) {
            v = _mm256_loadu_ps(reinterpret_cast<const float *>(a));
        }
    }

    void setZero()                           { v = _mm256_setzero_ps();}
    void store(std::complex<float> *a)       { _mm256_storeu_ps(reinterpret_cast<float* >(a), v); }
    void storeu(std::complex<float> *a)      { _mm256_storeu_ps(reinterpret_cast<float* >(a), v); }
    void store_partial(std::complex<float> *a, const int length)
    {
        ASSERT(length>=0 && length<=numElements);
        if (length==1) {
            _mm_store_sd(reinterpret_cast<double *>(a), _mm_castps_pd(_mm256_extractf128_ps(v, 0)));
        } else if (length==2) {
            _mm_storeu_ps(reinterpret_cast<float *>(a), _mm256_extractf128_ps(v, 0));
        } else if (length==3) {
            _mm_storeu_ps(reinterpret_cast<float *>(a), _mm256_extractf128_ps(v, 0));
            _mm_store_sd(reinterpret_cast<double *>(a+2), _mm_castps_pd(_mm256_extractf128_ps(v, 1)));
        } else if (length==4) {
            _mm256_storeu_ps(reinterpret_cast<float*>(a), v);
        }
    }
    void stream(std::complex<float> *a)      { _mm256_stream_ps(reinterpret_cast<float *>(a), v); }

private:
    __m256                                   v;

};

template <>
class Intrinsics<std::complex<double>, IntrinsicsLevel::AVX> {

public:
    typedef std::complex<double>             DataType;
    typedef double                           PrimitiveDataType;
    typedef __m256d                          IntrinsicsDataType;
    static  const int                        numElements = 2;

    Intrinsics(void)                         {}
    Intrinsics(__m256d val)                  {v = val;}
    Intrinsics(std::complex<double> *a)      {this->load(a);}
    Intrinsics(double a)                     {this->fill(a);}

    void operator=(std::complex<double> *a)  {this->load(a);}
    void operator=(__m256d a)                {v = a; }


    __m256d get(void) const                  {return v;}

    void fill(double a)                      { v = _mm256_broadcast_sd(&a); }
    void load(const std::complex<double> *a) { v = _mm256_loadu_pd(reinterpret_cast<const double* >(a));}
    void loadu(const std::complex<double> *a){ v = _mm256_loadu_pd(reinterpret_cast<const double* >(a));}
    void load_partial(const std::complex<double> *a, const int length)
    {
        ASSERT(length>=0 && length<=numElements);
        if (length==1) {
            v =  _mm256_insertf128_pd(v, _mm_loadu_pd(reinterpret_cast<const double*>(a)), 0);
        } else if (length==2) {
            v = _mm256_loadu_pd(reinterpret_cast<const double*>(a));
        }
    }
    void setZero()                           { v = _mm256_setzero_pd();}
    void store(std::complex<double> *a)      { _mm256_storeu_pd(reinterpret_cast<double* >(a), v);}
    void storeu(std::complex<double> *a)     { _mm256_storeu_pd(reinterpret_cast<double* >(a), v);}
    void store_partial(std::complex<double> *a, const int length)
    {
        ASSERT(length>=0 && length<=numElements);
        if (length==1) {
            _mm_storeu_pd(reinterpret_cast<double*>(a), _mm256_extractf128_pd(v,0));
        } else if (length==2) {
             _mm256_storeu_pd(reinterpret_cast<double*>(a), v);
        }

    }
    void stream(std::complex<double> *a)     { _mm256_stream_pd(reinterpret_cast<double* >(a), v); }
private:
    __m256d                                  v;

};

#endif // HAVE_AVX

#endif // PLAYGROUND_CXXBLAS_INTRINSICS_CLASSES_AVX_H

