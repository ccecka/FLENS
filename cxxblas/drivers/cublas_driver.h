#ifndef CXXBLAS_DRIVERS_CUBLAS_H
#define CXXBLAS_DRIVERS_CUBLAS_H 1

// NO, cublas provides a mapping to the cblas extern below
//#   define HAVE_CBLAS       1
//#   define HAVE_CBLAS_EXT

#define CBLAS_INT int
#define BLAS_IMPL "CUBLAS"
#ifndef CBLAS_INDEX
#define CBLAS_INDEX int
#endif // CBLAS_INDEX

#include <cublas.h>

float
cblas_sasum(int n, const float *x, int incX) {
  return cublasSasum(n, x, incX);
}

double
cblas_dasum(int n, const double *x, int incX) {
  return cublasDasum(n, x, incX);
}

float
cblas_scasum(int n, const float *x, int incX) {
  return cublasScasum(n, reinterpret_cast<const cuComplex *>(x), incX);
}

double
cblas_dzasum(int n, const double *x, int incX) {
  return cublasDzasum(n, reinterpret_cast<const cuDoubleComplex *>(x), incX);
}

void
cblas_saxpy(int n, float alpha, const float *x, int incX, float *y,
            int incY) {
  return cublasSaxpy(n, alpha, x, incX, y, incY);
}

void
cblas_daxpy(int n, double alpha, const double *x, int incX, double *y,
            int incY) {
  return cublasDaxpy(n, alpha, x, incX, y, incY);
}

void
cblas_caxpy(int n, const float *alpha, const float *x, int incX, float *y,
            int incY) {
  return cublasCaxpy(n, *reinterpret_cast<const cuComplex *>(alpha),
                     reinterpret_cast<const cuComplex *>(x), incX,
                     reinterpret_cast<cuComplex *>(y), incY);
}

void
cblas_zaxpy(int n, const double *alpha, const double *x, int incX,
            double *y, int incY) {
  return cublasZaxpy(n, *reinterpret_cast<const cuDoubleComplex *>(alpha),
                     reinterpret_cast<const cuDoubleComplex *>(x), incX,
                     reinterpret_cast<cuDoubleComplex *>(y), incY);
}

void
cblas_scopy(int n, const float *x, int incX, float *y, int incY) {
  return cublasScopy(n, x, incX, y, incY);
}

void
cblas_dcopy(int n, const double *x, int incX, double *y, int incY) {
  return cublasDcopy(n, x, incX, y, incY);
}

void
cblas_ccopy(int n, const float *x, int incX, float *y, int incY) {
  return cublasCcopy(n, reinterpret_cast<const cuComplex *>(x), incX,
                     reinterpret_cast<cuComplex *>(y), incY);
}

void
cblas_zcopy(int n, const double *x, int incX, double *y, int incY) {
  return cublasZcopy(n, reinterpret_cast<const cuDoubleComplex *>(x), incX,
                     reinterpret_cast<cuDoubleComplex *>(y), incY);
}

float
cblas_sdot(int n, const float *x, int incX, const float *y, int incY) {
  return cublasSdot(n, x, incX, y, incY);
}

double
cblas_ddot(int n, const double *x, int incX, const double *y, int incY) {
  return cublasDdot(n, x, incX, y, incY);
}

int
cblas_isamax(int n, const float *x, int incX) {
  return cublasIsamax(n, x, incX);
}

int
cblas_idamax(int n, const double *x, int incX) {
  return cublasIdamax(n, x, incX);
}

int
cblas_icamax(int n, const float *x, int incX) {
  return cublasIcamax(n, reinterpret_cast<const cuComplex *>(x), incX);
}

int
cblas_izamax(int n, const double *x, int incX) {
  return cublasIzamax(n, reinterpret_cast<const cuDoubleComplex *>(x), incX);
}

float
cblas_snrm2(int n, const float *X, int incX) {
  return cublasSnrm2(n, X, incX);
}

double
cblas_dnrm2(int n, const double *X, int incX) {
  return cublasDnrm2(n, X, incX);
}

float
cblas_scnrm2(int n, const float *X, int incX) {
  return cublasScnrm2(n, reinterpret_cast<const cuComplex *>(X), incX);
}

double
cblas_dznrm2(int n, const double *X, int incX) {
  return cublasDznrm2(n, reinterpret_cast<const cuDoubleComplex *>(X), incX);
}

void
cblas_srot(int n, float *X, int incX, float *Y, int incY, float c,
           float s) {
  return cublasSrot(n, X, incX, Y, incY, c, s);
}

void
cblas_drot(int n, double *X, int incX, double *Y, int incY, double c,
           double s) {
  return cublasDrot(n, X, incX, Y, incY, c, s);
}

void
cblas_srotg(float *a, float *b, float *c, float *s) {
  return cublasSrotg(a, b, c, s);
}

void
cblas_drotg(double *a, double *b, double *c, double *s) {
  return cublasDrotg(a, b, c, s);
}

void
cblas_srotm(int n, float *X, int incX, float *Y, int incY,
            const float *P) {
  return cublasSrotm(n, X, incX, Y, incY, P);
}

void
cblas_drotm(int n, double *X, int incX, double *Y, int incY,
            const double *P) {
  return cublasDrotm(n, X, incX, Y, incY, P);
}

void
cblas_srotmg(float *d1, float *d2, float *b1, float *b2, float *P) {
  return cublasSrotmg(d1, d2, b1, reinterpret_cast<const float *>(b2), P);
}

void
cblas_drotmg(double *d1, double *d2, double *b1, double *b2, double *P) {
  return cublasDrotmg(d1, d2, b1, reinterpret_cast<const double *>(b2), P);
}

void
cblas_sscal(int n, float alpha, float *x, int incX) {
  return cublasSscal(n, alpha, x, incX);
}

void
cblas_dscal(int n, double alpha, double *x, int incX) {
  return cublasDscal(n, alpha, x, incX);
}

void
cblas_cscal(int n, const float *alpha, float *x, int incX) {
  return cublasCscal(n, *reinterpret_cast<const cuComplex *>(alpha),
                     reinterpret_cast<cuComplex *>(x), incX);
}

void
cblas_zscal(int n, const double *alpha, double *x, int incX) {
  return cublasZscal(n, *reinterpret_cast<const cuDoubleComplex *>(alpha),
                     reinterpret_cast<cuDoubleComplex *>(x), incX);
}

void
cblas_csscal(int n, float alpha, float *x, int incX) {
  return cublasCsscal(n, alpha, reinterpret_cast<cuComplex *>(x), incX);
}

void
cblas_zdscal(int n, double alpha, double *x, int incX) {
  return cublasZdscal(n, alpha, reinterpret_cast<cuDoubleComplex *>(x), incX);
}

void
cblas_sswap(int n, float *x, int incX, float *y, int incY) {
  return cublasSswap(n, x, incX, y, incY);
}

void
cblas_dswap(int n, double *x, int incX, double *y, int incY) {
  return cublasDswap(n, x, incX, y, incY);
}

void
cblas_cswap(int n, float *x, int incX, float *y, int incY) {
  return cublasCswap(n, reinterpret_cast<cuComplex *>(x), incX,
                     reinterpret_cast<cuComplex *>(y), incY);
}

void
cblas_zswap(int n, double *x, int incX, double *y, int incY) {
  return cublasZswap(n, reinterpret_cast<cuDoubleComplex *>(x), incX,
                     reinterpret_cast<cuDoubleComplex *>(y), incY);
}

void
cblas_sgbmv(char order, char trans, int m, int n, int kl, int ku,
            float alpha, const float *A, int ldA, const float *x, int incX,
            float beta, float *y, int incY) {
  (void)order;
  assert(order == 102);
  return cublasSgbmv(trans, m, n, kl, ku, alpha, A, ldA, x, incX, beta, y,
                     incY);
}

void
cblas_dgbmv(char order, char trans, int m, int n, int kl, int ku,
            double alpha, const double *A, int ldA, const double *x,
            int incX, double beta, double *y, int incY) {
  (void)order;
  assert(order == 102);
  return cublasDgbmv(trans, m, n, kl, ku, alpha, A, ldA, x, incX, beta, y,
                     incY);
}

void
cblas_cgbmv(char order, char trans, int m, int n, int kl, int ku,
            const float *alpha, const float *A, int ldA, const float *x,
            int incX, const float *beta, float *y, int incY) {
  (void)order;
  assert(order == 102);
  return cublasCgbmv(trans, m, n, kl, ku,
                     *reinterpret_cast<const cuComplex *>(alpha),
                     reinterpret_cast<const cuComplex *>(A), ldA,
                     reinterpret_cast<const cuComplex *>(x), incX,
                     *reinterpret_cast<const cuComplex *>(beta),
                     reinterpret_cast<cuComplex *>(y), incY);
}

void
cblas_zgbmv(char order, char trans, int m, int n, int kl, int ku,
            const double *alpha, const double *A, int ldA, const double *x,
            int incX, const double *beta, double *y, int incY) {
  (void)order;
  assert(order == 102);
  return cublasZgbmv(trans, m, n, kl, ku,
                     *reinterpret_cast<const cuDoubleComplex *>(alpha),
                     reinterpret_cast<const cuDoubleComplex *>(A), ldA,
                     reinterpret_cast<const cuDoubleComplex *>(x), incX,
                     *reinterpret_cast<const cuDoubleComplex *>(beta),
                     reinterpret_cast<cuDoubleComplex *>(y), incY);
}

void
cblas_sgemv(char order, char trans, int m, int n, float alpha,
            const float *A, int ldA, const float *x, int incX, float beta,
            float *y, int incY) {
  (void)order;
  assert(order == 102);
  return cublasSgemv(trans, m, n, alpha, A, ldA, x, incX, beta, y, incY);
}

void
cblas_dgemv(char order, char trans, int m, int n, double alpha,
            const double *A, int ldA, const double *x, int incX,
            double beta, double *y, int incY) {
  (void)order;
  assert(order == 102);
  return cublasDgemv(trans, m, n, alpha, A, ldA, x, incX, beta, y, incY);
}

void
cblas_cgemv(char order, char trans, int m, int n, const float *alpha,
            const float *A, int ldA, const float *x, int incX,
            const float *beta, float *y, int incY) {
  (void)order;
  assert(order == 102);
  return cublasCgemv(trans, m, n, *reinterpret_cast<const cuComplex *>(alpha),
                     reinterpret_cast<const cuComplex *>(A), ldA,
                     reinterpret_cast<const cuComplex *>(x), incX,
                     *reinterpret_cast<const cuComplex *>(beta),
                     reinterpret_cast<cuComplex *>(y), incY);
}

void
cblas_zgemv(char order, char trans, int m, int n, const double *alpha,
            const double *A, int ldA, const double *x, int incX,
            const double *beta, double *y, int incY) {
  (void)order;
  assert(order == 102);
  return cublasZgemv(trans, m, n,
                     *reinterpret_cast<const cuDoubleComplex *>(alpha),
                     reinterpret_cast<const cuDoubleComplex *>(A), ldA,
                     reinterpret_cast<const cuDoubleComplex *>(x), incX,
                     *reinterpret_cast<const cuDoubleComplex *>(beta),
                     reinterpret_cast<cuDoubleComplex *>(y), incY);
}

void
cblas_ssbmv(char order, char upLo, int n, int k, float alpha,
            const float *A, int ldA, const float *x, int incX, float beta,
            float *y, int incY) {
  (void)order;
  assert(order == 102);
  return cublasSsbmv(upLo, n, k, alpha, A, ldA, x, incX, beta, y, incY);
}

void
cblas_dsbmv(char order, char upLo, int n, int k, double alpha,
            const double *A, int ldA, const double *x, int incX,
            double beta, double *y, int incY) {
  (void)order;
  assert(order == 102);
  return cublasDsbmv(upLo, n, k, alpha, A, ldA, x, incX, beta, y, incY);
}

void
cblas_ssymv(char order, char upLo, int n, float alpha, const float *A,
            int ldA, const float *x, int incX, float beta, float *y,
            int incY) {
  (void)order;
  assert(order == 102);
  return cublasSsymv(upLo, n, alpha, A, ldA, x, incX, beta, y, incY);
}

void
cblas_dsymv(char order, char upLo, int n, double alpha, const double *A,
            int ldA, const double *x, int incX, double beta, double *y,
            int incY) {
  (void)order;
  assert(order == 102);
  return cublasDsymv(upLo, n, alpha, A, ldA, x, incX, beta, y, incY);
}

void
cblas_sspmv(char order, char upLo, int n, float alpha, const float *Ap,
            const float *x, int incX, float beta, float *y, int incY) {
  (void)order;
  assert(order == 102);
  return cublasSspmv(upLo, n, alpha, Ap, x, incX, beta, y, incY);
}

void
cblas_dspmv(char order, char upLo, int n, double alpha, const double *Ap,
            const double *x, int incX, double beta, double *y, int incY) {
  (void)order;
  assert(order == 102);
  return cublasDspmv(upLo, n, alpha, Ap, x, incX, beta, y, incY);
}

void
cblas_chbmv(char order, char upLo, int n, int k, const float *alpha,
            const float *A, int ldA, const float *x, int incX,
            const float *beta, float *y, int incY) {
  (void)order;
  assert(order == 102);
  return cublasChbmv(upLo, n, k, *reinterpret_cast<const cuComplex *>(alpha),
                     reinterpret_cast<const cuComplex *>(A), ldA,
                     reinterpret_cast<const cuComplex *>(x), incX,
                     *reinterpret_cast<const cuComplex *>(beta),
                     reinterpret_cast<cuComplex *>(y), incY);
}

void
cblas_zhbmv(char order, char upLo, int n, int k, const double *alpha,
            const double *A, int ldA, const double *x, int incX,
            const double *beta, double *y, int incY) {
  (void)order;
  assert(order == 102);
  return cublasZhbmv(upLo, n, k,
                     *reinterpret_cast<const cuDoubleComplex *>(alpha),
                     reinterpret_cast<const cuDoubleComplex *>(A), ldA,
                     reinterpret_cast<const cuDoubleComplex *>(x), incX,
                     *reinterpret_cast<const cuDoubleComplex *>(beta),
                     reinterpret_cast<cuDoubleComplex *>(y), incY);
}

void
cblas_chemv(char order, char upLo, int n, const float *alpha,
            const float *A, int ldA, const float *x, int incX,
            const float *beta, float *y, int incY) {
  (void)order;
  assert(order == 102);
  return cublasChemv(upLo, n, *reinterpret_cast<const cuComplex *>(alpha),
                     reinterpret_cast<const cuComplex *>(A), ldA,
                     reinterpret_cast<const cuComplex *>(x), incX,
                     *reinterpret_cast<const cuComplex *>(beta),
                     reinterpret_cast<cuComplex *>(y), incY);
}

void
cblas_zhemv(char order, char upLo, int n, const double *alpha,
            const double *A, int ldA, const double *x, int incX,
            const double *beta, double *y, int incY) {
  (void)order;
  assert(order == 102);
  return cublasZhemv(upLo, n, *reinterpret_cast<const cuDoubleComplex *>(alpha),
                     reinterpret_cast<const cuDoubleComplex *>(A), ldA,
                     reinterpret_cast<const cuDoubleComplex *>(x), incX,
                     *reinterpret_cast<const cuDoubleComplex *>(beta),
                     reinterpret_cast<cuDoubleComplex *>(y), incY);
}

void
cblas_chpmv(char order, char upLo, int n, const float *alpha,
            const float *Ap, const float *x, int incX, const float *beta,
            float *y, int incY) {
  (void)order;
  assert(order == 102);
  return cublasChpmv(upLo, n, *reinterpret_cast<const cuComplex *>(alpha),
                     reinterpret_cast<const cuComplex *>(Ap),
                     reinterpret_cast<const cuComplex *>(x), incX,
                     *reinterpret_cast<const cuComplex *>(beta),
                     reinterpret_cast<cuComplex *>(y), incY);
}

void
cblas_zhpmv(char order, char upLo, int n, const double *alpha,
            const double *Ap, const double *x, int incX,
            const double *beta, double *y, int incY) {
  (void)order;
  assert(order == 102);
  return cublasZhpmv(upLo, n, *reinterpret_cast<const cuDoubleComplex *>(alpha),
                     reinterpret_cast<const cuDoubleComplex *>(Ap),
                     reinterpret_cast<const cuDoubleComplex *>(x), incX,
                     *reinterpret_cast<const cuDoubleComplex *>(beta),
                     reinterpret_cast<cuDoubleComplex *>(y), incY);
}

void
cblas_stbsv(char order, char upLo, char transA, char diag, int n, int k,
            const float *A, int lda, float *X, int incX) {
  (void)order;
  assert(order == 102);
  return cublasStbsv(upLo, transA, diag, n, k, A, lda, X, incX);
}

void
cblas_dtbsv(char order, char upLo, char transA, char diag, int n, int k,
            const double *A, int lda, double *X, int incX) {
  (void)order;
  assert(order == 102);
  return cublasDtbsv(upLo, transA, diag, n, k, A, lda, X, incX);
}

void
cblas_ctbsv(char order, char upLo, char transA, char diag, int n, int k,
            const float *A, int lda, float *X, int incX) {
  (void)order;
  assert(order == 102);
  return cublasCtbsv(upLo, transA, diag, n, k,
                     reinterpret_cast<const cuComplex *>(A), lda,
                     reinterpret_cast<cuComplex *>(X), incX);
}

void
cblas_ztbsv(char order, char upLo, char transA, char diag, int n, int k,
            const double *A, int lda, double *X, int incX) {
  (void)order;
  assert(order == 102);
  return cublasZtbsv(upLo, transA, diag, n, k,
                     reinterpret_cast<const cuDoubleComplex *>(A), lda,
                     reinterpret_cast<cuDoubleComplex *>(X), incX);
}

void
cblas_strsv(char order, char upLo, char transA, char diag, int n,
            const float *A, int lda, float *X, int incX) {
  (void)order;
  assert(order == 102);
  return cublasStrsv(upLo, transA, diag, n, A, lda, X, incX);
}

void
cblas_dtrsv(char order, char upLo, char transA, char diag, int n,
            const double *A, int lda, double *X, int incX) {
  (void)order;
  assert(order == 102);
  return cublasDtrsv(upLo, transA, diag, n, A, lda, X, incX);
}

void
cblas_ctrsv(char order, char upLo, char transA, char diag, int n,
            const float *A, int lda, float *X, int incX) {
  (void)order;
  assert(order == 102);
  return cublasCtrsv(upLo, transA, diag, n,
                     reinterpret_cast<const cuComplex *>(A), lda,
                     reinterpret_cast<cuComplex *>(X), incX);
}

void
cblas_ztrsv(char order, char upLo, char transA, char diag, int n,
            const double *A, int lda, double *X, int incX) {
  (void)order;
  assert(order == 102);
  return cublasZtrsv(upLo, transA, diag, n,
                     reinterpret_cast<const cuDoubleComplex *>(A), lda,
                     reinterpret_cast<cuDoubleComplex *>(X), incX);
}

void
cblas_stpsv(char order, char upLo, char transA, char diag, int n,
            const float *A, float *X, int incX) {
  (void)order;
  assert(order == 102);
  return cublasStpsv(upLo, transA, diag, n, A, X, incX);
}

void
cblas_dtpsv(char order, char upLo, char transA, char diag, int n,
            const double *A, double *X, int incX) {
  (void)order;
  assert(order == 102);
  return cublasDtpsv(upLo, transA, diag, n, A, X, incX);
}

void
cblas_ctpsv(char order, char upLo, char transA, char diag, int n,
            const float *A, float *X, int incX) {
  (void)order;
  assert(order == 102);
  return cublasCtpsv(upLo, transA, diag, n,
                     reinterpret_cast<const cuComplex *>(A),
                     reinterpret_cast<cuComplex *>(X), incX);
}

void
cblas_ztpsv(char order, char upLo, char transA, char diag, int n,
            const double *A, double *X, int incX) {
  (void)order;
  assert(order == 102);
  return cublasZtpsv(upLo, transA, diag, n,
                     reinterpret_cast<const cuDoubleComplex *>(A),
                     reinterpret_cast<cuDoubleComplex *>(X), incX);
}

void
cblas_stbmv(char order, char upLo, char transA, char diag, int n, int k,
            const float *A, int lda, float *x, int incX) {
  (void)order;
  assert(order == 102);
  return cublasStbmv(upLo, transA, diag, n, k, A, lda, x, incX);
}

void
cblas_dtbmv(char order, char upLo, char transA, char diag, int n, int k,
            const double *A, int lda, double *x, int incX) {
  (void)order;
  assert(order == 102);
  return cublasDtbmv(upLo, transA, diag, n, k, A, lda, x, incX);
}

void
cblas_ctbmv(char order, char upLo, char transA, char diag, int n, int k,
            const float *A, int lda, float *x, int incX) {
  (void)order;
  assert(order == 102);
  return cublasCtbmv(upLo, transA, diag, n, k,
                     reinterpret_cast<const cuComplex *>(A), lda,
                     reinterpret_cast<cuComplex *>(x), incX);
}

void
cblas_ztbmv(char order, char upLo, char transA, char diag, int N, int k,
            const double *A, int lda, double *x, int incX) {
  (void)order;
  assert(order == 102);
  return cublasZtbmv(upLo, transA, diag, N, k,
                     reinterpret_cast<const cuDoubleComplex *>(A), lda,
                     reinterpret_cast<cuDoubleComplex *>(x), incX);
}

void
cblas_strmv(char order, char upLo, char transA, char diag, int n,
            const float *A, int lda, float *x, int incX) {
  (void)order;
  assert(order == 102);
  return cublasStrmv(upLo, transA, diag, n, A, lda, x, incX);
}

void
cblas_dtrmv(char order, char upLo, char transA, char diag, int n,
            const double *A, int lda, double *x, int incX) {
  (void)order;
  assert(order == 102);
  return cublasDtrmv(upLo, transA, diag, n, A, lda, x, incX);
}

void
cblas_ctrmv(char order, char upLo, char transA, char diag, int n,
            const float *A, int lda, float *x, int incX) {
  (void)order;
  assert(order == 102);
  return cublasCtrmv(upLo, transA, diag, n,
                     reinterpret_cast<const cuComplex *>(A), lda,
                     reinterpret_cast<cuComplex *>(x), incX);
}

void
cblas_ztrmv(char order, char upLo, char transA, char diag, int N,
            const double *A, int lda, double *x, int incX) {
  (void)order;
  assert(order == 102);
  return cublasZtrmv(upLo, transA, diag, N,
                     reinterpret_cast<const cuDoubleComplex *>(A), lda,
                     reinterpret_cast<cuDoubleComplex *>(x), incX);
}

void
cblas_stpmv(char order, char upLo, char transA, char diag, int n,
            const float *Ap, float *x, int incX) {
  (void)order;
  assert(order == 102);
  return cublasStpmv(upLo, transA, diag, n, Ap, x, incX);
}

void
cblas_dtpmv(char order, char upLo, char transA, char diag, int n,
            const double *Ap, double *x, int incX) {
  (void)order;
  assert(order == 102);
  return cublasDtpmv(upLo, transA, diag, n, Ap, x, incX);
}

void
cblas_ctpmv(char order, char upLo, char transA, char diag, int n,
            const float *Ap, float *x, int incX) {
  (void)order;
  assert(order == 102);
  return cublasCtpmv(upLo, transA, diag, n,
                     reinterpret_cast<const cuComplex *>(Ap),
                     reinterpret_cast<cuComplex *>(x), incX);
}

void
cblas_ztpmv(char order, char upLo, char transA, char diag, int N,
            const double *Ap, double *x, int incX) {
  (void)order;
  assert(order == 102);
  return cublasZtpmv(upLo, transA, diag, N,
                     reinterpret_cast<const cuDoubleComplex *>(Ap),
                     reinterpret_cast<cuDoubleComplex *>(x), incX);
}

void
cblas_sger(char order, int m, int n, float alpha, const float *X, int incX,
           const float *Y, int incY, float *A, int lda) {
  (void)order;
  assert(order == 102);
  return cublasSger(m, n, alpha, X, incX, Y, incY, A, lda);
}

void
cblas_dger(char order, int m, int n, double alpha, const double *X,
           int incX, const double *Y, int incY, double *A, int lda) {
  (void)order;
  assert(order == 102);
  return cublasDger(m, n, alpha, X, incX, Y, incY, A, lda);
}

void
cblas_cgeru(char order, int m, int n, const float *alpha, const float *X,
            int incX, const float *Y, int incY, float *A, int lda) {
  (void)order;
  assert(order == 102);
  return cublasCgeru(m, n, *reinterpret_cast<const cuComplex *>(alpha),
                     reinterpret_cast<const cuComplex *>(X), incX,
                     reinterpret_cast<const cuComplex *>(Y), incY,
                     reinterpret_cast<cuComplex *>(A), lda);
}

void
cblas_cgerc(char order, int m, int n, const float *alpha, const float *X,
            int incX, const float *Y, int incY, float *A, int lda) {
  (void)order;
  assert(order == 102);
  return cublasCgerc(m, n, *reinterpret_cast<const cuComplex *>(alpha),
                     reinterpret_cast<const cuComplex *>(X), incX,
                     reinterpret_cast<const cuComplex *>(Y), incY,
                     reinterpret_cast<cuComplex *>(A), lda);
}

void
cblas_zgeru(char order, int m, int n, const double *alpha, const double *X,
            int incX, const double *Y, int incY, double *A, int lda) {
  (void)order;
  assert(order == 102);
  return cublasZgeru(m, n, *reinterpret_cast<const cuDoubleComplex *>(alpha),
                     reinterpret_cast<const cuDoubleComplex *>(X), incX,
                     reinterpret_cast<const cuDoubleComplex *>(Y), incY,
                     reinterpret_cast<cuDoubleComplex *>(A), lda);
}

void
cblas_zgerc(char order, int m, int n, const double *alpha, const double *X,
            int incX, const double *Y, int incY, double *A, int lda) {
  (void)order;
  assert(order == 102);
  return cublasZgerc(m, n, *reinterpret_cast<const cuDoubleComplex *>(alpha),
                     reinterpret_cast<const cuDoubleComplex *>(X), incX,
                     reinterpret_cast<const cuDoubleComplex *>(Y), incY,
                     reinterpret_cast<cuDoubleComplex *>(A), lda);
}

void
cblas_ssyr(char order, char upLo, int n, float alpha, const float *X,
           int incX, float *A, int lda) {
  (void)order;
  assert(order == 102);
  return cublasSsyr(upLo, n, alpha, X, incX, A, lda);
}

void
cblas_dsyr(char order, char upLo, int n, double alpha, const double *X,
           int incX, double *A, int lda) {
  (void)order;
  assert(order == 102);
  return cublasDsyr(upLo, n, alpha, X, incX, A, lda);
}

void
cblas_sspr(char order, char upLo, int n, float alpha, const float *X,
           int incX, float *A) {
  (void)order;
  assert(order == 102);
  return cublasSspr(upLo, n, alpha, X, incX, A);
}

void
cblas_dspr(char order, char upLo, int n, double alpha, const double *X,
           int incX, double *A) {
  (void)order;
  assert(order == 102);
  return cublasDspr(upLo, n, alpha, X, incX, A);
}

void
cblas_cher(char order, char upLo, int n, float alpha, const float *X,
           int incX, float *A, int lda) {
  (void)order;
  assert(order == 102);
  return cublasCher(upLo, n, alpha, reinterpret_cast<const cuComplex *>(X),
                    incX, reinterpret_cast<cuComplex *>(A), lda);
}

void
cblas_zher(char order, char upLo, int n, double alpha, const double *X,
           int incX, double *A, int lda) {
  (void)order;
  assert(order == 102);
  return cublasZher(upLo, n, alpha,
                    reinterpret_cast<const cuDoubleComplex *>(X), incX,
                    reinterpret_cast<cuDoubleComplex *>(A), lda);
}

void
cblas_chpr(char order, char upLo, int n, float alpha, const float *X,
           int incX, float *A) {
  (void)order;
  assert(order == 102);
  return cublasChpr(upLo, n, alpha, reinterpret_cast<const cuComplex *>(X),
                    incX, reinterpret_cast<cuComplex *>(A));
}

void
cblas_zhpr(char order, char upLo, int n, double alpha, const double *X,
           int incX, double *A) {
  (void)order;
  assert(order == 102);
  return cublasZhpr(upLo, n, alpha,
                    reinterpret_cast<const cuDoubleComplex *>(X), incX,
                    reinterpret_cast<cuDoubleComplex *>(A));
}

void
cblas_sspr2(char order, char upLo, int n, float alpha, const float *X,
            int incX, const float *Y, int incY, float *A) {
  (void)order;
  assert(order == 102);
  return cublasSspr2(upLo, n, alpha, X, incX, Y, incY, A);
}

void
cblas_dspr2(char order, char upLo, int n, double alpha, const double *X,
            int incX, const double *Y, int incY, double *A) {
  (void)order;
  assert(order == 102);
  return cublasDspr2(upLo, n, alpha, X, incX, Y, incY, A);
}

void
cblas_ssyr2(char order, char upLo, int n, float alpha, const float *X,
            int incX, const float *Y, int incY, float *A, int lda) {
  (void)order;
  assert(order == 102);
  return cublasSsyr2(upLo, n, alpha, X, incX, Y, incY, A, lda);
}

void
cblas_dsyr2(char order, char upLo, int n, double alpha, const double *X,
            int incX, const double *Y, int incY, double *A, int lda) {
  (void)order;
  assert(order == 102);
  return cublasDsyr2(upLo, n, alpha, X, incX, Y, incY, A, lda);
}

void
cblas_cher2(char order, char upLo, int n, const float *alpha,
            const float *X, int incX, const float *Y, int incY, float *A,
            int lda) {
  (void)order;
  assert(order == 102);
  return cublasCher2(upLo, n, *reinterpret_cast<const cuComplex *>(alpha),
                     reinterpret_cast<const cuComplex *>(X), incX,
                     reinterpret_cast<const cuComplex *>(Y), incY,
                     reinterpret_cast<cuComplex *>(A), lda);
}

void
cblas_zher2(char order, char upLo, int n, const double *alpha,
            const double *X, int incX, const double *Y, int incY,
            double *A, int lda) {
  (void)order;
  assert(order == 102);
  return cublasZher2(upLo, n, *reinterpret_cast<const cuDoubleComplex *>(alpha),
                     reinterpret_cast<const cuDoubleComplex *>(X), incX,
                     reinterpret_cast<const cuDoubleComplex *>(Y), incY,
                     reinterpret_cast<cuDoubleComplex *>(A), lda);
}

void
cblas_chpr2(char order, char upLo, int n, const float *alpha,
            const float *X, int incX, const float *Y, int incY, float *A) {
  (void)order;
  assert(order == 102);
  return cublasChpr2(upLo, n, *reinterpret_cast<const cuComplex *>(alpha),
                     reinterpret_cast<const cuComplex *>(X), incX,
                     reinterpret_cast<const cuComplex *>(Y), incY,
                     reinterpret_cast<cuComplex *>(A));
}

void
cblas_zhpr2(char order, char upLo, int n, const double *alpha,
            const double *X, int incX, const double *Y, int incY,
            double *A) {
  (void)order;
  assert(order == 102);
  return cublasZhpr2(upLo, n, *reinterpret_cast<const cuDoubleComplex *>(alpha),
                     reinterpret_cast<const cuDoubleComplex *>(X), incX,
                     reinterpret_cast<const cuDoubleComplex *>(Y), incY,
                     reinterpret_cast<cuDoubleComplex *>(A));
}

void
cblas_sgemm(char order, char transA, char transB, int m, int n, int k,
            float alpha, const float *A, int ldA, const float *B, int ldB,
            float beta, float *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasSgemm(transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C,
                     ldC);
}

void
cblas_dgemm(char order, char transA, char transB, int m, int n, int k,
            double alpha, const double *A, int ldA, const double *B,
            int ldB, double beta, double *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasDgemm(transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C,
                     ldC);
}

void
cblas_cgemm(char order, char transA, char transB, int m, int n, int k,
            const float *alpha, const float *A, int ldA, const float *B,
            int ldB, const float *beta, float *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasCgemm(transA, transB, m, n, k,
                     *reinterpret_cast<const cuComplex *>(alpha),
                     reinterpret_cast<const cuComplex *>(A), ldA,
                     reinterpret_cast<const cuComplex *>(B), ldB,
                     *reinterpret_cast<const cuComplex *>(beta),
                     reinterpret_cast<cuComplex *>(C), ldC);
}

void
cblas_zgemm(char order, char transA, char transB, int m, int n, int k,
            const double *alpha, const double *A, int ldA, const double *B,
            int ldB, const double *beta, double *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasZgemm(transA, transB, m, n, k,
                     *reinterpret_cast<const cuDoubleComplex *>(alpha),
                     reinterpret_cast<const cuDoubleComplex *>(A), ldA,
                     reinterpret_cast<const cuDoubleComplex *>(B), ldB,
                     *reinterpret_cast<const cuDoubleComplex *>(beta),
                     reinterpret_cast<cuDoubleComplex *>(C), ldC);
}

void
cblas_chemm(char order, char side, char upLo, int m, int n,
            const float *alpha, const float *A, int ldA, const float *B,
            int ldB, const float *beta, float *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasChemm(side, upLo, m, n,
                     *reinterpret_cast<const cuComplex *>(alpha),
                     reinterpret_cast<const cuComplex *>(A), ldA,
                     reinterpret_cast<const cuComplex *>(B), ldB,
                     *reinterpret_cast<const cuComplex *>(beta),
                     reinterpret_cast<cuComplex *>(C), ldC);
}

void
cblas_zhemm(char order, char side, char upLo, int m, int n,
            const double *alpha, const double *A, int ldA, const double *B,
            int ldB, const double *beta, double *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasZhemm(side, upLo, m, n,
                     *reinterpret_cast<const cuDoubleComplex *>(alpha),
                     reinterpret_cast<const cuDoubleComplex *>(A), ldA,
                     reinterpret_cast<const cuDoubleComplex *>(B), ldB,
                     *reinterpret_cast<const cuDoubleComplex *>(beta),
                     reinterpret_cast<cuDoubleComplex *>(C), ldC);
}

void
cblas_cherk(char order, char upLo, char trans, int n, int k, float alpha,
            const float *A, int ldA, float beta, float *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasCherk(upLo, trans, n, k, alpha,
                     reinterpret_cast<const cuComplex *>(A), ldA, beta,
                     reinterpret_cast<cuComplex *>(C), ldC);
}

void
cblas_zherk(char order, char upLo, char trans, int n, int k, double alpha,
            const double *A, int ldA, double beta, double *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasZherk(upLo, trans, n, k, alpha,
                     reinterpret_cast<const cuDoubleComplex *>(A), ldA, beta,
                     reinterpret_cast<cuDoubleComplex *>(C), ldC);
}

void
cblas_cher2k(char order, char upLo, char trans, int n, int k,
             const float *alpha, const float *A, int ldA, const float *B,
             int ldB, float beta, float *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasCher2k(upLo, trans, n, k,
                      *reinterpret_cast<const cuComplex *>(alpha),
                      reinterpret_cast<const cuComplex *>(A), ldA,
                      reinterpret_cast<const cuComplex *>(B), ldB, beta,
                      reinterpret_cast<cuComplex *>(C), ldC);
}

void
cblas_zher2k(char order, char upLo, char trans, int n, int k,
             const double *alpha, const double *A, int ldA,
             const double *B, int ldB, double beta, double *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasZher2k(upLo, trans, n, k,
                      *reinterpret_cast<const cuDoubleComplex *>(alpha),
                      reinterpret_cast<const cuDoubleComplex *>(A), ldA,
                      reinterpret_cast<const cuDoubleComplex *>(B), ldB, beta,
                      reinterpret_cast<cuDoubleComplex *>(C), ldC);
}

void
cblas_ssymm(char order, char side, char upLo, int m, int n, float alpha,
            const float *A, int ldA, const float *B, int ldB, float beta,
            float *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasSsymm(side, upLo, m, n, alpha, A, ldA, B, ldB, beta, C, ldC);
}

void
cblas_dsymm(char order, char side, char upLo, int m, int n, double alpha,
            const double *A, int ldA, const double *B, int ldB,
            double beta, double *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasDsymm(side, upLo, m, n, alpha, A, ldA, B, ldB, beta, C, ldC);
}

void
cblas_csymm(char order, char side, char upLo, int m, int n,
            const float *alpha, const float *A, int ldA, const float *B,
            int ldB, const float *beta, float *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasCsymm(side, upLo, m, n,
                     *reinterpret_cast<const cuComplex *>(alpha),
                     reinterpret_cast<const cuComplex *>(A), ldA,
                     reinterpret_cast<const cuComplex *>(B), ldB,
                     *reinterpret_cast<const cuComplex *>(beta),
                     reinterpret_cast<cuComplex *>(C), ldC);
}

void
cblas_zsymm(char order, char side, char upLo, int m, int n,
            const double *alpha, const double *A, int ldA, const double *B,
            int ldB, const double *beta, double *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasZsymm(side, upLo, m, n,
                     *reinterpret_cast<const cuDoubleComplex *>(alpha),
                     reinterpret_cast<const cuDoubleComplex *>(A), ldA,
                     reinterpret_cast<const cuDoubleComplex *>(B), ldB,
                     *reinterpret_cast<const cuDoubleComplex *>(beta),
                     reinterpret_cast<cuDoubleComplex *>(C), ldC);
}

void
cblas_ssyrk(char order, char upLo, char trans, int n, int k, float alpha,
            const float *A, int ldA, float beta, float *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasSsyrk(upLo, trans, n, k, alpha, A, ldA, beta, C, ldC);
}

void
cblas_dsyrk(char order, char upLo, char trans, int n, int k, double alpha,
            const double *A, int ldA, double beta, double *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasDsyrk(upLo, trans, n, k, alpha, A, ldA, beta, C, ldC);
}

void
cblas_csyrk(char order, char upLo, char trans, int n, int k,
            const float *alpha, const float *A, int ldA, const float *beta,
            float *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasCsyrk(upLo, trans, n, k,
                     *reinterpret_cast<const cuComplex *>(alpha),
                     reinterpret_cast<const cuComplex *>(A), ldA,
                     *reinterpret_cast<const cuComplex *>(beta),
                     reinterpret_cast<cuComplex *>(C), ldC);
}

void
cblas_zsyrk(char order, char upLo, char trans, int n, int k,
            const double *alpha, const double *A, int ldA,
            const double *beta, double *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasZsyrk(upLo, trans, n, k,
                     *reinterpret_cast<const cuDoubleComplex *>(alpha),
                     reinterpret_cast<const cuDoubleComplex *>(A), ldA,
                     *reinterpret_cast<const cuDoubleComplex *>(beta),
                     reinterpret_cast<cuDoubleComplex *>(C), ldC);
}

void
cblas_ssyr2k(char order, char upLo, char trans, int n, int k, float alpha,
             const float *A, int ldA, const float *B, int ldB, float beta,
             float *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasSsyr2k(upLo, trans, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
}

void
cblas_dsyr2k(char order, char upLo, char trans, int n, int k, double alpha,
             const double *A, int ldA, const double *B, int ldB,
             double beta, double *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasDsyr2k(upLo, trans, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
}

void
cblas_csyr2k(char order, char upLo, char trans, int n, int k,
             const float *alpha, const float *A, int ldA, const float *B,
             int ldB, const float *beta, float *C, int ldC) {
  (void)order;
  assert(order == 102);
  return cublasCsyr2k(upLo, trans, n, k,
                      *reinterpret_cast<const cuComplex *>(alpha),
                      reinterpret_cast<const cuComplex *>(A), ldA,
                      reinterpret_cast<const cuComplex *>(B), ldB,
                      *reinterpret_cast<const cuComplex *>(beta),
                      reinterpret_cast<cuComplex *>(C), ldC);
}

void
cblas_zsyr2k(char order, char upLo, char trans, int n, int k,
             const double *alpha, const double *A, int ldA,
             const double *B, int ldB, const double *beta, double *C,
             int ldC) {
  (void)order;
  assert(order == 102);
  return cublasZsyr2k(upLo, trans, n, k,
                      *reinterpret_cast<const cuDoubleComplex *>(alpha),
                      reinterpret_cast<const cuDoubleComplex *>(A), ldA,
                      reinterpret_cast<const cuDoubleComplex *>(B), ldB,
                      *reinterpret_cast<const cuDoubleComplex *>(beta),
                      reinterpret_cast<cuDoubleComplex *>(C), ldC);
}

void
cblas_strmm(char order, char side, char upLo, char transA, char diag,
            int m, int n, float alpha, const float *A, int ldA, float *B,
            int ldB) {
  (void)order;
  assert(order == 102);
  return cublasStrmm(side, upLo, transA, diag, m, n, alpha, A, ldA, B, ldB);
}

void
cblas_dtrmm(char order, char side, char upLo, char transA, char diag,
            int m, int n, double alpha, const double *A, int ldA,
            double *B, int ldB) {
  (void)order;
  assert(order == 102);
  return cublasDtrmm(side, upLo, transA, diag, m, n, alpha, A, ldA, B, ldB);
}

void
cblas_ctrmm(char order, char side, char upLo, char transA, char diag,
            int m, int n, const float *alpha, const float *A, int ldA,
            float *B, int ldB) {
  (void)order;
  assert(order == 102);
  return cublasCtrmm(side, upLo, transA, diag, m, n,
                     *reinterpret_cast<const cuComplex *>(alpha),
                     reinterpret_cast<const cuComplex *>(A), ldA,
                     reinterpret_cast<cuComplex *>(B), ldB);
}

void
cblas_ztrmm(char order, char side, char upLo, char transA, char diag,
            int m, int n, const double *alpha, const double *A, int ldA,
            double *B, int ldB) {
  (void)order;
  assert(order == 102);
  return cublasZtrmm(side, upLo, transA, diag, m, n,
                     *reinterpret_cast<const cuDoubleComplex *>(alpha),
                     reinterpret_cast<const cuDoubleComplex *>(A), ldA,
                     reinterpret_cast<cuDoubleComplex *>(B), ldB);
}

void
cblas_strsm(char order, char side, char upLo, char transA, char diag,
            int m, int n, float alpha, const float *A, int ldA, float *B,
            int ldB) {
  (void)order;
  assert(order == 102);
  return cublasStrsm(side, upLo, transA, diag, m, n, alpha, A, ldA, B, ldB);
}

void
cblas_dtrsm(char order, char side, char upLo, char transA, char diag,
            int m, int n, double alpha, const double *A, int ldA,
            double *B, int ldB) {
  (void)order;
  assert(order == 102);
  return cublasDtrsm(side, upLo, transA, diag, m, n, alpha, A, ldA, B, ldB);
}

void
cblas_ctrsm(char order, char side, char upLo, char transA, char diag,
            int m, int n, const float *alpha, const float *A, int ldA,
            float *B, int ldB) {
  (void)order;
  assert(order == 102);
  return cublasCtrsm(side, upLo, transA, diag, m, n,
                     *reinterpret_cast<const cuComplex *>(alpha),
                     reinterpret_cast<const cuComplex *>(A), ldA,
                     reinterpret_cast<cuComplex *>(B), ldB);
}

void
cblas_ztrsm(char order, char side, char upLo, char transA, char diag,
            int m, int n, const double *alpha, const double *A, int ldA,
            double *B, int ldB) {
  (void)order;
  assert(order == 102);
  return cublasZtrsm(side, upLo, transA, diag, m, n,
                     *reinterpret_cast<const cuDoubleComplex *>(alpha),
                     reinterpret_cast<const cuDoubleComplex *>(A), ldA,
                     reinterpret_cast<cuDoubleComplex *>(B), ldB);
}

#endif
