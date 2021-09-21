/* 
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *  
 *  Copyright (C) 2014-2020 Li Research Group (University of Washington)
 *  
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *  
 *  Contact the Developers:
 *    E-Mail: xsli@uw.edu
 *  
 */
#pragma once

#include <chronusq_sys.hpp>

#ifdef _CQ_HAS_BTAS
  #include <btas/btas.h>
#endif

#ifdef CQ_HAS_TA
  #include <tiledarray.h>
#endif


// Choose linear algebra headers
#ifdef _CQ_MKL
  #define MKL_Complex16 dcomplex // Redefine MKL complex type
  #define MKL_Complex8  std::complex<float> // Redefine MKL complex type 
  #define lapack_complex_float MKL_Complex8
  #define lapack_complex_double MKL_Complex16
  #define LAPACK_COMPLEX_CPP

  #ifndef CQ_HAS_TA
    #include <mkl.h> // MKL
  #endif

  #ifdef CQ_ENABLE_MPI
    #include <mkl_blacs.h>  
    #include <mkl_scalapack.h>  
    #include <mkl_pblas.h>

    #define CXXBLACS_BLACS_Complex16 double
    #define CXXBLACS_BLACS_Complex8  float
    
    #define CXXBLACS_HAS_BLACS
    #define CXXBLACS_HAS_PBLAS
    #define CXXBLACS_HAS_SCALAPACK
  #endif
#else

  #ifdef CQ_ENABLE_MPI
//  #error CXXBLAS + nonMKL Not Tested!
  #endif

  #define CXXBLACS_HAS_BLAS

  #ifndef CQ_HAS_TA
    #define CXXBLACS_BLAS_Complex16 std::complex<double>
    #define CXXBLACS_BLAS_Complex8  std::complex<float>
    //#define CXXBLACS_LAPACK_Complex16 double
    //#define CXXBLACS_LAPACK_Complex8  float
  #endif

  // Redefine OpenBLAS complex type
  #ifndef _CQ_HAS_BTAS
    #define LAPACK_COMPLEX_CPP
    #define lapack_complex_float std::complex<float> 
    #define lapack_complex_double dcomplex 
  #endif

  #ifndef CQ_HAS_TA
    #include <blas/fortran.h>
    #include <lapack/fortran.h>
  #endif

  extern "C" {
    int openblas_get_num_threads();
    void openblas_set_num_threads(int*);
  }


#ifdef CQ_HAS_TA
extern "C" {
  #define TA_Complex16 dcomplex // Redefine MKL complex type
  #define TA_Complex8  std::complex<float> // Redefine MKL complex type
  #define TA_INT int 
typedef TA_INT (*MKL_D_SELECT_FUNCTION_3) ( const double*, const double*, const double* );
typedef TA_INT (*MKL_Z_SELECT_FUNCTION_2) ( const TA_Complex16*, const TA_Complex16* );
void    dswap_(const TA_INT *n, double *x, const TA_INT *incx, double *y, const TA_INT *incy);
void    zswap_(const TA_INT *n, double *x, const TA_INT *incx, double *y, const TA_INT *incy);

double  dnrm2_(const TA_INT *n, const double *x, const TA_INT *incx);

double  dznrm2_(const TA_INT *n, const double *x, const TA_INT *incx);

void strmm_(const char *side, const char *uplo, const char *transa, const char *diag,
           const TA_INT *m, const TA_INT *n, const float *alpha, const float *a, const TA_INT *lda,
           float *b, const TA_INT *ldb);
void ctrmm_(const char *side, const char *uplo, const char *transa, const char *diag,
           const TA_INT *m, const TA_INT *n, const TA_Complex8 *alpha,
           const TA_Complex8 *a, const TA_INT *lda, TA_Complex8 *b, const TA_INT *ldb);
void dtrmm_(const char *side, const char *uplo, const char *transa, const char *diag,
           const TA_INT *m, const TA_INT *n, const double *alpha, const double *a, const TA_INT *lda,
           double *b, const TA_INT *ldb);
void ztrmm_(const char *side, const char *uplo, const char *transa, const char *diag,
           const TA_INT *m, const TA_INT *n, const TA_Complex16 *alpha,
           const TA_Complex16 *a, const TA_INT *lda, TA_Complex16 *b, const TA_INT *ldb);
void zgeev_( const char* jobvl, const char* jobvr, const TA_INT* n,
             TA_Complex16* a, const TA_INT* lda, TA_Complex16* w,
             TA_Complex16* vl, const TA_INT* ldvl, TA_Complex16* vr,
             const TA_INT* ldvr, TA_Complex16* work, const TA_INT* lwork,
             double* rwork, TA_INT* info );
void dgeev_( const char* jobvl, const char* jobvr, const TA_INT* n, double* a,
             const TA_INT* lda, double* wr, double* wi, double* vl,
             const TA_INT* ldvl, double* vr, const TA_INT* ldvr,
             double* work, const TA_INT* lwork, TA_INT* info );
void dgges_( const char* jobvsl, const char* jobvsr, const char* sort,
             MKL_D_SELECT_FUNCTION_3 selctg, const TA_INT* n, double* a,
             const TA_INT* lda, double* b, const TA_INT* ldb, TA_INT* sdim,
             double* alphar, double* alphai, double* beta, double* vsl,
             const TA_INT* ldvsl, double* vsr, const TA_INT* ldvsr,
             double* work, const TA_INT* lwork, TA_INT* bwork,
             TA_INT* info );
void zgges_( const char* jobvsl, const char* jobvsr, const char* sort,
             MKL_Z_SELECT_FUNCTION_2 selctg, const TA_INT* n,
             TA_Complex16* a, const TA_INT* lda, TA_Complex16* b,
             const TA_INT* ldb, TA_INT* sdim, TA_Complex16* alpha,
             TA_Complex16* beta, TA_Complex16* vsl, const TA_INT* ldvsl,
             TA_Complex16* vsr, const TA_INT* ldvsr, TA_Complex16* work,
             const TA_INT* lwork, double* rwork, TA_INT* bwork,
             TA_INT* info );
void dtgsen_( const TA_INT* ijob, const TA_INT* wantq, const TA_INT* wantz,
              const TA_INT* select, const TA_INT* n, double* a,
              const TA_INT* lda, double* b, const TA_INT* ldb,
              double* alphar, double* alphai, double* beta, double* q,
              const TA_INT* ldq, double* z, const TA_INT* ldz, TA_INT* m,
              double* pl, double* pr, double* dif, double* work,
              const TA_INT* lwork, TA_INT* iwork, const TA_INT* liwork,
              TA_INT* info );
void ztgsen_( const TA_INT* ijob, const TA_INT* wantq, const TA_INT* wantz,
              const TA_INT* select, const TA_INT* n, TA_Complex16* a,
              const TA_INT* lda, TA_Complex16* b, const TA_INT* ldb,
              TA_Complex16* alpha, TA_Complex16* beta, TA_Complex16* q,
              const TA_INT* ldq, TA_Complex16* z, const TA_INT* ldz,
              TA_INT* m, double* pl, double* pr, double* dif,
              TA_Complex16* work, const TA_INT* lwork, TA_INT* iwork,
              const TA_INT* liwork, TA_INT* info );
void ztgexc_( const TA_INT* wantq, const TA_INT* wantz, const TA_INT* n,
              TA_Complex16* a, const TA_INT* lda, TA_Complex16* b,
              const TA_INT* ldb, TA_Complex16* q, const TA_INT* ldq,
              TA_Complex16* z, const TA_INT* ldz, const TA_INT* ifst,
              TA_INT* ilst, TA_INT* info );
void dtgexc_( const TA_INT* wantq, const TA_INT* wantz, const TA_INT* n,
              double* a, const TA_INT* lda, double* b, const TA_INT* ldb,
              double* q, const TA_INT* ldq, double* z, const TA_INT* ldz,
              TA_INT* ifst, TA_INT* ilst, double* work, const TA_INT* lwork,
              TA_INT* info );
void zungqr_( const TA_INT* m, const TA_INT* n, const TA_INT* k,
              TA_Complex16* a, const TA_INT* lda, const TA_Complex16* tau,
              TA_Complex16* work, const TA_INT* lwork, TA_INT* info );
void zgeqrf_( const TA_INT* m, const TA_INT* n, TA_Complex16* a,
              const TA_INT* lda, TA_Complex16* tau, TA_Complex16* work,
              const TA_INT* lwork, TA_INT* info );
void dgeqrf_( const TA_INT* m, const TA_INT* n, double* a,
              const TA_INT* lda, double* tau, double* work,
              const TA_INT* lwork, TA_INT* info );
void dorgqr_( const TA_INT* m, const TA_INT* n, const TA_INT* k, double* a,
              const TA_INT* lda, const double* tau, double* work,
              const TA_INT* lwork, TA_INT* info );
void ztrtri_( const char* uplo, const char* diag, const TA_INT* n,
              TA_Complex16* a, const TA_INT* lda, TA_INT* info );
void dtrtri_( const char* uplo, const char* diag, const TA_INT* n, double* a,
              const TA_INT* lda, TA_INT* info );
void zhetrf_( const char* uplo, const TA_INT* n, TA_Complex16* a,
              const TA_INT* lda, TA_INT* ipiv, TA_Complex16* work,
              const TA_INT* lwork, TA_INT* info );
void dsytrf_( const char* uplo, const TA_INT* n, double* a,
              const TA_INT* lda, TA_INT* ipiv, double* work,
              const TA_INT* lwork, TA_INT* info );
void zgetri_( const TA_INT* n, TA_Complex16* a, const TA_INT* lda,
              const TA_INT* ipiv, TA_Complex16* work, const TA_INT* lwork,
              TA_INT* info );
void dgetri_( const TA_INT* n, double* a, const TA_INT* lda,
              const TA_INT* ipiv, double* work, const TA_INT* lwork,
              TA_INT* info );
void zpotri_( const char* uplo, const TA_INT* n, TA_Complex16* a,
              const TA_INT* lda, TA_INT* info );
void dpotri_( const char* uplo, const TA_INT* n, double* a,
              const TA_INT* lda, TA_INT* info );
void dsyev_( const char* jobz, const char* uplo, const TA_INT* n, double* a,
             const TA_INT* lda, double* w, double* work, const TA_INT* lwork,
             TA_INT* info );
void dggev_( const char* jobvl, const char* jobvr, const TA_INT* n, double* a,
             const TA_INT* lda, double* b, const TA_INT* ldb, double* alphar,
             double* alphai, double* beta, double* vl, const TA_INT* ldvl,
             double* vr, const TA_INT* ldvr, double* work,
             const TA_INT* lwork, TA_INT* info );
void zggev_( const char* jobvl, const char* jobvr, const TA_INT* n,
             TA_Complex16* a, const TA_INT* lda, TA_Complex16* b,
             const TA_INT* ldb, TA_Complex16* alpha, TA_Complex16* beta,
             TA_Complex16* vl, const TA_INT* ldvl, TA_Complex16* vr,
             const TA_INT* ldvr, TA_Complex16* work, const TA_INT* lwork,
             double* rwork, TA_INT* info );
void zheev_( const char* jobz, const char* uplo, const TA_INT* n,
             TA_Complex16* a, const TA_INT* lda, double* w,
             TA_Complex16* work, const TA_INT* lwork, double* rwork,
             TA_INT* info );
void dpotrf_( const char* uplo, const TA_INT* n, double* a,
              const TA_INT* lda, TA_INT* info );
void zpotrf_( const char* uplo, const TA_INT* n, TA_Complex16* a,
              const TA_INT* lda, TA_INT* info );
void zgetrf_( const TA_INT* m, const TA_INT* n, TA_Complex16* a,
              const TA_INT* lda, TA_INT* ipiv, TA_INT* info );
void dgetrf_( const TA_INT* m, const TA_INT* n, double* a,
              const TA_INT* lda, TA_INT* ipiv, TA_INT* info );
void dgesv_( const TA_INT* n, const TA_INT* nrhs, double* a,
             const TA_INT* lda, TA_INT* ipiv, double* b, const TA_INT* ldb,
             TA_INT* info );
void zgesv_( const TA_INT* n, const TA_INT* nrhs, TA_Complex16* a,
             const TA_INT* lda, TA_INT* ipiv, TA_Complex16* b,
             const TA_INT* ldb, TA_INT* info );
void dtrsm_(const char *side, const char *uplo, const char *transa, const char *diag,
           const TA_INT *m, const TA_INT *n, const double *alpha, const double *a, const TA_INT *lda,
           double *b, const TA_INT *ldb);
void ztrsm_(const char *side, const char *uplo, const char *transa, const char *diag,
            const TA_INT *m, const TA_INT *n, const double *alpha,
            const double *a, const TA_INT *lda, double *b, const TA_INT *ldb);
void zgesvd_( const char* jobu, const char* jobvt, const TA_INT* m,
              const TA_INT* n, TA_Complex16* a, const TA_INT* lda,
              double* s, TA_Complex16* u, const TA_INT* ldu,
              TA_Complex16* vt, const TA_INT* ldvt, TA_Complex16* work,
              const TA_INT* lwork, double* rwork, TA_INT* info );
void dgesvd_( const char* jobu, const char* jobvt, const TA_INT* m,
              const TA_INT* n, double* a, const TA_INT* lda, double* s,
              double* u, const TA_INT* ldu, double* vt, const TA_INT* ldvt,
              double* work, const TA_INT* lwork, TA_INT* info );
double LAPACKE_dlange( int matrix_layout, char norm, int m,
                           int n, const double* a, int lda );
double LAPACKE_zlange( int matrix_layout, char norm, int m,
                           int n, const lapack_complex_double* a,
                           int lda );
dcomplex    zdotc_(int*, const dcomplex  *, int *, const dcomplex  *, int *);
//dcomplex    zdotu_(int*, dcomplex  *, int *, dcomplex  *, int *);
void dsyr2k_(const char *uplo, const char *trans, const TA_INT *n, const TA_INT *k,
            const double *alpha, const double *a, const TA_INT *lda, const double *b, const TA_INT *ldb,
            const double *beta, double *c, const TA_INT *ldc);
void zsyr2k_(const char *uplo, const char *trans, const TA_INT *n, const TA_INT *k,
            const TA_Complex16 *alpha, const TA_Complex16 *a, const TA_INT *lda, const TA_Complex16 *b,
            const TA_INT *ldb,const TA_Complex16 *beta, TA_Complex16 *c, const TA_INT *ldc);
}
#endif


#endif

#ifndef CQ_HAS_TA
  #include <lapack.hh>
  #include <blas.hh>
#endif

#ifdef CQ_ENABLE_MPI
  #ifndef CQ_HAS_TA
    #define CXXBLACS_HAS_LAPACK
  #endif
  #include <cxxblacs.hpp>
#else
  #define CB_INT int32_t
#endif


#include <memmanager.hpp>
#include <Eigen/Core>

extern "C" void openblas_get_num_threads_(int*);
extern "C" void openblas_set_num_threads_(int*);

