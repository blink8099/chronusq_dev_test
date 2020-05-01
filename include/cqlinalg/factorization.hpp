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

#include <cqlinalg/cqlinalg_config.hpp>

namespace ChronusQ {

  /**
   *  \brief Computes the Cholesky factorization of an hermetian 
   *  positive definate matrix A. Smart wrapper around DPOTRF or 
   *  ZPOTRF depending on context.
   *
   *  See http://www.netlib.org/lapack/lapack-3.1.1/html/dpotrf.f.html or
   *      http://www.netlib.org/lapack/lapack-3.1.1/html/zpotrf.f.html for
   *  parameter documentation.
   */  
  template <typename _F>
  int Cholesky(char UPLO, int N, _F *A, int LDA);

  /**
   *  \brief Computes the inverse of an hermetian positive definate matrix A
   *  given its Cholesky decomposition. Smart wrapper around DPOTRI or 
   *  ZPOTRI depending on context.
   *
   *  Assumes A is the Cholesky factor upon entry
   *
   *  See http://www.netlib.org/lapack/lapack-3.1.1/html/dpotrf.f.html or
   *      http://www.netlib.org/lapack/lapack-3.1.1/html/zpotrf.f.html for
   *  parameter documentation.
   */  
  template <typename _F>
  int CholeskyInv(char UPLO, int N, _F *A, int LDA);


  /**
   *  \brief Computes the LU factorization of a matrix A. Smart wrapper
   *  around DGETRF or ZGETRF depending on context.
   *
   *  See http://www.netlib.org/lapack/lapack-3.1.1/html/dgetrf.f.html or
   *      http://www.netlib.org/lapack/lapack-3.1.1/html/zgetrf.f.html for
   *  parameter documentation.
   */ 
  template <typename _F>
  int LU(int M, int N, _F *A, int LDA, int *IPIV);


  /**
   *  \brief Computes the LU factorization of a matrix A. Wraps LU function
   *  with internal allocation of IPIV through a CQMemManager.
   */ 
  template <typename _F>
  int LU(int M, int N, _F *A, int LDA, CQMemManager &mem) {
    int *IPIV = mem.template malloc<int>(std::min(M,N));
    int INFO = LU(M,N,A,LDA,IPIV);
    mem.free(IPIV);
    return INFO;
  };

  /**
   *  \brief Computes the inverse of a non-singular matrix A. Initially
   *  computes the LU factorization then wraps DGETRI / ZGETRI for the
   *  matrix inversion depending on context.
   *
   *  See LU for docs on LU factorization. Allocates memory internally
   *  through CQMemManager
   *
   *  See http://www.netlib.org/lapack/lapack-3.1.1/html/dgetri.f.html or
   *      http://www.netlib.org/lapack/lapack-3.1.1/html/zgetri.f.html for
   *  parameter documentation.
   */ 
  template <typename _F>
  int LUInv(int N, _F *A, int LDA, CQMemManager &mem);


  /**
   *  \brief Computes the Bunch-Kaufman factorization of an hermetian 
   *  matrix A. Smart wrapper around DSYTRF or ZHETRF depending on context.
   *
   *  See http://www.netlib.org/lapack/lapack-3.1.1/html/dsytrf.f.html or
   *      http://www.netlib.org/lapack/lapack-3.1.1/html/zhetrf.f.html for
   *  parameter documentation.
   */  
  template <typename _F>
  int BunchKaufman(char UPLO, int N, _F *A, int LDA, int *IPIV, 
    CQMemManager &mem);


  /**
   *  \brief Computes the Bunch-Kaufman factorization of an hermetian 
   *  matrix A. Wraps BunchKaufman with internal allocation of IPIV
   */  
  template <typename _F>
  int BunchKaufman(char UPLO, int N, _F *A, int LDA, CQMemManager &mem) {

    int * IPIV = mem.malloc<int>(std::min(1,N));
    int INFO = BunchKaufman(UPLO,N,A,LDA,IPIV,mem);
    mem.free(IPIV);
    return INFO;

  }


  /**
   *  \brief Computes the inverse of a triangular matrix A. Smart wrapper 
   *  around DTRTRI or ZTRTRI depending on context.
   *
   *  See http://www.netlib.org/lapack/lapack-3.1.1/html/dtrtri.f.html or
   *      http://www.netlib.org/lapack/lapack-3.1.1/html/ztrtri.f.html for
   *  parameter documentation.
   */  
  template <typename _F>
  int TriInv(char UPLO, char DIAG, int N, _F *A, int LDA);







  /**
   *  \brief Computes the QR factorization of a general matrix A:
   *  returns Q in place of A and returns an upper triangular R.
   *  Smart wrapper around DGEQRF / DORGQR or ZGEQRF / ZUNGQR
   *  depending on the context
   */
  template <typename _F>
  int QR(int M, int N, _F *A, int LDA, _F *R, int LDR, CQMemManager &mem);


  /**
   *  \brief Computes the QR factorization of a general matrix A:
   *  returns Q in place of A and discards R
   *  Smart wrapper around DGEQRF / DORGQR or ZGEQRF / ZUNGQR
   *  depending on the context
   */
  template <typename _F>
  inline int QR(int M, int N, _F *A, int LDA, CQMemManager &mem) {

    int LDR = std::min(M,N);
    _F *R = mem.template malloc<_F>(LDR*LDR);

    int INFO = QR(M,N,A,LDA,R,LDR,mem);

    mem.free(R);

    return INFO;

  }





  /**
   *  \brief Computes the generalized Shur (QZ) factorization of general
   *  matricies (A,B). Smart wrapper around DGGES or ZGGES depending 
   *  on the context
   */
  template <typename _F>
  int QZ(char JOBVSL, char JOBVSR, int N, _F *A, int LDA, _F *B, int LDB,
    dcomplex *ALPHA, _F *BETA, _F *VSL, int LDVSL, _F *VSR, int LDVSR,
    CQMemManager &mem); 


  template <typename _F>
  inline int QZ(char JOBVSL, char JOBVSR, int N, _F *A, int LDA, _F *B, int LDB,
    dcomplex *W, _F *VSL, int LDVSL, _F *VSR, int LDVSR, CQMemManager &mem) {


    dcomplex *ALPHA = mem.template malloc<dcomplex>(N);
    _F       *BETA  = mem.template malloc<_F>(N);

    int INFO = QZ(JOBVSL,JOBVSR,N,A,LDA,B,LDB,ALPHA,BETA,VSL,LDVSL,VSR,LDVSR,
      mem);

    for(auto k = 0; k < N; k++) {

      if( std::abs(BETA[k]) < 1e-12 )
        std::cout << "WARNING: SMALL BETA INCURRED IN QZ" << std::endl;

      W[k] = ALPHA[k] / BETA[k];

    }

    mem.free(ALPHA,BETA);

    return INFO;

  } 

  template <typename _F>
  int OrdQZ(char JOBVSL, char JOBVSR, int N, _F *A, int LDA, _F *B, 
    int LDB, dcomplex *ALPHA, _F *BETA, double SIMGA, _F *VSL, 
    int LDVSL, _F *VSR, int LDVSR, CQMemManager &mem);

  template <typename _F>
  int OrdQZ2(char JOBVSL, char JOBVSR, int N, _F *A, int LDA, _F *B, 
    int LDB, dcomplex *ALPHA, _F *BETA, double hLim, double SIMGA, _F *VSL, 
    int LDVSL, _F *VSR, int LDVSR, CQMemManager &mem);

}; // namespace ChronusQ

