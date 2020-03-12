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
#include <cqlinalg/solve.hpp>

namespace ChronusQ {

  // Real wraps DGESV
  template <>
  int LinSolve(int N, int NRHS, double *A, int LDA, double *B, 
    int LDB, int *iPIV) {

    int INFO;

    dgesv_(&N,&NRHS,A,&LDA,iPIV,B,&LDB,&INFO);

    return INFO;
  }; // LinSolve (real)


  // Complex wraps ZGESV
  template <>
  int LinSolve(int N, int NRHS, dcomplex *A, int LDA, 
    dcomplex *B, int LDB, int* iPIV) {

    int INFO;

    zgesv_(&N,&NRHS,A,&LDA,iPIV,B,&LDB,&INFO);

    return INFO;
  }; // LinSolve (complex)



  // Real wraps DTRSM
  template <>
  void TriLinSolve(char SIDE, char UPLO, char TRANS, char DIAG, int M, int N, 
    double ALPHA, double *A, int LDA, double *B, int LDB){

#ifdef _CQ_MKL
    dtrsm(&SIDE,&UPLO,&TRANS,&DIAG,&M,&N,&ALPHA,A,&LDA,B,&LDB);
#else
    dtrsm_(&SIDE,&UPLO,&TRANS,&DIAG,&M,&N,&ALPHA,A,&LDA,B,&LDB);
#endif

  }; // TriLinSolve (real)


  // Complex wraps ZTRSM
  template <>
  void TriLinSolve(char SIDE, char UPLO, char TRANS, char DIAG, int M, int N, 
    dcomplex ALPHA, dcomplex *A, int LDA, dcomplex *B, int LDB){

#ifdef _CQ_MKL
    ztrsm(&SIDE,&UPLO,&TRANS,&DIAG,&M,&N,&ALPHA,A,&LDA,B,&LDB);
#else
    ztrsm_(&SIDE,&UPLO,&TRANS,&DIAG,&M,&N,reinterpret_cast<double*>(&ALPHA),
      reinterpret_cast<double*>(A),&LDA,reinterpret_cast<double*>(B),&LDB);
#endif

  }; // TriLinSolve (complex)





}; // namespace ChronusQ
