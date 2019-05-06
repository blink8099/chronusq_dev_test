/* 
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *  
 *  Copyright (C) 2014-2019 Li Research Group (University of Washington)
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
#ifndef __INCLUDED_CQLINALG_BLAS3_HPP__
#define __INCLUDED_CQLINALG_BLAS3_HPP__

#include <cqlinalg/cqlinalg_config.hpp>

namespace ChronusQ {

  template <typename _F1, typename _F2, typename _FScale>
  void Gemm(char TRANSA, char TRANSB, int M, int N, int K, _FScale ALPHA,
    _F1 *A, int LDA, _F2 *B, int LDB, _FScale BETA, _F2 *C, int LDC);

#ifdef CQ_ENABLE_MPI

  template <typename _F>
  void Gemm(char TRANSA, char TRANSB, CB_INT M, CB_INT N, CB_INT K, _F ALPHA,
    _F *A, CB_INT IA, CB_INT JA, CXXBLACS::ScaLAPACK_Desc_t DESCA, 
    _F *B, CB_INT IB, CB_INT JB, CXXBLACS::ScaLAPACK_Desc_t DESCB, 
    _F BETA, _F *C, CB_INT IC, CB_INT JC, CXXBLACS::ScaLAPACK_Desc_t DESCC) {

    CXXBLACS::PGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,IA,JA,DESCA,B,IB,JB,DESCB,
      BETA,C,IC,JC,DESCC);

  }

#endif


  template <typename _F, typename _FScale>
  void Trmm(char SIDE, char UPLO, char TRANSA, char DIAG, int M, int N, 
    _FScale ALPHA, _F *A, int LDA, _F *B, int LDB);




  /**
   *  \brief Returns constant time a vector plus a vector
   *
   *  Wraps BLAS functions. See
   *    http://http://www.netlib.org/lapack/lapack-3.1.1/html/dsyr2k.html
   *  parameter documentation.
   *    
   */
   void DSYR2K(char UPLO,char TRANS,int N,int K,double alpha,double *A,
        int LDA,double *B,int LDB,double beta, double *C,int LDC);

}; // namespace ChronusQ


#endif
