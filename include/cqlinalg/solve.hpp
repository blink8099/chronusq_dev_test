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
#ifndef __INCLUDED_CQLINALG_SOLVE_HPP__
#define __INCLUDED_CQLINALG_SOLVE_HPP__

#include <cqlinalg/cqlinalg_config.hpp>

namespace ChronusQ {

  /**
   *  \brief Solves a linear system AX = B. Smart wrapper around
   *  DGESV or ZGESV depending on context.
   *
   *  Assumes preallocated IPIV
   *
   *  See http://www.netlib.org/lapack/lapack-3.1.1/html/dgesv.f.html or
   *      http://www.netlib.org/lapack/lapack-3.1.1/html/zgesv.f.html for
   *  parameter documentation.
   */ 
  template <typename _F>
  int LinSolve(int N, int NRHS, _F *A, int LDA, _F *B, 
    int LDB, int* IPIV);



  /**
   *  \brief Solves a linear system AX = B. Smart wrapper around
   *  DGESV or ZGESV depending on context.
   *
   *  Allocates memory internally through CQMemManager.
   *
   *  See http://www.netlib.org/lapack/lapack-3.1.1/html/dgesv.f.html or
   *      http://www.netlib.org/lapack/lapack-3.1.1/html/zgesv.f.html for
   *  parameter documentation.
   */ 
  template <typename _F>
  inline int LinSolve(int N, int NRHS, _F *A, int LDA, _F *B, 
    int LDB, CQMemManager &mem) {

    int* iPIV = mem.malloc<int>(N);

    int INFO = LinSolve(N,NRHS,A,LDA,B,LDB,iPIV);

    mem.free(iPIV);

    return INFO;

  };


  /**
   *  \brief Solves a linear system AX = B. Smart wrapper around
   *  DGESV or ZGESV depending on context.
   *
   *  Allocates on the stack
   *
   *  See http://www.netlib.org/lapack/lapack-3.1.1/html/dgesv.f.html or
   *      http://www.netlib.org/lapack/lapack-3.1.1/html/zgesv.f.html for
   *  parameter documentation.
   */ 
  template <typename _F>
  inline int LinSolve(int N, int NRHS, _F *A, int LDA, _F *B, 
    int LDB) {

    std::vector<int> iPIV(N,0);
    return LinSolve(N,NRHS,A,LDA,B,LDB,&iPIV[0]);

  };


  /**
   *
   *  \brief Solves a triagular linear system OP(A)X = aB or X OP(A) = aB. 
   *  Smart wrapper around DTRSM and ZTRSM depending on context
   *   
   *  See http://www.netlib.org/lapack/lapack-3.1.1/html/dtrsm.f.html or
   *      http://www.netlib.org/lapack/lapack-3.1.1/html/ztrsm.f.html for
   *  parameter documentation.
   *
   */
  template <typename _F>
  void TriLinSolve(char SIDE, char UPLO, char TRANS, char DIAG, int M, int N, 
    _F ALPHA, _F *A, int LDA, _F *B, int LDB);




#ifdef CQ_ENABLE_MPI


  template <typename _F>
  inline CB_INT LinSolve(const CB_INT N, const CB_INT NRHS, _F *A, 
    const CB_INT IA, const CB_INT JA, const CXXBLACS::ScaLAPACK_Desc_t DESCA, 
    _F *B, const CB_INT IB, const CB_INT JB,
    const CXXBLACS::ScaLAPACK_Desc_t DESCB, CB_INT *IPIV) {

    return CXXBLACS::PGESV(N,NRHS,A,IA,JA,DESCA,IPIV,B,IB,JB,DESCB);

  }

  template <typename _F>
  inline CB_INT LinSolve(const CB_INT N, const CB_INT NRHS, _F *A, 
    const CB_INT IA, const CB_INT JA, const CXXBLACS::ScaLAPACK_Desc_t DESCA, 
    _F *B, const CB_INT IB, const CB_INT JB,
    const CXXBLACS::ScaLAPACK_Desc_t DESCB, CQMemManager &mem) {

    CB_INT* iPIV = mem.malloc<CB_INT>(DESCA[8] + DESCA[4]); // LLD + MB

    CB_INT INFO = LinSolve(N,NRHS,A,IA,JA,DESCA,B,IB,JB,DESCB,iPIV);

    mem.free(iPIV);

    return INFO;
  }

  template <typename _F>
  inline CB_INT LinSolve(const CB_INT N, const CB_INT NRHS, _F *A, 
    const CB_INT IA, const CB_INT JA, const CXXBLACS::ScaLAPACK_Desc_t DESCA, 
    _F *B, const CB_INT IB, const CB_INT JB,
    const CXXBLACS::ScaLAPACK_Desc_t DESCB) {

    std::vector<CB_INT> iPIV(DESCA[8] + DESCA[4],0); // LLD + MB
    return LinSolve(N,NRHS,A,IA,JA,DESCA,B,IB,JB,DESCB,&iPIV[0]);

  }


#endif

}; // namespace ChronusQ

#endif
