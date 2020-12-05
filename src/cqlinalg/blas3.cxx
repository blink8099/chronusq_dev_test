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
int LDCcense for more details.
 *  
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *  
 *  Contact the Developers:
 *    E-Mail: xsli@uw.edu
 *  
 */
#include <cqlinalg/blas3.hpp>

namespace ChronusQ {

  template<>
  void Gemm(char TRANSA, char TRANSB, int M, int N, int K, double ALPHA,
    const double *A, int LDA, const double *B, int LDB, double BETA, double *C, int LDC){
#ifdef _CQ_MKL
    dgemm
#else
    dgemm_
#endif
    (&TRANSA,&TRANSB,&M,&N,&K,&ALPHA,const_cast<double*>(A),&LDA,
     const_cast<double*>(B),&LDB,&BETA,C,&LDC);

  }; // GEMM (real,real,real)


  template<>
  void Gemm(char TRANSA, char TRANSB, int M, int N, int K, dcomplex ALPHA,
    const dcomplex *A, int LDA, const dcomplex *B, int LDB, dcomplex BETA, dcomplex *C,
    int LDC){
#ifdef _CQ_MKL
    zgemm(&TRANSA,&TRANSB,&M,&N,&K,&ALPHA,A,&LDA,B,&LDB,&BETA,C,&LDC);
#else
#ifdef CQ_HAS_TA
    zgemm_(&TRANSA,&TRANSB,&M,&N,&K,&ALPHA,A,&LDA,B,&LDB,&BETA,C,&LDC);
#else
    zgemm_(&TRANSA,&TRANSB,&M,&N,&K,reinterpret_cast<double*>(&ALPHA),
      const_cast<double*>(reinterpret_cast<const double*>(A)),&LDA,
      const_cast<double*>(reinterpret_cast<const double*>(B)),&LDB,
      reinterpret_cast<double*>(&BETA),reinterpret_cast<double*>(C),&LDC);
#endif
#endif

  }; // GEMM (complex,complex,complex)


  template<>
  void Gemm(char TRANSA, char TRANSB, int M, int N, int K, dcomplex ALPHA,
    const double *A, int LDA, const dcomplex *B, int LDB, dcomplex BETA, dcomplex *C,
    int LDC){
#ifdef _CQ_MKL
    dzgemm(&TRANSA,&TRANSB,&M,&N,&K,&ALPHA,A,&LDA,B,&LDB,&BETA,C,&LDC);
#else
    assert( (TRANSA == 'N' or TRANSA == 'T' or TRANSA == 'C')
            and (TRANSB == 'N' or TRANSB == 'T' or TRANSB == 'C') );

    int COLS_A = (TRANSA == 'N') ? K : M;
    int COLS_B = (TRANSB == 'N') ? N : K;

    Eigen::Map<
      const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>
    > AMap(A,LDA,COLS_A);

    Eigen::Map<
      const Eigen::Matrix<dcomplex,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>
    > BMap(B,LDB,COLS_B);

    Eigen::Map<
      Eigen::Matrix<dcomplex,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>
    > CMap(C,LDC,N);

    if(TRANSB == 'N') {
      if(TRANSA == 'N')
        CMap.block(0,0,M,N).noalias() = 
          ALPHA * AMap.block(0,0,M,K).cast<dcomplex>() *
          BMap.block(0,0,K,N) + BETA * CMap.block(0,0,M,N);
      else
        CMap.block(0,0,M,N).noalias() = 
          ALPHA * AMap.block(0,0,K,M).cast<dcomplex>().adjoint() *
          BMap.block(0,0,K,N) + BETA * CMap.block(0,0,M,N);
    } else if(TRANSB == 'C') {
      if(TRANSA == 'N')
        CMap.block(0,0,M,N).noalias() = 
          ALPHA * AMap.block(0,0,M,K).cast<dcomplex>() *
          BMap.block(0,0,N,K).adjoint() + BETA * CMap.block(0,0,M,N);
      else
        CMap.block(0,0,M,N).noalias() = 
          ALPHA * AMap.block(0,0,K,M).cast<dcomplex>().adjoint() *
          BMap.block(0,0,N,K).adjoint() + BETA * CMap.block(0,0,M,N);
    } else {
      if(TRANSA == 'N')
        CMap.block(0,0,M,N).noalias() =
          ALPHA * AMap.block(0,0,M,K).cast<dcomplex>() *
          BMap.block(0,0,N,K).transpose() + BETA * CMap.block(0,0,M,N);
      else
        CMap.block(0,0,M,N).noalias() =
          ALPHA * AMap.block(0,0,K,M).cast<dcomplex>().adjoint() *
          BMap.block(0,0,N,K).transpose() + BETA * CMap.block(0,0,M,N);
    }
#endif

  }; // GEMM (real,complex,complex)









  template<>
  void Trmm(char SIDE, char UPLO, char TRANSA, char DIAG, int M, int N, 
    double ALPHA, double *A, int LDA, double *B, int LDB){

#ifdef _CQ_MKL
    dtrmm
#else
    dtrmm_
#endif
    (&SIDE,&UPLO,&TRANSA,&DIAG,&M,&N,&ALPHA,A,&LDA,B,&LDB);

  }; // TRMM (real,real,real)


  template<>
  void Trmm(char SIDE, char UPLO, char TRANSA, char DIAG, int M, int N, 
    dcomplex ALPHA, dcomplex *A, int LDA, dcomplex *B, int LDB) {

#ifdef _CQ_MKL
    ztrmm(&SIDE,&UPLO,&TRANSA,&DIAG,&M,&N,&ALPHA,A,&LDA,B,&LDB);
#else
#ifdef CQ_HAS_TA
    ztrmm_(&SIDE,&UPLO,&TRANSA,&DIAG,&M,&N,&ALPHA,A,&LDA,B,&LDB);
#else
    ztrmm_(&SIDE,&UPLO,&TRANSA,&DIAG,&M,&N,reinterpret_cast<double*>(&ALPHA),
      reinterpret_cast<double*>(A),&LDA,reinterpret_cast<double*>(B),&LDB);
#endif
#endif
    

  }; // TRMM (complex,complex,complex)


  /*
   *  performs one of the symmetric rank 2k operations
   *  C := alpha*A*B' + alpha*B*A' + beta*C
   */
  void DSYR2K(char UPLO,char TRANS,int N,int K,double ALPHA,double *A,
    int LDA,double *B,int LDB,double BETA, double *C,int LDC){
#ifdef _CQ_MKL
    dsyr2k
#else
    dsyr2k_
#endif
      (&UPLO,&TRANS,&N,&K,&ALPHA,A,&LDA,B,&LDB,&BETA,C,&LDC);
  }; // DSYR2K

  /*
   *  performs one of the symmetric rank 2k operations
   *  C := alpha*A*B' + alpha*B*A' + beta*C
   */
  template <>
  void SYR2K(char UPLO,char TRANS,int N,int K,double ALPHA,double *A,
    int LDA,double *B,int LDB,double BETA, double *C,int LDC){
#ifdef _CQ_MKL
    dsyr2k
#else
    dsyr2k_
#endif
      (&UPLO,&TRANS,&N,&K,&ALPHA,A,&LDA,B,&LDB,&BETA,C,&LDC);
  }; // SYR2K

  template <>
  void SYR2K(char UPLO,char TRANS,int N,int K,dcomplex ALPHA,dcomplex *A,
    int LDA,dcomplex *B,int LDB,dcomplex BETA, dcomplex *C,int LDC){
#ifdef _CQ_MKL
    zsyr2k(&UPLO,&TRANS,&N,&K,&ALPHA,A,&LDA,B,&LDB,&BETA,C,&LDC);
#else
    zsyr2k_(&UPLO,&TRANS,&N,&K,reinterpret_cast<double*>(&ALPHA),
      reinterpret_cast<double*>(A),&LDA,reinterpret_cast<double*>(B),&LDB,
      reinterpret_cast<double*>(&BETA),reinterpret_cast<double*>(C),&LDC);
#endif
      
  }; // SYR2K



}; // namespace ChronusQ
