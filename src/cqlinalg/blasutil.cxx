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
#include <cqlinalg/blasutil.hpp>
#include <cqlinalg/blasext.hpp>
#include <cerr.hpp>

namespace ChronusQ {

  template <typename T> T ComplexScale();
  template <> double ComplexScale(){ return -1; }
  template <> dcomplex ComplexScale(){ return dcomplex(0,1); }

  template <typename _F1, typename _F2>
  void SpinScatter(size_t M, size_t N, const _F1 *AA, size_t LDAA,
      const _F1 *AB, size_t LDAB, const _F1 *BA, size_t LDBA,
      const _F1 *BB, size_t LDBB, _F2 *AS, size_t LDAS,
      _F2 *AZ, size_t LDAZ, _F2 *AY, size_t LDAY, _F2 *AX, size_t LDAX,
      bool zeroABBA, bool BBeqAA) {

    _F2 YFACT = ComplexScale<_F2>();

    if (BBeqAA) {
      if(AS) SetMat('N',M,N,_F2(2.),AA,LDAA,AS,LDAS);
      if(AZ) SetMat('N',M,N,_F2(0.0),AA,LDAA,AZ,LDAZ);
    } else {
      if(AS) MatAdd('N','N',M,N,_F2(1.),AA,LDAA,_F2(1.) ,BB,LDBB,AS,LDAS);
      if(AZ) MatAdd('N','N',M,N,_F2(1.),AA,LDAA,_F2(-1.),BB,LDBB,AZ,LDAZ);
    }

    if (zeroABBA) {
      if(AY) SetMat('N',M,N,_F2(0.0),AA,LDAA,AY,LDAY);
      if(AX) SetMat('N',M,N,_F2(0.0),AA,LDAA,AX,LDAX);
    } else {
      if(AY) MatAdd('N','N',M,N, YFACT ,AB,LDAB, -YFACT ,BA,LDBA,AY,LDAY);
      if(AX) MatAdd('N','N',M,N,_F2(1.),AB,LDAB,_F2(1.) ,BA,LDBA,AX,LDAX);
    }

  };

  template <typename _F1, typename _F2>
  void SpinScatter(size_t M, size_t N, const _F1 *A, size_t LDA, _F2 *AS, size_t LDAS,
      _F2 *AZ, size_t LDAZ, _F2 *AY, size_t LDAY, _F2 *AX, size_t LDAX,
      bool zeroABBA, bool BBeqAA) {

/*
    for(auto j = 0; j < N; j++)
    for(auto i = 0; i < M; i++) {
      AS[i + j*LDAS] = A[i + j*LDA] + A[(i+M) + (j+N)*LDA];
      AZ[i + j*LDAZ] = A[i + j*LDA] - A[(i+M) + (j+N)*LDA];
      AY[i + j*LDAY] = YFACT * (A[i + (j+N)*LDA] - A[(i+M) + j*LDA]);
      AX[i + j*LDAX] = A[i + (j+N)*LDA] + A[(i+M) + j*LDA];
    }
*/

    const _F1* A_AA = A;
    const _F1* A_AB = A_AA + N*LDA;
    const _F1* A_BA = A_AA + M;
    const _F1* A_BB = A_AB + M;

    SpinScatter(M,N,A_AA,LDA,A_AB,LDA,A_BA,LDA,A_BB,LDA,
                AS,LDAS,AZ,LDAZ,AY,LDAY,AX,LDAX,zeroABBA,BBeqAA);

  };

  template <typename _F1, typename _F2>
  void SpinScatter(size_t N, const _F1 *A, size_t LDA, _F2 *AS, size_t LDAS,
      _F2 *AZ, size_t LDAZ, _F2 *AY, size_t LDAY, _F2 *AX, size_t LDAX,
      bool zeroABBA, bool BBeqAA) {

   SpinScatter(N,N,A,LDA,AS,LDAS,AZ,LDAZ,AY,LDAY,AX,LDAX,zeroABBA,BBeqAA);

  };

  template <typename _F1, typename _F2>
  void SpinGather(size_t M, size_t N, _F1 *AA, size_t LDAA, _F1 *AB, size_t LDAB,
      _F1 *BA, size_t LDBA, _F1 *BB, size_t LDBB, const _F2 *AS, size_t LDAS,
      const _F2 *AZ, size_t LDAZ, const _F2 *AY, size_t LDAY, const _F2 *AX, size_t LDAX,
      bool zeroXY, bool zeroZ) {

    _F2 YFACT = 0.5*ComplexScale<_F2>();

    if (zeroZ) {
      if(AA) SetMat('N',M,N,_F2(0.5),AS,LDAS,AA,LDAA);
      if(BB) SetMat('N',M,N,_F2(0.5),AS,LDAS,BB,LDBB);
    } else {
      if(AA) MatAdd('N','N',M,N,_F2(0.5),AS,LDAS,_F2(0.5) ,AZ,LDAZ,AA,LDAA);
      if(BB) MatAdd('N','N',M,N,_F2(0.5),AS,LDAS,_F2(-0.5),AZ,LDAZ,BB,LDBB);
    }

    if (zeroXY) {
      if(BA) SetMat('N',M,N,_F2(0.0),AS,LDAS,BA,LDBA);
      if(AB) SetMat('N',M,N,_F2(0.0),AS,LDAS,AB,LDAB);
    } else {
      if(BA) MatAdd('N','N',M,N,_F2(0.5),AX,LDAX, YFACT,AY,LDAY,BA,LDBA);
      if(AB) MatAdd('N','N',M,N,_F2(0.5),AX,LDAX,-YFACT,AY,LDAY,AB,LDAB);
    }

  };

  template <typename _F1, typename _F2>
  void SpinGather(size_t M, size_t N, _F1 *A, size_t LDA, const _F2 *AS, size_t LDAS,
      const _F2 *AZ, size_t LDAZ, const _F2 *AY, size_t LDAY, const _F2 *AX, size_t LDAX,
      bool zeroXY, bool zeroZ) {

/*
    for(auto j = 0; j < N; j++)
    for(auto i = 0; i < N; i++) {
      A[i + j*LDA]         = 0.5 * (AS[i + j*LDAS] + AZ[i + j*LDAZ]);
      A[(i+N) + (j+N)*LDA] = 0.5 * (AS[i + j*LDAS] - AZ[i + j*LDAZ]);
      A[(i+N) + j*LDA]     = 0.5 * (AX[i + j*LDAS] + YFACT * AY[i + j*LDAZ]);
      A[i + (j+N)*LDA]     = 0.5 * (AX[i + j*LDAS] - YFACT * AY[i + j*LDAZ]);
    }
*/
    _F1* A_AA = A;
    _F1* A_AB = A_AA + N*LDA;
    _F1* A_BA = A_AA + M;
    _F1* A_BB = A_AB + M;

    SpinGather(M,N,A_AA,LDA,A_AB,LDA,A_BA,LDA,A_BB,LDA,
                AS,LDAS,AZ,LDAZ,AY,LDAY,AX,LDAX,zeroXY,zeroZ);

  };

  template <typename _F1, typename _F2>
  void SpinGather(size_t N, _F1 *A, size_t LDA, const _F2 *AS, size_t LDAS,
      const _F2 *AZ, size_t LDAZ, const _F2 *AY, size_t LDAY, const _F2 *AX, size_t LDAX,
      bool zeroXY, bool zeroZ) {

    SpinGather(N,N,A,LDA,AS,LDAS,AZ,LDAZ,AY,LDAY,AX,LDAX,zeroXY,zeroZ);

  };

  template
  void SpinScatter(size_t M, size_t N, const double *AA, size_t LDAA,
      const double *AB, size_t LDAB, const double *BA, size_t LDBA,
      const double *BB, size_t LDBB, double *AS, size_t LDAS,
      double *AZ, size_t LDAZ, double *AY, size_t LDAY, double *AX, size_t LDAX,
      bool zeroABBA, bool zeroBB);

  template
  void SpinScatter(size_t M, size_t N, const double *AA, size_t LDAA,
      const double *AB, size_t LDAB, const double *BA, size_t LDBA,
      const double *BB, size_t LDBB, dcomplex *AS, size_t LDAS,
      dcomplex *AZ, size_t LDAZ, dcomplex *AY, size_t LDAY, dcomplex *AX, size_t LDAX,
      bool zeroABBA, bool zeroBB);

  template
  void SpinScatter(size_t M, size_t N, const dcomplex *AA, size_t LDAA,
      const dcomplex *AB, size_t LDAB, const dcomplex *BA, size_t LDBA,
      const dcomplex *BB, size_t LDBB, dcomplex *AS, size_t LDAS,
      dcomplex *AZ, size_t LDAZ, dcomplex *AY, size_t LDAY, dcomplex *AX, size_t LDAX,
      bool zeroABBA, bool zeroBB);

  template
  void SpinScatter(size_t M, size_t N, const double *A, size_t LDA, double *AS, size_t LDAS,
      double *AZ, size_t LDAZ, double *AY, size_t LDAY, double *AX, size_t LDAX,
      bool zeroABBA, bool zeroBB);

  template
  void SpinScatter(size_t M, size_t N, const double *A, size_t LDA, dcomplex *AS, size_t LDAS,
      dcomplex *AZ, size_t LDAZ, dcomplex *AY, size_t LDAY, dcomplex *AX, size_t LDAX,
      bool zeroABBA, bool zeroBB);

  template
  void SpinScatter(size_t M, size_t N, const dcomplex *A, size_t LDA, dcomplex *AS, size_t LDAS,
      dcomplex *AZ, size_t LDAZ, dcomplex *AY, size_t LDAY, dcomplex *AX, size_t LDAX,
      bool zeroABBA, bool zeroBB);

  template
  void SpinScatter(size_t N, const double *A, size_t LDA, double *AS, size_t LDAS,
      double *AZ, size_t LDAZ, double *AY, size_t LDAY, double *AX, size_t LDAX,
      bool zeroABBA, bool zeroBB);

  template
  void SpinScatter(size_t N, const double *A, size_t LDA, dcomplex *AS, size_t LDAS,
      dcomplex *AZ, size_t LDAZ, dcomplex *AY, size_t LDAY, dcomplex *AX, size_t LDAX,
      bool zeroABBA, bool zeroBB);

  template
  void SpinScatter(size_t N, const dcomplex *A, size_t LDA, dcomplex *AS, size_t LDAS,
      dcomplex *AZ, size_t LDAZ, dcomplex *AY, size_t LDAY, dcomplex *AX, size_t LDAX,
      bool zeroABBA, bool zeroBB);

  template
  void SpinGather(size_t M, size_t N, double *AA, size_t LDAA, double *AB, size_t LDAB,
      double *BA, size_t LDBA, double *BB, size_t LDBB, const double *AS, size_t LDAS,
      const double *AZ, size_t LDAZ, const double *AY, size_t LDAY, const double *AX, size_t LDAX,
      bool zeroXY, bool zeroZ);

  template
  void SpinGather(size_t M, size_t N, dcomplex *AA, size_t LDAA, dcomplex *AB, size_t LDAB,
      dcomplex *BA, size_t LDBA, dcomplex *BB, size_t LDBB, const double *AS, size_t LDAS,
      const double *AZ, size_t LDAZ, const double *AY, size_t LDAY, const double *AX, size_t LDAX,
      bool zeroXY, bool zeroZ);

  template
  void SpinGather(size_t M, size_t N, dcomplex *AA, size_t LDAA, dcomplex *AB, size_t LDAB,
      dcomplex *BA, size_t LDBA, dcomplex *BB, size_t LDBB, const dcomplex *AS, size_t LDAS,
      const dcomplex *AZ, size_t LDAZ, const dcomplex *AY, size_t LDAY, const dcomplex *AX,
      size_t LDAX, bool zeroXY, bool zeroZ);

  template
  void SpinGather(size_t M, size_t N, double *A, size_t LDA, const double *AS, size_t LDAS,
      const double *AZ, size_t LDAZ, const double *AY, size_t LDAY, const double *AX, size_t LDAX,
      bool zeroXY, bool zeroZ);

  template
  void SpinGather(size_t M, size_t N, dcomplex *A, size_t LDA, const double *AS, size_t LDAS,
      const double *AZ, size_t LDAZ, const double *AY, size_t LDAY, const double *AX, size_t LDAX,
      bool zeroXY, bool zeroZ);

  template
  void SpinGather(size_t M, size_t N, dcomplex *A, size_t LDA, const dcomplex *AS, size_t LDAS,
      const dcomplex *AZ, size_t LDAZ, const dcomplex *AY, size_t LDAY, const dcomplex *AX,
      size_t LDAX, bool zeroXY, bool zeroZ);

  template
  void SpinGather(size_t N, double *A, size_t LDA, const double *AS, size_t LDAS,
      const double *AZ, size_t LDAZ, const double *AY, size_t LDAY, const double *AX, size_t LDAX,
      bool zeroXY, bool zeroZ);

  template
  void SpinGather(size_t N, dcomplex *A, size_t LDA, const double *AS, size_t LDAS,
      const double *AZ, size_t LDAZ, const double *AY, size_t LDAY, const double *AX, size_t LDAX,
      bool zeroXY, bool zeroZ);

  template
  void SpinGather(size_t N, dcomplex *A, size_t LDA, const dcomplex *AS, size_t LDAS,
      const dcomplex *AZ, size_t LDAZ, const dcomplex *AY, size_t LDAY, const dcomplex *AX,
      size_t LDAX, bool zeroXY, bool zeroZ);












  template <typename _F1, typename _F2, typename _FScale>
  void SetMat(char TRANS, size_t M, size_t N, _FScale ALPHA, const _F1 *A, size_t LDA,
    size_t SA, _F2 *B, size_t LDB, size_t SB) {

    assert( TRANS == 'N' or TRANS == 'R' or TRANS == 'T' or TRANS == 'C');

    using namespace Eigen;

    typedef const Matrix<_F1,Dynamic,Dynamic,ColMajor> F1Mat;
    typedef Matrix<_F2,Dynamic,Dynamic,ColMajor> F2Mat;
    typedef Stride<Dynamic,Dynamic> DynamicStride; 

    typedef Map<F1Mat,0,DynamicStride> F1Map;
    typedef Map<F2Mat,0,DynamicStride> F2Map;


    F1Map AMap(A,M,N, DynamicStride(LDA,SA));

    if      ( TRANS == 'N' ) {
      F2Map BMap(B,M,N, DynamicStride(LDB,SB));
      BMap = ALPHA * AMap;
    } else if ( TRANS == 'R' ) {
      F2Map BMap(B,M,N, DynamicStride(LDB,SB));
      BMap = ALPHA * AMap.conjugate();
    } else if ( TRANS == 'T' ) {
      F2Map BMap(B,N,M, DynamicStride(LDB,SB));
      BMap = ALPHA * AMap.transpose();
    } else if ( TRANS == 'C' ) {
      F2Map BMap(B,N,M, DynamicStride(LDB,SB));
      BMap = ALPHA * AMap.adjoint();
    }

  }

#ifdef _CQ_MKL

  template <>
  void SetMat(char TRANS, size_t M, size_t N, double ALPHA, const double *A,
    size_t LDA, size_t SA, double *B, size_t LDB, size_t SB) {

    if( SA != 1 or SB != 1)
      mkl_domatcopy2('C',TRANS,M,N,ALPHA,A,LDA,SA,B,LDB,SB);
    else
      mkl_domatcopy('C',TRANS,M,N,ALPHA,A,LDA,B,LDB);

  };

  template <>
  void SetMat(char TRANS, size_t M, size_t N, dcomplex ALPHA, const dcomplex *A,
    size_t LDA, size_t SA, dcomplex *B, size_t LDB, size_t SB) {

    if( SA != 1 or SB != 1)
      mkl_zomatcopy2('C',TRANS,M,N,ALPHA,A,LDA,SA,B,LDB,SB);
    else
      mkl_zomatcopy('C',TRANS,M,N,ALPHA,A,LDA,B,LDB);

  };

#else

  template
  void SetMat(char TRANS, size_t M, size_t N, double ALPHA, const double *A,
    size_t LDA, size_t SA, double *B, size_t LDB, size_t SB);

  template
  void SetMat(char TRANS, size_t M, size_t N, dcomplex ALPHA, const dcomplex *A,
    size_t LDA, size_t SA, dcomplex *B, size_t LDB, size_t SB);


#endif

  template
  void SetMat(char TRANS, size_t M, size_t N, double ALPHA, const double *A,
    size_t LDA, size_t SA, dcomplex *B, size_t LDB, size_t SB);

  template
  void SetMat(char TRANS, size_t M, size_t N, double ALPHA, const dcomplex *A,
    size_t LDA, size_t SA, dcomplex *B, size_t LDB, size_t SB);

  template
  void SetMat(char TRANS, size_t M, size_t N, dcomplex ALPHA, const double *A,
    size_t LDA, size_t SA, dcomplex *B, size_t LDB, size_t SB);


  /// \brief A2c = [ A  0 ]
  ///              [ 0  A ]
  template <typename _F1, typename _F2>
  void SetMatDiag(size_t M, size_t N, const _F1 *A, size_t LDA, _F2 *A2c, size_t LD2c) {
    SetMat('N',M,N,1.,A,LDA,A2c,LD2c);
    SetMat('N',M,N,1.,A,LDA,A2c + M + N*LD2c,LD2c);
    SetMat('N',M,N,0.,A,LDA,A2c + M,LD2c);
    SetMat('N',M,N,0.,A,LDA,A2c + N*LD2c,LD2c);
  }

  template void SetMatDiag(size_t, size_t, const double*, size_t, double*, size_t);
  template void SetMatDiag(size_t, size_t, const double*, size_t, dcomplex*, size_t);
  template void SetMatDiag(size_t, size_t, const dcomplex*, size_t, dcomplex*, size_t);




  template<>
  void SetMatRE(char TRANS, size_t M, size_t N, double ALPHA, const double *A,
    size_t LDA, dcomplex *B, size_t LDB) {

    SetMat(TRANS,M,N,ALPHA,A,LDA,1,reinterpret_cast<double*>(B),2*LDB,2);

  }; // SetMatRE (complex)

  template<>
  void SetMatRE(char TRANS, size_t M, size_t N, double ALPHA, const double *A,
    size_t LDA, double *B, size_t LDB) {

    SetMat(TRANS,M,N,ALPHA,A,LDA,1,B,LDB,1);
    

  }; // SetMatRE (real)


  template<>
  void SetMatIM(char TRANS, size_t M, size_t N, double ALPHA, const double *A,
    size_t LDA, dcomplex *B, size_t LDB) {

    SetMat(TRANS,M,N,ALPHA,A,LDA,1,reinterpret_cast<double*>(B)+1,2*LDB,2);

  }; // SetMatIM (complex)

  template<>
  void SetMatIM(char TRANS, size_t M, size_t N, double ALPHA, const double *A,
    size_t LDA, double *B, size_t LDB) {

    assert(false);

  }; // SetMatRM (real)

  template<>
  void GetMatRE(char TRANS, size_t M, size_t N, double ALPHA, const dcomplex *A,
    size_t LDA, double *B, size_t LDB) {

    SetMat(TRANS,M,N,ALPHA,reinterpret_cast<const double*>(A),2*LDA,2,B,LDA,1);

  }; // GetMatRE (complex)

  template<>
  void GetMatRE(char TRANS, size_t M, size_t N, double ALPHA, const double *A,
    size_t LDA, double *B, size_t LDB) {

    SetMat(TRANS,M,N,ALPHA,A,LDA,1,B,LDA,1);

  }; // GetMatRE (real)

  template<>
  void GetMatIM(char TRANS, size_t M, size_t N, double ALPHA, const dcomplex *A,
    size_t LDA, double *B, size_t LDB) {

    SetMat(TRANS,M,N,ALPHA,reinterpret_cast<const double*>(A)+1,2*LDA,2,B,LDA,1);

  }; // GetMatIM (complex)

  template<>
  void GetMatIM(char TRANS, size_t M, size_t N, double ALPHA, const double *A,
    size_t LDA, double *B, size_t LDB) {

    SetMat(TRANS,M,N,0.,A,LDA,1,B,LDA,1);

  }; // GetMatIM (real)









  // Non-contiguous sub matrix operations

  template <typename _F1, typename _F2>
  void SubMatSet(size_t M, size_t N, size_t MSub, size_t NSub, _F1 *ABig, 
    size_t LDAB, _F2 *ASmall, size_t LDAS, 
    std::vector<std::pair<size_t,size_t>> &SubMatCut) {

    
    Eigen::Map<
      Eigen::Matrix<_F1,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>
        ABigMap(ABig,LDAB,N);

    Eigen::Map<
      Eigen::Matrix<_F2,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>
        ASmallMap(ASmall,LDAS,NSub);

    size_t i(0);
    for( auto& iCut : SubMatCut ) {
      size_t deltaI = iCut.second - iCut.first;
      size_t j(0);
    for( auto& jCut : SubMatCut ) {
      size_t deltaJ = jCut.second - jCut.first;
    
      ASmallMap.block(i,j,deltaI,deltaJ).noalias() =
        ABigMap.block(iCut.first,jCut.first,deltaI,deltaJ);
    
      j += deltaJ;
    }
      i += deltaI;
    }
  };

  template <typename _F1, typename _F2>
  void SubMatGet(size_t M, size_t N, size_t MSub, size_t NSub, _F1 *ABig, 
    size_t LDAB, _F2 *ASmall, size_t LDAS, 
    std::vector<std::pair<size_t,size_t>> &SubMatCut) {

    
    Eigen::Map<
      Eigen::Matrix<_F1,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>
        ABigMap(ABig,LDAB,N);

    Eigen::Map<
      Eigen::Matrix<_F2,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>
        ASmallMap(ASmall,LDAS,NSub);

    size_t i(0);
    for( auto& iCut : SubMatCut ) {
      size_t deltaI = iCut.second - iCut.first;
      size_t j(0);
    for( auto& jCut : SubMatCut ) {
      size_t deltaJ = jCut.second - jCut.first;
    
      ABigMap.block(iCut.first,jCut.first,deltaI,deltaJ).noalias() =
        ASmallMap.block(i,j,deltaI,deltaJ);
    
      j += deltaJ;
    }
      i += deltaI;
    }
  };

  template <typename _F1, typename _F2>
  void SubMatInc(size_t M, size_t N, size_t MSub, size_t NSub, _F1 *ABig, 
    size_t LDAB, _F2 *ASmall, size_t LDAS, 
    std::vector<std::pair<size_t,size_t>> &SubMatCut) {

    
    Eigen::Map<
      Eigen::Matrix<_F1,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>
        ABigMap(ABig,LDAB,N);

    Eigen::Map<
      Eigen::Matrix<_F2,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>
        ASmallMap(ASmall,LDAS,NSub);

    size_t i(0);
    for( auto& iCut : SubMatCut ) {
      size_t deltaI = iCut.second - iCut.first;
      size_t j(0);
    for( auto& jCut : SubMatCut ) {
      size_t deltaJ = jCut.second - jCut.first;
    
      ASmallMap.block(i,j,deltaI,deltaJ).noalias() +=
        ABigMap.block(iCut.first,jCut.first,deltaI,deltaJ);
    
      j += deltaJ;
    }
      i += deltaI;
    }
  };

  template <typename _F1, typename _F2>
  void IncBySubMat(size_t M, size_t N, size_t MSub, size_t NSub, _F1 *ABig, 
    size_t LDAB, _F2 *ASmall, size_t LDAS, 
    std::vector<std::pair<size_t,size_t>> &SubMatCut) {

    
    Eigen::Map<
      Eigen::Matrix<_F1,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>
        ABigMap(ABig,LDAB,N);

    Eigen::Map<
      Eigen::Matrix<_F2,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>
        ASmallMap(ASmall,LDAS,NSub);

    size_t i(0);
    for( auto& iCut : SubMatCut ) {
      size_t deltaI = iCut.second - iCut.first;
      size_t j(0);
    for( auto& jCut : SubMatCut ) {
      size_t deltaJ = jCut.second - jCut.first;
    
      ABigMap.block(iCut.first,jCut.first,deltaI,deltaJ).noalias() +=
        ASmallMap.block(i,j,deltaI,deltaJ);
    
      j += deltaJ;
    }
      i += deltaI;
    }
  };

  // Instantiate functions

  template 
  void SubMatSet(size_t M, size_t N, size_t MSub, size_t NSub, double *ABig, 
    size_t LDAB, double *ASmall, size_t LDAS, 
    std::vector<std::pair<size_t,size_t>> &SubMatCut);

  template 
  void SubMatSet(size_t M, size_t N, size_t MSub, size_t NSub, dcomplex *ABig, 
    size_t LDAB, dcomplex *ASmall, size_t LDAS, 
    std::vector<std::pair<size_t,size_t>> &SubMatCut);

  template 
  void SubMatGet(size_t M, size_t N, size_t MSub, size_t NSub, double *ABig, 
    size_t LDAB, double *ASmall, size_t LDAS, 
    std::vector<std::pair<size_t,size_t>> &SubMatCut);

  template 
  void SubMatInc(size_t M, size_t N, size_t MSub, size_t NSub, double *ABig, 
    size_t LDAB, double *ASmall, size_t LDAS, 
    std::vector<std::pair<size_t,size_t>> &SubMatCut);

  template 
  void IncBySubMat(size_t M, size_t N, size_t MSub, size_t NSub, double *ABig, 
    size_t LDAB, double *ASmall, size_t LDAS, 
    std::vector<std::pair<size_t,size_t>> &SubMatCut);

  template 
  void IncBySubMat(size_t M, size_t N, size_t MSub, size_t NSub, dcomplex *ABig, 
    size_t LDAB, dcomplex *ASmall, size_t LDAS, 
    std::vector<std::pair<size_t,size_t>> &SubMatCut);

}; // namespace ChronusQ


