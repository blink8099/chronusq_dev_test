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
#include <cqlinalg/blasutil.hpp>
#include <cqlinalg/blasext.hpp>

namespace ChronusQ {

  template <typename T> T ComplexScale();
  template <> double ComplexScale(){ return -1; }
  template <> dcomplex ComplexScale(){ return dcomplex(0,1); }

  template <typename _F>
  void SpinScatter(size_t N, _F *A, size_t LDA, _F *AS, size_t LDAS,
    _F *AZ, size_t LDAZ, _F *AY, size_t LDAY, _F *AX, size_t LDAX) {

    assert(( std::is_same<_F,dcomplex>::value ));

    _F YFACT = ComplexScale<_F>();

/*
    for(auto j = 0; j < N; j++)
    for(auto i = 0; i < N; i++) {
      AS[i + j*LDAS] = A[i + j*LDA] + A[(i+N) + (j+N)*LDA];
      AZ[i + j*LDAZ] = A[i + j*LDA] - A[(i+N) + (j+N)*LDA];
      AY[i + j*LDAY] = YFACT * (A[i + (j+N)*LDA] - A[(i+N) + j*LDA]);
      AX[i + j*LDAX] = A[i + (j+N)*LDA] + A[(i+N) + j*LDA];
    }
*/

    _F* A_AA = A;
    _F* A_AB = A_AA + N*LDA;
    _F* A_BA = A_AA + N;
    _F* A_BB = A_AB + N;

    MatAdd('N','N',N,N,_F(1.),A_AA,LDA,_F(1.) ,A_BB,LDA,AS,LDAS);
    MatAdd('N','N',N,N,_F(1.),A_AA,LDA,_F(-1.),A_BB,LDA,AZ,LDAZ);
    MatAdd('N','N',N,N,YFACT ,A_AB,LDA,-YFACT ,A_BA,LDA,AY,LDAY);
    MatAdd('N','N',N,N,_F(1.),A_AB,LDA,_F(1.) ,A_BA,LDA,AX,LDAX);

  };

  template <typename _F>
  void SpinGather(size_t N, _F *A, size_t LDA, _F *AS, size_t LDAS,
    _F *AZ, size_t LDAZ, _F *AY, size_t LDAY, _F *AX, size_t LDAX) {

    assert(( std::is_same<_F,dcomplex>::value ));

    _F YFACT = 0.5*ComplexScale<_F>();

/*
    for(auto j = 0; j < N; j++)
    for(auto i = 0; i < N; i++) {
      A[i + j*LDA]         = 0.5 * (AS[i + j*LDAS] + AZ[i + j*LDAZ]);
      A[(i+N) + (j+N)*LDA] = 0.5 * (AS[i + j*LDAS] - AZ[i + j*LDAZ]);
      A[(i+N) + j*LDA]     = 0.5 * (AX[i + j*LDAS] + YFACT * AY[i + j*LDAZ]);
      A[i + (j+N)*LDA]     = 0.5 * (AX[i + j*LDAS] - YFACT * AY[i + j*LDAZ]);
    }
*/
    _F* A_AA = A;
    _F* A_AB = A_AA + N*LDA;
    _F* A_BA = A_AA + N;
    _F* A_BB = A_AB + N;

    MatAdd('N','N',N,N,_F(0.5),AS,LDAS,_F(0.5) ,AZ,LDAZ,A_AA,LDA);
    MatAdd('N','N',N,N,_F(0.5),AS,LDAS,_F(-0.5),AZ,LDAZ,A_BB,LDA);
    MatAdd('N','N',N,N,_F(0.5),AX,LDAS,YFACT   ,AY,LDAZ,A_BA,LDA);
    MatAdd('N','N',N,N,_F(0.5),AX,LDAS,-YFACT  ,AY,LDAZ,A_AB,LDA);

  };

  template
  void SpinScatter(size_t N, double *A, size_t LDA, double *AS, size_t LDAS,
    double *AZ, size_t LDAZ, double *AY, size_t LDAY, double *AX, size_t LDAX);

  template
  void SpinScatter(size_t N, dcomplex *A, size_t LDA, dcomplex *AS, size_t LDAS,
    dcomplex *AZ, size_t LDAZ, dcomplex *AY, size_t LDAY, dcomplex *AX, 
    size_t LDAX);

  template
  void SpinGather(size_t N, double *A, size_t LDA, double *AS, size_t LDAS,
    double *AZ, size_t LDAZ, double *AY, size_t LDAY, double *AX, size_t LDAX);

  template
  void SpinGather(size_t N, dcomplex *A, size_t LDA, dcomplex *AS, size_t LDAS,
    dcomplex *AZ, size_t LDAZ, dcomplex *AY, size_t LDAY, dcomplex *AX, 
    size_t LDAX);












  template <typename _F1, typename _F2, typename _FScale>
  void SetMat(char TRANS, size_t M, size_t N, _FScale ALPHA, _F1 *A, size_t LDA,
    size_t SA, _F2 *B, size_t LDB, size_t SB) {

    assert( TRANS == 'N' or TRANS == 'R' );

    using namespace Eigen;

    typedef Matrix<_F1,Dynamic,Dynamic,ColMajor> F1Mat;
    typedef Matrix<_F2,Dynamic,Dynamic,ColMajor> F2Mat;
    typedef Stride<Dynamic,Dynamic> DynamicStride; 

    typedef Map<F1Mat,0,DynamicStride> F1Map;
    typedef Map<F2Mat,0,DynamicStride> F2Map;


    F1Map AMap(A,M,N, DynamicStride(LDA,SA));
    F2Map BMap(B,M,N, DynamicStride(LDB,SB));

    if      ( TRANS == 'N' ) BMap = ALPHA * AMap;
    else if ( TRANS == 'R' ) BMap = ALPHA * AMap.conjugate();

  }

#ifdef _CQ_MKL

  template <>
  void SetMat(char TRANS, size_t M, size_t N, double ALPHA, double *A, 
    size_t LDA, size_t SA, double *B, size_t LDB, size_t SB) {

    if( SA != 1 or SB != 1)
      mkl_domatcopy2('C',TRANS,M,N,ALPHA,A,LDA,SA,B,LDB,SB);
    else
      mkl_domatcopy('C',TRANS,M,N,ALPHA,A,LDA,B,LDB);

  };

  template <>
  void SetMat(char TRANS, size_t M, size_t N, dcomplex ALPHA, dcomplex *A, 
    size_t LDA, size_t SA, dcomplex *B, size_t LDB, size_t SB) {

    if( SA != 1 or SB != 1)
      mkl_zomatcopy2('C',TRANS,M,N,ALPHA,A,LDA,SA,B,LDB,SB);
    else
      mkl_zomatcopy('C',TRANS,M,N,ALPHA,A,LDA,B,LDB);

  };

#else

  template
  void SetMat(char TRANS, size_t M, size_t N, double ALPHA, double *A, 
    size_t LDA, size_t SA, double *B, size_t LDB, size_t SB);

  template
  void SetMat(char TRANS, size_t M, size_t N, dcomplex ALPHA, dcomplex *A, 
    size_t LDA, size_t SA, dcomplex *B, size_t LDB, size_t SB);


#endif




  template<>
  void SetMatRE(char TRANS, size_t M, size_t N, double ALPHA, double *A, 
    size_t LDA, dcomplex *B, size_t LDB) {

    SetMat(TRANS,M,N,ALPHA,A,LDA,1,reinterpret_cast<double*>(B),2*LDB,2);

  }; // SetMatRE (complex)

  template<>
  void SetMatRE(char TRANS, size_t M, size_t N, double ALPHA, double *A, 
    size_t LDA, double *B, size_t LDB) {

    SetMat(TRANS,M,N,ALPHA,A,LDA,1,B,LDB,1);
    

  }; // SetMatRE (real)


  template<>
  void SetMatIM(char TRANS, size_t M, size_t N, double ALPHA, double *A, 
    size_t LDA, dcomplex *B, size_t LDB) {

    SetMat(TRANS,M,N,ALPHA,A,LDA,1,reinterpret_cast<double*>(B)+1,2*LDB,2);

  }; // SetMatIM (complex)

  template<>
  void SetMatIM(char TRANS, size_t M, size_t N, double ALPHA, double *A, 
    size_t LDA, double *B, size_t LDB) {

    assert(false);

  }; // SetMatRM (real)

  template<>
  void GetMatRE(char TRANS, size_t M, size_t N, double ALPHA, dcomplex *A, 
    size_t LDA, double *B, size_t LDB) {

    SetMat(TRANS,M,N,ALPHA,reinterpret_cast<double*>(A),2*LDA,2,B,LDA,1);

  }; // GetMatRE (complex)

  template<>
  void GetMatRE(char TRANS, size_t M, size_t N, double ALPHA, double *A, 
    size_t LDA, double *B, size_t LDB) {

    SetMat(TRANS,M,N,ALPHA,A,LDA,1,B,LDA,1);

  }; // GetMatRE (real)

  template<>
  void GetMatIM(char TRANS, size_t M, size_t N, double ALPHA, dcomplex *A, 
    size_t LDA, double *B, size_t LDB) {

    SetMat(TRANS,M,N,ALPHA,reinterpret_cast<double*>(A)+1,2*LDA,2,B,LDA,1);

  }; // GetMatIM (complex)

  template<>
  void GetMatIM(char TRANS, size_t M, size_t N, double ALPHA, double *A, 
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


