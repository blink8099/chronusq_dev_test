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
#ifndef __INCLUDED_CQLINALG_BLASUTIL_HPP__
#define __INCLUDED_CQLINALG_BLASUTIL_HPP__

#include <cqlinalg/cqlinalg_config.hpp>

namespace ChronusQ {



  // Spin Scatter a matrix
  template <typename _F>
  void SpinScatter(size_t N, _F *A, size_t LDA, _F *AS, size_t LDAS,
    _F *AZ, size_t LDAZ, _F *AY, size_t LDAY, _F *AX, size_t LDAX);

  // Spin Gather a matrix
  template <typename _F>
  void SpinGather(size_t N, _F *A, size_t LDA, _F *AS, size_t LDAS,
    _F *AZ, size_t LDAZ, _F *AY, size_t LDAY, _F *AX, size_t LDAX);





  // B = ALPHA * OP(A)
  template <typename _F1, typename _F2, typename _FScale>
  void SetMat(char TRANS, size_t M, size_t N, _FScale ALPHA, _F1 *A, size_t LDA,
    size_t SA, _F2 *B, size_t LDB, size_t SB);

  template <typename _F1, typename _F2, typename _FScale>
  void SetMat(char TRANS, size_t M, size_t N, _FScale ALPHA, _F1 *A, size_t LDA,
    _F2 *B, size_t LDB) {

    SetMat(TRANS,M,N,_F1(ALPHA),A,LDA,1,B,LDB,1);

  }


  // RE(B) = ALPHA * OP(A)
  template <typename _F>
  void SetMatRE(char TRANS, size_t M, size_t N, double ALPHA, double *A, 
    size_t LDA, _F *B, size_t LDB);

  // IM(B) = ALPHA * OP(A)
  template <typename _F>
  void SetMatIM(char TRANS, size_t M, size_t N, double ALPHA, double *A, 
    size_t LDA, _F *B, size_t LDB);

  // B = ALPHA * OP(RE(A))
  template <typename _F>
  void GetMatRE(char TRANS, size_t M, size_t N, double ALPHA, _F *A, 
    size_t LDA, double *B, size_t LDB);

  // B = ALPHA * OP(IM(A))
  template <typename _F>
  void GetMatIM(char TRANS, size_t M, size_t N, double ALPHA, _F *A, 
    size_t LDA, double *B, size_t LDB);
















  // Sub-matrix utilities for multiple, non-contiguous blocks


  /**
   *  \brief Set a packed submatrix to the corresponding blocks of
   *  the full matrix using sub matrix cuts
   *
   *  ASmall = ABig submatrix
   *
   *  \param [in] M       Number of rows in ABig
   *  \param [in] N       Number of columns in ABig
   *  \param [in] MSub    Number of rows in ASmall
   *  \param [in] NSub    Number of columns in ASmall
   *  \param [in] ABig    Raw storage of supermatrix
   *  \param [in] LDAB    Leading dimension of ABig
   *  \param [in/out] ASmall  Raw storage of submatrix
   *  \param [in] LDAS    Leading dimension of ASmall
   *  \param [in] SubMatCut Vector of pairs specifing the blocks of
   *                        the super matrix to be used
   *  
   */ 
  template <typename _F1, typename _F2>
  void SubMatSet(size_t M, size_t N, size_t MSub, size_t NSub, _F1 *ABig, 
    size_t LDAB, _F2 *ASmall, size_t LDAS, 
    std::vector<std::pair<size_t,size_t>> &SubMatCut);




  /**
   *  \brief Pack subblock of a matrix into a contiguous array
   *  using sub matrix cuts
   *
   *  ABig submatrix = ASmall
   *
   *  \param [in] M       Number of rows in ABig
   *  \param [in] N       Number of columns in ABig
   *  \param [in] MSub    Number of rows in ASmall
   *  \param [in] NSub    Number of columns in ASmall
   *  \param [in/out] ABig    Raw storage of supermatrix
   *  \param [in] LDAB    Leading dimension of ABig
   *  \param [in] ASmall  Raw storage of submatrix
   *  \param [in] LDAS    Leading dimension of ASmall
   *  \param [in] SubMatCut Vector of pairs specifing the blocks of
   *                        the super matrix to be used
   *  
   */ 
  template <typename _F1, typename _F2>
  void SubMatGet(size_t M, size_t N, size_t MSub, size_t NSub, _F1 *ABig, 
    size_t LDAB, _F2 *ASmall, size_t LDAS, 
    std::vector<std::pair<size_t,size_t>> &SubMatCut);




  /**
   *  \brief Increment a packed submatrix to the corresponding blocks of
   *  the full matrix using sub matrix cuts
   *
   *  ASmall += ABig submatrix
   *
   *  \param [in] M       Number of rows in ABig
   *  \param [in] N       Number of columns in ABig
   *  \param [in] MSub    Number of rows in ASmall
   *  \param [in] NSub    Number of columns in ASmall
   *  \param [in] ABig    Raw storage of supermatrix
   *  \param [in] LDAB    Leading dimension of ABig
   *  \param [in/out] ASmall  Raw storage of submatrix
   *  \param [in] LDAS    Leading dimension of ASmall
   *  \param [in] SubMatCut Vector of pairs specifing the blocks of
   *                        the super matrix to be used
   *  
   */ 
  template <typename _F1, typename _F2>
  void SubMatInc(size_t M, size_t N, size_t MSub, size_t NSub, _F1 *ABig, 
    size_t LDAB, _F2 *ASmall, size_t LDAS, 
    std::vector<std::pair<size_t,size_t>> &SubMatCut);




  /**
   *  \brief Increment a subblock of a matrix by a packed array
   *  using sub matrix cuts
   *
   *  ABig submatrix += ASmall
   *
   *  \param [in] M       Number of rows in ABig
   *  \param [in] N       Number of columns in ABig
   *  \param [in] MSub    Number of rows in ASmall
   *  \param [in] NSub    Number of columns in ASmall
   *  \param [in/out] ABig    Raw storage of supermatrix
   *  \param [in] LDAB    Leading dimension of ABig
   *  \param [in] ASmall  Raw storage of submatrix
   *  \param [in] LDAS    Leading dimension of ASmall
   *  \param [in] SubMatCut Vector of pairs specifing the blocks of
   *                        the super matrix to be used
   *  
   */ 
  template <typename _F1, typename _F2>
  void IncBySubMat(size_t M, size_t N, size_t MSub, size_t NSub, _F1 *ABig, 
    size_t LDAB, _F2 *ASmall, size_t LDAS, 
    std::vector<std::pair<size_t,size_t>> &SubMatCut);

}; // namespace ChronusQ


#endif
