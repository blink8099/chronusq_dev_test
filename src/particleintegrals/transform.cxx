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

#include <particleintegrals.hpp>
#include <particleintegrals/onepints.hpp>
#include <particleintegrals/twopints/incore4indexreleri.hpp>
#include <particleintegrals/twopints/incore4indextpi.hpp>
#include <particleintegrals/twopints/incoreritpi.hpp>
#include <particleintegrals/twopints/gtodirecttpi.hpp>
#include <particleintegrals/onepints/relativisticints.hpp>
#include <particleintegrals/multipoleints.hpp>
#include <integrals.hpp>
#include <cqlinalg.hpp>

namespace ChronusQ {

  /**
   *  \brief (p q | r s) = T(mu, p)^H @ T(lambda, r)^H @
   *             (mu nu | lambda sigma) @ T(nu, q) @ T(sigma, s)
   *
   *  \param [in]  TRANS     Whether transpose/adjoint T
   *  \param [in]  T         Transformation matrix
   *  \param [in]  LDT       Leading dimension of T
   *  \param [in]  off_sizes Vector of 4 pairs,
   *                         a pair of offset and size for each index.
   *  \param [out] out       Return the contraction result.
   *  \param [in]  increment Perform out += result if true
   */
  template <typename IntsT>
  template <typename TransT, typename OutT>
  void InCore4indexTPI<IntsT>::subsetTransform(
      char TRANS, const TransT* T, int LDT,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      OutT* out, bool increment) const {
    typedef typename std::conditional<
        (std::is_same<IntsT, dcomplex>::value or
         std::is_same<TransT, dcomplex>::value),
        dcomplex, double>::type ResultsT;
    CQMemManager &mem = this->memManager();
    size_t NB = this->nBasis();
    std::vector<size_t> offs, SCR_nRows{NB * NB * NB};
    for (const auto &off_size : off_sizes) {
      SCR_nRows.push_back(SCR_nRows.back() / NB * off_size.second);
      if (TRANS == 'T' or TRANS == 'C')
        offs.push_back(off_size.first);
      else if (TRANS == 'N')
        offs.push_back(off_size.first * LDT);
    }

    blas::Op OP_TRANS;
    if (TRANS == 'T') {
      OP_TRANS = blas::Op::Trans;
    } else if (TRANS == 'C') {
      OP_TRANS = blas::Op::ConjTrans;
    } else if (TRANS == 'N') {
      OP_TRANS = blas::Op::NoTrans;
    }

    ResultsT* SCR  = mem.malloc<ResultsT>(NB * std::max(SCR_nRows[1], SCR_nRows[3]));
    ResultsT* SCR2 = mem.malloc<ResultsT>(NB * SCR_nRows[2]);
    // SCR (nu lambda sigma, p) = (mu, nu | lambda sigma)^H @ T(mu, p)
    blas::gemm(blas::Layout::ColMajor,blas::Op::ConjTrans, OP_TRANS, SCR_nRows[0], off_sizes[0].second, NB,
        ResultsT(1.), pointer(), NB, T+offs[0], LDT,
        ResultsT(0.), SCR, SCR_nRows[0]);
    // SCR2(lambda sigma p, q) = SCR(nu, lambda sigma p)^H @ T(nu, q)
    blas::gemm(blas::Layout::ColMajor,blas::Op::ConjTrans, OP_TRANS, SCR_nRows[1], off_sizes[1].second, NB,
        ResultsT(1.), SCR, NB, T+offs[1], LDT,
        ResultsT(0.), SCR2, SCR_nRows[1]);
    // SCR(sigma p q, r) = SCR2(lambda, sigma p q)^H @ T(lambda, r)
    blas::gemm(blas::Layout::ColMajor,blas::Op::ConjTrans, OP_TRANS, SCR_nRows[2], off_sizes[2].second, NB,
        ResultsT(1.), SCR2,NB, T+offs[2], LDT,
        ResultsT(0.), SCR, SCR_nRows[2]);
    // (p q | r, s) = SCR(sigma, p q r)^H @ T(sigma, s)
    //              = T(mu, p)^H @ T(lambda, r)^H @
    //               (mu, nu | lambda sigma) @ T(nu, q) @ T(sigma, s)
    OutT outFactor = increment ? 1.0 : 0.0;
    blas::gemm(blas::Layout::ColMajor,blas::Op::ConjTrans, OP_TRANS, SCR_nRows[3], off_sizes[3].second, NB,
        OutT(1.),     SCR, NB, T+offs[3], LDT,
        outFactor,    out, SCR_nRows[3]);
    mem.free(SCR, SCR2);
  }
  template void InCore4indexTPI<double>::subsetTransform(
      char TRANS, const double* T, int LDT,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      double* out, bool increment) const;
  template void InCore4indexTPI<double>::subsetTransform(
      char TRANS, const dcomplex* T, int LDT,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      dcomplex* out, bool increment) const;
  template void InCore4indexTPI<dcomplex>::subsetTransform(
      char TRANS, const dcomplex* T, int LDT,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      dcomplex* out, bool increment) const;

  template <>
  template <>
  void InCore4indexTPI<dcomplex>::subsetTransform(
      char TRANS, const double* T, int LDT,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      dcomplex* out, bool increment) const {
    CQMemManager &mem = this->memManager();
    size_t NB = this->nBasis();
    std::vector<size_t> offs(4), SCR_nRows{NB * NB * NB};
    for (int i = 3; i >= 0; i--) {
      SCR_nRows.push_back(SCR_nRows.back() / NB * off_sizes[i].second);
      if (TRANS == 'T' or TRANS == 'C')
        offs[i] = off_sizes[i].first;
      else if (TRANS == 'N')
        offs[i] = off_sizes[i].first * LDT;
    }
    dcomplex* SCR  = mem.malloc<dcomplex>(NB * std::max(SCR_nRows[1], SCR_nRows[3]));
    dcomplex* SCR2 = mem.malloc<dcomplex>(NB * SCR_nRows[2]);
    
    if (TRANS == 'T' or TRANS == 'C')
      TRANS = 'N';
    else if (TRANS == 'N')
      TRANS = 'C';


    blas::Op OP_TRANS;
    if (TRANS == 'T') {
      OP_TRANS = blas::Op::Trans;
    } else if (TRANS == 'C') {
      OP_TRANS = blas::Op::ConjTrans;
    } else if (TRANS == 'N') {
      OP_TRANS = blas::Op::NoTrans;
    }
    // SCR (s, mu nu lambda) = T(sigma, s)^H @ (mu nu | lambda, sigma)^H
    blas::gemm(blas::Layout::ColMajor,OP_TRANS, blas::Op::ConjTrans, off_sizes[3].second, SCR_nRows[0], NB,
        dcomplex(1.), T+offs[3], LDT, pointer(), SCR_nRows[0],
        dcomplex(0.), SCR, off_sizes[3].second);
    // SCR2(r, s mu nu) = T(lambda, r)^H @ SCR(s mu nu lambda)^H
    blas::gemm(blas::Layout::ColMajor,OP_TRANS, blas::Op::ConjTrans, off_sizes[2].second, SCR_nRows[1], NB,
        dcomplex(1.), T+offs[2], LDT, SCR, SCR_nRows[1],
        dcomplex(0.), SCR2,off_sizes[2].second);
    // SCR(q, r s mu) = T(nu, q)^H @ SCR2(r s mu, nu)^H
    blas::gemm(blas::Layout::ColMajor,OP_TRANS, blas::Op::ConjTrans, off_sizes[1].second, SCR_nRows[2], NB,
        dcomplex(1.), T+offs[1], LDT, SCR2,SCR_nRows[2],
        dcomplex(0.), SCR, off_sizes[1].second);
    // (p, q | r s) = T(mu, p)^H @ SCR(q r s, mu)^H
    //              = T(mu, p)^H @ T(lambda, r)^H @
    //               (mu, nu | lambda sigma) @ T(nu, q) @ T(sigma, s)
    dcomplex outFactor = increment ? 1.0 : 0.0;
    blas::gemm(blas::Layout::ColMajor,OP_TRANS, blas::Op::ConjTrans, off_sizes[0].second, SCR_nRows[3], NB,
        dcomplex(1.), T+offs[0], LDT, SCR, SCR_nRows[3],
        outFactor,    out, off_sizes[0].second);
    mem.free(SCR, SCR2);
  }
  
 /**
   *  \brief 4 Components
   *         (p q | r s) = T(mu, p)^H @ T(lambda, r)^H @
   *             (mu nu | lambda sigma) @ T(nu, q) @ T(sigma, s)
   *
   *  \param [in]  TRANS     Whether transpose/adjoint T
   *  \param [in]  T(L/S)    Transformation matrix is as (Large, Small) 
   *  \param [in]  LDT(L/S)  Leading dimension of T(L/S)
   *  \param [in]  off_sizes Vector of 4 pairs,
   *                         a pair of offset and size for each index.
   *  \param [out] out       Return the contraction result.
   *  \param [in]  increment Perform out += result if true
   */
  template <typename IntsT>
  template <typename TransT, typename OutT>
  void InCore4indexRelERI<IntsT>::subsetTransform(
      char TRANS, const TransT* T, int LDT,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      OutT* out, bool increment) const {
    
    std::cout << "in 4C subsetTransform Now " << std::endl;
     
    // 4C subset transform, T is supposed to be at least 2*NB as (Large, Small) Component
    // if TRANS = 'N' or 'R' --> LDT = 2*NB
    // else if TRANS = 'T' or 'C' --> the other dimension is 2*NB 
    size_t NB  = this->nBasis();
    size_t NB2 = NB*NB; 
    CQMemManager &mem = this->memManager();
    
    std::vector<size_t> TCols;
    for (const auto &off_size : off_sizes) {
      TCols.push_back(off_size.first+off_size.second);
    }

    size_t TColMax = *std::max_element(TCols.begin(), TCols.end());
    TransT* TLarge = mem.template malloc<TransT>(NB*TColMax);
    TransT* TSmall = mem.template malloc<TransT>(NB*TColMax);
    
    size_t NBHalf = NB / 2;
    if (TRANS == 'N' or TRANS == 'R') { 
      SetMat('N', NBHalf, TColMax, TransT(1.), T,           LDT, TLarge, NB);
      SetMat('N', NBHalf, TColMax, TransT(1.), T +  NBHalf, LDT, TSmall, NB);
      SetMat('N', NBHalf, TColMax, TransT(1.), T +2*NBHalf, LDT, TLarge+NBHalf, NB);
      SetMat('N', NBHalf, TColMax, TransT(1.), T +3*NBHalf, LDT, TSmall+NBHalf, NB);
    } else if (TRANS == 'T' or TRANS == 'C') {
      SetMat('N', TColMax, NBHalf, TransT(1.), T,               LDT, TLarge, TColMax);
      SetMat('N', TColMax, NBHalf, TransT(1.), T +  NBHalf*LDT, LDT, TSmall, TColMax);
      SetMat('N', TColMax, NBHalf, TransT(1.), T +2*NBHalf*LDT, LDT, TLarge+NBHalf*TColMax, TColMax);
      SetMat('N', TColMax, NBHalf, TransT(1.), T +3*NBHalf*LDT, LDT, TSmall+NBHalf*TColMax, TColMax);
    } else {
      CErr("Wrong TRANS Input");
    }
    
    // NR LLLL part
    // std::cout << "----Transform LLLL " << std::endl;
    subsetTransformWithLSComps("LLLL", TRANS, TLarge, NB, TSmall, NB, 
      off_sizes, this->pointer(), out);
    
    if (this->nRelComp() > 0) {

      // Dirac-Coulomb Term 
      size_t out_LDA = off_sizes[0].second*off_sizes[1].second;
      bool outSymm = (off_sizes[0] == off_sizes[2] and off_sizes[1] == off_sizes[3]); 
      auto & spinor = components_[0];
      
//      if (outSymm) {
//        OutT * SCR = mem.template malloc<OutT>(out_LDA*out_LDA);
//        
//        // SSLL Transformation
//        subsetTransformWithLSComps("SSLL", TRANS, TLarge, NB, TSmall, NB, 
//          off_sizes, spinor.pointer(), SCR);
//        MatAdd('N', 'N', out_LDA, out_LDA, OutT(1.), out, out_LDA,
//          OutT(1.), SCR, out_LDA, out, out_LDA);
//
//        // LLSS Transformation
//        MatAdd('N', 'T', out_LDA, out_LDA, OutT(1.), out, out_LDA,
//          OutT(1.), SCR, out_LDA, out, out_LDA);
//        
//        mem.free(SCR);
//      } else {
        IntsT * SCR = mem.template malloc<IntsT>(NB2*NB2);
        
        // SSLL Transformation
        // std::cout << "----Transform SSLL " << std::endl;
        subsetTransformWithLSComps("SSLL", TRANS, TLarge, NB, TSmall, NB, 
          off_sizes, spinor.pointer(), out, true);

        // LLSS Transformation
        // std::cout << "----Transform LLSS " << std::endl;
        SetMat('T', NB2, NB2, IntsT(1.), spinor.pointer(), NB2, SCR, NB2);
        subsetTransformWithLSComps("LLSS", TRANS, TLarge, NB, TSmall, NB, 
          off_sizes, SCR, out, true);
        
        mem.free(SCR);
//      }
    }

    if (this->nRelComp() > 1) {
      CErr("Guant terms AO to MO transformation is not implemented");
    }      

    mem.free(TLarge, TSmall);
    std::cout << "4C subsetTransform FINISHED " << std::endl;
  }; // InCore4indexRelERI::subsetTransform
  
  template void InCore4indexRelERI<double>::subsetTransform(
      char TRANS, const double* T, int LDT,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      double* out, bool increment) const;
  template void InCore4indexRelERI<double>::subsetTransform(
      char TRANS, const dcomplex* T, int LDT,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      dcomplex* out, bool increment) const;
  template void InCore4indexRelERI<dcomplex>::subsetTransform(
      char TRANS, const double* T, int LDT,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      dcomplex* out, bool increment) const;
  template void InCore4indexRelERI<dcomplex>::subsetTransform(
      char TRANS, const dcomplex* T, int LDT,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      dcomplex* out, bool increment) const;
  
  /**
   *  \brief 4 Components AO to MO subset Tranfrom with LS Components
   *         (p q | r s) = T(mu, p)^H @ T(lambda, r)^H @
   *             (mu nu | lambda sigma) @ T(nu, q) @ T(sigma, s)
   *
   *  \param [in]  TRANS     Whether transpose/adjoint T
   *  \param [in]  T         Transformation matrix is as (Large, Small)
   *  \param [in]  LDT       Leading dimension of T
   *  \param [in]  off_sizes Vector of 4 pairs,
   *                         a pair of offset and size for each index.
   *  \param [out] out       Return the contraction result.
   *  \param [in]  increment Perform out += result if true
   */
  template <typename IntsT>
  template <typename TransT, typename OutT>
  void InCore4indexRelERI<IntsT>::subsetTransformWithLSComps(
      const std::string & LSComps, char TRANS, 
      const TransT* TL, int LDTL, const TransT* TS, int LDTS,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      const IntsT* in, OutT* out, bool increment) const {

    typedef typename std::conditional<
        (std::is_same<IntsT, dcomplex>::value or
         std::is_same<TransT, dcomplex>::value),
        dcomplex, double>::type ResultsT;
    
    size_t NB = this->nBasis();
    CQMemManager &mem = this->memManager();
    
    std::vector<size_t> offs, SCR_nRows{NB * NB * NB}, LDT;
    std::vector<const TransT*> T;

    for (auto i = 0ul; i < off_sizes.size(); i++) {

      SCR_nRows.push_back(SCR_nRows.back() / NB * off_sizes[i].second);
      
      if(LSComps[i] == 'L') { 
        T.push_back(TL);
        LDT.push_back(LDTL);
      } else if (LSComps[i] == 'S') {
        T.push_back(TS);
        LDT.push_back(LDTS);
      }

      if (TRANS == 'T' or TRANS == 'C')
        offs.push_back(off_sizes[i].first);
      else if (TRANS == 'N')
        offs.push_back(off_sizes[i].first * LDT[i]);
    }
    
    ResultsT* SCR  = mem.malloc<ResultsT>(NB * std::max(SCR_nRows[1], SCR_nRows[3]));
    ResultsT* SCR2 = mem.malloc<ResultsT>(NB * SCR_nRows[2]);
    
    blas::Op OP_TRANS;
    if (TRANS == 'T') {
      OP_TRANS = blas::Op::Trans;
    } else if (TRANS == 'C') {
      OP_TRANS = blas::Op::ConjTrans;
    } else if (TRANS == 'N') {
      OP_TRANS = blas::Op::NoTrans;
    }
    
    // SCR (nu lambda sigma, p) = (mu, nu | lambda sigma)^H @ T(mu, p)
    blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, OP_TRANS, 
        SCR_nRows[0], off_sizes[0].second, NB,
        ResultsT(1.), in, NB, T[0]+offs[0], LDT[0],
        ResultsT(0.), SCR, SCR_nRows[0]);
    // SCR2(lambda sigma p, q) = SCR(nu, lambda sigma p)^H @ T(nu, q)
    blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, OP_TRANS, 
        SCR_nRows[1], off_sizes[1].second, NB,
        ResultsT(1.), SCR, NB, T[1]+offs[1], LDT[1],
        ResultsT(0.), SCR2, SCR_nRows[1]);
    // SCR(sigma p q, r) = SCR2(lambda, sigma p q)^H @ T(lambda, r)
    blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, OP_TRANS, 
        SCR_nRows[2], off_sizes[2].second, NB,
        ResultsT(1.), SCR2,NB, T[2]+offs[2], LDT[2],
        ResultsT(0.), SCR, SCR_nRows[2]);
    // (p q | r, s) = SCR(sigma, p q r)^H @ T(sigma, s)
    //              = T(mu, p)^H @ T(lambda, r)^H @
    //               (mu, nu | lambda sigma) @ T(nu, q) @ T(sigma, s)
    OutT outFactor = increment ? 1.0 : 0.0;
    blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, OP_TRANS, 
        SCR_nRows[3], off_sizes[3].second, NB,
        OutT(1.),     SCR, NB, T[3]+offs[3], LDT[3],
        outFactor,    out, SCR_nRows[3]);
    mem.free(SCR, SCR2);
  };

  template void InCore4indexRelERI<double>::subsetTransformWithLSComps(
      const std::string & LSComps, char TRANS, 
      const double* TL, int LDTL, const double* TS, int LDTS,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      const double* in, double* out, bool increment) const;
  template void InCore4indexRelERI<double>::subsetTransformWithLSComps(
      const std::string & LSComps, char TRANS, 
      const dcomplex* TL, int LDTL, const dcomplex* TS, int LDTS,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      const double* in, dcomplex* out, bool increment) const;
  template void InCore4indexRelERI<dcomplex>::subsetTransformWithLSComps(
      const std::string & LSComps, char TRANS, 
      const dcomplex* TL, int LDTL, const dcomplex* TS, int LDTS,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      const dcomplex* in, dcomplex* out, bool increment) const;
  
  template <>
  template <>
  void InCore4indexRelERI<dcomplex>::subsetTransformWithLSComps(
      const std::string & LSComps, char TRANS, 
      const double* TL, int LDTL, const double* TS, int LDTS,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      const dcomplex* in, dcomplex* out, bool increment) const {
    
    CQMemManager &mem = this->memManager();
    size_t NB = this->nBasis();
    
    std::vector<size_t> offs, SCR_nRows{NB * NB * NB}, LDT;
    std::vector<const double*> T;

    for (auto i = 0ul; i < off_sizes.size(); i++) {
      SCR_nRows.push_back(SCR_nRows.back() / NB * off_sizes[i].second);
      
      if(LSComps[i] == 'L') { 
        T.push_back(TL);
        LDT.push_back(LDTL);
      } else if (LSComps[i] == 'S') {
        T.push_back(TS);
        LDT.push_back(LDTS);
      }
   
      if (TRANS == 'T' or TRANS == 'C')
        offs.push_back(off_sizes[i].first);
      else if (TRANS == 'N')
        offs.push_back(off_sizes[i].first * LDT[i]);
    }

    dcomplex* SCR  = mem.malloc<dcomplex>(NB * std::max(SCR_nRows[1], SCR_nRows[3]));
    dcomplex* SCR2 = mem.malloc<dcomplex>(NB * SCR_nRows[2]);
    
    if (TRANS == 'T' or TRANS == 'C')
      TRANS = 'N';
    else if (TRANS == 'N')
      TRANS = 'C';
    
    blas::Op OP_TRANS;
    if (TRANS == 'T') {
      OP_TRANS = blas::Op::Trans;
    } else if (TRANS == 'C') {
      OP_TRANS = blas::Op::ConjTrans;
    } else if (TRANS == 'N') {
      OP_TRANS = blas::Op::NoTrans;
    }
    
    // SCR (s, mu nu lambda) = T(sigma, s)^H @ (mu nu | lambda, sigma)^H
    blas::gemm(blas::Layout::ColMajor,OP_TRANS, blas::Op::ConjTrans,
        off_sizes[3].second, SCR_nRows[0], NB,
        dcomplex(1.), T[3]+offs[3], LDT[3], in, SCR_nRows[0],
        dcomplex(0.), SCR, off_sizes[3].second);
    // SCR2(r, s mu nu) = T(lambda, r)^H @ SCR(s mu nu lambda)^H
    blas::gemm(blas::Layout::ColMajor,OP_TRANS, blas::Op::ConjTrans,
        off_sizes[2].second, SCR_nRows[1], NB,
        dcomplex(1.), T[2]+offs[2], LDT[2], SCR, SCR_nRows[1],
        dcomplex(0.), SCR2,off_sizes[2].second);
    // SCR(q, r s mu) = T(nu, q)^H @ SCR2(r s mu, nu)^H
    blas::gemm(blas::Layout::ColMajor,OP_TRANS, blas::Op::ConjTrans,
        off_sizes[1].second, SCR_nRows[2], NB,
        dcomplex(1.), T[1]+offs[1], LDT[1], SCR2,SCR_nRows[2],
        dcomplex(0.), SCR, off_sizes[1].second);
    // (p, q | r s) = T(mu, p)^H @ SCR(q r s, mu)^H
    //              = T(mu, p)^H @ T(lambda, r)^H @
    //               (mu, nu | lambda sigma) @ T(nu, q) @ T(sigma, s)
    dcomplex outFactor = increment ? 1.0 : 0.0;
    blas::gemm(blas::Layout::ColMajor,OP_TRANS, blas::Op::ConjTrans,
        off_sizes[0].second, SCR_nRows[3], NB,
        dcomplex(1.), T[0]+offs[0], LDT[0], SCR, SCR_nRows[3],
        outFactor,    out, off_sizes[0].second);
    mem.free(SCR, SCR2);
  };

  /**
   *  \brief (p q | r s) = T(mu, p)^H @ T(lambda, r)^H @
   *             (mu nu | lambda sigma) @ T(nu, q) @ T(sigma, s)
   *
   *  \param [in] TRANS Whether transpose/adjoint T
   *  \param [in] T     Transformation matrix
   *  \param [in] NT    Number of columns for T
   *  \param [in] LDT   Leading dimension of T
   *
   *  \return InCore4indexERI object with element type derived from
   *          IntsT and TransT.
   */
  template <typename IntsT>
  template <typename TransT>
  InCore4indexTPI<typename std::conditional<
  (std::is_same<IntsT, dcomplex>::value or
   std::is_same<TransT, dcomplex>::value),
  dcomplex, double>::type> InCore4indexTPI<IntsT>::transform(
      char TRANS, const TransT* T, int NT, int LDT) const{
    InCore4indexTPI<typename std::conditional<
        (std::is_same<IntsT, dcomplex>::value or
         std::is_same<TransT, dcomplex>::value),
        dcomplex, double>::type> transInts(this->memManager(), NT);
    subsetTransform(TRANS,T,LDT,{{0,NT},{0,NT},{0,NT},{0,NT}},
                    transInts.pointer(),false);
    return transInts;
  }
  
  template InCore4indexTPI<double> InCore4indexTPI<double>::transform(char TRANS, const double* T, int NT, int LDT) const;
  template InCore4indexTPI<dcomplex> InCore4indexTPI<double>::transform(char TRANS, const dcomplex* T, int NT, int LDT) const;
  /**
   *  \brief B(L, p, q) = T(mu, p)^H @ B(L, mu, nu) @ T(nu, q)
   *
   *  \param [in]  TRANS     Whether transpose/adjoint T
   *  \param [in]  T         Transformation matrix
   *  \param [in]  LDT       Leading dimension of T
   *  \param [in]  off_sizes Vector of 2 pairs,
   *                         a pair of offset and size for each index.
   *  \param [out] out       Return the contraction result.
   *  \param [in]  increment Perform out += result if true
   */
  template <typename IntsT>
  template <typename TransT, typename OutT>
  void InCoreRITPI<IntsT>::subsetTransform(
      char TRANS, const TransT* T, int LDT,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      OutT* out, bool increment) const {
    typedef typename std::conditional<
        (std::is_same<IntsT, dcomplex>::value or
         std::is_same<TransT, dcomplex>::value),
        dcomplex, double>::type ResultsT;
    std::vector<size_t> offs;
    for (const auto &off_size : off_sizes) {
      if (TRANS == 'T' or TRANS == 'C')
        offs.push_back(off_size.first);
      else if (TRANS == 'N')
        offs.push_back(off_size.first * LDT);
    }

    blas::Op OP_TRANS;
    if (TRANS == 'T') {
      OP_TRANS = blas::Op::Trans;
    } else if (TRANS == 'C') {
      OP_TRANS = blas::Op::ConjTrans;
    } else if (TRANS == 'N') {
      OP_TRANS = blas::Op::NoTrans;
    }


    CQMemManager &mem = this->memManager();
    size_t NB   = this->nBasis();
    size_t NBRI = nRIBasis();
    IntsT* SCR = mem.malloc<IntsT>(NB * NB * NBRI);
    ResultsT* SCR2 = mem.malloc<ResultsT>(NB * NBRI * off_sizes[0].second);
    // SCR(mu nu, L) = ( L | mu nu )^T
    SetMat('T', NBRI, NB*NB, IntsT(1.),
         pointer(), NBRI, 1, SCR, NB*NB, 1);
    // SCR2(nu L, p) = SCR(mu, nu L)^H @ T(mu, p)
    blas::gemm(blas::Layout::ColMajor,blas::Op::ConjTrans, OP_TRANS, NB*NBRI, off_sizes[0].second, NB,
        ResultsT(1.), SCR, NB, T+offs[0], LDT,
        ResultsT(0.), SCR2, NB*NBRI);
    // ( L | p, q ) = SCR2(nu, L p)^H @ T(nu, q)
    //              = T(mu, p)^H @ ( L | mu nu ) @ T(nu, q)
    OutT outFactor = increment ? 1.0 : 0.0;
    blas::gemm(blas::Layout::ColMajor,blas::Op::ConjTrans, OP_TRANS, NBRI*off_sizes[0].second, off_sizes[1].second, NB,
        ResultsT(1.), SCR2,NB, T+offs[1], LDT,
        outFactor, out, NBRI*off_sizes[0].second);
    mem.free(SCR, SCR2);
  }
  template void InCoreRITPI<double>::subsetTransform(
      char TRANS, const double* T, int LDT,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      double* out, bool increment) const;
  template void InCoreRITPI<double>::subsetTransform(
      char TRANS, const dcomplex* T, int LDT,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      dcomplex* out, bool increment) const;
  template void InCoreRITPI<dcomplex>::subsetTransform(
      char TRANS, const dcomplex* T, int LDT,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      dcomplex* out, bool increment) const;

  template <>
  template <>
  void InCoreRITPI<dcomplex>::subsetTransform(
      char TRANS, const double* T, int LDT,
      const std::vector<std::pair<size_t,size_t>> &off_sizes,
      dcomplex* out, bool increment) const {
    std::vector<size_t> offs;
    for (const auto &off_size : off_sizes) {
      if (TRANS == 'T' or TRANS == 'C')
        offs.push_back(off_size.first);
      else if (TRANS == 'N')
        offs.push_back(off_size.first * LDT);
    }
    CQMemManager &mem = this->memManager();
    size_t NB   = this->nBasis();
    size_t NBRI = nRIBasis();
    dcomplex* SCR  = mem.malloc<dcomplex>(off_sizes[1].second * NBRI * NB);
    dcomplex* SCR2 = mem.malloc<dcomplex>(
          off_sizes[0].second * off_sizes[1].second * NBRI);
    if (TRANS == 'T' or TRANS == 'C')
      TRANS = 'N';
    else if (TRANS == 'N')
      TRANS = 'C';

    blas::Op OP_TRANS;
    if (TRANS == 'T') {
      OP_TRANS = blas::Op::Trans;
    } else if (TRANS == 'C') {
      OP_TRANS = blas::Op::ConjTrans;
    } else if (TRANS == 'N') {
      OP_TRANS = blas::Op::NoTrans;
    }


    // SCR(q, L mu) = T(nu, q)^H @ ( L | mu, nu )^H
    blas::gemm(blas::Layout::ColMajor,OP_TRANS, blas::Op::ConjTrans, off_sizes[1].second, NBRI*NB, NB,
        dcomplex(1.), T+offs[1], LDT, pointer(), NBRI*NB,
        dcomplex(0.), SCR, off_sizes[1].second);
    // SCR2(p, q L) = T(mu, p)^H @ SCR(q L, mu)^H
    blas::gemm(blas::Layout::ColMajor,OP_TRANS, blas::Op::ConjTrans, off_sizes[0].second, off_sizes[1].second*NBRI, NB,
        dcomplex(1.), T+offs[0], LDT, SCR, off_sizes[1].second*NBRI,
        dcomplex(0.), SCR2,off_sizes[0].second);
    // ( L | p q ) = SCR2(p q, L)^T
    //             = T(mu, p)^H @ ( L | mu nu ) @ T(nu, q)
    size_t pq_size = off_sizes[0].second * off_sizes[1].second;
    if (increment)
      SetMat('T', pq_size, NBRI, dcomplex(1.), SCR2, pq_size, 1, out, NBRI, 1);
    else
      MatAdd('T', 'N', NBRI, pq_size, dcomplex(1.), SCR2, pq_size,
             dcomplex(1.), out, NBRI, out, NBRI);
    mem.free(SCR, SCR2);
  }

  /**
   *  \brief B(L, p, q) = T(mu, p)^H @ B(L, mu, nu) @ T(nu, q)
   *
   *  \param [in] TRANS Whether transpose/adjoint T
   *  \param [in] T     Transformation matrix
   *  \param [in] NT    Number of columns for T
   *  \param [in] LDT   Leading dimension of T
   *
   *  \return InCoreRIERI object with element type derived from
   *          IntsT and TransT.
   */
  template <typename IntsT>
  template <typename TransT>
  InCoreRITPI<typename std::conditional<
  (std::is_same<IntsT, dcomplex>::value or
   std::is_same<TransT, dcomplex>::value),
  dcomplex, double>::type> InCoreRITPI<IntsT>::transform(
      char TRANS, const TransT* T, int NT, int LDT) const {
    InCoreRITPI<typename std::conditional<
        (std::is_same<IntsT, dcomplex>::value or
         std::is_same<TransT, dcomplex>::value),
        dcomplex, double>::type> transInts(this->memManager(), NT, nRIBasis());
    subsetTransform(TRANS,T,LDT,{{0,NT},{0,NT}},transInts.pointer(),false);
    return transInts;
  }

  template InCoreRITPI<double> InCoreRITPI<double>::transform(char TRANS, const double* T, int NT, int LDT) const;
  template InCoreRITPI<dcomplex> InCoreRITPI<double>::transform(char TRANS, const dcomplex* T, int NT, int LDT) const;

#define IF_TRANSFORM_INTS(EI) \
  if (tID == typeid(EI<double>))\
    return std::dynamic_pointer_cast<ParticleIntegrals>(\
        std::make_shared<EI<TransT>>(\
            dynamic_cast<const EI<double>&>(ints)\
                .transform(TRANS, T, NT, LDT)));\
  if (tID == typeid(EI<dcomplex>))\
    return std::dynamic_pointer_cast<ParticleIntegrals>(\
        std::make_shared<EI<dcomplex>>(\
            dynamic_cast<const EI<dcomplex>&>(ints)\
                .transform(TRANS, T, NT, LDT)))

  template <typename TransT>
  std::shared_ptr<ParticleIntegrals> ParticleIntegrals::transform(
      const ParticleIntegrals &ints, char TRANS, const TransT* T, int NT, int LDT) {
    const std::type_info &tID(typeid(ints));
    IF_TRANSFORM_INTS(OnePInts);
    IF_TRANSFORM_INTS(OnePRelInts);
    IF_TRANSFORM_INTS(InCore4indexTPI);
    IF_TRANSFORM_INTS(InCoreRITPI);
    IF_TRANSFORM_INTS(MultipoleInts);
    CErr("Transformation NYI for requested type of ElectronIntegrals");
    return nullptr;
  }

  template std::shared_ptr<ParticleIntegrals> ParticleIntegrals::transform(
      const ParticleIntegrals &ints, char TRANS, const double* T, int NT, int LDT);
  template std::shared_ptr<ParticleIntegrals> ParticleIntegrals::transform(
      const ParticleIntegrals &ints, char TRANS, const dcomplex* T, int NT, int LDT);

  template <typename IntsT>
  template <typename TransT>
  Integrals<typename std::conditional<
  (std::is_same<IntsT, dcomplex>::value or
   std::is_same<TransT, dcomplex>::value),
  dcomplex, double>::type> Integrals<IntsT>::transform(
      const std::vector<OPERATOR> &ops, const std::vector<std::string> &miscOps,
      char TRANS, const TransT* T, int NT, int LDT) const {
    typedef typename std::conditional<
        (std::is_same<IntsT, dcomplex>::value or
         std::is_same<TransT, dcomplex>::value),
        dcomplex, double>::type ResultT;
    Integrals<ResultT> transInts;
    for (const OPERATOR op : ops)
      switch (op) {
      case OVERLAP:
        transInts.overlap = std::dynamic_pointer_cast<OnePInts<ResultT>>(
            ParticleIntegrals::transform(*overlap, TRANS, T, NT, LDT));
        break;
      case KINETIC:
        transInts.kinetic = std::dynamic_pointer_cast<OnePInts<ResultT>>(
            ParticleIntegrals::transform(*kinetic, TRANS, T, NT, LDT));
        break;
      case NUCLEAR_POTENTIAL:
        transInts.potential = std::dynamic_pointer_cast<OnePInts<ResultT>>(
            ParticleIntegrals::transform(*potential, TRANS, T, NT, LDT));
        break;
      case LEN_ELECTRIC_MULTIPOLE:
        transInts.lenElectric = std::dynamic_pointer_cast<MultipoleInts<ResultT>>(
            ParticleIntegrals::transform(*lenElectric, TRANS, T, NT, LDT));
        break;
      case VEL_ELECTRIC_MULTIPOLE:
        transInts.velElectric = std::dynamic_pointer_cast<MultipoleInts<ResultT>>(
            ParticleIntegrals::transform(*velElectric, TRANS, T, NT, LDT));
        break;
      case MAGNETIC_MULTIPOLE:
        transInts.magnetic = std::dynamic_pointer_cast<MultipoleInts<ResultT>>(
            ParticleIntegrals::transform(*magnetic, TRANS, T, NT, LDT));
        break;
      case ELECTRON_REPULSION:
        transInts.TPI = std::dynamic_pointer_cast<TwoPInts<ResultT>>(
            ParticleIntegrals::transform(*TPI, TRANS, T, NT, LDT));
        break;
      }
    for (const std::string &op : miscOps)
      transInts.misc[op] = ParticleIntegrals::transform(*misc.at(op), TRANS, T, NT, LDT);
    return transInts;
  }

  template Integrals<double> Integrals<double>::transform(
      const std::vector<OPERATOR>&, const std::vector<std::string>&,
      char TRANS, const double* T, int NT, int LDT) const;
  template Integrals<dcomplex> Integrals<double>::transform(
      const std::vector<OPERATOR>&, const std::vector<std::string>&,
      char TRANS, const dcomplex* T, int NT, int LDT) const;
  template Integrals<dcomplex> Integrals<dcomplex>::transform(
      const std::vector<OPERATOR>&, const std::vector<std::string>&,
      char TRANS, const double* T, int NT, int LDT) const;
  template Integrals<dcomplex> Integrals<dcomplex>::transform(
      const std::vector<OPERATOR>&, const std::vector<std::string>&,
      char TRANS, const dcomplex* T, int NT, int LDT) const;

}; // namespace ChronusQ
