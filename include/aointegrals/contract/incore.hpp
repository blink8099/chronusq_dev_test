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
#ifndef __INCLUDED_AOINTEGRALS_CONTRACT_INCORE_HPP__
#define __INCLUDED_AOINTEGRALS_CONTRACT_INCORE_HPP__


#include <aointegrals.hpp>
#include <util/matout.hpp>
#include <util/threads.hpp>
#include <cqlinalg/blas3.hpp>
#include <cqlinalg/blasext.hpp>

// Use stupid but bullet proof incore contraction for debug
//#define _BULLET_PROOF_INCORE

namespace ChronusQ {

  /**
   *  \brief Perform various tensor contractions of the full ERI
   *  tensor in core. Wraps other helper functions and provides
   *  loop structure
   *
   *  Currently supports
   *    - Coulomb-type (34,12) contractions
   *    - Exchange-type (23,12) contractions
   *
   *  Works with both real and complex matricies
   *
   *  \param [in/out] list Contains information pertinent to the 
   *    matricies to be contracted with. See TwoBodyContraction
   *    for details
   */ 
  template <typename IntsT>
  template <typename TT>
  void AOIntegrals<IntsT>::twoBodyContractIncore(
    MPI_Comm comm, std::vector<TwoBodyContraction<TT>> &list) {

    ROOT_ONLY(comm);

    auto topIncore = std::chrono::high_resolution_clock::now();

    // Loop over matricies to contract with
    for(auto &C : list) {

      // Coulomb-type (34,12) ERI contraction
      // AX(mn) = (mn | kl) X(kl)
      if( C.contType == COULOMB ) JContractIncore(comm,C);

      // Exchange-type (23,12) ERI contraction
      // AX(mn) = (mk |ln) X(kl)
      else if( C.contType == EXCHANGE ) KContractIncore(comm,C);

    } // loop over matricies

    auto botIncore = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> durIncore = botIncore - topIncore;


#ifdef _REPORT_INTEGRAL_TIMINGS
    std::cerr << "Incore Contraction took " << durIncore.count() << " s\n\n";
#endif
  }; // AOIntegrals::twoBodyContractIncore


  /**
   *  \brief Perform a Coulomb-type (34,12) ERI contraction with
   *  a one-body operator.
   */   
  template <typename IntsT>
  template <typename TT>
  void AOIntegrals<IntsT>::JContractIncore(MPI_Comm comm, TwoBodyContraction<TT> &C) {

    IntsT *X  = reinterpret_cast<IntsT*>(C.X);
    IntsT *AX = reinterpret_cast<IntsT*>(C.AX);

    // Extract the real part of X if X is Hermetian and if the ints are
    // real
    const bool extractRealPartX = 
      C.HER and std::is_same<IntsT,double>::value and 
      std::is_same<TT,dcomplex>::value;

    // Allocate scratch if IntsT and TT are different
    const bool allocAXScratch = not std::is_same<IntsT,TT>::value;


    if( extractRealPartX ) {

      X = memManager_.malloc<IntsT>(nSQ_);
      for(auto k = 0ul; k < nSQ_; k++) X[k] = std::real(C.X[k]);

    }

    if( allocAXScratch ) {

      AX = memManager_.malloc<IntsT>(nSQ_);
      std::fill_n(AX,nSQ_,0.);

    }


    #ifdef _BULLET_PROOF_INCORE

    size_t NB3 = basisSet_.nBasis * nSQ_;
    for(auto i = 0; i < basisSet_.nBasis; ++i)
    for(auto j = 0; j < basisSet_.nBasis; ++j)
    for(auto k = 0; k < basisSet_.nBasis; ++k)
    for(auto l = 0; l < basisSet_.nBasis; ++l) 

      C.AX[i + j*basisSet_.nBasis] +=ERI[i + j*basisSet_.nBasis + l*nSQ_ + k*NB3] *
        C.X[k + l*basisSet_.nBasis];

    #else


    Gemm('C','N',nSQ_,1,nSQ_,IntsT(1.),ERI,nSQ_,X,nSQ_,IntsT(0.),AX,nSQ_);

    // if Complex ints + Hermitian, conjugate
    if( std::is_same<IntsT,dcomplex>::value and C.HER )
      IMatCopy('R',basisSet_.nBasis,basisSet_.nBasis,
        IntsT(1.),AX,basisSet_.nBasis,basisSet_.nBasis);

    // If non-hermetian, transpose
    if( not C.HER )  {

      IMatCopy('T',basisSet_.nBasis,basisSet_.nBasis,
        IntsT(1.),AX,basisSet_.nBasis,basisSet_.nBasis);

    }

    #endif

    // Cleanup temporaries
    if( extractRealPartX ) memManager_.free(X);
    if( allocAXScratch ) {

      std::copy_n(AX,nSQ_,C.AX);
      memManager_.free(AX);

    }

  }; // AOIntegrals::JContractIncore


  template <typename IntsT>
  template <typename TT>
  void AOIntegrals<IntsT>::KContractIncore(MPI_Comm comm, TwoBodyContraction<TT> &C) {

    size_t NB3 = basisSet_.nBasis * nSQ_;

    #ifdef _BULLET_PROOF_INCORE

    for(auto i = 0; i < basisSet_.nBasis; ++i)
    for(auto j = 0; j < basisSet_.nBasis; ++j)
    for(auto k = 0; k < basisSet_.nBasis; ++k)
    for(auto l = 0; l < basisSet_.nBasis; ++l) {
      C.AX[i + j*basisSet_.nBasis] +=
        ERI[i + k*basisSet_.nBasis + l*nSQ_ + j*NB3] *
        C.X[k + l*basisSet_.nBasis];
    }

    #else

    size_t LAThreads = GetLAThreads();
    SetLAThreads(1);

    #pragma omp parallel for
    for(auto nu = 0; nu < basisSet_.nBasis; nu++) 
      Gemm('N','N',basisSet_.nBasis,1,nSQ_,TT(1.),
        ERI  + nu * NB3,              basisSet_.nBasis,
        C.X,                          nSQ_,TT(0.),
        C.AX + nu * basisSet_.nBasis, basisSet_.nBasis);

    SetLAThreads(LAThreads);

    #endif

  }; // AOIntegrals::KContractIncore

}; // namespace ChronusQ

#endif

