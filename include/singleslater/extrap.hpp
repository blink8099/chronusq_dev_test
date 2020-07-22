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

#include <extrapolate.hpp>
#include <singleslater.hpp>
#include <util/matout.hpp>
#include <cqlinalg/blas1.hpp>

namespace ChronusQ {

 /**
   *  \brief Control routine for DIIS and damping for SCF
   *  
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::modifyFock() {

    ROOT_ONLY(comm); 

    // Static Damping
    scfControls.doDamp &= (std::dynamic_pointer_cast<ROFock<MatsT,IntsT>>(fockBuilder) == nullptr);
    if (scfControls.doDamp) fockDamping();

    // DIIS extrapolation
    if (scfControls.diisAlg == NONE) return;

    if (scfControls.diisAlg == CDIIS) {
      size_t nExtrap = std::min(scfConv.nSCFIter+1,scfControls.nKeep);
      scfDIIS(nExtrap);
    } else CErr("Only CDIIS is implemented so far",std::cout);

  }; // SingleSlater<T>::modifyFock



 /**
   *  \brief Static Fock Damping routine
   *
   *  F^k = (1-dp)*F^k + dp*F^{k-1}
   *  
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::fockDamping() {

    // Don't damp the first iteration.  We don't want 
    // to use the guess Fock and it's not saved anyway.
    if(scfConv.nSCFIter == 0) return;

    size_t NB = this->aoints.basisSet().nBasis;
    double dp = scfControls.dampParam;
   
    // Damp the current orthonormal Fock matrix 
    if( not savFile.exists() )
      for(auto i = 0; i < fockMatrixOrtho.size(); i++)
        MatAdd('N','N', NB, NB, MatsT(1-dp), fockMatrixOrtho[i], NB, MatsT(dp), 
          prevFock[i], NB, fockMatrixOrtho[i], NB);
    else {

      MatsT* FSCR = this->memManager.template malloc<MatsT>(NB*NB);
      const std::array<std::string,4> spinLabel =
        { "SCALAR", "MZ", "MY", "MX" };

      for(auto i = 0; i < fockMatrixOrtho.size(); i++) {

        savFile.readData("/SCF/FOCK_ORTHO_" + spinLabel[i],FSCR);

        MatAdd('N','N', NB, NB, MatsT(1-dp), fockMatrixOrtho[i], NB, MatsT(dp), 
         FSCR, NB, fockMatrixOrtho[i], NB);

      }

      this->memManager.free(FSCR);
    }

  }; // SingleSlater<T>::fockDamping


 /**
   *  \brief Commutator DIIS
   *
   *  Saves the AO fock and density matrices, evaluates the [F,D]
   *  commutator and uses this to extrapolate the Fock and density. 
   *
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::scfDIIS(size_t nExtrap) {

    // Save the current AO Fock and density matrices
    size_t NB    = this->aoints.basisSet().nBasis;
    size_t iDIIS = scfConv.nSCFIter % scfControls.nKeep;
    for(auto i = 0; i < this->fockMatrix.size(); i++) {
      std::copy_n(this->fockMatrix[i],NB*NB,diisFock[iDIIS][i]);
      std::copy_n(this->onePDM[i],NB*NB,diisOnePDM[iDIIS][i]);
    }

    // Evaluate orthonormal [F,D] and store in diisError
    FDCommutator(diisError[iDIIS]);

    scfConv.nrmFDC = 0.;
    for(auto &E : diisError[iDIIS])
      scfConv.nrmFDC = std::max(scfConv.nrmFDC,TwoNorm<double>(NB*NB,E,1));

    // Just save the Fock, density, and commutator for the first iteration
    if (scfConv.nSCFIter == 0) return;
      
    // Build the B matrix and return the coefficients for the extrapolation
    size_t nMat = fockMatrixOrtho.size();
    DIIS<MatsT> extrap(nExtrap,nMat,NB*NB,diisError);


    if(extrap.extrapolate()) { 
      // Extrapolate Fock and density matrices using DIIS coefficients
      for(auto i = 0; i < fockMatrixOrtho.size(); i++) {
        std::fill_n(fockMatrix[i],NB*NB,0.);
        std::fill_n(this->onePDM[i],NB*NB,0.);
        for(auto j = 0; j < nExtrap; j++) {
          MatAdd('N','N', NB, NB, MatsT(1.), fockMatrix[i], NB, MatsT(extrap.coeffs[j]), 
            diisFock[j][i], NB, fockMatrix[i], NB);
          MatAdd('N','N', NB, NB, MatsT(1.), this->onePDM[i], NB, 
            MatsT(extrap.coeffs[j]), diisOnePDM[j][i], NB, this->onePDM[i], NB);
        } 
      }
    } else {
      std::cout << "\n    *** WARNING: DIIS Inversion Failed -- "
                << " Defaulting to Fixed-Point step ***\n" << std::endl;
    }

    // Transform AO fock into the orthonormal basis
    ao2orthoFock();

  }; // SingleSlater<T>::scfDIIS



 /**
   *  \brief Allocates storage for different extrapolation approaches to SCF 
   *  
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::allocExtrapStorage() {

    ROOT_ONLY(comm);

    diisFock.clear();
    diisOnePDM.clear();
    diisError.clear();
    prevFock.clear();

    size_t FSize = memManager.template getSize(fockMatrix[SCALAR]);

    // Allocate memory to store previous orthonormal Focks and densities for DIIS
    if (scfControls.diisAlg != NONE) {
      for(auto i = 0; i < scfControls.nKeep; i++) {
        diisFock.emplace_back();
        diisOnePDM.emplace_back();
        diisError.emplace_back();
        for(auto j = 0; j < this->fockMatrix.size(); j++) {
          diisFock[i].emplace_back(memManager.template malloc<MatsT>(FSize));
          diisOnePDM[i].emplace_back(memManager.template malloc<MatsT>(FSize));
          diisError[i].emplace_back(memManager.template malloc<MatsT>(FSize));
          std::fill_n(diisFock[i][j],FSize,0.);
          std::fill_n(diisOnePDM[i][j],FSize,0.);
          std::fill_n(diisError[i][j],FSize,0.);
        } 
      }
    }

    // Allocate memory to store previous orthonormal Fock for damping 
    if( scfControls.doDamp and not savFile.exists() ) 
      for(auto i = 0; i < this->fockMatrix.size(); i++) 
        prevFock.emplace_back(memManager.template malloc<MatsT>(FSize));

  }; // SingleSlater<T>::allocExtrapStorage



 /**
   *  \brief Deallocates storage for different extrapolation approaches to SCF 
   *  
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::deallocExtrapStorage() {

    ROOT_ONLY(comm);

    // Deallocate memory to store previous orthonormal Focks and densities for DIIS
    if (scfControls.diisAlg != NONE) {
      for(auto i = 0; i < scfControls.nKeep; i++) {
        for(auto j = 0; j < this->fockMatrix.size(); j++) {
          memManager.free(diisFock[i][j]);
          memManager.free(diisOnePDM[i][j]);
          memManager.free(diisError[i][j]);
        } 
      }
    }

    // Deallocate memory to store previous orthonormal Fock for damping 
    for( auto &F : prevFock ) memManager.free(F);

  }; // SingleSlater<T>::deallocExtrapStorage



 /**
   *  \brief Form the orthonormal [F,D] commutator and store in diisError 
   *  
   *  The commutator is formed using FD and its adjoint
   *              [F,D] = FD - Adj(FD)
   *
   *  TODO: This routine only works for R/UHF right now
   *
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::FDCommutator(oper_t_coll &FDC) {

    size_t OSize = memManager.template getSize(fockMatrix[SCALAR]);
    size_t NB    = this->aoints.basisSet().nBasis;
    MatsT* SCR     = memManager.template malloc<MatsT>(nC*nC*NB*NB);
    bool iRO = (std::dynamic_pointer_cast<ROFock<MatsT,IntsT>>(fockBuilder) != nullptr);

    if(this->nC == 1) {
      // FD(S) = F(S)D(S)
      Gemm('N', 'N', NB, NB, NB, MatsT(1.), fockMatrixOrtho[SCALAR], NB, 
        onePDMOrtho[SCALAR], NB, MatsT(0.), FDC[SCALAR], NB);

      // FD(S) += F(z)D(z)
      if(nC == 2 or !iCS and !iRO) {
        Gemm('N', 'N', NB, NB, NB, MatsT(1.), fockMatrixOrtho[MZ], NB, 
          onePDMOrtho[MZ], NB, MatsT(0.), SCR, NB);
        MatAdd('N','N', NB, NB, MatsT(1.), FDC[SCALAR], NB, MatsT(1.), 
          SCR, NB, FDC[SCALAR], NB);
      }

      // Form {FD - DF}(S)
      std::copy_n(FDC[SCALAR],OSize,SCR);
      MatAdd('N','C', NB, NB, MatsT(1.), FDC[SCALAR], NB, MatsT(-1.), 
        SCR, NB, FDC[SCALAR], NB);


      if(nC == 2 or !iCS and !iRO) {
        // FD(z) = F(S)D(z) + F(z)D(S)
        Gemm('N', 'N', NB, NB, NB, MatsT(1.), fockMatrixOrtho[SCALAR], NB, 
          onePDMOrtho[MZ], NB, MatsT(0.), FDC[MZ], NB);
        Gemm('N', 'N', NB, NB, NB, MatsT(1.), fockMatrixOrtho[MZ], NB, 
          onePDMOrtho[SCALAR], NB, MatsT(0.), SCR, NB);
        MatAdd('N','N', NB, NB, MatsT(1.), FDC[MZ], NB, MatsT(1.), 
          SCR, NB, FDC[MZ], NB);

        // Form {FD - DF}(z)
        std::copy_n(FDC[MZ],OSize,SCR);
        MatAdd('N','C', NB, NB, MatsT(1.), FDC[MZ], NB, MatsT(-1.), 
          SCR, NB, FDC[MZ], NB);
      }
    } else {

      MatsT* FO = memManager.template malloc<MatsT>(4*NB*NB);
      MatsT* DO = memManager.template malloc<MatsT>(4*NB*NB);

      // Gather the orthonormal Fock and densities
      SpinGather(NB,FO,2*NB,fockMatrixOrtho[SCALAR],NB,fockMatrixOrtho[MZ],
        NB,fockMatrixOrtho[MY],NB,fockMatrixOrtho[MX],NB);
      SpinGather(NB,DO,2*NB,onePDMOrtho[SCALAR],NB,onePDMOrtho[MZ],
        NB,onePDMOrtho[MY],NB,onePDMOrtho[MX],NB);

      // Compute FD product
      Gemm('N','N',2*NB,2*NB,2*NB,MatsT(1.),FO,2*NB,DO,2*NB,MatsT(0.),SCR,2*NB);
      
      // Compute FD - DF (Store in FO scratch)
      MatAdd('N','C',2*NB,2*NB,MatsT(1.),SCR,2*NB,MatsT(-1.),SCR,2*NB,
        FO,2*NB);

      // Scatter Product into FDC
      SpinScatter(NB,FO,2*NB,FDC[SCALAR],NB,FDC[MZ],
        NB,FDC[MY],NB,FDC[MX],NB);

      memManager.free(FO,DO);
    }


    memManager.free(SCR);

  }; // SingleSlater<T>::FDCommutator





}; // namespace ChronusQ

