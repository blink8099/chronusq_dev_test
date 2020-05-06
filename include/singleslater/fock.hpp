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

#include <singleslater.hpp>
#include <corehbuilder.hpp>
#include <fockbuilder.hpp>

#include <util/time.hpp>
#include <cqlinalg/blasext.hpp>

#include <cqlinalg.hpp>
#include <cqlinalg/svd.hpp>
#include <cqlinalg/blasutil.hpp>
#include <util/matout.hpp>
#include <util/threads.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>


//#define _DEBUGORTHO

namespace ChronusQ {

  /**
   *  \brief Forms the Fock matrix for a single slater determinant using
   *  the 1PDM.
   *
   *  \param [in] increment Whether or not the Fock matrix is being 
   *  incremented using a previous density
   *
   *  Populates / overwrites fock strorage
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::formFock(
    EMPerturbation &pert, bool increment, double xHFX) {

    fockBuilder->formFock(*this, pert, increment, xHFX);

  }; // SingleSlater::fockFock


  /**
   *  \brief Compute the Core Hamiltonian.
   *
   *  \param [in] typ Which Hamiltonian to build
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::formCoreH(EMPerturbation& emPert) {

    ROOT_ONLY(comm);

    if( coreH.size() != 0 )
      CErr("Recomputing the CoreH is not well-defined behaviour",std::cout);

    size_t NB = this->aoints.basisSet().nBasis;
    size_t nSQ  = NB*NB;

    coreH.emplace_back(memManager.malloc<MatsT>(nSQ));
    if(not iCS and nC == 1 and this->aoints.basisSet().basisType == COMPLEX_GIAO) 
      coreH.emplace_back(memManager.malloc<MatsT>(nSQ));
    else if(nC == 2) {
      coreH.emplace_back(memManager.malloc<MatsT>(nSQ));
      coreH.emplace_back(memManager.malloc<MatsT>(nSQ));
      coreH.emplace_back(memManager.malloc<MatsT>(nSQ));
    }


    // Compute core Hamiltonian
    coreHBuilder->computeCoreH(emPert,coreH);


    // Compute Orthonormalization trasformations
    computeOrtho();


    // Save the Core Hamiltonian
    if( savFile.exists() ) {

      size_t NB = this->aoints.basisSet().nBasis;
      const std::array<std::string,4> spinLabel =
        { "SCALAR", "MZ", "MY", "MX" };

      for(auto i = 0; i < coreH.size(); i++)
        savFile.safeWriteData("INTS/CORE_HAMILTONIAN_" +
          spinLabel[i], coreH[i], {NB,NB});

    }


  }; // SingleSlater<MatsT,IntsT>::computeCoreH


  /**
   *  \brief Allocate, compute and store the orthonormalization matricies 
   *  over the CGTO basis.
   *
   *  Computes either the Lowdin or Cholesky transformation matricies based
   *  on orthoType
   */ 
  template <typename MatsT, typename IntsT> 
  void SingleSlater<MatsT,IntsT>::computeOrtho() {

    size_t NB = this->aoints.basisSet().nBasis;
    size_t nSQ  = NB*NB;

    // Allocate orthogonalization matricies
    ortho1 = memManager.malloc<MatsT>(nSQ);
    ortho2 = memManager.malloc<MatsT>(nSQ);

    std::fill_n(ortho1,nSQ,0.);
    std::fill_n(ortho2,nSQ,0.);

    // Allocate scratch
    MatsT* SCR1 = memManager.malloc<MatsT>(nSQ);

    // Copy the overlap over to scratch space
    std::copy_n(this->aoints.overlap,nSQ,SCR1);

    if(orthoType == LOWDIN) {

      // Allocate more scratch
      MatsT* sE   = memManager.malloc<MatsT>(NB);
      MatsT* SCR2 = memManager.malloc<MatsT>(nSQ);

      
      // Diagonalize the overlap in scratch S = V * s * V**T
      HermetianEigen('V','U',NB,SCR1,NB,sE,memManager);


      if( std::abs( sE[0] ) < 1e-10 )
        CErr("Contracted Basis Set is Linearly Dependent!");

      // Compute X = V * s^{-1/2} 
      for(auto j = 0; j < NB; j++)
      for(auto i = 0; i < NB; i++)
        SCR2[i + j*NB] = 
          SCR1[i + j*NB] / std::sqrt(sE[j]);

      // Compute O1 = X * V**T
      Gemm('N','C',NB,NB,NB,
        static_cast<MatsT>(1.),SCR2,NB,SCR1,NB,
        static_cast<MatsT>(0.),ortho1,NB);


      // Compute X = V * s^{1/2} in place (by multiplying by s)
      for(auto j = 0; j < NB; j++)
      for(auto i = 0; i < NB; i++)
        SCR2[i + j*NB] = 
          SCR2[i + j*NB] * sE[j];

      // Compute O2 = X * V**T
      Gemm('N','C',NB,NB,NB,
        static_cast<MatsT>(1.),SCR2,NB,SCR1,NB,
        static_cast<MatsT>(0.),ortho2,NB);

#ifdef _DEBUGORTHO
      // Debug code to validate the Lowdin orthogonalization

      std::cerr << "Debugging Lowdin Orthogonalization" << std::endl;
      double maxDiff(-10000000);

      // Check that ortho1 and ortho2 are inverses of eachother
      Gemm('N','N',NB,NB,NB,
        static_cast<MatsT>(1.),ortho1,NB,ortho2,NB,
        static_cast<MatsT>(0.),SCR1,NB);
      
      for(auto j = 0; j < NB; j++)
      for(auto i = 0; i < NB; i++) {

        if( i == j ) maxDiff = 
          std::max(maxDiff, std::abs(1. - SCR1[i + j*NB]));
        else maxDiff = 
          std::max(maxDiff,std::abs(SCR1[i + j*NB])); 

      }

      std::cerr << "  Ortho1 * Ortho2 = I: " << maxDiff << std::endl;

      // Check that ortho2 * ortho2 is the overlap
      Gemm('N','N',NB,NB,NB,
        static_cast<MatsT>(1.),ortho2,NB,ortho2,NB,
        static_cast<MatsT>(0.),SCR1,NB);
      
      maxDiff = -100000;

      for(auto j = 0; j < NB; j++)
      for(auto i = 0; i < NB; i++) {

          maxDiff = std::max(maxDiff,
          std::abs(SCR1[i + j*NB] - 
            this->aoints.overlap[i + j*NB])); 

      }

      std::cerr << "  Ortho2 * Ortho2 = S: " << maxDiff << std::endl;

      // Check that ortho1 * ortho1 is the inverse of the overlap
      Gemm('N','N',NB,NB,NB,
        static_cast<MatsT>(1.),ortho1,NB,ortho1,NB,
        static_cast<MatsT>(0.),SCR1,NB);
      Gemm('N','N',NB,NB,NB,
        static_cast<MatsT>(1.),SCR1,NB,reinterpret_cast<MatsT*>(this->aoints.overlap),NB,
        static_cast<MatsT>(0.),SCR2,
        NB);
      
      maxDiff = -10000;
      for(auto j = 0; j < NB; j++)
      for(auto i = 0; i < NB; i++) {

        if( i == j ) maxDiff = 
          std::max(maxDiff, std::abs(1. - SCR2[i + j*NB]));
        else maxDiff = 
          std::max(maxDiff,std::abs(SCR2[i + j*NB])); 

      }

      std::cerr << "  Ortho1 * Ortho1 * S = I: " << maxDiff << std::endl;

#endif

      // Free Scratch Space
      memManager.free(sE,SCR2);

    } else if(orthoType == CHOLESKY) {

      std::cout << 
      "*** WARNING: Cholesky orthogonalization has not yet been confirmed ***" 
      << std::endl;

      // Compute the Cholesky factorization of the overlap S = L * L**T
      Cholesky('L',NB,SCR1,NB);

      // Copy the lower triangle to ortho2 (O2 = L)
      for(auto j = 0; j < NB; j++)
      for(auto i = j; i < NB; i++)
        ortho2[i + j*NB] = SCR1[i + j*NB];

      // Compute the inverse of the overlap using the Cholesky factors
      CholeskyInv('L',NB,SCR1,NB);

      // O1 = O2**T * S^{-1}
      Gemm('T','N',NB,NB,NB,
        static_cast<MatsT>(1.),ortho2,NB,SCR1,NB,
        static_cast<MatsT>(0.),ortho1,NB);

      // Remove upper triangle junk from O1
      for(auto j = 0; j < NB; j++)
      for(auto i = 0; i < j               ; i++)
        ortho1[i + j*NB] = 0.;

#ifdef _DEBUGORTHO
      // Debug code to validate the Lowdin orthogonalization

      std::cerr << "Debugging Cholesky Orthogonalization" << std::endl;

      // Debug code to validate the Cholesky orthogonalization
      MatsT* SCR2 = memManager.malloc<MatsT>(nSQ);
        
      double maxDiff = -1000;
      Gemm('T','N',NB,NB,NB,
        static_cast<MatsT>(1.),ortho1,NB,reinterpret_cast<MatsT*>(this->aoints.overlap),NB,
        static_cast<MatsT>(0.),SCR1,
        NB);
      Gemm('N','N',NB,NB,NB,
        static_cast<MatsT>(1.),SCR1,NB,ortho1,NB,
        static_cast<MatsT>(0.),SCR2,
        NB);

      for(auto j = 0; j < NB; j++)
      for(auto i = 0; i < NB; i++) {

        if( i == j ) maxDiff = 
          std::max(maxDiff, std::abs(1. - SCR2[i + j*NB]));
        else maxDiff = 
          std::max(maxDiff,std::abs(SCR2[i + j*NB])); 

      }

      std::cerr << "Ortho1**T * S ** Ortho1 = I: " << maxDiff << std::endl;

      memManager.free(SCR2); // Free SCR2
#endif
        

    }

    memManager.free(SCR1); // Free SCR1

  }; // computeOrtho

  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::MOFOCK() {

    const size_t NB   = this->aoints.basisSet().nBasis;
    const size_t NB2  = NB * NB;
    const size_t NBC  = nC * NB;
    const size_t NBC2 = NBC * NBC;


    if( fockMO.empty() ) {
      fockMO.emplace_back(memManager.template malloc<MatsT>(NBC2));
      if( nC == 1 and not iCS )
        fockMO.emplace_back(memManager.template malloc<MatsT>(NBC2));
    }

    if( MPIRank(comm) == 0 ) {
      MatsT* SCR1 = memManager.template malloc<MatsT>(NBC2);

      MatsT* FSTORAGE = fockMatrix[0];

      if( nC == 1 and not iCS ) {

        MatAdd('N','N',NB,NB,MatsT(1.),fockMatrix[0],NB,MatsT(1.) ,fockMatrix[1],NB,fockMO[0],NB);
        MatAdd('N','N',NB,NB,MatsT(1.),fockMatrix[0],NB,MatsT(-1.),fockMatrix[1],NB,fockMO[1],NB);

        FSTORAGE = fockMO[0];

      } else if( nC == 2 ) {

        SpinGather(NB,fockMO[0],NBC,fockMatrix[SCALAR],NB,fockMatrix[MZ],NB,
          fockMatrix[MY],NB,fockMatrix[MX],NB);

        FSTORAGE = fockMO[0];

      }

      
      MatsT tfact = (nC == 2) ? 1. : 0.5;

      Gemm('C','N',NBC,NBC,NBC,MatsT(1.),this->mo1,NBC,FSTORAGE,NBC,MatsT(0.),SCR1,NBC);
      Gemm('N','N',NBC,NBC,NBC,tfact,SCR1,NBC,this->mo1,NBC,MatsT(0.),
          fockMO[0],NBC);

      if( nC == 1 and not iCS ) {
        Gemm('C','N',NB,NB,NB,MatsT(1.),this->mo2,NB,fockMO[1],NB,MatsT(0.),SCR1,NB);
        Gemm('N','N',NB,NB,NB,tfact,SCR1,NB,this->mo2,NB,MatsT(0.),fockMO[1],NB);
      }

      memManager.free(SCR1);
    }

  //prettyPrintSmart(std::cout,"MOF",fockMO[0],NBC,NBC,NBC);

#ifdef CQ_ENABLE_MPI

    if(MPISize(comm) > 1) {
      std::cerr  << "  *** Scattering MO-FOCK ***\n";
      for(int k = 0; k < fockMO.size(); k++)
        MPIBCast(fockMO[k],NBC*NBC,0,comm);
    }

#endif
  };

}; // namespace ChronusQ

