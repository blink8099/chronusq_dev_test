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
#include <physcon.hpp>

#include <util/timer.hpp>
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

    ProgramTimer::tick("Form Core H");

    if( coreH != nullptr )
      CErr("Recomputing the CoreH is not well-defined behaviour",std::cout);

    size_t NB = this->basisSet().nBasis;
    if( nC == 4 ) NB = 2 * NB;

    if(not iCS and nC == 1 and this->basisSet().basisType == COMPLEX_GIAO)
      coreH = std::make_shared<PauliSpinorSquareMatrices<MatsT>>(memManager, NB, false);
    else if(nC == 2 or nC == 4)
      coreH = std::make_shared<PauliSpinorSquareMatrices<MatsT>>(memManager, NB, true);
    else
      coreH = std::make_shared<PauliSpinorSquareMatrices<MatsT>>(memManager, NB, false, false);


    // Make a copy of HamiltonianOptions
    HamiltonianOptions hamiltonianOptions = coreHBuilder->getHamiltonianOptions();

    // Prepare one-electron integrals
    std::vector<std::pair<OPERATOR,size_t>> ops;
    if (hamiltonianOptions.basisType == REAL_GTO)
      ops = {{OVERLAP,0}, {KINETIC,0}, {NUCLEAR_POTENTIAL,0},
             {LEN_ELECTRIC_MULTIPOLE,3},
             {VEL_ELECTRIC_MULTIPOLE,3}, {MAGNETIC_MULTIPOLE,2}};
    else
      ops = {{OVERLAP,0}, {KINETIC,0}, {NUCLEAR_POTENTIAL,0},
             {LEN_ELECTRIC_MULTIPOLE,3},
             {MAGNETIC_MULTIPOLE,1}};

    // Multipole integrals NYI for 4C
    if (nC == 4) ops.resize(3);

    // In case of X2C coreHBuilder, here we only compute
    // non-relativistic one electron integrals for contracted basis functions.
    // Relativistic integrals will be computed for uncontracted basis functions
    // in X2C CoreHBuilder.
    if (hamiltonianOptions.x2cType != X2C_TYPE::OFF) {
      hamiltonianOptions.OneEScalarRelativity = false;
      hamiltonianOptions.OneESpinOrbit = false;
    }

    this->aoints.computeAOOneP(memManager,this->molecule(),
        this->basisSet(),emPert, ops, hamiltonianOptions); // compute the necessary 1e ints

    // Compute core Hamiltonian
    coreHBuilder->computeCoreH(emPert,coreH);


    // Compute Orthonormalization trasformations
    computeOrtho();


    // Save the Core Hamiltonian
    if( savFile.exists() ) {

      const std::array<std::string,4> spinLabel =
        { "SCALAR", "MZ", "MY", "MX" };

      std::vector<MatsT*> CH(coreH->SZYXPointers());
      for(auto i = 0; i < CH.size(); i++)
        savFile.safeWriteData("INTS/CORE_HAMILTONIAN_" +
          spinLabel[i], CH[i], {NB,NB});

    }

    ProgramTimer::tock("Form Core H");

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

    size_t NB = this->basisSet().nBasis;
    if( nC == 4 ) NB = 2 * NB;
    size_t nSQ  = NB*NB;

    // Allocate orthogonalization matricies
    ortho[0].clear();
    ortho[1].clear();

    // Allocate scratch
    MatsT* SCR1 = memManager.malloc<MatsT>(nSQ);
    std::fill_n(SCR1,nSQ,0.);

    // Copy the overlap over to scratch space
    if ( nC != 4 ) {
      std::copy_n(this->aoints.overlap->pointer(),nSQ,SCR1);
    } else if( nC == 4 ) {
      // HBL 4C May need a Ints type check (SetMat) to capture GIAO.
      SetMatRE('N',NB/2,NB/2,1.,
               reinterpret_cast<double*>(this->aoints.overlap->pointer()),NB/2,
               SCR1,NB);
      SetMatRE('N',NB/2,NB/2,1./(2*SpeedOfLight*SpeedOfLight),
               reinterpret_cast<double*>(this->aoints.kinetic->pointer()),NB/2,
               SCR1+NB*NB/2+NB/2,NB);
      //prettyPrintSmart(std::cout,"S Metric",SCR1,NB,NB,NB);
    }

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
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::ConjTrans, 
                 NB, NB, NB, static_cast<MatsT>(1.0), SCR2, NB, SCR1, NB, static_cast<MatsT>(0.0), ortho[0].pointer(), NB);


      // Compute X = V * s^{1/2} in place (by multiplying by s)
      for(auto j = 0; j < NB; j++)
      for(auto i = 0; i < NB; i++)
        SCR2[i + j*NB] = 
          SCR2[i + j*NB] * sE[j];

      // Compute O2 = X * V**T
      blas::gemm(blas::Layout::ColMajor,blas::Op::NoTrans,blas::Op::ConjTrans,NB,NB,NB,
        static_cast<MatsT>(1.),SCR2,NB,SCR1,NB,
        static_cast<MatsT>(0.),ortho[1].pointer(),NB);

#ifdef _DEBUGORTHO
      // Debug code to validate the Lowdin orthogonalization

      std::cerr << "Debugging Lowdin Orthogonalization" << std::endl;
      double maxDiff(-10000000);

      // Check that ortho1 and ortho2 are inverses of eachother
      blas::gemm(blas::Layout::ColMajor,blas::Op::NoTrans,blas::Op::NoTrans,NB,NB,NB,
        static_cast<MatsT>(1.),ortho[0].pointer(),NB,ortho[1].pointer(),NB,
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
      blas::gemm(blas::Layout::ColMajor,blas::Op::NoTrans,blas::Op::NoTrans,NB,NB,NB,
        static_cast<MatsT>(1.),ortho[1].pointer(),NB,ortho[1].pointer(),NB,
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
      blas::gemm(blas::Layout::ColMajor,blas::Op::NoTrans,blas::Op::NoTrans,NB,NB,NB,
        static_cast<MatsT>(1.),ortho[0].pointer(),NB,ortho[0].pointer(),NB,
        static_cast<MatsT>(0.),SCR1,NB);
      blas::gemm(blas::Layout::ColMajor,blas::Op::NoTrans,blas::Op::NoTrans,NB,NB,NB,
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
      lapack::potrf(lapack::Uplo::Lower,NB,SCR1,NB);

      // Copy the lower triangle to ortho2 (O2 = L)
      for(auto j = 0; j < NB; j++)
      for(auto i = j; i < NB; i++)
        ortho[1](i,j) = SCR1[i + j*NB];

      // Compute the inverse of the overlap using the Cholesky factors
      lapack::potri(lapack::Uplo::Lower,NB,SCR1,NB);

      // O1 = O2**T * S^{-1}
      blas::gemm(blas::Layout::ColMajor,blas::Op::Trans,blas::Op::NoTrans,NB,NB,NB,
        static_cast<MatsT>(1.),ortho[1].pointer(),NB,SCR1,NB,
        static_cast<MatsT>(0.),ortho[0].pointer(),NB);

      // Remove upper triangle junk from O1
      for(auto j = 0; j < NB; j++)
      for(auto i = 0; i < j ; i++)
        ortho[0](i,j) = 0.;

#ifdef _DEBUGORTHO
      // Debug code to validate the Lowdin orthogonalization

      std::cerr << "Debugging Cholesky Orthogonalization" << std::endl;

      // Debug code to validate the Cholesky orthogonalization
      MatsT* SCR2 = memManager.malloc<MatsT>(nSQ);
        
      double maxDiff = -1000;
      blas::gemm(blas::Layout::ColMajor,blas::Op::Trans,blas::Op::NoTrans,NB,NB,NB,
        static_cast<MatsT>(1.),ortho1,NB,reinterpret_cast<MatsT*>(this->aoints.overlap),NB,
        static_cast<MatsT>(0.),SCR1,
        NB);
      blas::gemm(blas::Layout::ColMajor,blas::Op::NoTrans,blas::Op::NoTrans,NB,NB,NB,
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

    const size_t NB   = this->nAlphaOrbital();
    const size_t NBC  = nC * NB;

    if( fockMO.empty() ) {
      fockMO.emplace_back(memManager, NBC);
      if( nC == 1 and not iCS )
        fockMO.emplace_back(memManager, NBC);
    }

    if( MPIRank(comm) == 0 ) {

      if ( nC == 2 )
        fockMO[0] = fockMatrix->template spinGather<MatsT>();
      else if ( nC == 1 )
        fockMO = fockMatrix->template spinGatherToBlocks<MatsT>(false, not iCS);

      fockMO[0] = fockMO[0].transform('N',this->mo[0].pointer(),NBC,NBC);

      if( nC == 1 and not iCS ) {

        fockMO[1] = fockMO[1].transform('N',this->mo[1].pointer(),NBC,NBC);

      }

    }

  //prettyPrintSmart(std::cout,"MOF",fockMO[0].pointer(),NBC,NBC,NBC);

#ifdef CQ_ENABLE_MPI

    if(MPISize(comm) > 1) {
      std::cerr  << "  *** Scattering MO-FOCK ***\n";
      for(int k = 0; k < fockMO.size(); k++)
        MPIBCast(fockMO[k].pointer(),NBC*NBC,0,comm);
    }

#endif
  };

}; // namespace ChronusQ

