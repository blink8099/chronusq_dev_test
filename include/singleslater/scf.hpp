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
#include <util/matout.hpp>
#include <util/math.hpp>
#include <cqlinalg/blas1.hpp>
#include <cqlinalg/blasutil.hpp>
#include <cqlinalg/matfunc.hpp>

// SCF definitions for SingleSlaterBase
#include <singleslater/base/scf.hpp> 


namespace ChronusQ {

  /**
   *  \brief Saves the current state of wave function
   *
   *  Saves a copy of the current AO 1PDM and orthonormal Fock
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::saveCurrentState() {

    ROOT_ONLY(comm); 


    // Checkpoint if file exists
    if( savFile.exists() ) {

      size_t NB = this->nAlphaOrbital();
      size_t NBC = this->nC * NB;

      size_t t_hash = std::is_same<MatsT,double>::value ? 1 : 2;

      // Save Field type
      std::string prefix = "SCF/";
      if (this->particle.charge == 1.0)
        prefix = "PROT_" + prefix;
      
      savFile.safeWriteData(prefix + "FIELD_TYPE",&t_hash,{1});

      savFile.safeWriteData(prefix + "1PDM", *this->onePDM);

      savFile.safeWriteData(prefix + "FOCK", *fockMatrix);

      savFile.safeWriteData(prefix + "1PDM_ORTHO", *onePDMOrtho);

      savFile.safeWriteData(prefix + "FOCK_ORTHO", *fockMatrixOrtho);

      savFile.safeWriteData("SCF/ORTHO", ortho[0].pointer(), {NB,NB});

      savFile.safeWriteData("SCF/ORTHO_INV", ortho[1].pointer(), {NB,NB});

      // Save MOs
      savFile.safeWriteData(prefix + "MO1", this->mo[0].pointer(), {NBC,NBC});
      if( this->nC == 1 and not this->iCS )
        savFile.safeWriteData(prefix + "MO2", this->mo[1].pointer(), {NBC,NBC});

      // Save Energies
      savFile.safeWriteData(prefix + "TOTAL_ENERGY",&this->totalEnergy,
        {1});
      savFile.safeWriteData(prefix + "ONE_BODY_ENERGY",&this->OBEnergy,
        {1});
      savFile.safeWriteData(prefix + "MANY_BODY_ENERGY",&this->MBEnergy,
        {1});

      // Save Multipoles
      savFile.safeWriteData(prefix + "LEN_ELECTRIC_DIPOLE",&this->elecDipole[0],
        {3});
      savFile.safeWriteData(prefix + "LEN_ELECTRIC_QUADRUPOLE",
        &this->elecQuadrupole[0][0], {3,3});
      savFile.safeWriteData(prefix + "LEN_ELECTRIC_OCTUPOLE",
        &this->elecOctupole[0][0][0], {3,3,3});

      // Save Spin
      savFile.safeWriteData(prefix + "S_EXPECT",&this->SExpect[0],{3});
      savFile.safeWriteData(prefix + "S_SQUARED",&this->SSq,{1});
      

    // If file doesnt exist, checkpoint important bits in core
    } else {

      // Copy over current AO density matrix
      *curOnePDM = *this->onePDM;

      // Copy the previous orthonormal Fock matrix for damping. It's the 
      // previous Fock since saveCurrentState is called at the beginning 
      // of the SCF loop. 
      if ( scfControls.doExtrap and scfControls.doDamp) {
          
        // Avoid saving the guess Fock for extrapolation
        if (scfConv.nSCFIter > 0) {
          prevFock = fockMatrixOrtho;
        }
      }

    }

  }; // SingleSlater<MatsT>::saveCurrentState()


  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT, IntsT>::ConventionalSCF(bool modF) {

    // Transform AO fock into the orthonormal basis (on root MPI process)
    ao2orthoFock();

    // Modify fock matrix if requested (on root MPI process)
    if( modF ) modifyFock();

    // Diagonalize the orthonormal fock Matrix (on root MPI process)
    diagOrthoFock();

    // Form the orthonormal density (in the AO storage, 
    // updates all MPI processes)
    formDensity();

    // Copy the AO storage to orthonormal storage and back transform
    // the density into the AO basis. This is because ortho2aoDen
    // requires the onePDMOrtho storage is populated.
    //
    // *** Replicated on all MPI processes ***
    *onePDMOrtho = *this->onePDM;

    // Transform the orthonormal density to the AO basis 
    // (on root MPI process)
    ortho2aoDen();

    ortho2aoMOs();

  }

  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::NewtonRaphsonSCF() {

    MatsT* C = getNRCoeffs();

    // MO(:,i) = MO(:,i) + \sum_a C(a,i) MO(:,a)
    const size_t NB   = this->nAlphaOrbital();
    const size_t NB2  = NB * NB;
    const size_t NBC  = nC * NB;
    const size_t NBC2 = NBC * NBC;

    const size_t NO    = (nC == 2) ? this->nO : this->nOA;
    const size_t NV    = (nC == 2) ? this->nV : this->nVA;
    const size_t nOAVA = this->nOA * this->nVA;
    const size_t nOBVB = this->nOB * this->nVB;

    for(auto i = 0ul, ai = 0ul; i < NO;  i++)
    for(auto a = NO           ; a < NBC; a++, ai++) 
      AXPY(NBC,-C[ai],this->mo[0].pointer() + a*NBC,1,this->mo[0].pointer() + i*NBC,1);

    if( nC == 1 and not iCS )
      for(auto i = 0ul, ai = 0ul; i < this->nOB;  i++)
      for(auto a = this->nOB   ; a < NB; a++, ai++) 
        AXPY(NB,-C[ai + nOAVA],this->mo[1].pointer() + a*NB,1,this->mo[1].pointer() + i*NB,1);

    this->memManager.free(C);


    orthoAOMO();
    formDensity();

  }



  /**
   *  \brief Obtain a new set of orbitals given a Fock matrix.
   *
   *  Currently implements the fixed-point SCF procedure.
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::getNewOrbitals(EMPerturbation &pert, 
      bool frmFock) {

    bool increment = scfControls.doIncFock and 
                     scfConv.nSCFIter % scfControls.nIncFock != 0 and
                     scfControls.guess != RANDOM;

    // Form the Fock matrix D(k) -> F(k)
    if( frmFock ) {
      ProgramTimer::timeOp("Form Fock", [&](){
        formFock(pert,increment);
      });
    }

    if( scfControls.scfAlg == _NEWTON_RAPHSON_SCF and scfConv.nSCFIter > 0 ) {
      if(std::dynamic_pointer_cast<ROFock<MatsT,IntsT>>(fockBuilder) != nullptr)
        CErr("Newton Raphson SCF not implemented for ROHF",std::cout);
      else scfControls.scfStep = _NEWTON_RAPHSON_STEP;
    }

    if( scfControls.scfStep == _CONVENTIONAL_SCF_STEP )
      ConventionalSCF(scfControls.doExtrap and frmFock);
    else
      NewtonRaphsonSCF();

#ifdef CQ_ENABLE_MPI
    // Broadcast the AO 1PDM to all MPI processes
    if( MPISize(comm) > 1 ) {
      std::cerr  << "  *** Scattering the 1PDM ***\n";
      for(auto mat : this->onePDM->SZYXPointers())
        MPIBCast(mat,memManager.template getSize(mat),0,comm);
    }
#endif

  }; // SingleSlater<T>::getNewOrbitals



  template <typename MatsT, typename IntsT>
  bool SingleSlater<MatsT, IntsT>::checkStability() {

    double W;
    MatsT* J;
    std::tie(W,J) = this->getStab();
    std::cout << "  * LOWEST STABILITY EIGENVALUE " << 
      std::scientific << W << "\n";

    if( W < 0. and std::abs(W) > 1e-08 )
      std::cout << "  * LOWEST STABILITY EIGENVALUE NEGATIVE: " 
        << "PERFORMING THOULESS ROTATION\n";
    else { 

      std::cout << "  * LOWEST STABILITY EIGENVALUE POSITIVE: " 
        << "WAVE FUNCTION 2nd ORDER STABLE\n";

      this->memManager.free(J); return true; 

    }

    const size_t NB   = this->nAlphaOrbital();
    const size_t NB2  = NB * NB;
    const size_t NBC  = nC * NB;
    const size_t NBC2 = NBC * NBC;

    const size_t NO    = (nC == 2) ? this->nO : this->nOA;
    const size_t NV    = (nC == 2) ? this->nV : this->nVA;
    const size_t nOAVA = this->nOA * this->nVA;
    const size_t nOBVB = this->nOB * this->nVB;


    MatsT* ROT    = this->memManager.template malloc<MatsT>(NBC2);
    MatsT* EXPROT = this->memManager.template malloc<MatsT>(NBC2);
    std::fill_n(ROT,NBC2,0.);

    for(auto i = 0ul, ai = 0ul; i < NO;  i++)
    for(auto a = NO;            a < NBC; a++, ai++) {

      ROT[a + i*NBC] =  J[ai];
      ROT[i + a*NBC] = -SmartConj(J[ai]);

    }      

    // FIXME: need to generalize MatExp to take non-hermetian and real
    // matricies
    //MatExp('D',NBC,T(-1.),ROT,NBC,EXPROT,NBC,this->memManager);

    // Taylor
    MatsT s = 1.;
    std::copy_n(ROT,NBC2,EXPROT); // n = 1
    Scale(NBC2,-s,EXPROT,1);
    for(auto j = 0; j < NBC; j++) EXPROT[j*(NBC+1)] += 1.; // n = 0

    MatsT* SCR  = this->memManager.template malloc<MatsT>(NBC2);
    MatsT* SCR2 = this->memManager.template malloc<MatsT>(NBC2);
    std::copy_n(ROT,NBC2,SCR);

    size_t tayMax = 30; 
    for(auto n = 2; n <= tayMax; n++) {

      MatsT* M = nullptr;
      if( n % 2 ) {
        Gemm('N','N',NBC,NBC,NBC,MatsT(1.),ROT,NBC,SCR2,NBC,MatsT(0.),SCR,NBC);
        M = SCR;
      } else {
        Gemm('N','N',NBC,NBC,NBC,MatsT(1.),ROT,NBC,SCR,NBC,MatsT(0.),SCR2,NBC);
        M = SCR2;
      }

      MatsT fact = std::pow(-s,n)/factorial(n);
      MatAdd('N','N',NBC,NBC,MatsT(1.),EXPROT,NBC,fact,M,NBC,
        EXPROT,NBC);

    }

    /*
    Gemm('C','N',NBC,NBC,NBC,T(1.),EXPROT,NBC,EXPROT,NBC,T(0.),SCR,NBC);
   // prettyPrintSmart(std::cerr,"ROT",ROT,NBC,NBC,NBC);
   // prettyPrintSmart(std::cerr,"EXPROT",EXPROT,NBC,NBC,NBC);
    prettyPrintSmart(std::cout,"SCR",SCR,NBC,NBC,NBC);
   // CErr();
   */


    // MO1 = MO1 * EXPROT
    Gemm('N','N',NBC,NBC,NBC,MatsT(1.),this->mo[0].pointer(),NBC,EXPROT,NBC,MatsT(0.),
      ROT,NBC);
    std::copy_n(ROT,NBC2,this->mo[0].pointer());




    orthoAOMO();
    this->formDensity();
    this->formDelta();
    this->memManager.free(J,ROT,EXPROT);

    return false;

  }





  /**
   *  \brief Evaluate SCF convergence based on various criteria.
   *
   *  Checks the norm of [F,D], if converged -> SCF converged.
   *
   *  Checks change in energy and density between SCF iterations,
   *    if *both* converged -> SCF converged.
   */ 
  template <typename MatsT, typename IntsT>
  bool SingleSlater<MatsT, IntsT>::evalConver(EMPerturbation &pert) {

    bool isConverged;

    formDelta(); // Get change in density on all MPI processes

    // Compute all SCF convergence information on root process
    if( MPIRank(comm) == 0 ) {
      
      // Check energy convergence
        
      // Save copy of old Energy
      double oldEnergy = this->totalEnergy;

      // Compute new energy (with new Density)
      this->computeProperties(pert);
      scfConv.deltaEnergy = this->totalEnergy - oldEnergy;

      bool energyConv = std::abs(scfConv.deltaEnergy) < 
                        scfControls.eneConvTol;

      /*
      bool energySuperConv = std::abs(scfConv.deltaEnergy) < 
                             1e-2 * scfControls.eneConvTol;
                             */
      bool energySuperConv = false;



      // Check density convergence

      size_t NB    = basisSet().nBasis;
      size_t DSize = NB*NB;
      scfConv.RMSDenScalar = 
        TwoNorm<double>(DSize,deltaOnePDM->S().pointer(),1) / NB;
      scfConv.RMSDenMag = 0.;
      for(auto i = 1; i < deltaOnePDM->nComponent(); i++)
        scfConv.RMSDenMag += std::pow(TwoNorm<double>(DSize,
            (*deltaOnePDM)[static_cast<PAULI_SPINOR_COMPS>(i)].pointer(),1),2.);
 
      scfConv.RMSDenMag = std::sqrt(scfConv.RMSDenMag) / NB;
      
 
      bool denConv = scfConv.RMSDenScalar < scfControls.denConvTol;
 
      // Check FP convergence
      bool FDConv(false);
 
      isConverged = FDConv or (energyConv and denConv) or energySuperConv;
 
      // Toggle damping based on energy difference
      if( scfControls.doExtrap ) {
        // TODO: should enable print statements only when print flag is high 
        // enough 
        bool largeEDiff = 
          std::abs(scfConv.deltaEnergy) > scfControls.dampError;
 
        if( scfControls.doDamp and not largeEDiff and 
            scfControls.dampParam > 0.) {
 
          if( printLevel > 0 )
            std::cout << 
              "    *** Damping Disabled - Energy Difference Fell Below " <<
              scfControls.dampError << " ***" << std::endl;
 
          scfControls.dampParam = 0.;
 
        } else if( scfControls.doDamp and largeEDiff and 
                   scfControls.dampParam <= 0.) {
 
          if( printLevel > 0 )
            std::cout << "    *** Damping Enabled Due to "<<
              scfControls.dampError << " Oscillation in Energy ***" << std::endl;
 
          scfControls.dampParam = scfControls.dampStartParam;
 
        }
      }

    }


    // If converged and NR, check for saddle point and poke along
    // that direction ala https://doi.org/10.1063/1.4918561
    if( scfControls.scfAlg == _NEWTON_RAPHSON_SCF and isConverged )
      isConverged = checkStability();


#ifdef CQ_ENABLE_MPI
    // Broadcast whether or not we're converged to ensure that all
    // MPI processes exit the SCF simultaneously
    if( MPISize(comm) > 1 ) MPIBCast(isConverged,0,comm);
#endif

    return isConverged;

  }; // SingleSlater<MatsT>::evalConver





  /**
   *  \brief Computes the change in the current wave function
   *
   *  Saves onePDM - curOnePDM in deltaOnePDM
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::formDelta() {

    size_t NB = basisSet().nBasis;
    if (nC == 4) NB *= 2;

    // Compute difference on root MPI process
    if( MPIRank(comm) == 0 ) {
      if( not savFile.exists() )
        *deltaOnePDM = *this->onePDM - *curOnePDM;
      else {

        PauliSpinorSquareMatrices<MatsT> DENSCR(this->memManager, NB,
            this->onePDM->hasXY(), this->onePDM->hasZ());

        std::string prefix = "/SCF/";
        if (this->particle.charge == 1.0)
          prefix = "/PROT_SCF/";
        savFile.readData(prefix + "1PDM",DENSCR);

        //DENSCR.output(std::cout, "old_den", true);
        *deltaOnePDM = *this->onePDM - DENSCR;

      }
    }

#ifdef CQ_ENABLE_MPI
    // Broadcast the change in the 1PDM
    if( MPISize(comm) > 1 ) 
      for(auto &X : deltaOnePDM->SZYXPointers()) {
        MPIBCast(X,NB*NB,0,comm);
      }
#endif

  }; // SingleSlater<MatsT>:formDelta

  



  /**
   *  \brief Diagonalize the orthonormal fock matrix
   *
   *  General purpose routine which diagonalizes the orthonormal
   *  fock matrix and stores a set of orthonormal MO coefficients
   *  (in WaveFunction::mo1 and possibly WaveFunction::mo2) and
   *  orbital energies. General for both 1 and 2 spin components
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::diagOrthoFock() {

    ROOT_ONLY(comm); 
    size_t NB = this->nAlphaOrbital() * nC;
    size_t NB2 = NB*NB;
    bool iRO = (std::dynamic_pointer_cast<ROFock<MatsT,IntsT>>(fockBuilder) != nullptr);

    // Copy over the fockMatrixOrtho into MO storage
    if(nC == 1 and iCS) 
      this->mo = fockMatrixOrtho->template spinGatherToBlocks<MatsT>(false,false);
    else if(iRO)
      this->mo[0] = fockMatrixOrtho->S();
    else if(nC == 1)
      this->mo = fockMatrixOrtho->template spinGatherToBlocks<MatsT>(false);
    else {
      this->mo[0] = fockMatrixOrtho->template spinGather<MatsT>();

//    prettyPrintSmart(std::cout,"Orthormal Fock",this->mo[0].pointer(),NB,NB,NB);
    }

    // Diagonalize the Fock Matrix
    int INFO = HermetianEigen('V', 'L', NB, this->mo[0].pointer(), NB, this->eps1,
      memManager );
    if( INFO != 0 ) CErr("HermetianEigen failed in Fock1",std::cout);

    if(iRO) {
      this->mo[1] = this->mo[0]; // for ROHF
      std::copy_n(this->eps1, NB, this->eps2);
    } else if(nC == 1 and not iCS) {
      INFO = HermetianEigen('V', 'L', NB, this->mo[1].pointer(), NB, this->eps2,
        memManager );
      if( INFO != 0 ) CErr("HermetianEigen failed in Fock2",std::cout);
    }

#if 0
    printMO(std::cout);
#endif

  }; // SingleSlater<MatsT>::diagOrthoFock

  /**
   *  \brief Diagonalize the AO fock matrix
   *
   *  General purpose routine which diagonalizes the orthonormal
   *  fock matrix and stores a set of orthonormal MO coefficients
   *  (in WaveFunction::mo1 and possibly WaveFunction::mo2) and
   *  orbital energies. General for both 1 and 2 spin components
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::diagAOFock() {

    ROOT_ONLY(comm);
    size_t NB = this->nAlphaOrbital() * nC;
    size_t NB2 = NB*NB;
    bool iRO = (std::dynamic_pointer_cast<ROFock<MatsT,IntsT>>(fockBuilder) != nullptr);

    // Copy over the fockMatrix into MO storage
    if(nC == 1 and iCS)
      this->mo = fockMatrix->template spinGatherToBlocks<MatsT>(false,false);
    else if(iRO)
      this->mo[0] = fockMatrix->S();
    else if(nC == 1)
      this->mo = fockMatrix->template spinGatherToBlocks<MatsT>(false);
    else {
      this->mo[0] = fockMatrix->template spinGather<MatsT>();

    prettyPrintSmart(std::cout,"AO Fock",this->mo[0].pointer(),NB,NB,NB);

    }

    // Diagonalize the Fock Matrix
    int INFO = HermetianEigen('V', 'L', NB, this->mo[0].pointer(), NB, this->eps1,
      memManager );
    if( INFO != 0 ) CErr("HermetianEigen failed in Fock1",std::cout);

    if(iRO) {
      this->mo[1] = this->mo[0]; // for ROHF
      std::copy_n(this->eps1, NB, this->eps2);
    } else if(nC == 1 and not iCS) {
      INFO = HermetianEigen('V', 'L', NB, this->mo[1].pointer(), NB, this->eps2,
        memManager );
      if( INFO != 0 ) CErr("HermetianEigen failed in Fock2",std::cout);
    }

#if 0
    printMO(std::cout);
#endif

  }; // SingleSlater<MatsT>::diagOrthoFock


  /**
   *  \brief Transforms all of the spin components of the AO fock
   *  matrix to the orthonormal basis.
   *
   *  Populates / overwrites fockMatrixOrtho storage
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::ao2orthoFock() {

    ROOT_ONLY(comm);

    size_t NB = ortho[0].dimension();
    *fockMatrixOrtho = fockMatrix->transform(
          'N', ortho[0].pointer(), NB, NB);

  }; // SingleSlater<MatsT>::ao2orthoFock



  /**
   *  \brief Transforms all of the spin compoenents of the orthonormal
   *  1PDM to the AO basis.
   *
   *  Populates / overwrites onePDM storage
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::ortho2aoDen() {

    ROOT_ONLY(comm);

    size_t NB = ortho[0].dimension();
    *this->onePDM = onePDMOrtho->transform(
          'C', ortho[0].pointer(), NB, NB);

#if 0
    print1PDMOrtho(std::cout);
#endif

  }; // SingleSlater<MatsT>::ao2orthoFock

  
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::ortho2aoMOs() {

    size_t NB = this->nAlphaOrbital();

    // Transform MOs on MPI root as slave processes do not have
    // updated MO coefficients
    if( MPIRank(comm) == 0 ) {
      MatsT* SCR = this->memManager.template malloc<MatsT>(this->nC*NB*NB);

      // Transform the (top half) of MO1
      Gemm('N','N',NB,this->nC*NB,NB,MatsT(1.),this->ortho[0].pointer(),NB,
          this->mo[0].pointer(),this->nC*NB,MatsT(0.),SCR,NB);
      SetMat('N',NB,this->nC*NB,MatsT(1.),SCR,NB,this->mo[0].pointer(),this->nC*NB);
      
      if( this->nC == 2 ) {

        // Transform the bottom half of MO1
        Gemm('N','N',NB,this->nC*NB,NB,MatsT(1.),this->ortho[0].pointer(),NB,
            this->mo[0].pointer() + NB,this->nC*NB,MatsT(0.),SCR,NB);
        SetMat('N',NB,this->nC*NB,MatsT(1.),SCR,NB,this->mo[0].pointer() + NB,this->nC*NB);

      } else if( this->nC == 4 ) {

        //Second part of MO
        Gemm('N','N',NB,this->nC*NB,NB,MatsT(1.),this->ortho[0].pointer(),NB,
          this->mo[0].pointer() + NB,this->nC*NB,MatsT(0.),SCR,NB);
        SetMat('N',NB,this->nC*NB,MatsT(1.),SCR,NB,this->mo[0].pointer() + NB,this->nC*NB);

        //Third part of MO
        Gemm('N','N',NB,this->nC*NB,NB,MatsT(1.),this->ortho[0].pointer(),NB,
          this->mo[0].pointer() + 2*NB,this->nC*NB,MatsT(0.),SCR,NB);
        SetMat('N',NB,this->nC*NB,MatsT(1.),SCR,NB,this->mo[0].pointer() + 2*NB,this->nC*NB);

        //Fourth part of MO
        Gemm('N','N',NB,this->nC*NB,NB,MatsT(1.),this->ortho[0].pointer(),NB,
          this->mo[0].pointer() + 3*NB,this->nC*NB,MatsT(0.),SCR,NB);
        SetMat('N',NB,this->nC*NB,MatsT(1.),SCR,NB,this->mo[0].pointer() + 3*NB,this->nC*NB);
       

      } else if( not this->iCS ) {

        // Transform MO2
        Gemm('N','N',NB,NB,NB,MatsT(1.),this->ortho[0].pointer(),NB,
            this->mo[1].pointer(),NB,MatsT(0.),SCR,NB);
        SetMat('N',NB,NB,MatsT(1.),SCR,NB,this->mo[1].pointer(),NB);

      }

      this->memManager.free(SCR);

    }

#ifdef CQ_ENABLE_MPI


    // Broadcast the updated MOs to all MPI processes
    if( MPISize(comm) > 1 ) {

      std::cerr  << "  *** Scattering the AO-MOs ***\n";
      MPIBCast(this->mo[0].pointer(),nC*nC*NB*NB,0,comm);
      if( nC == 1 and not iCS )
        MPIBCast(this->mo[1].pointer(),nC*nC*NB*NB,0,comm);

      std::cerr  << "  *** Scattering EPS ***\n";
      MPIBCast(this->eps1,nC*NB,0,comm);
      if( nC == 1 and not iCS )
        MPIBCast(this->eps2,nC*NB,0,comm);

      std::cerr  << "  *** Scattering FOCK ***\n";
      for(MatsT *mat : fockMatrix->SZYXPointers())
        MPIBCast(mat,NB*NB,0,comm);

    }

#endif

    MOFOCK(); // Form the MO fock matrix

#if 0
    // Check proper orthonormalized wrt overlap

    T* SCR2 = this->memManager.template malloc<T>(this->nC*this->nC*NB*NB);
    T* SCR3 = this->memManager.template malloc<T>(this->nC*this->nC*NB*NB);


    // MO1 inner product
    Gemm('N','N',NB,this->nC*NB,NB,T(1.),this->aoints.overlap,NB,
      this->mo[0].pointer(),this->nC*NB,T(0.),SCR2,this->nC*NB);
    if(this->nC == 2)
      Gemm('N','N',NB,this->nC*NB,NB,T(1.),this->aoints.overlap,NB,
        this->mo[0].pointer()+NB,this->nC*NB,T(0.),SCR2+NB,this->nC*NB);
   
    Gemm('C','N',this->nC*NB,this->nC*NB,this->nC*NB,T(1.),this->mo[0].pointer(),
      this->nC*NB,SCR2,this->nC*NB,T(0.),SCR3,this->nC*NB);

    for(auto i = 0; i < this->nC*NB; i++)
      SCR3[i*(this->nC*NB + 1)] -= 1.;


    std::cerr << "Error in orthonormazation of MO1 = " 
      << MatNorm<double>('F',this->nC*NB,this->nC*NB,SCR3,this->nC*NB)
      << std::endl;
             


    if(this->nC == 1 and not this->iCS) {
      Gemm('N','N',NB,NB,NB,T(1.),this->aoints.overlap,NB,this->mo[1].pointer(),NB,T(0.),SCR2,NB);
      Gemm('C','N',NB,NB,NB,T(1.),this->mo[1].pointer(),NB,SCR2,NB,T(0.),SCR3,NB);

      for(auto i = 0; i < this->nC*NB; i++)
        SCR3[i*(this->nC*NB + 1)] -= 1.;

      std::cerr << "Error in orthonormazation of MO2 = " 
        << MatNorm<double>('F',NB,NB,SCR3,NB) << std::endl;
    }

    this->memManager.free(SCR2,SCR3);

#endif

  }; // SingleSlater<MatsT>::ortho2aoMOs


  /**
   *  \brief Initializes the environment for the SCF caluclation.
   *
   *  Allocate memory for extrapolation and compute the energy
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::SCFInit() {

    // Allocate additional storage if doing some type of 
    // extrapolation during the SCF procedure
    if ( scfControls.doExtrap ) allocExtrapStorage();

  }; // SingleSlater<MatsT>::SCFInit




  /**
   *  \brief Finalizes the environment for the SCF caluclation.
   *
   *  Deallocate the memory allocated for extrapolation.
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::SCFFin() {

    // Deallocate extrapolation storage
    if ( scfControls.doExtrap ) deallocExtrapStorage();

  }; // SingleSlater<MatsT>::SCFFin




  /**
   *  \brief Reorthogonalize the MOs wrt overlap
   */
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT, IntsT>::orthoAOMO() {

    const size_t NB   = this->nAlphaOrbital();
    const size_t NB2  = NB * NB;
    const size_t NBC  = this->nC * NB;
    const size_t NBC2 = NBC * NBC;

    if( this->nC == 4 ) CErr("orthoAOMO NYI for 4c",std::cout);

    // Transform MOs on MPI root as slave processes do not have
    // updated MO coefficients
    if( MPIRank(comm) == 0 ) {
      // Reorthogonalize MOs wrt S
      MatsT* SCR  = this->memManager.template malloc<MatsT>(NBC*NBC);
      MatsT* SCR2 = this->memManager.template malloc<MatsT>(NBC*NBC);

      // SCR2 = C**H S C
      Gemm('N','N',NB,NBC,NB,MatsT(1.),this->aoints.overlap->pointer(),NB,
           this->mo[0].pointer(),NBC,MatsT(0.),SCR,NBC);
      if( this->nC == 2 )
        Gemm('N','N',NB,NBC,NB,MatsT(1.),this->aoints.overlap->pointer(),NB,
             this->mo[0].pointer()+NB,NBC,MatsT(0.),SCR+NB,NBC);

      Gemm('C','N',NBC,NBC,NBC,MatsT(1.),this->mo[0].pointer(),NBC,SCR,NBC,MatsT(0.),SCR2,NBC);

      // SCR2 = L L**H -> L
      int INFO = Cholesky('L',NBC,SCR2,NBC);

      // SCR2 = L^-1
      INFO = TriInv('L','N',NBC,SCR2,NBC);

      // MO1 = MO1 * L^-H
      Trmm('R','L','C','N',NBC,NBC,MatsT(1.),SCR2,NBC,this->mo[0].pointer(),NBC);

      // Reorthogonalize MOB
      if( this->nC == 1 and not this->iCS ) {
        Gemm('N','N',NB,NB,NB,MatsT(1.),this->aoints.overlap->pointer(),NB,
             this->mo[1].pointer(),NB,MatsT(0.),SCR,NB);
        Gemm('C','N',NB,NB,NB,MatsT(1.),this->mo[1].pointer(),NBC,SCR,NB,MatsT(0.),SCR2,NB);

        // SCR2 = L L**H -> L
        INFO = Cholesky('L',NB,SCR2,NB);

        // SCR2 = L^-1
        INFO = TriInv('L','N',NB,SCR2,NB);

        // MO2 = MO2 * L^-H
        Trmm('R','L','C','N',NB,NB,MatsT(1.),SCR2,NB,this->mo[1].pointer(),NB);

      }

      this->memManager.free(SCR,SCR2);
    }

#ifdef CQ_ENABLE_MPI


    // Broadcast the updated MOs to all MPI processes
    if( MPISize(comm) > 1 ) {

      std::cerr  << "  *** Scattering the AO-MOs ***\n";
      MPIBCast(this->mo[0].pointer(),nC*nC*NB*NB,0,comm);
      if( nC == 1 and not iCS )
        MPIBCast(this->mo[1].pointer(),nC*nC*NB*NB,0,comm);

      std::cerr  << "  *** Scattering EPS ***\n";
      MPIBCast(this->eps1,nC*NB,0,comm);
      if( nC == 1 and not iCS )
        MPIBCast(this->eps2,nC*NB,0,comm);

      std::cerr  << "  *** Scattering FOCK ***\n";
      for(MatsT *mat : fockMatrix->SZYXPointers())
        MPIBCast(mat,NB*NB,0,comm);

    }

#endif

  }

}; // namespace ChronusQ

