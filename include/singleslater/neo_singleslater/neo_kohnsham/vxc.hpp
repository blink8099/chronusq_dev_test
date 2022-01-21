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

#include <singleslater/kohnsham.hpp>
#include <singleslater/neo_singleslater/neo_kohnsham.hpp>

#include <grid/integrator.hpp>
#include <basisset/basisset_util.hpp>
#include <cqlinalg/blasutil.hpp>
#include <cqlinalg/blasext.hpp>

#include <util/threads.hpp>

namespace ChronusQ {

  /**
   *  \brief assemble the VXC for all density component over batch 
   *  of points
   *
   *  It handles submatrix of the VXC (for a given
   *  subset of shell to be evaluated for the provided batch)
   *  and assemable them
   *
   *  This function is integrated by the BeckeIntegrator
   *  object
   */
  template <typename MatsT, typename IntsT>
  void NEOKohnSham<MatsT,IntsT>::formVXC() {

    
#if VXC_DEBUG_LEVEL >= 1
    // TIMING 
    auto topMem = std::chrono::high_resolution_clock::now();
#endif

    assert( this->intParam.nRad % this->intParam.nRadPerBatch == 0 );




    // Parallelism

    size_t nthreads = GetNumThreads();
    size_t LAThreads = GetLAThreads();
    size_t mpiRank   = MPIRank(this->comm);
    size_t mpiSize   = MPISize(this->comm);

    // MPI Communicator for numerical integration
    // *** Assumes that MPI will be done only across atoms in integration

    size_t nAtoms = this->molecule().nAtoms;
    int color = ((mpiSize < nAtoms) or 
                 (mpiRank < nAtoms)) ? 1 : MPI_UNDEFINED;
                                                            
                  
    MPI_Comm intComm = MPICommSplit(this->comm,color,mpiRank);







#ifdef CQ_ENABLE_MPI
    if( intComm != MPI_COMM_NULL ) 
#endif
    {
  
  
  
  
  
  
      // Define several useful quantities for later on
      size_t NPtsMaxPerBatch = this->intParam.nRadPerBatch * this->intParam.nAng;
  
      bool isGGA = std::any_of(this->functionals.begin(),this->functionals.end(),
                     [](std::shared_ptr<DFTFunctional> &x) {
                       return x->isGGA(); 
                     }); 

      bool epcisGGA = std::any_of(epc_functionals.begin(),epc_functionals.end(),
                       [](std::shared_ptr<DFTFunctional> &x) {
                       return x->isGGA(); 
                     }); 
  
      // Turn off LA threads
      SetLAThreads(1);
  
      BasisSet &basis = this->basisSet();
      size_t NB     = basis.nBasis;
      size_t NB2    = NB*NB;
      size_t NTNB2  = nthreads * NB2;
      size_t NTNPPB = nthreads*NPtsMaxPerBatch;

      // Auxiliary basis set information for NEO-DFT
      BasisSet &aux_basis = this->aux_neoss->basisSet();
      size_t aux_NB    = aux_basis.nBasis;
      size_t aux_NB2   = aux_NB*aux_NB;
      size_t aux_NTNB2 = nthreads * aux_NB2;


      // Clean up all VXC components for a the evaluation for a new 
      // batch of points
  
      std::vector<std::vector<double*>> integrateVXC;
      double* intVXC_RAW = (nthreads == 1) ? nullptr :
        this->memManager.template malloc<double>(this->VXC->nComponent() * NTNB2);
  
      std::vector<double*> VXC_SZYX = this->VXC->SZYXPointers(); 
      for(auto k = 0; k < VXC_SZYX.size(); k++) {
        integrateVXC.emplace_back();

        if( nthreads != 1 ) 
          for(auto ith = 0; ith < nthreads; ith++)
            integrateVXC.back().emplace_back(
              intVXC_RAW + (ith + k*nthreads) * NB2
            );

        else 
          integrateVXC.back().emplace_back(VXC_SZYX[k]);

      }
      
      for(auto &X : integrateVXC) for(auto &Y : X) std::fill_n(Y,NB2,0.);
  
      std::vector<double> integrateXCEnergy(nthreads,0.);
  
      // Start Debug quantities
#if VXC_DEBUG_LEVEL >= 3
      // tmp VXC submat
      double *tmpS      = this->memManager.template malloc<double>(NB2); 
      std::fill_n(tmpS,basis.nBasis*basis.nBasis,0.);
#endif

#if VXC_DEBUG_LEVEL >= 2
      double sumrho   = 0.;
      double sumspin  = 0.;
      double sumgamma = 0.;
#endif
      // END Debug quantities
 
      // Allocating Memory
      // ----------------------------------------------------------//
    
      this->XCEnergy = 0.;
      EPCEnergy = 0.;
      double *SCRATCHNBNB = 
        this->memManager.template malloc<double>(NTNB2); 
      double *SCRATCHNBNP = 
        this->memManager.template malloc<double>(NTNPPB*NB); 

      double *DenS, *DenZ, *DenX ;
      DenS = this->memManager.template malloc<double>(NTNPPB);

      if( this->onePDM->hasZ() )
        DenZ = this->memManager.template malloc<double>(NTNPPB);

      if( this->onePDM->hasXY() )
        CErr("Relativistic NEO-Kohn-Sham NYI!", std::cout);

      double *epsEval = this->memManager.template malloc<double>(NTNPPB);

      // Density U-Variables
      double *U_n   = 
        this->memManager.template malloc<double>(2*NTNPPB); 
      double *dVU_n = 
        this->memManager.template malloc<double>(2*NTNPPB); 

      double *ZrhoVar1, *ZgammaVar1, *ZgammaVar2;

      ZrhoVar1 = this->memManager.template malloc<double>(NTNPPB);
      
      // These quantities are only used for GGA functionals
      double *GDenS, *GDenZ, *U_gamma, *dVU_gamma;
      if( isGGA ) {
        GDenS = this->memManager.template malloc<double>(3*NTNPPB);
        ZgammaVar1 = this->memManager.template malloc<double>(NTNPPB);
        ZgammaVar2 = this->memManager.template malloc<double>(NTNPPB);

        if( this->onePDM->hasZ() )
          GDenZ = this->memManager.template malloc<double>(3*NTNPPB);

        if( this->onePDM->hasXY() )
          CErr("Relativistic NEO-Kohn-Sham NYI!", std::cout);

        // Gamma U-Variables
        U_gamma = this->memManager.template malloc<double>(3*NTNPPB); 
        dVU_gamma = this->memManager.template malloc<double>(3*NTNPPB); 
      }

      // These quantities are used when there are multiple xc functionals
      double *epsSCR, *dVU_n_SCR, *dVU_gamma_SCR;
      if( this->functionals.size() > 1 ) {
        epsSCR    = this->memManager.template malloc<double>(NTNPPB);
        dVU_n_SCR    = this->memManager.template malloc<double>(2*NTNPPB);
        if(isGGA) 
          dVU_gamma_SCR = 
            this->memManager.template malloc<double>(3*NTNPPB);
      }

      // ZMatrix
      double *ZMAT = this->memManager.template malloc<double>(NTNPPB*NB);
 
      // Decide if we need to allocate space for real part of the 
      // densities and copy over the real parts
      std::shared_ptr<PauliSpinorSquareMatrices<double>> Re1PDM;
      if (std::is_same<MatsT,double>::value)
        Re1PDM = std::dynamic_pointer_cast<PauliSpinorSquareMatrices<double>>(
            this->onePDM);
      else
        Re1PDM = std::make_shared<PauliSpinorSquareMatrices<double>>(
           this->onePDM->real_part());


      // Scratch pointers for auxiliary systems 
      // ---------------NEO------------------------------------------------
      double *AUX_SCRATCHNBNB = 
        this->memManager.template malloc<double>(aux_NTNB2);
      double *AUX_SCRATCHNBNP = 
        this->memManager.template malloc<double>(NTNPPB*aux_NB);        

      // Density pointers for auxiliary system
      double *aux_DenS, *aux_DenZ;
      aux_DenS = this->memManager.template malloc<double>(NTNPPB);

        if ( this->aux_neoss->onePDM->hasZ() )
          aux_DenZ = this->memManager.template malloc<double>(NTNPPB);

        if ( this->aux_neoss->onePDM->hasXY() )
          CErr("Relativistic NEO-Kohn-Sham NYI!", std::cout);

      // NEO auxiliary Density U-Variables
      double *aux_U_n   = 
        this->memManager.template malloc<double>(2*NTNPPB);
      double *aux_dVU_n  = 
        this->memManager.template malloc<double>(2*NTNPPB);

      double *aux_ZrhoVar1, *aux_ZgammaVar1, *aux_ZgammaVar2;
      // These quantities are only used for GGA functionals
      double *aux_GDenS, *aux_GDenZ, *aux_U_gamma, *aux_dVU_gamma;

      aux_ZrhoVar1 = this->memManager.template malloc<double>(NTNPPB);

      if( epcisGGA ) {
        aux_GDenS = this->memManager.template malloc<double>(3*NTNPPB);
        aux_ZgammaVar1 = this->memManager.template malloc<double>(NTNPPB);
        aux_ZgammaVar2 = this->memManager.template malloc<double>(NTNPPB);

        if( this->aux_neoss->onePDM->hasZ() )
          aux_GDenZ = this->memManager.template malloc<double>(3*NTNPPB);

        if( this->aux_neoss->onePDM->hasXY() )
          CErr("Relativistic NEO-Kohn-Sham NYI!", std::cout);

        // Gamma U-Variables
        aux_U_gamma = this->memManager.template malloc<double>(3*NTNPPB); 
        aux_dVU_gamma = this->memManager.template malloc<double>(3*NTNPPB); 
      }

      double *aux_epsSCR, *aux_dVU_n_SCR, *aux_dVU_gamma_SCR;
      if( this->functionals.size() > 1 ) {
        aux_epsSCR    = this->memManager.template malloc<double>(NTNPPB);
        aux_dVU_n_SCR    = this->memManager.template malloc<double>(2*NTNPPB);
        if(epcisGGA) 
          aux_dVU_gamma_SCR = 
            this->memManager.template malloc<double>(3*NTNPPB);
      }

      std::shared_ptr<PauliSpinorSquareMatrices<double>> aux_Re1PDM;
      if( std::is_same<MatsT,double>::value )
         aux_Re1PDM = std::dynamic_pointer_cast<PauliSpinorSquareMatrices<double>>(
           this->aux_neoss->onePDM);
       else
         aux_Re1PDM = std::make_shared<PauliSpinorSquareMatrices<double>>(
           this->aux_neoss->onePDM->real_part());


      // ---------------end NEO------------------------------------------------

      // --------------Cross Memory--------------------------------------------
      double *cross_U_gamma, *cross_dVU_gamma, *ZgammaVar3;
      if (isGGA and epcisGGA) {
        cross_U_gamma = this->memManager.template malloc<double>(4*NTNPPB); 
        cross_dVU_gamma = this->memManager.template malloc<double>(4*NTNPPB);
        ZgammaVar3 = this->memManager.template malloc<double>(NTNPPB);
      }
      // --------------end cross-----------------------------------------------
 

      // -------------------------------------------------------------//
      // End allocating Memory

#if VXC_DEBUG_LEVEL >= 1
      // TIMING
      auto botMem = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> durevalDen(0.)  ;
      std::chrono::duration<double> durmkAuxVar(0.) ;
      std::chrono::duration<double> durloadVXCder(0.) ;
      std::chrono::duration<double> durenergy_vxc(0.) ;
      std::chrono::duration<double> durconstructZVars(0.) ;
      std::chrono::duration<double> durformZ_vxc(0.) ;
      std::chrono::duration<double> durDSYR2K(0.) ;
      std::chrono::duration<double> durIncBySubMat(0.) ;
#endif

      auto vxcbuild = [&](size_t &res, std::vector<cart_t> &batch, 
        std::vector<double> &weights, std::vector<size_t> NBE_vec, 
        std::vector<double*> BasisEval_vec, 
        std::vector<std::vector<size_t>> & batchEvalShells_vec, 
        std::vector<std::vector<std::pair<size_t,size_t>>> & subMatCut_vec) {

#if VXC_DEBUG_LEVEL > 3
        prettyPrintSmart(std::cout,"BASIS SCR",BasisEval,NBE,
          4*batch.size(),NBE);
#endif

        // main system
        size_t NBE = NBE_vec[0];
        double * BasisEval = BasisEval_vec[0];
        std::vector<size_t> & batchEvalShells = batchEvalShells_vec[0];
        std::vector<std::pair<size_t,size_t>> & subMatCut = subMatCut_vec[0];

        // auxiliary system
        size_t aux_NBE = NBE_vec[1];
        double * aux_BasisEval = BasisEval_vec[1];
        std::vector<size_t> & aux_batchEvalShells = batchEvalShells_vec[1];
        std::vector<std::pair<size_t,size_t>> & aux_subMatCut = subMatCut_vec[1];


        // intParam.epsilon / ntotalpts (NANG * NRAD * NATOMS)
        double epsScreen = this->intParam.epsilon / nAtoms /
          this->intParam.nAng / this->intParam.nRad;

        epsScreen = std::max(epsScreen,
                             std::numeric_limits<double>::epsilon());

        size_t NPts = batch.size();
        size_t IOff = NBE*NPts;

        size_t thread_id = GetThreadID();
        size_t TIDNPPB   = thread_id * NPtsMaxPerBatch;

#if VXC_DEBUG_LEVEL >= 1
        // TIMING
        auto topevalDen = std::chrono::high_resolution_clock::now();
#endif

        // --------------Main System------------------------------------------
        // Setup local pointers
        double * SCRATCHNBNB_loc = SCRATCHNBNB + thread_id * NB2;
        double * SCRATCHNBNP_loc = SCRATCHNBNP + thread_id * NB*NPtsMaxPerBatch;

        // local density pointers 
        double * DenS_loc = DenS + TIDNPPB;
        double * DenZ_loc = DenZ + TIDNPPB;

        // local density gradient pointers
        double * GDenS_loc = GDenS + 3*TIDNPPB;
        double * GDenZ_loc = GDenZ + 3*TIDNPPB;

        // local vxc energy and U, V pointer
        double * epsEval_loc   = epsEval   +  TIDNPPB;
        double * U_n_loc       = U_n       + 2*TIDNPPB;
        double * dVU_n_loc     = dVU_n     + 2*TIDNPPB;
        double * U_gamma_loc   = U_gamma   + 3*TIDNPPB;
        double * dVU_gamma_loc = dVU_gamma + 3*TIDNPPB;

        // local gradient pointer
        double * ZrhoVar1_loc   = ZrhoVar1   + TIDNPPB;
        double * ZgammaVar1_loc = ZgammaVar1 + TIDNPPB;
        double * ZgammaVar2_loc = ZgammaVar2 + TIDNPPB;

        // local multiple functional pointer
        double * epsSCR_loc        = epsSCR        +   TIDNPPB;
        double * dVU_n_SCR_loc     = dVU_n_SCR     + 2*TIDNPPB;
        double * dVU_gamma_SCR_loc = dVU_gamma_SCR + 3*TIDNPPB;

        // local Zmat
        double *ZMAT_loc = ZMAT + NB * TIDNPPB;

        // ---------------End Main System------------------------------------


        // ---------------Auxiliary System-----------------------------------
        // aux local scratch pointers 
        double * AUX_SCRATCHNBNB_loc, * AUX_SCRATCHNBNP_loc;
        AUX_SCRATCHNBNB_loc = AUX_SCRATCHNBNB + thread_id * aux_NB2;
        AUX_SCRATCHNBNP_loc = AUX_SCRATCHNBNP + thread_id * aux_NB*NPtsMaxPerBatch;         

        // aux local density pointers
        double * aux_DenS_loc, * aux_DenZ_loc;
        aux_DenS_loc = aux_DenS + TIDNPPB;
        aux_DenZ_loc = aux_DenZ + TIDNPPB;

        // aux local density gradient pointers
        double *aux_GDenS_loc, *aux_GDenZ_loc;
        aux_GDenS_loc = aux_GDenS + 3*TIDNPPB;
        aux_GDenZ_loc = aux_GDenZ + 3*TIDNPPB;

        // aux local U and V pointers
        double * aux_U_n_loc, * aux_dVU_n_loc, * aux_U_gamma_loc, * aux_dVU_gamma_loc;
        aux_U_n_loc       = aux_U_n       + 2*TIDNPPB;
        aux_dVU_n_loc     = aux_dVU_n     + 2*TIDNPPB;
        aux_U_gamma_loc   = aux_U_gamma   + 3*TIDNPPB;
        aux_dVU_gamma_loc = aux_dVU_gamma + 3*TIDNPPB;

        // aux local Z pointers
        double * aux_ZrhoVar1_loc, * aux_ZgammaVar1_loc, * aux_ZgammaVar2_loc;        
        aux_ZrhoVar1_loc   = aux_ZrhoVar1   + TIDNPPB;
        aux_ZgammaVar1_loc = aux_ZgammaVar1 + TIDNPPB;
        aux_ZgammaVar2_loc = aux_ZgammaVar2 + TIDNPPB;

        // aux local multiple functional pointers
        double * aux_epsSCR_loc, * aux_dVU_n_SCR_loc, * aux_dVU_gamma_SCR_loc;
        aux_epsSCR_loc        = aux_epsSCR        +   TIDNPPB;
        aux_dVU_n_SCR_loc     = aux_dVU_n_SCR     + 2*TIDNPPB;
        aux_dVU_gamma_SCR_loc = aux_dVU_gamma_SCR + 3*TIDNPPB;

        // ---------------End Auxiliary System--------------------------------

        // ---------------Cross Between Main and Auxiliary system-------------
        double * cross_U_gamma_loc, *cross_dVU_gamma_loc, *ZgammaVar3_loc;
        if (isGGA and epcisGGA) {
          cross_U_gamma_loc = cross_U_gamma + 4*TIDNPPB;
          cross_dVU_gamma_loc = cross_dVU_gamma + 4*TIDNPPB;
          ZgammaVar3_loc = ZgammaVar3 + TIDNPPB;
        }
        // ---------------End Cross-------------------------------------------

        // This evaluates the V variables for all components of the main system
        // (Scalar, MZ (UKS) and Mx, MY (2 Comp))
        evalDen((isGGA ? GRADIENT : NOGRAD), NPts, NBE, NB, subMatCut, 
            SCRATCHNBNB_loc, SCRATCHNBNP_loc, Re1PDM->S().pointer(), DenS_loc, 
            GDenS_loc, GDenS_loc + NPts, GDenS_loc + 2*NPts, BasisEval);

        // evaluate density for auxiliary components
        evalDen((epcisGGA ? GRADIENT : NOGRAD), NPts, aux_NBE, 
          aux_NB, aux_subMatCut, 
          AUX_SCRATCHNBNB_loc, AUX_SCRATCHNBNP_loc, aux_Re1PDM->S().pointer(), aux_DenS_loc,
          aux_GDenS_loc, aux_GDenS_loc + NPts, aux_GDenS_loc + 2*NPts, aux_BasisEval);


#if VXC_DEBUG_LEVEL < 3
        // Coarse screen on Density
        double MaxDenS_loc = *std::max_element(DenS_loc,DenS_loc+NPts);
        if (MaxDenS_loc < epsScreen) { return; }
#endif

        if ( this->onePDM->hasZ() )
          evalDen((isGGA ? GRADIENT : NOGRAD), NPts, NBE, NB, subMatCut, 
                SCRATCHNBNB_loc ,SCRATCHNBNP_loc, Re1PDM->Z().pointer(), DenZ_loc, 
                GDenZ_loc, GDenZ_loc + NPts, GDenZ_loc + 2*NPts, BasisEval);

        if ( this->aux_neoss->onePDM->hasZ() )
          evalDen((epcisGGA ? GRADIENT : NOGRAD), NPts, aux_NBE, 
            aux_NB, aux_subMatCut, 
            AUX_SCRATCHNBNB_loc, AUX_SCRATCHNBNP_loc, aux_Re1PDM->Z().pointer(), 
            aux_DenZ_loc,
            aux_GDenZ_loc, aux_GDenZ_loc + NPts, aux_GDenZ_loc + 2*NPts, aux_BasisEval);

        if ( this->onePDM->hasXY() )
          CErr("Relativistic NEO-Kohn-Sham NYI!");

        if ( this->aux_neoss->onePDM->hasXY() )
          CErr("Relativistic NEO-Kohn-Sham NYI!");


#if VXC_DEBUG_LEVEL >= 1
        // TIMING
        auto botevalDen  = std::chrono::high_resolution_clock::now();
        auto topmkAuxVar = std::chrono::high_resolution_clock::now();
#endif
        
        // V -> U variables for NEO-DFT kernal derivatives 
        mkAuxVar(aux_neoks->onePDM,
          epcisGGA,epsScreen,NPts,
          aux_DenS_loc,aux_DenZ_loc,nullptr,nullptr,
          aux_GDenS_loc,aux_GDenS_loc + NPts,aux_GDenS_loc + 2*NPts,
          aux_GDenZ_loc,aux_GDenZ_loc + NPts,aux_GDenZ_loc + 2*NPts,
          nullptr,nullptr,nullptr,
          nullptr,nullptr,nullptr,
          nullptr,
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, 
          aux_U_n_loc, aux_U_gamma_loc
        );


        // V -> U variables for evaluating the kernel derivatives.
        mkAuxVar(this->onePDM,
          isGGA,epsScreen,NPts,
          DenS_loc,DenZ_loc,nullptr,nullptr,
          GDenS_loc,GDenS_loc + NPts,GDenS_loc + 2*NPts,
          GDenZ_loc,GDenZ_loc + NPts,GDenZ_loc + 2*NPts,
          nullptr,nullptr,nullptr,
          nullptr,nullptr,nullptr,
          nullptr, 
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, 
          U_n_loc,U_gamma_loc
        );


        // Cross V -> U variables
        if (isGGA and epcisGGA)
          mkCrossAuxVar(false,this->particle.charge < 0,
            this->onePDM, aux_neoks->onePDM, epsScreen,NPts,
            GDenS_loc,GDenS_loc + NPts,GDenS_loc + 2*NPts,
            nullptr,nullptr,nullptr,
            nullptr,nullptr,nullptr,
            GDenZ_loc,GDenZ_loc + NPts,GDenZ_loc + 2*NPts,
            aux_GDenS_loc,aux_GDenS_loc + NPts,aux_GDenS_loc + 2*NPts,
            nullptr,nullptr,nullptr,
            nullptr,nullptr,nullptr,
            aux_GDenZ_loc,aux_GDenZ_loc + NPts,aux_GDenZ_loc + 2*NPts,
            cross_U_gamma_loc);



#if VXC_DEBUG_LEVEL >= 1
        // TIMING
        auto botmkAuxVar = std::chrono::high_resolution_clock::now();
#endif


#if VXC_DEBUG_LEVEL >= 2
        assert(nthreads == 1);
        // Debug int
        for(auto iPt = 0; iPt < NPts; iPt++) { 
          sumrho  += weights[iPt] * (U_n[2*iPt] + U_n[2*iPt + 1]);
          sumspin += weights[iPt] * (U_n[2*iPt] - U_n[2*iPt + 1]);
          if(isGGA) 
            sumgamma += weights[iPt] * 
              ( U_gamma[3*iPt] + U_gamma[3*iPt+1] + U_gamma[3*iPt+2]);
        };
        // end debug
#endif
      
#if VXC_DEBUG_LEVEL >= 1
        // TIMING
        auto toploadVXCder = std::chrono::high_resolution_clock::now();
#endif

        // Get NEO-DFT Energy derivatives wrt U variables
        loadVXCder(
          this->functionals,
          NPts, U_n_loc, U_gamma_loc,
          epsEval_loc, dVU_n_loc, dVU_gamma_loc,
          epsSCR_loc, dVU_n_SCR_loc, dVU_gamma_SCR_loc);

        loadEPCVXCder(this->particle.charge < 0, 
          this->epc_functionals,
          NPts, U_n_loc, U_gamma_loc, aux_U_n_loc,
          aux_U_gamma_loc, cross_U_gamma_loc, epsEval_loc, dVU_n_loc,
          dVU_gamma_loc, cross_dVU_gamma_loc, epsSCR_loc, dVU_n_SCR_loc, 
          dVU_gamma_SCR_loc, cross_dVU_gamma_loc);


  
#if VXC_DEBUG_LEVEL >= 1
        // TIMING
        auto botloadVXCder = std::chrono::high_resolution_clock::now();
        auto topenergy_vxc = std::chrono::high_resolution_clock::now();
#endif

        // Compute for the current batch the XC energy and increment the 
        // total XC energy.
        integrateXCEnergy[thread_id] += 
          energy_vxc(NPts, weights, epsEval_loc, DenS_loc);

#if VXC_DEBUG_LEVEL >= 1
        // TIMING
        auto botenergy_vxc     = 
          std::chrono::high_resolution_clock::now();
        auto topconstructZVars = 
          std::chrono::high_resolution_clock::now();
#endif
   
        // Construct the required quantities for the formation of the Z 
        // vector (SCALAR) given the kernel derivatives wrt U variables. 

        constructZVars(this->onePDM, SCALAR, isGGA,NPts,dVU_n_loc,dVU_gamma_loc,
          ZrhoVar1_loc, ZgammaVar1_loc, ZgammaVar2_loc);

        // Construct the required quantities for the formation of the Z
        // vector for EPC-19 functional
        if (isGGA and epcisGGA)
          constructEPCZVars(this->particle.charge < 0,
            SCALAR,NPts,cross_dVU_gamma_loc, ZgammaVar3_loc);


#if VXC_DEBUG_LEVEL >= 1
        // TIMING
        auto botconstructZVars = 
          std::chrono::high_resolution_clock::now();
        auto topformZ_vxc      = 
          std::chrono::high_resolution_clock::now();
#endif

        // Creating ZMAT (SCALAR) according to 
        //   J. Chem. Theory Comput. 2011, 7, 3097–3104 Eq. 15 
        if (isGGA and epcisGGA)
          formZ_vxc_epc(this->onePDM, aux_neoks->onePDM,
            SCALAR, isGGA, NPts, NBE, IOff, epsScreen, weights,
            ZrhoVar1_loc, ZgammaVar1_loc, ZgammaVar2_loc, ZgammaVar3_loc,
            DenS_loc, DenZ_loc, nullptr, nullptr, GDenS_loc, GDenZ_loc, nullptr,
            nullptr, aux_DenS_loc, aux_DenZ_loc, nullptr, nullptr,
            aux_GDenS_loc, aux_GDenZ_loc, nullptr, nullptr,
            BasisEval, ZMAT_loc);
        else
          formZ_vxc(this->onePDM, SCALAR,isGGA, NPts, NBE, IOff, epsScreen, weights, 
            ZrhoVar1_loc, ZgammaVar1_loc, ZgammaVar2_loc, DenS_loc, 
            DenZ_loc, nullptr, nullptr, GDenS_loc, GDenZ_loc, nullptr, 
            nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr,
            nullptr, BasisEval, ZMAT_loc);

#if VXC_DEBUG_LEVEL >= 1
        // TIMING
        auto botformZ_vxc = std::chrono::high_resolution_clock::now();
#endif

        bool evalZ = true;

#if VXC_DEBUG_LEVEL < 3
        // Coarse screen on ZMat
        double MaxBasis = *std::max_element(BasisEval,BasisEval+IOff);
        double MaxZ     = *std::max_element(ZMAT_loc,ZMAT_loc+IOff);
        evalZ = ( std::abs(2 * MaxBasis * MaxZ) > epsScreen); 
#endif

        if (evalZ) {

 #if VXC_DEBUG_LEVEL >= 1
          auto topDSYR2K    = std::chrono::high_resolution_clock::now();
 #endif
          // Creating according to 
          //   J. Chem. Theory Comput. 2011, 7, 3097–3104 Eq. 14 
          //
          // Z -> VXC (submat - SCALAR)
          blas::syr2k(blas::Layout::ColMajor,blas::Uplo::Lower,blas::Op::NoTrans,NBE,NPts,1.,BasisEval,NBE,ZMAT_loc,NBE,0.,
            SCRATCHNBNB_loc,NBE);

 #if VXC_DEBUG_LEVEL >= 1
          // TIMING
          auto botDSYR2K      = std::chrono::high_resolution_clock::now();
          durDSYR2K += botDSYR2K - topDSYR2K;
          auto topIncBySubMat = std::chrono::high_resolution_clock::now();
 #endif

          // Locating the submatrix in the right position given the 
          // subset of shells for the given batch.
          IncBySubMat(NB,NB,NBE,NBE,integrateVXC[SCALAR][thread_id],NB,
            SCRATCHNBNB_loc,NBE,subMatCut);

 #if VXC_DEBUG_LEVEL >= 1
          // TIMING
          auto botIncBySubMat = std::chrono::high_resolution_clock::now();
          durIncBySubMat += botIncBySubMat - topIncBySubMat;
 #endif
        }



#if VXC_DEBUG_LEVEL > 3
        prettyPrintSmart(std::cerr,"Basis   ",BasisEval,NBE,NPts,NBE);
        prettyPrintSmart(std::cerr,"BasisX  ",BasisEval+NBE*NPts,NBE,
          NPts,NBE);
        prettyPrintSmart(std::cerr,"BasisY  ",BasisEval+2*NBE*NPts,
          NBE,NPts,NBE);
        prettyPrintSmart(std::cerr,"BasisZ  ",BasisEval+3*NBE*NPts,
          NBE,NPts,NBE);
        prettyPrintSmart(std::cerr,"ZMAT  ",ZMAT_loc,NBE,NPts,NBE);
#endif

#if VXC_DEBUG_LEVEL >= 1
        // TIMING
        durevalDen += botevalDen - topevalDen;
        durmkAuxVar += botmkAuxVar - topmkAuxVar;
        durloadVXCder += botloadVXCder - toploadVXCder;
        durenergy_vxc += botenergy_vxc - topenergy_vxc;
        durconstructZVars += botconstructZVars - topconstructZVars;
        durformZ_vxc += botformZ_vxc - topformZ_vxc;
#endif

#if VXC_DEBUG_LEVEL >= 3
        // Create Numerical Overlap
        for(auto iPt = 0; iPt < NPts; iPt++)
          blas::gemm(blas::Layout::ColMajor,blas::Op::NoTrans,blas::Op::ConjTrans,NB,NB,1,weights[iPt],BasisEval + iPt*NB,NB, 
            BasisEval + iPt*NB,NB, 1.,tmpS,NB);
#endif

        if( not this->onePDM->hasZ() ) return;

        //
        // ---------------   UKS or 2C ------------- Mz ---------------
        //    See J. Chem. Theory Comput. 2017, 13, 2591-2603  
        //

        // Construct the required quantities for the formation of 
        // the Z vector (Mz) given the kernel derivatives wrt U 
        // variables.
        constructZVars(this->onePDM, MZ,isGGA,NPts,dVU_n_loc,dVU_gamma_loc,ZrhoVar1_loc,
          ZgammaVar1_loc, ZgammaVar2_loc);

        // Construct the required quantities for the formation of the Z
        // vector for EPC-19 functional
        if (isGGA and epcisGGA)
          constructEPCZVars(this->particle.charge < 0, 
            MZ,NPts,cross_dVU_gamma_loc, ZgammaVar3_loc);

        // Creating ZMAT (Mz) according to 
        //   J. Chem. Theory Comput. 2011, 7, 3097–3104 Eq. 15 
        if (isGGA and epcisGGA)
          formZ_vxc_epc(this->onePDM, aux_neoks->onePDM,
            MZ, isGGA, NPts, NBE, IOff, epsScreen, weights,
            ZrhoVar1_loc, ZgammaVar1_loc, ZgammaVar2_loc, ZgammaVar3_loc,
            DenS_loc, DenZ_loc, nullptr, nullptr, GDenS_loc, GDenZ_loc, nullptr,
            nullptr, aux_DenS_loc, aux_DenZ_loc, nullptr, nullptr,
            aux_GDenS_loc, aux_GDenZ_loc, nullptr, nullptr,
            BasisEval, ZMAT_loc);
        else
          formZ_vxc(this->onePDM,MZ,isGGA, NPts, NBE, IOff, epsScreen, weights, 
            ZrhoVar1_loc, ZgammaVar1_loc, ZgammaVar2_loc, DenS_loc, 
            DenZ_loc, nullptr, nullptr, GDenS_loc, GDenZ_loc, nullptr, 
            nullptr, nullptr, nullptr, 
            nullptr, nullptr, nullptr, 
            nullptr, BasisEval, ZMAT_loc);


#if VXC_DEBUG_LEVEL < 3
        MaxZ     = *std::max_element(ZMAT_loc,ZMAT_loc+IOff);
        evalZ = ( std::abs(2 * MaxBasis * MaxZ) > epsScreen); 
#endif
        // Coarse screen on ZMat
        if(evalZ) {
  
          // Creating according to 
          //   J. Chem. Theory Comput. 2011, 7, 3097–3104 Eq. 14 
          //
          // Z -> VXC (submat)
          blas::syr2k(blas::Layout::ColMajor,blas::Uplo::Lower,blas::Op::NoTrans,NBE,NPts,1.,BasisEval,NBE,ZMAT_loc,NBE,0.,
            SCRATCHNBNB_loc,NBE);
    
    
          // Locating the submatrix in the right position given 
          // the subset of shells for the given batch.
          IncBySubMat(NB,NB,NBE,NBE,integrateVXC[MZ][thread_id],NB,
            SCRATCHNBNB_loc,NBE,subMatCut);           

        }
 


        if( not this->onePDM->hasXY() ) return;
        
        CErr("Relativistic NEO-Kohn-Sham NYI!",std::cout);

      }; // VXC integrate



      // Create the BeckeIntegrator object
      BeckeIntegrator<EulerMac> 
        integrator(intComm,this->memManager,this->molecule(),basis,aux_basis,
        EulerMac(this->intParam.nRad), this->intParam.nAng, this->intParam.nRadPerBatch,
          (isGGA ? GRADIENT : NOGRAD), (epcisGGA ? GRADIENT : NOGRAD), 
          this->intParam.epsilon);

      // Integrate the VXC
      integrator.integrate<size_t>(vxcbuild);




#if VXC_DEBUG_LEVEL >= 1
      // TIMING
      auto toptransform    = std::chrono::high_resolution_clock::now();
#endif

      // Finishing up the VXC
      // factor in the 4 pi (Lebedev) and built the upper triagolar part
      // since we create only the lower triangular. For all components
      for(auto k = 0; k < VXC_SZYX.size(); k++) {
        if( nthreads == 1 )
          blas::scal(NB2,4*M_PI,VXC_SZYX[k],1);
        else
          for(auto ithread = 0; ithread < nthreads; ithread++)
            MatAdd('N','N',NB,NB,((ithread == 0) ? 0. : 1.),VXC_SZYX[k],NB,
              4*M_PI,integrateVXC[k][ithread],NB, VXC_SZYX[k],NB);
        
        HerMat('L',NB,VXC_SZYX[k],NB);
      }

      for(auto &X : integrateXCEnergy)
        this->XCEnergy += 4*M_PI*X;


// Combine MPI results
#ifdef CQ_ENABLE_MPI

      double* mpiScr;
      if( mpiRank == 0 ) mpiScr = this->memManager.template malloc<double>(NB*NB);

      for(auto &V : VXC_SZYX) {

        mxx::reduce(V,NB*NB,mpiScr,0,std::plus<double>(),intComm);

        if( mpiRank == 0 ) std::copy_n(mpiScr,NB*NB,V);

      }

      if( mpiRank == 0 ) this->memManager.free(mpiScr);

      this->XCEnergy = mxx::reduce(this->XCEnergy,0,std::plus<double>(),intComm);

#endif

#if VXC_DEBUG_LEVEL >= 1
      // TIMING
      auto bottransform = std::chrono::high_resolution_clock::now();
#endif

#if VXC_DEBUG_LEVEL >= 3
      // DebugPrint
      std::cerr << std::endl;
      blas::scal(NB2,4*M_PI,tmpS,1);
      prettyPrintSmart(std::cerr,"Analytic  Overlap",
        this->aoints.overlap,NB,NB,NB);
      prettyPrintSmart(std::cerr,"Numeric  Overlap",tmpS,NB,NB,NB);
      std::cerr << std::endl;
      std::cerr << std::endl;
      for(auto i = 0; i < NB2; i++)
        tmpS[i] = std::abs(tmpS[i] - this->aoints.overlap[i]);
      std::cerr << "MAX DIFF OVERLAP = " << 
        *std::max_element(tmpS,tmpS+NB2) << std::endl;
#endif



#if VXC_DEBUG_LEVEL >= 2
      // DEBUG
      std::cerr << std::scientific << "\n";
      std::cerr << "N     electrons      = " << 4*M_PI*sumrho << "\n";
      std::cerr << "N unp electrons      = " << 4*M_PI*sumspin << "\n";
      std::cerr << "sum gamma        = " << 4*M_PI*sumgamma << "\n";
      std::cerr << "EXC              = " << XCEnergy << "\n";

      prettyPrintSmart(std::cerr,"onePDM Scalar",this->onePDM[SCALAR],
        NB,NB,NB);
      prettyPrintSmart(std::cerr,"Numerical Scalar VXC ",
        integrateVXC[SCALAR][0],NB,NB,NB);

      if( not this->iCS or this->nC > 1 ) { 
        prettyPrintSmart(std::cerr,"onePDM Mz",this->onePDM[MZ],
          NB,NB,NB);
        prettyPrintSmart(std::cerr,"Numerical Mz VXC",
          integrateVXC[MZ][0],NB,NB,NB);

        if( this->onePDM.size() > 2 ) {
          prettyPrintSmart(std::cerr,"onePDM My",this->onePDM[MY],
            NB,NB,NB);
          prettyPrintSmart(std::cerr,"Numerical My VXC",
            integrateVXC[MY][0],NB,NB,NB);

          prettyPrintSmart(std::cerr,"onePDM Mx",this->onePDM[MX],
            NB,NB,NB);
          prettyPrintSmart(std::cerr,"Numerical Mx VXC",
            integrateVXC[MX][0],NB,NB,NB);
        }
      }
#endif

      // Freeing the memory
      // ----------------Main System-------------------------------------------- //
      this->memManager.free(SCRATCHNBNB,SCRATCHNBNP,DenS,epsEval,U_n,
        dVU_n, ZrhoVar1,ZMAT);
      if( isGGA )  
        this->memManager.free(ZgammaVar1,ZgammaVar2,GDenS,U_gamma,
          dVU_gamma);

      if( this->onePDM->hasZ() ) {
        this->memManager.free(DenZ);
        if( isGGA )  this->memManager.free(GDenZ);
      }

      if( this->onePDM->hasXY() )
        CErr("Relativistic NEO-Kohn-Sham NYI!", std::cout);

      if( this->functionals.size() > 1 ) {
        this->memManager.free(epsSCR,dVU_n_SCR);
        if( isGGA ) this->memManager.free(dVU_gamma_SCR);
      }

      if( nthreads != 1 ) this->memManager.free(intVXC_RAW);

      Re1PDM = nullptr;

      // -----------------End Main System-------------------------------------------- //

      // ----------------Auxiliary System-------------------------------------------- //
      this->memManager.free(AUX_SCRATCHNBNB,AUX_SCRATCHNBNP,aux_DenS,aux_U_n,aux_dVU_n,aux_ZrhoVar1);
      if(epcisGGA )  
        this->memManager.free(aux_ZgammaVar1,aux_ZgammaVar2,aux_GDenS,aux_U_gamma,
          aux_dVU_gamma);

      if( this->aux_neoss->onePDM->hasZ() ) {
        this->memManager.free(aux_DenZ);
        if ( epcisGGA ) this->memManager.free(aux_GDenZ);
      }

      if( this->aux_neoss->onePDM->hasXY() )
        CErr("Relativistic NEO-Kohn-Sham NYI!", std::cout);

      if( epc_functionals.size() > 1 ) {
        this->memManager.free(aux_epsSCR,aux_dVU_n_SCR);
        if( epcisGGA ) this->memManager.free(aux_dVU_gamma_SCR);
      }

      aux_Re1PDM = nullptr;

      // -----------------End Aux-------------------------------------------- //
      // -----------------Cross---------------------------------------------- //
      if(isGGA and epcisGGA)
        this->memManager.free(cross_U_gamma, cross_dVU_gamma, ZgammaVar3);
      // -----------------End Cross------------------------------------------ //
      // End freeing the memory


#if VXC_DEBUG_LEVEL >= 1
     // TIMING
     double d_batch = this->aoints.molecule().nAtoms * 
                        intParam.nRad / intParam.nRadPerBatch;

     std::chrono::duration<double> durMem = botMem - topMem;
     std::chrono::duration<double> durtransform = 
       bottransform - toptransform;

     std::cerr << std::scientific << "\n";
     std::cerr << "Mem " << durMem.count()/d_batch << "\n";
     std::cerr << "transform " << durtransform.count()/d_batch << "\n";
     std::cerr << "evalDen " << durevalDen.count()/d_batch << "\n";
     std::cerr << "mkAuxVar " << durmkAuxVar.count()/d_batch << "\n";
     std::cerr << "loadVXCder " << durloadVXCder.count()/d_batch << "\n";
     std::cerr << "energy_vxc " << durenergy_vxc.count()/d_batch << "\n";
     std::cerr << "constructZVars " << durconstructZVars.count()/d_batch 
               << "\n";
     std::cerr << "formZ_vxc " << durformZ_vxc.count()/d_batch << "\n";
     std::cerr << "DSYR2K " << durDSYR2K.count()/d_batch << "\n";
     std::cerr << "IncBySubMat " << durIncBySubMat.count()/d_batch;
     std::cerr <<  "\n\n\n";
#endif


  
      // Turn back on LA threads
      SetLAThreads(LAThreads);

      MPICommFree(intComm); // Free communicator

    } // Valid intComm

    MPI_Barrier(this->comm); // Syncronize the MPI processes

  }; // NEOKohnSham::formVXC


}; // namespace ChronusQ
