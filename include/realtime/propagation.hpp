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

#include <realtime.hpp>
#include <cqlinalg/blas1.hpp>
#include <cqlinalg/blas3.hpp>
#include <cqlinalg/blasutil.hpp>
#include <cqlinalg/matfunc.hpp>
#include <matrix.hpp>

#include <util/matout.hpp>
#include <util/timer.hpp>
#include <unsupported/Eigen/MatrixFunctions>

template <size_t N, typename T>
std::array<T,N> valarray2array(const std::valarray<T> &x) {
  assert( x.size() == N );

  std::array<T,N> arr;
  std::copy_n(&x[0],N,arr.begin());
  return arr;
 
};

namespace ChronusQ {

  template <template <typename, typename> class _SSTyp, typename IntsT>
  void RealTime<_SSTyp,IntsT>::doPropagation(bool doGradient) {

    ProgramTimer::tick("Real Time Total");

    printRTHeader();

    size_t NB = propagator_.nAlphaOrbital();

    if ( savFile.exists() )
      if ( restart )
        restoreState();

    // Upon entry to RT, assume only the orthonormal density is valid
    propagator_.computeOrtho();
    propagator_.ortho2aoDen();
    propagator_.ortho2aoMOs();

    // propagator_.onePDMOrtho->output(std::cout, "TD 1PDM Ortho", true);
    // propagator_.onePDM->output(std::cout, "TD 1PDM", true);

    bool Start(false); // Start the MMUT iterations
    bool FinMM(false); // Wrap up the MMUT iterations

    if(doGradient) {

      EMPerturbation pert_t = pert.getPert(curState.xTime);

      // Form the Fock matrix at the current time
      formFock(false,curState.xTime);

      // Compute properties for D(k) 
      propagator_.computeProperties(pert_t);

      // Print progress line in the output file
      printRTStep();

    }

    // size_t maxStep = (size_t)((intScheme.tMax + intScheme.deltaT/2)/intScheme.deltaT);
    size_t maxStep = (size_t)((intScheme.tMax + intScheme.deltaT/4)/intScheme.deltaT);

    curState.xTime = intScheme.restoreStep * intScheme.deltaT;

    for( curState.iStep = intScheme.restoreStep; 
         curState.iStep < maxStep;
         curState.xTime += intScheme.deltaT, curState.iStep++) {

      ProgramTimer::tick("Real Time Iter");

      // Perturbation for the current time
      EMPerturbation pert_t = pert.getPert(curState.xTime);





#if 1
      // Determine the step type for the current integration step 
      if( intScheme.intAlg == MMUT ) {

        // "Start" the MMUT if this is the first step or we just
        // "Finished" the MMUT segment
        Start = ( curState.iStep == intScheme.restoreStep ) or FinMM;

        // "Start" the MMUT if the current step index is a restart
        // step
        if( intScheme.iRstrt > 0 ) 
          Start = Start or ( curState.iStep % intScheme.iRstrt == 0 );

        // "Finish" the MMUT if this is the last step
        FinMM = ( curState.iStep == maxStep );

        // TODO: "Finish" the MMUT if the field turns on or off
        FinMM = FinMM or pert.isFieldDiscontinuous(curState.xTime, intScheme.deltaT);
          
        // "Finish" the MMUT if the next step will be a restart step
        if( intScheme.iRstrt > 0 ) 
          FinMM = FinMM or ( (curState.iStep + 1) % intScheme.iRstrt == 0 );

        // If "Starting" or "Finishing" the MMUT, the step type is
        // the specified restart step type, else it is the MMUT step
        if( Start or FinMM ) curState.curStep = intScheme.rstStep;
        else                 curState.curStep = ModifiedMidpoint;

        if( Start or FinMM ) std::cout << "  *** Restarting MMUT ***\n";

      // For non leapfrog scheme, the step type is constant
      } else if ( intScheme.intAlg == ExpMagnus2 )
        curState.curStep = ExplicitMagnus2;
#else
      curState.curStep = ForwardEuler;
#endif












      // Handle density copies / swaps for the current step
      //  + Determine the step size
        
      if( curState.curStep == ModifiedMidpoint ) {
        // Swap the saved density with the SingleSlater density
          
        // DOSav(k) = DO(k)
        // DO(k)    = DO(k-1)
        std::shared_ptr<PauliSpinorSquareMatrices<dcomplex>> tmp = DOSav;
        DOSav = propagator_.onePDMOrtho;
        propagator_.onePDMOrtho = tmp;


        curState.stepSize = 2. * intScheme.deltaT;

      } else {
        // Save a copy of the SingleSlater density in the saved density
        // storage 
          
        // DOSav(k) = DO(k)
        *DOSav = *propagator_.onePDMOrtho;

     
        curState.stepSize = intScheme.deltaT;

      }





     
      // Form the Fock matrix at the current time
      formFock(false,curState.xTime);

      // Compute properties for D(k) 
      propagator_.computeProperties(pert_t);

      // Save data
      // TODO: Fix this when we have a stable definition of MD + electronic steps
      // saveState(pert_t);

      // Save D(k) if doing Magnus 2
      std::shared_ptr<PauliSpinorSquareMatrices<dcomplex>> den_k;
      std::shared_ptr<PauliSpinorSquareMatrices<dcomplex>> denOrtho_k;
      if ( curState.curStep == ExplicitMagnus2 ) {
        den_k = std::make_shared<PauliSpinorSquareMatrices<dcomplex>>(*propagator_.onePDM);
        denOrtho_k = std::make_shared<PauliSpinorSquareMatrices<dcomplex>>(*propagator_.onePDMOrtho);
      }


      // Print progress line in the output file
      printRTStep();




      // Orthonormalize the AO Fock matrix
      // F(k) -> FO(k)
      propagator_.ao2orthoFock();


      // Form the propagator from the orthonormal Fock matrix
      // FO(k) -> U**H(k) = exp(- i * dt * FO(k) )
      formPropagator();

      // Propagator the orthonormal density matrix
      // DO (in propagator_) will now store DO(k+1)
      //
      // DO(k+1) = U**H(k) * DO * U(k)
      // - Where DO is what is currently stored in propagator_
      //
      // ***
      // This function also transforms DO(k+1) to the AO
      // basis ( DO(k+1) -> D(k+1) in propagator_ ) and
      // computes the change in density from the previous 
      // AO density ( delD = D(k+1) - D(k) ) 
      // ***
      propagateWFN();

      //
      // Second order magnus
      //
      if ( curState.curStep == ExplicitMagnus2 ) {

        // F(k)
        PauliSpinorSquareMatrices<dcomplex> fock_k(*propagator_.fockMatrix);
        
        // F(k + 1)
        formFock(false, curState.xTime + intScheme.deltaT);

        // Store 0.5 * ( F(k) + F(k+1) ) in propagator_.fockMatrix
        *propagator_.fockMatrix = 0.5 * (fock_k + *propagator_.fockMatrix);


        // Restore old densities
        *propagator_.onePDM = *den_k;
        *propagator_.onePDMOrtho = *denOrtho_k;

        // Repeat formation of propagator and propagation
        propagator_.ao2orthoFock();
        formPropagator();
        propagateWFN();

      }  // End 2nd order magnus

      ProgramTimer::tock("Real Time Iter");

    } // Time loop


    ProgramTimer::tock("Real Time Total");

  //mathematicaPrint(std::cerr,"Dipole-X",&data.ElecDipole[0][0],
  //  curState.iStep,1,curState.iStep,3);

  }; // RealTime::doPropagation


  /**
   *  \brief Form the adjoint of the unitary propagator
   *
   *  \f[
   *    U = \exp\left( -i \delta t F \right) 
   *      = \exp\left( -\frac{i\delta t}{2} 
   *                    \left(F^S \otimes I_2 + F^k \sigma_k\right) \right) 
   *      = \frac{1}{2}U^S \otimes I_2 + \frac{1}{2} U^k \otimes \sigma_k
   *  \f]
   */ 
  template <template <typename, typename> class _SSTyp, typename IntsT>
  void RealTime<_SSTyp,IntsT>::formPropagator() {

    ProgramTimer::tick("Propagator Formation");

    size_t NB = propagator_.nAlphaOrbital();

    // Form U

    // Restricted
    if( not UH->hasZ() ) {
      // See docs for factor of 2
      MatExp('D',NB,dcomplex(0.,-curState.stepSize/2.),
        propagator_.fockMatrixOrtho->S().pointer(),NB,UH->S().pointer(),NB,memManager_);

      Scale(NB*NB,dcomplex(2.),UH->S().pointer(),1);

    // Unrestricted
    } else if( not UH->hasXY() ) {

      // Transform SCALAR / MZ -> ALPHA / BETA
      std::vector<SquareMatrix<dcomplex>> Fblocks =
          propagator_.fockMatrixOrtho->template spinGatherToBlocks<dcomplex>(false);
      std::vector<SquareMatrix<dcomplex>> UHblocks;
      UHblocks.reserve(2);
      UHblocks.emplace_back(memManager_, NB);
      UHblocks.emplace_back(memManager_, NB);

      MatExp('D',NB,dcomplex(0.,-curState.stepSize),
        Fblocks[0].pointer(),NB,UHblocks[0].pointer(),NB,memManager_);
      MatExp('D',NB,dcomplex(0.,-curState.stepSize),
        Fblocks[1].pointer(),NB,UHblocks[1].pointer(),NB,memManager_);

      // Transform ALPHA / BETA -> SCALAR / MZ
      *UH = PauliSpinorSquareMatrices<dcomplex>::
          spinBlockScatterBuild<dcomplex>(UHblocks[0],UHblocks[1]);

    // Generalized (2C)
    } else {

      SquareMatrix<dcomplex> F2C(propagator_.fockMatrixOrtho->template spinGather<dcomplex>());
      SquareMatrix<dcomplex> UH2C(memManager_, 2*NB);

      MatExp('D',2*NB,dcomplex(0.,-curState.stepSize),
             F2C.pointer(),2*NB,UH2C.pointer(),2*NB, memManager_);

      *UH = UH2C.template spinScatter<dcomplex>();

    }

    ProgramTimer::tock("Propagator Formation");
#if 0

    prettyPrintSmart(std::cout,"UH Scalar",UH->S().pointer(),NB,NB,NB);
    if( UH->hasZ() )
      prettyPrintSmart(std::cout,"UH MZ",UH->Z().pointer(),NB,NB,NB);

#endif
    
  }; // RealTime::formPropagator



  
  template <template <typename, typename> class _SSTyp, typename IntsT>
  void RealTime<_SSTyp,IntsT>::propagateWFN() {

    ProgramTimer::tick("Propagate WFN");

    size_t NB = propagator_.nAlphaOrbital();
    size_t NC = propagator_.nC;
    dcomplex *SCR  = memManager_.template malloc<dcomplex>(NC*NC*NB*NB);
    dcomplex *SCR1 = memManager_.template malloc<dcomplex>(NC*NC*NB*NB);

    if( not UH->hasXY() ) {

      // Create X(S) = (U**H * DO)(S) in SCR 

      // SCR = 0.5 * U(S)**H * DO(S)
      Gemm('N','N',NB,NB,NB,dcomplex(0.5),UH->S().pointer(),NB,
        propagator_.onePDMOrtho->S().pointer(),NB,dcomplex(0.),SCR,NB);

      // SCR += 0.5 * U(Z)**H * DO(Z)
      if( UH->hasZ() )
        Gemm('N','N',NB,NB,NB,dcomplex(0.5),UH->Z().pointer(),NB,
          propagator_.onePDMOrtho->Z().pointer(),NB,dcomplex(1.),SCR,NB);





      // Create X(Z) = (U**H * DO)(Z) in SCR1
        
      if( propagator_.onePDMOrtho->hasZ() ) {

        // SCR1 = 0.5 * U(S)**H * DO(Z)
        Gemm('N','N',NB,NB,NB,dcomplex(0.5),UH->S().pointer(),NB,
          propagator_.onePDMOrtho->Z().pointer(),NB,dcomplex(0.),SCR1,NB);

        // SCR1 += 0.5 * U(Z)**H * DO(S)
        if( UH->hasZ() )
          Gemm('N','N',NB,NB,NB,dcomplex(0.5),UH->Z().pointer(),NB,
            propagator_.onePDMOrtho->S().pointer(),NB,dcomplex(1.),SCR1,NB);

      }






      // DO(S) = 0.5 * ( X(S) * U(S) + X(Z) * U(Z) )
      //       = 0.5 * ( SCR  * U(S) + SCR1 * U(Z) )

      // DO(S) = 0.5 * SCR * U(S)
      Gemm('N','C',NB,NB,NB,dcomplex(0.5),SCR,NB,UH->S().pointer(),NB,
           dcomplex(0.),propagator_.onePDMOrtho->S().pointer(),NB);
 
      // DO(S) += 0.5 * SCR1 * U(Z)
      if( UH->hasZ() )
        Gemm('N','C',NB,NB,NB,dcomplex(0.5),SCR1,NB,UH->Z().pointer(),NB,
             dcomplex(1.),propagator_.onePDMOrtho->S().pointer(),NB);


      if( UH->hasZ() ) {

        // DO(Z) = 0.5 * ( X(S) * U(Z) + X(Z) * U(S) )
        //       = 0.5 * ( SCR  * U(Z) + SCR1 * U(S) )
          
        // DO(Z) = 0.5 * SCR * U(Z)
        Gemm('N','C',NB,NB,NB,dcomplex(0.5),SCR,NB,UH->Z().pointer(),NB,
             dcomplex(0.),propagator_.onePDMOrtho->Z().pointer(),NB);
 
        // DO(Z) += 0.5 * SCR1 * U(S)
        Gemm('N','C',NB,NB,NB,dcomplex(0.5),SCR1,NB,UH->S().pointer(),NB,
             dcomplex(1.),propagator_.onePDMOrtho->Z().pointer(),NB);

      }

    } else {

      // Gather DO
      SquareMatrix<dcomplex> DO(propagator_.onePDMOrtho->template spinGather<dcomplex>());

      // Gather UH into SCR
      SquareMatrix<dcomplex> UHblockForm(UH->template spinGather<dcomplex>());

      // SCR1 = U**H * DO
      Gemm('N','N',2*NB,2*NB,2*NB,dcomplex(1.),UHblockForm.pointer(),2*NB,
           DO.pointer(),2*NB,dcomplex(0.),SCR1,2*NB);

      // DO = SCR1 * U
      Gemm('N','C',2*NB,2*NB,2*NB,dcomplex(1.),SCR1,2*NB,
           UHblockForm.pointer(),2*NB,dcomplex(0.),DO.pointer(),2*NB);

      // Scatter DO
      *propagator_.onePDMOrtho = DO.template spinScatter<dcomplex>();

    }


    propagator_.ortho2aoDen();

    memManager_.free(SCR,SCR1);

    ProgramTimer::tock("Propagate WFN");

  }; // RealTime::propagatorWFN


  template <template <typename, typename> class _SSTyp, typename IntsT>
  void RealTime<_SSTyp,IntsT>::createRTDataSets(size_t maxPoints) {
    std::cout << "create datasets with size: " << maxPoints << std::endl;

    if( maxPoints == 0 ) 
      maxPoints = intScheme.tMax / intScheme.deltaT + 1;
    
    savFile.createGroup("RT");

    savFile.createDataSet<double>("RT/TIME",
        {maxPoints});
    savFile.createDataSet<double>("RT/ENERGY",
        {maxPoints});
    savFile.createDataSet<double>("RT/LEN_ELEC_DIPOLE",
        {maxPoints,3});
    savFile.createDataSet<double>("RT/LEN_ELEC_DIPOLE_FIELD",
        {maxPoints,3});
  }; // RealTime::createRTDataSets


  template <template <typename, typename> class _SSTyp, typename IntsT>
  void RealTime<_SSTyp,IntsT>::restoreState() {

    hsize_t maxPoints = intScheme.tMax / intScheme.deltaT + 1;

    if ( savFile.getDims("RT/TIME")[0] != maxPoints )
      CErr("Mismatched requested and saved propagation length!");

    // Restore time dependent density
    try {
      savFile.readData("RT/TD_1PDM", *propagator_.onePDM);
      savFile.readData("RT/TD_1PDM_ORTHO", *propagator_.onePDMOrtho);
    } catch(...) { }

    // Find last time step that was checkpointed
    double* timeData = memManager_.template malloc<double>(maxPoints);
    savFile.readData("RT/TIME", timeData);
    int offset = *timeData < 1e-10 ? -1 : 0;
    size_t restoreStep = offset + std::distance( timeData, 
      std::find_if( timeData+1, timeData+maxPoints,
        [](double x){ return x < 1e-10; }
      )
    );
    memManager_.free(timeData);

    std::cout << "  *** Restoring from step " << restoreStep << " (";
    std::cout << std::setprecision(4) << restoreStep * intScheme.deltaT;
    std::cout << " AU) ***" << std::endl;

    intScheme.restoreStep = restoreStep;

  }; // RealTime::restoreState


  template <template <typename, typename> class _SSTyp, typename IntsT>
  void RealTime<_SSTyp,IntsT>::saveState(EMPerturbation& pert_t) {

    size_t NB = propagator_.nAlphaOrbital();

    data.Time.push_back(curState.xTime);
    data.Energy.push_back(propagator_.totalEnergy);
    data.ElecDipole.push_back(propagator_.elecDipole);
    if( pert_t.fields.size() > 0 )
      data.ElecDipoleField.push_back( pert_t.getDipoleAmp(Electric) );

    // Write to file
    if( savFile.exists() ) {

      hsize_t nSteps = 0;

      if (curState.iStep % intScheme.iSave == 0 
          and curState.iStep != intScheme.restoreStep)
        nSteps = intScheme.iSave;
      else if ( curState.iStep >= intScheme.tMax / intScheme.deltaT )
        nSteps = curState.iStep % intScheme.iSave + 1;

      hsize_t lastPos = curState.iStep - nSteps + 1;
      hsize_t memLastPos = lastPos - intScheme.restoreStep;

      if (nSteps != 0) {
        std::cout << "  *** Saving data to binary file ***" << std::endl;
        savFile.partialWriteData("RT/TIME", &data.Time[0], {lastPos},
            {nSteps}, {memLastPos}, {data.Time.size()});
        savFile.partialWriteData("RT/ENERGY", &data.Energy[0], {lastPos},
            {nSteps}, {memLastPos}, {data.Time.size()});
        savFile.partialWriteData("RT/LEN_ELEC_DIPOLE", &data.ElecDipole[0][0],
          {lastPos, 0}, {nSteps, 3}, {memLastPos, 0}, {data.Time.size(), 3});

        if( data.ElecDipoleField.size() > 0 )
          savFile.partialWriteData("RT/LEN_ELEC_DIPOLE_FIELD",
            &data.ElecDipoleField[0][0], {lastPos,0}, {nSteps,3},
            {memLastPos, 0}, {data.Time.size(), 3});

        savFile.safeWriteData("RT/TD_1PDM", *propagator_.onePDM);
        if ( curState.curStep == ModifiedMidpoint )
          savFile.safeWriteData("RT/TD_1PDM_ORTHO",*DOSav);
        else
          savFile.safeWriteData("RT/TD_1PDM_ORTHO",*propagator_.onePDMOrtho);
      }
    }
  }; // RealTime::saveState


  template <template <typename, typename> class _SSTyp, typename IntsT>
  void RealTime<_SSTyp,IntsT>::updateAOProperties(double t) {
    // Form AO density
    propagator_.computeOrtho();
    propagator_.ortho2aoDen();

    // Form fock matrix
    formFock(false, t);
    // Compute properties
    EMPerturbation pert_t = pert.getPert(t);
    propagator_.computeProperties(pert_t);

  }; // RealTime::orthoAndFock

}; // namespace ChronusQ

