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
#include <singleslater/neo_singleslater.hpp>
#include <util/matout.hpp>

namespace ChronusQ {

  /**
   *  \brief Initializes the environment for the NEO-SCF calculation
   *
   *  Allocate memory for extrapolation and compute the energy
   */
  template <typename MatsT, typename IntsT>
  void NEOSingleSlater<MatsT,IntsT>::SCFInit() {

    // main system
    this->SingleSlater<MatsT,IntsT>::SCFInit();

    // auxiliary system
    this->aux_neoss->SingleSlater<MatsT,IntsT>::SCFInit();

  }; // NEOSingleSlater<MatsT>::SCFInit

  /**
   *  \brief Finalizes the environment for the NEO-SCF calculation
   *
   */
  template <typename MatsT, typename IntsT>
  void NEOSingleSlater<MatsT,IntsT>::SCFFin() {

    // main system
    this->SingleSlater<MatsT,IntsT>::SCFFin();

    // auxiliary system
    this->aux_neoss->SingleSlater<MatsT,IntsT>::SCFFin();

  }; // NEOSingleSlater<MatsT>::SCFFin

  /**
   *  \brief Save the current state for the NEO-SCF calculation
   *
   *  Allocate memory for extrapolation and compute the energy
   */
  template <typename MatsT, typename IntsT>
  void NEOSingleSlater<MatsT,IntsT>::saveCurrentState() {

    // main system
    this->SingleSlater<MatsT,IntsT>::saveCurrentState();

    // auxiliary system
    this->aux_neoss->SingleSlater<MatsT,IntsT>::saveCurrentState();

  }; // NEOSingleSlater<MatsT>::saveCurrentState

  /**
   *  \brief Performs the NEO self-consistent field procedure given set of 
   *  orbitals.
   */
  template <typename MatsT, typename IntsT>
  void NEOSingleSlater<MatsT,IntsT>::SCF(EMPerturbation &pert) {

    this->printLevel = 1;
    this->aux_neoss->printLevel = 1;

    // initialization
    SCFInit(); 

    // Initialize type independent parameters
    bool isConverged = false;
    this->scfControls.dampParam = this->scfControls.dampStartParam;
    
    this->scfControls.doIncFock = false;

    if( this->scfControls.scfAlg == _NEWTON_RAPHSON_SCF )
      this->scfControls.doExtrap = false;

    if( this->scfControls.scfAlg == _SKIP_SCF )
      isConverged = true;

    // Compute initial properties
    this->computeTotalProperties(pert);

    if ( this->printLevel > 0 and MPIRank(this->comm) == 0 ) {
      this->printSCFHeader(std::cout,pert);
      printSCFMacroProg(std::cout,false);
    }

    for( this->scfConv.nSCFMacroIter = 0; this->scfConv.nSCFMacroIter < this->scfControls.maxSCFIter; 
         this->scfConv.nSCFMacroIter++) {

      // Save current state of the wave function (method specific)
      saveCurrentState();

      // Converge proton SCF
      this->aux_neoss->SingleSlaterBase::SCF(pert);

      // Converge electron SCF
      this->SingleSlaterBase::SCF(pert);

      // Exit loop on convergence
      if(isConverged) break;

      // Evaluate convergence
      isConverged = evalMacroConver(pert);

      // Print out iteration information
      if ( this->printLevel > 0 and (MPIRank(this->comm) == 0)) printSCFMacroProg(std::cout,true);

    }; // Iteration loop

    // Save current state of the wave function (method specific)
    saveCurrentState();

    // finalize SCF
    SCFFin();

    // Compute initial properties
    this->computeTotalProperties(pert);

    if(not isConverged)
      CErr(std::string("NEO-SCF Failed to converge within ") + 
        std::to_string(this->scfControls.maxSCFIter) + 
        std::string(" iterations"));
    else if ( this->printLevel > 0 ) {
      std::cout << std::endl << "NEO-SCF Completed: E("
                << this->refShortName_ << ") = " << std::fixed
                << std::setprecision(10) << this->totalMacroEnergy
                << " Eh after " << this->scfConv.nSCFMacroIter
                << " SCF Iteration" << std::endl;
    }

    if( this->printLevel > 0 ) std::cout << BannerEnd << std::endl;

    if( this->printLevel > 0 ) {
      this->printMOInfo(std::cout);
      this->printMultipoles(std::cout);
      this->printSpin(std::cout);
      this->printMiscProperties(std::cout);

      this->aux_neoss->printMOInfo(std::cout);
      this->aux_neoss->printMultipoles(std::cout);
      this->aux_neoss->printSpin(std::cout);
    }

  }; // NEOSingleSlater::SCF()

  /**
   *  \brief Evaluate the Macro NEO-SCF convergence based on various criteria
   *  
   *  Checks change in energy and density between macro SCF iterations,
   */
  template <typename MatsT, typename IntsT>
  bool NEOSingleSlater<MatsT,IntsT>::evalMacroConver(EMPerturbation &pert) {

    bool isConverged; 

    // Compute all SCF convergence information on root process
    if( MPIRank(this->comm) == 0 ) {
      
      // Save copy of old energy
      double oldEnergy = totalMacroEnergy;

      // Compute new energy
      computeTotalProperties(pert);

      // Compute the difference between current and old energy
      this->scfConv.deltaEnergy = totalMacroEnergy - oldEnergy;

      bool energyConv = std::abs(this->scfConv.deltaEnergy) < 
                        this->scfControls.eneConvTol;

      isConverged = energyConv;
    }

#ifdef CQ_ENABLE_MPI
    // Broadcast whether or not we're converged to ensure that all
    // MPI processes exit the NEO-SCF simultaneously
    if( MPISize(this->comm) > 1 ) MPIBCast(isConverged,0,this->comm);
#endif
    
    return isConverged;

  }; 

  /**
   *  \brief Print the current macro convergence information of the NEO-SCF
   *  procedure
   */ 
  template <typename MatsT, typename IntsT>
  void NEOSingleSlater<MatsT,IntsT>::printSCFMacroProg(std::ostream &out,
    bool printDiff) {

    // SCF Iteration
    out << "  SCFIt Macro: " <<std::setw(6) << std::left;

    if( printDiff ) out << this->scfConv.nSCFMacroIter + 1;
    else            out << 0;

    // Current Total Energy
    out << std::setw(18) << std::fixed << std::setprecision(10)
                         << std::left << totalMacroEnergy;

    if( printDiff ) {
      out << std::scientific << std::setprecision(7);
      // Current Change in Energy
      out << std::setw(14) << std::right << this->scfConv.deltaEnergy;
      out << "   ";
      //out << std::setw(13) << std::right << scfConv.RMSDenScalar;
      //if(not iCS or nC > 1) {
      //  out << "   ";
      //  out << std::setw(13) << std::right << scfConv.RMSDenMag;
      //}
    }
  
    out << std::endl;
  }; // NEOSingleSlater<T>::printSCFMacroProg


}; // namespace ChronusQ
