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
#include <singleslater/neoss.hpp>
#include <util/matout.hpp>

namespace ChronusQ {

  /**
   *  \brief Performs the NEO self-consistent field procedure given set of 
   *  orbitals.
   */
  template <typename MatsT, typename IntsT>
  void NEOSS<MatsT,IntsT>::SCF(EMPerturbation &pert) {

    applyToEach([](SubSSPtr& ss){ ss->printLevel = 1; });

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
    this->computeProperties(pert);

    if ( this->printLevel > 0 and MPIRank(this->comm) == 0 ) {
      this->printSCFHeader(std::cout,pert);
      printSCFProg(std::cout,false);
    }

    for( this->scfConv.nSCFMacroIter = 0; this->scfConv.nSCFMacroIter < this->scfControls.maxSCFIter; 
         this->scfConv.nSCFMacroIter++) {

      // Save current state of the wave function (method specific)
      saveCurrentState();

      // Perform the SCF on each subsystem
      applyToEach([&](SubSSPtr& ss){ ss->SCF(pert); });

      // Exit loop on convergence
      // NOTE: "Break" can be placed after isCovnerged
      if(isConverged) break;

      // Evaluate convergence
      isConverged = evalConver(pert);

      // Print out iteration information
      if ( this->printLevel > 0 and (MPIRank(this->comm) == 0)) printSCFProg(std::cout,true);

    }; // Iteration loop

    // Save current state of the wave function (method specific)
    saveCurrentState();

    // finalize SCF
    SCFFin();

    // Compute initial properties
    this->computeProperties(pert);

    if(not isConverged)
      CErr(std::string("NEO-SCF Failed to converge within ") + 
        std::to_string(this->scfControls.maxSCFIter) + 
        std::string(" iterations"));
    else if ( this->printLevel > 0 ) {
      std::cout << std::endl << "NEO-SCF Completed: E("
                << this->refShortName_ << ") = " << std::fixed
                << std::setprecision(10) << this->totalEnergy
                << " Eh after " << this->scfConv.nSCFMacroIter
                << " SCF Iteration" << std::endl;
    }

    if( this->printLevel > 0 ) std::cout << BannerEnd << std::endl;

    if( this->printLevel > 0 ) {
      applyToEach([](SubSSPtr& ss) {
        ss->printMOInfo(std::cout);
        ss->printSpin(std::cout);
        ss->printMiscProperties(std::cout);
      });
    }
    this->printMultipoles(std::cout);
  }; // NEOSingleSlater::SCF()

}; // namespace ChronusQ

