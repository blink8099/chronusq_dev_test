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
#include <cqlinalg.hpp>
#include <util/matout.hpp>
#include <corehbuilder/nonrel.hpp>
#include <particleintegrals/twopints/incore4indextpi.hpp>
#include <singleslater/neo_singleslater.hpp>

namespace ChronusQ {


  /**
   *  \brief Form a set of guess orbitals for the neo single slater
   *  determinant SCF
   */
  template <typename MatsT, typename IntsT>
  void NEOSingleSlater<MatsT,IntsT>::formGuess(const SingleSlaterOptions&) {

    if( this->printLevel > 0 )
      std::cout << "  *** Forming Initial Guess Density for NEO-SCF Procedure ***"
                << std::endl << std::endl;

    // electronic guess
    // SAD guess is not supported by NEO
    if (this->scfControls.guess == SAD)
      CErr("SAD guess is not supported by NEO", std::cout);
    else if (this->scfControls.guess == CORE)
      this->CoreGuess();
    else if (this->scfControls.guess == RANDOM)
      this->RandomGuess();
    else if (this->scfControls.guess == READMO)
      this->ReadGuessMO();
    else if (this->scfControls.guess == READDEN)
      this->ReadGuess1PDM();
    else 
      CErr("Unknown choice for SCF.GUESS",std::cout);

    // protonic guess
    if (this->scfControls.prot_guess == SAD)
      CErr("SAD guess is not supported by NEO", std::cout);
    else if (this->scfControls.prot_guess == CORE)
      this->aux_neoss->CoreGuess();
    else if (this->scfControls.prot_guess == RANDOM)
      this->aux_neoss->RandomGuess();
    else if (this->scfControls.prot_guess == READMO)
      this->aux_neoss->ReadGuessMO();
    else if (this->scfControls.prot_guess == READDEN)
      this->aux_neoss->ReadGuess1PDM();
    else 
      CErr("Unknown choice for SCF.PROT_GUESS",std::cout);

    EMPerturbation pert; // Dummy EM perturbation
    formFock(pert, false);
    this->aux_neoss->formFock(pert, false);
    
    // Common to all guess, form new set of orbitals from 
    // initial guess at Fock
    this->getNewOrbitals(pert,false);
    this->aux_neoss->getNewOrbitals(pert,false);

  }; // NEOSingleSlater<T>::formGuess()

}; // namespace ChronusQ
