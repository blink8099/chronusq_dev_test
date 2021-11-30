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
#include <corehbuilder.hpp>
#include <corehbuilder/x2c.hpp>
#include <fockbuilder.hpp>

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
   *  \brief Forms the Fock matrix for a NEO single slater determinant using
   *  the 1PDM.
   *
   *  \param [in] increment Whether or not the Fock matrix is being 
   *  incremented using a previous density
   *
   *  Populates / overwrites fock strorage
   */ 
  template <typename MatsT, typename IntsT>
  void NEOSingleSlater<MatsT,IntsT>::formFock(
    EMPerturbation &pert, bool increment, double xHFX) {

    // form Fock matrix for the underlying single slater object
    SingleSlater<MatsT,IntsT>::formFock(pert, increment, xHFX);

    // form the epJ contribution to the Fock matrix
    this->fockBuilder->formepJ(*this, *this->aux_neoss, increment, xHFX);

  }; // NEOSingleSlater::fockFock


  /**
   *  \brief Compute the Core Hamiltonian.
   *
   *  \param [in] typ Which Hamiltonian to build
   */ 
  template <typename MatsT, typename IntsT>
  void NEOSingleSlater<MatsT,IntsT>::formCoreH(EMPerturbation& emPert, bool save) {

    // form core Hamiltonian for the main system
    SingleSlater<MatsT,IntsT>::formCoreH(emPert, save);

    // form core Hamiltonian for the auxiliary system
    this->aux_neoss->SingleSlater<MatsT,IntsT>::formCoreH(emPert, save);    

  }; // NEOSingleSlater<MatsT,IntsT>::computeCoreH

  /**
   *  \brief Compute the overall properties of the electron-proton system
   */
  template <typename MatsT, typename IntsT>
  void NEOSingleSlater<MatsT,IntsT>::removeNucMultipoleContrib() {

    // remove nuclear contribution
    if (this->particle.charge < 0.) {
      for (auto &atom : this->molecule().atoms)
        if (atom.quantum)
          MatAdd('N','N',3,1,1.,&this->elecDipole[0],3,-atom.nucCharge,
            &atom.coord[0],3,&this->elecDipole[0],3);

      // remove quantum Nuclear contributions to the quadrupoles
      for(auto &atom : this->molecule().atoms)
        if (atom.quantum)
          for(size_t iXYZ = 0; iXYZ < 3; iXYZ++)
          for(size_t jXYZ = 0; jXYZ < 3; jXYZ++) 
            this->elecQuadrupole[iXYZ][jXYZ] -=
              atom.nucCharge * atom.coord[iXYZ] * atom.coord[jXYZ];

      // remove quantum Nuclear contributions to the octupoles
      for(auto &atom : this->molecule().atoms)
        if (atom.quantum)
          for(size_t iXYZ = 0; iXYZ < 3; iXYZ++)
          for(size_t jXYZ = 0; jXYZ < 3; jXYZ++)
          for(size_t kXYZ = 0; kXYZ < 3; kXYZ++)
            this->elecOctupole[iXYZ][jXYZ][kXYZ] -=
              atom.nucCharge * atom.coord[iXYZ] * atom.coord[jXYZ] *
              atom.coord[kXYZ];
    }
    else {
      for (auto &atom : this->molecule().atoms)
        MatAdd('N','N',3,1,1.,&this->elecDipole[0],3,-atom.nucCharge,
          &atom.coord[0],3,&this->elecDipole[0],3);

      // remove all Nuclear contributions to the quadrupoles
      for(auto &atom : this->molecule().atoms)
      for(size_t iXYZ = 0; iXYZ < 3; iXYZ++)
      for(size_t jXYZ = 0; jXYZ < 3; jXYZ++) 
        this->elecQuadrupole[iXYZ][jXYZ] -=
          atom.nucCharge * atom.coord[iXYZ] * atom.coord[jXYZ];

      // remove all Nuclear contributions to the octupoles
      for(auto &atom : this->molecule().atoms)
      for(size_t iXYZ = 0; iXYZ < 3; iXYZ++)
      for(size_t jXYZ = 0; jXYZ < 3; jXYZ++)
      for(size_t kXYZ = 0; kXYZ < 3; kXYZ++)
        this->elecOctupole[iXYZ][jXYZ][kXYZ] -=
          atom.nucCharge * atom.coord[iXYZ] * atom.coord[jXYZ] *
          atom.coord[kXYZ];
    }

  }; // NEOSingleSlater<MatsT,IntsT>::computeTotalProperties


  /**
   *  \brief Compute the overall properties of the electron-proton system
   */
  template <typename MatsT, typename IntsT>
  void NEOSingleSlater<MatsT,IntsT>::computeTotalProperties(EMPerturbation& emPert) {

    // electron
    this->computeProperties(emPert);
    this->removeNucMultipoleContrib(); 

    double EPJEnergy = -1.0 * this->template computeOBProperty<double,DENSITY_TYPE::SCALAR>(epJMatrix->pointer());
    this->MBEnergy += 0.5 * EPJEnergy;

    // Assemble total energy
    this->totalEnergy = 
      this->OBEnergy + this->MBEnergy + this->molecule().nucRepEnergy;


    // proton
    this->aux_neoss->computeProperties(emPert);
    this->aux_neoss->removeNucMultipoleContrib();

    // set total macro energy
    totalMacroEnergy = this->totalEnergy;

    // Add proton one-body energy
    totalMacroEnergy += this->aux_neoss->OBEnergy;

    // Add proton two-body energy
    totalMacroEnergy += this->aux_neoss->PPEnergy;

    // Add proton electric field energy
    double aux_field_delta(0.);

    // Purely electric field contribution here
    auto protDipoleField = emPert.getDipoleAmp(Electric);
    aux_field_delta += 
      -1.0 * protDipoleField[0] * this->aux_neoss->elecDipole[0] + 
      -1.0 * protDipoleField[1] * this->aux_neoss->elecDipole[1] + 
      -1.0 * protDipoleField[2] * this->aux_neoss->elecDipole[2]; 

    totalMacroEnergy += aux_field_delta; 

  }; // NEOSingleSlater<MatsT,IntsT>::computeTotalProperties

}; // namespace ChronusQ

