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

namespace ChronusQ {

template<typename MatsT, typename IntsT>
void NEOHartreeFock<MatsT, IntsT>::buildModifyOrbitals() {
  // Modify SCFControls
  this->scfControls.printLevel    = this->printLevel;
  this->scfControls.refLongName_  = this->refLongName_;
  this->scfControls.refShortName_ = this->refShortName_;


  // Initialize ModifyOrbitalOptions
  ModifyOrbitalsOptions<MatsT> modOrbOpt;

  // Bind Lambdas to std::functions
  modOrbOpt.printProperties   = [this]() { this->printProperties(); };
  modOrbOpt.saveCurrentState  = [this]() { this->saveCurrentState(); };
  modOrbOpt.formFock          = [this](EMPerturbation& pert) { this->formBothFock(pert); };
  modOrbOpt.computeProperties = [this](EMPerturbation& pert) { this->computeTotalProperties(pert); };
  modOrbOpt.formDensity       = [this]() { this->formDensity(); };
  modOrbOpt.getFock           = [this]() { return this->getFock(); };
  modOrbOpt.getOnePDM         = [this]() { return this->getOnePDM(); };
  modOrbOpt.getOrtho          = [this]() { return this->getOrtho(); };
  modOrbOpt.getTotalEnergy    = [this]() { return this->getTotalEnergy(); };

  // Make ModifyOrbitals based on scfControls
  if( this->scfControls.scfAlg == _CONVENTIONAL_SCF ) {
    this->modifyOrbitals = std::dynamic_pointer_cast<ModifyOrbitals<MatsT>>(
      std::make_shared<ConventionalSCF<MatsT>>(this->scfControls, this->comm, modOrbOpt, this->memManager));
  } else if( this->scfControls.scfAlg == _NEWTON_RAPHSON_SCF ) {

    // Generate NRRotationOptions
    std::vector<NRRotOptions> rotOpt = NEOSingleSlater<MatsT,IntsT>::buildRotOpt();

    // Create NRSCF object
    this->modifyOrbitals = std::dynamic_pointer_cast<ModifyOrbitals<MatsT>>(
      std::make_shared<NewtonRaphsonSCF<MatsT>>(rotOpt, this->scfControls, this->comm, modOrbOpt, this->memManager));
  } else {
    this->scfControls.doExtrap = false;
    this->modifyOrbitals       = std::dynamic_pointer_cast<ModifyOrbitals<MatsT>>(
      std::make_shared<SkipSCF<MatsT>>(this->scfControls, this->comm, modOrbOpt, this->memManager));
  }
};   // NEOHartreeFock<MatsT,IntsT> :: buildModifyOrbital

};   // namespace ChronusQ
