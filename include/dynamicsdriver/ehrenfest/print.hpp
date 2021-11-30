/* 
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *  
 *  Copyright (C) 2014-2019 Li Research Group (University of Washington)
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
#ifndef __INCLUDED_EHRENFEST_PRINT_HPP__
#define __INCLUDED_EHRENFEST_PRINT_HPP__

#include <ehrenfest.hpp>
#include <cxxapi/output.hpp>
#include <physcon.hpp>

namespace ChronusQ {

  void EFFormattedLine(std::ostream &out, std::string s) {
    out << std::setw(38) << "  " + s << std::endl;
  }

  template <typename T>
  void EFFormattedLine(std::ostream &out, std::string s, T v) {
    out << std::setw(38) << "  " + s << v << std::endl;
  }

  template <typename T, typename U>
  void EFFormattedLine(std::ostream &out, std::string s, T v, U u) {
    out << std::setw(38) << "  " + s << v << u << std::endl;
  }

  template<template <typename, typename> class _SSTyp, typename IntsT>
  void Ehrenfest<_SSTyp,IntsT>::printEFHeader() {

    std::cout << BannerTop << std::endl;
    std::cout << "Ehrenfest Dynamics Settings:" << std::endl << std::endl;

    std::cout << std::left << std::setprecision(7);
    std::string AUTime = " \u0127 / Eh";

    EFFormattedLine(std::cout,"* Simulation Paramaters:");

    int nSteps = this->intScheme.MaxTN / this->intScheme.dTN;
    EFFormattedLine(std::cout,"Simulation Time:",this->intScheme.MaxTN,AUTime);
    EFFormattedLine(std::cout,"Number of Verlet Steps:",nSteps);
    EFFormattedLine(std::cout,"Verlet Step Size:",this->intScheme.dTN,AUTime);
    EFFormattedLine(std::cout,"Number of Midpoint Fock Update per Verlet Step:",this->intScheme.N);
    EFFormattedLine(std::cout,"Number of MMUT per Midpoint Fock Update Step:",this->intScheme.M);

  };

  template<template <typename, typename> class _SSTyp, typename IntsT>
  void Ehrenfest<_SSTyp,IntsT>::printEFStep(double Time) {

    std::cout << std::fixed << std::right;

    std::cout << "Nuc Kinetic Energy is " << kineticEng << std::endl;
  
    // print out energy
    double totalEnergy = this->rt->propagator_->totalEnergy; 

    // add nuclear kinetic energy 
    totalEnergy += kineticEng;

    if(not this->aux_rt)
      std::cout << "Total Energy" << "  " << std::setw(8) << Time << std::setw(16) << totalEnergy << " ";
    else {
      totalEnergy += this->aux_rt->propagator_->OBEnergy;
      totalEnergy += this->aux_rt->propagator_->PPEnergy;
      totalEnergy += this->aux_rt->propagator_->FieldEnergy;
      std::cout << "Total Energy" << std::setw(16) << totalEnergy << " ";
    }
    std::cout << std::endl;

    std::cout << std::setprecision(8);

    // print out nulear position
    std::cout << "Nuclear Position (Angstrom):" << std::endl;
    for(size_t ic = 0; ic < this->nAtoms; ic++) {
      std::cout << std::setw(16) << this->current_x[ic][0] * AngPerBohr;
      std::cout << std::setw(16) << this->current_x[ic][1] * AngPerBohr;
      std::cout << std::setw(16) << this->current_x[ic][2] * AngPerBohr;
      std::cout << std::endl;
    }

    // print out nulear momentum
    std::cout << "Nuclear Momentum:" << std::endl;
    for(size_t ic = 0; ic < this->nAtoms; ic++) {
      std::cout << std::setw(16) << this->current_p[ic][0];
      std::cout << std::setw(16) << this->current_p[ic][1];
      std::cout << std::setw(16) << this->current_p[ic][2];
      std::cout << std::endl;
    }

  };

}; // namespace ChronusQ

#endif
