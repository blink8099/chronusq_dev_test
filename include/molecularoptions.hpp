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
#include <physcon.hpp>

namespace ChronusQ {


  struct MolecularOptions {

    size_t nNuclearSteps;      // number of steps for molecular dynamics
    size_t nMidpointFockSteps; // number of elctronic steps for electronic dynamics
    size_t nElectronicSteps;   // number of elctronic steps for electronic dynamics

    //Molecular Dynamics Options
    double timeStepAU;          // timestep for molecular dynamics in a.u.
    double timeStepFS;    // timestep for molecular dynamics in fs

    MolecularOptions(const double step = 0.05, const size_t nsteps = 2000) :
      timeStepFS(0.05), nNuclearSteps(nsteps) {
      timeStepAU = timeStepFS/FSPerAUTime;
      nMidpointFockSteps = 1;
      nElectronicSteps = 50;
    }
    //Geometry Optimization Options

  }; // struct MolecularOptions

  std::ostream& operator<<(std::ostream&, const MolecularOptions&);

}; // namespace ChronusQ

