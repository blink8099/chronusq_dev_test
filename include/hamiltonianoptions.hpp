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
#include <basisset.hpp>


namespace ChronusQ {

  enum KineticBalance {RKBPauli, RKBSpinor, UKBScalar};

  struct HamiltonianOptions {

    // Number of components
    size_t nComponents = 1; // 1, 2, or 4

    // Time-reversal symmetry
    bool KramersSymmetry = false; // Whether or not to use the Kramers' symmetry

    // Integral Options
    BASIS_FUNCTION_TYPE basisType = REAL_GTO; //GTO or GIAO
    bool finiteWidthNuc = false; // Use finite nuclei in integral evaluations
    bool Libcint = false; // Use Libcint library instead of Libint

    // One-Component Options
    bool PerturbativeScalarRelativity = false; // Add perturbative scalar relativity
    bool PerturbativeSpinOrbit = false; // Add perturbative spin-orbit

    // Two-Component Options
    bool OneEScalarRelativity = true; //scalar relativity
    bool OneESpinOrbit = true; //spin-orbit relativity
    bool Boettger = true; // Use Boetteger factor to scale one-electron spin-orbit
    bool AtomicMeanField = false; // Use atomic mean field two-electron spin-orbit

    // Four-Component Options
    KineticBalance kineticBalance = RKBPauli; // Choose the kinetic-balance condition; currently, only RKBPauli is implemented
    bool BareCoulomb = true; // Do bare Coulomb only in Restricted-Kinetic balance (RKB)
    bool DiracCoulomb = true; // Dirac-Coulomb without SSSS
    bool DiracCoulombSSSS = false; // SSSS to Dirac-Coulomb
    bool Gaunt = false; // Gaunt
    bool Gauge = false; // Gauge

  }; // struct HamiltonianOptions

  std::ostream& operator<<(std::ostream&, const HamiltonianOptions&);

}; // namespace ChronusQ

