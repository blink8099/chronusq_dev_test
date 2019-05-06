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
#ifndef __INCLUDED_AOINTEGRALS_PRINT_HPP__
#define __INCLUDED_AOINTEGRALS_PRINT_HPP__
#include <aointegrals.hpp>
#include <cxxapi/output.hpp>


namespace ChronusQ {

  std::ostream& operator<<(std::ostream &out, const AOIntegralsBase &aoints) {

    out << "\nIntegral Engine Settings:\n" << BannerTop << "\n\n" ;
    out << std::left;


    // XXX: Hard code this for now
    out << "  " << std::setw(28) << "ERI Engine:" << "Libint2" << std::endl;
    out << "  " << std::setw(28) << "One-Body Engine:" 
        << "Libint2 + In-House" << std::endl;


/*
    out << std::endl;
    out << "  " << std::setw(28) << "Core Hamiltonian:";
    if(aoints.coreType == NON_RELATIVISTIC) 
      out << "Non-Relativistic";
    else                      
      out << "Relativistic (X2C)";
    out << std::endl;
    
    if(aoints.coreType == RELATIVISTIC_X2C_1E)
      out << "    * Using Finite Width Gaussian Nuclei\n\n";
*/


    out << std::endl;
    out << "  Property Integrals:\n";
    out << "    * Will Compute Length Gauge Electric Multipoles up to Octupole"
        << std::endl;
    out << "    * Will Compute Velocity Gauge Electric Multipoles up to Octupole"
        << std::endl;
    out << "    * Will Compute Magnetic Multipoles up to Quadrupole"
        << std::endl;
    out << std::endl;


    out << std::endl;
    out << "  " << std::setw(28) << "ERI Contraction Algorithm:";
    if(aoints.contrAlg == INCORE) out << "INCORE (Gemm)";
    else                      out << "DIRECT";
    out << std::endl;

    if( aoints.contrAlg == DIRECT )
      out << "    * Schwartz Screening Threshold = " 
          << aoints.threshSchwartz << "\n";
    

    out << std::endl << BannerEnd << std::endl;

    return out; // return std::ostream reference

  }; // operator<<(AOIntegrals)

}; // namespace ChronusQ

#endif
