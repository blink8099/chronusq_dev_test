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

#include <chronusq_sys.hpp>
#include <quantum/base.hpp>
#include <aointegrals.hpp>

namespace ChronusQ {

  /**
   *  \brief The WaveFunctionBase class. The abstraction of information
   *  relating to the WaveFunction class which are independent of storage
   *  type.
   *
   *  Specializes QuantumBase interface.
   *
   *  See WaveFunction for further docs.
   */ 
  class WaveFunctionBase : virtual public QuantumBase {

  public:
    // Member data


    size_t nO;  ///< Total number of occupied orbitals
    size_t nV;  ///< Total number of virtual orbitals
    size_t nOA; ///< Number of occupied alpha orbitals (nC == 1)
    size_t nOB; ///< Number of occupied beta orbitals  (nC == 1)
    size_t nVA; ///< Number of virtual alpha orbitals  (nC == 1)
    size_t nVB; ///< Number of virtual beta orbitals   (nC == 1)

    
    // Disable default constructor
    WaveFunctionBase() = delete;

    // Default copy and move constructors
    WaveFunctionBase(const WaveFunctionBase &) = default;
    WaveFunctionBase(WaveFunctionBase &&)      = default;



    /**
     *  WaveFunctionBase Constructor. Constructs a WaveFunctionBase object
     *
     *  \param [in] aoi  AOIntegrals object (which handels the BasisSet, etc)
     *  \param [in] _nC  Number of spin components (1 and 2 are supported)
     *  \param [in] iCS  Whether or not to treat as closed shell
     */ 
    WaveFunctionBase(MPI_Comm c, CQMemManager &mem, size_t _nC, bool iCS) : 
      QuantumBase(c, mem,_nC,iCS) { }; // WaveFunctionBase ctor 

    // Print Functions
    virtual void printMO(std::ostream&)  = 0;
    virtual void printEPS(std::ostream&) = 0;

    virtual void printMOInfo(std::ostream&) = 0;

  }; // class WaveFunctionBase

}; // namespace ChronusQ

