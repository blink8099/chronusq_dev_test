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
#include <aointegrals.hpp>

namespace ChronusQ {

  /**
   *  \brief Class to use in-house CQ GTO integral engine
   *
   *  Libint integrals are done in AOIntegrals
   */
  class GTOIntegrals : public AOIntegrals<double>{

    private:

    public:

  }; // class CQGTOIntegrals

  /**
   *  \brief Class to handle the evaluation and storage of 
   *  integral matrices representing quantum mechanical operators in
   *  a finite Gauge-Including Atomic Orbital (GIAO) basis.
   *
   *  Real-valued arithmetics are kept in AOIntegrals
   */
  class GIAOIntegrals : public AOIntegrals<dcomplex> {
    private:

    // 1-e builders for in-house integral code
    template <size_t NOPER, bool SYMM, typename F>
    std::vector<dcomplex*> OneEDriverLocalGIAO(const F&,std::vector<libint2::Shell>&);

    public:

    /**
     *  GIAOIntegrals Constructor. Constructs a GIAOIntegrals object.
     *
     *  \param [in] memManager Memory manager for matrix allocation
     *  \param [in] mol        Molecule object for molecular specification
     *  \param [in] basis      The GTO basis for integral evaluation
     */ 
    GIAOIntegrals(CQMemManager &memManager, Molecule &mol, BasisSet &basis) :
      AOIntegrals<dcomplex>(memManager,mol,basis){}

    // Integral evaluation
    virtual void computeAOOneE(EMPerturbation&,OneETerms&);     // Evaluate the 1-e ints (general)
    virtual void computeERI(EMPerturbation&);                   // Evaluate ERIs (general)

    void computeAOOneEGIAO(EMPerturbation&,OneETerms&); // Evaluate the 1-e ints in the GIAO basis
    void computeERIGIAO(EMPerturbation&);               // Evaluate ERIs in the GIAO basis

  }; // class GIAOIntegrals


}; // namespace ChronusQ

