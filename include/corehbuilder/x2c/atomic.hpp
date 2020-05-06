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

#include <corehbuilder/x2c.hpp>

namespace ChronusQ {

  /**
   *  \brief The AtomicX2C class. A class to compute X2C Core Hamiltonian.
   *  Stores intermediate matrices.
   */
  template <typename MatsT, typename IntsT>
  class AtomicX2C : public X2C<MatsT, IntsT> {
  protected:
    X2C_TYPE type_ = {true, true, false};
    std::vector<X2C<MatsT, IntsT>> atoms_;
  private:
  public:

    // Constructors

    // Disable default constructor
    AtomicX2C() = delete;

    // Default copy and move constructors
    AtomicX2C(const AtomicX2C &) = default;
    AtomicX2C(AtomicX2C &&)      = default;

    /**
     * \brief Constructor
     *
     *  \param [in] memManager Memory manager for matrix allocation
     *  \param [in] mol        Molecule object for molecular specification
     *  \param [in] basis      The GTO basis for integral evaluation
     */
    AtomicX2C(CQMemManager &mem, const Molecule &mol, const BasisSet &basis) :
      X2C<MatsT, IntsT>(mem, mol, basis) { ; }


    // Public Member functions

    void computeCoreH(EMPerturbation& emPert, std::vector<MatsT*>& CH);

  };

}; // namespace ChronusQ
