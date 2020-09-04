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
#include <memmanager.hpp>
#include <basisset.hpp>
#include <molecule.hpp>
#include <fields.hpp>
#include <hamiltonianoptions.hpp>

#include <type_traits>

namespace ChronusQ {

  /**
   *  The operator types to evaluate integrals.
   */
  enum OPERATOR {
    ELECTRON_REPULSION,
    OVERLAP,
    KINETIC,
    NUCLEAR_POTENTIAL,
    LEN_ELECTRIC_MULTIPOLE,
    VEL_ELECTRIC_MULTIPOLE,
    MAGNETIC_MULTIPOLE
  };

  /**
   *  \brief Templated class to handle the evaluation and storage of
   *  one electron integral matrix in a finite basis set.
   */
  class ElectronIntegrals {

  protected:
    size_t NB;
    CQMemManager &memManager_; ///< CQMemManager to allocate matricies

  public:

    // Constructor
    ElectronIntegrals() = delete;
    ElectronIntegrals( const ElectronIntegrals & ) = default;
    ElectronIntegrals( ElectronIntegrals && ) = default;
    ElectronIntegrals(CQMemManager &mem, size_t nb):
        NB(nb), memManager_(mem) {}

    CQMemManager& memManager() const { return memManager_; }
    size_t nBasis() const{ return NB; }

    // Computation interfaces
    /// Evaluate AO Integrals according to a basis set.
    virtual void computeAOInts(BasisSet&, Molecule&, EMPerturbation&,
        OPERATOR, const HamiltonianOptions&) = 0;

    virtual void clear() = 0;

    virtual void output(std::ostream&, const std::string& = "",
                        bool printFull = false) const = 0;

    template <typename TransT>
    static std::shared_ptr<ElectronIntegrals> transform(
        const ElectronIntegrals&, char TRANS, const TransT* T, int NT, int LDT);

    virtual ~ElectronIntegrals() {}

  }; // class ElectronIntegrals

  std::ostream& operator<<(std::ostream &out,
                           const ElectronIntegrals &ints);

}; // namespace ChronusQ
