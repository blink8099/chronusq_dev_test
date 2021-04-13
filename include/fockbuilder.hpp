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

#include <fields.hpp>
#include <singleslater.hpp>
#include <singleslater/neo_singleslater.hpp>

namespace ChronusQ {

  /**
   * \brief The FockBuilder class
   */
  template <typename MatsT, typename IntsT>
  class FockBuilder {

    template <typename MatsU, typename IntsU>
    friend class FockBuilder;

  protected:
    HamiltonianOptions hamiltonianOptions_; ///< One electron terms to be computed

  public:

    // Constructors
    FockBuilder() = delete;
    FockBuilder(HamiltonianOptions hamiltonianOptions):
      hamiltonianOptions_(hamiltonianOptions) {}

    // Different type
    template <typename MatsU>
    FockBuilder(const FockBuilder<MatsU,IntsT> &);
    template <typename MatsU>
    FockBuilder(FockBuilder<MatsU,IntsT> &&);

    // Virtual destructor
    virtual ~FockBuilder() {}


    // Public member functions
    const HamiltonianOptions& getHamiltonianOptions() const {
      return hamiltonianOptions_;
    }

    // Form the Hartree-Fock perturbation tensor (see include/fockbuilder/impl.hpp for docs)
    virtual void formGD(SingleSlater<MatsT,IntsT> &, EMPerturbation &, bool increment = false, double xHFX = 1.);

    // Form the EPJ contribution to Fock matrix (see include/fockbuilder/impl.hpp for docs)
    virtual void formepJ(NEOSingleSlater<MatsT,IntsT> &, NEOSingleSlater<MatsT,IntsT> &, bool increment = false, double xHFX = 1.);

    // Form a fock matrix (see include/fockbuilder/impl.hpp for docs)
    virtual void formFock(SingleSlater<MatsT,IntsT> &, EMPerturbation &, bool increment = false, double xHFX = 1.);

    // Compute the gradient
    virtual void getGrad() {}

    // Pointer convertor
    template <typename MatsU>
    static std::shared_ptr<FockBuilder<MatsU,IntsT>>
    convert(const std::shared_ptr<FockBuilder<MatsT,IntsT>>&);


  };

}
