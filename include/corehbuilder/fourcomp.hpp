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

#include <corehbuilder.hpp>

namespace ChronusQ {

  /**
   * \brief The FourComponent class
   */
  template <typename MatsT, typename IntsT>
  class FourComponent : public CoreHBuilder<MatsT,IntsT> {
  protected:

  public:

    // Constructors

    // Disable default constructor
    FourComponent() = delete;
    FourComponent(Integrals<IntsT> &aoints):
      CoreHBuilder<MatsT,IntsT>(aoints, {true,true,true}) {}

    // Same or Different type
    template <typename MatsU>
    FourComponent(const FourComponent<MatsU,IntsT> &other):
      CoreHBuilder<MatsT,IntsT>(other) {}
    template <typename MatsU>
    FourComponent(FourComponent<MatsU,IntsT> &&other):
      CoreHBuilder<MatsT,IntsT>(other) {}

    // Virtual destructor
    virtual ~FourComponent() {}

    // Public member functions

    // Compute core Hamitlonian
    virtual void computeCoreH(EMPerturbation&,
        std::shared_ptr<PauliSpinorSquareMatrices<MatsT>>) {
      CErr("4C NYI",std::cout);
    }

    // Compute the gradient
    virtual void getGrad() {
      CErr("4C CoreH gradient NYI",std::cout);
    }

  };

}
