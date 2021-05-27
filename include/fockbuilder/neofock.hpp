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

#include <fockbuilder.hpp>

namespace ChronusQ {

  template <typename MatsT, typename IntsT>
  class NEOFockBuilder: public FockBuilder<MatsT,IntsT> {

    template<typename MatsU, typename IntsU>
    friend class NEOFockBuilder;

    protected:

      // "Other" single slater with which to contract
      SingleSlater<MatsT,IntsT>* aux_ss = nullptr;
      SquareMatrix<MatsT>* outMat = nullptr;
      TPIContractions<MatsT,IntsT>* contraction = nullptr;

    public:

    //
    // Constructors
    // XXX: Constructors do NOT populate the protected members - not even the
    //      copy/move (since they can't convert types)
    //
    NEOFockBuilder() = delete;
    NEOFockBuilder(HamiltonianOptions hamiltonianOptions) :
      FockBuilder<MatsT,IntsT>(hamiltonianOptions) { }

    // Other type constructors
    template <typename MatsU>
    NEOFockBuilder(const NEOFockBuilder<MatsU,IntsT> &other ) : 
      FockBuilder<MatsT,IntsT>( dynamic_cast<const FockBuilder<MatsU,IntsT>&>(other) )
      { }
    template <typename MatsU>
    NEOFockBuilder(FockBuilder<MatsU,IntsT> &&other ) :
      FockBuilder<MatsT,IntsT>( dynamic_cast<FockBuilder<MatsU,IntsT>&&>(other) )
      { }

    // Setters
    void setAux(SingleSlater<MatsT,IntsT>* ss) {
      aux_ss = ss;
    }

    void setOutput(SquareMatrix<MatsT>* out) {
      outMat = out;
    }

    void setContraction(TPIContractions<MatsT,IntsT>* cont) {
      contraction = cont;
    }

    // Inter-SingleSlater interaction
    void formepJ(SingleSlater<MatsT,IntsT>&, bool increment = false);

    // Interface method
    virtual void formFock(SingleSlater<MatsT,IntsT>&, EMPerturbation&,
      bool increment = false, double xHFX = 1.);

  };

}

#include <fockbuilder/neofock/impl.hpp>
