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

#include <fockbuilder/neofock.hpp>

namespace ChronusQ {

  template <typename MatsT, typename IntsT>
  void NEOFockBuilder<MatsT,IntsT>::formepJ(SingleSlater<MatsT,IntsT>& ss,
    bool increment)
  {
    // Validate internal pointers
    if( aux_ss == nullptr )
      CErr("aux_ss uninitialized in formepJ!");
    if( outMat == nullptr )
      CErr("outMat uninitialized in formepJ!");
    if( contraction == nullptr )
      CErr("contraction uninitialized in formepJ!");

    // Decide onePDM to use
    PauliSpinorSquareMatrices<MatsT>& contract1PDM
        = increment ? *aux_ss->deltaOnePDM : *aux_ss->onePDM;

    size_t NB = ss.basisSet().nBasis;

    // Zero out J and K[i]
    if(not increment)
      outMat->clear();
  
    std::vector<TwoBodyContraction<MatsT>> contract =
      { {contract1PDM.S().pointer(), outMat->pointer(), true, COULOMB} };

    EMPerturbation pert;
    contraction->twoBodyContract(ss.comm, contract, pert);
  }

  template <typename MatsT, typename IntsT>
  void NEOFockBuilder<MatsT,IntsT>::formFock(
    SingleSlater<MatsT,IntsT>& ss, EMPerturbation& empert, bool increment,
    double xHFX)
  {
    FockBuilder<MatsT,IntsT>::formFock(ss, empert, increment, xHFX);

    formepJ(ss, increment);

    *ss.twoeH -= 2. * *outMat;
    *ss.fockMatrix -= 2. * *outMat;
  }

}
