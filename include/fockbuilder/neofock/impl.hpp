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

#include <particleintegrals/twopints/incore4indextpi.hpp>
#include <particleintegrals/twopints/gtodirecttpi.hpp>
#include <particleintegrals/gradints/direct.hpp>
#include <particleintegrals/gradints/incore.hpp>

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
  std::vector<double> NEOFockBuilder<MatsT,IntsT>::formepJGrad(
    SingleSlater<MatsT,IntsT>& ss, EMPerturbation& pert, double xHFX) {

    if( aux_ss == nullptr )
      CErr("aux_ss uninitialized in formepJGrad!");
    if( gradTPI == nullptr )
      CErr("gradTPI uninitialized in formepJGrad!");

    size_t NB = ss.basisSet().nBasis;
    size_t nGrad = 3*ss.molecule().nAtoms;
    CQMemManager& mem = ss.memManager;

    // Form contraction
    std::unique_ptr<GradContractions<MatsT,IntsT>> contract = nullptr;
    if ( std::dynamic_pointer_cast<InCore4indexTPI<IntsT>>((*gradTPI)[0]) ) {
      contract = std::make_unique<InCore4indexGradContraction<MatsT,IntsT>>(*gradTPI);
    }
    else if ( std::dynamic_pointer_cast<DirectTPI<IntsT>>((*gradTPI)[0]) ) {
      contract = std::make_unique<DirectGradContraction<MatsT,IntsT>>(*gradTPI);
    }
    else
      CErr("Gradients of RI NYI!");
    // Assume that the order of the TPI is the same between the regular and
    //   gradient integrals
    contract->contractSecond = contraction->contractSecond;

    // Create contraction list
    std::vector<std::vector<TwoBodyContraction<MatsT>>> cList;

    std::vector<SquareMatrix<MatsT>> JList;
    JList.reserve(nGrad);

    for( auto iGrad = 0; iGrad < nGrad; iGrad++ ) {
      std::vector<TwoBodyContraction<MatsT>> tempCont;

      // Coulomb
      JList.emplace_back(mem, NB);
      JList.back().clear();
      tempCont.push_back(
         {aux_ss->onePDM->S().pointer(), JList.back().pointer(), true, COULOMB}
      );

      cList.push_back(tempCont);
    }

    // Contract to J/K
    contract->gradTwoBodyContract(MPI_COMM_WORLD, true, cList, pert);

    // Contract to gradient
    std::vector<double> gradient;
    PauliSpinorSquareMatrices<MatsT> twoEGrad(mem, NB, false, false);

    for( auto iGrad = 0; iGrad < nGrad; iGrad++ ) {

      // G[S] = -2 * J
      // TODO: The negative here is implicitly taking care of the
      //   electron/proton charges.
      twoEGrad.S() = -2. * JList[iGrad];

      double gradVal = ss.template computeOBProperty<double,SCALAR>(
        twoEGrad.S().pointer()
      );
      gradient.push_back(0.25*gradVal);

    }

    return gradient; 

  }

  template <typename MatsT, typename IntsT>
  void NEOFockBuilder<MatsT,IntsT>::formFock(
    SingleSlater<MatsT,IntsT>& ss, EMPerturbation& empert, bool increment,
    double xHFX)
  {
    if( upstream == nullptr )
      CErr("Upstream FockBuilder uninitialized in formFock!");

    // Call all upstream FockBuilders
    upstream->formFock(ss, empert, increment, xHFX);

    formepJ(ss, increment);

    *ss.twoeH -= 2. * *outMat;
    *ss.fockMatrix -= 2. * *outMat;
  }

  template <typename MatsT, typename IntsT>
  std::vector<double> NEOFockBuilder<MatsT,IntsT>::getGDGrad(
    SingleSlater<MatsT,IntsT>& ss, EMPerturbation& pert, double xHFX) {
    if( upstream == nullptr )
      CErr("Upstream FockBuilder uninitialized in getGDGrad!");

    size_t nGrad = 3*ss.molecule().nAtoms;

    std::vector<double> gradient = upstream->getGDGrad(ss, pert, xHFX);
    std::vector<double> epjGrad = formepJGrad(ss, pert, xHFX);

    std::transform(gradient.begin(), gradient.end(), epjGrad.begin(),
                   gradient.begin(), std::plus<double>());

    return gradient;

  };

}
