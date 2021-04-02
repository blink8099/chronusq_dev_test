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
#include <util/time.hpp>
#include <cqlinalg.hpp>
#include <matrix.hpp>
#include <electronintegrals/twoeints/incorerieri.hpp>
#include <electronintegrals/gradints/incore.hpp>
#include <electronintegrals/gradints/direct.hpp>
#include <fockbuilder/rofock/impl.hpp>

#include <typeinfo>

namespace ChronusQ {

  /**
   *  Constructs a FockBuilder object from another of a another (possibly the
   *  same) type by copy.
   *
   *  \param [in] other FockBuilder object to copy
   */
  template <typename MatsT, typename IntsT>
  template <typename MatsU>
  FockBuilder<MatsT,IntsT>::FockBuilder(const FockBuilder<MatsU,IntsT> &other) {}

  /**
   *  Constructs a FockBuilder object from another of a another (possibly the
   *  same) by move.
   *
   *  \warning Deallocates the passed FockBuilder object
   *
   *  \param [in] other FockBuilder object to move
   */
  template <typename MatsT, typename IntsT>
  template <typename MatsU>
  FockBuilder<MatsT,IntsT>::FockBuilder(FockBuilder<MatsU,IntsT> &&other) {}

  /**
   *  \brief Forms the Hartree-Fock perturbation tensor
   *
   *  Populates / overwrites GD storage (and JScalar and K storage)
   */
  template <typename MatsT, typename IntsT>
  void FockBuilder<MatsT,IntsT>::formGD(SingleSlater<MatsT,IntsT> &ss,
    EMPerturbation &pert, bool increment, double xHFX) {

    typedef MatsT*                    oper_t;
    typedef std::vector<oper_t>       oper_t_coll;

    // Decide list of onePDMs to use
    PauliSpinorSquareMatrices<MatsT> &contract1PDM
        = increment ? *ss.deltaOnePDM : *ss.onePDM;

    size_t NB = ss.basisSet().nBasis;

    // Zero out J and K[i]
    if(not increment) {
      ss.coulombMatrix->clear();
      ss.exchangeMatrix->clear();
    }

    std::vector<TwoBodyContraction<MatsT>> contract =
      { {contract1PDM.S().pointer(), ss.coulombMatrix->pointer(), true, COULOMB} };

    // Determine how many (if any) exchange terms to calculate
    if( std::abs(xHFX) > 1e-12 and not increment and
        (ss.scfControls.guess != SAD or ss.scfConv.nSCFIter != 0) and
        std::dynamic_pointer_cast<InCoreRIERIContraction<MatsT, IntsT>>(ss.ERI)) {
      ROOT_ONLY(ss.comm);
      auto rieri = std::dynamic_pointer_cast<InCoreRIERIContraction<MatsT, IntsT>>(ss.ERI);

      if(ss.nC == 1) {
        SquareMatrix<MatsT> AAblock(ss.exchangeMatrix->memManager(), NB);
        rieri->KCoefContract(ss.comm, ss.nOA, ss.mo[0].pointer(), AAblock.pointer());
        if(ss.iCS) {
          *ss.exchangeMatrix = PauliSpinorSquareMatrices<MatsT>::spinBlockScatterBuild(AAblock);
        } else {
          SquareMatrix<MatsT> BBblock(ss.exchangeMatrix->memManager(), NB);
          rieri->KCoefContract(ss.comm, ss.nOB, ss.mo[1].pointer(), BBblock.pointer());
          *ss.exchangeMatrix = PauliSpinorSquareMatrices<MatsT>::spinBlockScatterBuild(AAblock, BBblock);
        }
      } else {
        SquareMatrix<MatsT> spinBlockForm(ss.exchangeMatrix->memManager(), NB*2);
        InCoreRIERI<MatsT> rierispinor(dynamic_cast<InCoreRIERI<IntsT>&>(rieri->ints()).
                                       template spatialToSpinBlock<MatsT>());
        InCoreRIERIContraction<MatsT,MatsT>(rierispinor)
            .KCoefContract(ss.comm, ss.nO, ss.mo[0].pointer(), spinBlockForm.pointer());
        *ss.exchangeMatrix = spinBlockForm.template spinScatter<MatsT>();
      }

    } else if( std::abs(xHFX) > 1e-12 ) {
      contract.push_back(
          {contract1PDM.S().pointer(), ss.exchangeMatrix->pointer(), true, EXCHANGE}
      );
      if (ss.exchangeMatrix->hasZ())
        contract.push_back(
            {contract1PDM.Z().pointer(), ss.exchangeMatrix->Z().pointer(), true, EXCHANGE}
        );
      if (ss.exchangeMatrix->hasXY()) {
        contract.push_back(
            {contract1PDM.Y().pointer(), ss.exchangeMatrix->Y().pointer(), true, EXCHANGE}
        );
        contract.push_back(
            {contract1PDM.X().pointer(), ss.exchangeMatrix->X().pointer(), true, EXCHANGE}
        );
      }
    }

    ss.ERI->twoBodyContract(ss.comm, contract, pert);

    ROOT_ONLY(ss.comm); // Return if not root (J/K only valid on root process)


    // Form GD: G[D] = 2.0*J[D] - K[D]
    if( std::abs(xHFX) > 1e-12 ) {
      *ss.twoeH = -xHFX * *ss.exchangeMatrix;
    } else {
      ss.twoeH->clear();
    }
    // G[D] += 2*J[D]
    *ss.twoeH += 2.0 * *ss.coulombMatrix;

#if 0
  //printJ(std::cout);
    printK(std::cout);
  //printGD(std::cout);
#endif

  } // FockBuilder::formGD

  /**
   *  \brief Forms the Fock matrix for a single slater determinant using
   *  the 1PDM.
   *
   *  \param [in] increment Whether or not the Fock matrix is being
   *  incremented using a previous density
   *
   *  Populates / overwrites fock strorage in SingleSlater &ss
   */
  template <typename MatsT, typename IntsT>
  void FockBuilder<MatsT,IntsT>::formFock(SingleSlater<MatsT,IntsT> &ss,
    EMPerturbation &pert, bool increment, double xHFX) {

    size_t NB = ss.basisSet().nBasis;
    size_t NB2 = NB*NB;

    auto GDStart = tick(); // Start time for G[D]

    // Form G[D]
    formGD(ss,pert,increment,xHFX);

    ss.GDDur = tock(GDStart); // G[D] Duraction

    ROOT_ONLY(ss.comm);

    // Form Fock
    *ss.fockMatrix = *ss.coreH + *ss.twoeH;

    // Add in the electric field contributions
    // FIXME: the magnetic field contribution should go here as well to allow for RT
    // manipulation

    if( pert_has_type(pert,Electric) ) {

      auto dipAmp = pert.getDipoleAmp(Electric);

      for(auto i = 0;    i < 3;     i++)
        ss.fockMatrix->S() -=
          2. * dipAmp[i] * (*ss.aoints.lenElectric)[i].matrix();

    }

#if 0
    ss.printFock(std::cout);
#endif
  }

  /**
   *  \brief The pointer convertor. This static function converts
   *  the underlying polymorphism correctly to hold a different
   *  type of matrices. It is called when the corresponding
   *  SingleSlater object is being converted.
   */
  template <typename MatsT, typename IntsT>
  template <typename MatsU>
  std::shared_ptr<FockBuilder<MatsU,IntsT>>
  FockBuilder<MatsT,IntsT>::convert(const std::shared_ptr<FockBuilder<MatsT,IntsT>>& fb) {

    if (not fb) return nullptr;

    const std::type_info &tID(typeid(*fb));

    if (tID == typeid(ROFock<MatsT,IntsT>)) {
      return std::make_shared<ROFock<MatsU,IntsT>>(
               *std::dynamic_pointer_cast<ROFock<MatsT,IntsT>>(fb));

    } else {
      return std::make_shared<FockBuilder<MatsU,IntsT>>(
               *std::dynamic_pointer_cast<FockBuilder<MatsT,IntsT>>(fb));
    }

  } // FockBuilder<MatsT,IntsT>::convert

  
  template <typename MatsT, typename IntsT>
  std::vector<double> FockBuilder<MatsT,IntsT>::getGDGrad(
    SingleSlater<MatsT,IntsT>& ss, EMPerturbation& pert, double xHFX) {

    size_t NB = ss.basisSet().nBasis;
    size_t nGrad = 3*ss.molecule().nAtoms;
    CQMemManager& mem = ss.memManager;

    bool hasXY = ss.exchangeMatrix->hasXY();
    bool hasZ = ss.exchangeMatrix->hasZ();

    if( not ss.aoints.gradERI )
      CErr("Gradient ERI missing in FockBuilder::getGDGrad!");

    GradInts<TwoEInts,IntsT>& gradERI = *ss.aoints.gradERI;

    // Form contraction
    // TODO: There's gotta be a better way to do this...
    std::unique_ptr<GradContractions<MatsT,IntsT>> contract = nullptr;
    if ( std::dynamic_pointer_cast<InCore4indexERI<IntsT>>(gradERI[0]) ) {
      contract = std::make_unique<InCore4indexGradContraction<MatsT,IntsT>>(gradERI);
    }
    else if ( std::dynamic_pointer_cast<DirectERI<IntsT>>(gradERI[0]) ) {
      contract = std::make_unique<DirectGradContraction<MatsT,IntsT>>(gradERI);
    }
    else
      CErr("Gradients of RI NYI!");

    // Create contraction list
    std::vector<std::vector<TwoBodyContraction<MatsT>>> cList;

    std::vector<SquareMatrix<MatsT>> JList;
    std::vector<PauliSpinorSquareMatrices<MatsT>> KList;

    JList.reserve(nGrad);
    KList.reserve(nGrad);

    for( auto iGrad = 0; iGrad < nGrad; iGrad++ ) {
      std::vector<TwoBodyContraction<MatsT>> tempCont;

      // Coulomb
      JList.emplace_back(mem, NB);
      tempCont.push_back(
         {ss.onePDM->S().pointer(), JList.back().pointer(), true, COULOMB}
      );

      // Exchange
      if( std::abs(xHFX) > 1e-12 ) {

        KList.emplace_back(mem, NB, hasXY, hasZ);

        tempCont.push_back(
          {ss.onePDM->S().pointer(), KList.back().S().pointer(), true, EXCHANGE}
        );

        if (hasZ) {
          tempCont.push_back(
            {ss.onePDM->Z().pointer(), KList.back().Z().pointer(), true, EXCHANGE}
          );
        }
        if (hasXY) {
          tempCont.push_back(
            {ss.onePDM->Y().pointer(), KList.back().Y().pointer(), true, EXCHANGE}
          );
          tempCont.push_back(
            {ss.onePDM->X().pointer(), KList.back().X().pointer(), true, EXCHANGE}
          );
        }
      }


      cList.push_back(tempCont);
    }

    // Contract to J/K
    contract->gradTwoBodyContract(MPI_COMM_WORLD, true, cList, pert);

    // Contract to gradient
    std::vector<double> gradient;
    PauliSpinorSquareMatrices<MatsT> twoEGrad(mem, NB, hasXY, hasZ);

    for( auto iGrad = 0; iGrad < nGrad; iGrad++ ) {

      // Scale K by alpha
      twoEGrad = -xHFX * KList[iGrad];

      // G[S] = 2 * J[S] + alpha * K[S]
      twoEGrad.S() += 2. * JList[iGrad];

      double gradVal = ss.template computeOBProperty<double,SCALAR>(
        twoEGrad.S().pointer()
      );
      if( hasZ )
        gradVal += ss.template computeOBProperty<double,MZ>(
          twoEGrad.Z().pointer()
        );
      if( hasXY ) {
        gradVal += ss.template computeOBProperty<double,MY>(
          twoEGrad.Y().pointer()
        );
        gradVal += ss.template computeOBProperty<double,MX>(
          twoEGrad.X().pointer()
        );
      }

      gradient.push_back(0.25*gradVal);

    }

    return gradient;

  } // FockBuilder::getGDGrad

}; // namespace ChronusQ
