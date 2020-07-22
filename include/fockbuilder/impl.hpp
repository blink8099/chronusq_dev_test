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
    oper_t_coll &contract1PDM  = increment ? ss.deltaOnePDM : ss.onePDM;

    size_t NB = ss.aoints.basisSet().nBasis;
    size_t NB2 = NB*NB;

    size_t mpiRank   = MPIRank(ss.comm);
    bool   isNotRoot = mpiRank != 0;


    // Zero out J
    if(not increment) memset(ss.coulombMatrix,0.,NB2*sizeof(MatsT));

    std::vector<TwoBodyContraction<MatsT>> contract =
      { {true, contract1PDM[SCALAR], ss.coulombMatrix, true, COULOMB} };

    // Determine how many (if any) exchange terms to calculate
    if( std::abs(xHFX) > 1e-12 )
    for(auto i = 0; i < ss.exchangeMatrix.size(); i++) {
      contract.push_back(
          {true, contract1PDM[i], ss.exchangeMatrix[i], true, EXCHANGE}
      );

      // Zero out K[i]
      if(not increment) memset(ss.exchangeMatrix[i],0,NB2*sizeof(MatsT));
    }

    ss.aoints.twoBodyContract(ss.comm, contract, pert);

    ROOT_ONLY(ss.comm); // Return if not root (J/K only valid on root process)


    // Form GD: G[D] = 2.0*J[D] - K[D]
    if( std::abs(xHFX) > 1e-12 ) {
      for(auto i = 0; i < ss.exchangeMatrix.size(); i++)
        MatAdd('N','N', NB, NB, MatsT(0.), ss.twoeH[i], NB, MatsT(-xHFX), ss.exchangeMatrix[i], NB, ss.twoeH[i], NB);
    } else {
      for(auto i = 0; i < ss.fockMatrix.size(); i++) memset(ss.twoeH[i],0,NB2*sizeof(MatsT));
    }
    // G[D] += 2*J[D]
    MatAdd('N','N', NB, NB, MatsT(1.), ss.twoeH[SCALAR], NB, MatsT(2.), ss.coulombMatrix, NB, ss.twoeH[SCALAR], NB);

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

    size_t NB = ss.aoints.basisSet().nBasis;
    size_t NB2 = NB*NB;

    auto GDStart = tick(); // Start time for G[D]

    // Form G[D]
    formGD(ss,pert,increment,xHFX);

    ss.GDDur = tock(GDStart); // G[D] Duraction

    ROOT_ONLY(ss.comm);


    // Zero out the Fock
    for(auto &F : ss.fockMatrix) std::fill_n(F,NB2,MatsT(0.));

    // Copy over the Core Hamiltonian
    for(auto i = 0; i < ss.coreH.size(); i++)
      SetMat('N',NB,NB,MatsT(1.),ss.coreH[i],NB,ss.fockMatrix[i],NB);

    // Add in the two electron term
    for(auto i = 0ul; i < ss.fockMatrix.size(); i++)
      MatAdd('N','N', NB, NB, MatsT(1.), ss.fockMatrix[i], NB, MatsT(1.), ss.twoeH[i], NB, ss.fockMatrix[i], NB);




    // Add in the electric field contributions
    // FIXME: the magnetic field contribution should go here as well to allow for RT
    // manipulation

    if( pert_has_type(pert,Electric) ) {

      auto dipAmp = pert.getDipoleAmp(Electric);

      // Because IntsT and MatsT need not be the same: MatAdd not possible
      for(auto i = 0;    i < 3;     i++)
      for(auto k = 0ul;  k < NB*NB; k++)
        ss.fockMatrix[SCALAR][k] -=
          2. * dipAmp[i] * ss.aoints.lenElecDipole[i][k];

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

    const std::type_index tIndex(typeid(*fb));

    if (tIndex == std::type_index(typeid(ROFock<MatsT,IntsT>))) {
      return std::make_shared<ROFock<MatsU,IntsT>>(
               *std::dynamic_pointer_cast<ROFock<MatsT,IntsT>>(fb));

    } else {
      return std::make_shared<FockBuilder<MatsU,IntsT>>(
               *std::dynamic_pointer_cast<FockBuilder<MatsT,IntsT>>(fb));
    }

  } // FockBuilder<MatsT,IntsT>::convert

}; // namespace ChronusQ
