/* 
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *  
 *  Copyright (C) 2014-2020 Li Research Group (University of Washington)
 *  
 *  This program is free software; you ca redistribute it and/or modify
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

#include <singleslater/neoss.hpp>
#include <cerr.hpp>

namespace ChronusQ {

#define TRY_REF(_REF, input, output) \
  if( output == nullptr ) \
    if( auto casted = std::dynamic_pointer_cast<_REF<MatsU,IntsT>>(input) ) \
      output = std::dynamic_pointer_cast<SingleSlater<MatsT,IntsT>>( \
        std::make_shared< _REF<MatsT,IntsT> >(*casted) \
      );

  // Helper function to handle determining type of subsystem
  template <typename MatsT, typename IntsT, typename MatsU>
  std::shared_ptr<SingleSlater<MatsT,IntsT>> makeNewSS(const std::shared_ptr<SingleSlater<MatsU,IntsT>>& old_ss) {
    std::shared_ptr<SingleSlater<MatsT,IntsT>> new_ss = nullptr;

    TRY_REF(KohnSham, old_ss, new_ss);
    TRY_REF(HartreeFock, old_ss, new_ss);

    if( new_ss == nullptr )
      CErr("Unrecognized reference in constructing NEOSS!");

    return new_ss;
  };

  template <typename MatsT, typename IntsT>
  template <typename MatsU>
  NEOSS<MatsT,IntsT>::NEOSS(const NEOSS<MatsU,IntsT>& other, int dummy) :
    SingleSlater<MatsT,IntsT>(dynamic_cast<const SingleSlater<MatsU,IntsT>&>(other), dummy),
    WaveFunctionBase(dynamic_cast<const WaveFunctionBase&>(other)),
    QuantumBase(dynamic_cast<const QuantumBase&>(other)),
    order_(other.order_)
  {
    // Loop over all old subsystems
    for( auto& x: other.subsystems ) {

      // Easier to read names
      auto& label = x.first;
      auto& old_sys = x.second;

      // Find a non-NEO fockbuilder
      auto baseFock = old_sys->fockBuilder.get();
      if( auto p = dynamic_cast<NEOFockBuilder<MatsU,IntsT>*>(baseFock) ) {
        baseFock = p->getNonNEOUpstream();
      }

      // Assume just non-relativistic Fock here
      // TODO: Make this general to all fockbuilders
      auto newFock = std::make_shared<FockBuilder<MatsT,IntsT>>(*baseFock);

      // Get new SingleSlater object
      SubSSPtr new_sys = makeNewSS<MatsT>(old_sys);
      new_sys->fockBuilder = newFock;

      // Add the new subsystem to the current object
      addSubsystem(label, new_sys, other.interIntegrals.at(label));
    }

  }; // Other type copy constructor

  template <typename MatsT, typename IntsT>
  template <typename MatsU>
  NEOSS<MatsT,IntsT>::NEOSS(NEOSS<MatsU,IntsT>&& other, int dummy) :
    SingleSlater<MatsT,IntsT>(dynamic_cast<SingleSlater<MatsU,IntsT>&&>(other), dummy),
    WaveFunctionBase(dynamic_cast<WaveFunctionBase&&>(other)),
    QuantumBase(dynamic_cast<QuantumBase&&>(other)),
    order_(other.order_)
  {
    CErr("NEOSS move constructor not fully implemented");
  }; // Other type move constructor

  template <typename MatsT, typename IntsT>
  std::vector<double> NEOSS<MatsT,IntsT>::getGrad(EMPerturbation& pert,
    bool equil, bool saveInts) {

    size_t nAtoms = this->molecule().nAtoms;
    size_t nGrad = 3*nAtoms;
    std::vector<double> gradient(nGrad, 0.);
    return gradient;

  };

} // namespace ChronusQ
