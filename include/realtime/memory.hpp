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

#include <realtime.hpp>


namespace ChronusQ {

  template <template <typename, typename> class _SSTyp, typename IntsT>
  template <typename MatsT>
  void RealTime<_SSTyp,IntsT>::alloc() {

    // XXX: Member functions can't be partially specialized,
    //   so we're just going to cram this all into a single if statement...
    if( std::is_same<NEOSS<dcomplex,IntsT>,_SSTyp<dcomplex,IntsT>>::value ) {
      auto ref = dynamic_cast<NEOSS<MatsT,IntsT>*>(reference_);
      auto map = ref->getSubsystemMap();

      assert( !map.empty() );

      // Assume all subsystems are the same type
      bool isHF = std::dynamic_pointer_cast<HartreeFock<MatsT,IntsT>>(
        map.begin()->second);

      for( auto& system: map ) {

        auto baseFock = system.second->fockBuilder.get();
        if( auto p = dynamic_cast<NEOFockBuilder<MatsT,IntsT>*>(baseFock) ) {
          baseFock = p->getNonNEOUpstream();
        }

        // Assume just non-relativistic Fock here
        // TODO: Make this general to all fockbuilders
        auto newFock = std::make_shared<FockBuilder<dcomplex,IntsT>>(*baseFock);

        std::shared_ptr<SingleSlater<dcomplex,IntsT>> newSS;
        if( isHF ) {
          newSS = std::make_shared<HartreeFock<dcomplex,IntsT>>(
            dynamic_cast<HartreeFock<MatsT,IntsT>&>(*system.second)
          );
        }
        else {
          newSS = std::make_shared<KohnSham<dcomplex,IntsT>>(
            dynamic_cast<KohnSham<MatsT,IntsT>&>(*system.second)
          );
        }

        newSS->fockBuilder = newFock;

        // Stupid, stupid C++
        auto casted = dynamic_cast<NEOSS<dcomplex,IntsT>&>(propagator_);
        casted.addSubsystem(system.first, newSS);

        // Information for RealTime only
        size_t NB = newSS->onePDM->dimension();
        bool hasZ = newSS->onePDM->hasZ();
        bool hasXY= newSS->onePDM->hasXY();

        systems_.push_back(newSS.get());

        DOSav.push_back(
          std::make_shared<PauliSpinorSquareMatrices<dcomplex>>(
            memManager_, NB, hasXY, hasZ
          ));
        UH.push_back(
          std::make_shared<PauliSpinorSquareMatrices<dcomplex>>(
            memManager_, NB, hasXY, hasZ
          ));

      }

    }
    else {

      size_t NB = propagator_.onePDM->dimension();
      bool hasZ = propagator_.onePDM->hasZ();
      bool hasXY= propagator_.onePDM->hasXY();

      systems_.push_back(&propagator_);

      DOSav.emplace_back(
        std::make_shared<PauliSpinorSquareMatrices<dcomplex>>(
          memManager_, NB, hasXY, hasZ
        ));
      UH.emplace_back(
        std::make_shared<PauliSpinorSquareMatrices<dcomplex>>(
          memManager_, NB, hasXY, hasZ
        ));

    }

  };

}; // namespace ChronusQ


