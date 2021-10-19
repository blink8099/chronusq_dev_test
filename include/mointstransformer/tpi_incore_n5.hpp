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

#include <mointstransformer.hpp>
#include <util/timer.hpp>
#include <cqlinalg.hpp>
#include <matrix.hpp>
#include <particleintegrals/twopints/incoreritpi.hpp>
#include <particleintegrals/twopints/incore4indexreleri.hpp>
#include <particleintegrals/twopints/incore4indextpi.hpp>

namespace ChronusQ {

  template <typename MatsT, typename IntsT>
  std::shared_ptr<InCore4indexTPI<MatsT>> 
  MOIntsTransformer<MatsT,IntsT>::getAOTPIInCore(bool cacheAOTPI) {
      
      // cache AOTPI
      auto AOTPI = ints_cache_.getIntegral<InCore4indexTPI,MatsT>("AOTPI");

      if (not AOTPI) { 
        if(ss_.nC == 1) {
          AOTPI = std::make_shared<InCore4indexTPI<MatsT>>(
                    *std::dynamic_pointer_cast<InCore4indexTPI<IntsT>>(ss_.aoints.TPI)); 
        } else if (ss_.nC == 2) {
          std::cout << "  * Using bare Coulomb Operator for 2e Integrals" << std::endl;
          AOTPI = std::make_shared<InCore4indexTPI<MatsT>>(
            std::dynamic_pointer_cast<InCore4indexTPI<IntsT>>(ss_.aoints.TPI)
              ->template spatialToSpinBlock<MatsT>()); 
        } else if (ss_.nC == 4) {
          AOTPI = std::dynamic_pointer_cast<InCore4indexTPI<MatsT>>(
            std::make_shared<InCore4indexRelERI<MatsT>>(
              std::dynamic_pointer_cast<InCore4indexRelERI<IntsT>>(ss_.aoints.TPI)
                ->template spatialToSpinBlock<MatsT>()));
        }

        if (cacheAOTPI) ints_cache_.addIntegral("AOTPI", AOTPI);
      } 

     return AOTPI; 
  }; // MOIntsTransformer::cacheAOTPIInCore
  
  /**
   *  \brief subset tranform TPI using incore  
   */
  template <typename MatsT, typename IntsT>
  void MOIntsTransformer<MatsT,IntsT>::subsetTransformTPIInCoreN5(
    const std::vector<std::pair<size_t,size_t>> & off_sizes, MatsT * MOTPI, bool cacheIntermediates) {
      
      auto AOTPI = getAOTPIInCore(cacheIntermediates); 
       
      size_t nAO = ss_.mo[0].dimension();
      auto MO = ss_.mo[0].pointer();
      
      if (ss_.nC != 4) {
        AOTPI->subsetTransform('N', MO, nAO, off_sizes, MOTPI);
      } else {
        auto AOTPI4C = std::dynamic_pointer_cast<InCore4indexRelERI<MatsT>>(AOTPI);
        AOTPI4C->subsetTransform('N', MO, nAO, off_sizes, MOTPI);
      }

      return; 
  }; // MOIntsTransformer<MatsT,IntsT>::subsetTransformTPIInCoreN5
  
}; // namespace ChronusQ
