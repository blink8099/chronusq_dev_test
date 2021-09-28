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
  void MOIntsTransformer<MatsT,IntsT>::cacheAOTPIInCore() {
      
      // cache AOTPI
      if (not AOTPI_) {
        if(ss_.nC == 1) {
          AOTPI_ = std::make_shared<InCore4indexTPI<MatsT>>(
                    *std::dynamic_pointer_cast<InCore4indexTPI<IntsT>>(ss_.aoints.TPI)); 
        } else if (ss_.nC == 2) {
          std::cout << "  * Using bare Coulomb Operator for 2e Integrals" << std::endl;
          AOTPI_ = std::make_shared<InCore4indexTPI<MatsT>>(
            std::dynamic_pointer_cast<InCore4indexTPI<IntsT>>(ss_.aoints.TPI)
              ->template spatialToSpinBlock<MatsT>()); 
        } else if (ss_.nC == 4) {
          AOTPI_ = std::dynamic_pointer_cast<InCore4indexTPI<MatsT>>(
            std::make_shared<InCore4indexRelERI<MatsT>>(
              std::dynamic_pointer_cast<InCore4indexRelERI<IntsT>>(ss_.aoints.TPI)
                ->template spatialToSpinBlock<MatsT>()));
        }
      }

     return; 
  }; // MOIntsTransformer::cacheAOTPIInCore
  
  /**
   *  \brief subset tranform TPI using incore  
   */
  template <typename MatsT, typename IntsT>
  void MOIntsTransformer<MatsT,IntsT>::subsetTransformTPIInCoreN5(
    const std::vector<std::pair<size_t,size_t>> & off_sizes, MatsT * MOTPI, bool antiSymm) {
      
      cacheAOTPIInCore(); 
       
      size_t nAO = ss_.mo[0].dimension();
      size_t poff = off_sizes[0].first;
      size_t qoff = off_sizes[1].first;
      size_t roff = off_sizes[2].first;
      size_t soff = off_sizes[3].first;
      size_t np = off_sizes[0].second;
      size_t nq = off_sizes[1].second;
      size_t nr = off_sizes[2].second;
      size_t ns = off_sizes[3].second;
      size_t npq = np * nq;
      size_t nrs = nr * ns;
      size_t nps = np * ns;
      size_t npqr = npq * nr;
      size_t npsr = nps * nr;
      size_t npqrs = npq * nrs;
      
      auto MO = ss_.mo[0].pointer();
      
      // get the Coulomb part
      if (ss_.nC != 4) {
        AOTPI_->subsetTransform('N', MO, nAO, off_sizes, MOTPI);
      } else {
        auto AOTPI4C = std::dynamic_pointer_cast<InCore4indexRelERI<MatsT>>(AOTPI_);
        AOTPI4C->subsetTransform('N', MO, nAO, off_sizes, MOTPI);
      }

      // get exchange part if anti-symmetrize MOTPI
      if (antiSymm) {
        
        MatsT * SCR  = memManager_.malloc<MatsT>(npqrs);
        bool qsSymm = (qoff == soff) and (nq == ns); 
        if (qsSymm) {
          SetMat('N', npq, nrs, MatsT(1.), MOTPI, npq, SCR, npq);
        } else { 
          std::vector<std::pair<size_t, size_t>> exchange_off_sizes 
            = {{poff,np},{soff,ns},{roff,nr},{qoff,nq}};
          if (ss_.nC != 4) {
            AOTPI_->subsetTransform('N', MO, nAO, exchange_off_sizes, SCR); 
          } else {
            auto AOTPI4C = std::dynamic_pointer_cast<InCore4indexRelERI<MatsT>>(AOTPI_);
            AOTPI4C->subsetTransform('N', MO, nAO, exchange_off_sizes, SCR); 
          }
        } 
      
      // Anti-symmetrize MOTPI
#pragma omp parallel for schedule(static) collapse(4) default(shared)       
        for (auto s = 0ul; s < ns; s++)
        for (auto r = 0ul; r < nr; r++) 
        for (auto q = 0ul; q < nq; q++)
        for (auto p = 0ul; p < np; p++) {     
            MOTPI[p + q*np + r*npq + s*npqr] -= SCR[p + s*np + r*nps + q*npsr]; 
        }
        
        memManager_.free(SCR);
      }
      
      return; 
  }; // MOIntsTransformer<MatsT,IntsT>::subsetTransformTPIInCoreN5
  
}; // namespace ChronusQ
