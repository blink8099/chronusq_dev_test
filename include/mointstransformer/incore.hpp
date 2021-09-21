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
  void MOIntsTransformer<MatsT,IntsT>::cacheAOERIInCore() {
      
      // cache AOERI
      if (not AOERI_) {
        if(ss_.nC == 1) {
          AOERI_ = std::make_shared<InCore4indexTPI<MatsT>>(
                    *std::dynamic_pointer_cast<InCore4indexTPI<IntsT>>(ss_.aoints.TPI)); 
        } else if (ss_.nC == 2) {
          std::cout << "  * Using bare Coulomb Operator for 2e Integrals" << std::endl;
          AOERI_ = std::make_shared<InCore4indexTPI<MatsT>>(
            std::dynamic_pointer_cast<InCore4indexTPI<IntsT>>(ss_.aoints.TPI)
              ->template spatialToSpinBlock<MatsT>()); 
        } else if (ss_.nC == 4) {
          AOERI_ = std::dynamic_pointer_cast<InCore4indexTPI<MatsT>>(
            std::make_shared<InCore4indexRelERI<MatsT>>(
              std::dynamic_pointer_cast<InCore4indexRelERI<IntsT>>(ss_.aoints.TPI)
                ->template spatialToSpinBlock<MatsT>()));
        }
      }

     return; 
  }; // MOIntsTransformer::cacheAOERIInCore
  
  /**
   *  \brief subset tranform ERI using incore  
   */
  template <typename MatsT, typename IntsT>
  void MOIntsTransformer<MatsT,IntsT>::subsetTransformERIInCoreN5(
    const std::vector<std::pair<size_t,size_t>> & off_sizes, MatsT * MOERI, bool antiSymm) {
      
      cacheAOERIInCore(); 
       
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
        AOERI_->subsetTransform('N', MO, nAO, off_sizes, MOERI);
      } else {
        auto AOERI4C = std::dynamic_pointer_cast<InCore4indexRelERI<MatsT>>(AOERI_);
        AOERI4C->subsetTransform('N', MO, nAO, off_sizes, MOERI);
      }

      // get exchange part if anti-symmetrize MOERI
      if (antiSymm) {
        
        MatsT * SCR  = memManager_.malloc<MatsT>(npqrs);
        bool qsSymm = (qoff == soff) and (nq == ns); 
        if (qsSymm) {
          SetMat('N', npq, nrs, MatsT(1.), MOERI, npq, SCR, npq);
        } else { 
          std::vector<std::pair<size_t, size_t>> exchange_off_sizes 
            = {{poff,np},{soff,ns},{roff,nr},{qoff,nq}};
          if (ss_.nC != 4) {
            AOERI_->subsetTransform('N', MO, nAO, exchange_off_sizes, SCR); 
          } else {
            auto AOERI4C = std::dynamic_pointer_cast<InCore4indexRelERI<MatsT>>(AOERI_);
            AOERI4C->subsetTransform('N', MO, nAO, exchange_off_sizes, SCR); 
          }
        } 
      
      // Anti-symmetrize MOERI
#pragma omp parallel for schedule(static) collapse(4) default(shared)       
        for (auto s = 0ul; s < ns; s++)
        for (auto r = 0ul; r < nr; r++) 
        for (auto q = 0ul; q < nq; q++)
        for (auto p = 0ul; p < np; p++) {     
            MOERI[p + q*np + r*npq + s*npqr] -= SCR[p + s*np + r*nps + q*npsr]; 
        }
        
        memManager_.free(SCR);
      }
      
      return; 
  }; // MOIntsTransformer<MatsT,IntsT>::subsetTransformERIInCoreN5
  
}; // namespace ChronusQ
