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
#include <mointstransformer/moranges.hpp>
#include <mointstransformer/hcore.hpp>
#include <mointstransformer/tpi_ssfock.hpp>
#include <mointstransformer/tpi_incore_n5.hpp>
#include <util/timer.hpp>

namespace ChronusQ {

  /**
   *  \brief main interface to transform HCore 
   */
  template <typename MatsT, typename IntsT>
  void MOIntsTransformer<MatsT,IntsT>::transformHCore(MatsT* MOHCore, const std::string & moType) {

    auto off_sizes = parseMOType(moType);
    subsetTransformHCore(off_sizes, MOHCore);

  }; // MOIntsTransformer::transformHCore
  
  /**
   *  \brief main interface to transform TPI
   */
  template <typename MatsT, typename IntsT>
  void MOIntsTransformer<MatsT,IntsT>::transformTPI(EMPerturbation & pert, 
    MatsT* MOTPI, const std::string & moType, bool cacheIntermediates, bool withExchange) {
    
    auto off_sizes = parseMOType(moType);
    
    // get the Coulomb part
    if (TPITransAlg_ == DIRECT_N6 or TPITransAlg_ == INCORE_N6) {
      subsetTransformTPISSFockN6(pert, off_sizes, MOTPI, moType, cacheIntermediates);
    } else if (TPITransAlg_ == INCORE_N5) {
      subsetTransformTPIInCoreN5(off_sizes, MOTPI, cacheIntermediates);
    } else {
      CErr("DIRECT_N5 NYI");
    }

    // get exchange part if needed
    // for 2c/4c: it's antisymmetrized integrals (pq|rs) - (ps|rq)
    // for 1c: it's scaled for spins, so exchange part will be scaled with 0.5. 
    //         So (pq|rs) - 0.5 * (ps|rq) will be computed
    if (withExchange) {
      
      size_t qoff = off_sizes[1].first;
      size_t soff = off_sizes[3].first;
      size_t np = off_sizes[0].second;
      size_t nq = off_sizes[1].second;
      size_t nr = off_sizes[2].second;
      size_t ns = off_sizes[3].second;
      size_t npq = np * nq;
      size_t nps = np * ns;
      size_t npqr = npq * nr;
      size_t npsr = nps * nr;
      
      bool qsSymm = (qoff == soff) and (nq == ns); 
      
      MatsT fc = ss_.nC == 1 ? 0.5: 1.0;

      if (qsSymm) {
        #pragma omp parallel for schedule(static) collapse(2) default(shared)       
        for (auto r = 0ul; r < nr; r++) 
        for (auto p = 0ul; p < np; p++) {
          MatsT tmp1, tmp2;
          size_t pqrs, psrq;
          auto prnpq = p + r*npq;
          auto prnps = p + r*nps;
          for (auto s = 0ul; s < ns; s++)
          for (auto q = 0ul; q <= s; q++) {
            pqrs = prnpq + q*np + s*npqr;
            psrq = prnps + s*np + q*npsr;
            tmp1 = MOTPI[pqrs];
            tmp2 = MOTPI[psrq];
            MOTPI[pqrs] = tmp1 - fc * tmp2;
            MOTPI[psrq] = tmp2 - fc * tmp1;
          }
        }
      } else {
        
        MatsT * SCR = memManager_.malloc<MatsT>(npqr*ns);
        std::string moType_exchange = "";
        moType_exchange += moType[0];
        moType_exchange += moType[3];
        moType_exchange += moType[2];
        moType_exchange += moType[1];
        transformTPI(pert, SCR, moType_exchange, true, false); 

#pragma omp parallel for schedule(static) collapse(2) default(shared)       
        for (auto s = 0ul; s < ns; s++)
        for (auto r = 0ul; r < nr; r++) 
        for (auto q = 0ul; q < nq; q++) 
        for (auto p = 0ul; p < np; p++) { 
          MOTPI[p + q*np + r*npq + s*npqr] -= fc * SCR[p + s*np + r*nps + q*npsr]; 
        }
        
        memManager_.free(SCR);
      }
    }  
  
  }; // MOIntsTransformer::transformTPI 

}; // namespace ChronusQ
