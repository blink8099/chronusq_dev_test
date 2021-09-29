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
#include <mointstransformer/ssfock.hpp>
#include <mointstransformer/incore.hpp>
#include <util/timer.hpp>
#include <cqlinalg.hpp>
#include <matrix.hpp>
#include <cqlinalg/blas1.hpp>
#include <cqlinalg/blasutil.hpp>
#include <particleintegrals/twopints/incoreritpi.hpp>

namespace ChronusQ {

  /**
   *  \brief set up MO ranges 
   */
  template <typename MatsT, typename IntsT>
  void MOIntsTransformer<MatsT,IntsT>::addMORanges(const std::set<char> & symSet,
    const std::pair<size_t, size_t> & range) {
      
      symbol_sets_.push_back(symSet);
      mo_ranges_.push_back(range);
 
  }; // MOIntsTransformer::addMORanges  
  
  template <typename MatsT, typename IntsT>
  void MOIntsTransformer<MatsT,IntsT>::setMORanges(size_t nFrozenCore, size_t nFrozenVirt) {
    
      // set 4C for no-pair approximation
      size_t fourCompOffset = (ss_.nC == 4) ? ss_.nAlphaOrbital() * 2: 0;

      size_t nO = ss_.nO - nFrozenCore;
      size_t nV = ss_.nV - nFrozenVirt; 
      size_t nT = nO + nV;
      
      resetMORanges();

      // general indices
      addMORanges({'p','q','r','s'}, {fourCompOffset + nFrozenCore, nT});
      
      // hole indices
      addMORanges({'i','j','k','l'}, {fourCompOffset + nFrozenCore, nO});
      
      // particle indices
      addMORanges({'a','b','c','d'}, {fourCompOffset + ss_.nO, nV}); 

  }; // MOIntsTransformer::setMORanges
  
  /**
   *  \brief parsing the mo ints types to offsizes 
   */
  template <typename MatsT, typename IntsT>
  std::vector<std::pair<size_t,size_t>> 
    MOIntsTransformer<MatsT,IntsT>::parseMOType(const std::string & moType) {
      
      std::vector<std::pair<size_t,size_t>> off_sizes;
      
      bool foundMOType;
      for (const auto& ch: moType) {
        
        foundMOType = false;
        for (auto i = 0ul; i < symbol_sets_.size(); i++) {
          if (symbol_sets_[i].count(ch)) {
            off_sizes.push_back(mo_ranges_[i]);
            foundMOType = true;
            break;
          }
        }
        if (not foundMOType) CErr("Wrong MO Type in parseMOIntsType");
      }
      
      return off_sizes;
  }; // MOIntsTransformer::parseMOIntsType
  
  /**
   *  \brief main interface to transform HCore 
   */
  template <typename MatsT, typename IntsT>
  void MOIntsTransformer<MatsT,IntsT>::transformHCore(MatsT* MOHCore, const std::string & moType) {

    auto off_sizes = parseMOType(moType);
    subsetTransformHCore(off_sizes, MOHCore);

  }; // MOIntsTransformer::transformHCore
  
  /**
   *  \brief transform a subset of HCore 
   */
  template <typename MatsT, typename IntsT>
  void MOIntsTransformer<MatsT,IntsT>::subsetTransformHCore(
    const std::vector<std::pair<size_t,size_t>> &off_sizes, MatsT* MOHCore) {
    
    size_t nAO = ss_.nAlphaOrbital() * ss_.nC;
    auto AOHCore = ss_.coreH->template spinGather<MatsT>();
    AOHCore.subsetTransform('N', ss_.mo[0].pointer(), nAO, off_sizes, MOHCore, false); 
  
  }; // MOIntsTransformer::subsetTransformHCore 
  
  /**
   *  \brief main interface to transform HCore 
   */
  template <typename MatsT, typename IntsT>
  void MOIntsTransformer<MatsT,IntsT>::transformTPI(EMPerturbation & pert, 
    MatsT* MOTPI, const std::string & moType, bool antiSymm) {
    
    if (ss_.nC == 1) CErr("transformTPI not implemented for 1C");

    auto off_sizes = parseMOType(moType);
    
    // get the Coulomb part
    if (TPITransAlg_ == DIRECT_N6 or TPITransAlg_ == INCORE_N6) {
      subsetTransformTPISSFockN6(pert, off_sizes, MOTPI);
    } else if (TPITransAlg_ == INCORE_N5) {
      subsetTransformTPIInCoreN5(off_sizes, MOTPI);
    } else {
      CErr("DIRECT_N5 NYI");
    }

    // get exchange part if anti-symmetrize MOTPI
    if (antiSymm) {
      
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
      MatsT * SCR = nullptr;
      if (qsSymm) {
         SCR = MOTPI;
      } else {
        SCR = memManager_.malloc<MatsT>(npqr*ns);
        std::string moType_exchange = "";
        moType_exchange += moType[0];
        moType_exchange += moType[3];
        moType_exchange += moType[2];
        moType_exchange += moType[1];
        transformTPI(pert, SCR, moType_exchange, false); 
      }
    
#pragma omp parallel for schedule(static) collapse(2) default(shared)       
      for (auto s = 0ul; s < ns; s++)
      for (auto r = 0ul; r < nr; r++) 
      for (auto q = 0ul; q <= s; q++) 
      for (auto p = 0ul; p < np; p++) { 
          MOTPI[p + q*np + r*npq + s*npqr] -= SCR[p + s*np + r*nps + q*npsr]; 
          MOTPI[p + s*np + r*npq + q*npqr] = - MOTPI[p + q*np + r*npq + s*npqr]; 
      }
        
      if(not qsSymm) memManager_.free(SCR);

    }
  }; // MOIntsTransformer::transformTPI 

}; // namespace ChronusQ
