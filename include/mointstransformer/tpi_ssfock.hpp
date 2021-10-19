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
#include <cqlinalg/blas1.hpp>
#include <cqlinalg/blasutil.hpp>

namespace ChronusQ {

  #define ALLOCATE_AND_CLEAR_CACHE_IF_NECESSARY(ALLOCATION) \
    try { ALLOCATION; } \
    catch (...) { \
      ints_cache_.clear(); \
      try { ALLOCATION; } \
      catch (...) { CErr("Not Enough Memory in subsetTransformTPISSFockN6");} \
    }
 
  /**
   *  \brief transform AO TPI to form MO TPI
   *  thru SingleSlater formfock 
   */
  template <typename MatsT, typename IntsT>
  void MOIntsTransformer<MatsT,IntsT>::subsetTransformTPISSFockN6(EMPerturbation & pert, 
    const std::vector<std::pair<size_t,size_t>> &off_sizes, MatsT* MOTPI, 
    const std::string & moType, bool cacheIntermediates) {

    size_t poff = off_sizes[0].first;
    size_t qoff = off_sizes[1].first;
    size_t roff = off_sizes[2].first;
    size_t soff = off_sizes[3].first;
    size_t np = off_sizes[0].second;
    size_t nq = off_sizes[1].second;
    size_t nr = off_sizes[2].second;
    size_t ns = off_sizes[3].second;
    bool pqSymm = (poff == qoff) and (np == nq); 
    std::string moType_cache = "HalfTMOTPI-" 
      + getUniqueSymbol(moType[0]) + getUniqueSymbol(moType[1]);
    
    // check pq and rs can be swapped to accelarate the computation 
    bool rsSymm = (roff == soff) and (nr == ns); 
    bool swap_pq_rs = false;
    if (rsSymm) {
      std::string rs_moType_cache = "HalfTMOTPI-" 
        + getUniqueSymbol(moType[2]) + getUniqueSymbol(moType[3]); 
      if (not pqSymm) swap_pq_rs = true;
      else {
        auto pqCache = ints_cache_.getIntegral<InCore4indexTPI, MatsT>(moType_cache); 
        auto rsCache = ints_cache_.getIntegral<InCore4indexTPI, MatsT>(rs_moType_cache); 
        
        if (not pqCache and rsCache) swap_pq_rs = true;
      }
    
      if (swap_pq_rs) {
        std::swap(poff, roff); 
        std::swap(qoff, soff); 
        std::swap(np, nr); 
        std::swap(nq, ns); 
        pqSymm = true;
        moType_cache = rs_moType_cache; 
      }
    }
    
    size_t npq = np * nq;
    size_t npqr = npq * nr;

    size_t nAO = ss_.nAlphaOrbital() * ss_.nC;
    size_t NB  = ss_.nAlphaOrbital();
    size_t nAO2 = nAO * nAO;
    
    MatsT * halfTMOTPI_ptr = nullptr, *dummy_ptr = nullptr;
    std::shared_ptr<InCore4indexTPI<MatsT>> halfTMOTPI = nullptr; 
    if (not pqSymm) cacheIntermediates = false;
    
    if (cacheIntermediates) {
      halfTMOTPI = ints_cache_.getIntegral<InCore4indexTPI, MatsT>(moType_cache);
      if (halfTMOTPI) halfTMOTPI_ptr = halfTMOTPI->pointer();
    } 
    
    //
    // 1/2 transformation to obtain halfTMOTPI_ptr(mu, nu, p, q)
    //
    if (not halfTMOTPI) {
      
      ALLOCATE_AND_CLEAR_CACHE_IF_NECESSARY(
        if (pqSymm) {
          halfTMOTPI = std::make_shared<InCore4indexTPI<MatsT>>(memManager_, nAO, np);
          halfTMOTPI_ptr = halfTMOTPI->pointer();
        } else {  
          halfTMOTPI_ptr  = memManager_.malloc<MatsT>(nAO2 * npq);
        }
      )

      SquareMatrix<MatsT> SCR(memManager_, nAO);
      std::vector<std::shared_ptr<PauliSpinorSquareMatrices<MatsT>>> pq1PDMs, pqMOTPIs, dummy;
      MatsT *MOTPIpq_ptr = nullptr, *density_ptr = nullptr; 
      bool is4C = ss_.nC == 4;
      bool is2C = ss_.nC == 2;
      bool is1C = ss_.nC == 1;
      size_t pqSCRSize = is4C ? 2*NB: NB;
      
      // find out maximum batch size base on current memory limit
      
      // SCR size form fockbuilder
      size_t fockGDSCRSize = ss_.fockBuilder->formRawGDSCRSizePerBatch(ss_, true); 
      
      // SCR size for spinor density and half-transformed integrals
      fockGDSCRSize += is4C ? pqSCRSize*pqSCRSize: 2*pqSCRSize*pqSCRSize;  
      size_t maxNBatch = 0;
      double allow_extra = 0.2;
      
      ALLOCATE_AND_CLEAR_CACHE_IF_NECESSARY(
        maxNBatch = memManager_.max_avail_allocatable<MatsT>(size_t(fockGDSCRSize*(1.+allow_extra))); 
        if(maxNBatch == 0) 
          CErr(" Memory is not enough for 1 density in subsetTransformTPISSFockN6");
      )

      std::vector<std::pair<size_t,size_t>> pqJobs;
      
      for (auto q = 0ul; q <  nq; q++) 
      for (auto p = 0ul; p <  np; p++) {
        pqJobs.push_back({p,q});
        if (pqSymm and p == q) break; 
      }
      
      size_t NJob = pqJobs.size();
      size_t NJobComplete  = 0ul;
      size_t NJobToDo = 0ul;

      while (NJobComplete < NJob) { 
        
        size_t p, q;
        NJobToDo = std::min(maxNBatch, NJob-NJobComplete);

        std::cout << "NJob         = " << NJob << std::endl;
        std::cout << "NJobComplete = " << NJobComplete << std::endl;
        std::cout << "NJobToDo     = " << NJobToDo << std::endl;
        std::cout << "maxNBatch    = " << maxNBatch << std::endl;

        
        // build fake densities in batch
        for (auto i = 0ul; i < NJobToDo; i++) { 
          
          p = pqJobs[i + NJobComplete].first; 
          q = pqJobs[i + NJobComplete].second; 
          
          pq1PDMs.push_back(std::make_shared<PauliSpinorSquareMatrices<MatsT>>(memManager_, pqSCRSize, is4C, is4C));  
          
          // 4C will reuse pq1PDM as densities are component scattered anyways
          if (is4C) pqMOTPIs.push_back(pq1PDMs.back());
          else pqMOTPIs.push_back(std::make_shared<PauliSpinorSquareMatrices<MatsT>>(memManager_, pqSCRSize, is4C, is4C));  
          
          if (is1C) {
            density_ptr = pq1PDMs.back()->S().pointer();
          } else {
            density_ptr = SCR.pointer();
          }
        
          blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::ConjTrans, 
            nAO, nAO, 1, MatsT(1.), ss_.mo[0].pointer() + (q+qoff)*nAO, nAO,
            ss_.mo[0].pointer() + (p+poff)*nAO, nAO, MatsT(0.), density_ptr, nAO);
          
          if (not is1C) *pq1PDMs.back() = SCR.template spinScatter<MatsT>(is4C, is4C); 
        
        }
        
        // do contraction to get half-transformed integrals
        if (is1C) ss_.fockBuilder->formRawGDInBatches(ss_, pert, false, 0., false, pq1PDMs, pqMOTPIs, dummy, dummy);
        else ss_.fockBuilder->formRawGDInBatches(ss_, pert, false, 0., false, pq1PDMs, dummy, dummy, pqMOTPIs);
        
        // copy over to SCR
        for (auto i = 0ul; i < NJobToDo; i++) { 
          p = pqJobs[i + NJobComplete].first; 
          q = pqJobs[i + NJobComplete].second; 
           
          if (not is1C) {
            SCR = pqMOTPIs[i]->template spinGather<MatsT>();
            MOTPIpq_ptr = SCR.pointer(); 
          } else MOTPIpq_ptr = pqMOTPIs[i]->S().pointer();
          
          SetMat('N', nAO, nAO, MatsT(1.), MOTPIpq_ptr, nAO, halfTMOTPI_ptr + (p + q*np)*nAO2, nAO);
          if (pqSymm and p < q)  
            SetMat('C', nAO, nAO, MatsT(1.), MOTPIpq_ptr, nAO, halfTMOTPI_ptr + (q + p*np)*nAO2, nAO); 
        }

        // increment and clear SCR after job done
        NJobComplete += NJobToDo; 
        pq1PDMs.clear();
        pqMOTPIs.clear();
      
      } // main loop
      
      if (cacheIntermediates) ints_cache_.addIntegral(moType_cache, halfTMOTPI);
     
    } //  if there is no cache

    MatsT * SCR = nullptr;
    
    ALLOCATE_AND_CLEAR_CACHE_IF_NECESSARY(
      SCR  = memManager_.malloc<MatsT>(nAO * npqr);
    )
    
    char TransMOTPI = swap_pq_rs ? 'N': 'T';

    PairTransformation('N', ss_.mo[0].pointer(), nAO, roff, soff,
      'N', halfTMOTPI_ptr, nAO, nAO, npq, 
      TransMOTPI, MOTPI, nr, ns, dummy_ptr, SCR, false); 

    // free memories
    halfTMOTPI = nullptr;
    if (not pqSymm) memManager_.free(halfTMOTPI_ptr);
    if (SCR) memManager_.free(SCR);
 
  }; // MOIntsTransformer::subsetTransformTPISSFockN6
  

}; // namespace ChronusQ
