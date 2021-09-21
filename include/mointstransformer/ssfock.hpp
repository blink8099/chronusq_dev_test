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

  /**
   *  \brief transform ERI to form anti-symmetric ERI
   *  thru SingleSlater formfock 
   */
  template <typename MatsT, typename IntsT>
  void MOIntsTransformer<MatsT,IntsT>::subsetTransformERISSFockN6(EMPerturbation & pert, 
    const std::vector<std::pair<size_t,size_t>> &off_sizes, MatsT* MOERI, bool antiSymm) {

    // disable 1C case
    if (ss_.nC == 1) CErr("ERI Transformation thru SSFOCK_N6 NYI for 1C");
    
    size_t poff = off_sizes[0].first;
    size_t qoff = off_sizes[1].first;
    size_t roff = off_sizes[2].first;
    size_t soff = off_sizes[3].first;
    size_t np = off_sizes[0].second;
    size_t nq = off_sizes[1].second;
    size_t nr = off_sizes[2].second;
    size_t ns = off_sizes[3].second;
    size_t npq = np * nq;
    size_t npqr = npq * nr;

    size_t nAO = ss_.nAlphaOrbital() * ss_.nC;
    size_t nAO2 = nAO * nAO;
    
    SquareMatrix<MatsT> spinBlockForm1PDM(memManager_, nAO);
    MatsT * SCR  = memManager_.malloc<MatsT>(nAO2 * npq);
    
    bool pqSymm = (poff == qoff) and (np == nq); 
    double xHFX = antiSymm ? 1.0: 0.0; 
    std::cout << "antiSymm ="  << antiSymm << " xHFX = " << xHFX << std::endl;
    
    // 1/2 transformation to obtain SCR(mu, nu, p, q)
    for (auto q = 0; q <  nq; q++) 
    for (auto p = 0; p <  np; p++) {
      
      // outer product to make a fake ss onePDM of Dqp
      Gemm('N', 'C', nAO, nAO, 1, MatsT(1.), ss_.mo[0].pointer() + (q+qoff)*nAO, nAO,
        ss_.mo[0].pointer() + (p+poff)*nAO, nAO, MatsT(0.), spinBlockForm1PDM.pointer(), nAO);
      
      // Hack SS to get ASYMERIpq
      *(ss_.onePDM) = spinBlockForm1PDM.template spinScatter<MatsT>(true, true);
      ss_.fockBuilder->formGD(ss_, pert, false, xHFX, false);
      auto MOERIpq = ss_.twoeH->template spinGather<MatsT>();

      // copy
      SetMat('N', nAO, nAO, MatsT(1.), MOERIpq.pointer(), nAO, SCR + (p + q*np)*nAO2, nAO);
      if (pqSymm) { 
         if (p < q) SetMat('C', nAO, nAO, MatsT(1.), MOERIpq.pointer(), nAO, SCR + (q + p*np)*nAO2, nAO); 
         else if (p == q) break;
      }
     
    } // 1/2 transformation

    MatsT * SCR2 = nullptr;
    
    if (ns == nAO) SCR2 = MOERI;
    else SCR2  = memManager_.malloc<MatsT>(nAO * npqr);
    
    // 3/4 transfromation: SCR2(nu p q, r) = SCR(mu,nu p q)^H * C(mu, r)
    Gemm('C', 'N', nAO*npq, nr, nAO,
        MatsT(1.), SCR, nAO, ss_.mo[0].pointer() + roff*nAO, nAO,
        MatsT(0.), SCR2, nAO*npq);
    
    // 4/4 transfromation: (p q| r, s) = SCR2(nu, p q r)^H * C(nu, s)
    Gemm('C', 'N', npqr, ns, nAO,
        MatsT(1.), SCR2, nAO, ss_.mo[0].pointer() + soff*nAO, nAO,
        MatsT(0.), SCR, npqr);
    
    SetMat('N', npq, nr*ns, MatsT(1.), SCR, npq, MOERI, npq); 
    memManager_.free(SCR);
    if (ns != nAO) memManager_.free(SCR2);
  
  }; // MOIntsTransformer::subsetTransformERISSFockN6
  

}; // namespace ChronusQ
