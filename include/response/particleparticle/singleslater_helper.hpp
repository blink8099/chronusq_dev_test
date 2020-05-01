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

#include <response/particleparticle.hpp>

#include <cqlinalg/blas1.hpp>
#include <cqlinalg/factorization.hpp>

#include <util/threads.hpp>

namespace ChronusQ {

  template <typename Reference, typename Resp, typename U>
  void formLinearTrans_direct_helper(MPI_Comm c, Resp & res, bool doTDA,
    ParticleParticleProp_SpinSep spinSepOp, ParticleParticleTDA tdaMAT,
    std::vector<RESPONSE_CONTRACTION<U>> x, double mu = 0.) {

    Reference &hf = dynamic_cast<Reference&>(*res.ref());
    for(auto &X : x) {

      size_t N  = X.N;
      size_t nV = X.nVec;

      if( X.AX ) std::fill_n(X.AX,N*nV,0.);
      hf.ppLinearTrans(c,X.nVec,doTDA,spinSepOp,tdaMAT,X.X,X.AX,mu);

      //if( X.AX and incMet and not doAPB_AMB )
      //  SetMat('N', N/2, nV, U(-1.), X.AX + (N/2), N, X.AX + (N/2), N);

    }

  }

};

