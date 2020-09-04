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

#include <cqlinalg.hpp>
#include <cqlinalg/blasutil.hpp>
#include <util/time.hpp>
#include <util/matout.hpp>
#include <electronintegrals/twoeints/incore4indexeri.hpp>
#include <electronintegrals/twoeints/incorerieri.hpp>
#include <electronintegrals/twoeints/gtodirecteri.hpp>

#include <util/threads.hpp>
#include <chrono>
// Debug directives
//#define _DEBUGORTHO
//#define __DEBUGERI__


namespace ChronusQ {



  /**
   *  \brief Allocate and evaluate the Schwartz bounds over the
   *  CGTO shell pairs.
   */ 
//  template <>
//  void AOIntegrals<dcomplex>::computeSchwartz() {
//    CErr("Only real GTOs are allowed",std::cout);
//  };
  template <typename IntsT>
  void DirectERI<IntsT>::computeSchwartz() {

    CQMemManager &memManager_ = this->memManager();

    if( schwartz() != nullptr ) memManager_.free(schwartz());

    // Allocate the schwartz tensor
    size_t nShell = basisSet().nShell;
    schwartz() = memManager_.malloc<double>(nShell*nShell);

    // Define the libint2 integral engine
    libint2::Engine engine(libint2::Operator::coulomb,
      basisSet().maxPrim,basisSet().maxL,0);

    engine.set_precision(0.); // Don't screen prims during evaluation

    const auto &buf_vec = engine.results();

    auto topSch = std::chrono::high_resolution_clock::now();
  
    size_t n1,n2;
    for(auto s1(0ul); s1 < basisSet().nShell; s1++) {
      n1 = basisSet().shells[s1].size(); // Size shell 1
    for(auto s2(0ul); s2 <= s1; s2++) {
      n2 = basisSet().shells[s2].size(); // Size shell 2



      // Evaluate the shell quartet (s1 s2 | s1 s2)
      engine.compute(
        basisSet().shells[s1],
        basisSet().shells[s2],
        basisSet().shells[s1],
        basisSet().shells[s2]
      );

      if(buf_vec[0] == nullptr) continue;

      // Allocate space to hold the diagonals
      double* diags = memManager_.malloc<double>(n1*n2);

      for(auto i(0), ij(0); i < n1; i++)
      for(auto j(0); j < n2; j++, ij++)
        diags[i + j*n1] = buf_vec[0][ij*n1*n2 + ij];


      schwartz()[s1 + s2*basisSet().nShell] =
        std::sqrt(MatNorm<double>('I',n1,n2,diags,n1));

      // Free up space
      memManager_.free(diags);

    } // loop s2
    } // loop s1

    auto botSch = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> durSch = botSch - topSch;

    HerMat('L',basisSet().nShell,schwartz(),basisSet().nShell);

#if 0
    prettyPrintSmart(std::cout,"Schwartz",schwartz,basisSet_.nShell,
      basisSet_.nShell,basisSet_.nShell);
#endif

  }; // DirectERI<double>::computeSchwartz
  template void DirectERI<double>::computeSchwartz();
  template void DirectERI<dcomplex>::computeSchwartz();

}; // namespace ChronusQ

