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

namespace ChronusQ {

  /**
   *  \brief transform a subset of HCore 
   */
  template <typename MatsT, typename IntsT>
  void MOIntsTransformer<MatsT,IntsT>::subsetTransformHCore(
    const std::vector<std::pair<size_t,size_t>> &off_sizes, MatsT* MOHCore) {
    
    size_t nAO = ss_.nAlphaOrbital() * ss_.nC;
    
    // populate AOHCore
    if (not AOHCore_) {
      if(ss_.nC == 1) {
        AOHCore_ = std::make_shared<SquareMatrix<MatsT>>(0.5*ss_.coreH->S());
      } else { 
        AOHCore_ = std::make_shared<SquareMatrix<MatsT>>(
          ss_.coreH->template spinGather<MatsT>());
      }
    }
    
    AOHCore_->subsetTransform('N', ss_.mo[0].pointer(), nAO, off_sizes, MOHCore, false); 
  
  }; // MOIntsTransformer::subsetTransformHCore 

}; // namespace ChronusQ
