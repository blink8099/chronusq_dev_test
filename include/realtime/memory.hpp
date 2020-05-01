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

#include <realtime.hpp>


namespace ChronusQ {

  template <template <typename, typename> class _SSTyp, typename IntsT>
  void RealTime<_SSTyp,IntsT>::alloc() {

    size_t OSize = 
      memManager_.template getSize<dcomplex>(propagator_.onePDM[0]);

    for(auto i = 0; i < propagator_.onePDM.size(); i++) {
      DOSav.emplace_back(memManager_.template malloc<dcomplex>(OSize));
      UH.emplace_back(memManager_.template malloc<dcomplex>(OSize));
    }

  };


  template <template <typename, typename> class _SSTyp, typename IntsT>
  void RealTime<_SSTyp,IntsT>::dealloc() {

    for(auto &X : DOSav) memManager_.free(X);
    for(auto &X : UH)    memManager_.free(X);

  };

}; // namespace ChronusQ


