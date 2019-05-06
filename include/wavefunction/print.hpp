/* 
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *  
 *  Copyright (C) 2014-2019 Li Research Group (University of Washington)
 *  
 *  This program is free software; you ca redistribute it and/or modify
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
#ifndef __INCLUDED_WAVEFUNCTION_PRINT_HPP__
#define __INCLUDED_WAVEFUNCTION_PRINT_HPP__

#include <wavefunction.hpp>
#include <util/matout.hpp>

namespace ChronusQ {

  template <typename MatsT, typename IntsT>
  void WaveFunction<MatsT,IntsT>::printMO(std::ostream &out) {

    size_t NB = aoints.basisSet().nBasis * this->nC;
   
    prettyPrintSmart(std::cout,"MO 1",mo1,NB,NB,NB);
    if( mo2 != nullptr )
      prettyPrintSmart(std::cout,"MO 2",mo2,NB,NB,NB);

  }; // WaveFunction<T,MatsT,IntsT>::printMO

  template <typename MatsT, typename IntsT>
  void WaveFunction<MatsT,IntsT>::printEPS(std::ostream &out) {

    size_t NB = aoints.basisSet().nBasis * this->nC;
   
    prettyPrintSmart(std::cout,"EPS 1",eps1,NB,NB,NB);
    if( eps2 != nullptr )
      prettyPrintSmart(std::cout,"EPS 2",eps2,NB,NB,NB);

  }; // WaveFunction<T,MatsT,IntsT>::printEPS


}; // namespace ChronusQ

#endif
