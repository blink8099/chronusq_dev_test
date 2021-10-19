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
      size_t oneCompFactor  = (ss_.nC == 1) ? 2: 1;

      size_t nO = ss_.nO/oneCompFactor - nFrozenCore;
      size_t nV = ss_.nV/oneCompFactor - nFrozenVirt; 
      size_t nT = nO + nV;
      
      std::cout << "nO = " << nO << std::endl;
      std::cout << "nV = " << nV << std::endl;

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
  
  /*
   * \brief get unique symbols
   */ 
  template <typename MatsT, typename IntsT>
  char MOIntsTransformer<MatsT,IntsT>::getUniqueSymbol(char type) {
  
      for (auto i = 0ul; i < symbol_sets_.size(); i++) {
        if (symbol_sets_[i].count(type)) {
          return *symbol_sets_[i].begin();
        }
      }
      CErr("Wrong MO Type in parseMOIntsType");
  }; // MOIntsTransformer::getUniqueSymbol

  /**
   *  \brief print offsizes 
   */
  template <typename MatsT, typename IntsT>
  void MOIntsTransformer<MatsT,IntsT>::printOffSizes(
    const std::vector<std::pair<size_t,size_t>> & off_size) { 
    std::cout << "----MO Ranges: " << std::endl;
    
    size_t counter = 0;
    for (auto & size: off_size) {
      counter ++;
      std::cout << "Index " << counter << ": from " << std::setw(5)
                << size.first + 1 << " to " << std::setw(5)
                << size.first + size.second << std::endl;
    }
    std::cout << std::endl;
  }


}; // namespace ChronusQ
