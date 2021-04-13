/* 
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *  
 *  Copyright (C) 2014-2020 Li Research Group (University of Washington)
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
#pragma once

#include <singleslater.hpp>
#include <singleslater/neo_singleslater.hpp>
#include <util/preprocessor.hpp>
#include <quantum/preprocessor.hpp>
#include <corehbuilder/impl.hpp>
//#include <particleintegrals/squarematrix/impl.hpp>
#include <particleintegrals/twopints/impl.hpp>
#include <fockbuilder/rofock.hpp>

namespace ChronusQ {

  /**
   *  Constructs a NEOSingleSlater object from another of another (possibly the 
   *  same) type by copy
   *
   *  \param [in] other NEOSingleSlater object to copy
   *  \param [in] dummy Dummy argument to fix calling signiture for delegation 
   *    to copy constructor
   */
  template <typename MatsT, typename IntsT>
  template <typename MatsU>
  NEOSingleSlater<MatsT,IntsT>::NEOSingleSlater(const NEOSingleSlater<MatsU,IntsT> &other,
                                                int dummy) :
    SingleSlater<MatsT,IntsT>(dynamic_cast<const SingleSlater<MatsU,IntsT>&>(other),dummy),
    //aux_neoss(std::make_shared<NEOSingleSlater<MatsT,IntsT>>(*other.aux_neoss)),
    epJMatrix(std::make_shared<PauliSpinorSquareMatrices<MatsT>>(*other.epJMatrix)),
    EPAI(TPIContractions<MatsU,IntsT>::template convert<MatsT>(other.EPAI))
    { 
      // Allocate NEOSingleSlater Object
      alloc(); 
    
    }; // NEOSingleSlater<MatsT>::NEOSingleSlater(const NEOSingleSlater<U> &)

  /**
   *  Constructs a NEOSingleSlater object from another of a another (possibly the
   *  same) by move
   *
   *  \warning Deallocates the passed NEOSingleSlater object
   *
   *  \param [in] other NEOSingleSlater object to move
   *  \param [in] dummy Dummy argument to fix calling signiture for delegation 
   *    to move constructor
   */
  template <typename MatsT, typename IntsT>
  template <typename MatsU>
  NEOSingleSlater<MatsT,IntsT>::NEOSingleSlater(NEOSingleSlater<MatsU,IntsT> &&other,
                                                int dummy) :
    SingleSlater<MatsT,IntsT>(dynamic_cast<SingleSlater<MatsU,IntsT>&&>(std::move(other)),dummy),
    //aux_neoss(std::make_shared<NEOSingleSlater<MatsT,IntsT>>(std::move(*other.aux_neoss))),
    epJMatrix(std::make_shared<PauliSpinorSquareMatrices<MatsT>>(std::move(*other.epJMatrix))),
    EPAI(TPIContractions<MatsU,IntsT>::template convert<MatsT>(other.EPAI))
    {}; // NEOSingleSlater<MatsT>::NEOSingleSlater(NEOSingleSlater<U> &&)

  
  // Delegate the copy constructor to the conversion constructors
  template <typename MatsT, typename IntsT>
  NEOSingleSlater<MatsT,IntsT>::NEOSingleSlater(const NEOSingleSlater<MatsT,IntsT> &other) : 
    NEOSingleSlater(other,0){ };
  template <typename MatsT, typename IntsT>
  NEOSingleSlater<MatsT,IntsT>::NEOSingleSlater(NEOSingleSlater<MatsT,IntsT> &&other) : 
    NEOSingleSlater(std::move(other),0){ };

  /**
   *  Allocate the internal memory for a NEOSingleSlater object
   */
  template <typename MatsT, typename IntsT>
  void NEOSingleSlater<MatsT,IntsT>::alloc() {
    
    size_t NB = this->basisSet().nBasis;

    epJMatrix = std::make_shared<SquareMatrix<MatsT>>(this->memManager, NB);

  }; // NEOSingleSlater<MatsT>::alloc

  /**
   *  Deallocates the internal memory for a NEOSingleSlater object
   */
  template <typename MatsT, typename IntsT>
  void NEOSingleSlater<MatsT,IntsT>::dealloc() {

    epJMatrix = nullptr;

  }; // NEOSingleSlater<MatsT>::dealloc

}; // namespace ChronusQ

// Other implementation files
#include <singleslater/neo_singleslater/guess.hpp>        // Guess header
#include <singleslater/neo_singleslater/scf.hpp>          // SCF heeader
#include <singleslater/neo_singleslater/fock.hpp>         // Fock matrix header

#include <singleslater/neo_singleslater/neo_kohnsham.hpp> // NEO-KS headers
#include <singleslater/neo_singleslater/neo_kohnsham/impl.hpp> // NEO-KS headers
