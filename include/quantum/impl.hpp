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
#ifndef __INCLUDED_QUANTUM_IMPL_HPP__
#define __INCLUDED_QUANTUM_IMPL_HPP__

#include <quantum.hpp>
#include <util/preprocessor.hpp>
#include <quantum/preprocessor.hpp>

// Template for a collective operation on the members of a 
// Quantum object
  
#define Quantum_COLLECTIVE_OP(OP_MEMBER,OP_VEC_OP) \
  /* Handle densities */\
  OP_VEC_OP(MatsT,this,other,memManager,onePDM); 



namespace ChronusQ {

  /**
   *  Constructs a Quantum object from another of a another (possibly the same) 
   *  type by copy.
   *
   *  \param [in] other Quantum object to copy
   *  \param [in] dummy Dummy argument to fix calling signature for delegation 
   *    to copy constructor
   */ 
  template <typename MatsT>
  template <typename MatsU>
  Quantum<MatsT>::Quantum(const Quantum<MatsU> &other, int dummy) : 
    QuantumBase(dynamic_cast<const QuantumBase&>(other)) {

    #ifdef _QuantumDebug
    std::cout << "Quantum<T>::Quantum(const Quantum<U>&) (this = " << this 
              << ", other = " << &other << ")" << std::endl;
    #endif

    Quantum_COLLECTIVE_OP(COPY_OTHER_MEMBER,COPY_OTHER_MEMBER_VEC_OP);

  }; // Quantum<T>::Quantum(const Quantum<U> &)
    

  /**
   *  Constructs a Quantum object from another of a another (possibly the same) 
   *  type by move.
   *
   *  \warning Deallocates the passed Quantum object
   *
   *  \param [in] other Quantum object to move
   *  \param [in] dummy Dummy argument to fix calling signature for delegation 
   *    to move constructor
   */ 
  template <typename MatsT>
  template <typename MatsU>
  Quantum<MatsT>::Quantum(Quantum<MatsU> &&other, int dummy) : 
    QuantumBase(dynamic_cast<QuantumBase&&>(std::move(other))) {

    #ifdef _QuantumDebug
    std::cout << "Quantum<T>::Quantum(Quantum<U>&&) (this = " << this 
              << ", other = " << &other << ")" << std::endl;
    #endif

    Quantum_COLLECTIVE_OP(MOVE_OTHER_MEMBER,MOVE_OTHER_MEMBER_VEC_OP);

  }; // Quantum<T>::Quantum(Quantum<U> &&)

  // Delagate the copy constructor to the conversion constructors
  template <typename MatsT>
  Quantum<MatsT>::Quantum(const Quantum<MatsT> &other) : Quantum(other,0){ };
  template <typename MatsT>
  Quantum<MatsT>::Quantum(Quantum<MatsT> &&other) : Quantum(std::move(other),0){ };




  /**
   *  Allocates the internal memory a Quantum object
   *
   *  \param [in] N Dimension of density matricies
   */ 
  template <typename MatsT>
  void Quantum<MatsT>::alloc(size_t N) {

    #ifdef _QuantumDebug
    std::cout << "Quantum::alloc (this = " << this << ")" << std::endl;
    #endif

    SPIN_OPERATOR_ALLOC(N,onePDM);

  }; // Quantum<T>::alloc


  /**
   *  Deallocates the internal memory a Quantum object
   */ 
  template <typename MatsT>
  void Quantum<MatsT>::dealloc() {

    #ifdef _QuantumDebug
    std::cout << "Quantum::dealloc (this = " << this << ")" << std::endl;
    #endif

    // Deallocate the 1PDM
    DEALLOC_VEC_OP(memManager,onePDM);

  }; // Quantum<T>::dealloc

}; // namespace ChronusQ


// Other headers
#include <quantum/print.hpp>

#endif
