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
#include <util/preprocessor.hpp>
#include <quantum/preprocessor.hpp>
#include <corehbuilder/nonrel.hpp>
#include <corehbuilder/fourcomp.hpp>
#include <corehbuilder/x2c.hpp>
#include <fockbuilder/rofock.hpp>

// Template for a collective operation on the members of a 
// SingleSlater object
  

// FIXME: For copy and move, this only populates the lists, not the
// explicit pointers
#define SingleSlater_COLLECTIVE_OP(OP_OP,OP_VEC_OP) \
  /* Handle Operators */\
  OP_VEC_OP(MatsT,this,other,memManager,fockMatrix); \
  OP_VEC_OP(MatsT,this,other,memManager,fockMatrixOrtho); \
  OP_OP(MatsT,this,other,memManager,coulombMatrix); \
  OP_VEC_OP(MatsT,this,other,memManager,exchangeMatrix); \
  OP_VEC_OP(MatsT,this,other,memManager,twoeH);\
  OP_VEC_OP(MatsT,this,other,memManager,onePDMOrtho);\
  OP_VEC_OP(MatsT,this,other,memManager,curOnePDM);\
  OP_VEC_OP(MatsT,this,other,memManager,deltaOnePDM);\
  OP_OP(MatsT,this,other,memManager,ortho1);\
  OP_OP(MatsT,this,other,memManager,ortho2);\
  OP_VEC_OP(MatsT,this,other,memManager,coreH);\
  OP_VEC_OP(MatsT,this,other,memManager,coreHPerturbed);

namespace ChronusQ {

  /**
   *  Constructs a SingleSlater object from another of a another (possibly the 
   *  same) type by copy.
   *
   *  \param [in] other SingleSlater object to copy
   *  \param [in] dummy Dummy argument to fix calling signature for delegation 
   *    to copy constructor
   */ 
  template <typename MatsT, typename IntsT>
  template <typename MatsU> 
  SingleSlater<MatsT,IntsT>::SingleSlater(const SingleSlater<MatsU,IntsT> &other,int dummy) : 
    //orthoType(other.orthoType),coreType(other.coreType),
    QuantumBase(dynamic_cast<const QuantumBase&>(other)),
    WaveFunctionBase(dynamic_cast<const WaveFunctionBase&>(other)),
    SingleSlaterBase(dynamic_cast<const SingleSlaterBase&>(other)),
    WaveFunction<MatsT,IntsT>(dynamic_cast<const WaveFunction<MatsU,IntsT>&>(other)),
    coreHBuilder(CoreHBuilder<MatsU,IntsT>::template convert<MatsT>(other.coreHBuilder)),
    fockBuilder(FockBuilder<MatsU,IntsT>::template convert<MatsT>(other.fockBuilder)) {

#ifdef _SingleSlaterDebug
    std::cout << "SingleSlater<MatsT>::SingleSlater(const SingleSlater<U>&) "
              << "(this = " << this << ", other = " << &other << ")" 
              << std::endl;
#endif

    SingleSlater_COLLECTIVE_OP(COPY_OTHER_MEMBER_OP, COPY_OTHER_MEMBER_VEC_OP);

  }; // SingleSlater<MatsT>::SingleSlater(const SingleSlater<U> &)



  /**
   *  Constructs a SingleSlater object from another of a another (possibly the 
   *  same) by move.
   *
   *  \warning Deallocates the passed SingleSlater object
   *
   *  \param [in] other SingleSlater object to move
   *  \param [in] dummy Dummy argument to fix calling signature for delegation 
   *    to move constructor
   */ 
  template <typename MatsT, typename IntsT>
  template <typename MatsU> 
  SingleSlater<MatsT,IntsT>::SingleSlater(SingleSlater<MatsU,IntsT> &&other,int dummy) : 
    //orthoType(other.orthoType),coreType(other.coreType),
    QuantumBase(dynamic_cast<QuantumBase&&>(std::move(other))),
    WaveFunctionBase(dynamic_cast<WaveFunctionBase&&>(std::move(other))),
    SingleSlaterBase(dynamic_cast<SingleSlaterBase&&>(std::move(other))),
    WaveFunction<MatsT,IntsT>(dynamic_cast<WaveFunction<MatsU,IntsT>&&>(std::move(other))),
    coreHBuilder(CoreHBuilder<MatsU,IntsT>::template convert<MatsT>(other.coreHBuilder)),
    fockBuilder(FockBuilder<MatsU,IntsT>::template convert<MatsT>(other.fockBuilder)) {

#ifdef _SingleSlaterDebug
    std::cout << "SingleSlater<MatsT>::SingleSlater(SingleSlater<U>&&) "
              << "(this = " << this << ", other = " << &other << ")" 
              << std::endl;
#endif

    SingleSlater_COLLECTIVE_OP(MOVE_OTHER_MEMBER_OP, MOVE_OTHER_MEMBER_VEC_OP);

  }; // SingleSlater<MatsT>::SingleSlater(SingleSlater<U> &&)


  // Delagate the copy constructor to the conversion constructors
  template <typename MatsT, typename IntsT>
  SingleSlater<MatsT,IntsT>::SingleSlater(const SingleSlater<MatsT,IntsT> &other) : 
    SingleSlater(other,0){ };
  template <typename MatsT, typename IntsT>
  SingleSlater<MatsT,IntsT>::SingleSlater(SingleSlater<MatsT,IntsT> &&other) : 
    SingleSlater(std::move(other),0){ };


  /**
   *  Allocates the internal memory a SingleSlater object
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::alloc() {

#ifdef _SingleSlaterDebug
    std::cout << "SingleSlater::alloc (this = " << this << ")" << std::endl;
#endif

    size_t NB = this->aoints.basisSet().nBasis;

    SPIN_OPERATOR_ALLOC(NB,fockMatrix);
    SPIN_OPERATOR_ALLOC(NB,fockMatrixOrtho);
    SPIN_OPERATOR_ALLOC(NB,exchangeMatrix);
    SPIN_OPERATOR_ALLOC(NB,twoeH);
    SPIN_OPERATOR_ALLOC(NB,onePDMOrtho);
    //SPIN_OPERATOR_ALLOC(NB,coreH);
    //SPIN_OPERATOR_ALLOC(NB,coreHPerturbed);
//    SPIN_OPERATOR_ALLOC(NB,ortho1);
//    SPIN_OPERATOR_ALLOC(NB,ortho2);

    coulombMatrix = memManager.template malloc<MatsT>(NB*NB);

    SPIN_OPERATOR_ALLOC(NB,curOnePDM);
    SPIN_OPERATOR_ALLOC(NB,deltaOnePDM);

  }; // SingleSlater<MatsT>::alloc


  /**
   *  Deallocates the internal memory a SingleSlater object
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::dealloc() {

#ifdef _SingleSlaterDebug
    std::cout << "SingleSlater::dealloc (this = " << this << ")" << std::endl;
#endif

    SingleSlater_COLLECTIVE_OP(DEALLOC_OP_5, DEALLOC_VEC_OP_5);

  }; // SingleSlater<MatsT>::dealloc

}; // namespace ChronusQ


// Other implementation files
#include <singleslater/quantum.hpp>   // Quantum declarations
#include <singleslater/fock.hpp>      // Fock matrix header
#include <singleslater/guess.hpp>     // Guess header
#include <singleslater/scf.hpp>       // SCF header
#include <singleslater/extrap.hpp>    // Extrapolate header
#include <singleslater/print.hpp>     // Print header
#include <singleslater/pop.hpp>       // Population analysis

#include <singleslater/kohnsham/impl.hpp> // KS headers
#include <singleslater/kohnsham/fxc.hpp> // KS headers
#include <singleslater/ortho.hpp>

#include <singleslater/hartreefock/scf.hpp> 
#include <singleslater/kohnsham/scf.hpp>  


