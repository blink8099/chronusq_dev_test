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

#include <aointegrals.hpp>
#include <util/preprocessor.hpp>


// Template for a collective operation on the members of an 
// AOIntegrals object
  
#define AOIntegrals_COLLECTIVE_OP(OP_MEMBER, OP_OP, OP_VEC_OP) \
    /* Copy over meta  */ \
    OP_OP(double,this,other,memManager_,schwartz); \
    \
    /* 1-e Integrals */ \
    OP_OP(IntsT,this,other,memManager_,overlap); \
    OP_OP(IntsT,this,other,memManager_,kinetic); \
    OP_OP(IntsT,this,other,memManager_,potential); \
    OP_VEC_OP(IntsT,this,other,memManager_,lenElecDipole); \
    OP_VEC_OP(IntsT,this,other,memManager_,lenElecQuadrupole); \
    OP_VEC_OP(IntsT,this,other,memManager_,lenElecOctupole); \
    OP_VEC_OP(IntsT,this,other,memManager_,velElecDipole); \
    OP_VEC_OP(IntsT,this,other,memManager_,velElecQuadrupole); \
    OP_VEC_OP(IntsT,this,other,memManager_,velElecOctupole); \
    OP_VEC_OP(IntsT,this,other,memManager_,magDipole); \
    OP_VEC_OP(IntsT,this,other,memManager_,magQuadrupole); \
    \
    /* 2-e Integrals */ \
    OP_OP(IntsT,this,other,memManager_,ERI)



namespace ChronusQ {

  /**
   *  Constructs an AOIntegrals object to another by copy.
   *
   *  \param [in] other AOIntegrals object to copy
   */ 
  template <typename IntsT> 
  AOIntegrals<IntsT>::AOIntegrals(const AOIntegrals<IntsT> &other) :
    AOIntegrals(other.memManager_, other.molecule_, other.basisSet_){

    AOIntegrals_COLLECTIVE_OP(COPY_OTHER_MEMBER,COPY_OTHER_MEMBER_OP,
      COPY_OTHER_MEMBER_VEC_OP);

  }; // AOIntegrals::AOIntegrals(const AOIntegrals &other)

  /**
   *  Constructs an AOIntegrals object to another by move.
   *
   *  \warning Deallocates the passed AOIntegrals object
   *
   *  \param [in] other AOIntegrals object to move
   */ 
  template <typename IntsT> 
  AOIntegrals<IntsT>::AOIntegrals(AOIntegrals<IntsT> &&other) :
    AOIntegrals(other.memManager_, other.molecule_, other.basisSet_){

    AOIntegrals_COLLECTIVE_OP(COPY_OTHER_MEMBER,MOVE_OTHER_MEMBER_OP,
      MOVE_OTHER_MEMBER_VEC_OP);

  }; // AOIntegrals::AOIntegrals(AOIntegrals &&other)


  /**
   *  Deallocates the internal memory an AOIntegrals object
   */ 
  template <typename IntsT> void AOIntegrals<IntsT>::dealloc() {

    AOIntegrals_COLLECTIVE_OP(DUMMY3,DEALLOC_OP_5,DEALLOC_VEC_OP_5);

  }; // AOIntegrals::dealloc()

}; // namespace ChronusQ

// Other implementation files
#include <aointegrals/contract.hpp>
//#include <aointegrals/print.hpp>


