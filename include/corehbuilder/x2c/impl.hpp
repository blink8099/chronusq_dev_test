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

#include <corehbuilder/x2c.hpp>
#include <util/preprocessor.hpp>

// FIXME: For copy and move, this only populates the lists, not the
// explicit pointers
#define X2C_COLLECTIVE_OP(OP_OP,OP_VEC_OP) \
  /* Handle Operators */\
  OP_OP(IntsT,this,other,memManager_,mapPrim2Cont);\
  OP_OP(MatsT,this,other,memManager_,W);\
  OP_OP(IntsT,this,other,memManager_,UK);\
  OP_OP(double,this,other,memManager_,p);\
  OP_OP(MatsT,this,other,memManager_,X);\
  OP_OP(MatsT,this,other,memManager_,Y);\
  OP_OP(MatsT,this,other,memManager_,UL);\
  OP_OP(MatsT,this,other,memManager_,US);


namespace ChronusQ {

  template <typename MatsT, typename IntsT>
  X2C<MatsT,IntsT>::X2C(const X2C<MatsT,IntsT> &other) :
    CoreHBuilder<MatsT,IntsT>(other), memManager_(other.memManager_),
    molecule_(other.molecule_), basisSet_(other.basisSet_),
    uncontractedBasis_(other.uncontractedBasis_),
    uncontractedInts_(other.uncontractedInts_) {

    X2C_COLLECTIVE_OP(COPY_OTHER_MEMBER_OP, COPY_OTHER_MEMBER_VEC_OP)

  }

  template <typename MatsT, typename IntsT>
  X2C<MatsT,IntsT>::X2C(X2C<MatsT,IntsT> &&other) :
    CoreHBuilder<MatsT,IntsT>(other), memManager_(other.memManager_),
    molecule_(other.molecule_), basisSet_(other.basisSet_),
    uncontractedBasis_(other.uncontractedBasis_),
    uncontractedInts_(other.uncontractedInts_) {

    X2C_COLLECTIVE_OP(MOVE_OTHER_MEMBER_OP, MOVE_OTHER_MEMBER_VEC_OP)

  }

  template <typename MatsT, typename IntsT>
  void X2C<MatsT,IntsT>::dealloc() {

    X2C_COLLECTIVE_OP(DEALLOC_OP_5, DEALLOC_VEC_OP_5)

  }

}; // namespace ChronusQ