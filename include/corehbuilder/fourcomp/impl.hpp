/*
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *
 *  Copyright (C) 2014-2022 Li Research Group (University of Washington)
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

#include <corehbuilder/fourcomp.hpp>

namespace ChronusQ {

  template <typename MatsT, typename IntsT>
  FourComponent<MatsT,IntsT>::FourComponent(const FourComponent<MatsT,IntsT> &other) :
    FourComponent(other,0) {}

  template <typename MatsT, typename IntsT>
  FourComponent<MatsT,IntsT>::FourComponent(FourComponent<MatsT,IntsT> &&other) :
    FourComponent(std::move(other),0) {}

  template <typename MatsT, typename IntsT>
  template <typename MatsU>
  FourComponent<MatsT,IntsT>::FourComponent(const FourComponent<MatsU,IntsT> &other, int dummy) :
    CoreHBuilder<MatsT,IntsT>(other) {}

  template <typename MatsT, typename IntsT>
  template <typename MatsU>
  FourComponent<MatsT,IntsT>::FourComponent(FourComponent<MatsU,IntsT> &&other, int dummy) :
    CoreHBuilder<MatsT,IntsT>(other) {}
}
