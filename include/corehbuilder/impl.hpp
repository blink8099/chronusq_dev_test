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

#include <corehbuilder.hpp>
#include <corehbuilder/nonrel/impl.hpp>
#include <corehbuilder/x2c.hpp>
#include <corehbuilder/fourcomp.hpp>

#include <typeinfo>
#include <memory>


namespace ChronusQ {

  /**
   *  \brief The pointer convertor. This static function converts
   *  the underlying polymorphism correctly to hold a different
   *  type of matrices. It is called when the corresponding
   *  SingleSlater object is being converted.
   */
  template <typename MatsT, typename IntsT>
  template <typename MatsU>
  std::shared_ptr<CoreHBuilder<MatsU,IntsT>>
  CoreHBuilder<MatsT,IntsT>::convert(const std::shared_ptr<CoreHBuilder<MatsT,IntsT>>& ch) {

    if (not ch) return nullptr;

    const std::type_index tIndex(typeid(*ch));

    if (tIndex == std::type_index(typeid(NRCoreH<MatsT,IntsT>))) {
      return std::make_shared<NRCoreH<MatsU,IntsT>>(
               *std::dynamic_pointer_cast<NRCoreH<MatsT,IntsT>>(ch));

    } else if (tIndex == std::type_index(typeid(X2C<MatsT,IntsT>))) {
      return std::make_shared<X2C<MatsU,IntsT>>(
               *std::dynamic_pointer_cast<X2C<MatsT,IntsT>>(ch));

    } else if (tIndex == std::type_index(typeid(FourComponent<MatsT,IntsT>))) {
      return std::make_shared<FourComponent<MatsU,IntsT>>(
               *std::dynamic_pointer_cast<FourComponent<MatsT,IntsT>>(ch));

    } else {
      std::stringstream ss;
      ss << "CoreHBuilder implementation \"" << typeid(*ch).name() << "\" not registered in convert." << std::endl;
      CErr(ss.str(),std::cout);
    }

    return nullptr;

  }

}
