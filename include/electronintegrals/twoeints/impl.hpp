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

#include <electronintegrals/twoeints.hpp>
#include <electronintegrals/twoeints/incore4indexeri.hpp>
#include <electronintegrals/twoeints/gtodirecteri.hpp>
#include <electronintegrals/twoeints/giaodirecteri.hpp>
#include <electronintegrals/twoeints/incorerieri.hpp>

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
  std::shared_ptr<ERIContractions<MatsU,IntsT>>
  ERIContractions<MatsT,IntsT>::convert(const std::shared_ptr<ERIContractions<MatsT,IntsT>>& ch) {

    if (not ch) return nullptr;

    const std::type_info &tID(typeid(*ch));

    if (tID == typeid(InCore4indexERIContraction<MatsT,IntsT>)) {
      return std::make_shared<InCore4indexERIContraction<MatsU,IntsT>>(
               *std::dynamic_pointer_cast<InCore4indexERIContraction<MatsT,IntsT>>(ch));

    } else if (tID == typeid(GTODirectERIContraction<MatsT,IntsT>)) {
      return std::make_shared<GTODirectERIContraction<MatsU,IntsT>>(
               *std::dynamic_pointer_cast<GTODirectERIContraction<MatsT,IntsT>>(ch));

    } else if (tID == typeid(GIAODirectERIContraction)) {
      return std::dynamic_pointer_cast<ERIContractions<MatsU,IntsT>>(ch);

    } else if (tID == typeid(InCoreRIERIContraction<MatsT,IntsT>)) {
      return std::make_shared<InCoreRIERIContraction<MatsU,IntsT>>(
               *std::dynamic_pointer_cast<InCoreRIERIContraction<MatsT,IntsT>>(ch));

    } else {
      std::stringstream errMsg;
      errMsg << "ERIContractions implementation \"" << tID.name() << "\" not registered in convert." << std::endl;
      CErr(errMsg.str(),std::cout);
    }

    return nullptr;

  }

}
