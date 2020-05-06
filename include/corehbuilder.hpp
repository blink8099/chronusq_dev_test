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

#include <fields.hpp>
#include <singleslater.hpp>
#include <aointegrals.hpp>

namespace ChronusQ {

  /**
   * \brief The CoreHBuilder class
   */
  template <typename MatsT, typename IntsT>
  class CoreHBuilder {

    template <typename MatsU, typename IntsU>
    friend class CoreHBuilder;

  protected:
    AOIntegrals<IntsT> &aoints_;
    OneETerms           oneETerms_; ///< One electron terms to be computed

  public:

    // Constructors

    // Disable default constructor
    CoreHBuilder() = delete;
    CoreHBuilder(AOIntegrals<IntsT> &aoints, OneETerms oneETerms):
      aoints_(aoints), oneETerms_(oneETerms) {}

    // Same or Different type
    template <typename MatsU>
    CoreHBuilder(const CoreHBuilder<MatsU,IntsT> &other):
      aoints_(other.aoints_), oneETerms_(other.oneETerms_) {}
    template <typename MatsU>
    CoreHBuilder(CoreHBuilder<MatsU,IntsT> &&other):
      aoints_(other.aoints_), oneETerms_(other.oneETerms_) {}

    // Virtual destructor
    virtual ~CoreHBuilder() {}


    // Public member functions

    // Compute various core Hamitlonian
    virtual void computeCoreH(EMPerturbation &, std::vector<MatsT*>&) = 0;

    // Compute the gradient
    virtual void getGrad() = 0;

    // Pointer convertor
    template <typename MatsU>
    static std::shared_ptr<CoreHBuilder<MatsU,IntsT>>
    convert(const std::shared_ptr<CoreHBuilder<MatsT,IntsT>>&);

  };

}
