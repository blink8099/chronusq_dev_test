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

#include <electronintegrals.hpp>
#include <electronintegrals/oneeints.hpp>

namespace ChronusQ {

  /**
   *  \brief Templated class to handle the evaluation and storage of
   *  dipole, quadrupole, and octupole integral matrices in a finite
   *  basis set.
   *
   *  Templated over storage type (IntsT) to allow for a seamless
   *  interface to both real- and complex-valued basis sets
   *  (e.g., GTO and GIAO)
   */
  template <template <typename> class IntClass, typename IntsT>
  class GradInts : public ElectronIntegrals {

    template <template <typename> class, typename>
    friend class GradInts;

  protected:
    std::vector<IntClass<IntsT>> components_;
    size_t nAtoms_;

  public:

    //
    // Constructors
    //
    
    GradInts() = delete;
    GradInts( const GradInts & ) = default;
    GradInts( GradInts && ) = default;

    // Main constructor
    GradInts(CQMemManager &mem, size_t nBasis, size_t nAtoms):
        ElectronIntegrals(mem, nBasis), nAtoms_(nAtoms) {

      components_.reserve(3*nAtoms_);

      for (size_t i = 0; i < 3*nAtoms_; i++) {
        components_.emplace_back(mem, nBasis);
      }

    }

    // Converter constructor
    template <typename IntsU>
    GradInts( const GradInts<IntClass,IntsU> &other, int = 0 ):
        ElectronIntegrals(other), nAtoms_(other.nAtoms_) {

      if (std::is_same<IntsU, dcomplex>::value
          and std::is_same<IntsT, double>::value)
        CErr("Cannot create a Real GradInts from a Complex one.");

      components_.reserve(other.components_.size());

      for (auto &p : other.components_)
        components_.emplace_back(p);
    }


    //
    // Access
    //

    // Element access by internal storage
    OneEInts<IntsT>& operator[](size_t i) {
      return components_[i];
    }
    const OneEInts<IntsT>& operator[](size_t i) const {
      return components_[i];
    }

    // Element access by atom index and cartesian index
    OneEInts<IntsT>& integralByAtomCart(size_t atom, size_t xyz) {
      return components_[3*atom + xyz];
    }
    const OneEInts<IntsT>& integralByAtomCart(size_t atom, size_t xyz) const {
      return components_[3*atom + xyz];
    }


    //
    // Interface methods
    //

    virtual void computeAOInts(BasisSet&, Molecule&, EMPerturbation&,
        OPERATOR, const AOIntsOptions&);

    virtual void clear() {
      for (OneEInts<IntsT>& c : components_)
        c.clear();
    }

    virtual void output(std::ostream &out, const std::string &s = "",
                        bool printFull = false) const {

      if (printFull) {

        std::string oeiStr;
        if (s == "")
          oeiStr = "GradInts.";
        else
          oeiStr = "GradInts[" + s + "].";

        for (size_t i = 0; i < components_.size(); i++)
          prettyPrintSmart(out, oeiStr + std::to_string(i),
              operator[](i).pointer(),
              this->nBasis(), this->nBasis(), this->nBasis());

      } else {
        std::string oeiStr;
        if (s == "")
          oeiStr = "OneE Gradient Integral";
        else
          oeiStr = "GradInts[" + s + "]";
        out << oeiStr << " with " << nAtoms_ << " atoms.";
        out << std::endl;
      }

    }

    // Conversion to other type
    template <typename TransT>
    GradInts<IntClass, typename std::conditional<
    (std::is_same<IntsT, dcomplex>::value or
     std::is_same<TransT, dcomplex>::value),
    dcomplex, double>::type> transform(
        char TRANS, const TransT* T, int NT, int LDT) const {

      GradInts<IntClass, typename std::conditional<
      (std::is_same<IntsT, dcomplex>::value or
       std::is_same<TransT, dcomplex>::value),
      dcomplex, double>::type> transInts(
          memManager(), NT, nAtoms_);

      for (size_t i = 0; i < components_.size(); i++)
        transInts[i] = (*this)[i].transform(TRANS, T, NT, LDT);

      return transInts;
    }

    ~GradInts() {}

  }; // class GradInts


}; // namespace ChronusQ

