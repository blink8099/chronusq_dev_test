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
  template <typename IntsT>
  class MultipoleInts : public ElectronIntegrals {

    template <typename IntsU>
    friend class MultipoleInts;

  protected:
    size_t highOrder_ = 0;
    bool symmetric_ = false;
    std::vector<OneEInts<IntsT>> components_;

    void orderCheck(size_t order) const {
      if (order > highOrder())
        CErr("Order of XYZ components exceeds highest order in this MultipoleInts.");
      if (order <= 0)
        CErr("No zeroth-order components in MultipoleInts.");
    }

    size_t nComponentsOnOrder(size_t order) const {
      if (symmetric()) {
        return (order + 2) * (order + 1) / 2;
      } else {
        size_t pow = 3;
        for (size_t i = 2; i <= order; i++) {
          pow *= 3;
        }
        return pow;
      }
    }

    size_t cumeComponents(size_t order) const {
      size_t count = 0;
      if (symmetric()) {
        for (size_t i = 1; i <= order; i++)
          count += (i + 2) * (i + 1) / 2;
      } else {
        size_t pow = 3;
        for (size_t i = 1; i <= order; i++) {
          count += pow;
          pow *= 3;
        }
      }
      return count;
    }

    size_t index(std::vector<size_t> comps) const {
      size_t order = comps.size();
      orderCheck(order);
      size_t count = cumeComponents(order-1);
      if (symmetric()) {
        std::array<size_t, 3> ncomps{0,0,0};
        for (size_t i = 0; i < order; i++) {
          if (comps[i] > 2)
            CErr("MultipoleInts Component label must be"
                 " combinations of 'X', 'Y', and 'Z'");
          ncomps[comps[i]]++;
        }
        size_t yz = ncomps[1] + ncomps[2];
        count += yz * (yz+1) / 2 + ncomps[2];
      } else {
        size_t pow = 1;
        for (int i = order - 1; i >= 0; i--) {
          if (comps[i] > 2)
            CErr("MultipoleInts Component label must be"
                 " combinations of 'X', 'Y', and 'Z'");
          count += comps[i] * pow;
          pow *= 3;
        }
      }
      return count;
    }

    std::vector<size_t> indices(std::string s) const {
      std::vector<size_t> inds(s.size());
      std::transform(s.begin(), s.end(), inds.begin(),
          [](unsigned char c){ return std::toupper(c) - 'X';});
      return inds;
    }

    std::string label(size_t i) const {
      size_t order = 1, orderNComp = 3;
      while (i >= orderNComp) {
        i -= orderNComp;
        orderNComp = nComponentsOnOrder(++order);
      }

      std::string s;
      if (symmetric()) {
        size_t yz = static_cast<size_t>(std::sqrt(2*i+0.25)-0.5);
        std::array<size_t, 3> ncomps{0,0,0};
        ncomps[0] = order - yz;
        ncomps[2] = i - yz*(yz+1)/2;
        ncomps[1] = yz - ncomps[2];
        for (size_t j = 0; j < 3; j++)
          for (size_t k = 0; k < ncomps[j]; k++)
            s += static_cast<char>('X'+j);
      } else
        for (size_t j = 0; j < order; j++) {
          s = static_cast<char>('X'+i%3) + s;
          i /= 3;
        }
      return s;
    }

  public:

    // Constructor
    MultipoleInts() = delete;
    MultipoleInts( const MultipoleInts & ) = default;
    MultipoleInts( MultipoleInts && ) = default;
    MultipoleInts(CQMemManager &mem, size_t nb, size_t order, bool symm):
        ElectronIntegrals(mem, nb), highOrder_(order), symmetric_(symm) {
      if (order == 0)
        CErr("MultipoleInts order must be at least 1.");
      size_t size = cumeComponents(order);
      components_.reserve(size);
      for (size_t i = 0; i < size; i++) {
        components_.emplace_back(mem, nb);
      }
    }

    template <typename IntsU>
    MultipoleInts( const MultipoleInts<IntsU> &other, int = 0 ):
        ElectronIntegrals(other),
        highOrder_(other.highOrder_), symmetric_(other.symmetric_) {
      if (std::is_same<IntsU, dcomplex>::value
          and std::is_same<IntsT, double>::value)
        CErr("Cannot create a Real MultipoleInts from a Complex one.");
      components_.reserve(other.components_.size());
      for (auto &p : other.components_)
        components_.emplace_back(p);
    }

    size_t size() const { return components_.size(); }
    bool symmetric() const { return symmetric_; }
    size_t highOrder() const { return highOrder_; }

    OneEInts<IntsT>& operator[](size_t i) {
      return components_[i];
    }

    const OneEInts<IntsT>& operator[](size_t i) const {
      return components_[i];
    }

    OneEInts<IntsT>& operator[](std::string s) {
      return components_[index(indices(s))];
    }

    const OneEInts<IntsT>& operator[](std::string s) const {
      return components_[index(indices(s))];
    }

    std::vector<IntsT*> pointersByOrder(size_t order) {
      orderCheck(order);
      size_t iStart = cumeComponents(order - 1);
      size_t iEnd = cumeComponents(order);
      std::vector<IntsT*> ps(iEnd - iStart);
      std::transform(components_.begin()+iStart,
          components_.begin()+iEnd, ps.begin(),
          [](OneEInts<IntsT> &oei){ return oei.pointer(); });
      return ps;
    }

    std::vector<IntsT*> dipolePointers() {
      return pointersByOrder(1);
    }

    std::vector<IntsT*> quadrupolePointers() {
      return pointersByOrder(2);
    }

    std::vector<IntsT*> octupolePointers() {
      return pointersByOrder(3);
    }

    // Computation interfaces
    virtual void computeAOInts(BasisSet&, Molecule&, EMPerturbation&,
        OPERATOR, const AOIntsOptions&);

    virtual void output(std::ostream &out, const std::string &s = "",
                        bool printFull = false) const {
      if (printFull) {
        std::string oeiStr;
        if (s == "")
          oeiStr = "MultipoleInts.";
        else
          oeiStr = "MultipoleInts[" + s + "].";
        for (size_t i = 0; i < size(); i++)
          prettyPrintSmart(out, oeiStr+label(i), operator[](i).pointer(),
              this->nBasis(), this->nBasis(), this->nBasis());
      } else {
        std::string oeiStr;
        if (s == "")
          oeiStr = "Multipole integral";
        else
          oeiStr = "MultipoleInts[" + s + "]";
        out << oeiStr << " up to ";
        switch (highOrder()) {
        case 1:
          out << "Dipole";
          break;
        case 2:
          out << "Quadrupole";
          break;
        case 3:
          out << "Octupole";
          break;
        default:
          out << highOrder() << "-th order";
          break;
        }
        out << std::endl;
      }
    }

    template <typename TransT>
    MultipoleInts<typename std::conditional<
    (std::is_same<IntsT, dcomplex>::value or
     std::is_same<TransT, dcomplex>::value),
    dcomplex, double>::type> transform(
        char TRANS, const TransT* T, int NT, int LDT) const {
      MultipoleInts<typename std::conditional<
      (std::is_same<IntsT, dcomplex>::value or
       std::is_same<TransT, dcomplex>::value),
      dcomplex, double>::type> transInts(
          memManager(), NT, highOrder(), symmetric());
      for (size_t i = 0; i < size(); i++)
        transInts[i] = (*this)[i].transform(TRANS, T, NT, LDT);
      return transInts;
    }

    ~MultipoleInts() {}

  }; // class MultipoleInts

}; // namespace ChronusQ
