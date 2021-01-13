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
#include <matrix.hpp>
#include <libint2/shell.h>

namespace ChronusQ {

  /**
   *  \brief Templated class to handle the evaluation and storage of
   *  one electron integral matrix in a finite basis set.
   *
   *  Templated over storage type (IntsT) to allow for a seamless
   *  interface to both real- and complex-valued basis sets
   *  (e.g., GTO and GIAO)
   */
  template <typename IntsT>
  class OneEInts : public ElectronIntegrals {

    template <typename IntsU>
    friend class OneEInts;

  protected:
    SquareMatrix<IntsT> mat_; ///< One Electron integrals (2 index)

  public:

    // Constructor
    OneEInts() = delete;
    OneEInts(CQMemManager &mem, size_t nb):
        ElectronIntegrals(mem, nb), mat_(mem, nb) {}
    OneEInts( const OneEInts &other ) = default;
    template <typename IntsU>
    OneEInts( const OneEInts<IntsU> &other, int = 0 ):
        ElectronIntegrals(other), mat_(other.mat_) {}
    OneEInts( OneEInts &&other ) = default;
    OneEInts( const SquareMatrix<IntsT> &other ):
        ElectronIntegrals(other.memManager(), other.dimension()),
        mat_(other) {}
    template <typename IntsU>
    OneEInts( const SquareMatrix<IntsU> &other, int = 0 ):
        ElectronIntegrals(other.memManager(), other.dimension()),
        mat_(other) {}
    OneEInts( SquareMatrix<IntsT> &&other ):
        ElectronIntegrals(other.memManager(), other.dimension()),
        mat_(std::move(other)) {}

    OneEInts& operator=( const OneEInts &other ) {
      NB = other.nBasis();
      mat_ = other.mat_;
      return *this;
    }
    OneEInts& operator=( OneEInts &&other ) {
      NB = other.nBasis();
      mat_ = std::move(other.mat_);
      return *this;
    }
    template <typename IntsU>
    OneEInts& operator=( const SquareMatrix<IntsU> &other ) {
      NB = other.dimension();
      mat_ = other;
      return *this;
    }
    template <typename IntsU>
    OneEInts& operator=( SquareMatrix<IntsU> &&other ) {
      NB = other.dimension();
      mat_ = std::move(other);
      return *this;
    }

    IntsT& operator()(size_t p, size_t q) {
      return mat_(p,q);
    }
    IntsT operator()(size_t p, size_t q) const {
      return mat_(p,q);
    }

    // Matrix direct access
    SquareMatrix<IntsT>& matrix() { return mat_; }
    const SquareMatrix<IntsT>& matrix() const { return mat_; }
    IntsT* pointer() { return mat_.pointer(); }
    const IntsT* pointer() const { return mat_.pointer(); }

    // Computation interfaces
    virtual void computeAOInts(BasisSet&, Molecule&, EMPerturbation&,
        OPERATOR, const AOIntsOptions&);

    virtual void clear() { mat_.clear(); }

    virtual void output(std::ostream &out, const std::string &s = "",
                        bool printFull = false) const {
      std::string oeiStr;
      if (s == "")
        oeiStr = "One-electron integral";
      else
        oeiStr = "OEI[" + s + "]";
      if (printFull)
        prettyPrintSmart(out, oeiStr, pointer(), this->nBasis(),
                         this->nBasis(), this->nBasis());
      else {
        out << oeiStr << std::endl;
      }
    }

    template <typename IntsU>
    PauliSpinorSquareMatrices<IntsU> spinScatter() const {
      return mat_.template spinScatter<IntsU>();
    }

    template <typename IntsU>
    OneEInts<IntsU> spatialToSpinBlock() const {
      return mat_.template spatialToSpinBlock<IntsU>();
    }

    template <typename TransT>
    OneEInts<typename std::conditional<
    (std::is_same<IntsT, dcomplex>::value or
     std::is_same<TransT, dcomplex>::value),
    dcomplex, double>::type> transform(
        char TRANS, const TransT* T, int NT, int LDT) const {
      return mat_.transform(TRANS, T, NT, LDT);
    }

    template <typename TransT, typename OutT>
    void subsetTransform(
        char TRANS, const TransT* T, int LDT,
        const std::vector<std::pair<size_t,size_t>> &off_size,
        OutT* out, bool increment = false) const {
      return mat_.subsetTransform(TRANS, T, LDT, off_size, out, increment);
    }

    static void OneEDriverLibint(libint2::Operator, Molecule&,
        BasisSet&, std::vector<IntsT*>, size_t deriv=0);
    template <size_t NOPER, bool SYMM, typename F>
    static void OneEDriverLocal(const F&,
        std::vector<libint2::Shell>&, std::vector<IntsT*>);

    // Pointer convertor
    template <typename IntsU>
    static std::shared_ptr<OneEInts<IntsU>>
    convert(const std::shared_ptr<OneEInts<IntsT>>&);

    ~OneEInts() {}

  }; // class OneEInts

}; // namespace ChronusQ
