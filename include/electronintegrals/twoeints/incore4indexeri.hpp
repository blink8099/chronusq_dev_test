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

#include <electronintegrals/twoeints.hpp>
#include <cxxapi/output.hpp>

namespace ChronusQ {

  template <typename IntsT>
  class InCore4indexERI : public TwoEInts<IntsT> {

    template <typename IntsU>
    friend class InCore4indexERI;

  protected:
    size_t NB2, NB3;
    IntsT* ERI = nullptr; ///< Electron-Electron repulsion integrals (4 index)

  public:

    // Constructor
    InCore4indexERI() = delete;
    InCore4indexERI(CQMemManager &mem, size_t nb):
        TwoEInts<IntsT>(mem, nb) {
      NB2 = this->nBasis()*this->nBasis();
      NB3 = NB2 * this->nBasis();
      malloc();
    }
    InCore4indexERI( const InCore4indexERI &other ):
        InCore4indexERI(other.memManager(), other.nBasis()) {
      std::copy_n(other.ERI, NB2*NB2, ERI);
    }
    template <typename IntsU>
    InCore4indexERI( const InCore4indexERI<IntsU> &other, int = 0 ):
        InCore4indexERI(other.memManager(), other.nBasis()) {
      if (std::is_same<IntsU, dcomplex>::value
          and std::is_same<IntsT, double>::value)
        CErr("Cannot create a Real InCore4indexERI from a Complex one.");
      std::copy_n(other.ERI, NB2*NB2, ERI);
    }
    InCore4indexERI( InCore4indexERI &&other ): TwoEInts<IntsT>(std::move(other)),
        NB2(other.NB2), NB3(other.NB3), ERI(other.ERI) { other.ERI = nullptr; }

    InCore4indexERI& operator=( const InCore4indexERI &other ) {
      if (this != &other) { // self-assignment check expected
        if (this->nBasis() != other.nBasis()) {
          this->NB = other.NB;
          NB2 = other.NB2;
          NB3 = other.NB3;
          malloc(); // reallocate memory
        }
        std::copy_n(other.ERI, NB2*NB2, ERI);
      }
      return *this;
    }
    InCore4indexERI& operator=( InCore4indexERI &&other ) {
      if (this != &other) { // self-assignment check expected
        this->memManager().free(ERI);
        this->NB = other.NB;
        NB2 = other.NB2;
        NB3 = other.NB3;
        ERI = other.ERI;
        other.ERI = nullptr;
      }
      return *this;
    }

    // Single element interfaces
    virtual IntsT operator()(size_t p, size_t q, size_t r, size_t s) const {
      return ERI[p + q*this->nBasis() + r*NB2 + s*NB3];
    }
    IntsT& operator()(size_t p, size_t q, size_t r, size_t s) {
      return ERI[p + q*this->nBasis() + r*NB2 + s*NB3];
    }
    virtual IntsT operator()(size_t pq, size_t rs) const {
      return ERI[pq + rs*NB2];
    }
    IntsT& operator()(size_t pq, size_t rs) {
      return ERI[pq + rs*NB2];
    }

    // Tensor direct access
    IntsT* pointer() { return ERI; }
    const IntsT* pointer() const { return ERI; }

    // Computation interfaces
    virtual void computeAOInts(BasisSet&, Molecule&, EMPerturbation&,
        OPERATOR, const AOIntsOptions&);

    virtual void clear() {
      std::fill_n(ERI, NB2*NB2, IntsT(0.));
    }

    virtual void output(std::ostream &out, const std::string &s = "",
                        bool printFull = false) const {
      if (s == "")
        out << "  Electron repulsion integral:" << std::endl;
      else
        out << "  ERI[" << s << "]:" << std::endl;
      out << "  " << std::setw(28) << "  Contraction Algorithm:";
      out << "INCORE (Gemm)";
      out << std::endl;
      if (printFull) {
        out << bannerTop << std::endl;
        size_t NB = this->nBasis();
        out << std::scientific << std::left << std::setprecision(8);
        for(auto i = 0ul; i < NB; i++)
        for(auto j = 0ul; j < NB; j++)
        for(auto k = 0ul; k < NB; k++)
        for(auto l = 0ul; l < NB; l++){
          out << "    (" << i << "," << j << "|" << k << "," << l << ")  ";
          out << operator()(i,j,k,l) << std::endl;
        };
        out << bannerEnd << std::endl;
      }
    }

    void malloc() {
      if(ERI) this->memManager().free(ERI);
      size_t NB4 = NB2*NB2;
      try { ERI = this->memManager().template malloc<IntsT>(NB4); }
      catch(...) {
        std::cout << std::fixed;
        std::cout << "Insufficient memory for the full ERI tensor ("
                  << (NB4/1e9) * sizeof(double) << " GB)" << std::endl;
        std::cout << std::endl << this->memManager() << std::endl;
        CErr();
      }
    }

    template <typename IntsU>
    InCore4indexERI<IntsU> spatialToSpinBlock() const;

    template <typename TransT>
    InCore4indexERI<typename std::conditional<
    (std::is_same<IntsT, dcomplex>::value or
     std::is_same<TransT, dcomplex>::value),
    dcomplex, double>::type> transform(
        char TRANS, const TransT* T, int NT, int LDT) const;

    template <typename TransT, typename OutT>
    void subsetTransform(
        char TRANS, const TransT* T, int LDT,
        const std::vector<std::pair<size_t,size_t>> &off_size,
        OutT* out, bool increment = false) const;

    virtual ~InCore4indexERI() {
      if(ERI) this->memManager().free(ERI);
    }

  }; // class InCore4indexERI

  template <typename MatsT, typename IntsT>
  class InCore4indexERIContraction : public ERIContractions<MatsT,IntsT> {

    template <typename MatsU, typename IntsU>
    friend class InCore4indexERIContraction;

  public:

    // Constructors

    InCore4indexERIContraction() = delete;
    InCore4indexERIContraction(TwoEInts<IntsT> &eri):
      ERIContractions<MatsT,IntsT>(eri) {}

    template <typename MatsU>
    InCore4indexERIContraction(
        const InCore4indexERIContraction<MatsU,IntsT> &other, int dummy = 0 ):
      InCore4indexERIContraction(other.ints_) {}
    template <typename MatsU>
    InCore4indexERIContraction(
        InCore4indexERIContraction<MatsU,IntsT> &&other, int dummy = 0 ):
      InCore4indexERIContraction(other.ints_) {}

    InCore4indexERIContraction( const InCore4indexERIContraction &other ):
      InCore4indexERIContraction(other, 0) {}
    InCore4indexERIContraction( InCore4indexERIContraction &&other ):
      InCore4indexERIContraction(std::move(other), 0) {}

    // Computation interfaces
    virtual void twoBodyContract(
        MPI_Comm comm,
        const bool,
        std::vector<TwoBodyContraction<MatsT>> &list,
        EMPerturbation&) const;

    virtual void JContract(
        MPI_Comm,
        TwoBodyContraction<MatsT>&) const;

    virtual void KContract(
        MPI_Comm,
        TwoBodyContraction<MatsT>&) const;

    virtual ~InCore4indexERIContraction() {}

  }; // class InCore4indexERIContraction

}; // namespace ChronusQ
