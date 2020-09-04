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
#include <electronintegrals/twoeints/incore4indexeri.hpp>
#include <cqlinalg/blas1.hpp>
#include <cqlinalg/blasutil.hpp>
#include <cxxapi/output.hpp>

namespace ChronusQ {

  template <typename IntsT>
  class InCoreRIERI : public TwoEInts<IntsT> {

    template <typename IntsU>
    friend class InCoreRIERI;

  protected:
    size_t NBRI, NBNBRI;
    IntsT* ERI3J = nullptr;    ///< Electron-Electron repulsion integrals (3 index)

  public:

    // Constructor
    InCoreRIERI() = delete;
    InCoreRIERI(CQMemManager &mem, size_t nb):
        TwoEInts<IntsT>(mem, nb), NBRI(0), NBNBRI(0) {}
    InCoreRIERI(CQMemManager &mem, size_t nb, size_t nbri):
        TwoEInts<IntsT>(mem, nb), NBRI(nbri) {
      NBNBRI = this->nBasis()*nRIBasis();
      malloc();
    }
    InCoreRIERI( const InCoreRIERI &other ):
        InCoreRIERI(other.memManager(), other.nBasis(), other.nRIBasis()) {
      std::copy_n(other.ERI3J, this->nBasis()*NBNBRI, ERI3J);
    }
    template <typename IntsU>
    InCoreRIERI( const InCoreRIERI<IntsU> &other, int = 0 ):
        InCoreRIERI(other.memManager(), other.nBasis(), other.nRIBasis()) {
      if (std::is_same<IntsU, dcomplex>::value
          and std::is_same<IntsT, double>::value)
        CErr("Cannot create a Real InCoreRIERI from a Complex one.");
      std::copy_n(other.ERI3J, this->nBasis()*NBNBRI, ERI3J);
    }
    InCoreRIERI( InCoreRIERI &&other ): TwoEInts<IntsT>(std::move(other)),
        NBRI(other.NBRI), NBNBRI(other.NBNBRI), ERI3J(other.ERI3J) {
      other.ERI3J = nullptr;
    }

    InCoreRIERI& operator=( const InCoreRIERI &other ) {
      if (this != &other) { // self-assignment check expected
        if (this->nBasis() != other.nBasis()) {
          this->NB = other.NB;
          NBRI = other.NBRI;
          NBNBRI = other.NBNBRI;
          malloc(); // reallocate memory
        }
        std::copy_n(other.ERI3J, this->nBasis()*NBNBRI, ERI3J);
      }
      return *this;
    }
    InCoreRIERI& operator=( InCoreRIERI &&other ) {
      if (this != &other) { // self-assignment check expected
        this->memManager().free(ERI3J);
        this->NB = other.NB;
        NBRI = other.NBRI;
        NBNBRI = other.NBNBRI;
        ERI3J = other.ERI3J;
        other.ERI3J = nullptr;
      }
      return *this;
    }

    size_t nRIBasis() const { return NBRI; }
    void setNRIBasis(size_t nbri) {
      if(NBRI != nbri) {
        NBRI = nbri;
        NBNBRI = this->NB * NBRI;
        this->malloc();
      }
    }

    // Single element interfaces
    virtual IntsT operator()(size_t p, size_t q, size_t r, size_t s) const {
      return operator()(p+q*this->nBasis(), r+s*this->nBasis());
    }
    virtual IntsT operator()(size_t pq, size_t rs) const {
      return InnerProd<IntsT>(NBRI, &ERI3J[pq*NBRI], 1, &ERI3J[rs*NBRI], 1);
    }
    IntsT& operator()(size_t L, size_t p, size_t q) {
      return ERI3J[L + p*NBRI + q*NBNBRI];
    }
    IntsT operator()(size_t L, size_t p, size_t q) const {
      return ERI3J[L + p*NBRI + q*NBNBRI];
    }

    // Tensor direct access
    IntsT* pointer() { return ERI3J; }
    const IntsT* pointer() const { return ERI3J; }

    // Computation interfaces
    virtual void computeAOInts(BasisSet&, Molecule&, EMPerturbation&,
        OPERATOR, const HamiltonianOptions&) {
      CErr("AO integral evaluation is NOT implemented in super class InCoreRIERI.");
    }

    virtual void clear() {
      std::fill_n(ERI3J, this->nBasis()*NBNBRI, IntsT(0.));
    }

    virtual void output(std::ostream &out, const std::string &s = "",
                        bool printFull = false) const {
      if (s == "")
        out << "  Electron repulsion integral:" << std::endl;
      else
        out << "  ERI[" << s << "]:" << std::endl;
      out << "  " << std::setw(28) << "  Contraction Algorithm:";
      out << "INCORE RI (Gemm)";
      out << std::endl;
      if (printFull) {
        out << bannerTop << std::endl;
        size_t NB = this->nBasis(), NBRI = nRIBasis();
        out << std::scientific << std::left << std::setprecision(8);
        for(auto L = 0ul; L < NBRI; L++)
        for(auto i = 0ul; i < NB; i++)
        for(auto j = 0ul; j < NB; j++){
          out << "    (" << L << "|" << i << "," << j << ")  ";
          out << (*this)(L,i,j) << std::endl;
        };
        out << bannerEnd << std::endl;
      }
    }

    InCore4indexERI<IntsT> to4indexERI() {
      InCore4indexERI<IntsT> eri4i(this->memManager(), this->nBasis());
      size_t NB2 = this->nBasis() * this->nBasis();
      Gemm('T','N',NB2,NB2,NBRI,IntsT(1.),pointer(),NBRI,
           pointer(),NBRI,IntsT(0.),eri4i.pointer(),NB2);
      return eri4i;
    }

    template <typename IntsU>
    InCoreRIERI<IntsU> spatialToSpinBlock() const;

    template <typename TransT>
    InCoreRIERI<typename std::conditional<
    (std::is_same<IntsT, dcomplex>::value or
     std::is_same<TransT, dcomplex>::value),
    dcomplex, double>::type> transform(
        char TRANS, const TransT* T, int NT, int LDT) const;

    template <typename TransT, typename OutT>
    void subsetTransform(
        char TRANS, const TransT* T, int LDT,
        const std::vector<std::pair<size_t,size_t>> &off_size,
        OutT* out, bool increment = false) const;

    void malloc() {
      if (ERI3J) this->memManager().free(ERI3J);
      size_t NB3 = this->nBasis()*NBNBRI;
      try { ERI3J = this->memManager().template malloc<IntsT>(NB3); }
      catch(...) {
        std::cout << std::fixed;
        std::cout << "Insufficient memory for the full RI-ERI tensor ("
                  << (NB3/1e9) * sizeof(double) << " GB)" << std::endl;
        std::cout << std::endl << this->memManager() << std::endl;
        CErr();
      }
    }

    virtual ~InCoreRIERI() {
      if(ERI3J) this->memManager().free(ERI3J);
      //if(ERI3K) this->memManager_.free(ERI3K);
    }

  }; // class InCoreRIERI

  template <typename IntsT>
  class InCoreAuxBasisRIERI : public InCoreRIERI<IntsT> {

    template <typename IntsU>
    friend class InCoreAuxBasisRIERI;

  protected:
    std::shared_ptr<BasisSet> auxBasisSet_ = nullptr; ///< BasisSet for the GTO basis defintion

  public:

    // Constructor
    InCoreAuxBasisRIERI() = delete;
    InCoreAuxBasisRIERI(CQMemManager &mem, size_t nb):
        InCoreRIERI<IntsT>(mem, nb) {}
    InCoreAuxBasisRIERI(CQMemManager &mem, size_t nb, size_t nbri):
        InCoreRIERI<IntsT>(mem, nb, nbri) {}
    InCoreAuxBasisRIERI(CQMemManager &mem, size_t nb,
        std::shared_ptr<BasisSet> auxBasisSet):
        InCoreRIERI<IntsT>(mem, nb, auxBasisSet->nBasis),
        auxBasisSet_(auxBasisSet) {}
    InCoreAuxBasisRIERI( const InCoreAuxBasisRIERI& ) = default;
    template <typename IntsU>
    InCoreAuxBasisRIERI( const InCoreAuxBasisRIERI<IntsU> &other, int = 0 ):
        InCoreRIERI<IntsT>(other) {
      auxBasisSet_ = other.auxBasisSet_;
    }
    InCoreAuxBasisRIERI( InCoreAuxBasisRIERI &&other ) = default;

    InCoreAuxBasisRIERI& operator=( const InCoreAuxBasisRIERI& ) = default;
    InCoreAuxBasisRIERI& operator=( InCoreAuxBasisRIERI&& ) = default;

    void setAuxBasisSet(std::shared_ptr<BasisSet> auxbasisSet) {
      auxBasisSet_ = auxbasisSet;
      this->setNRIBasis(auxBasisSet_->nBasis);
    }
    std::shared_ptr<BasisSet> auxbasisSet() const { return auxBasisSet_; }

    // Computation interfaces
    virtual void computeAOInts(BasisSet&, Molecule&, EMPerturbation&,
        OPERATOR, const HamiltonianOptions&);

    virtual void output(std::ostream &out, const std::string &s = "",
                        bool printFull = false) const {
      if (s == "")
        out << "  Electron repulsion integral:" << std::endl;
      else
        out << "  ERI[" << s << "]:" << std::endl;
      out << "  " << std::setw(28) << "  Contraction Algorithm:";
      out << "INCORE auxiliary basis RI (Gemm)";
      out << std::endl;
      if (printFull) {
        out << bannerTop << std::endl;
        size_t NB = this->nBasis(), NBRI = this->nRIBasis();
        out << std::scientific << std::left << std::setprecision(8);
        for(auto L = 0ul; L < NBRI; L++)
        for(auto i = 0ul; i < NB; i++)
        for(auto j = 0ul; j < NB; j++){
          out << "    (" << L << "|" << i << "," << j << ")  ";
          out << (*this)(L,i,j) << std::endl;
        };
        out << bannerEnd << std::endl;
      }
    }

    virtual ~InCoreAuxBasisRIERI() {}

  }; // class InCoreAuxBasisRIERI

  template <typename IntsT>
  class InCoreCholeskyRIERI : public InCoreRIERI<IntsT> {

    template <typename IntsU>
    friend class InCoreCholeskyRIERI;

  protected:
    double delta_; // Maximum error allowed in the decomposition
    std::vector<size_t> pivots; // List of selected pivots

  public:

    // Constructor
    InCoreCholeskyRIERI() = delete;
    InCoreCholeskyRIERI(CQMemManager &mem, size_t nb):
        InCoreRIERI<IntsT>(mem, nb), delta_(0.0) {}
    InCoreCholeskyRIERI(CQMemManager &mem, size_t nb, double delta):
        InCoreRIERI<IntsT>(mem, nb), delta_(delta) {}
    InCoreCholeskyRIERI( const InCoreCholeskyRIERI& ) = default;
    template <typename IntsU>
    InCoreCholeskyRIERI( const InCoreCholeskyRIERI<IntsU> &other, int = 0 ):
        InCoreRIERI<IntsT>(other) {
      delta_ = other.delta_;
    }
    InCoreCholeskyRIERI( InCoreCholeskyRIERI &&other ) = default;

    InCoreCholeskyRIERI& operator=( const InCoreCholeskyRIERI& ) = default;
    InCoreCholeskyRIERI& operator=( InCoreCholeskyRIERI&& ) = default;

    void setDelta( double delta ) { delta_ = delta; }
    double delta() const { return delta_; }
    const std::vector<size_t>& getPivots() const { return pivots; }

    // Computation interfaces
    virtual void computeAOInts(BasisSet&, Molecule&, EMPerturbation&,
                               OPERATOR, const HamiltonianOptions&);

    virtual void output(std::ostream &out, const std::string &s = "",
                        bool printFull = false) const {
      if (s == "")
        out << "  Electron repulsion integral:" << std::endl;
      else
        out << "  ERI[" << s << "]:" << std::endl;
      out << "  " << std::setw(28) << "  Contraction Algorithm:";
      out << "INCORE Cholesky decomposition RI (Gemm)";
      out << std::endl;
      if (printFull) {
        out << bannerTop << std::endl;
        size_t NB = this->nBasis(), NBRI = this->nRIBasis();
        out << std::scientific << std::left << std::setprecision(8);
        for(auto L = 0ul; L < NBRI; L++)
        for(auto i = 0ul; i < NB; i++)
        for(auto j = 0ul; j < NB; j++){
          out << "    (" << L << "|" << i << "," << j << ")  ";
          out << (*this)(L,i,j) << std::endl;
        };
        out << bannerEnd << std::endl;
      }
    }

    virtual ~InCoreCholeskyRIERI() {}

  }; // class InCoreCholeskyRIERI

  template <typename MatsT, typename IntsT>
  class InCoreRIERIContraction : public InCore4indexERIContraction<MatsT,IntsT> {

    template <typename MatsU, typename IntsU>
    friend class InCoreRIERIContraction;

  public:

    // Constructors

    InCoreRIERIContraction() = delete;
    InCoreRIERIContraction(TwoEInts<IntsT> &eri):
      InCore4indexERIContraction<MatsT,IntsT>(eri) {}

    template <typename MatsU>
    InCoreRIERIContraction(
        const InCoreRIERIContraction<MatsU,IntsT> &other, int dummy = 0 ):
      InCoreRIERIContraction(other.ints_) {}
    template <typename MatsU>
    InCoreRIERIContraction(
        InCoreRIERIContraction<MatsU,IntsT> &&other, int dummy = 0 ):
      InCoreRIERIContraction(other.ints_) {}

    InCoreRIERIContraction( const InCoreRIERIContraction &other ):
      InCoreRIERIContraction(other, 0) {}
    InCoreRIERIContraction( InCoreRIERIContraction &&other ):
      InCoreRIERIContraction(std::move(other), 0) {}

    // Computation interfaces
    virtual void JContract(
        MPI_Comm,
        TwoBodyContraction<MatsT>&) const;

    virtual void KContract(
        MPI_Comm,
        TwoBodyContraction<MatsT>&) const;

    void KCoefContract(
        MPI_Comm, size_t nO, MatsT *X, MatsT *AX) const;

    virtual ~InCoreRIERIContraction() {}

  }; // class ERInts

}; // namespace ChronusQ
