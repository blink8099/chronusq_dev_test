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

namespace ChronusQ {

  template <typename IntsT>
  class DirectERI : public TwoEInts<IntsT> {

    template <typename IntsU>
    friend class DirectERI;

  protected:
    BasisSet &basisSet_;
    double threshSchwartz_ = 1e-12; ///< Schwartz screening threshold
    double* schwartz_ = nullptr;   ///< Schwartz bounds for the ERIs

  public:

    // Constructor
    DirectERI() = delete;
    DirectERI(CQMemManager &mem, BasisSet &basis, double threshSchwartz):
        TwoEInts<IntsT>(mem, basis.nBasis), basisSet_(basis),
        threshSchwartz_(threshSchwartz) {}
    DirectERI( const DirectERI &other ):
        DirectERI(other.memManager(), other.nBasis(), other.threshSchwartz_) {
      std::copy_n(other.schwartz_, basisSet_.nShell*basisSet_.nShell, schwartz_);
    }
    template <typename IntsU>
    DirectERI( const DirectERI<IntsU> &other, int = 0 ):
        DirectERI(other.memManager_, other.basisSet_, other.threshSchwartz_) {
      if (other.schwartz_) {
        size_t NS = basisSet().nShell;
        schwartz_ = this->memManager().template malloc<double>(NS*NS);
        std::copy_n(other.schwartz_, NS*NS, schwartz_);
      }
    }
    DirectERI( DirectERI &&other ): TwoEInts<IntsT>(std::move(other)),
        basisSet_(other.basis), threshSchwartz_(other.threshSchwartz_),
        schwartz_(other.schwartz_) { other.schwartz_ = nullptr; }

    BasisSet& basisSet() { return basisSet_; }
    double threshSchwartz() const { return threshSchwartz_; }
    double*& schwartz() { return schwartz_; }

    // Single element interfaces
    virtual IntsT operator()(size_t p, size_t q, size_t r, size_t s) const {
      CErr("NYI");
    }
    virtual IntsT operator()(size_t pq, size_t rs) const {
      CErr("NYI");
    }

    // Computation interfaces
    virtual void computeAOInts(BasisSet&, Molecule&, EMPerturbation&,
        OPERATOR, const AOIntsOptions&) {}

    virtual void clear() {}

    void computeSchwartz();

    virtual void output(std::ostream &out, const std::string &s = "",
                        bool printFull = false) const {
      if (s == "")
        out << "  Electron repulsion integral:" << std::endl;
      else
        out << "  ERI[" << s << "]:" << std::endl;
      out << "  " << std::setw(28) << "  Contraction Algorithm:";
      out << "DIRECT";
      out << std::endl;

      out << "    * Schwartz Screening Threshold = "
          << threshSchwartz() << std::endl;

      if (printFull) {
        CErr("Printing Full ERI tensor for Direct contraction NYI.", out);
      }
    }

    virtual ~DirectERI() {
      if(schwartz_) this->memManager().free(schwartz_);
    }

  }; // class DirectERI

  template <typename MatsT, typename IntsT>
  class GTODirectERIContraction : public ERIContractions<MatsT,IntsT> {

    template <typename MatsU, typename IntsU>
    friend class GTODirectERIContraction;

  public:

    // Constructors

    GTODirectERIContraction() = delete;
    GTODirectERIContraction(TwoEInts<IntsT> &eri):
      ERIContractions<MatsT,IntsT>(eri) {

      if (typeid(eri) != typeid(DirectERI<IntsT>))
        CErr("GTODirectERIContraction expect a DirectERI reference.");

    }

    template <typename MatsU>
    GTODirectERIContraction(
        const GTODirectERIContraction<MatsU,IntsT> &other, int dummy = 0 ):
      GTODirectERIContraction(other.ints_) {}
    template <typename MatsU>
    GTODirectERIContraction(
        GTODirectERIContraction<MatsU,IntsT> &&other, int dummy = 0 ):
      GTODirectERIContraction(other.ints_) {}

    GTODirectERIContraction( const GTODirectERIContraction &other ):
      GTODirectERIContraction(other, 0) {}
    GTODirectERIContraction( GTODirectERIContraction &&other ):
      GTODirectERIContraction(std::move(other), 0) {}

    /**
     *  \brief Perform various tensor contractions of the ERI tensor
     *  directly. Wraps other helper functions and provides
     *  loop structure
     *
     *  Currently supports
     *    - Coulomb-type (34,12) contractions
     *    - Exchange-type (23,12) contractions
     *
     *  Works with both real and complex matricies
     *
     *  \param [in/out] list Contains information pertinent to the
     *    matricies to be contracted with. See TwoBodyContraction
     *    for details
     */
    virtual void twoBodyContract(
        MPI_Comm c,
        const bool screen,
        std::vector<TwoBodyContraction<MatsT>> &list,
        EMPerturbation &pert) const {

        directScaffoldNew(c, screen, list);
    }

    void directScaffoldNew(
        MPI_Comm,
        const bool,
        std::vector<TwoBodyContraction<MatsT>>&) const;

    void directScaffold(
        MPI_Comm,
        const bool,
        std::vector<TwoBodyContraction<MatsT>>&,
        EMPerturbation&) const;

    virtual ~GTODirectERIContraction() {}

  }; // class GTODirectERIContraction

}; // namespace ChronusQ
