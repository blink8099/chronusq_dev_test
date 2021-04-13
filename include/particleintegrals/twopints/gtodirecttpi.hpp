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

#include <particleintegrals/twopints.hpp>

namespace ChronusQ {

  template <typename IntsT>
  class DirectTPI : public TwoPInts<IntsT> {

    template <typename IntsU>
    friend class DirectTPI;

  protected:
    BasisSet &basisSet_;
    BasisSet &basisSet2_;
    double threshSchwartz_ = 1e-12; ///< Schwartz screening threshold
    double* schwartz_ = nullptr;   ///< Schwartz bounds for the TPIs
    double* schwartz2_ = nullptr;  ///< second Schwartz bounds for the TPIs
    Molecule &molecule_;

  public:

    // Constructor
    DirectTPI() = delete;
    DirectTPI(CQMemManager &mem, BasisSet &basis, BasisSet &basis2, Molecule &mol, double threshSchwartz):
        TwoPInts<IntsT>(mem, basis.nBasis, basis2.nBasis), 
        basisSet_(basis), basisSet2_(basis2),molecule_(mol),
        threshSchwartz_(threshSchwartz) {}
    DirectTPI( const DirectTPI &other ):
        DirectTPI(other.memManager(), other.nBasis(), other.nBasis2(), other.threshSchwartz_) {
      std::copy_n(other.schwartz_, basisSet_.nShell*basisSet_.nShell, schwartz_);
      std::copy_n(other.schwartz2_, basisSet2_.nShell*basisSet2_.nShell, schwartz2_);
    }
    template <typename IntsU>
    DirectTPI( const DirectTPI<IntsU> &other, int = 0 ):
        DirectTPI(other.memManager_, other.basisSet_, other.basisSet2_, other.molecule_, other.threshSchwartz_) {
      if (other.schwartz_) {
        size_t NS = basisSet().nShell;
        schwartz_ = this->memManager().template malloc<double>(NS*NS);
        std::copy_n(other.schwartz_, NS*NS, schwartz_);
      }
      if (other.schwartz2_) {
        size_t NS = basisSet2().nShell;
        schwartz2_ = this->memManager().template malloc<double>(NS*NS);
        std::copy_n(other.schwartz2_, NS*NS, schwartz2_);
      }
    }
    DirectTPI( DirectTPI &&other ): TwoPInts<IntsT>(std::move(other)),
        basisSet_(other.basisSet_), basisSet2_(other.basisSet2_), 
        threshSchwartz_(other.threshSchwartz_),
        molecule_(other.molecule_),
        schwartz_(other.schwartz_), schwartz2_(other.schwartz2_) 
        { other.schwartz_ = nullptr; 
          other.schwartz2_ = nullptr;
        }

    BasisSet& basisSet() { return basisSet_; }
    BasisSet& basisSet2() { return basisSet2_; }
    Molecule& molecule() { return molecule_; }
    double threshSchwartz() const { return threshSchwartz_; }
    double*& schwartz()  { return schwartz_; }
    double*& schwartz2() { return schwartz2_; }

    // Single element interfaces
    virtual IntsT operator()(size_t p, size_t q, size_t r, size_t s) const {
      CErr("NYI");
      return 0;
    }
    virtual IntsT operator()(size_t pq, size_t rs) const {
      CErr("NYI");
      return 0;
    }

    // Computation interfaces
    virtual void computeAOInts(BasisSet&, Molecule&, EMPerturbation&,
        OPERATOR, const HamiltonianOptions&) {}

    virtual void computeAOInts(BasisSet&, BasisSet&, Molecule&, EMPerturbation&,
        OPERATOR, const HamiltonianOptions&) {}

    virtual void clear() {}

    void computeSchwartz();

    virtual void output(std::ostream &out, const std::string &s = "",
                        bool printFull = false) const {
      if (s == "")
        out << "  Two particle integral:" << std::endl;
      else
        out << "  TPI[" << s << "]:" << std::endl;
      out << "  " << std::setw(28) << "  Contraction Algorithm:";
      out << "DIRECT";
      out << std::endl;

      out << "    * Schwartz Screening Threshold = "
          << threshSchwartz() << std::endl;

      if (printFull) {
        CErr("Printing Full ERI tensor for Direct contraction NYI.", out);
      }
    }

    virtual ~DirectTPI() {
      if(schwartz_)  this->memManager().free(schwartz_);
      if(schwartz2_) this->memManager().free(schwartz2_);
    }

  }; // class DirectTPI

  template <typename MatsT, typename IntsT>
  class GTODirectTPIContraction : public TPIContractions<MatsT,IntsT> {

    template <typename MatsU, typename IntsU>
    friend class GTODirectTPIContraction;

  public:

    // Constructors

    GTODirectTPIContraction() = delete;
    GTODirectTPIContraction(TwoPInts<IntsT> &tpi):
      TPIContractions<MatsT,IntsT>(tpi) {

      if (typeid(tpi) != typeid(DirectTPI<IntsT>))
        CErr("GTODirectTPIContraction expect a DirectTPI reference.");

    }

    template <typename MatsU>
    GTODirectTPIContraction(
        const GTODirectTPIContraction<MatsU,IntsT> &other, int dummy = 0 ):
      GTODirectTPIContraction(other.ints_) {}
    template <typename MatsU>
    GTODirectTPIContraction(
        GTODirectTPIContraction<MatsU,IntsT> &&other, int dummy = 0 ):
      GTODirectTPIContraction(other.ints_) {}

    GTODirectTPIContraction( const GTODirectTPIContraction &other ):
      GTODirectTPIContraction(other, 0) {}
    GTODirectTPIContraction( GTODirectTPIContraction &&other ):
      GTODirectTPIContraction(std::move(other), 0) {}

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

    void direct4CScaffold(
        MPI_Comm,
        const bool,
        std::vector<TwoBodyContraction<MatsT>>&) const;

    void directScaffold(
        MPI_Comm,
        const bool,
        std::vector<TwoBodyContraction<MatsT>>&,
        EMPerturbation&) const;

    virtual ~GTODirectTPIContraction() {}

  }; // class GTODirectTPIContraction

}; // namespace ChronusQ
