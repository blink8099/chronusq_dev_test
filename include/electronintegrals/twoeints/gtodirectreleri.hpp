/*
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *
 *  Copyright (C) 2014-2020 Li Research Group (University of WashinGIAOn)
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

#include <electronintegrals/twoeints/gtodirecteri.hpp>

namespace ChronusQ {

  template <typename MatsT, typename IntsT>
  class GTODirectRelERIContraction : public GTODirectERIContraction<MatsT,IntsT> {

  public:

    IntsT *ERI4DCB = nullptr; // Storage for 3-index DCB ERI intermediates

    // Constructors

    GTODirectRelERIContraction() = delete;
    GTODirectRelERIContraction(TwoEInts<IntsT> &eri):
      GTODirectERIContraction<MatsT,IntsT>(eri) {

      if (typeid(eri) != typeid(DirectERI<IntsT>))
        CErr("GTODirectRelERIContraction expect a DirectERI<IntsT> reference.");

    }

    GTODirectRelERIContraction( const GTODirectRelERIContraction &other ):
      GTODirectRelERIContraction(other.ints_) {}
    GTODirectRelERIContraction( GTODirectRelERIContraction &&other ):
      GTODirectRelERIContraction(other.ints_) {}

    // Computation interfaces
    virtual void twoBodyContract(
        MPI_Comm comm,
        const bool screen,
        std::vector<TwoBodyContraction<MatsT>> &list,
        EMPerturbation&) const {
      directScaffoldLibcint(comm, screen, list);
     // directScaffold(comm, screen, list);
//      twoBodyContract3Index(comm, list);
    }

    void directScaffold(
        MPI_Comm,
        const bool,
        std::vector<TwoBodyContraction<MatsT>>&) const;

    void directScaffoldLibcint(
        MPI_Comm,
        const bool,
        std::vector<TwoBodyContraction<MatsT>>&) const;

    void twoBodyContract3Index(
        MPI_Comm,
        std::vector<TwoBodyContraction<MatsT>>&) const;

    void JContract3Index(
        MPI_Comm,
        TwoBodyContraction<MatsT>&) const;

    void KContract3Index(
        MPI_Comm,
        TwoBodyContraction<MatsT>&) const;

    void computeERI3Index(size_t); // Evaluate Spin-Own-Orbit ERIs in the CGTO basis

    virtual ~GTODirectRelERIContraction() {}

  }; // class GTODirectRelERIContraction

}; // namespace ChronusQ
