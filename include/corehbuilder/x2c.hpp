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

#include <corehbuilder.hpp>
#include <molecule.hpp>
#include <memmanager.hpp>
#include <basisset.hpp>
#include <fields.hpp>
#include <integrals.hpp>

namespace ChronusQ {

  template <typename MatsT, typename IntsT>
  class AtomicX2C;

  /**
   *  \brief The X2C class. A class to compute X2C Core Hamiltonian.
   *  Stores intermediate matrices.
   */
  template <typename MatsT, typename IntsT>
  class X2C : public CoreHBuilder<MatsT,IntsT> {

    template <typename MatsU, typename IntsU>
    friend class X2C;
    template <typename MatsU, typename IntsU>
    friend class AtomicX2C;

  protected:

    CQMemManager    &memManager_;        ///< CQMemManager to allocate matricies
    Molecule         molecule_;          ///< Molecule object for nuclear potential
    BasisSet         basisSet_;          ///< BasisSet for original basis defintion
    BasisSet         uncontractedBasis_; ///< BasisSet for uncontracted basis defintion
    Integrals<IntsT> uncontractedInts_;  ///< AOIntegrals for uncontracted basis
    size_t           nPrimUse_;          ///< Number of primitives used in p space

  public:

    // Operator storage
    IntsT*  mapPrim2Cont = nullptr;
    std::shared_ptr<SquareMatrix<MatsT>> W  = nullptr; ///< W = (\sigma p) V (\sigma p)
    IntsT*  UK = nullptr; ///< K transformation between p- and R-space
    double* p  = nullptr; ///< p momentum eigens
    MatsT*  X  = nullptr; ///< X = S * L^-1
    MatsT*  Y  = nullptr; ///< Y = sqrt(1 + X**H * X)
    MatsT*  UL = nullptr; ///< Picture change matrix of large component
    MatsT*  US = nullptr; ///< Picture change matrix of small component


    // Constructors

    // Disable default constructor
    X2C() = delete;

    // Default copy and move constructors
    X2C(const X2C<MatsT,IntsT> &);
    X2C(X2C<MatsT,IntsT> &&);

    /**
     * \brief Constructor
     *
     *  \param [in] aoints             Reference to the global AOIntegrals
     *  \param [in] memManager         Memory manager for matrix allocation
     *  \param [in] mol                Molecule object for molecular specification
     *  \param [in] basis              The GTO basis for integral evaluation
     *  \param [in] hamiltonianOptions Flags for AO integrals evaluation
     */
    X2C(Integrals<IntsT> &aoints, CQMemManager &mem,
        const Molecule &mol, const BasisSet &basis, HamiltonianOptions hamiltonianOptions) :
      CoreHBuilder<MatsT,IntsT>(aoints, hamiltonianOptions),
      memManager_(mem),molecule_(mol), basisSet_(basis),
      uncontractedBasis_(basisSet_.uncontractBasis()) {}

    // Different type
    template <typename MatsU>
    X2C(const X2C<MatsU,IntsT> &other, int dummy = 0);
    template <typename MatsU>
    X2C(X2C<MatsU,IntsT> &&     other, int dummy = 0);

    /**
     *  Destructor.
     *
     *  Destructs a X2C object
     */
    virtual ~X2C() { dealloc(); }


    // Public Member functions

    // Deallocation (see include/x2c/impl.hpp for docs)
    virtual void dealloc();

    // Compute core Hamitlonian
    virtual void computeCoreH(EMPerturbation&,
        std::shared_ptr<PauliSpinorSquareMatrices<MatsT>>);
    virtual void computeX2C(EMPerturbation&,
        std::shared_ptr<PauliSpinorSquareMatrices<MatsT>>);
    virtual void computeU();
    virtual void computeX2C_UDU(EMPerturbation&,
        std::shared_ptr<PauliSpinorSquareMatrices<MatsT>>);
    virtual void computeX2C_corr(EMPerturbation&,
        std::shared_ptr<PauliSpinorSquareMatrices<MatsT>>);
    void BoettgerScale(std::shared_ptr<PauliSpinorSquareMatrices<MatsT>>);

    // Compute the gradient
    virtual std::vector<double> getGrad(EMPerturbation&, SingleSlater<MatsT,IntsT>&) {
      CErr("X2C CoreH gradient NYI",std::cout);
    }

  };

}; // namespace ChronusQ
