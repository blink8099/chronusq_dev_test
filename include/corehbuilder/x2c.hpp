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
#include <aointegrals.hpp>

namespace ChronusQ {

  struct X2C_TYPE {
    bool atomic;
    bool isolateAtom;
    bool diagonalOnly;
  };

  /**
   *  \brief The X2C class. A class to compute X2C Core Hamiltonian.
   *  Stores intermediate matrices.
   */
  template <typename MatsT, typename IntsT>
  class X2C : public CoreHBuilder<MatsT,IntsT> {

    template <typename MatsU, typename IntsU>
    friend class X2C;

  protected:

    CQMemManager      &memManager_;        ///< CQMemManager to allocate matricies
    Molecule           molecule_;          ///< Molecule object for nuclear potential
    BasisSet           basisSet_;          ///< BasisSet for original basis defintion
    BasisSet           uncontractedBasis_; ///< BasisSet for uncontracted basis defintion
    AOIntegrals<IntsT> uncontractedInts_;  ///< AOIntegrals for uncontracted basis

  public:

    // Operator storage
    IntsT*  mapPrim2Cont = nullptr;
    MatsT*  W  = nullptr; ///< W = (\sigma p) V (\sigma p)
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
     *  \param [in] aoints     Reference to the global AOIntegrals
     *  \param [in] memManager Memory manager for matrix allocation
     *  \param [in] mol        Molecule object for molecular specification
     *  \param [in] basis      The GTO basis for integral evaluation
     */
    X2C(AOIntegrals<IntsT> &aoints, CQMemManager &mem,
        const Molecule &mol, const BasisSet &basis) :
      CoreHBuilder<MatsT,IntsT>(aoints, {true,true,true}),
      memManager_(mem),molecule_(mol), basisSet_(basis),
      uncontractedBasis_(basisSet_.uncontractBasis()),
      uncontractedInts_(memManager_,molecule_,uncontractedBasis_) {}

    /**
     * \brief Constructor
     *
     *  \param [in] aoints     Reference to the global AOIntegrals
     */
    X2C(AOIntegrals<IntsT> &aoints) :
      X2C(aoints, aoints.memManager(), aoints.molecule(), aoints.basisSet()) {}

    // Different type
    template <typename MatsU>
    X2C(const X2C<MatsU,IntsT> &other, int dummy = 0) :
      CoreHBuilder<MatsT,IntsT>(other), memManager_(other.memManager_),
      molecule_(other.molecule_), basisSet_(other.basisSet_),
      uncontractedBasis_(other.uncontractedBasis_),
      uncontractedInts_(other.uncontractedInts_) {
      CErr("X2C MatsT should always be dcomplex.",std::cout);
    }
    template <typename MatsU>
    X2C(X2C<MatsU,IntsT> &&     other, int dummy = 0) :
      CoreHBuilder<MatsT,IntsT>(other), memManager_(other.memManager_),
      molecule_(other.molecule_), basisSet_(other.basisSet_),
      uncontractedBasis_(other.uncontractedBasis_),
      uncontractedInts_(other.uncontractedInts_) {
      CErr("X2C MatsT should always be dcomplex.",std::cout);
    }

    /**
     *  Destructor.
     *
     *  Destructs a X2C object
     */
    virtual ~X2C() { dealloc(); }


    // Public Member functions

    // Deallocation (see include/x2c/impl.hpp for docs)
    void dealloc();

    // Compute core Hamitlonian
    void computeU();
    virtual void computeCoreH(EMPerturbation&, std::vector<MatsT*>&);

    // Compute the gradient
    virtual void getGrad() {
      CErr("X2C CoreH gradient NYI",std::cout);
    }

  protected:

    // Compute One-electron integrals
    virtual void computeAOOneE(EMPerturbation&);
    virtual void computeX2C(EMPerturbation&, std::vector<MatsT*>&);
    virtual void computeX2C_UDU(EMPerturbation&, std::vector<MatsT*>&);

  };

  template <typename T>
  void formW(size_t NP, dcomplex *W, size_t LDW, T* pVdotP, size_t LDD, T* pVxPZ,
    size_t LDZ, T* pVxPY, size_t LDY, T* pVxPX, size_t LDX);

}; // namespace ChronusQ
