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
#include <util/mpi.hpp>

namespace ChronusQ {

  enum TWOBODY_CONTRACTION_TYPE {
    COULOMB, ///< (mn | kl) X(lk)
    EXCHANGE,///< (mn | kl) X(nk)
    PAIR     ///< (mn | kl) X(nl)
  }; ///< 2-Body Tensor Contraction Specification


  // ERI transpose type
  enum INTEGRAL_TRANSPOSE {
    TRANS_NONE,
    TRANS_MNKL,
    TRANS_KL,
    TRANS_MN
  };


  /**
   *  The TwoBodyContraction struct. Stores information
   *  pertinant for a two body operator contraction with
   *  a one body (2 index) operator. z.B. The density matrix.
   */
  template <typename T>
  struct TwoBodyContraction {


    T*  X;  ///< 1-Body (2 index) operator to contraction
    T*  AX; ///< 1-Body (2 index) storage for the contraction

    bool HER; ///< Whether or not X is hermetian

    TWOBODY_CONTRACTION_TYPE contType;

    double* ERI4 = nullptr;

    INTEGRAL_TRANSPOSE intTrans;


  }; // struct TwoBodyContraction


  /**
   *  \brief Templated class to handle the evaluation and storage of
   *  electron-electron repulsion integral tensors in a finite basis
   *  set.
   *
   *  Templated over storage type (IntsT) to allow for a seamless
   *  interface to both real- and complex-valued basis sets
   *  (e.g., GTO and GIAO)
   */
  template <typename IntsT>
  class TwoEInts : public ElectronIntegrals {

  public:

    // Constructors

    TwoEInts() = delete;
    TwoEInts( const TwoEInts & ) = default;
    TwoEInts( TwoEInts && ) = default;

    TwoEInts(CQMemManager &mem, size_t nb):
        ElectronIntegrals(mem, nb) {}

    template <typename IntsU>
    TwoEInts( const TwoEInts<IntsU> &other, int = 0 ):
        TwoEInts(other.memManager(), other.nBasis()) {
      if (std::is_same<IntsU, dcomplex>::value
          and std::is_same<IntsT, double>::value)
        CErr("Cannot create a Real TwoEInts from a Complex one.");
    }

    // Single element interfaces
    virtual IntsT operator()(size_t, size_t, size_t, size_t) const = 0;
    virtual IntsT operator()(size_t, size_t) const = 0;

    //virtual TensorContraction ERITensor();

    virtual ~TwoEInts() {}

  }; // class ERInts

  /**
   *  \brief Templated class to define the interface to perform
   *  transformations and contractions of ERInts. Handles the
   *  contraction of 2-body (3,4 index) integrals with
   *  1-body (2 index) operators.
   *
   *  Templated over matrix type (MatsT) to allow for a seamless
   *  interface to both real- and complex-valued coefficients
   *  and density.
   */
  template <typename MatsT, typename IntsT>
  class ERIContractions {

    template <typename MatsU, typename IntsU>
    friend class ERIContractions;

  protected:
    TwoEInts<IntsT> &ints_;

  public:

    // Constructors

    ERIContractions() = delete;
    ERIContractions(TwoEInts<IntsT> &eri): ints_(eri) {}
    template <typename MatsU>
    ERIContractions( const ERIContractions<MatsU,IntsT> &other, int dummy = 0 ):
      ERIContractions(other.ints_) {}
    template <typename MatsU>
    ERIContractions( ERIContractions<MatsU,IntsT> &&other, int dummy = 0 ):
      ERIContractions(other.ints_) {}

    ERIContractions( const ERIContractions &other ):
      ERIContractions(other, 0) {}
    ERIContractions( ERIContractions &&other ):
      ERIContractions(std::move(other), 0) {}

    TwoEInts<IntsT>& ints() { return ints_; }
    const TwoEInts<IntsT>& ints() const { return ints_; }

    // Computation interfaces

    /**
     *  Contract the two body potential with one body (2 index) operators.
     *
     *  Smartly determines whether to do the contraction directly, incore
     *  or using density fitting depending on context
     *
     *  \param [in/ont] contList List of one body operators for contraction.
     */
    virtual void twoBodyContract(
        MPI_Comm,
        const bool,
        std::vector<TwoBodyContraction<MatsT>>&,
        EMPerturbation&) const = 0;

    inline void twoBodyContract(
        MPI_Comm comm,
        const bool screen,
        std::vector<TwoBodyContraction<MatsT>> &contList) const {
      EMPerturbation pert;
      twoBodyContract(comm,screen,contList,pert);
    }

    inline void twoBodyContract(
        MPI_Comm comm,
        std::vector<TwoBodyContraction<MatsT>> &contList,
        EMPerturbation &pert) const {
      twoBodyContract(comm,true,contList,pert);
    }

    inline void twoBodyContract(
        MPI_Comm comm,
        std::vector<TwoBodyContraction<MatsT>> &contList) const {
      twoBodyContract(comm,true,contList);
    }

    // Destructor
    virtual ~ERIContractions() {}

    // Pointer convertor
    template <typename MatsU>
    static std::shared_ptr<ERIContractions<MatsU,IntsT>>
    convert(const std::shared_ptr<ERIContractions<MatsT,IntsT>>&);

  }; // class ERIContractions

}; // namespace ChronusQ
