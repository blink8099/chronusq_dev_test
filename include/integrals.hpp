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

#include <chronusq_sys.hpp>
#include <util/files.hpp>
#include <fields.hpp>
#include <particleintegrals/onepints.hpp>
#include <particleintegrals/twopints.hpp>
#include <particleintegrals/multipoleints.hpp>

namespace ChronusQ {

  enum ORTHO_TYPE {
    LOWDIN,
    CHOLESKY
  }; ///< Orthonormalization Scheme





  /**
   *  \brief Abstract Base class for AOIntegrals
   *
   *  Stores type independent members and interfaces for templated the
   *  AOIntegrals class
   *
   */
  struct IntegralsBase {
    
    SafeFile savFile; ///< Hard storage of integrals
    HamiltonianOptions options_;

    // Default copy and move ctors
    IntegralsBase( const IntegralsBase & ) = default;
    IntegralsBase( IntegralsBase && )      = default;
    IntegralsBase& operator=(const IntegralsBase&) = default;
    // Remove default ctor
    IntegralsBase() = default;

    // Interfaces
    virtual void computeAOOneP(CQMemManager &mem, Molecule &mol,
        BasisSet &basis, EMPerturbation&,
        const std::vector<std::pair<OPERATOR,size_t>>&,
        const HamiltonianOptions&) = 0;

    virtual void computeAOTwoE(BasisSet&, Molecule&, EMPerturbation&) = 0;
    virtual void computeAOTwoE(BasisSet&, BasisSet&, Molecule&, EMPerturbation&) = 0;

    // Print (see src/aointegrals/print.cxx for docs)
    template <typename G> 
      friend std::ostream & operator<<(std::ostream &, const IntegralsBase& );

    virtual ~IntegralsBase() {}

  };


  /**
   *  \brief Templated class to handle the evaluation and storage of 
   *  integral matrices representing quantum mechanical operators in
   *  a finite basis set.
   *
   *  Templated over storage type (IntsT) to allow for a seamless
   *  interface to both real- and complex-valued basis sets 
   *  (e.g., GTO and GIAO)
   *
   *  Real-valued arithmetics are kept in AOIntegrals
   */
  template <typename IntsT>
  class Integrals : public IntegralsBase {

  public:

    // 1-particle storage
    std::shared_ptr<OnePInts<IntsT>> overlap   = nullptr;   ///< Overlap matrix
    std::shared_ptr<OnePInts<IntsT>> kinetic   = nullptr;   ///< Kinetic matrix
    std::shared_ptr<OnePInts<IntsT>> potential = nullptr;   ///< Nuclear potential matrix

    std::shared_ptr<MultipoleInts<IntsT>> lenElectric = nullptr;
    std::shared_ptr<MultipoleInts<IntsT>> velElectric = nullptr;
    std::shared_ptr<MultipoleInts<IntsT>> magnetic = nullptr;

    // 2-particle storage
    std::shared_ptr<TwoPInts<IntsT>> TPI = nullptr;

    // miscellaneous storage
    std::map<std::string, std::shared_ptr<ParticleIntegrals>> misc;

    // Constructors
    Integrals() = default;

    Integrals(const Integrals &) = default; // Copy constructor
    Integrals(Integrals &&)      = default; // Move constructor
    Integrals& operator=(const Integrals&) = default;

    // Destructor.
    ~Integrals() {}

    // Integral evaluation
    // Evaluate the 1-particle ints (general)
    virtual void computeAOOneP(CQMemManager &mem, Molecule &mol,
        BasisSet &basis, EMPerturbation&,
        const std::vector<std::pair<OPERATOR,size_t>>&,
        const HamiltonianOptions&);

    virtual void computeAOTwoE(BasisSet& basis, Molecule& mol,
      EMPerturbation& emPert) {
      TPI->computeAOInts(basis, mol, emPert, ELECTRON_REPULSION,
                         options_);
    }

    virtual void computeAOTwoE(BasisSet& basis, BasisSet& basis2, 
                               Molecule& mol, EMPerturbation& emPert) {
      TPI->computeAOInts(basis, basis2, mol, emPert, EP_ATTRACTION,
                         options_);
    }

    template <typename MatsT>
    Integrals<typename std::conditional<
    (std::is_same<IntsT, dcomplex>::value or
     std::is_same<MatsT, dcomplex>::value),
    dcomplex, double>::type> transform(
        const std::vector<OPERATOR>&, const std::vector<std::string>&,
        char TRANS, const MatsT* T, int NT, int LDT) const;

  }; // class Integrals


}; // namespace ChronusQ
