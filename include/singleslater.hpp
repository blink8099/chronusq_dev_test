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
#include <wavefunction.hpp>
#include <singleslater/base.hpp>
#include <electronintegrals/twoeints.hpp>
#include <matrix.hpp>

// Debug print triggered by Wavefunction
  
#ifdef _WaveFunctionDebug
  #define _SingleSlaterDebug
#endif

namespace ChronusQ {

  // Declaration of CoreH and Fock builders.
  template <typename MatsT, typename IntsT>
  class CoreHBuilder;
  template <typename MatsT, typename IntsT>
  class FockBuilder;

  /**
   *  \brief The SingleSlater class. The typed abstract interface for all
   *  classes for which the wave function is described by a single slater
   *  determinant (HF, KS, PHF, etc).
   *
   *  Adds knowledge of storage type to SingleSlaterBase
   *
   *  Specializes the WaveFunction class of the same type
   */ 
  template <typename MatsT, typename IntsT>
  class SingleSlater : public SingleSlaterBase, public WaveFunction<MatsT,IntsT> {

    template <typename MatsU, typename IntsU>
    friend class SingleSlater;

  protected:

    // Useful typedefs
    typedef MatsT*                    oper_t;
    typedef std::vector<oper_t>       oper_t_coll;
    typedef std::vector<oper_t_coll>  oper_t_coll2;

    BasisSet &basisSet_; ///< BasisSet for the GTO basis defintion

  private:
  public:

    typedef MatsT value_type;
    typedef IntsT ints_type;

    //ORTHO_TYPE            orthoType; ///< Orthogonalization scheme

    // Operator storage

    // AO Fock Matrix
    std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> fockMatrix; ///< List of populated AO Fock matricies
    std::vector<SquareMatrix<MatsT>> fockMO;     ///< Fock matrix in the MO basis

    // Orthonormal Fock
    std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> fockMatrixOrtho; ///< List of populated orthonormal Fock matricies

    // Coulomb (J[D])
    std::shared_ptr<SquareMatrix<MatsT>> coulombMatrix; ///< scalar Coulomb Matrix

    // Exchange (K[D])
    std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> exchangeMatrix; ///< List of populated exact (HF) exchange matricies

    // Two-electron Hamiltonian (G[D])
    std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> twoeH; ///< List of populated HF perturbation tensors

    // Orthonormal density
    std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> onePDMOrtho; ///< List of populated orthonormal 1PDM matricies


    // Current / change in state information (for use with SCF)
    std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> curOnePDM; ///< List of the current 1PDMs
    std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> deltaOnePDM; ///< List of the changes in the 1PDMs

    // Stores the previous Fock matrix to use for damping
    std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> prevFock; ///< AO Fock from the previous SCF iteration

    // Stores the previous matrices to use for DIIS 
    std::vector<PauliSpinorSquareMatrices<MatsT>> diisFock; ///< List of AO Fock matrices for DIIS extrap
    std::vector<PauliSpinorSquareMatrices<MatsT>> diisOnePDM; ///< List of AO Density matrices for DIIS extrap
    std::vector<PauliSpinorSquareMatrices<MatsT>> diisError; ///< List of orthonormal [F,D] for DIIS extrap

    // 1-e integrals
    ///< ortho[0] : Orthogonalization matrix which S -> I
    ///< ortho[1] : Inverse of ortho[0]
    std::vector<SquareMatrix<MatsT>> ortho;

    std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> coreH; ///< Core Hamiltonian (scalar and magnetization)
    std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> coreHPerturbed; ///< Perturbed Core Hamiltonian (scalar and magnetization)

    // CoreH and Fock builders
    std::shared_ptr<ERIContractions<MatsT,IntsT>> ERI; ///< ERIContractions
    std::shared_ptr<CoreHBuilder<MatsT,IntsT>> coreHBuilder; ///< Builder for CoreH
    std::shared_ptr<FockBuilder<MatsT,IntsT>> fockBuilder; ///< Builder for Fock

    // Method specific propery storage
    std::vector<double> mullikenCharges;
    std::vector<double> lowdinCharges;



    // Constructors
      
    /**
     *  SingleSlater Constructor. Constructs a SingleSlater object
     *
     *  \param [in] aoi  AOIntegrals object (which handels the BasisSet, etc)
     *  \param [in] args Parameter pack for the remaining parameters of the
     *                   WaveFunction constructor. See include/wavefunction.hpp
     *                   for details. 
     */ 
    template <typename... Args>
    SingleSlater(MPI_Comm c, CQMemManager &mem, Molecule &mol, BasisSet &basis,
                 Integrals<IntsT> &aoi, Args... args) :
      SingleSlaterBase(c,mem,args...), WaveFunctionBase(c,mem,args...),
      QuantumBase(c,mem,args...),
      WaveFunction<MatsT,IntsT>(c,mem,mol,basis.nBasis,aoi,args...),
      basisSet_(basis)
      //, coreType(NON_RELATIVISTIC), orthoType(LOWDIN)
    {
      // Allocate SingleSlater Object
      alloc();

      // Determine Real/Complex part of method string
      if(std::is_same<MatsT,double>::value) {
        refLongName_  = "Real ";
        refShortName_ = "R-";
      } else {
        refLongName_  = "Complex ";
        refShortName_ = "C-";
      
      }

    }; // SingleSlater constructor

    // See include/singleslater/impl.hpp for documentation 
    // on the following constructors

    // Different type
    template <typename MatsU> 
      SingleSlater(const SingleSlater<MatsU,IntsT> &, int dummy = 0);
    template <typename MatsU> 
      SingleSlater(SingleSlater<MatsU,IntsT> &&     , int dummy = 0);

    // Same type
    SingleSlater(const SingleSlater<MatsT,IntsT> &);
    SingleSlater(SingleSlater<MatsT,IntsT> &&);     

    /**
     *  Destructor.
     *
     *  Destructs a SingleSlater object
     */ 
    ~SingleSlater() { dealloc(); }



    // Public Member functions

    BasisSet& basisSet() { return basisSet_; }
      
      

    // Deallocation (see include/singleslater/impl.hpp for docs)
    void alloc();
    void dealloc();


    // Declarations from QuantumBase 
    // (see include/singleslater/quantum.hpp for docs)
    void formDensity();
    void computeEnergy();
    void computeMultipole(EMPerturbation &);
    void computeSpin();

    // Compute various core Hamitlonian
    void formCoreH(EMPerturbation&); // Compute the CH
//  void updateCoreH(EMPerturbation &);
    void computeOrtho();  // Evaluate orthonormalization transformations

    // Method specific properties
    void populationAnalysis();
    void methodSpecificProperties() {
      populationAnalysis();
    }





    // Form a fock matrix (see include/singleslater/fock.hpp for docs)
    virtual void formFock(EMPerturbation &, bool increment = false, double xHFX = 1.);

    // Form initial guess orbitals
    // see include/singleslater/guess.hpp for docs)
    void formGuess();
    void CoreGuess();
    void SADGuess();
    void RandomGuess();
    void ReadGuessMO();
    void ReadGuess1PDM();
    void FchkGuessMO();

    // Fchk-related functions 
    std::vector<int> fchkToCQMO();
    void reorderAngMO(std::vector<int> sl, MatsT* tmo, int sp);
    void reorderSpinMO();

    // SCF procedural functions (see include/singleslater/scf.hpp for docs)

    // Transformation functions to and from the orthonormal basis
    void ao2orthoFock();
    void ortho2aoDen();
    void ortho2aoMOs();

    // Evaluate convergence
    bool evalConver(EMPerturbation &);

    // Obtain new orbitals
    void getNewOrbitals(EMPerturbation &, bool frmFock = true);
    void ConventionalSCF(bool modF);
    void NewtonRaphsonSCF();
    virtual MatsT* getNRCoeffs() = 0;

    // Misc procedural
    void diagOrthoFock();
    void diagAOFock();
    void FDCommutator(PauliSpinorSquareMatrices<MatsT>&);
    virtual void saveCurrentState();
    virtual void formDelta();
    void orthoAOMO();
    void SCFInit();
    void SCFFin();

    // Stability and reopt
    virtual std::pair<double,MatsT*> getStab() = 0;
    bool checkStability();

    // Print functions
    void printFock(std::ostream& )    ;
    void print1PDMOrtho(std::ostream&);
    void printGD(std::ostream&)       ;
    void printJ(std::ostream&)        ;
    void printK(std::ostream&)        ;
    void printMiscProperties(std::ostream&);
    void printMOInfo(std::ostream&); 
    virtual void printFockTimings(std::ostream&);

    // SCF extrapolation functions (see include/singleslater/extrap.hpp for docs)
    void allocExtrapStorage();
    void deallocExtrapStorage();
    void modifyFock();
    void fockDamping();
    void scfDIIS(size_t);




    // MO Transformations
    void MOFOCK();

  }; // class SingleSlater

}; // namespace ChronusQ

// Include declaration of CoreHBuilder and FockBuilder
#include <corehbuilder.hpp>
#include <fockbuilder.hpp>

// Include headers for specializations of SingleSlater
#include <singleslater/hartreefock.hpp> // HF specialization
#include <singleslater/kohnsham.hpp>    // KS specialization

