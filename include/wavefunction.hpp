/* 
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *  
 *  Copyright (C) 2014-2019 Li Research Group (University of Washington)
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
#ifndef __INCLUDED_WAVEFUNCTION_HPP__
#define __INCLUDED_WAVEFUNCTION_HPP__

#include <chronusq_sys.hpp>
#include <quantum.hpp>
#include <wavefunction/base.hpp>

// Debug print triggered by Quantum
  
#ifdef _QuantumDebug
  #define _WaveFunctionDebug
#endif

namespace ChronusQ {


  /**
   *  \brief The WaveFunction class. The typed abstract interface for
   *  all classes which admit a well defined wave function (HF, KS, etc).
   *
   *  Adds knowledge of storage type to WaveFunctionBase
   *
   *  Specializes the Quantum class of the same type; 
   */
  template <typename MatsT, typename IntsT>
  class WaveFunction : virtual public WaveFunctionBase, public Quantum<MatsT> {
  protected:

    // Useful typedefs
    typedef MatsT*               oper_t;
    typedef std::vector<oper_t>  oper_t_coll;

  public:
    
    AOIntegrals<IntsT> &aoints; ///< AOIntegrals for the evaluation of GTO integrals

    // Operator storage
    oper_t  mo1;  ///< Full (nC > 1) / ALPHA (nC == 1) MO coefficient matrix
    oper_t  mo2;  ///< BETA (nC == 1) MO coefficient matrix
    double* eps1; ///< Full (nC > 1) / ALPHA (nC == 1) Fock eigenvalues
    double* eps2; ///< BETA (nC == 1) Fock eigenvalues

    // Constructors

    // Disable default constructor
    WaveFunction() = delete;

    /**
     *  WaveFunction Constructor. Constructs a WaveFunction object
     *
     *  \param [in] aoi  AOIntegrals object (which handels the BasisSet, etc)
     *  \param [in] _nC  Number of spin components (1 and 2 are supported)
     *  \param [in] iCS  Whether or not to treat as closed shell
     */ 
    WaveFunction(MPI_Comm c, AOIntegrals<IntsT> &aoi, size_t _nC, bool iCS) :
      QuantumBase(c, aoi.memManager(),_nC,iCS),
      WaveFunctionBase(c, aoi.memManager(),_nC,iCS),
      Quantum<MatsT>(c, aoi.memManager(),_nC,iCS,aoi.basisSet().nBasis), 
      mo1(nullptr), mo2(nullptr), eps1(nullptr), eps2(nullptr), aoints(aoi) {

      // Compute meta data

      this->nO = this->aoints.molecule().nTotalE;
      this->nV = 2*aoints.basisSet().nBasis - nO;

      if( this->iCS ) {
        this->nOA = this->nO / 2; this->nOB = this->nO / 2;
        this->nVA = this->nV / 2; this->nVB = this->nV / 2;
      } else {
        size_t nSingleE = aoints.molecule().multip - 1;
        this->nOB = (this->nO - nSingleE) / 2;
        this->nOA = this->nOB + nSingleE;
        this->nVA = aoints.basisSet().nBasis - this->nOA;
        this->nVB = aoints.basisSet().nBasis - this->nOB;
      }

      // Allocate internal memory
      if(aoints.basisSet().nBasis != 0) alloc();

    };

    // See include/wavefunction/impl.hpp for documentation 
    // on the following constructors

    // Different type
    template <typename MatsU> 
      WaveFunction(const WaveFunction<MatsU,IntsT> &, int dummy = 0);
    template <typename MatsU> 
      WaveFunction(WaveFunction<MatsU,IntsT> &&     , int dummy = 0);

    // Same type
    WaveFunction(const WaveFunction &);
    WaveFunction(WaveFunction &&);     
    

    /**
     *  Deconstructor
     */ 
    ~WaveFunction(){ dealloc(); }


    // Member functions

    // Deallocation (see include/wavefunction/impl.hpp for docs)
    void alloc();
    void dealloc();

    // Print Functions
    void printMO(std::ostream&) ;
    void printEPS(std::ostream&);

  }; // class WaveFunction

}; // namespace ChronusQ

#endif
