/* 
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *  
 *  Copyright (C) 2014-2017 Li Research Group (University of Washington)
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
#ifndef __INCLUDED_QUANTUM_HPP__
#define __INCLUDED_QUANTUM_HPP__

#include <chronusq_sys.hpp>
#include <quantum/base.hpp>

// Debug print (triggers WaveFunction, etc)
//#define _QuantumDebug

namespace ChronusQ {

  /**
   *  \brief The Quantum class. The typed abstract interface for all classes 
   *  which can define a 1PDM and compute 1 and 2 body properties.
   *
   *  Adds knowledge of storage type and general property evaluation schemes to 
   *  QuantumBase.
   *
   */
  template <typename T>
  class Quantum : virtual public QuantumBase {
  protected:

    // Useful typedefs
    typedef T*                   oper_t;
    typedef std::vector<oper_t>  oper_t_coll;
    

  private:

    // Helper functions for the automatic evaluation of properties
    // see include/quantum/properties.hpp for documentation

    template <typename RetTyp, DENSITY_TYPE DenTyp, typename Op>
    RetTyp OperatorSpinCombine(const Op&);

  public:


    // 1PDM storage

    oper_t_coll onePDM;  ///< 1PDM array (Scalar + Magnetization)

    // Constructors
      
    // Disable default constructor
    Quantum() = delete;

    /**
     *  Quantum Constructor. Constructs a Quantum object.
     *
     *  \param [in] mem   CQMemManager to handle to allocation of densities
     *  \param [in] _nC   Number of spin components (1 and 2 are supported)
     *  \param [in] _iCS  Whether or not system is closed shell
     *                    (only used when _nC == 1)
     *  \param [in] N     Dimension of the density matricies to be allocated
     */ 
    Quantum(CQMemManager &mem, size_t _nC = 1, bool _iCS = true, 
      size_t N = 0, bool doAlloc = true): QuantumBase(mem,_nC,_iCS) {

        // Allocate densities
        if( N != 0 and doAlloc ) alloc(N);

    }; // Quantum::Quantum
    
       
    // See include/quantum/impl.hpp for documentation 
    // on the following constructors

    // Different type
    template <typename U> Quantum(const Quantum<U> &, int dummy = 0);
    template <typename U> Quantum(Quantum<U> &&     , int dummy = 0);

    // Same type
    Quantum(const Quantum &);
    Quantum(Quantum &&);     

    /**
     *  Deconstructor
     */ 
    ~Quantum(){ dealloc(); }

    // Member functions

    // Deallocation (see include/quantum/impl.hpp for docs)
    void alloc(size_t);
    void dealloc();


    // Public interfaces for property evaluation
      
    /**
     *  \brief Computes a 1-body property through a trace with the
     *  proper components of the 1PDM.
     *
     *  \param [template] DenTyp Which spin component of the 1PDM to trace with
     *  \param [in]       op     Square matrix to trace with 1PDM
     *  \returns          Trace of op with the DenTyp 1PDM (cast to type RetTyp)
     */ 
    template <typename RetTyp, DENSITY_TYPE DenTyp, typename Op>
    inline RetTyp computeOBProperty(const Op &op) {
      return OperatorSpinCombine<RetTyp,DenTyp>(op);
    }; // Quantum<T>::computeOBProperty (single operator)

    /**
     *  \brief Computes set of a 1-body properties through a traces with the
     *  proper components of the 1PDM.
     *
     *  \param [template] DenTyp Which spin component of the 1PDM to trace with
     *  \param [in]       opv    List of qquare matrices to trace with 1PDM
     *  \returns          List of traces of opv with the DenTyp 1PDM (cast to type RetTyp)
     */ 
    template <typename RetTyp, DENSITY_TYPE DenTyp, typename Op>
    inline std::vector<RetTyp> computeOBProperty(const std::vector<Op> &opv) {
      std::vector<RetTyp> results;
      for(auto &op : opv) 
        results.emplace_back(computeOBProperty<RetTyp,DenTyp>(op));
      return results;
    }; // Quantum<T>::computeOBProperty (many operators)

    // Print functions
    void print1PDM(std::ostream&);

  }; // class Quantum

}; // namespace ChronusQ

#endif
