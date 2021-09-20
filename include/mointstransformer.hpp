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

#include <fields.hpp>
#include <singleslater.hpp>
#include <singleslater/neo_singleslater.hpp>
#include <particleintegrals/twopints/incoreritpi.hpp>

#include <type_traits>

namespace ChronusQ {

  /**
   *  The operator types to evaluate integrals.
   */
  enum ERI_TRANSFORMATION_ALG {
      SSFOCK_N6 = 0, // hack thru ss.formfock
      INCORE_N5 = 1 //NYI
  };
  
  /**
   *  \brief Templated class to handle AO to MO integral transformation
   *  for both one-body and two-body terms 
   */

  template<typename MatsT, typename IntsT>
  class MOIntsTransformer {

  protected:

    CQMemManager &memManager_; ///< CQMemManager to allocate matricies
    ERI_TRANSFORMATION_ALG ERITransAlg_;
    SingleSlater<MatsT,IntsT> & ss_;

    // variables for moints type
    std::vector<std::set<char>> symbol_sets_;
    std::vector<std::pair<size_t, size_t>> mo_ranges_;
    
  public:

    // Constructor
    MOIntsTransformer() = delete;
    MOIntsTransformer( const MOIntsTransformer & ) = default;
    MOIntsTransformer( MOIntsTransformer && ) = default;
    MOIntsTransformer( CQMemManager &mem, SingleSlater<MatsT,IntsT> & ss,
      ERI_TRANSFORMATION_ALG alg = SSFOCK_N6):
        memManager_(mem), ss_(ss), ERITransAlg_(alg) {
      
        if (alg == INCORE_N5) {
          CErr("INCORE N5 NYI !");   
        }
        
        // set default MO ranges
        setMORanges();
    };
    
    // Methods to parse types of integral indices
    void resetMORanges() {
      symbol_sets_.clear();
      mo_ranges_.clear();
    };
    void addMORanges(const std::set<char> &, const std::pair<size_t, size_t> &);
    void setMORanges(size_t nFrozenCore = 0, size_t nFrozenVirt = 0);  
    // void setMORanges(MOSpacePartition); TODO: implement for CAS type   
    std::vector<std::pair<size_t,size_t>> parseMOType(const std::string &);

    // Methods to transform HCore 
    void transformHCore(MatsT * MOHCore, const std::string & moType = "pq");
    void subsetTransformHCore(const std::vector<std::pair<size_t,size_t>> &, MatsT*);
    
    // Methods to transform ERI 
    void transformERI(EMPerturbation & pert, MatsT* MOERI, const std::string & moType = "pqrs");
    void subsetTransformERISSFockN6(EMPerturbation &, const std::vector<std::pair<size_t,size_t>> &, MatsT*);
    void subsetTransformERIInCoreN5(const std::vector<std::pair<size_t,size_t>> &, MatsT*);

    virtual ~MOIntsTransformer() {};

  }; // class MOIntsTransformer

}; // namespace ChronusQ
