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
   *  \brief Templated class to handle AO to MO integral transformation
   *  for both one-body and two-body terms 
   */

  template<typename MatsT, typename IntsT>
  class MOIntsTransformer {

  protected:

    CQMemManager &memManager_; ///< CQMemManager to allocate matricies
    TPI_TRANSFORMATION_ALG TPITransAlg_;
    SingleSlater<MatsT,IntsT> & ss_;
    std::shared_ptr<InCore4indexTPI<MatsT>> AOTPI_ = nullptr;

    // variables for moints type
    std::vector<std::set<char>> symbol_sets_;
    std::vector<std::pair<size_t, size_t>> mo_ranges_;
    
  public:

    // Constructor
    MOIntsTransformer() = delete;
    MOIntsTransformer( const MOIntsTransformer & ) = default;
    MOIntsTransformer( MOIntsTransformer && ) = default;
    MOIntsTransformer( CQMemManager &mem, SingleSlater<MatsT,IntsT> & ss,
      TPI_TRANSFORMATION_ALG alg = DIRECT_N6):
        memManager_(mem), ss_(ss), TPITransAlg_(alg) {
      
        if (ss.nC == 1) {
          CErr("1C MOInts NYI !");   
        } else if (alg == DIRECT_N5) {
          CErr("DIRECT N5 MOIntsTransformer NYI !");   
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
    
    // Methods to transform TPI 
    void transformTPI(EMPerturbation & pert, MatsT* MOTPI, 
      const std::string & moType = "pqrs", bool antiSymm = true);
    void subsetTransformTPISSFockN6(EMPerturbation &, 
      const std::vector<std::pair<size_t,size_t>> &, MatsT*, bool antiSymm = true);
    void cacheAOTPIInCore();
    void subsetTransformTPIInCoreN5(const std::vector<std::pair<size_t,size_t>> &, 
      MatsT*, bool antiSymm = true);

    virtual ~MOIntsTransformer() {};

  }; // class MOIntsTransformer

}; // namespace ChronusQ
