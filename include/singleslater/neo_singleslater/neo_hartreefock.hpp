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
#include <singleslater/neo_singleslater.hpp>
#include <singleslater/hartreefock.hpp>

namespace ChronusQ {

  /**
   *  \brief The NEO-Hartree-Fock class.
   *
   *  Trivially specializes the NEOSingleSlater class for a NEO-Hartree-Fock description of
   *  the many-body wave function
   */
  template <typename MatsT, typename IntsT>
  class NEOHartreeFock : public NEOSingleSlater<MatsT,IntsT>, public HartreeFock<MatsT,IntsT> {

    template <typename MatsU, typename IntsU>
    friend class NEOHartreeFock;

    protected:
    
    // pointer that points to the auxiliary class object
    std::shared_ptr<NEOHartreeFock<MatsT,IntsT>> aux_neohf;

    public:
    
    // Trivially inherit ctors from NEOSingleSlater<T>
    template <typename... Args>
    NEOHartreeFock(MPI_Comm c, CQMemManager &mem, Molecule &mol, 
                   BasisSet &basis, Integrals<IntsT> &aoi, Args... args) :
      WaveFunctionBase(c,mem,args...),
      QuantumBase(c,mem,args...),
      NEOSingleSlater<MatsT,IntsT>(c,mem,mol,basis,aoi,args...),
      HartreeFock<MatsT,IntsT>(c,mem,mol,basis,aoi,args...),
      SingleSlater<MatsT,IntsT>(c,mem,mol,basis,aoi,args...) {}; 

    // Allow for reference name specification
    template <typename... Args>
    NEOHartreeFock(std::string rL, std::string rS, MPI_Comm c,
                   CQMemManager &mem, Molecule &mol, BasisSet &basis,
                   Integrals<IntsT> &aoi, Args... args) :
      WaveFunctionBase(c,mem,args...),
      QuantumBase(c,mem,args...),
      NEOSingleSlater<MatsT,IntsT>(c,mem,basis,aoi,args...),
      HartreeFock<MatsT,IntsT>(rL,rS,c,mem,mol,basis,aoi,args...), 
      SingleSlater<MatsT,IntsT>(c,mem,mol,basis,aoi,args...) { };  

    // Copy and Move ctors

    template <typename MatsU>
      NEOHartreeFock(const NEOHartreeFock<MatsU,IntsT> &other, int dummy = 0) :
        NEOSingleSlater<MatsT,IntsT>(dynamic_cast<const NEOSingleSlater<MatsU,IntsT>&>(other),dummy),
        HartreeFock<MatsT,IntsT>(dynamic_cast<const HartreeFock<MatsU,IntsT>&>(other),dummy),
        SingleSlater<MatsT,IntsT>(dynamic_cast<const SingleSlater<MatsU,IntsT>&>(other),dummy)
        //aux_neohf(std::make_shared<NEOHartreeFock<MatsT,IntsT>>(*other.aux_neohf))
        { };

    template <typename MatsU>
      NEOHartreeFock(NEOHartreeFock<MatsU,IntsT> &&other, int dummy = 0) :
        NEOSingleSlater<MatsT,IntsT>(dynamic_cast<NEOSingleSlater<MatsU,IntsT>&&>(std::move(other)),dummy),
        HartreeFock<MatsT,IntsT>(dynamic_cast<HartreeFock<MatsU,IntsT>&&>(std::move(other)),dummy),
        SingleSlater<MatsT,IntsT>(dynamic_cast<SingleSlater<MatsU,IntsT>&&>(std::move(other)),dummy)
        //aux_neohf(std::make_shared<NEOHartreeFock<MatsT,IntsT>>(std::move(*other.aux_neohf)))
        { };

    NEOHartreeFock(const NEOHartreeFock<MatsT,IntsT> &other) : 
      NEOHartreeFock(other,0) { };

    NEOHartreeFock(NEOHartreeFock<MatsT,IntsT> &&other) :
      NEOHartreeFock(std::move(other),0) { };

    void getAux(std::shared_ptr<NEOHartreeFock<MatsT,IntsT>> neo_hf) 
    { 
      aux_neohf = neo_hf; 
      NEOSingleSlater<MatsT,IntsT>::getAux(neo_hf);
    };

  }; // NEOHartreeFock class

}
