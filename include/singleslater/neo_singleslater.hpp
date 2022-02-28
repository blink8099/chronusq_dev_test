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
#include <singleslater.hpp>
#include <matrix.hpp>
//#include <singleslater/base.hpp>

namespace ChronusQ {

  /**
   *  \brief The NEO SingleSlater class
   *
   */
  template <typename MatsT, typename IntsT>
  class NEOSingleSlater : virtual public SingleSlater<MatsT,IntsT> {

    template <typename MatsU, typename IntsU>
    friend class NEOSingleSlater;
    
    protected:
      
      // pointer that points to the auxiliary class object 
      std::shared_ptr<NEOSingleSlater<MatsT,IntsT>> aux_neoss; 

     private:
     public:

      double totalMacroEnergy;

       // Constructors
       template <typename... Args>
       NEOSingleSlater(MPI_Comm c, CQMemManager &mem, 
                       Molecule &mol, BasisSet &basis, 
                       Integrals<IntsT> &aoi, Args... args) : 
         SingleSlater<MatsT,IntsT>(c,mem,mol,basis,aoi,args...)
       {
         
         // Allocate NEOSingleSlater Object
         alloc(); 
         
       }; // NEOSingleSater constructor

       // See include/singleslater/neo_singleslater/impl.hpp for documentation
       // on the following constructors

       // Different type
       template <typename MatsU>
         NEOSingleSlater(const NEOSingleSlater<MatsU,IntsT> &, int dummy = 0);
       template <typename MatsU>
         NEOSingleSlater(NEOSingleSlater<MatsU,IntsT> &&     , int dummy = 0);

       // Same type
       NEOSingleSlater(const NEOSingleSlater<MatsT,IntsT> &);
       NEOSingleSlater(NEOSingleSlater<MatsT,IntsT> &&);

       /**
        *  Destructor.
        *
        *  Destructs a NEOSingleSlater object
        */ 
       ~NEOSingleSlater() { dealloc(); }

       virtual void getAux(std::shared_ptr<NEOSingleSlater<MatsT,IntsT>> neo_ss) 
       { 
         aux_neoss = neo_ss; 
       }

       std::shared_ptr<NEOSingleSlater<MatsT,IntsT>> returnAux()
       {
         return aux_neoss;
       }

       // Deallocation (see include/singleslater/neo_singleslater/impl.hpp for docs)
       void alloc();
       void dealloc();

       Particle mainParticle() const{ return this->particle; }

       // ep Coulomb
       std::shared_ptr<SquareMatrix<MatsT>> epJMatrix; ///< electron-proton coulomb matrix

       // EPAI contractions 
       std::shared_ptr<TPIContractions<MatsT,IntsT>> EPAI; 

       // Compute the CH
       void formCoreH(EMPerturbation&); 

       // Form initial guess orbitals
       void formGuess(const SingleSlaterOptions&);
       
       // Form a fock matrix
       virtual void formFock(EMPerturbation &, bool increment = false, double xHFX = 1.);

       // SCF Functions
       inline double getTotalEnergy() { return this->totalMacroEnergy; };
       std::vector<std::shared_ptr<SquareMatrix<MatsT>>> getOnePDM();
       std::vector<std::shared_ptr<SquareMatrix<MatsT>>> getFock();
       std::vector<std::shared_ptr<Orthogonalization<MatsT>>> getOrtho();
       void formDensity(bool computeAuxDen = true);
       void printProperties();
       void formBothFock(EMPerturbation&, bool increment = false, double xHFX = 1.);
       void runModifyOrbitals(EMPerturbation&);
       std::vector<NRRotOptions> buildRotOpt();


       // Compute Energy
       //virtual void computeEnergy();

       // Save the current state in NEO-SCF
       void saveCurrentState();

       // Compute properties of the overall electron-proton system
       virtual void computeTotalProperties(EMPerturbation &);

       // Compute multipole information
       void removeNucMultipoleContrib(); 
  }; // class NEOSingleSlater

} // namespace ChronusQ

// Include headers for specifications of NEOSingleSlater
#include <singleslater/neo_singleslater/neo_hartreefock.hpp> // NEO-KS specification
#include <singleslater/neo_singleslater/neo_kohnsham.hpp> // NEO-KS specification
