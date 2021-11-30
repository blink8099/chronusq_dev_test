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

      double totalMacroEnergy;

     private:
     public:

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

       // Deallocation (see include/singleslater/neo_singleslater/impl.hpp for docs)
       void alloc();
       void dealloc();

       Particle mainParticle() const{ return this->particle; }

       // ep Coulomb
       std::shared_ptr<SquareMatrix<MatsT>> epJMatrix; ///< electron-proton coulomb matrix

       // EPAI contractions 
       std::shared_ptr<TPIContractions<MatsT,IntsT>> EPAI; 

       // Compute the CH
       void formCoreH(EMPerturbation&, bool); 

       // Form initial guess orbitals
       void formGuess();
       
       // Form a fock matrix
       virtual void formFock(EMPerturbation &, bool increment = false, double xHFX = 1.);

       // Compute Energy
       //virtual void computeEnergy();

       // Initialize SCF procedure
       void SCFInit();

       // Perform an NEO-SCF procedure 
       void SCF(EMPerturbation &);

       // Print Macro SCF information
       void printSCFMacroProg(std::ostream &, bool);

       // Save the current state in NEO-SCF
       void saveCurrentState();

       // Finalizes the SCF procedure
       void SCFFin();

       // Evaluate the NEO-SCF convergence
       bool evalMacroConver(EMPerturbation &);

       // Compute properties of the overall electron-proton system
       virtual void computeTotalProperties(EMPerturbation &);

       // Compute multipole information
       void removeNucMultipoleContrib(); 
  }; // class NEOSingleSlater

} // namespace ChronusQ

// Include headers for specifications of NEOSingleSlater
#include <singleslater/neo_singleslater/neo_hartreefock.hpp> // NEO-KS specification
#include <singleslater/neo_singleslater/neo_kohnsham.hpp> // NEO-KS specification
