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
#include <util/typedefs.hpp>
#include <atom.hpp>
#include <cerr.hpp>

#include <libint2/shell.h>

namespace ChronusQ {


  /**
   *  \brief The Molecule struct. Contains information pertinant to
   *  the overall molecular structure.
   */
  struct Molecule {

    size_t nAtoms;   ///< Number of atoms in the Molecule
    size_t multip;   ///< Spin multiplicity XXX: This implies <S^2>
    size_t nTotalE;  ///< Total number of electrons in the Molecule

    int  charge;   ///< Overall charge of the Molecule (atomic units)

    double   nucRepEnergy; ///< Nuclear-Nuclear repulsion energy
    
    cart_t    COM; ///< Center-of-mass of the Molecule
    cart_t    COC; ///< Center-of-charge of the Molecule
    cartmat_t MOI; ///< Moment of inertia of the Molecule

    dynmat_t nucRepForce; ///< Nuclear gradient contribution
    dynmat_t RIJ;         ///< Nuclear distance matrix

    std::vector<Atom> atoms; ///< The Atoms of which the Molecule consists


    std::vector<libint2::Shell> chargeDist;



    /**
     *  Default constructor
     *
     *  \param [in] C       Total Molecular charge (A.U.)
     *  \param [in] M       Spin Multiplicity XXX: This implies <S^2>
     *  \param [in] _atoms  Atoms from which to construct Molecule object
     */ 
    Molecule(const int C = 0, const int M = 0, std::vector<Atom> _atoms = {}) :
      charge(C), multip(M), nAtoms(_atoms.size()), atoms(std::move(_atoms)){ 
    
      if(nAtoms > 0) update();
    };

    /**
     *  Copy constructor.
     *
     *  Copies the contents of one Molecule object to another
     */ 
    Molecule(const Molecule &) = default;


    /**
     *  Move constructor.
     *
     *  Moves the contents of one Molecule object to another
     */ 
    Molecule(Molecule &&)      = default;


    /**
     *  Copy assignment operator
     *
     *  Assigns one Molecule object to another through a copy
     */ 
    Molecule& operator=(const Molecule&) = default;


    /**
     *  Move assignment operator
     *
     *  Assigns one Molecule object to another through a move (rvalue reference)
     */ 
    Molecule& operator=(Molecule&&)  = default;

    // Ouputs Molecule info, see src/molecule/molecule.cxx for documentation
    friend std::ostream& operator<<(std::ostream &, const Molecule&);

    

    /**
     *  \brief Sets the atoms array of the Molecule object. Recomputes
     *  all member data to fit with the new atoms array
     *
     *  \param [in] _atoms A collection of Atom object to construct a Molecule object
     */ 
    void setAtoms(std::vector<Atom> _atoms) {
      atoms = std::move(_atoms);
      nAtoms = atoms.size();
      update();
    }

    /**
     *  \brief Update Molecule member data
     *
     *  Populates or repopulates the member data for a Molecule
     *  object.
     */ 
    inline void update() {

      // Compute the total number of 
      nTotalE = std::accumulate(atoms.begin(),atoms.end(),-charge,
                  [&](int c, const Atom &a){ return a.atomicNumber + c; }
                );
      
      if(not ((nTotalE % 2) != 0 xor (multip % 2) != 0) or 
         multip > nTotalE + 1) {
        std::stringstream ss;
        ss << "Multiplicity = " << multip << " is not compatible with "
           << "total electrons = " << nTotalE;
        CErr(ss.str(),std::cout);
      }

      computeRIJ();
      computeNNRep();
      computeNNX();
      computeCOM();
      computeCOC();
      computeMOI();
      computeCDist();

    }

    private:

      // Functions to compute member data for Molecule object
      // (See src/molecule/molecule.cxx for documentation)
      void computeRIJ();
      void computeNNRep();
      void computeNNX();
      void computeCOM();
      void computeCOC();
      void computeMOI();
      void computeCDist();

  }; // Molecule struct

}; // namespace ChronusQ

