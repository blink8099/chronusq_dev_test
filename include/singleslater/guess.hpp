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

#include <singleslater.hpp>
#include <cqlinalg.hpp>
#include <util/matout.hpp>
#include <corehbuilder/nonrel.hpp>
#include <electronintegrals/twoeints/incore4indexeri.hpp>

namespace ChronusQ {

  /**
   *  \brief List of default atomic multiplicities for 
   *  atomic SCF.
   *
   *  Atomic multiplicities obtained from physics.nist.gov 
   */ 
  static std::map<size_t,size_t> defaultMult(
    {
      { 1  , 2 }, // H
      { 2  , 1 }, // He
      { 3  , 2 }, // Li
      { 4  , 1 }, // Be
      { 5  , 2 }, // B
      { 6  , 3 }, // C
      { 7  , 4 }, // N
      { 8  , 3 }, // O
      { 9  , 2 }, // F
      { 10 , 1 }, // Ne
      { 11 , 2 }, // Na
      { 12 , 1 }, // Mg
      { 13 , 2 }, // Al
      { 14 , 3 }, // Si
      { 15 , 4 }, // P
      { 16 , 3 }, // S
      { 17 , 2 }, // Cl
      { 18 , 1 }, // Ar
      { 19 , 2 }, // K
      { 20 , 1 }, // Ca
      { 21 , 2 }, // Sc
      { 22 , 3 }, // Ti
      { 23 , 4 }, // V
      { 24 , 7 }, // Cr
      { 25 , 6 }, // Mn
      { 26 , 5 }, // Fe
      { 27 , 4 }, // Co
      { 28 , 3 }, // Ni
      { 29 , 2 }, // Cu
      { 30 , 1 }, // Zn
      { 31 , 2 }, // Ga
      { 32 , 3 }, // Ge
      { 33 , 4 }, // As
      { 34 , 3 }, // Se
      { 35 , 2 }, // Br
      { 36 , 1 }, // Kr
      { 37 , 2 }, // Rb
      { 38 , 1 }, // Sr
      { 39 , 2 }, // Y 
      { 40 , 3 }, // Zr 
      { 41 , 6 }, // Nb 
      { 42 , 7 }, // Mo 
      { 43 , 6 }, // Tc
      { 44 , 5 }, // Ru
      { 45 , 4 }, // Rh
      { 46 , 1 }, // Pd
      { 47 , 2 }, // Ag
      { 48 , 1 }, // Cd
      { 49 , 2 }, // In
      { 50 , 3 }, // Sn
      { 51 , 4 }, // Sb
      { 52 , 3 }, // Te
      { 53 , 2 }, // I 
      { 54 , 1 }, // Xe
      { 55 , 2 }, // Cs
      { 56 , 1 }, // Ba
      { 57 , 2 }, // La
      { 58 , 1 }, // Ce
      { 59 , 4 }, // Pr
      { 60 , 5 }, // Nd
      { 61 , 6 }, // Pm
      { 62 , 7 }, // Sm
      { 63 , 8 }, // Eu
      { 64 , 9 }, // Gd
      { 65 , 6 }, // Tb
      { 66 , 5 }, // Dy
      { 67 , 4 }, // Ho
      { 68 , 3 }, // Er
      { 69 , 2 }, // Tm
      { 70 , 1 }, // Yb
      { 71 , 2 }, // Lu
      { 72 , 3 }, // Hf
      { 73 , 4 }, // Ta
      { 74 , 5 }, // W 
      { 75 , 6 }, // Re
      { 76 , 5 }, // Os
      { 77 , 4 }, // Ir
      { 78 , 3 }, // Pt
      { 79 , 2 }, // Au
      { 80 , 1 }, // Hg
      { 81 , 2 }, // Tl
      { 82 , 3 }, // Pb
      { 83 , 4 }, // Bi
      { 84 , 3 }, // Po
      { 85 , 2 }, // At
      { 86 , 1 }, // Rn
      { 87 , 2 }, // Fr
      { 88 , 1 }, // Ra
      { 89 , 2 }, // Ac
      { 90 , 3 }, // Th
      { 91 , 4 }, // Pa
      { 92 , 5 }, // U 
      { 93 , 6 }, // Np
      { 94 , 7 }, // Pu
      { 95 , 8 }, // Am
      { 96 , 9 }, // Cm
      { 97 , 6 }, // Bk
      { 98 , 5 }, // Cf
      { 99 , 4 }, // Es
    }
  );


  /**
   *  \brief Forms a set of guess orbitals for a single slater
   *  determininant SCF in various ways
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::formGuess() {

    if( printLevel > 0 )
      std::cout << "  *** Forming Initial Guess Density for SCF Procedure ***"
                << std::endl << std::endl;

    if( this->molecule().nAtoms == 1  and scfControls.guess == SAD ) {
      CoreGuess();
    } else if( scfControls.guess == CORE ) CoreGuess();
    else if( scfControls.guess == SAD ) SADGuess();
    else if( scfControls.guess == RANDOM ) RandomGuess();
    else if( scfControls.guess == READMO ) ReadGuessMO(); 
    else if( scfControls.guess == READDEN ) ReadGuess1PDM(); 
    else CErr("Unknown choice for SCF.GUESS",std::cout);


    // Common to all guess: form new set of orbitals from
    // initial guess at Fock.
    EMPerturbation pert; // Dummy EM perturbation
    getNewOrbitals(pert,false);
    
    // If RANDOM guess, scale the densites appropriately
    // *** Replicates on all MPI processes ***
    if( scfControls.guess == RANDOM ) {

      size_t NB = basisSet().nBasis;

      double TS = 
        this->template computeOBProperty<double,SCALAR>(this->aoints.overlap->pointer());

      double TZ = 
        this->template computeOBProperty<double,MZ>(this->aoints.overlap->pointer());
      double TY = 
        this->template computeOBProperty<double,MY>(this->aoints.overlap->pointer());
      double TX = 
        this->template computeOBProperty<double,MX>(this->aoints.overlap->pointer());

      double magNorm = std::sqrt(TZ*TZ + TY*TY + TX*TX);

      this->onePDM->S() *= MatsT(this->nO)/MatsT(TS);

      double factorXYZ = magNorm > 1e-10 ? (this->nOA - this->nOB)/magNorm : 0.0;
      if (this->onePDM->hasZ())
        this->onePDM->Z() *= factorXYZ;
      if (this->onePDM->hasXY()) {
        this->onePDM->Y() *= factorXYZ;
        this->onePDM->X() *= factorXYZ;
      }

    }

//prettyPrintSmart(std::cout,"1pdm guess",this->onePDM[0],this->nOrbital(),this->nOrbital(),this->nOrbital());

  }; // SingleSlater<T>::formGuess

  /**
   *  \brief Populates the initial Fock matrix with the core
   *  hamiltonian.
   *
   *  I.e. Neglecting electron-electron interaction. While this
   *  is the simplest choice for an initial guess, it is often
   *  a very poor guess.
   */
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::CoreGuess() {

    if( printLevel > 0 )
      std::cout << "    * Forming the Core Hamiltonian Guess (F = H)\n\n";

    // Form the core guess Fock matrix on the root MPI process
    if( MPIRank(comm) == 0 )
      // Copy over the Core Hamiltonian
      *fockMatrix = *coreH;

#ifdef CQ_ENABLE_MPI
    // BCast random fock to all MPI processes
    if( MPISize(comm) > 1 ) {
      std::cerr  << "  *** Scattering the CORE GUESS Fock ***\n";
      size_t NB = this->fockMatrix->dimension();
      for(auto mat : this->fockMatrix->SZYXPointers())
        MPIBCast(mat,NB*NB,0,comm);
    }
#endif

  }; // SingleSlater::CoreGuess 

  /**
   *  
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::SADGuess() {



    // ROOT Communicator
    int color = (MPIRank(comm) == 0) ? 1 : MPI_UNDEFINED;
    MPI_Comm rcomm = MPICommSplit(comm,color,0);

    //std::cerr << "RCOMM " << rcomm << std::endl;

    if( printLevel > 0 )
      std::cout << "    * Forming the Superposition of Atomic Densities Guess"
                << " (SAD)\n\n";

    size_t NB = basisSet().nBasis;
    EMPerturbation pert;

    // Zero out the densities (For all MPI processes)
    this->onePDM->clear();


    // Determine the unique atoms
    std::vector<Atom> uniqueElements(this->molecule().atoms);

    // Sort the Atoms by atomic number for std::unique
    std::sort(uniqueElements.begin(),uniqueElements.end(),
      [](Atom &a, Atom &b) {
        return a.atomicNumber < b.atomicNumber;
      });

    // Obtain unique elements on sorted list
    auto it = std::unique(uniqueElements.begin(),uniqueElements.end(),
                [](Atom &a, Atom &b){ 
                  return a.atomicNumber == b.atomicNumber;
                });

    // Remove excess elements
    uniqueElements.resize(std::distance(uniqueElements.begin(),it));

    if( printLevel > 0 )
      std::cout << "  *** Found " << uniqueElements.size() 
                << " unique Atoms in Molecule Specification ***\n";


    // Set all atomic centers to origin
    for(auto &u : uniqueElements) u.coord = {0.,0.,0.};

    // Create a map from atoms in molecule to unique atom index
    std::map<size_t,size_t> mapAtom2Uniq;
    for(auto iAtm = 0ul; iAtm < this->molecule().nAtoms; iAtm++) {
      size_t curAtomicNumber = this->molecule().atoms[iAtm].atomicNumber;

      auto el = std::find_if(uniqueElements.begin(),uniqueElements.end(),
                  [&](Atom &a){return a.atomicNumber == curAtomicNumber;});

      if( el == uniqueElements.end() ) 
        CErr("Whoops! Can't find atom in unique list! Contact a developer!");

      mapAtom2Uniq[iAtm] = std::distance(uniqueElements.begin(),el);
    }

    if( printLevel > 0 )
      std::cout << "  *** Running " << uniqueElements.size() 
                << " Atomic SCF calculations for SAD Guess ***\n\n";


    if( MPIRank(comm) == 0 )
    for(auto iUn = 0; iUn < uniqueElements.size(); iUn++) {

      // Get default multiplicity for the Atom
      size_t defaultMultip;
      try { 
        defaultMultip = defaultMult[uniqueElements[iUn].atomicNumber];
      } catch(...) {
        CErr("AtomZ = " + std::to_string(uniqueElements[iUn].atomicNumber)
             + " not supported for SAD Guess");
      }


      std::string multipName;
      switch( defaultMultip ) {

        case 1: multipName = "Singlet"; break;
        case 2: multipName = "Doublet"; break;
        case 3: multipName = "Triplet"; break;
        case 4: multipName = "Quadruplet"; break;
        case 5: multipName = "Quintuplet"; break;
        case 6: multipName = "Sextuplet"; break;
        case 7: multipName = "Septuplet"; break;
        case 8: multipName = "Octuplet"; break;
  
        default: multipName = "UNKNOWN"; break;

      }


      if( printLevel > 0 )
        std::cout << "    * Running AtomZ = " 
                  << uniqueElements[iUn].atomicNumber << " as a " 
                  << multipName << std::endl;

      BASIS_FUNCTION_TYPE basisType = basisSet().basisType;

      Molecule atom(0,defaultMultip,{ uniqueElements[iUn] });
      BasisSet basis(basisSet().basisName,
        basisSet().basisDef, basisSet().inputDef,
        atom, basisType, basisSet().forceCart, false);
     
      std::shared_ptr<Integrals<IntsT>> aointsAtom =
          std::make_shared<Integrals<IntsT>>();
      aointsAtom->ERI = std::make_shared<InCore4indexERI<IntsT>>(
          memManager,basis.nBasis);

      std::shared_ptr<SingleSlater<MatsT,IntsT>> ss =
          std::dynamic_pointer_cast<SingleSlater<MatsT,IntsT>> (
            std::make_shared<HartreeFock<MatsT,IntsT>>(
              rcomm,memManager,atom,basis,*aointsAtom,1, ( defaultMultip == 1 )
            )
          );;
      ss->ERI = std::make_shared<InCore4indexERIContraction<MatsT,IntsT>>(
          *aointsAtom->ERI);

      ss->printLevel = 0;
      ss->scfControls.doIncFock = false;        
      ss->scfControls.dampError = 1e-4;
      ss->scfControls.nKeep     = 8;

      AOIntsOptions aoiOptions{basisType, false, false, false};
      ss->coreHBuilder = std::make_shared<NRCoreH<MatsT,IntsT>>(
            ss->aoints, aoiOptions);
      ss->fockBuilder = std::make_shared<FockBuilder<MatsT,IntsT>>();

      ss->formCoreH(pert);
      aointsAtom->ERI->computeAOInts(basis, atom, pert,
          ELECTRON_REPULSION, aoiOptions);


      ss->formGuess();
      ss->SCF(pert);

      size_t NBbasis = basis.nBasis;

      // Place the atomic densities into the guess density
      // *** This only places it into root MPI process ***

      for(auto iAtm = 0; iAtm < mapAtom2Uniq.size(); iAtm++)
        if( mapAtom2Uniq[iAtm] == iUn ) {
          SetMat('N',NBbasis,NBbasis,MatsT(1.),
              ss->onePDM->S().pointer(),NBbasis,
              this->onePDM->S().pointer() + basisSet().mapCen2BfSt[iAtm]*(1+NB), NB);
        }

    }

    // Spin-Average the SAD density (on MPI root)
    if( this->onePDM->hasZ() and MPIRank(comm) == 0 ) {
      this->onePDM->Z() = (MatsT(this->nOA - this->nOB) / MatsT(this->nO)) * this->onePDM->S();
    }


#ifdef CQ_ENABLE_MPI
    // Broadcast the 1PDM to all MPI processes
    if( MPISize(comm) > 1 ) {
      std::cerr  << "  *** Scattering the SAD Density ***\n";
      for(MatsT *mat : this->onePDM->SZYXPointers())
        MPIBCast(mat,NB*NB,0,comm);
    }

    // Free the ROOT communicator
    if( rcomm != MPI_COMM_NULL) MPICommFree(rcomm);
#endif


    if( printLevel > 0 )
      std::cout << std::endl
                << "  *** Forming Initial Fock Matrix from SAD Density ***\n\n";

    this->formFock(pert,false);


  }; // SingleSlater<T>::SADGuess



  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::RandomGuess() {

    size_t NB = basisSet().nBasis;

    // Set up random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
//  std::uniform_real_distribution<> dis(-5,5);
    std::normal_distribution<> dis(0,5);

    // Form the random Fock matrix on the root MPI process
    if( MPIRank(comm) == 0 ) {

      // Copy over the Core Hamiltonian
      *fockMatrix = *coreH;

      // Randomize the Fock matricies
      for(auto F : this->fockMatrix->SZYXPointers()) {
        for(auto k = 0ul; k < NB*NB; k++) F[k] = dis(gen);
        HerMat('L',NB,F,NB);
      }

    }

#ifdef CQ_ENABLE_MPI
    // BCast random fock to all MPI processes
    if( MPISize(comm) > 1 ) {
      std::cerr  << "  *** Scattering the RANDOM Fock ***\n";
      for(auto mat : this->fockMatrix->SZYXPointers())
        MPIBCast(mat,NB*NB,0,comm);
    }
#endif

  }

  /**
   *  \brief Reads in 1PDM from bin file. 
   *
   **/
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::ReadGuess1PDM() {

    if( printLevel > 0 )
      std::cout << "    * Reading in guess density from file "
        << savFile.fName() << "\n";

    size_t t_hash = typeid(MatsT).hash_code();
    size_t d_hash = typeid(double).hash_code();
    size_t c_hash = typeid(dcomplex).hash_code();

    size_t savHash;
    try{
      savFile.readData("/SCF/FIELD_TYPE", &savHash);
    } catch (...) {
      CErr("Cannot find /SCF/FIELD_TYPE on rstFile!",std::cout);
    }



    if( t_hash != savHash ) {
  
      bool t_is_double  = t_hash == d_hash;
      bool t_is_complex = t_hash == c_hash;
  
      bool s_is_double  = savHash == d_hash;
      bool s_is_complex = savHash == c_hash;

      std::string t_field = t_is_double ? "REAL" : "COMPLEX";
      std::string s_field = s_is_double ? "REAL" : "COMPLEX";

      std::string message = "/SCF/FIELD_TYPE on disk (" + s_field +
        ") is incompatible with current FIELD_TYPE (" + t_field + ")";

      CErr(message,std::cout);
    }


 
    // dimension of 1PDM 
    auto NB = basisSet().nBasis;
    auto NB2 = NB*NB;


    auto DSdims = savFile.getDims( "SCF/1PDM_SCALAR" );
    auto DZdims = savFile.getDims( "SCF/1PDM_MZ" );
    auto DYdims = savFile.getDims( "SCF/1PDM_MY" );
    auto DXdims = savFile.getDims( "SCF/1PDM_MX" );

    bool hasDS = DSdims.size() != 0;
    bool hasDZ = DZdims.size() != 0;
    bool hasDY = DYdims.size() != 0;
    bool hasDX = DXdims.size() != 0;

    bool r2DS = DSdims.size() == 2;
    bool r2DZ = DZdims.size() == 2;
    bool r2DY = DYdims.size() == 2;
    bool r2DX = DXdims.size() == 2;


    // Errors in 1PDM SCALAR
    if( not hasDS )
      CErr("SCF/1PDM_SCALAR does not exist in " + savFile.fName(), std::cout); 

    else if( not r2DS ) 
      CErr("SCF/1PDM_SCALAR not saved as a rank-2 tensor in " + 
          savFile.fName(), std::cout); 

    else if( DSdims[0] != NB or DSdims[1] != NB ) {

      std::cout << "    * Incompatible SCF/1PDM_SCALAR:";
      std::cout << "  Recieved (" << DSdims[0] << "," << DSdims[1] << ")"
        << " :"; 
      std::cout << "  Expected (" << NB << "," << NB << ")"; 
      CErr("Wrong dimension of 1PDM SCALAR!",std::cout);

    }

    // Read in 1PDM SCALAR
    std::cout << "    * Found SCF/1PDM_SCALAR !" << std::endl;
    savFile.readData("/SCF/1PDM_SCALAR",this->onePDM->S().pointer());


    // Oddities in Restricted
    if( this->nC == 1 and this->iCS ) {

      if( hasDZ )
        std::cout << "    * WARNING: Reading in SCF/1PDM_SCALAR as "
          << "restricted guess but " << savFile.fName() 
          << " contains SCF/1PDM_MZ" << std::endl;

      if( hasDY )
        std::cout << "    * WARNING: Reading in SCF/1PDM_SCALAR as "
          << "restricted guess but " << savFile.fName() 
          << " contains SCF/1PDM_MY" << std::endl;

      if( hasDX )
        std::cout << "    * WARNING: Reading in SCF/1PDM_SCALAR as "
          << "restricted guess but " << savFile.fName() 
          << " contains SCF/1PDM_MX" << std::endl;

    }


    // MZ
    if( this->nC == 2 or not this->iCS ) {

      if( not hasDZ ) {

        std::cout <<  "    * WARNING: SCF/1PDM_MZ does not exist in "
          << savFile.fName() << " -- Zeroing out SCF/1PDM_MZ" << std::endl;

        this->onePDM->Z().clear();


      } else if( not r2DZ ) 
        CErr("SCF/1PDM_MZ not saved as a rank-2 tensor in " + 
            savFile.fName(), std::cout); 

      else if( DZdims[0] != NB or DZdims[1] != NB ) {

        std::cout << "    * Incompatible SCF/1PDM_MZ:";
        std::cout << "  Recieved (" << DZdims[0] << "," << DZdims[1] << ")"
          << " :"; 
        std::cout << "  Expected (" << NB << "," << NB << ")"; 
        CErr("Wrong dimension of 1PDM MZ!",std::cout);

      } else {

        std::cout << "    * Found SCF/1PDM_MZ !" << std::endl;
        savFile.readData("SCF/1PDM_MZ",this->onePDM->Z().pointer());

      }

      // Oddities in Unrestricted
      if( this->nC == 2 ) {

        if( hasDY )
          std::cout << "    * WARNING: Reading in SCF/1PDM_MZ as "
            << "unrestricted guess but " << savFile.fName() 
            << " contains SCF/1PDM_MY" << std::endl;

        if( hasDX )
          std::cout << "    * WARNING: Reading in SCF/1PDM_MZ as "
            << "unrestricted guess but " << savFile.fName() 
            << " contains SCF/1PDM_MX" << std::endl;

      }

    }


    if( this->nC == 2 ) {

      if( not hasDY ) {

        std::cout <<  "    * WARNING: SCF/1PDM_MY does not exist in "
          << savFile.fName() << " -- Zeroing out SCF/1PDM_MY" << std::endl;

        this->onePDM->Y().clear();


      } else if( not r2DY ) 
        CErr("SCF/1PDM_MY not saved as a rank-2 tensor in " + 
            savFile.fName(), std::cout); 

      else if( DYdims[0] != NB or DYdims[1] != NB ) {

        std::cout << "    * Incompatible SCF/1PDM_MY:";
        std::cout << "  Recieved (" << DYdims[0] << "," << DYdims[1] << ")"
          << " :"; 
        std::cout << "  Expected (" << NB << "," << NB << ")"; 
        CErr("Wrong dimension of 1PDM MY!",std::cout);

      } else {

        std::cout << "    * Found SCF/1PDM_MY !" << std::endl;
        savFile.readData("SCF/1PDM_MY",this->onePDM->Y().pointer());

      }


      if( not hasDX ) {

        std::cout <<  "    * WARNING: SCF/1PDM_MX does not exist in "
          << savFile.fName() << " -- Zeroing out SCF/1PDM_MX" << std::endl;

        this->onePDM->X().clear();


      } else if( not r2DX ) 
        CErr("SCF/1PDM_MX not saved as a rank-2 tensor in " + 
            savFile.fName(), std::cout); 

      else if( DXdims[0] != NB or DXdims[1] != NB ) {

        std::cout << "    * Incompatible SCF/1PDM_MX:";
        std::cout << "  Recieved (" << DXdims[0] << "," << DXdims[1] << ")"
          << " :"; 
        std::cout << "  Expected (" << NB << "," << NB << ")"; 
        CErr("Wrong dimension of 1PDM MX!",std::cout);

      } else {

        std::cout << "    * Found SCF/1PDM_MX !" << std::endl;
        savFile.readData("SCF/1PDM_MX",this->onePDM->X().pointer());

      }


    }




    std::cout << "\n" << std::endl;
    if( printLevel > 0 )
      std::cout << std::endl
                << "  *** Forming Initial Fock Matrix from Guess Density ***\n\n";

    std::cout << "\n" << std::endl;
    EMPerturbation pert;
    formFock(pert,false);

  } // SingleSlater<T>::ReadGuess1PDM()



  /**
   *  \brief Reads in MOs from bin file. 
   *
   **/
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::ReadGuessMO() {

    if( printLevel > 0 )
      std::cout << "    * Reading in guess orbitals from file "
        << savFile.fName() << "\n";
 
    size_t t_hash = typeid(MatsT).hash_code();
    size_t d_hash = typeid(double).hash_code();
    size_t c_hash = typeid(dcomplex).hash_code();

    size_t savHash; 

    try{
      savFile.readData("/SCF/FIELD_TYPE", &savHash);
    } catch (...) {
      CErr("Cannot find /SCF/FIELD_TYPE on rstFile!",std::cout);
    }


    if( t_hash != savHash ) {
  
      bool t_is_double  = t_hash == d_hash;
      bool t_is_complex = t_hash == c_hash;
  
      bool s_is_double  = savHash == d_hash;
      bool s_is_complex = savHash == c_hash;

      std::string t_field = t_is_double ? "REAL" : "COMPLEX";
      std::string s_field = s_is_double ? "REAL" : "COMPLEX";

      std::string message = "/SCF/FIELD_TYPE on disk (" + s_field +
        ") is incompatible with current FIELD_TYPE (" + t_field + ")";

      CErr(message,std::cout);
    }

    // dimension of mo1 and mo2
    auto NB = this->nC * this->nAlphaOrbital();
    auto NB2 = NB*NB;

    auto MO1dims = savFile.getDims( "SCF/MO1" );
    auto MO2dims = savFile.getDims( "SCF/MO2" );


    // Find errors in MO1
    if( MO1dims.size() == 0 ) 
      CErr("SCF/MO1 does not exist in " + savFile.fName(), std::cout); 

    if( MO1dims.size() != 2 ) 
      CErr("SCF/MO1 not saved as a rank-2 tensor in " + savFile.fName(), 
          std::cout); 

    if( MO1dims[0] != NB or MO1dims[1] != NB ) {

      std::cout << "    * Incompatible SCF/MO1:";
      std::cout << "  Recieved (" << MO1dims[0] << "," << MO1dims[1] << ")"
        << " :"; 
      std::cout << "  Expected (" << NB << "," << NB << ")"; 
      CErr("Wrong number of MO coefficients!",std::cout);

    }



    // MO2 + RHF is odd, print warning 
    if( MO2dims.size() != 0 and this->nC == 1 and this->iCS )
      std::cout << "    * WARNING: Reading in SCF/MO1 as restricted guess "
                << "but " << savFile.fName() << " contains SCF/MO2"
                << std::endl;


    // Read in MO1
    std::cout << "    * Found SCF/MO1 !" << std::endl;
    savFile.readData("SCF/MO1",this->mo[0].pointer());





    // Unrestricted calculations
    if( this->nC == 1 and not this->iCS ) {

      if( MO2dims.size() == 0 )
        std::cout << "    * WARNING: SCF/MO2 does not exist in "
          << savFile.fName() << " -- Copying SCF/MO1 -> SCF/MO2 " << std::endl;

      if( MO2dims.size() > 2  ) 

        CErr("SCF/MO2 not saved as a rank-2 tensor in " + savFile.fName(), 
            std::cout); 

      else if( MO2dims[0] != NB or MO2dims[1] != NB ) {

        std::cout << "    * Incompatible SCF/MO2:";
        std::cout << "  Recieved (" << MO2dims[0] << "," << MO2dims[1] << ")"
          << " :"; 
        std::cout << "  Expected (" << NB << "," << NB << ")"; 
        CErr("Wrong number of MO coefficients!",std::cout);

      }


      // Read in MO2
      if( MO2dims.size() == 0 )
        this->mo[1] = this->mo[0];
      else {
        std::cout << "    * Found SCF/MO2 !" << std::endl;
        savFile.readData("SCF/MO2",this->mo[1].pointer());
      }

    }

    // Save MO coefficents if doing a "reset" after the SCF
    if ( scfControls.resetMOCoeffs ) {
      savFile.safeWriteData("SCF/OLD_MO1", this->mo[0].pointer(), {NB, NB});
      if ( this->nC == 1 and not this->iCS )
        savFile.safeWriteData("SCF/OLD_MO2", this->mo[1].pointer(), {NB, NB});
    }


    // Form density from MOs
    formDensity();

    std::cout << "\n" << std::endl;
    if( printLevel > 0 )
      std::cout << std::endl
                << "  *** Forming Initial Fock Matrix from Guess Density ***\n\n";

    std::cout << "\n" << std::endl;
    EMPerturbation pert;
    formFock(pert,false);

  } // SingleSlater<T>::ReadGuessMO()

}; // namespace ChronusQ

