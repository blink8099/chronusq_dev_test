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

#include <cxxapi/input.hpp>
#include <cxxapi/output.hpp>
#include <cxxapi/options.hpp>
#include <cxxapi/boilerplate.hpp>
#include <cxxapi/procedural.hpp>

#include <util/files.hpp>
#include <util/matout.hpp>
#include <util/mpi.hpp>
#include <util/threads.hpp>
#include <util/timer.hpp>

#include <memmanager.hpp>
#include <cerr.hpp>
#include <molecule.hpp>
#include <basisset.hpp>
#include <integrals.hpp>
#include <singleslater.hpp>
#include <response.hpp>
#include <realtime.hpp>
#include <itersolver.hpp>
#include <coupledcluster.hpp>
#include <mcwavefunction.hpp>
#include <mcscf.hpp>

#include <cqlinalg/blasext.hpp>
#include <cqlinalg/eig.hpp>

#include <physcon.hpp>

#include <corehbuilder/x2c.hpp>
#include <corehbuilder/nonrel.hpp>
#include <fockbuilder/matrixfock.hpp>


//#include <cubegen.hpp>

namespace ChronusQ {

#ifdef ENABLE_BCAST_COUNTER
  int bcastCounter = 0;
#endif

  template <typename MatsT>
  void GatherUSpin(const SquareMatrix<MatsT> &UL, const SquareMatrix<MatsT> &US, MatsT *U);

  template <typename MatsT>
  void ReOrganizeMOSpin(const SquareMatrix<MatsT> &moSpin, SquareMatrix<MatsT> &mo);

  void RunChronusQ(std::string inFileName,
    std::string outFileName, std::string rstFileName,
    std::string scrFileName) {

    // Check to make sure input and output file name are different.
    if( inFileName == outFileName )
      CErr("Input file name and output file name cannot be identical.");

    int rank = MPIRank();
    int size = MPISize();

    // Redirect output to output file if not STDOUT
    std::shared_ptr<std::ofstream> outfile;
    std::streambuf *coutbuf = std::cout.rdbuf();

    if( outFileName.compare("STDOUT") and (rank == 0) ) {

      outfile = std::make_shared<std::ofstream>(outFileName);
      std::cout.rdbuf(outfile->rdbuf());

    }


    // Setup MPI rank files

    std::shared_ptr<std::ofstream> rankfile;
    std::streambuf *cerrbuf = std::cerr.rdbuf();

    if( size > 1 ) {
      
      std::string rankFileName = outFileName + ".mpi." + 
        std::to_string(rank);

      rankfile = std::make_shared<std::ofstream>(rankFileName);
      std::cerr.rdbuf(rankfile->rdbuf());

      std::cerr << "Hello from RANK = " << rank << " / SIZE = " << size 
                << "\n\n";

    }


    std::ostream &output = (rank == 0) ? std::cout : std::cerr;


    // Output CQ header
    CQOutputHeader(output);
    if(rankfile and rank == 0) CQOutputHeader(std::cerr);


    // Parse Input File
    CQInputFile input(inFileName);


    CQINPUT_VALID(output,input);

    // Dump contents of input file into output file
    if( rank == 0 ) {
      std::cout << "\n\n\n";
      std::cout << "Input File:\n" << BannerTop << std::endl;
      std::ifstream inStream(inFileName);
      std::istreambuf_iterator<char> begin_src(inStream);
      std::istreambuf_iterator<char> end_src;
      std::ostreambuf_iterator<char> begin_dest(std::cout);
      std::copy(begin_src,end_src,begin_dest);
      inStream.close();
      std::cout << BannerEnd << "\n\n\n" << std::endl;
    }



    // Determine JOB type
    std::string jobType;
    
    try {
      jobType = input.getData<std::string>("QM.JOB");
    } catch (...) {
      CErr("Must Specify QM.JOB",output);
    }

    bool doNEO = false;
    
    if ( input.containsSection("SCF") ) {
      try {
        doNEO = input.getData<bool>("SCF.NEO");
      } catch(...) { ; }
    }

    auto memManager = CQMiscOptions(output,input);


    // Create Molecule and BasisSet objects
    Molecule mol(std::move(CQMoleculeOptions(output,input,scrFileName)));

    // Create BasisSet objects
    std::shared_ptr<BasisSet> basis = CQBasisSetOptions(output,input,mol,"BASIS");
    std::shared_ptr<BasisSet> dfbasis = CQBasisSetOptions(output,input,mol,"DFBASIS");

    auto aoints = CQIntsOptions(output,input,*memManager,mol,basis,dfbasis,nullptr);

    // Create BasisSet and integral objects for nuclear orbitals 
    std::shared_ptr<BasisSet> prot_basis = 
      doNEO ? CQBasisSetOptions(output,input,mol,"PBASIS") : nullptr;
    auto prot_aoints = 
      doNEO ? CQIntsOptions(output,input,*memManager,mol,prot_basis,dfbasis,nullptr,"PINTS"): nullptr;
    auto ep_aoints   = 
      doNEO? CQIntsOptions(output,input,*memManager,mol,basis,dfbasis,prot_basis,"EPINTS") : nullptr;

    std::shared_ptr<SingleSlaterBase> ss  = nullptr;
    std::shared_ptr<SingleSlaterBase> pss = nullptr;

    SingleSlaterOptions ssOptions;

    // NEO calculation
    if (doNEO) {
      
      // construct the neo single slater objects for electron and proton
      std::vector<std::shared_ptr<SingleSlaterBase>> neo_vec = 
        CQNEOSingleSlaterOptions(output,input,*memManager,mol,
                                 *basis,*prot_basis,
                                 aoints, prot_aoints,
                                 ep_aoints);

      ss  = neo_vec[0]; // electron
      pss = neo_vec[1]; // proton

    }

    // EM Perturbation for SCF
    EMPerturbation emPert;

    // SCF options for electrons
    SCFControls scfControls = CQSCFOptions(output,input,emPert);

    // SCF options for protons
    if (doNEO) {
      ss->scfControls = scfControls;
      pss->scfControls = scfControls;
    } else {
      ssOptions = CQSingleSlaterOptions(output,input,mol,*basis,aoints);
      ssOptions.scfControls = scfControls;
      ss = ssOptions.buildSingleSlater(output,*memManager,mol,*basis,aoints);

      // MO swapping
      HandleOrbitalSwaps(output, input, *ss);
    }

    bool rstExists = false;
    if( ss->scfControls.guess == READMO or 
        ss->scfControls.guess == READDEN and
        scrFileName.empty() )
      rstExists = true;
    else if( ss->scfControls.guess == READDEN
        and not scrFileName.empty() )
      ss->scrBinFileName = scrFileName;
    if( ss->scfControls.guess == FCHKMO )
      ss->fchkFileName = scrFileName;

    if (doNEO) {
      if( pss->scfControls.guess == READMO or 
          pss->scfControls.guess == READDEN ) 
        rstExists = true;
      if( pss->scfControls.guess == FCHKMO )
        pss->fchkFileName = scrFileName;
    }

    // Create the restart and scratch files
    SafeFile rstFile(rstFileName, rstExists);
    //SafeFile scrFile(scrFileName);

    if( not rstExists and rank == 0 ) rstFile.createFile();


    if( rank == 0 ) {
      ss->savFile     = rstFile;
      aoints->savFile = rstFile;
      if (doNEO) { 
        pss->savFile         = rstFile;
        prot_aoints->savFile = rstFile;
        ep_aoints->savFile   = rstFile;
      }
    }


    if( not jobType.compare("SCF") or not jobType.compare("RT") or 
        not jobType.compare("RESP") or not jobType.compare("CC") or 
        not jobType.compare("MCSCF") ) {

      if (ssOptions.hamiltonianOptions.x2cType != X2C_TYPE::OFF) {

        compute_X2C_CoreH_Fock(*memManager, mol, *basis, aoints, emPert, ss, ssOptions);

      }

      ss->formCoreH(emPert);

      // If INCORE, compute and store the ERIs
      aoints->computeAOTwoE(*basis, mol, emPert);

      if (doNEO) { 

        if(auto p = std::dynamic_pointer_cast<Integrals<double>>(prot_aoints))
          prot_aoints->computeAOTwoE(*prot_basis, mol, emPert);
        else
          CErr("NEO with complex integrals NYI!",output);

        if(auto p = std::dynamic_pointer_cast<Integrals<double>>(ep_aoints))
          ep_aoints->computeAOTwoE(*basis, *prot_basis, mol, emPert); 

      }

      // Note, these guessSSOptions does not apply to NEO guess
      SingleSlaterOptions guessSSOptions(ssOptions);
      guessSSOptions.refOptions.isKSRef = false;
      guessSSOptions.refOptions.nC = 1;
      guessSSOptions.hamiltonianOptions.OneEScalarRelativity = false;
      guessSSOptions.hamiltonianOptions.OneESpinOrbit = false;

      ss->formGuess(guessSSOptions);
      ss->SCF(emPert);
    }


    if( not jobType.compare("RT") ) {

      // FIXME: Need to implement RT-NEO
      if (doNEO)
        CErr("RT-NEO NYI!",output);

      if( MPISize() > 1 ) CErr("RT + MPI NYI!",output);

      auto rt = CQRealTimeOptions(output,input,ss,emPert);
      rt->savFile = rstFile;
      rt->doPropagation();

    }

    if( not jobType.compare("RESP") ) {

      // FIXME: Need to implement TD-NEO
      if (doNEO)
        CErr("RESP-NEO NYI!",output);

      auto resp = CQResponseOptions(output,input,ss);
      resp->savFile = rstFile;
      resp->run();

      if( MPIRank(MPI_COMM_WORLD) == 0 ) resp->printResults(output);
      MPI_Barrier(MPI_COMM_WORLD);

    }
    
    if( not jobType.compare("CC")){  

      // FIXME: Need to implement NEO-CC
      if (doNEO)
        CErr("NEO-CC NYI!",output);

      #ifdef CQ_HAS_TA
        auto cc = CQCCOptions(output, input, ss);
        cc->run(); 
      #else
        CErr("TiledArray must be compiled to use Coupled-Cluster code!");
      #endif
    }

    if ( not jobType.compare("MCSCF") ) {
      auto mcscf = CQMCSCFOptions(output,input,ss);
      mcscf->savFile = rstFile;
      mcscf->run(emPert);
    }
    
    ProgramTimer::tock("Chronus Quantum");
    printTimerSummary(std::cout);
     
    // Output CQ footer
    CQOutputFooter(output);

    // Reset std::cout and std::cerr
    if(outfile)  std::cout.rdbuf(coutbuf);
    if(rankfile) std::cerr.rdbuf(cerrbuf);

  }; // RunChronusQ


 }; // namespace ChronusQ
