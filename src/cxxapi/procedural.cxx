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

#include <util/matout.hpp>
#include <util/threads.hpp>
#include <util/mpi.hpp>
#include <util/files.hpp>

#include <memmanager.hpp>
#include <cerr.hpp>
#include <molecule.hpp>
#include <basisset.hpp>
#include <integrals.hpp>
#include <singleslater.hpp>
#include <response.hpp>
#include <realtime.hpp>
#include <itersolver.hpp>

#include <findiff/geomgrad.hpp>
#include <electronintegrals/gradints.hpp>
#include <electronintegrals/twoeints/incore4indexeri.hpp>
#include <electronintegrals/twoeints/gtodirecteri.hpp>
#include <electronintegrals/gradints/incore.hpp>
#include <electronintegrals/gradints/direct.hpp>

#include <cqlinalg/blasext.hpp>
#include <cqlinalg/eig.hpp>

#include <physcon.hpp>


//#include <cubegen.hpp>

namespace ChronusQ {

#ifdef ENABLE_BCAST_COUNTER
  int bcastCounter = 0;
#endif

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


    auto memManager = CQMiscOptions(output,input);


    // Create Molecule and BasisSet objects
    Molecule mol(std::move(CQMoleculeOptions(output,input,scrFileName)));
    std::shared_ptr<BasisSet> basis = CQBasisSetOptions(output,input,mol,"BASIS");
    std::shared_ptr<BasisSet> dfbasis = CQBasisSetOptions(output,input,mol,"DFBASIS");

    auto aoints = CQIntsOptions(output,input,*memManager,basis,dfbasis);

    auto ss = CQSingleSlaterOptions(output,input,*memManager,mol,*basis,aoints);

    // EM Perturbation for SCF
    EMPerturbation emPert;

    CQSCFOptions(output,input,*ss,emPert);
       


    bool rstExists = false;
    if( ss->scfControls.guess == READMO or 
        ss->scfControls.guess == READDEN ) 
      rstExists = true;
    if( ss->scfControls.guess == FCHKMO )
      ss->fchkFileName = scrFileName;

    // Create the restart and scratch files
    SafeFile rstFile(rstFileName, rstExists);
    //SafeFile scrFile(scrFileName);

    if( not rstExists and rank == 0 ) rstFile.createFile();


    if( rank == 0 ) {
      ss->savFile     = rstFile;
      aoints->savFile = rstFile;
    }


    if( not jobType.compare("SCF") or not jobType.compare("RT") or 
        not jobType.compare("RESP") ) {

      ss->formCoreH(emPert);

      // If INCORE, compute and store the ERIs
      if(auto p = std::dynamic_pointer_cast<Integrals<double>>(aoints))
        p->ERI->computeAOInts(*basis, mol, emPert, ELECTRON_REPULSION,
                              {basis->basisType, false, false, false});
      else if(auto p = std::dynamic_pointer_cast<Integrals<dcomplex>>(aoints))
        p->ERI->computeAOInts(*basis, mol, emPert, ELECTRON_REPULSION,
                              {basis->basisType, false, false, false});

      ss->formGuess();
      ss->SCF(emPert);
    }

    std::cout << "Energy: " << ss->totalEnergy;
    std::cout << "OneE Energy: " << ss->OBEnergy;
    std::cout << "TwoE Energy: " << ss->MBEnergy;

    // Numerical gradient
    // size_t acc = 16;
    // if ( acc != 0 ) {
    //   NumGradient grad(input, ss, basis);
    //   // grad.intGrad<double>(acc);
    //   grad.eriGrad<double>(acc);
    // }

    std::vector<std::shared_ptr<DirectERI<double>>> ints;
    for ( auto i = 0; i < mol.atoms.size() * 3; i++ )
      ints.push_back(
        std::make_shared<DirectERI<double>>(*memManager, *basis, 1e-12)
      );

    auto casted = std::dynamic_pointer_cast<Integrals<double>>(aoints);
    casted->gradERI = std::make_shared<GradInts<TwoEInts,double>>(
      *memManager, basis->nBasis, mol.atoms.size(), ints
    );


    aoints->computeGradInts(*memManager, mol, *basis, emPert, {{OVERLAP,1},
        {KINETIC,1}, {NUCLEAR_POTENTIAL,1}, {ELECTRON_REPULSION,1}},
        {basis->basisType, false, false, false});

    // casted->gradERI->output(std::cout, "", true);

    InCore4IndexGradContraction<double,double> contract(*(casted->gradERI));

    auto ssd = std::dynamic_pointer_cast<SingleSlater<double,double>>(ss);

    auto NB = basis->nBasis;
    std::vector<std::vector<TwoBodyContraction<double>>> doubleList;

    std::vector<double*> JList;
    std::vector<std::vector<double*>> KList;

    for ( auto iGrad = 0; iGrad < 3*mol.nAtoms; iGrad++ ) {

      std::vector<TwoBodyContraction<double>> contList;

      auto JGrad = memManager->template malloc<double>(NB*NB);
      std::fill_n(JGrad, NB*NB, 0.);
      std::vector<double*> KGrad;
      for ( auto i = 0; i < ssd->onePDM->nComponent(); i++ ) {
        std::cout << "K index " << i << std::endl;
        KGrad.push_back(memManager->template malloc<double>(NB*NB));
        std::fill_n(KGrad[i], NB*NB, 0.);
      }

      JList.push_back(JGrad);
      KList.push_back(KGrad);

      TwoBodyContraction<double> JGradCont;
      JGradCont.AX = JGrad;
      JGradCont.X  = ssd->onePDM->S().pointer();
      JGradCont.HER = true;
      JGradCont.contType = COULOMB;
      contList.push_back(JGradCont);

      TwoBodyContraction<double> K0GradCont;
      K0GradCont.AX = KGrad[0];
      K0GradCont.X  = ssd->onePDM->S().pointer();
      K0GradCont.HER = true;
      K0GradCont.contType = EXCHANGE;
      contList.push_back(K0GradCont);

      if (ssd->onePDM->hasZ()) {
        TwoBodyContraction<double> K1GradCont;
        K1GradCont.AX = KGrad[1];
        K1GradCont.X  = ssd->onePDM->Z().pointer();
        K1GradCont.HER = true;
        K1GradCont.contType = EXCHANGE;
        contList.push_back(K1GradCont);
      }

      if (ssd->onePDM->hasXY()) {
        TwoBodyContraction<double> K2GradCont;
        K2GradCont.AX = KGrad[2];
        K2GradCont.X  = ssd->onePDM->Y().pointer();
        K2GradCont.HER = true;
        K2GradCont.contType = EXCHANGE;
        contList.push_back(K2GradCont);

        TwoBodyContraction<double> K3GradCont;
        K3GradCont.AX = KGrad[3];
        K3GradCont.X  = ssd->onePDM->X().pointer();
        K3GradCont.HER = true;
        K3GradCont.contType = EXCHANGE;
        contList.push_back(K3GradCont);
      }

      doubleList.push_back(contList);
    }

    GradContractions<double,double>& ccast = contract;
    ccast.gradTwoBodyContract(MPI_COMM_WORLD, false, doubleList, emPert);


    for ( auto i = 0, idx = 0; i < mol.atoms.size(); i++ ) {
      for ( auto xyz = 0; xyz < 3; xyz++, idx++ ) {
        std::cout << "  Atom " << i << ", Cart " << xyz << std::endl;
        prettyPrintSmart(std::cout, "Analytic J Gradient",
          JList[idx], NB, NB, NB);
      }
    }

    for ( auto i = 0, idx = 0; i < mol.atoms.size(); i++ ) {
      for ( auto xyz = 0; xyz < 3; xyz++, idx++ ) {
        std::cout << "  Atom " << i << ", Cart " << xyz << std::endl;
        prettyPrintSmart(std::cout, "Analytic K Gradient",
          KList[idx][SCALAR], NB, NB, NB);
      }
    }

    std::cout << "================================================" << std::endl;

    for ( auto i = 0, idx = 0; i < mol.atoms.size(); i++ ) {
      for ( auto xyz = 0; xyz < 3; xyz++, idx++ ) {
        std::cout << "  Atom " << i << ", Cart " << xyz << std::endl;
        prettyPrintSmart(std::cout, "Analytic K Gradient",
          KList[idx][MZ], NB, NB, NB);
      }
    }

    /*
    auto casted = std::dynamic_pointer_cast<Integrals<double>>(aoints);
    auto NB = basis->nBasis;
    for ( auto i = 0, idx = 0; i < mol.atoms.size(); i++ ) {
      for ( auto xyz = 0; xyz < 3; xyz++, idx++ ) {
        std::cout << "  Atom " << i << ", Cart " << xyz << std::endl;
        prettyPrintSmart(std::cout, "Analytic Overlap Gradient",
          (*casted->gradOverlap)[idx]->pointer(), NB, NB, NB);
      }
    }

    for ( auto i = 0, idx = 0; i < mol.atoms.size(); i++ ) {
      for ( auto xyz = 0; xyz < 3; xyz++, idx++ ) {
        std::cout << "  Atom " << i << ", Cart " << xyz << std::endl;
        prettyPrintSmart(std::cout, "Analytic Kinetic Gradient",
          (*casted->gradKinetic)[idx]->pointer(), NB, NB, NB);
      }
    }

    for ( auto i = 0, idx = 0; i < mol.atoms.size(); i++ ) {
      for ( auto xyz = 0; xyz < 3; xyz++, idx++ ) {
        std::cout << "  Atom " << i << ", Cart " << xyz << std::endl;
        prettyPrintSmart(std::cout, "Analytic Potential Gradient",
          (*casted->gradPotential)[idx]->pointer(), NB, NB, NB);
      }
    }
    */

    if( not jobType.compare("RT") ) {

      if( MPISize() > 1 ) CErr("RT + MPI NYI!",output);

      auto rt = CQRealTimeOptions(output,input,ss,emPert);
      rt->savFile = rstFile;
      rt->doPropagation();

    }

    if( not jobType.compare("RESP") ) {

      auto resp = CQResponseOptions(output,input,ss);
      resp->savFile = rstFile;
      resp->run();

      if( MPIRank(MPI_COMM_WORLD) == 0 ) resp->printResults(output);
      MPI_Barrier(MPI_COMM_WORLD);

    }

    // Output CQ footer
    CQOutputFooter(output);

    // Reset std::cout and std::cerr
    if(outfile)  std::cout.rdbuf(coutbuf);
    if(rankfile) std::cerr.rdbuf(cerrbuf);

  }; // RunChronusQ


 }; // namespace ChronusQ
