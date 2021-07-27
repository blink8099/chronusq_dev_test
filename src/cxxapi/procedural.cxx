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

#include <findiff/geomgrad.hpp>
#include <particleintegrals/gradints.hpp>
#include <particleintegrals/twopints/incore4indextpi.hpp>
#include <particleintegrals/twopints/gtodirecttpi.hpp>
#include <particleintegrals/gradints/incore.hpp>
#include <particleintegrals/gradints/direct.hpp>

#include <cqlinalg/blasext.hpp>
#include <cqlinalg/eig.hpp>

#include <geometrymodifier.hpp>
#include <geometrymodifier/moleculardynamics.hpp>
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
    else
      ss = CQSingleSlaterOptions(output,input,*memManager,mol,*basis,aoints);

    // EM Perturbation for SCF
    EMPerturbation emPert;

    // SCF options for electrons
    CQSCFOptions(output,input,*ss,emPert);

    // SCF options for protons
    if (doNEO)
      CQSCFOptions(output,input,*pss,emPert);

    bool rstExists = false;
    if( ss->scfControls.guess == READMO or 
        ss->scfControls.guess == READDEN ) 
      rstExists = true;
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
        not jobType.compare("RESP") or not jobType.compare("CC") ) {

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

      ss->formGuess();
      ss->SCF(emPert);
    }

    std::cout << "Energy: " << ss->totalEnergy;
    std::cout << "OneE Energy: " << ss->OBEnergy;
    std::cout << "TwoE Energy: " << ss->MBEnergy;

    // Numerical gradient
    size_t acc = 0;
    if ( acc != 0 ) {
      NumGradient grad(input, ss, basis);
      grad.doGrad(acc);
      // grad.eriGrad<double>(acc);
    }

    std::vector<std::shared_ptr<InCore4indexTPI<double>>> ints;
    for ( auto i = 0; i < mol.atoms.size() * 3; i++ )
      ints.push_back(
        std::make_shared<InCore4indexTPI<double>>(*memManager, basis->nBasis)
      );

    auto casted = std::dynamic_pointer_cast<Integrals<double>>(aoints);
    casted->gradERI = std::make_shared<GradInts<TwoPInts,double>>(
      *memManager, basis->nBasis, mol.atoms.size(), ints
    );


    // aoints->computeGradInts(*memManager, mol, *basis, emPert, {{OVERLAP,1},
    //     {KINETIC,1}, {NUCLEAR_POTENTIAL,1}, {ELECTRON_REPULSION,1}},
    //     {basis->basisType, false, false, false});

#if 0 //BOMD
    auto ssd = std::dynamic_pointer_cast<SingleSlater<double,double>>(ss);

    MolecularOptions molecularOptions;
    mol.geometryModifier = std::make_shared<MolecularDynamics>(molecularOptions,mol);
    auto BOMD = std::dynamic_pointer_cast<MolecularDynamics>(mol.geometryModifier);
    BOMD->initializeMD(mol);

    auto totalTimeFS = 0.0;
    auto totalTimeAU = 0.0;
    auto ETot0 = 0.0;
    auto ETot = 0.0;
    auto ETotPrevious = 0.0;
    for(auto iStep = 0; iStep < molecularOptions.nNulcearSteps; iStep++){

      aoints->computeAOTwoE(*basis, mol, emPert);
      ssd->formCoreH(emPert);
      ssd->SCF(emPert);
      auto grad = ssd->getGrad(emPert, false, false);

      //molecularOptions.timeStepFS = 0.0;
      //molecularOptions.timeStepAU = 0.0;

      std::cout << std::endl;
      std::cout << "MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD"<<std::endl;
      std::cout << "Molecular Dynamics Information for Step "<<std::setw(8)<<iStep<<std::endl;

      std::cout << std::scientific<<std::setprecision(12);

      std::cout << "Nuclear Repulsion Energy = "<<mol.nucRepEnergy<<std::endl;

      std::cout << std::defaultfloat<<std::setprecision(8);
      BOMD->updateNuclearCoordinates(true,mol,grad,iStep==0,iStep==molecularOptions.nNulcearSteps);

      std::cout << std::endl<<"Time (fs): "<< std::right<<std::setw(16)<<totalTimeFS
                << "  Time (au): "<<std::right<< std::setw(16)<<totalTimeAU<<std::endl;

      std::cout << std::scientific<<std::setprecision(8);
      std::cout <<  "EKin= " << std::right << std::setw(16) << BOMD->nuclearKineticEnergy
                << "  EPot= " << std::right << std::setw(16) << ssd->totalEnergy<<" a.u."<<std::endl;

      ETot = ssd->totalEnergy+BOMD->nuclearKineticEnergy;
      if(iStep == 0) {
        ETot0   = ETot;
	ETotPrevious = ETot;
      }
      std::cout << "ETot= " << std::right << std::setw(16) << ETot
                << " ΔETot (current-previous)= " << std::right << std::setw(16)<< ETot-ETotPrevious
                << " ΔETot (cumulative)= " << std::right << std::setw(16)<< ETot-ETot0<< " a.u."<<std::endl;
      ETotPrevious = ETot;


      basis->updateNuclearCoordinates(mol);

      totalTimeFS += molecularOptions.timeStepFS;
      totalTimeAU += molecularOptions.timeStepAU;

    }
#endif //BOMD

#if 1 //Ehrenfest
    auto ssd = std::dynamic_pointer_cast<SingleSlater<dcomplex,double>>(ss);
    auto rt = CQRealTimeOptions(output,input,ss,emPert);
    rt->savFile = rstFile;


    MolecularOptions molecularOptions;
    mol.geometryModifier = std::make_shared<MolecularDynamics>(molecularOptions,mol);
    auto BOMD = std::dynamic_pointer_cast<MolecularDynamics>(mol.geometryModifier);
    BOMD->initializeMD(mol);

    auto totalTimeFS = 0.0;
    auto totalTimeAU = 0.0;
    auto ETot0 = 0.0;
    auto ETot = 0.0;
    auto ETotPrevious = 0.0;

    rt->intScheme.deltaT = molecularOptions.timeStepAU/
                           (molecularOptions.nMidpointFockSteps*molecularOptions.nElectronicSteps);

    for(auto outerStep = 0; outerStep < molecularOptions.nNuclearSteps; outerStep++){

      basis->updateNuclearCoordinates(mol);
      aoints->computeAOTwoE(*basis, mol, emPert);
      rt->formCoreH(emPert);

      std::cout << std::endl;
      std::cout << "MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD-MD"<<std::endl;
      std::cout << "Molecular Dynamics Information for Step "<<std::setw(8)<<outerStep<<std::endl;



      std::cout << std::defaultfloat<<std::setprecision(8);

      std::cout << std::endl<<"Time (fs): "<< std::right<<std::setw(16)<<totalTimeFS
                << "  Time (au): "<<std::right<< std::setw(16)<<totalTimeAU<<std::endl;

      ETot = rt->totalEnergy()+BOMD->nuclearKineticEnergy;
      if(outerStep == 0) {
        ETot0   = ETot;
	ETotPrevious = ETot;
      }

      std::cout << std::scientific<<std::setprecision(8);
      std::cout <<  "EKin= " << std::right << std::setw(16) << BOMD->nuclearKineticEnergy
                << "  EPot= " << std::right << std::setw(16) << rt->totalEnergy()
                << " ETot= " << std::right << std::setw(16) << ETot << " a.u."<<std::endl;
      std::cout << "ΔETot (current-previous)= " << std::right << std::setw(16)<< ETot-ETotPrevious
                << " ΔETot (cumulative)= " << std::right << std::setw(16)<< ETot-ETot0<< " a.u."<<std::endl;
      ETotPrevious = ETot;


      auto grad = rt->getGrad(emPert);

      for(auto middleStep = 0; middleStep < molecularOptions.nMidpointFockSteps+1; middleStep++) {

        molecularOptions.timeStepAU = molecularOptions.timeStepAU/(molecularOptions.nMidpointFockSteps+1);
        BOMD->updateNuclearCoordinates(middleStep==0, mol,grad,middleStep==0,middleStep==molecularOptions.nMidpointFockSteps);
        molecularOptions.timeStepAU = molecularOptions.timeStepAU*(molecularOptions.nMidpointFockSteps+1);


        basis->updateNuclearCoordinates(mol);
        aoints->computeAOTwoE(*basis, mol, emPert);
        rt->formCoreH(emPert);


        rt->intScheme.restoreStep = outerStep*molecularOptions.nMidpointFockSteps*molecularOptions.nElectronicSteps
                                  + middleStep*molecularOptions.nElectronicSteps;

        // the last step is used to advance nuclear positions and compute energy only
        if(middleStep != molecularOptions.nMidpointFockSteps)
          rt->intScheme.tMax = molecularOptions.timeStepAU*outerStep
                             + (middleStep+1)*molecularOptions.timeStepAU/molecularOptions.nMidpointFockSteps;
	else rt->intScheme.tMax = molecularOptions.timeStepAU*(outerStep+1);


        rt->doPropagation();
        std::cout << std::scientific<<std::setprecision(12);
        std::cout << "Nuclear Repulsion Energy = "<<mol.nucRepEnergy<<std::endl;

      }


      //molecularOptions.timeStepFS = 0.0;
      //molecularOptions.timeStepAU = 0.0;

      totalTimeFS += molecularOptions.timeStepFS;
      totalTimeAU += molecularOptions.timeStepAU;

    }

#endif //Ehrenfest



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

    ProgramTimer::tock("Chronus Quantum");
    printTimerSummary(std::cout);
     
    // Output CQ footer
    CQOutputFooter(output);

    // Reset std::cout and std::cerr
    if(outfile)  std::cout.rdbuf(coutbuf);
    if(rankfile) std::cerr.rdbuf(cerrbuf);

  }; // RunChronusQ


 }; // namespace ChronusQ
