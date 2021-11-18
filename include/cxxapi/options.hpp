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

#include <cxxapi/input.hpp>
#include <cxxapi/procedural.hpp>

#include <molecule.hpp>
#include <basisset.hpp>
#include <integrals.hpp>
#include <singleslater.hpp>
#include <realtime.hpp>
#include <response.hpp>
#include <coupledcluster.hpp>
#include <corehbuilder/x2c/atomic.hpp>



// Preprocessor directive to aid the digestion of optional 
// input arguments
#define OPTOPT(x) try{ x; } catch(...) { ; }

namespace ChronusQ {

  // Type of SingleSlater object
  enum RefType { 
    isRawRef,  // non-specified
    isRRef,    // RHF/DFT
    isURef,    // UHF/DFT
    isRORef,   // ROHF/DFT
    isGRef,    // GHF/DFT
    isTwoCRef, // Two-component
    isX2CRef,  // X2C
    isFourCRef // Four-component
  };

  // Type of Job
  enum JobType {
    SCF,
    RT,
    RESP,
    CC,
    BOMD,
    EHRENFEST
  };

  // Tedious, but there isn't an easier way to do this
  inline JobType parseJob(std::string jobStr) {
    JobType job;
    if( jobStr == "SCF" ) {
      job = SCF;
    }
    else if( jobStr == "RT" ) {
      job = RT;
    }
    else if( jobStr == "RESP" ) {
      job = RESP;
    }
    else if( jobStr == "CC" ) {
      job = CC;
    }
    else if( jobStr == "BOMD" ) {
      job = BOMD;
    }
    else if( jobStr == "EHRENFEST" ) {
      job = EHRENFEST;
    }
    else {
      jobStr = "Unrecognized job type \"" + jobStr + "\"!";
      CErr(jobStr);
    }
    return job;
  };

  // A struct that stores reference information
  struct RefOptions {
    
    std::string RCflag = "REAL"; // Real or Complex
    
    RefType refType = isRRef;    // R/U/G/2c/X2C/4c

    bool isKSRef = false;        // HF or DFT

    size_t nC = 1;               // number of component
    bool iCS = true;             // closed shell or not

    std::string funcName;        // DFT functional name
  };

  // Function definitions ofr option parsing. 
  // See src/cxxapi/input/*opts.cxx for documentation

  // Parse the options relating to the Molecule object
  Molecule CQMoleculeOptions(std::ostream &, CQInputFile &, std::string &);

  void CQMOLECULE_VALID(std::ostream&, CQInputFile &);

  void parseGeomInp(Molecule &, std::string &, std::ostream &, bool);

  void parseGeomFchk(Molecule &, std::string &, std::ostream &);

  RefOptions parseRef(std::ostream &, Molecule &, std::vector<std::string> &);

  void buildFunclist(std::vector<std::shared_ptr<DFTFunctional>> &,
    std::string);

  void parseIntParam(std::ostream &, CQInputFile &, IntegrationParam &);

  void parseHamiltonianOptions(std::ostream &, CQInputFile &, 
    BasisSet &basis, std::shared_ptr<IntegralsBase> aoints,
    RefOptions &refOptions, HamiltonianOptions &hamiltonianOptions);

  bool parseAtomicType(std::ostream &, CQInputFile &, ATOMIC_X2C_TYPE &);

  // Parse the options relating to the BasisSet
  std::shared_ptr<BasisSet> CQBasisSetOptions(std::ostream &, CQInputFile &,
    Molecule &, std::string);

  void CQBASIS_VALID(std::ostream&, CQInputFile &, std::string);

  // Parse the options relating to the SingleSlater 
  // (and variants)
  std::shared_ptr<SingleSlaterBase> CQSingleSlaterOptions(
      std::ostream &, CQInputFile &,
      CQMemManager &mem, Molecule &mol, BasisSet &basis,
      std::shared_ptr<IntegralsBase>);

  // Parse the options relating to the NEOSingleSlater
  // (and variants)
  std::vector<std::shared_ptr<SingleSlaterBase>> CQNEOSingleSlaterOptions(
      std::ostream &, CQInputFile &,
      CQMemManager &mem, Molecule &mol,
      BasisSet &ebasis, BasisSet &pbasis,
      std::shared_ptr<IntegralsBase> eaoints,
      std::shared_ptr<IntegralsBase> paoints,
      std::shared_ptr<IntegralsBase> epaoints);

  std::shared_ptr<SingleSlaterBase> CQNEOSSOptions(
      std::ostream &, CQInputFile &,
      CQMemManager &mem, Molecule &mol,
      BasisSet &ebasis, BasisSet &pbasis,
      std::shared_ptr<IntegralsBase> eaoints,
      std::shared_ptr<IntegralsBase> paoints,
      std::shared_ptr<IntegralsBase> epaoints);

  void CQQM_VALID(std::ostream&, CQInputFile &);
  void CQDFTINT_VALID(std::ostream&, CQInputFile &);

  // Parse RT options
  std::shared_ptr<RealTimeBase> CQRealTimeOptions(
    std::ostream &, CQInputFile &, std::shared_ptr<SingleSlaterBase> &,
    EMPerturbation &
  );

  void CQRT_VALID(std::ostream&, CQInputFile &);

  // Parse Response options
  std::shared_ptr<ResponseBase> CQResponseOptions(
    std::ostream &, CQInputFile &, std::shared_ptr<SingleSlaterBase>
  );

  void CQRESPONSE_VALID(std::ostream&, CQInputFile &);
  void CQMOR_VALID(std::ostream&, CQInputFile &);

  // Parse integral options
  std::shared_ptr<IntegralsBase> CQIntsOptions(std::ostream &, 
    CQInputFile &, CQMemManager &, Molecule &,
    std::shared_ptr<BasisSet>, std::shared_ptr<BasisSet>,
    std::shared_ptr<BasisSet>, std::string int_sec = "INTS");

  void CQINTS_VALID(std::ostream&, CQInputFile &);

  // Parse the SCF options
  void CQSCFOptions(std::ostream&, CQInputFile&,
    SingleSlaterBase &, EMPerturbation &);

  void CQSCF_VALID(std::ostream&, CQInputFile &);

  // Parse CC options
#ifdef CQ_HAS_TA
  std::shared_ptr<CCBase> CQCCOptions(std::ostream &, 
     CQInputFile &, std::shared_ptr<SingleSlaterBase> &);
#endif  
  void CQCC_VALID(std::ostream &, CQInputFile &);


  // Parse geometry modifier options
  JobType CQGeometryOptions(std::ostream& out, CQInputFile& input, 
    JobType job, Molecule& mol, std::shared_ptr<SingleSlaterBase> ss,
    std::shared_ptr<RealTimeBase>& rt, std::shared_ptr<IntegralsBase> epints,
    EMPerturbation& emPert);

  JobType CQDynamicsOptions(std::ostream& out, CQInputFile& input, 
    JobType job, Molecule& mol, std::shared_ptr<SingleSlaterBase> ss,
    std::shared_ptr<RealTimeBase>& rt, std::shared_ptr<IntegralsBase> epints,
    EMPerturbation& emPert);

  void CQDYNAMICS_VALID( std::ostream& out, CQInputFile& input );


  std::shared_ptr<CQMemManager> CQMiscOptions(std::ostream &,
    CQInputFile &);

  void CQMISC_VALID(std::ostream&, CQInputFile &);


  inline void CQINPUT_VALID(std::ostream &out, CQInputFile &input) {

    CQMOLECULE_VALID(out,input);
    CQBASIS_VALID(out,input,"BASIS");
    CQBASIS_VALID(out,input,"DFBASIS");
    CQINTS_VALID(out,input);
    CQQM_VALID(out,input);
    CQDFTINT_VALID(out,input);
    CQSCF_VALID(out,input);
    CQRT_VALID(out,input);
    CQRESPONSE_VALID(out,input);
    CQMOR_VALID(out,input);
    CQMISC_VALID(out,input);
    CQCC_VALID(out,input);
    CQDYNAMICS_VALID(out,input);

  }


};




