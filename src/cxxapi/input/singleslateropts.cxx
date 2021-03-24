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
#include <cxxapi/options.hpp>
#include <cxxapi/output.hpp>
#include <cerr.hpp>
#include <corehbuilder.hpp>
#include <corehbuilder/nonrel.hpp>
#include <corehbuilder/x2c.hpp>
#include <corehbuilder/x2c/atomic.hpp>
#include <corehbuilder/fourcomp.hpp>
#include <fockbuilder.hpp>
#include <fockbuilder/rofock.hpp>
#include <fockbuilder/fourcompfock.hpp>
#include <electronintegrals/twoeints.hpp>
#include <electronintegrals/twoeints/gtodirecteri.hpp>
#include <electronintegrals/twoeints/incore4indexeri.hpp>
#include <electronintegrals/twoeints/giaodirecteri.hpp>
#include <electronintegrals/twoeints/incorerieri.hpp>
#include <electronintegrals/twoeints/incore4indexreleri.hpp>
#include <electronintegrals/twoeints/gtodirectreleri.hpp>

namespace ChronusQ {

  /**
   *
   *  Check valid keywords in the section.
   *
  */
  void CQQM_VALID( std::ostream &out, CQInputFile &input ) {

    // Allowed keywords
    std::vector<std::string> allowedKeywords = {
      "REFERENCE",
      "JOB",
      "X2CTYPE",
      "SPINORBITSCALING",
      "ATOMICX2C"
    };

    // Specified keywords
    std::vector<std::string> qmKeywords = input.getDataInSection("QM");

    // Make sure all of basisKeywords in allowedKeywords
    for( auto &keyword : qmKeywords ) {
      auto ipos = std::find(allowedKeywords.begin(),allowedKeywords.end(),keyword);
      if( ipos == allowedKeywords.end() ) 
        CErr("Keyword QM." + keyword + " is not recognized",std::cout);// Error
    }
    // Check for disallowed combinations (if any)
  }

  /**
   *
   *  Check valid keywords in the section.
   *
  */
  void CQDFTINT_VALID( std::ostream &out, CQInputFile &input ) {

    // Allowed keywords
    std::vector<std::string> allowedKeywords = {
      "EPS",
      "NANG",
      "NRAD",
      "NMACRO"
    };

    // Specified keywords
    std::vector<std::string> dftintKeywords = input.getDataInSection("DFTINT");

    // Make sure all of basisKeywords in allowedKeywords
    for( auto &keyword : dftintKeywords ) {
      auto ipos = std::find(allowedKeywords.begin(),allowedKeywords.end(),keyword);
      if( ipos == allowedKeywords.end() ) 
        CErr("Keyword DFTINT." + keyword + " is not recognized",std::cout);// Error
    }
    // Check for disallowed combinations (if any)
  }

  /**
   *  \brief Construct a SingleSlater object using the input 
   *  file.
   *
   *  \param [in] out    Output device for data / error output.
   *  \param [in] input  Input file datastructure
   *  \param [in] aoints AOIntegrals object for SingleSlater
   *                     construction
   *
   *  \returns shared_ptr to a SingleSlaterBase object
   *    constructed from the input options.
   *
   */ 
  std::shared_ptr<SingleSlaterBase> CQSingleSlaterOptions(
    std::ostream &out, CQInputFile &input,
    CQMemManager &mem, Molecule &mol, BasisSet &basis,
    std::shared_ptr<IntegralsBase> aoints) {

    out << "  *** Parsing QM.REFERENCE options ***\n";

    // Initialize HamiltonianOptions
    HamiltonianOptions hamiltonianOptions;

    // Attempt to find reference
    std::string reference;
    try { 
      reference = input.getData<std::string>("QM.REFERENCE");
    } catch(...) {
      CErr("QM.REFERENCE Keyword not found!",out);
    }

    // Digest reference string
    // Trim Spaces
    trim(reference);

    // Split into tokens
    std::vector<std::string> tokens;
    split(tokens,reference);
    for(auto &X : tokens) trim(X);

    std::string RCflag;

    // Determine the Real/Complex flag
    if( tokens.size() == 1 )      RCflag = "AUTO";
    else if( tokens.size() == 2 ) RCflag = tokens[0];
    else CErr("QM.REFERENCE Field not valid",out);


    // Kohn-Sham Keywords
    std::vector<std::string> KSRefs {
      "SLATER", 
      "B88",
      "LSDA",
      "SVWN5",
      "BLYP",
      "PBEXPBEC",
      "B3LYP",
      "B3PW91",
      "PBE0",
      "BHANDHLYP",
      "BHANDH",
    };

    // All reference keywords
    std::vector<std::string> rawRefs(KSRefs);
    rawRefs.insert(rawRefs.begin(),"HF");

    // Construct R/U/RO/G/X2C/4C reference keywords
    std::vector<std::string> RRefs, URefs, RORefs, GRefs, X2CRefs, TwoCRefs, FourCRefs;
    for(auto &f : rawRefs) {
      RRefs.emplace_back( "R" + f );
      URefs.emplace_back( "U" + f );
      GRefs.emplace_back( "G" + f );
      X2CRefs.emplace_back( "X2C" + f );
      TwoCRefs.emplace_back( "2C" + f );
    }
    RORefs.emplace_back( "ROHF" );
    FourCRefs.emplace_back( "4CHF" );

    // This is the reference string to be parsed
    std::string refString = tokens.back();
   

    // Determine type of reference
    bool isRawRef = 
      std::find(rawRefs.begin(),rawRefs.end(),refString) != rawRefs.end();
    bool isRRef   = 
      std::find(RRefs.begin(),RRefs.end(),refString) != RRefs.end();
    bool isURef   = 
      std::find(URefs.begin(),URefs.end(),refString) != URefs.end();
    bool isRORef  =
      std::find(RORefs.begin(),RORefs.end(),refString) != RORefs.end();
    bool isGRef   = 
      std::find(GRefs.begin(),GRefs.end(),refString) != GRefs.end();
    bool isTwoCRef   =
      std::find(TwoCRefs.begin(),TwoCRefs.end(),refString) != TwoCRefs.end();
    bool isX2CRef = 
      std::find(X2CRefs.begin(),X2CRefs.end(),refString) != X2CRefs.end();
    bool isFourCRef = 
      std::find(FourCRefs.begin(),FourCRefs.end(),refString) != FourCRefs.end();

    // Throw an error if not a valid reference keyword
    if( not isRawRef and not isRRef and not isURef and not isRORef and 
        not isGRef and not isX2CRef and not isTwoCRef and not isFourCRef )
      CErr(refString + " is not a valid QM.REFERENCE",out);

    // Cleanup the reference string
    if( not isRawRef )
      if( isX2CRef ) refString.erase(0,3);
      else if( isFourCRef or isTwoCRef ) refString.erase(0,2);
      else           refString.erase(0,1);


    // Handle KS related queries
    bool isKSRef = 
      std::find(KSRefs.begin(),KSRefs.end(),refString) != KSRefs.end();

    std::string funcName;
    if( isKSRef )
      funcName = refString;

    // Build Functional List
    std::vector<std::shared_ptr<DFTFunctional>> funcList;
    if( isKSRef ) {
      if( not funcName.compare("B88") )
        funcList.push_back(
          std::dynamic_pointer_cast<DFTFunctional>(
            std::make_shared<BEightyEight>()
          )
        );

      if( not funcName.compare("SLATER") )
        funcList.push_back(
          std::dynamic_pointer_cast<DFTFunctional>(
            std::make_shared<SlaterExchange>()
          )
        );

      if( not funcName.compare("LSDA") or not funcName.compare("LDA") ) {

        funcList.push_back(
          std::dynamic_pointer_cast<DFTFunctional>(
            std::make_shared<SlaterExchange>()
          )
        );

        funcList.push_back(
          std::dynamic_pointer_cast<DFTFunctional>(
            std::make_shared<VWNV>()
          )
        );

      }

      if( not funcName.compare("BLYP") ) {

        funcList.push_back(
          std::dynamic_pointer_cast<DFTFunctional>(
            std::make_shared<BEightyEight>()
          )
        );

        funcList.push_back(
          std::dynamic_pointer_cast<DFTFunctional>(
            std::make_shared<LYP>()
          )
        );

      }

      if( not funcName.compare("SVWN5") ) {

        funcList.push_back(
          std::dynamic_pointer_cast<DFTFunctional>(
            std::make_shared<SlaterExchange>()
          )
        );

        funcList.push_back(
          std::dynamic_pointer_cast<DFTFunctional>(
            std::make_shared<VWNV_G>()
          )
        );

      }

      if( not funcName.compare("PBEXPBEC") ) {

        funcList.push_back(
          std::dynamic_pointer_cast<DFTFunctional>(
            std::make_shared<PBEX>()
          )
        );

        funcList.push_back(
          std::dynamic_pointer_cast<DFTFunctional>(
            std::make_shared<PBEC>()
          )
        );

      }

      if( not funcName.compare("B3LYP") ) 
        funcList.push_back(
          std::dynamic_pointer_cast<DFTFunctional>(
            std::make_shared<B3LYP>()
          )
        );

      if( not funcName.compare("B3PW91") ) 
        funcList.push_back(
          std::dynamic_pointer_cast<DFTFunctional>(
            std::make_shared<B3PW91>()
          )
        );

      if( not funcName.compare("PBE0") ) 
        funcList.push_back(
          std::dynamic_pointer_cast<DFTFunctional>(
            std::make_shared<PBE0>()
          )
        );

      if( not funcName.compare("BHANDH") )
        funcList.push_back(
          std::dynamic_pointer_cast<DFTFunctional>(
            std::make_shared<BHANDH>()
          )
        );

      if( not funcName.compare("BHANDHLYP") )
        funcList.push_back(
          std::dynamic_pointer_cast<DFTFunctional>(
            std::make_shared<BHANDHLYP>()
          )
        );

    }
      


    // Setup references
    size_t nC = 1; bool iCS;

    // Raw reference
    if( isRawRef ) {
      out << "  *** Auto-determination of reference: " << refString << " -> ";
      iCS = mol.multip == 1;

      if(iCS) out << "R" << refString;
      else    out << "U" << refString;

      out << " ***" << std::endl;
      
    } else if( isRRef )
      if( mol.multip != 1 )
        CErr("Spin-Restricted Reference only valid for singlet spin multiplicities",out);
      else
        iCS = true;
    else if( isURef  or isRORef )
      iCS = false;
    else if( isGRef or isTwoCRef or isX2CRef ) {
      iCS = false; nC = 2;
    }
    else if( isFourCRef ) {
      iCS = false; nC = 4;
    }

    // Sanity Checks
    bool isGIAO = basis.basisType == COMPLEX_GIAO;

    if( nC == 2 and not RCflag.compare("REAL") )
      CErr("Real + Two-Component not valid",out);

    if( nC == 4 and not RCflag.compare("REAL") )
      CErr("Real + Four-Component not valid",out);

    if( isGIAO and not RCflag.compare("REAL") )
      CErr("Real + GIAO not valid",out);

    if( isGIAO and isKSRef )
      CErr("KS + GIAO not valid",out);

    if( isGIAO and isX2CRef )
      CErr("X2C + GIAO not valid",out);







    // Determine Real/Complex if need be
    if(not RCflag.compare("AUTO") ) {
      if( nC == 2 or nC == 4 )
        RCflag = "COMPLEX";
      else
        RCflag = "REAL";

      out << "  *** Auto-determination of wave function field: AUTO -> " 
          << RCflag << " ***" << std::endl;
    }

    out << "\n\n";


    // Override core hamiltoninan type for X2C
      

    // FIXME: Should put this somewhere else
    // Parse KS integration

    IntegrationParam intParam;

    if( input.containsSection("DFTINT") and isKSRef ) {

      OPTOPT( intParam.epsilon = input.getData<double>("DFTINT.EPS")  );
      OPTOPT( intParam.nAng    = input.getData<size_t>("DFTINT.NANG") );
      OPTOPT( intParam.nRad    = input.getData<size_t>("DFTINT.NRAD") );
      OPTOPT( intParam.nRadPerBatch    = input.getData<size_t>("DFTINT.NMACRO") );

    }

    if( isKSRef ) {

      out << "\nDFT Integration Settings:\n" << BannerTop << "\n\n" ;
      out << std::left;

      out << "  " << std::setw(28) << "Screening Tolerance:";
      out << intParam.epsilon << std::endl;

      out << "  " << std::setw(28) << "Angular Grid:";
      out <<  "Lebedev (" << intParam.nAng << ")" << std::endl;
      out << "  " << std::setw(28) << "Radial Grid:";
      out <<  "Euler-Maclaurin (" << intParam.nRad << ")" << std::endl;
      out << "  " << std::setw(28) << "Macro Batch Size:";
      out <<  intParam.nRadPerBatch << " Radial Points" << std::endl;

      out << std::endl << BannerEnd << std::endl;

    }




  #define KS_LIST(T) \
    funcName,funcList,MPI_COMM_WORLD,intParam,mem,mol,basis,dynamic_cast<Integrals<T>&>(*aoints),nC,iCS

  #define HF_LIST(T) \
    MPI_COMM_WORLD,mem,mol,basis,dynamic_cast<Integrals<T>&>(*aoints),nC,iCS


    // Construct the SS object
    std::shared_ptr<SingleSlaterBase> ss;

    if( not RCflag.compare("REAL") )
      if( isKSRef )
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<KohnSham<double,double>>( KS_LIST(double) )
          );
      else if(isRORef)
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<double,double>>(
            "Real Restricted Open-shell Hartree-Fock", "R-ROHF", HF_LIST(double) )
          );
      else if( not isGIAO )
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<double,double>>( HF_LIST(double) )
          );
      else
        CErr("GIAO + REAL is not a valid option.",out);

    else if( not RCflag.compare("COMPLEX") and not isGIAO )
      if( isKSRef and isX2CRef)
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<KohnSham<dcomplex,double>>(
              "Exact Two Component", "X2C-", KS_LIST(double)
            )
          );
      else if( isKSRef )
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<KohnSham<dcomplex,double>>( KS_LIST(double) )
          );
      else if( isX2CRef )
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<dcomplex,double>>(
              "Exact Two Component","X2C-",HF_LIST(double)
            )
          );
      else if( isRORef )
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<dcomplex,double>>(
              "Complex Restricted Open-shell Hartree-Fock", "C-ROHF", HF_LIST(double)
            )
          );
      else if( isFourCRef )
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<dcomplex,double>>(
              "Four Component","4C-",HF_LIST(double)
            )
          );
      else // isGRef or isTwoCRef
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<dcomplex,double>>( HF_LIST(double) )
          );
    else
      if( isKSRef and isX2CRef )
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<KohnSham<dcomplex,dcomplex>>(
              "Exact Two Component", "X2C-", KS_LIST(dcomplex)
            )
          );
      else if( isKSRef )
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<KohnSham<dcomplex,dcomplex>>( KS_LIST(dcomplex) )
          );
      else if( isX2CRef )
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<dcomplex,dcomplex>>(
              "Exact Two Component","X2C-",HF_LIST(dcomplex)
            )
          );
      else if(isRORef)
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<dcomplex,dcomplex>>(
              "Complex Restricted Open-shell Hartree-Fock", "C-ROHF", HF_LIST(dcomplex)
            )
          );
      else if( isFourCRef )
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<dcomplex,dcomplex>>(
              "Four Component","4C-",HF_LIST(dcomplex)
            )
          );
      else // isGRef or isTwoCRef
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<dcomplex,dcomplex>>( HF_LIST(dcomplex) )
          );




    // Parse hamiltonianOptions
    hamiltonianOptions.basisType = basis.basisType;

    std::string X;

    // Parse X2C option
    // X2CType = off (default), spinfree, onee, twoe
    X = "DEFAULT";
    OPTOPT( X = input.getData<std::string>("QM.X2CTYPE")  );
    trim(X);
    if( not X.compare("SPINFREE") ) {

      hamiltonianOptions.OneEScalarRelativity = true;
      hamiltonianOptions.OneESpinOrbit = false;
      hamiltonianOptions.Boettger = false;
      hamiltonianOptions.AtomicMeanField = false;

    } else if( not X.compare("ONEE") or not X.compare("ONEELECTRON")
               or ( not X.compare("DEFAULT") and isX2CRef ) ) {
      // Legacy X2C- reference is equilvalent to 2C- reference + OneE-X2C

      hamiltonianOptions.OneEScalarRelativity = true;
      hamiltonianOptions.OneESpinOrbit = true;
      hamiltonianOptions.Boettger = true;
      hamiltonianOptions.AtomicMeanField = false;

    } else if( not X.compare("TWOE") or not X.compare("TWOELECTRON")) {

      CErr(X + " NYI",out);

    } else if( not X.compare("OFF")
               or ( not X.compare("DEFAULT") and not isX2CRef ) ) {

      if ( isFourCRef ) {

        hamiltonianOptions.OneEScalarRelativity = true;
        hamiltonianOptions.OneESpinOrbit = true;

      } else {

        hamiltonianOptions.OneEScalarRelativity = false;
        hamiltonianOptions.OneESpinOrbit = false;

      }

      hamiltonianOptions.Boettger = false;
      hamiltonianOptions.AtomicMeanField = false;

    } else  {

      CErr(X + " not a valid QM.X2CTYPE",out);

    }



    // Parse one-electron spin-orbie scaling option
    // SpinOrbitScaling  = noscaling, boettger (dafault), atomicmeanfield (amfi)
    X = "DEFAULT"; // Unspecified value
    OPTOPT( X = input.getData<std::string>("QM.SPINORBITSCALING")  );
    trim(X);
    if( not X.compare("NOSCALING") ) {

      hamiltonianOptions.Boettger = false;
      hamiltonianOptions.AtomicMeanField = false;

    } else if( not X.compare("BOETTGER") ) {

      if( not hamiltonianOptions.OneESpinOrbit ) 
        CErr("Spin-Orbit Scaling = "+ X + " is not compatible with X2CType = SpinFree",out);
      hamiltonianOptions.Boettger = true;
      hamiltonianOptions.AtomicMeanField = false;

    } else if( not X.compare("AMFI") or not X.compare("ATOMICMEANFIELD")) {

      if( not hamiltonianOptions.OneESpinOrbit ) 
        CErr("Spin-Orbit Scaling = "+ X + " is not compatible with X2CType = SpinFree",out);
      hamiltonianOptions.Boettger = false;
      hamiltonianOptions.AtomicMeanField = true;

    } else if( not X.compare("DEFAULT") ) {

      if ( hamiltonianOptions.OneESpinOrbit and not isFourCRef) {

        hamiltonianOptions.Boettger = true;
        hamiltonianOptions.AtomicMeanField = false;

      } else {

        hamiltonianOptions.Boettger = false;
        hamiltonianOptions.AtomicMeanField = false;

      }

    } else {

      CErr(X + " not a valid QM.X2CTYPE",out);

    }



    // Parse Atomic X2C option
    // AtomicX2C  = ALH, ALU, DLH, DLU, OFF (default)
    bool atomic = false;
    ATOMIC_X2C_TYPE atomicX2CType = {false,false};
    X = "OFF";
    OPTOPT( X = input.getData<std::string>("QM.ATOMICX2C")  );
    trim(X);
    if( not X.compare("ALH") ) {
      atomic = true;
      atomicX2CType = {true,true};
    } else if( not X.compare("ALU") ) {
      atomic = true;
      atomicX2CType = {true,false};
    } else if( not X.compare("DLH") ) {
      atomic = true;
      atomicX2CType = {false,true};
    } else if( not X.compare("DLU") ) {
      atomic = true;
      atomicX2CType = {false,false};
    } else if( not X.compare("OFF") ){
      atomic = false;
    } else {
      CErr(X + " not a valid QM.ATOMICX2C",out);
    }


    // Parse Finite Width Nuclei
    std::string finiteCore = "DEFAULT";
    OPTOPT( finiteCore = input.getData<std::string>("INTS.FINITENUCLEI"); )
    trim(finiteCore);
    if( not finiteCore.compare("TRUE") )
      hamiltonianOptions.finiteWidthNuc = true;
    else if( not finiteCore.compare("FALSE") )
      hamiltonianOptions.finiteWidthNuc = false;
    else if( not finiteCore.compare("DEFAULT") )
      hamiltonianOptions.finiteWidthNuc = isFourCRef or hamiltonianOptions.OneEScalarRelativity;
    else
      CErr(finiteCore + " not a valid INTS.ALG",out);


    // Parse Integral library
    OPTOPT( hamiltonianOptions.Libcint = input.getData<bool>("INTS.LIBCINT") )

    if (hamiltonianOptions.Libcint) {
      if (basis.forceCart)
        CErr("Libcint + cartesian GTO NYI.");
      if (auto aoi = std::dynamic_pointer_cast<Integrals<double>>(aoints))
        if (auto rieri = std::dynamic_pointer_cast<InCoreAuxBasisRIERI<double>>(aoi->ERI))
          if (rieri->auxbasisSet()->forceCart)
            CErr("Libcint + cartesian GTO NYI.");
    }


    // Parse 4C options
    OPTOPT( hamiltonianOptions.DiracCoulomb = input.getData<bool>("INTS.DIRACCOULOMB") )
    OPTOPT( hamiltonianOptions.DiracCoulomb = input.getData<bool>("INTS.DC") )
    OPTOPT( hamiltonianOptions.DiracCoulombSSSS = input.getData<bool>("INTS.SSSS") )
    OPTOPT( hamiltonianOptions.BareCoulomb = input.getData<bool>("INTS.BARECOULOMB") )
    OPTOPT( hamiltonianOptions.Gauge = input.getData<bool>("INTS.GAUGE") )
    OPTOPT( hamiltonianOptions.Gaunt = input.getData<bool>("INTS.GAUNT") )

    try{
      if ( input.getData<bool>("INTS.BREIT") ) {
        CErr("Breit Hamiltonian is NYI.",out);
        hamiltonianOptions.DiracCoulomb = true;
        hamiltonianOptions.Gaunt = true;
        hamiltonianOptions.Gauge = true;
      }
    } catch(...) {}

    if (not isFourCRef) {

      hamiltonianOptions.BareCoulomb = false;
      hamiltonianOptions.DiracCoulomb = false;
      hamiltonianOptions.DiracCoulombSSSS = false;
      hamiltonianOptions.Gaunt = false;
      hamiltonianOptions.Gauge = false;

    }

    // update IntegralsBase options
    aoints->options_ = hamiltonianOptions;

    // Construct CoreHBuilder
    if( isFourCRef ) {

      if(auto p = std::dynamic_pointer_cast<SingleSlater<double,double>>(ss)) {

        CErr("4C + Real WFN is not a valid option",std::cout);
      } else if(auto p = std::dynamic_pointer_cast<SingleSlater<dcomplex,double>>(ss)) {

        p->coreHBuilder = std::make_shared<FourComponent<dcomplex,double>>(
            *std::dynamic_pointer_cast<Integrals<double>>(aoints),
            mem, mol, basis, hamiltonianOptions);

        p->fockBuilder = std::make_shared<FourCompFock<dcomplex,double>>(hamiltonianOptions);

      } else if(auto p = std::dynamic_pointer_cast<SingleSlater<dcomplex,dcomplex>>(ss)) {

        CErr("4C + GIAO NYI",std::cout);

      } else {

        CErr("Complex INT + Real WFN is not a valid option",std::cout);

      }
    } else if( hamiltonianOptions.OneEScalarRelativity ) {

      if(auto p = std::dynamic_pointer_cast<SingleSlater<dcomplex,double>>(ss)) {

        if (atomic)
          p->coreHBuilder = std::make_shared<AtomicX2C<dcomplex,double>>(
              *std::dynamic_pointer_cast<Integrals<double>>(aoints),
              mem, mol, basis, hamiltonianOptions, atomicX2CType);
        else
          p->coreHBuilder = std::make_shared<X2C<dcomplex,double>>(
              *std::dynamic_pointer_cast<Integrals<double>>(aoints),
              mem, mol, basis, hamiltonianOptions);

        p->fockBuilder = std::make_shared<FockBuilder<dcomplex,double>>(hamiltonianOptions);

      } else if(auto p = std::dynamic_pointer_cast<SingleSlater<double,double>>(ss)) {

        if (not hamiltonianOptions.OneESpinOrbit) {

          if (atomic)
            p->coreHBuilder = std::make_shared<AtomicX2C<double,double>>(
                *std::dynamic_pointer_cast<Integrals<double>>(aoints),
                mem, mol, basis, hamiltonianOptions, atomicX2CType);
          else
            p->coreHBuilder = std::make_shared<X2C<double,double>>(
                *std::dynamic_pointer_cast<Integrals<double>>(aoints),
                mem, mol, basis, hamiltonianOptions);

          p->fockBuilder = std::make_shared<FockBuilder<double,double>>(hamiltonianOptions);

        } else

          CErr("OneE-X2C + Real WFN is not a valid option",std::cout);

      } else if (std::dynamic_pointer_cast<SingleSlater<dcomplex,dcomplex>>(ss)) {

        CErr("X2C + Complex Ints NYI",std::cout);

      } else {

        CErr("Complex INT + Real WFN is not a valid option",std::cout);

      }
    } else {

      if(auto p = std::dynamic_pointer_cast<SingleSlater<double,double>>(ss)) {

        p->coreHBuilder = std::make_shared<NRCoreH<double,double>>(
            *std::dynamic_pointer_cast<Integrals<double>>(aoints), hamiltonianOptions);

        if(isRORef) p->fockBuilder = std::make_shared<ROFock<double,double>>(hamiltonianOptions);
        else p->fockBuilder = std::make_shared<FockBuilder<double,double>>(hamiltonianOptions);

      } else if(auto p = std::dynamic_pointer_cast<SingleSlater<dcomplex,double>>(ss)) {

        p->coreHBuilder = std::make_shared<NRCoreH<dcomplex,double>>(
            *std::dynamic_pointer_cast<Integrals<double>>(aoints), hamiltonianOptions);

        if(isRORef) p->fockBuilder = std::make_shared<ROFock<dcomplex,double>>(hamiltonianOptions);
        else p->fockBuilder = std::make_shared<FockBuilder<dcomplex,double>>(hamiltonianOptions);

      } else if (auto p = std::dynamic_pointer_cast<SingleSlater<dcomplex,dcomplex>>(ss)) {

        p->coreHBuilder = std::make_shared<NRCoreH<dcomplex,dcomplex>>(
            *std::dynamic_pointer_cast<Integrals<dcomplex>>(aoints), hamiltonianOptions);

        if(isRORef) p->fockBuilder = std::make_shared<ROFock<dcomplex,dcomplex>>(hamiltonianOptions);
        else p->fockBuilder = std::make_shared<FockBuilder<dcomplex,dcomplex>>(hamiltonianOptions);

      } else {
        CErr("Complex INT + Real WFN is not a valid option",std::cout);
      }
    }



    // Construct ERIContractions
    if(isFourCRef) {

      size_t nERI4DCB = 0; // Bare-Coulomb
      if( hamiltonianOptions.Gaunt ) nERI4DCB = 23; // Dirac-Coulomb-Gaunt
      else if( hamiltonianOptions.DiracCoulomb ) nERI4DCB = 4; // Dirac-Coulomb


      if(auto p = std::dynamic_pointer_cast<SingleSlater<double,double>>(ss)) {

        CErr("Real INT + Real Four-component WFN NYI",std::cout);

      } else if(auto p = std::dynamic_pointer_cast<SingleSlater<dcomplex,double>>(ss)) {

        std::shared_ptr<TwoEInts<double>> &ERI =
            std::dynamic_pointer_cast<Integrals<double>>(aoints)->ERI;

        if (auto eri_typed = std::dynamic_pointer_cast<InCore4indexERI<double>>(ERI)) {

          ERI = std::make_shared<InCore4indexRelERI<double>>(mem,basis.nBasis,nERI4DCB);

          p->ERI = std::make_shared<InCore4indexRelERIContraction<dcomplex,double>>(*ERI);

        } else if (auto eri_typed = std::dynamic_pointer_cast<DirectERI<double>>(ERI)) {

          p->ERI = std::make_shared<GTODirectRelERIContraction<dcomplex,double>>(*eri_typed);

        } else {
          CErr("Invalid ERInts type for Four-component Wavefunction<dcomplex,double>",std::cout);
        }

      } else if (auto p = std::dynamic_pointer_cast<SingleSlater<dcomplex,dcomplex>>(ss)) {

        CErr("Complex INT Four-component Wavefunction method NYI",std::cout);

      } else {

        CErr("Complex INT + Real WFN is not a valid option",std::cout);

      }
    } else if(auto p = std::dynamic_pointer_cast<SingleSlater<double,double>>(ss)) {

      std::shared_ptr<TwoEInts<double>> ERI =
          std::dynamic_pointer_cast<Integrals<double>>(aoints)->ERI;

      if (auto eri_typed = std::dynamic_pointer_cast<DirectERI<double>>(ERI)) {

        p->ERI = std::make_shared<GTODirectERIContraction<double,double>>(*eri_typed);

      } else if (auto eri_typed = std::dynamic_pointer_cast<InCoreRIERI<double>>(ERI)) {

        p->ERI = std::make_shared<InCoreRIERIContraction<double,double>>(*eri_typed);

      } else if (auto eri_typed = std::dynamic_pointer_cast<InCore4indexERI<double>>(ERI)) {

        p->ERI = std::make_shared<InCore4indexERIContraction<double,double>>(*eri_typed);

      } else {

        CErr("Invalid ERInts type for Wavefunction<double,double>",std::cout);

      }
    } else if(auto p = std::dynamic_pointer_cast<SingleSlater<dcomplex,double>>(ss)) {

      std::shared_ptr<TwoEInts<double>> ERI =
          std::dynamic_pointer_cast<Integrals<double>>(aoints)->ERI;

      if (auto eri_typed = std::dynamic_pointer_cast<DirectERI<double>>(ERI)) {

        p->ERI = std::make_shared<GTODirectERIContraction<dcomplex,double>>(*eri_typed);

      } else if (auto eri_typed = std::dynamic_pointer_cast<InCoreRIERI<double>>(ERI)) {

        p->ERI = std::make_shared<InCoreRIERIContraction<dcomplex,double>>(*eri_typed);

      } else if (auto eri_typed = std::dynamic_pointer_cast<InCore4indexERI<double>>(ERI)) {

        p->ERI = std::make_shared<InCore4indexERIContraction<dcomplex,double>>(*eri_typed);

      } else {

        CErr("Invalid ERInts type for Wavefunction<dcomplex,double>",std::cout);

      }
    } else if (auto p = std::dynamic_pointer_cast<SingleSlater<dcomplex,dcomplex>>(ss)) {

      std::shared_ptr<TwoEInts<dcomplex>> ERI =
          std::dynamic_pointer_cast<Integrals<dcomplex>>(aoints)->ERI;

      if (auto eri_typed = std::dynamic_pointer_cast<InCore4indexERI<dcomplex>>(ERI)) {

        p->ERI = std::make_shared<InCore4indexERIContraction<dcomplex,dcomplex>>(*eri_typed);

      } else if (auto eri_typed = std::dynamic_pointer_cast<DirectERI<dcomplex>>(ERI)) {

        p->ERI = std::make_shared<GIAODirectERIContraction>(*eri_typed);

      } else {

        CErr("Invalid ERInts type for Wavefunction<dcomplex,dcomplex>",std::cout);

      }
    } else {

      CErr("Complex INT + Real WFN is not a valid option",std::cout);

    }





    out << hamiltonianOptions << std::endl;





    return ss;

  }; // CQSingleSlaterOptions






  /**
   *  Outputs relevant information for the HamiltonianOptions struct
   *  to a specified output.
   *
   *  \param [in/out] out     Ouput device
   *  \param [in]     options HamiltonianOptions object to output.
   */
  std::ostream& operator<<(std::ostream &out, const HamiltonianOptions &options) {

    out << std::endl << "Hamiltonian Options";
    out << ":" << std::endl << BannerTop << std::endl << std::endl;


    const int fieldNameWidth(40);

    out << "  " << std::setw(fieldNameWidth) << "Integral:" << std::endl;
    out << bannerMid << std::endl;
    out << "  " << std::setw(fieldNameWidth) << "Basis Type:";
    switch (options.basisType) {
    case REAL_GTO:
      out << "REAL_GTO";
      break;
    case COMPLEX_GIAO:
      out << "COMPLEX_GIAO";
      break;
    case COMPLEX_GTO:
      out << "COMPLEX_GTO";
      break;
    }
    out << std::endl;
    out << "  " << std::setw(fieldNameWidth) << "Finite Width Nuclei:"
        << (options.finiteWidthNuc ? "True" : "False") << std::endl;
    out << "  " << std::setw(fieldNameWidth) << "Using Libcint:"
        << (options.Libcint ? "True" : "False") << std::endl;
    out << std::endl;


    out << "  " << std::setw(fieldNameWidth) << "One-Component Options:" << std::endl;
    out << bannerMid << std::endl;
    out << "  " << std::setw(fieldNameWidth) << "Perturbative Scalar Relativity:"
        << (options.PerturbativeScalarRelativity ? "On" : "Off") << std::endl;
    out << "  " << std::setw(fieldNameWidth) << "Perturbative Spin-orbit Relativity:"
        << (options.PerturbativeSpinOrbit ? "On" : "Off") << std::endl;
    out << std::endl;


    out << "  " << std::setw(fieldNameWidth) << "Two-Component Options:" << std::endl;
    out << bannerMid << std::endl;
    out << "  " << std::setw(fieldNameWidth) << "One-Electron Scalar Relativity:"
        << (options.OneEScalarRelativity ? "On" : "Off") << std::endl;
    out << "  " << std::setw(fieldNameWidth) << "One-Electron Spin-orbit Relativity:"
        << (options.OneESpinOrbit ? "On" : "Off") << std::endl;
    out << "  " << std::setw(fieldNameWidth) << "Boettger Spin-orbit Scaling:"
        << (options.Boettger ? "On" : "Off") << std::endl;
    out << "  " << std::setw(fieldNameWidth) << "Atomic Mean Field Spin-orbit:"
        << (options.AtomicMeanField ? "On" : "Off") << std::endl;
    out << std::endl;


    out << "  " << std::setw(fieldNameWidth) << "Four-Component Options:" << std::endl;
    out << bannerMid << std::endl;
    out << "  " << std::setw(fieldNameWidth) << "Bare Coulomb Term:"
        << (options.BareCoulomb ? "On" : "Off") << std::endl;
    out << "  " << std::setw(fieldNameWidth) << "Dirac Coulomb Term:"
        << (options.DiracCoulomb ? "On" : "Off") << std::endl;
    out << "  " << std::setw(fieldNameWidth) << "Dirac Coulomb SSSS Term:"
        << (options.DiracCoulombSSSS ? "On" : "Off") << std::endl;
    out << "  " << std::setw(fieldNameWidth) << "Gaunt Term:"
        << (options.Gaunt ? "On" : "Off") << std::endl;
    out << "  " << std::setw(fieldNameWidth) << "Gauge Term:"
        << (options.Gauge ? "On" : "Off") << std::endl;


    out << std::endl << BannerEnd << std::endl;

    return out; // Return std::ostream reference

  }

}; // namespace ChronusQ
