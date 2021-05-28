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
#include <physcon.hpp>
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
#include <particleintegrals/twopints.hpp>
#include <particleintegrals/twopints/gtodirecttpi.hpp>
#include <particleintegrals/twopints/incore4indextpi.hpp>
#include <particleintegrals/twopints/giaodirecteri.hpp>
#include <particleintegrals/twopints/incoreritpi.hpp>
#include <particleintegrals/twopints/incore4indexreleri.hpp>
#include <particleintegrals/twopints/gtodirectreleri.hpp>

#include <singleslater/neoss.hpp>

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
      "NUCREFERENCE",
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
   * \brief Parse the SingleSlater Referece information using
   * the input file.
   * 
   * \param [in]  out     Output device for data / error output.
   * \param [in]  tokens  Vector of string that contains reference information
   *
   * \returns RefOptions object that stores all reference options.
   *
   */
  RefOptions parseRef(std::ostream &out, 
    Molecule &mol, std::vector<std::string> &tokens) {

    // Initialize return
    RefOptions ref;

    // Determine the Real/Complex flag
    if( tokens.size() == 1 )      ref.RCflag = "AUTO";
    else if( tokens.size() == 2 ) ref.RCflag = tokens[0];
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
      "EPC17",
      "EPC19"
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
    if ( std::find(rawRefs.begin(),rawRefs.end(),refString) != rawRefs.end() )
      ref.refType = isRawRef;
    else if ( std::find(RRefs.begin(),RRefs.end(),refString) != RRefs.end() )
      ref.refType = isRRef;
    else if ( std::find(URefs.begin(),URefs.end(),refString) != URefs.end() )
      ref.refType = isURef;
    else if ( std::find(RORefs.begin(),RORefs.end(),refString) != RORefs.end() )
      ref.refType = isRORef;
    else if ( std::find(GRefs.begin(),GRefs.end(),refString) != GRefs.end() )
      ref.refType = isGRef;
    else if ( std::find(TwoCRefs.begin(),TwoCRefs.end(),refString) != TwoCRefs.end() )
      ref.refType = isTwoCRef;
    else if ( std::find(X2CRefs.begin(),X2CRefs.end(),refString) != X2CRefs.end() )
      ref.refType = isX2CRef;
    else if ( std::find(FourCRefs.begin(),FourCRefs.end(),refString) != FourCRefs.end() )
      ref.refType = isFourCRef;
    else 
      CErr(refString + " is not a valid QM.REFERENCE",out);


    // Cleanup the reference string
    if( ref.refType != isRawRef )
      if( ref.refType == isX2CRef )                       
        refString.erase(0,3);
      else if( ref.refType == isFourCRef or ref.refType == isTwoCRef ) 
        refString.erase(0,2);
      else                                                
        refString.erase(0,1);


    // Handle KS related queries
    ref.isKSRef = 
      std::find(KSRefs.begin(),KSRefs.end(),refString) != KSRefs.end();

    if( ref.isKSRef )
      ref.funcName = refString;

    // Raw reference
    if( ref.refType == isRawRef ) {
      out << "  *** Auto-determination of reference: " << refString << " -> ";
      ref.iCS = mol.multip == 1;

      if(ref.iCS) out << "R" << refString;
      else        out << "U" << refString;

      out << " ***" << std::endl;
      
    } else if( ref.refType == isRRef )
      if( mol.multip != 1 )
        CErr("Spin-Restricted Reference only valid for singlet spin multiplicities",out);
      else
        ref.iCS = true;
    else if( ref.refType == isURef or ref.refType == isRORef )
      ref.iCS = false;
    else if( ref.refType == isGRef or ref.refType == isTwoCRef or ref.refType == isX2CRef ) {
      ref.iCS = false; ref.nC = 2;
    }
    else if( ref.refType == isFourCRef ) {
      ref.iCS = false; ref.nC = 4;
    }

    // Determine Real/Complex if need be
    if(not ref.RCflag.compare("AUTO") ) {
      if( ref.nC == 2 or ref.nC == 4 )
        ref.RCflag = "COMPLEX";
      else
        ref.RCflag = "REAL";

      out << "  *** Auto-determination of wave function field: AUTO -> " 
          << ref.RCflag << " ***" << std::endl;
    }

    out << "\n\n";

    return ref;
  }

  /**
   * \brief Construct a list of DFT Functional objects based on input name.
   * 
   * \param [in]  funcName   Input functional name
   * \param [out] funcList   Vector that stores constructed DFT functional object
   *
   */
  void buildFunclist(std::vector<std::shared_ptr<DFTFunctional>> &funcList,
    std::string funcName) {

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

    if (not funcName.compare("EPC17"))
      funcList.push_back(
        std::dynamic_pointer_cast<DFTFunctional>(
          std::make_shared<EPC17>("EPC-17")
        )
      );

    if (not funcName.compare("EPC19"))
      funcList.push_back(
        std::dynamic_pointer_cast<DFTFunctional>(
          std::make_shared<EPC19>("EPC-19")
        )
      );
  }

  /**
   * \brief Parse the Integration information using
   * the input file.
   * 
   * \param [in]  out       Output device for data / error output.
   * \param [in]  input     Input file datastructure
   * \param [out] intParam  Object that stores parsed info
   *
   */
  void parseIntParam(std::ostream &out, CQInputFile &input, 
    IntegrationParam &intParam) {

    if( input.containsSection("DFTINT") ) {

      OPTOPT( intParam.epsilon = input.getData<double>("DFTINT.EPS")  );
      OPTOPT( intParam.nAng    = input.getData<size_t>("DFTINT.NANG") );
      OPTOPT( intParam.nRad    = input.getData<size_t>("DFTINT.NRAD") );
      OPTOPT( intParam.nRadPerBatch    = input.getData<size_t>("DFTINT.NMACRO") );

    }


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

  /**
   *  \brief Parse Hamiltonian options using the input 
   *  file.
   *
   *  \param [in] out    Output device for data / error output.
   *  \param [in] input  Input file datastructure
   *  \param [in] basis  Basis Set
   *  \param [in] aoints AOIntegrals object for SingleSlater
   *                     construction
   *  \param [in]  refOptions Object that stores reference info
   *  \param [out] hamiltonianOptions Objects that get parsed
   *
   *
   */
  void parseHamiltonianOptions(std::ostream &out, CQInputFile &input, 
    BasisSet &basis, std::shared_ptr<IntegralsBase> aoints,
    RefOptions &refOptions, HamiltonianOptions &hamiltonianOptions, std::string section) {

    // Parse hamiltonianOptions
    hamiltonianOptions.basisType = basis.basisType;

    std::string X;

    // Parse X2C option
    // X2CType = off (default), spinfree, onee, twoe
    X = "DEFAULT";
    OPTOPT( X = input.getData<std::string>(section + ".X2CTYPE")  );
    trim(X);
    if( not X.compare("SPINFREE") ) {

      hamiltonianOptions.OneEScalarRelativity = true;
      hamiltonianOptions.OneESpinOrbit = false;
      hamiltonianOptions.Boettger = false;
      hamiltonianOptions.AtomicMeanField = false;

    } else if( not X.compare("ONEE") or not X.compare("ONEELECTRON")
               or ( not X.compare("DEFAULT") and refOptions.refType == isX2CRef ) ) {
      // Legacy X2C- reference is equilvalent to 2C- reference + OneE-X2C

      hamiltonianOptions.OneEScalarRelativity = true;
      hamiltonianOptions.OneESpinOrbit = true;
      hamiltonianOptions.Boettger = true;
      hamiltonianOptions.AtomicMeanField = false;

    } else if( not X.compare("TWOE") or not X.compare("TWOELECTRON")) {

      CErr(X + " NYI",out);

    } else if( not X.compare("OFF")
               or ( not X.compare("DEFAULT") and refOptions.refType != isX2CRef ) ) {

      if ( refOptions.refType == isFourCRef ) {

        hamiltonianOptions.OneEScalarRelativity = true;
        hamiltonianOptions.OneESpinOrbit = true;

      } else {

        hamiltonianOptions.OneEScalarRelativity = false;
        hamiltonianOptions.OneESpinOrbit = false;

      }

      hamiltonianOptions.Boettger = false;
      hamiltonianOptions.AtomicMeanField = false;

    } else  {

      CErr(X + " not a valid " + section + ".X2CTYPE",out);

    }



    // Parse one-electron spin-orbie scaling option
    // SpinOrbitScaling  = noscaling, boettger (dafault), atomicmeanfield (amfi)
    X = "DEFAULT"; // Unspecified value
    OPTOPT( X = input.getData<std::string>(section + ".SPINORBITSCALING")  );
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

      if ( hamiltonianOptions.OneESpinOrbit and refOptions.refType != isFourCRef) {

        hamiltonianOptions.Boettger = true;
        hamiltonianOptions.AtomicMeanField = false;

      } else {

        hamiltonianOptions.Boettger = false;
        hamiltonianOptions.AtomicMeanField = false;

      }

    } else {

      CErr(X + " not a valid " + section + ".X2CTYPE",out);

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
      hamiltonianOptions.finiteWidthNuc = refOptions.refType == isFourCRef or hamiltonianOptions.OneEScalarRelativity;
    else
      CErr(finiteCore + " not a valid INTS.ALG",out);


    // Parse Integral library
    OPTOPT( hamiltonianOptions.Libcint = input.getData<bool>("INTS.LIBCINT") )

    if (hamiltonianOptions.Libcint) {
      if (basis.forceCart)
        CErr("Libcint + cartesian GTO NYI.");
      if (auto aoi = std::dynamic_pointer_cast<Integrals<double>>(aoints))
        if (auto rieri = std::dynamic_pointer_cast<InCoreAuxBasisRIERI<double>>(aoi->TPI))
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

    if (refOptions.refType != isFourCRef) {

      hamiltonianOptions.BareCoulomb = false;
      hamiltonianOptions.DiracCoulomb = false;
      hamiltonianOptions.DiracCoulombSSSS = false;
      hamiltonianOptions.Gaunt = false;
      hamiltonianOptions.Gauge = false;

    }

  }

  /**
   *  \brief Parse atomic X2C options using the input 
   *  file.
   *
   *  \param [in]  out            Output device for data / error output.
   *  \param [in]  input          Input file datastructure
   *  \param [out] atomicX2CType  Objects that get parsed
   *
   *  \returns boolean that tells whether atomic X2C is used.
   *
   */
  bool parseAtomicType(std::ostream &out, CQInputFile &input, 
    ATOMIC_X2C_TYPE &atomicX2CType, std::string section) {

    // Parse Atomic X2C option
    // AtomicX2C  = ALH, ALU, DLH, DLU, OFF (default)
    bool atomic = false;
    std::string X = "OFF";
    OPTOPT( X = input.getData<std::string>(section + ".ATOMICX2C")  );
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
      CErr(X + " not a valid " + section + ".ATOMICX2C",out);
    }

    return atomic;

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
  std::shared_ptr<SingleSlaterBase> buildSingleSlater(
    std::ostream &out, CQInputFile &input,
    CQMemManager &mem, Molecule &mol, BasisSet &basis,
    std::shared_ptr<IntegralsBase> aoints, Particle p, std::string section) {

    out << "  *** Parsing " << section << ".REFERENCE options ***\n";

    // Initialize HamiltonianOptions
    HamiltonianOptions hamiltonianOptions;

    // Initialize ReferenceOptions
    RefOptions refOptions;

    // Initialize Atomic X2C options
    ATOMIC_X2C_TYPE atomicX2CType = {false,false};

    // Attempt to find reference
    std::string reference;
    try { 
      reference = input.getData<std::string>(section + ".REFERENCE");
    } catch(...) {
      CErr(section + ".REFERENCE Keyword not found!",out);
    }

    // Digest reference string
    // Trim Spaces
    trim(reference);

    // Split into tokens
    std::vector<std::string> tokens;
    split(tokens,reference);
    for(auto &X : tokens) trim(X);

    // Parse reference information
    refOptions = parseRef(out,mol,tokens);

    // Build Functional List
    std::vector<std::shared_ptr<DFTFunctional>> funcList;
    if( refOptions.isKSRef )
      buildFunclist(funcList, refOptions.funcName);

    // Sanity Checks
    bool isGIAO = basis.basisType == COMPLEX_GIAO;

    if( refOptions.nC == 2 and not refOptions.RCflag.compare("REAL") )
      CErr("Real + Two-Component not valid",out);

    if( refOptions.nC == 4 and not refOptions.RCflag.compare("REAL") )
      CErr("Real + Four-Component not valid",out);

    if( isGIAO and not refOptions.RCflag.compare("REAL") )
      CErr("Real + GIAO not valid",out);

    if( isGIAO and refOptions.isKSRef )
      CErr("KS + GIAO not valid",out);

    if( isGIAO and refOptions.refType == isX2CRef )
      CErr("X2C + GIAO not valid",out);


    // Override core hamiltoninan type for X2C
      

    // FIXME: Should put this somewhere else
    // Parse KS integration

    IntegrationParam intParam;

    if( refOptions.isKSRef )
     parseIntParam(out, input, intParam);


  #define KS_LIST(T) \
    refOptions.funcName,funcList,MPI_COMM_WORLD,intParam,mem,mol,basis,dynamic_cast<Integrals<T>&>(*aoints),refOptions.nC,refOptions.iCS,p

  #define HF_LIST(T) \
    MPI_COMM_WORLD,mem,mol,basis,dynamic_cast<Integrals<T>&>(*aoints),refOptions.nC,refOptions.iCS,p


    // Construct the SS object
    std::shared_ptr<SingleSlaterBase> ss;

    if( not refOptions.RCflag.compare("REAL") )
      if( refOptions.isKSRef )
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<KohnSham<double,double>>( KS_LIST(double) )
          );
      else if(refOptions.refType == isRORef)
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

    else if( not refOptions.RCflag.compare("COMPLEX") and not isGIAO )
      if( refOptions.isKSRef and refOptions.refType == isX2CRef)
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<KohnSham<dcomplex,double>>(
              "Exact Two Component", "X2C-", KS_LIST(double)
            )
          );
      else if( refOptions.isKSRef )
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<KohnSham<dcomplex,double>>( KS_LIST(double) )
          );
      else if( refOptions.refType == isX2CRef )
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<dcomplex,double>>(
              "Exact Two Component","X2C-",HF_LIST(double)
            )
          );
      else if( refOptions.refType == isRORef )
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<dcomplex,double>>(
              "Complex Restricted Open-shell Hartree-Fock", "C-ROHF", HF_LIST(double)
            )
          );
      else if( refOptions.refType == isFourCRef )
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
      if( refOptions.isKSRef and refOptions.refType == isX2CRef )
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<KohnSham<dcomplex,dcomplex>>(
              "Exact Two Component", "X2C-", KS_LIST(dcomplex)
            )
          );
      else if( refOptions.isKSRef )
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<KohnSham<dcomplex,dcomplex>>( KS_LIST(dcomplex) )
          );
      else if( refOptions.refType == isX2CRef )
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<dcomplex,dcomplex>>(
              "Exact Two Component","X2C-",HF_LIST(dcomplex)
            )
          );
      else if( refOptions.refType == isRORef )
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<dcomplex,dcomplex>>(
              "Complex Restricted Open-shell Hartree-Fock", "C-ROHF", HF_LIST(dcomplex)
            )
          );
      else if( refOptions.refType == isFourCRef )
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
    parseHamiltonianOptions(out,input,basis,aoints,refOptions,hamiltonianOptions,section);
    hamiltonianOptions.particle = p;

    // Parse Atomic X2C
    bool atomic = parseAtomicType(out,input,atomicX2CType,section);

    // update IntegralsBase options
    aoints->options_ = hamiltonianOptions;

    // Construct CoreHBuilder
    if( refOptions.refType == isFourCRef ) {

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

        if(refOptions.refType == isRORef) p->fockBuilder = std::make_shared<ROFock<double,double>>(hamiltonianOptions);
        else p->fockBuilder = std::make_shared<FockBuilder<double,double>>(hamiltonianOptions);

      } else if(auto p = std::dynamic_pointer_cast<SingleSlater<dcomplex,double>>(ss)) {

        p->coreHBuilder = std::make_shared<NRCoreH<dcomplex,double>>(
            *std::dynamic_pointer_cast<Integrals<double>>(aoints), hamiltonianOptions);

        if(refOptions.refType == isRORef) p->fockBuilder = std::make_shared<ROFock<dcomplex,double>>(hamiltonianOptions);
        else p->fockBuilder = std::make_shared<FockBuilder<dcomplex,double>>(hamiltonianOptions);

      } else if (auto p = std::dynamic_pointer_cast<SingleSlater<dcomplex,dcomplex>>(ss)) {

        p->coreHBuilder = std::make_shared<NRCoreH<dcomplex,dcomplex>>(
            *std::dynamic_pointer_cast<Integrals<dcomplex>>(aoints), hamiltonianOptions);

        if(refOptions.refType == isRORef) p->fockBuilder = std::make_shared<ROFock<dcomplex,dcomplex>>(hamiltonianOptions);
        else p->fockBuilder = std::make_shared<FockBuilder<dcomplex,dcomplex>>(hamiltonianOptions);

      } else {
        CErr("Complex INT + Real WFN is not a valid option",std::cout);
      }
    }



    // Construct ERIContractions
    if(refOptions.refType == isFourCRef) {

      size_t nERI4DCB = 0; // Bare-Coulomb
      if( hamiltonianOptions.Gaunt ) nERI4DCB = 23; // Dirac-Coulomb-Gaunt
      else if( hamiltonianOptions.DiracCoulomb ) nERI4DCB = 4; // Dirac-Coulomb


      if(auto p = std::dynamic_pointer_cast<SingleSlater<double,double>>(ss)) {

        CErr("Real INT + Real Four-component WFN NYI",std::cout);

      } else if(auto p = std::dynamic_pointer_cast<SingleSlater<dcomplex,double>>(ss)) {

        std::shared_ptr<TwoPInts<double>> &TPI =
            std::dynamic_pointer_cast<Integrals<double>>(aoints)->TPI;

        if (auto tpi_typed = std::dynamic_pointer_cast<InCore4indexTPI<double>>(TPI)) {

          TPI = std::make_shared<InCore4indexRelERI<double>>(mem,basis.nBasis,nERI4DCB);

          p->TPI = std::make_shared<InCore4indexRelERIContraction<dcomplex,double>>(*TPI);

        } else if (auto tpi_typed = std::dynamic_pointer_cast<DirectTPI<double>>(TPI)) {

          p->TPI = std::make_shared<GTODirectRelERIContraction<dcomplex,double>>(*tpi_typed);

        } else {
          CErr("Invalid TPInts type for Four-component Wavefunction<dcomplex,double>",std::cout);
        }

      } else if (auto p = std::dynamic_pointer_cast<SingleSlater<dcomplex,dcomplex>>(ss)) {

        CErr("Complex INT Four-component Wavefunction method NYI",std::cout);

      } else {

        CErr("Complex INT + Real WFN is not a valid option",std::cout);

      }
    } else if(auto p = std::dynamic_pointer_cast<SingleSlater<double,double>>(ss)) {

      std::shared_ptr<TwoPInts<double>> TPI =
          std::dynamic_pointer_cast<Integrals<double>>(aoints)->TPI;

      if (auto tpi_typed = std::dynamic_pointer_cast<DirectTPI<double>>(TPI)) {

        p->TPI = std::make_shared<GTODirectTPIContraction<double,double>>(*tpi_typed);

      } else if (auto tpi_typed = std::dynamic_pointer_cast<InCoreRITPI<double>>(TPI)) {

        p->TPI = std::make_shared<InCoreRITPIContraction<double,double>>(*tpi_typed);

      } else if (auto tpi_typed = std::dynamic_pointer_cast<InCore4indexTPI<double>>(TPI)) {

        p->TPI = std::make_shared<InCore4indexTPIContraction<double,double>>(*tpi_typed);

      } else {

        CErr("Invalid TPInts type for Wavefunction<double,double>",std::cout);

      }
    } else if(auto p = std::dynamic_pointer_cast<SingleSlater<dcomplex,double>>(ss)) {

      std::shared_ptr<TwoPInts<double>> TPI =
          std::dynamic_pointer_cast<Integrals<double>>(aoints)->TPI;

      if (auto tpi_typed = std::dynamic_pointer_cast<DirectTPI<double>>(TPI)) {

        p->TPI = std::make_shared<GTODirectTPIContraction<dcomplex,double>>(*tpi_typed);

      } else if (auto tpi_typed = std::dynamic_pointer_cast<InCoreRITPI<double>>(TPI)) {

        p->TPI = std::make_shared<InCoreRITPIContraction<dcomplex,double>>(*tpi_typed);

      } else if (auto tpi_typed = std::dynamic_pointer_cast<InCore4indexTPI<double>>(TPI)) {

        p->TPI = std::make_shared<InCore4indexTPIContraction<dcomplex,double>>(*tpi_typed);

      } else {

        CErr("Invalid TPInts type for Wavefunction<dcomplex,double>",std::cout);

      }
    } else if (auto p = std::dynamic_pointer_cast<SingleSlater<dcomplex,dcomplex>>(ss)) {

      std::shared_ptr<TwoPInts<dcomplex>> TPI =
          std::dynamic_pointer_cast<Integrals<dcomplex>>(aoints)->TPI;

      if (auto tpi_typed = std::dynamic_pointer_cast<InCore4indexTPI<dcomplex>>(TPI)) {

        p->TPI = std::make_shared<InCore4indexTPIContraction<dcomplex,dcomplex>>(*tpi_typed);

      } else if (auto tpi_typed = std::dynamic_pointer_cast<DirectTPI<dcomplex>>(TPI)) {

        p->TPI = std::make_shared<GIAODirectERIContraction>(*tpi_typed);

      } else {

        CErr("Invalid TPInts type for Wavefunction<dcomplex,dcomplex>",std::cout);

      }
    } else {

      CErr("Complex INT + Real WFN is not a valid option",std::cout);

    }





    out << hamiltonianOptions << std::endl;





    return ss;

  }; // CQSingleSlaterOptions


  /**
   *  \brief Construct a NEOSingleSlater object using the input 
   *  file.
   *
   *  \param [in] out         Output device for data / error output.
   *  \param [in] input       Input file datastructure
   *  \param [in] elec_aoints AOIntegrals object for NEOSingleSlater
   *                          construction (electrons)
   *  \param [in] prot_aoints AOIntegrals object for NEOSingleSlater
   *                          construction (protons)
   *  \param [in] ep_aoints   AOIntegrals object for NEOSingleSlater
   *                          construction (electron-proton)
   *
   *
   *  \returns shared_ptr to a NEOSingleSlater object
   *    constructed from the input options.
   *
   */ 
  std::vector<std::shared_ptr<SingleSlaterBase>> CQNEOSingleSlaterOptions(
    std::ostream &out, CQInputFile &input,
    CQMemManager &mem, Molecule &mol,
    BasisSet &ebasis, BasisSet &pbasis,
    std::shared_ptr<IntegralsBase> eaoints, 
    std::shared_ptr<IntegralsBase> paoints,
    std::shared_ptr<IntegralsBase> epaoints) {

    // Initialize electron ReferenceOptions
    RefOptions elec_refOptions;

    // Initialize ReferenceOptions
    RefOptions prot_refOptions;

    out << "  *** Parsing QM.REFERENCE options ***\n";

    // Attempt to find reference
    std::string reference;
    try { 
      reference = input.getData<std::string>("QM.REFERENCE");
    } catch(...) {
      CErr("QM.REFERENCE Keyword not found!",out);
    }

    out << "  *** Parsing QM.NUCREFERENCE options ***\n";

    // Attempt to find proton reference 
    std::string preference;
    try {
      preference = input.getData<std::string>("QM.NUCREFERENCE");
    } catch(...) {
      CErr("QM.NUCREFERENCE Keyword not found in NEO calculation!",out);
    }

    // Digest reference string
    // Trim Spaces
    trim(reference);

    // Split into tokens
    std::vector<std::string> tokens;
    split(tokens,reference);
    for(auto &X : tokens) trim(X);

    // Trim Spaces for proton
    trim(preference);

    // Split into tokens
    std::vector<std::string> ptokens;
    split(ptokens,preference);
    for(auto &X : ptokens) trim(X);

    // Parse electron reference information
    elec_refOptions = parseRef(out,mol,tokens);

    // Parse proton reference information
    prot_refOptions = parseRef(out,mol,ptokens);

    // Throw an error if not a valid reference keyword
    if( elec_refOptions.refType != isRawRef and elec_refOptions.refType != isRRef and elec_refOptions.refType != isURef )
      CErr(tokens.back() + " is not a valid QM.REFERENCE for NEO",out);

    // Throw an error if not a valid proton reference keyword
    if( prot_refOptions.refType != isRawRef and prot_refOptions.refType != isURef )
      CErr(ptokens.back() + " is not a valid QM.NUCREFERENCE for NEO",out);

    if ( (not elec_refOptions.isKSRef and prot_refOptions.isKSRef) 
         or (elec_refOptions.isKSRef and not prot_refOptions.isKSRef) )
      CErr("Mixing DFT with HF is not allowed",out);

    // Sanity check for electron functional names
    std::vector<std::shared_ptr<DFTFunctional>> funcList;
    if( elec_refOptions.isKSRef )
      if (not elec_refOptions.funcName.compare("EPC17") or not elec_refOptions.funcName.compare("EPC19"))
        CErr("Find invalid KS functional",out);
      else
        buildFunclist(funcList, elec_refOptions.funcName);
        

    // Sanity check for epc functional names
    std::vector<std::shared_ptr<DFTFunctional>> epc_funcList;
    if( prot_refOptions.isKSRef )
      if (prot_refOptions.funcName.compare("EPC17") and prot_refOptions.funcName.compare("EPC19"))
        CErr("Find invalid EPC functional",out);
      else
         buildFunclist(epc_funcList, prot_refOptions.funcName);       

    // Manually set proton reference
    prot_refOptions.iCS = false;
    prot_refOptions.nC = 1;
    if (prot_refOptions.refType == isRawRef)
      prot_refOptions.refType == isURef;
    
    // Sanity Checks
    bool eisGIAO = ebasis.basisType == COMPLEX_GIAO;
    bool pisGIAO = pbasis.basisType == COMPLEX_GIAO;

    if( eisGIAO or pisGIAO )
      CErr("GIAO is not supported by NEO",out);


    // Override core hamiltoninan type for X2C
      

    // FIXME: Should put this somewhere else
    // Parse KS integration

    IntegrationParam intParam;

    if( elec_refOptions.isKSRef )
     parseIntParam(out, input, intParam);

  // electron 
  Particle elec = {-1.0, 1.0};

  // proton
  Particle prot = {1.0, ProtMassPerE};


  #define eKS_LIST(T) \
    prot_refOptions.funcName,epc_funcList,elec_refOptions.funcName,funcList,MPI_COMM_WORLD,intParam,mem,mol,ebasis,dynamic_cast<Integrals<T>&>(*eaoints),elec_refOptions.nC,elec_refOptions.iCS,elec

  #define eHF_LIST(T) \
    MPI_COMM_WORLD,mem,mol,ebasis,dynamic_cast<Integrals<T>&>(*eaoints),elec_refOptions.nC,elec_refOptions.iCS,elec

  #define pKS_LIST(T) \
    prot_refOptions.funcName,epc_funcList,prot_refOptions.funcName,epc_funcList,MPI_COMM_WORLD,intParam,mem,mol,pbasis,dynamic_cast<Integrals<T>&>(*paoints),1,false,prot

  #define pHF_LIST(T) \
    MPI_COMM_WORLD,mem,mol,pbasis,dynamic_cast<Integrals<T>&>(*paoints),1,false,prot


    // Construct the electron SS object
    std::shared_ptr<SingleSlaterBase> elec_ss;

    // Construct the proton SS object
    std::shared_ptr<SingleSlaterBase> prot_ss;

    if( not elec_refOptions.RCflag.compare("REAL") )
      if(  elec_refOptions.isKSRef ) {
        
        // NEO Kohn-Sham object for electron
        auto neo_ess = std::make_shared<NEOKohnSham<double,double>>( eKS_LIST(double) );

        // neo single slater object for proton
        auto neo_pss = std::make_shared<NEOKohnSham<double,double>>( pKS_LIST(double) );

        // connects electron and proton
        neo_ess->getAux(neo_pss);
        neo_pss->getAux(neo_ess);

        // cast to singleslater base class
        elec_ss = std::dynamic_pointer_cast<SingleSlaterBase>(neo_ess);
        prot_ss = std::dynamic_pointer_cast<SingleSlaterBase>(neo_pss);

      }
      else {

        // neo single slater object for electron
        auto neo_ess = std::make_shared<NEOHartreeFock<double,double>>( eHF_LIST(double) );

        // neo single slater object for proton
        auto neo_pss = std::make_shared<NEOHartreeFock<double,double>>( pHF_LIST(double) );

        // connects electron and proton
        neo_ess->getAux(neo_pss);
        neo_pss->getAux(neo_ess);

        // cast to singleslater base class
        elec_ss = std::dynamic_pointer_cast<SingleSlaterBase>(neo_ess);
        prot_ss = std::dynamic_pointer_cast<SingleSlaterBase>(neo_pss);
      }

    else if( not  elec_refOptions.RCflag.compare("COMPLEX") and not eisGIAO and not pisGIAO)
      if(  elec_refOptions.isKSRef ) {
        // single slater object for electron
        auto neo_ess = std::make_shared<NEOKohnSham<dcomplex,double>> ( eKS_LIST(double) );

        // single slater object for proton
        auto neo_pss = std::make_shared<NEOKohnSham<dcomplex,double>> ( pKS_LIST(double) );

        // connects electron and proton
        neo_ess->getAux(neo_pss);
        neo_pss->getAux(neo_ess);

        // cast to singleslater base class
        elec_ss = std::dynamic_pointer_cast<SingleSlaterBase>(neo_ess);
        prot_ss = std::dynamic_pointer_cast<SingleSlaterBase>(neo_pss);

      }
      else {
        // single slater object for electron
        auto neo_ess = std::make_shared<NEOHartreeFock<dcomplex,double>> ( eHF_LIST(double) );

        // single slater object for proton
        auto neo_pss = std::make_shared<NEOHartreeFock<dcomplex,double>> ( pHF_LIST(double) );

        // connects electron and proton
        neo_ess->getAux(neo_pss);
        neo_pss->getAux(neo_ess);

        // cast to singleslater base class
        elec_ss = std::dynamic_pointer_cast<SingleSlaterBase>(neo_ess);
        prot_ss = std::dynamic_pointer_cast<SingleSlaterBase>(neo_pss);
      }
    else
      CErr("GIAO is not supported by NEO", out);

    // overall Hamiltonian options
    HamiltonianOptions hamiltonianOptions;

    // Parse hamiltonianOptions
    hamiltonianOptions.basisType = ebasis.basisType;

    std::string X;

    // Parse X2C option
    hamiltonianOptions.OneEScalarRelativity = false;
    hamiltonianOptions.OneESpinOrbit = false;
    hamiltonianOptions.Boettger = false;
    hamiltonianOptions.AtomicMeanField = false;



    // Parse one-electron spin-orbie scaling option
    // SpinOrbitScaling  = noscaling, boettger (dafault), atomicmeanfield (amfi)
    hamiltonianOptions.Boettger = false;
    hamiltonianOptions.AtomicMeanField = false;


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
    hamiltonianOptions.finiteWidthNuc = false;

    // Parse Integral library
    OPTOPT( hamiltonianOptions.Libcint = input.getData<bool>("INTS.LIBCINT") )

    // Parse 4C options
    hamiltonianOptions.BareCoulomb = false;
    hamiltonianOptions.DiracCoulomb = false;
    hamiltonianOptions.DiracCoulombSSSS = false;
    hamiltonianOptions.Gaunt = false;
    hamiltonianOptions.Gauge = false;


    // electron and proton HamiltonianOptions
    HamiltonianOptions elec_aoiOptions = hamiltonianOptions;
    HamiltonianOptions prot_aoiOptions = hamiltonianOptions;

    elec_aoiOptions.particle = elec;
    prot_aoiOptions.particle = prot;

    // Construct CoreHBuilder
    if(auto p = std::dynamic_pointer_cast<NEOSingleSlater<double,double>>(elec_ss)) {
      p->coreHBuilder = std::make_shared<NRCoreH<double,double>>(
          *std::dynamic_pointer_cast<Integrals<double>>(eaoints), elec_aoiOptions);
      p->fockBuilder = std::make_shared<FockBuilder<double,double>>(elec_aoiOptions);
      if (auto q = std::dynamic_pointer_cast<NEOSingleSlater<double,double>>(prot_ss)) {
        q->coreHBuilder = std::make_shared<NRCoreH<double,double>>(
            *std::dynamic_pointer_cast<Integrals<double>>(paoints), prot_aoiOptions);
        q->fockBuilder = std::make_shared<FockBuilder<double,double>>(prot_aoiOptions);
      }
    } else if(auto p = std::dynamic_pointer_cast<NEOSingleSlater<dcomplex,double>>(elec_ss)) {
      p->coreHBuilder = std::make_shared<NRCoreH<dcomplex,double>>(
          *std::dynamic_pointer_cast<Integrals<double>>(eaoints), elec_aoiOptions);
      p->fockBuilder = std::make_shared<FockBuilder<dcomplex,double>>(elec_aoiOptions);
      if (auto q = std::dynamic_pointer_cast<NEOSingleSlater<dcomplex,double>>(prot_ss)) {
        q->coreHBuilder = std::make_shared<NRCoreH<dcomplex,double>>(
            *std::dynamic_pointer_cast<Integrals<double>>(paoints), prot_aoiOptions);
        q->fockBuilder = std::make_shared<FockBuilder<dcomplex,double>>(prot_aoiOptions);
      }
    } else {
      CErr("Complex INT is not a valid option for NEO",std::cout);
    }



    // Construct ERIContractions
    if(auto p = std::dynamic_pointer_cast<NEOSingleSlater<double,double>>(elec_ss)) {

      auto q = std::dynamic_pointer_cast<NEOSingleSlater<double,double>>(prot_ss);
      std::shared_ptr<TwoPInts<double>> ERI =
          std::dynamic_pointer_cast<Integrals<double>>(eaoints)->TPI;
      std::shared_ptr<TwoPInts<double>> PRI =
          std::dynamic_pointer_cast<Integrals<double>>(paoints)->TPI;
      std::shared_ptr<TwoPInts<double>> EPAI =
          std::dynamic_pointer_cast<Integrals<double>>(epaoints)->TPI;

      // electron 2-particle
      if (auto eri_typed = std::dynamic_pointer_cast<DirectTPI<double>>(ERI)) {
        p->TPI = std::make_shared<GTODirectTPIContraction<double,double>>(*eri_typed);
      } else if (auto eri_typed = std::dynamic_pointer_cast<InCoreRITPI<double>>(ERI)) {
        p->TPI = std::make_shared<InCoreRITPIContraction<double,double>>(*eri_typed);
      } else if (auto eri_typed = std::dynamic_pointer_cast<InCore4indexTPI<double>>(ERI)) {
        p->TPI = std::make_shared<InCore4indexTPIContraction<double,double>>(*eri_typed);
      } else {
        CErr("Invalid ERInts type for Wavefunction<double,double>",std::cout);
      }

      // proton 2-particle
      if (auto eri_typed = std::dynamic_pointer_cast<DirectTPI<double>>(PRI)) {
        q->TPI = std::make_shared<GTODirectTPIContraction<double,double>>(*eri_typed);
      } else if (auto eri_typed = std::dynamic_pointer_cast<InCoreRITPI<double>>(PRI)) {
        q->TPI = std::make_shared<InCoreRITPIContraction<double,double>>(*eri_typed);
      } else if (auto eri_typed = std::dynamic_pointer_cast<InCore4indexTPI<double>>(PRI)) {
        q->TPI = std::make_shared<InCore4indexTPIContraction<double,double>>(*eri_typed);
      } else {
        CErr("Invalid ERInts type for Wavefunction<double,double>",std::cout);
      }

      // electron-proton 2-particle
      if (auto eri_typed = std::dynamic_pointer_cast<DirectTPI<double>>(EPAI)) {
        p->EPAI = std::make_shared<GTODirectTPIContraction<double,double>>(*eri_typed);
        q->EPAI = std::make_shared<GTODirectTPIContraction<double,double>>(*eri_typed);
        q->EPAI->contractSecond = true;
      } else if (auto eri_typed = std::dynamic_pointer_cast<InCore4indexTPI<double>>(EPAI)) {
        p->EPAI = std::make_shared<InCore4indexTPIContraction<double,double>>(*eri_typed);
        q->EPAI = std::make_shared<InCore4indexTPIContraction<double,double>>(*eri_typed);
        q->EPAI->contractSecond = true;
      } else {
        CErr("Invalid ERInts type for Wavefunction<double,double>",std::cout);
      }
    } else if(auto p = std::dynamic_pointer_cast<NEOSingleSlater<dcomplex,double>>(elec_ss)) {
      auto q = std::dynamic_pointer_cast<NEOSingleSlater<dcomplex,double>>(prot_ss);
      std::shared_ptr<TwoPInts<double>> ERI =
          std::dynamic_pointer_cast<Integrals<double>>(eaoints)->TPI;
      std::shared_ptr<TwoPInts<double>> PRI =
          std::dynamic_pointer_cast<Integrals<double>>(paoints)->TPI;
      std::shared_ptr<TwoPInts<double>> EPAI =
          std::dynamic_pointer_cast<Integrals<double>>(epaoints)->TPI;
      
      // electron 2-particle
      if (auto eri_typed = std::dynamic_pointer_cast<DirectTPI<double>>(ERI)) {
        p->TPI = std::make_shared<GTODirectTPIContraction<dcomplex,double>>(*eri_typed);
      } else if (auto eri_typed = std::dynamic_pointer_cast<InCoreRITPI<double>>(ERI)) {
        p->TPI = std::make_shared<InCoreRITPIContraction<dcomplex,double>>(*eri_typed);
      } else if (auto eri_typed = std::dynamic_pointer_cast<InCore4indexTPI<double>>(ERI)) {
        p->TPI = std::make_shared<InCore4indexTPIContraction<dcomplex,double>>(*eri_typed);
      } else {
        CErr("Invalid ERInts type for Wavefunction<dcomplex,double>",std::cout);
      }

      // proton 2-particle
      if (auto eri_typed = std::dynamic_pointer_cast<DirectTPI<double>>(PRI)) {
        q->TPI = std::make_shared<GTODirectTPIContraction<dcomplex,double>>(*eri_typed);
      } else if (auto eri_typed = std::dynamic_pointer_cast<InCoreRITPI<double>>(PRI)) {
        q->TPI = std::make_shared<InCoreRITPIContraction<dcomplex,double>>(*eri_typed);
      } else if (auto eri_typed = std::dynamic_pointer_cast<InCore4indexTPI<double>>(PRI)) {
        q->TPI = std::make_shared<InCore4indexTPIContraction<dcomplex,double>>(*eri_typed);
      } else {
        CErr("Invalid ERInts type for Wavefunction<dcomplex,double>",std::cout);
      }

      // elctron-proton 2-particle
      if (auto eri_typed = std::dynamic_pointer_cast<DirectTPI<double>>(EPAI)) {
        p->EPAI = std::make_shared<GTODirectTPIContraction<dcomplex,double>>(*eri_typed);
        q->EPAI = std::make_shared<GTODirectTPIContraction<dcomplex,double>>(*eri_typed);
        q->EPAI->contractSecond = true;
      } else if (auto eri_typed = std::dynamic_pointer_cast<InCore4indexTPI<double>>(EPAI)) {
        p->EPAI = std::make_shared<InCore4indexTPIContraction<dcomplex,double>>(*eri_typed);
        q->EPAI = std::make_shared<InCore4indexTPIContraction<dcomplex,double>>(*eri_typed);
        q->EPAI->contractSecond = true;
      } else {
        CErr("Invalid ERInts type for Wavefunction<dcomplex,double>",std::cout);
      }
    } else {
      CErr("Complex INT is not a valid option for NEO",std::cout);
    }











    return {elec_ss, prot_ss};

  };





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


  // Regular SingleSlater wrapper
  std::shared_ptr<SingleSlaterBase> CQSingleSlaterOptions(
    std::ostream &out, CQInputFile &input,
    CQMemManager &mem, Molecule &mol, BasisSet &basis,
    std::shared_ptr<IntegralsBase> aoints) {

    return buildSingleSlater(out, input, mem, mol, basis, aoints, {-1., 1.}, "QM");

  }


  // NEO SingleSlater wrapper
  std::shared_ptr<SingleSlaterBase> CQNEOSSOptions(
    std::ostream &out, CQInputFile &input,
    CQMemManager &mem, Molecule &mol,
    BasisSet &ebasis, BasisSet &pbasis,
    std::shared_ptr<IntegralsBase> eaoints, 
    std::shared_ptr<IntegralsBase> paoints,
    std::shared_ptr<IntegralsBase> epaoints) {

    Particle p{-1., 1.};
#define NEO_LIST(T) \
    MPI_COMM_WORLD,mem,mol,ebasis,dynamic_cast<Integrals<T>&>(*epaoints),1,false,p

    auto ess = buildSingleSlater(out, input, mem, mol, ebasis, eaoints, {-1., 1.}, "QM");
    auto pss = buildSingleSlater(out, input, mem, mol, pbasis, paoints, {1., ProtMassPerE}, "PROTQM");

    std::shared_ptr<SingleSlaterBase> neoss;

    if(auto ess_t = std::dynamic_pointer_cast<SingleSlater<double,double>>(ess)) {
      if(auto pss_t = std::dynamic_pointer_cast<SingleSlater<double,double>>(pss)) {
        auto neoss_t = std::make_shared<NEOSS<double,double>>(NEO_LIST(double));
        neoss_t->addSubsystem("Electronic", ess_t);
        neoss_t->addSubsystem("Protonic", pss_t);
        neoss_t->setOrder({"Protonic", "Electronic"});
        neoss = std::dynamic_pointer_cast<SingleSlaterBase>(neoss_t);
      }
      else
        CErr("Electrons and protons must use the same field (real/real) or (complex/complex)");
    }
    else if(auto ess_t = std::dynamic_pointer_cast<SingleSlater<dcomplex,double>>(ess)) {
      if(auto pss_t = std::dynamic_pointer_cast<SingleSlater<dcomplex,double>>(pss)) {
        auto neoss_t = std::make_shared<NEOSS<dcomplex,double>>(NEO_LIST(double));
        neoss_t->addSubsystem("Electronic", ess_t);
        neoss_t->addSubsystem("Protonic", pss_t);
        neoss_t->setOrder({"Protonic", "Electronic"});
        neoss = std::dynamic_pointer_cast<SingleSlaterBase>(neoss_t);
      }
      else
        CErr("Electrons and protons must use the same field (real/real) or (complex/complex)");
    }
    else {
      CErr("NEO + GIAO NYI!");
    }

    return neoss;

  }


}; // namespace ChronusQ
