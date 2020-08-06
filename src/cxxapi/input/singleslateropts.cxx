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
#include <fockbuilder/rofock.hpp>
#include <electronintegrals/twoeints.hpp>
#include <electronintegrals/twoeints/gtodirecteri.hpp>
#include <electronintegrals/twoeints/incore4indexeri.hpp>
#include <electronintegrals/twoeints/giaodirecteri.hpp>
#include <electronintegrals/twoeints/incorerieri.hpp>

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
      "X2CTYPE"
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

    // All refernce keywords
    std::vector<std::string> rawRefs(KSRefs);
    rawRefs.insert(rawRefs.begin(),"HF");

    // Construct R/U/RO/G/X2C reference keywords
    std::vector<std::string> RRefs, URefs, RORefs, GRefs, X2CRefs;
    for(auto &f : rawRefs) {
      RRefs.emplace_back( "R" + f );
      URefs.emplace_back( "U" + f );
      RORefs.emplace_back( "RO" + f);
      GRefs.emplace_back( "G" + f );
      X2CRefs.emplace_back( "X2C" + f );
    }


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
    bool isX2CRef = 
      std::find(X2CRefs.begin(),X2CRefs.end(),refString) != X2CRefs.end();

    // Throw an error if not a valid reference keyword
    if( not isRawRef and not isRRef and not isURef and not isRORef and 
        not isGRef and not isX2CRef )
      CErr(refString + " is not a valid QM.REFERENCE",out);

    // Cleanup the reference string
    if( not isRawRef )
      if( isX2CRef ) refString.erase(0,3);
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
    else if( isGRef or isX2CRef ) {
      iCS = false; nC = 2;
    }

    // Sanity Checks
    bool isGIAO = basis.basisType == COMPLEX_GIAO;

    if( nC == 2 and not RCflag.compare("REAL") )
      CErr("Real + Two-Component not valid",out);

    if( isGIAO and not RCflag.compare("REAL") )
      CErr("Real + GIAO not valid",out);

    if( isGIAO and isKSRef )
      CErr("KS + GIAO not valid",out);

    if( isGIAO and isX2CRef )
      CErr("X2C + GIAO not valid",out);







    // Determine Real/Complex if need be
    if(not RCflag.compare("AUTO") ) {
      if( nC == 2 )
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
      else if(not isGIAO)
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<double,double>>( HF_LIST(double) )
          );
      else
        CErr("GIAO + REAL is not a valid option.",out);

    else if( not RCflag.compare("COMPLEX") and not isGIAO)
      if( isKSRef and isX2CRef)
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<KohnSham<dcomplex,double>>(
              "Exact Two Component", "X2C-", KS_LIST(double)
            )
          );
      else if( isKSRef)
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<KohnSham<dcomplex,double>>( KS_LIST(double) )
          );
      else if( isX2CRef)
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<dcomplex,double>>(
              "Exact Two Component","X2C-",HF_LIST(double)
            )
          );
      else if(isRORef)
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<dcomplex,double>>(
            "Complex Restricted Open-shell Hartree-Fock", "C-ROHF", HF_LIST(double) )
          );
      else
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<dcomplex,double>>( HF_LIST(double) )
          );
    else
      if( isKSRef and isX2CRef)
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<KohnSham<dcomplex,dcomplex>>(
              "Exact Two Component", "X2C-", KS_LIST(dcomplex)
            )
          );
      else if( isKSRef)
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<KohnSham<dcomplex,dcomplex>>( KS_LIST(dcomplex) )
          );
      else if( isX2CRef)
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<dcomplex,dcomplex>>(
              "Exact Two Component","X2C-",HF_LIST(dcomplex)
            )
          );
      else if(isRORef)
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<dcomplex,dcomplex>>(
            "Complex Restricted Open-shell Hartree-Fock", "C-ROHF", HF_LIST(dcomplex) )
          );
      else
        ss = std::dynamic_pointer_cast<SingleSlaterBase>(
            std::make_shared<HartreeFock<dcomplex,dcomplex>>( HF_LIST(dcomplex) )
          );





    // Parse X2C option
    std::string x2c_str;
    bool atomic = false;
    ATOMIC_X2C_TYPE x2ctype = {false,false};
    bool scalarOnly = false;
    OPTOPT( x2c_str = input.getData<std::string>("QM.X2CTYPE")  );
    trim(x2c_str);
    split(tokens,x2c_str);
    for(auto &X : tokens) {
      trim(X);
      if( not X.compare("FULL") ) {
        atomic = false;
        x2ctype = {false,false};
      } else if( not X.compare("ALH") ) {
        atomic = true;
        x2ctype = {true,true};
      } else if( not X.compare("ALU") ) {
        atomic = true;
        x2ctype = {true,false};
      } else if( not X.compare("DLH") ) {
        atomic = true;
        x2ctype = {false,true};
      } else if( not X.compare("DLU") ) {
        atomic = true;
        x2ctype = {false,false};
      } else if( not X.compare("SO") )
        scalarOnly = false;
      else if( not X.compare("SCALAR") )
        scalarOnly = true;
      else
        CErr(X + " not a valid QM.X2CTYPE",out);
    }


    // Parse Finite Width Nuclei
    AOIntsOptions aoiOptions{basis.basisType,false,false,false,false,false,false,false};
    std::string finiteCore = "DEFAULT";
    OPTOPT( finiteCore = input.getData<std::string>("INTS.FINITE_NUCLEI"); )
    trim(finiteCore);
    if( not finiteCore.compare("TRUE") )
      aoiOptions.finiteWidthNuc = true;
    else if( not finiteCore.compare("FALSE") )
      aoiOptions.finiteWidthNuc = false;
    else if( not finiteCore.compare("DEFAULT") )
      aoiOptions.finiteWidthNuc = isX2CRef or scalarOnly;
    else
      CErr(finiteCore + " not a valid INTS.ALG",out);
    OPTOPT( aoiOptions.DiracCoulomb = input.getData<bool>("INTS.DC") )
    OPTOPT( aoiOptions.SSSS = input.getData<bool>("INTS.SSSS") )
    OPTOPT( aoiOptions.Gaunt = input.getData<bool>("INTS.DCB") )
    OPTOPT( aoiOptions.Gauge = input.getData<bool>("INTS.GAUGE") )
    if(aoiOptions.Gaunt) aoiOptions.DiracCoulomb = true;


    // Construct CoreHBuilder
    if( isX2CRef or scalarOnly ) {
      aoiOptions.OneEScalarRelativity = true;
      aoiOptions.OneESpinOrbit = not scalarOnly;
      if(auto p = std::dynamic_pointer_cast<SingleSlater<dcomplex,double>>(ss)) {
        if (atomic)
          p->coreHBuilder = std::make_shared<AtomicX2C<dcomplex,double>>(
              *std::dynamic_pointer_cast<Integrals<double>>(aoints),
              mem, mol, basis, aoiOptions, x2ctype);
        else
          p->coreHBuilder = std::make_shared<X2C<dcomplex,double>>(
              *std::dynamic_pointer_cast<Integrals<double>>(aoints),
              mem, mol, basis, aoiOptions);
        p->fockBuilder = std::make_shared<FockBuilder<dcomplex,double>>();
      } else if(auto p = std::dynamic_pointer_cast<SingleSlater<double,double>>(ss)) {
        if (scalarOnly) {
          if (atomic)
            p->coreHBuilder = std::make_shared<AtomicX2C<double,double>>(
                *std::dynamic_pointer_cast<Integrals<double>>(aoints),
                mem, mol, basis, aoiOptions, x2ctype);
          else
            p->coreHBuilder = std::make_shared<X2C<double,double>>(
                *std::dynamic_pointer_cast<Integrals<double>>(aoints),
                mem, mol, basis, aoiOptions);
          p->fockBuilder = std::make_shared<FockBuilder<double,double>>();
        } else
          CErr("SOX2C + Real WFN is not a valid option",std::cout);
      } else if (std::dynamic_pointer_cast<SingleSlater<dcomplex,dcomplex>>(ss)) {
        CErr("X2C + Complex Ints NYI",std::cout);
      } else {
        CErr("Complex INT + Real WFN is not a valid option",std::cout);
      }
    } else {
      if(auto p = std::dynamic_pointer_cast<SingleSlater<double,double>>(ss)) {
        p->coreHBuilder = std::make_shared<NRCoreH<double,double>>(
            *std::dynamic_pointer_cast<Integrals<double>>(aoints), aoiOptions);
        if(isRORef) p->fockBuilder = std::make_shared<ROFock<double,double>>();
        else p->fockBuilder = std::make_shared<FockBuilder<double,double>>();
      } else if(auto p = std::dynamic_pointer_cast<SingleSlater<dcomplex,double>>(ss)) {
        p->coreHBuilder = std::make_shared<NRCoreH<dcomplex,double>>(
            *std::dynamic_pointer_cast<Integrals<double>>(aoints), aoiOptions);
        if(isRORef) p->fockBuilder = std::make_shared<ROFock<dcomplex,double>>();
        else p->fockBuilder = std::make_shared<FockBuilder<dcomplex,double>>();
      } else if (auto p = std::dynamic_pointer_cast<SingleSlater<dcomplex,dcomplex>>(ss)) {
        p->coreHBuilder = std::make_shared<NRCoreH<dcomplex,dcomplex>>(
            *std::dynamic_pointer_cast<Integrals<dcomplex>>(aoints), aoiOptions);
        if(isRORef) p->fockBuilder = std::make_shared<ROFock<dcomplex,dcomplex>>();
        else p->fockBuilder = std::make_shared<FockBuilder<dcomplex,dcomplex>>();
      } else {
        CErr("Complex INT + Real WFN is not a valid option",std::cout);
      }
    }



    // Construct ERIContractions
    if(auto p = std::dynamic_pointer_cast<SingleSlater<double,double>>(ss)) {
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











    return ss;

  }; // CQSingleSlaterOptions

}; // namespace ChronusQ
