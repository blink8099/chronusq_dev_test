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
#include <cerr.hpp>

namespace ChronusQ {


  void CQSCF_VALID( std::ostream &out, CQInputFile &input ) {

    // Allowed keywords
    std::vector<std::string> allowedKeywords = {
      "ENETOL",
      "DENTOL",
      "MAXITER",
      "INCFOCK",
      "NINCFOCK",
      "GUESS",
      "ALG",
      "EXTRAP",
      "DIIS",
      "NKEEP",
      "DAMP",
      "DAMPPARAM",
      "DAMPERROR",
      "FIELD",
      "PRINTMOS"
    };

    // Specified keywords
    std::vector<std::string> scfKeywords = input.getDataInSection("SCF");

    // Make sure all of scfKeywords in allowedKeywords

    for( auto &keyword : scfKeywords ) {
      auto ipos = std::find(allowedKeywords.begin(),allowedKeywords.end(),keyword);
      if( ipos == allowedKeywords.end() ) 
        CErr("Keyword SCF." + keyword + " is not recognized",std::cout);// Error
    }
    // Check for disallowed combinations (if any)
  }


  void HandlePostSCFRestarts(std::ostream &out, CQInputFile &input,
    SingleSlaterBase &ss) {

    bool restart = false;
    
    if ( input.containsSection("RT") )
      OPTOPT( restart |= input.getData<bool>("RT.RESTART"); )
    // Add additional checks like the one below for restarting other post SCF
    // RESP restart is NYI, so this is a commented out placeholder example
    //
    // else if ( input.containsSection("RESP") )
    //   OPTOPT( restart |= input.getData<bool>("RESP.RESTART"); )

    if ( restart ) {
      out << "  *** RESTART requested -- SCF.GUESS set to READMO and SCF.ALG set to SKIP ***";
      out << std::endl;
      ss.scfControls.guess = READMO;
      ss.scfControls.scfAlg = _SKIP_SCF;
    }

  }

  void CQSCFOptions(std::ostream &out, CQInputFile &input,
    SingleSlaterBase &ss, EMPerturbation &pert) {

    // SCF section not required
    if( not input.containsSection("SCF") ) {
     
      // Restart jobs
      HandlePostSCFRestarts(out, input, ss);

      return;
    }

    // Optionally parse guess

    // Energy convergence tolerance
    OPTOPT( ss.scfControls.eneConvTol = 
              input.getData<double>("SCF.ENETOL"); )

    // Energy convergence tolerance
    OPTOPT( ss.scfControls.denConvTol = 
              input.getData<double>("SCF.DENTOL"); )

    // Maximum SCF iterations
    OPTOPT( ss.scfControls.maxSCFIter = 
              input.getData<size_t>("SCF.MAXITER"); )


    // Incremental Fock Options
    OPTOPT(
      ss.scfControls.doIncFock = input.getData<bool>("SCF.INCFOCK");
    )
    OPTOPT(
      ss.scfControls.nIncFock = input.getData<size_t>("SCF.NINCFOCK");
    )


    // Guess
    OPTOPT(
      std::string guessString = input.getData<std::string>("SCF.GUESS");

      if( not guessString.compare("CORE") )
        ss.scfControls.guess = CORE;
      else if( not guessString.compare("SAD") )
        ss.scfControls.guess = SAD;
      else if( not guessString.compare("RANDOM") )
        ss.scfControls.guess = RANDOM;
      else if( not guessString.compare("READMO") )
        ss.scfControls.guess = READMO;
      else if( not guessString.compare("READDEN") )
        ss.scfControls.guess = READDEN;
      else
        CErr("Unrecognized entry for SCF.GUESS");
    )


    // ALGORITHM
    OPTOPT(

        std::string algString = input.getData<std::string>("SCF.ALG");
        if( not algString.compare("CONVENTIONAL") )
          ss.scfControls.scfAlg = _CONVENTIONAL_SCF;
        else if( not algString.compare("NR") )
          ss.scfControls.scfAlg = _NEWTON_RAPHSON_SCF;
        else if( not algString.compare("SKIP") )
          ss.scfControls.scfAlg = _SKIP_SCF;
        else CErr("Unrecognized entry for SCF.ALG!");

    )


    // Restart jobs
    HandlePostSCFRestarts(out, input, ss);


    // Toggle extrapolation in its entireity
    OPTOPT(
      ss.scfControls.doExtrap = 
        input.getData<bool>("SCF.EXTRAP");
    )

    // Handle DIIS options
    OPTOPT(
      bool doDIIS = input.getData<bool>("SCF.DIIS");
      if( not doDIIS )
        ss.scfControls.diisAlg = NONE;
    );

    // Number of terms for keep for DIIS
    OPTOPT( ss.scfControls.nKeep = 
              input.getData<size_t>("SCF.NKEEP"); )

    // Parse Damping options
      
    OPTOPT(
      ss.scfControls.doDamp = input.getData<bool>("SCF.DAMP");
    );

    OPTOPT(
      ss.scfControls.dampStartParam = 
        input.getData<double>("SCF.DAMPPARAM");
    );

    OPTOPT(
      ss.scfControls.dampError = 
        input.getData<double>("SCF.DAMPERROR");
    );


    // SCF Field
    auto handleField = [&]() {
      std::string fieldStr;
      OPTOPT(
        fieldStr = input.getData<std::string>("SCF.FIELD");
      )
      if( fieldStr.empty() ) return;

      std::vector<std::string> tokens;
      split(tokens,fieldStr);

      if( tokens.size() < 4 )
        CErr(fieldStr + "is not a valid SCF Field specification");

      std::string fieldTypeStr = tokens[0];

      EMFieldTyp fieldType;
      if( not fieldTypeStr.compare("ELECTRIC") )
        fieldType = Electric;
      else if( not fieldTypeStr.compare("MAGNETIC") )
        fieldType = Magnetic;  
      else
        CErr(fieldTypeStr + "not a valid Field type");

      if( tokens.size() == 4 ) { 

        cart_t field = {std::stod(tokens[1]), std::stod(tokens[2]), 
                        std::stod(tokens[3])};

        pert.addField(fieldType,field);

      } else
        CErr("Non Dipole fields NYI");
    };

    handleField();






    // Printing Options
    OPTOPT( ss.scfControls.printMOCoeffs = 
      input.getData<bool>("SCF.PRINTMOS") );







    // Handling eqivalences in the input options


    // Setting the damp param to 0. is equivalent to
    // turning damping off
    if( ss.scfControls.dampStartParam == 0. )
      ss.scfControls.doDamp = false;

    // Turning off both damping and DIIS is equivalent
    // to turning off extrapolation entirely
    if( not ss.scfControls.doDamp and 
        ss.scfControls.diisAlg == NONE )
      ss.scfControls.doExtrap = false;


  }; // CQSCFOptions

}; // namespace ChronusQ
