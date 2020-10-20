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
#include <chronusq_sys.hpp>
#include <cxxapi/options.hpp>
#include <cxxapi/procedural.hpp>
#include <cerr.hpp>

namespace ChronusQ {

  /**
   *
   *  Check valid keywords in the section.
   *
  */
  void CQMOLECULE_VALID( std::ostream &out, CQInputFile &input ) {

    // Allowed keywords
    std::vector<std::string> allowedKeywords = {
      "CHARGE",
      "MULT",
      "READGEOM",
      "GEOM"
    };

    // Specified keywords
    std::vector<std::string> moleculeKeywords = input.getDataInSection("MOLECULE");

    // Make sure all of basisKeywords in allowedKeywords
    for( auto &keyword : moleculeKeywords ) {
      auto ipos = std::find(allowedKeywords.begin(),allowedKeywords.end(),keyword);
      if( ipos == allowedKeywords.end() ) 
        CErr("Keyword MOLECULE." + keyword + " is not recognized",std::cout);// Error
    }
    // Check for disallowed combinations (if any)
  }

  /**
   *  Construct a Molecule object using the input file.
   *
   *  \param [in] out   Output device for data output
   *  \param [in] input Input file datastructure
   *
   *  \returns Appropriate Molecule object for the input parameters
   */
  Molecule CQMoleculeOptions(std::ostream &out, CQInputFile &input, std::string &scrName) {

    Molecule mol;
    
    // Obtain Charge
    try { mol.charge = input.getData<int>("MOLECULE.CHARGE"); }
    catch (...) {
      CErr("Unable to set Molecular Charge!", out);
    }

    // Obtain Multiplicity
    try { mol.multip = input.getData<size_t>("MOLECULE.MULT"); }
    catch (...) {
      //CErr("Unable to set Molecular Spin Multiplicity!", out);
    }

    // Parse Geometry Read Options
    std::string geomReadStr;
    try { geomReadStr = input.getData<std::string>("MOLECULE.READGEOM"); }
    catch (...) {
      geomReadStr="INPUTFILE";
    }

    // Parse Geometry
    std::string geomStr;
    if( geomReadStr == "INPUTFILE" ){
      try { geomStr = input.getData<std::string>("MOLECULE.GEOM"); }
      catch (...) {
        CErr("Unable to find Molecular Geometry in Input File!", out);
      }
    }

    // Parse different files depending on READGEOM
    if( geomReadStr == "INPUTFILE" ) parseGeomInp(mol,geomStr,out);
    else if( geomReadStr == "FCHK" ) parseGeomFchk(mol,scrName,out);
    else CErr("INVALID OPTION FOR READGEOM KEYWORD!");

    return mol; // Return Molecule object (no intermediates)

  }; // CQMoleculeOptions

  void parseGeomInp( Molecule &mol, std::string &geomStr, std::ostream &out ) {

    std::istringstream geomStream; geomStream.str(geomStr);
    std::vector<std::string> tokens;
    std::vector<Atom> atoms;
    std::locale loc;

    // Loop over lines of geometry specification
    for(std::string line; std::getline(geomStream, line); ){
      split(tokens,line," \t");

      if( tokens.size() == 0 ) continue;

      std::string atmSymb = tokens[0];

      bool hasDig = std::any_of(atmSymb.begin(),atmSymb.end(),
        [&](char a) {return std::isdigit(a,loc); });
      bool hasAlpha = std::any_of(atmSymb.begin(),atmSymb.end(),
        [&](char a) {return std::isalpha(a,loc); });

      
      bool isAtNum   = hasDig   and not hasAlpha;
      bool isComIso  = hasAlpha and not hasDig  ;
      bool isSpecIso = hasDig   and     hasAlpha;



      if( isAtNum ) {

        auto it = 
        std::find_if(atomicReference.begin(),atomicReference.end(),
          [&](std::pair<std::string,Atom> st){ 
            return st.second.atomicNumber == std::stoi(atmSymb);}
           );

        atoms.emplace_back((it == atomicReference.end() ? "X" : defaultIsotope[it->first]));

        
      } else atoms.emplace_back(isComIso ? defaultIsotope[atmSymb] : atmSymb);
      
      // Convert to Bohr
      atoms.back().coord[0] = std::stod(tokens[1]) / AngPerBohr;
      atoms.back().coord[1] = std::stod(tokens[2]) / AngPerBohr;
      atoms.back().coord[2] = std::stod(tokens[3]) / AngPerBohr;
      
    }

    if ( atoms.size() == 0 )
      CErr("MOLECULE.GEOM must not be empty and must be indented");
    // Set the Atoms vector in Molecule (calls Molecule::update())
    mol.setAtoms(atoms);

    // Output Molecule data
    out << mol << std::endl;

  } // parseGeomInp

  void parseGeomFchk( Molecule &mol, std::string &fchkName, std::ostream &out ) {

    std::ifstream fchkFile;
    std::vector<Atom> atoms;
    fchkFile.open(fchkName);

    if ( fchkFile.good() ){
      std::cout << "fchkFile found in parseGeomFchk" << "\n";
    }else{
      CErr("Could not find fchkFile. Use -s flag.");
    }

    // Boolean used for fchk reading
    bool readAtNum=false, readCoord=false;
    // Counter for atomic coordinates
    int coordCount=0, atomCount=0;

    // TODO: Add non-default isotopes
    // Parse fchk file
    while( not fchkFile.eof() ) {

      std::string line;
      std::getline(fchkFile,line);

      // Determine position of first and last non-space character
      size_t firstNonSpace = line.find_first_not_of(" ");

      // Skip blank lines
      if( firstNonSpace == std::string::npos ) continue;

      // Strip trailing spaces
      trim_right(line);
      line =
         line.substr(firstNonSpace,line.length()-firstNonSpace);

      // Split the line into tokens, trim spaces
      std::vector<std::string> tokens;
      split(tokens,line,"   ");
      for(auto &X : tokens) { trim(X); }

      // Finding atomic number in fchk file
      if ( tokens.size() > 4 ){
        if ( tokens[0] == "Atomic" and tokens[1] == "numbers" ){
          readAtNum = true;
          continue;
        }
      }

      // Ending atomic number in fchk file
      if ( tokens.size() > 4 ){
        if ( tokens[0] == "Nuclear" and tokens[1] == "charges" ){
          readAtNum = false;
          continue;
        }
      }

      // Finding coordinates in fchk file
      if ( tokens.size() > 4 ){
        if ( tokens[0] == "Current" and tokens[1] == "cartesian" ){
          readCoord = true;
          continue;
        }
      }

      // Ending atomic number in fchk file
      if ( tokens.size() > 5 ){
        if ( tokens[2] == "symbols" and tokens[3] == "in" ){
          readCoord = false;
          continue;
        }
      }

      // Read in atomic numbers
      if ( readAtNum ){
        for(int i=0; i<tokens.size(); i++){

          auto it =
          std::find_if(atomicReference.begin(),atomicReference.end(),
            [&](std::pair<std::string,Atom> st){
              return st.second.atomicNumber == std::stoi(tokens[i]);}
             );

          // Atomic symbols are currently the first isotope listed in atomicReference
          // Trimming everything after the hyphen and passing into defaultIsotope
          std::string atmSymb = it->first.substr(0,it->first.find("-",0));

          atoms.emplace_back((it == atomicReference.end() ? "X" : defaultIsotope[atmSymb]));

        }
      } // Read in atomic numbers

      // Read in cartesian coordinates from fchk (in Bohr)
      if ( readCoord ){
        for(int i=0; i<tokens.size(); i++){

          atoms[atomCount].coord[coordCount] = std::stod(tokens[i]);

          coordCount = coordCount + 1;

          // Reset for x, y, z
          if( coordCount == 3 ){
            coordCount = 0;
            atomCount = atomCount + 1;
          }

        }
      } // Read in cartesian coordinates

    } // End of fchk file parsing

    // Set the Atoms vector in Molecule (calls Molecule::update())
    mol.setAtoms(atoms);

    // Output Molecule data
    out << mol << std::endl;

  } // parseGeomFchk

}; // namespace ChronusQ

