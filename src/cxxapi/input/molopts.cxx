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
  Molecule CQMoleculeOptions(std::ostream &out, CQInputFile &input) {

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

    // Parse Geometry
    std::string geomStr;
    try { geomStr = input.getData<std::string>("MOLECULE.GEOM"); }
    catch (...) {
      CErr("Unable to find Molecular Geometry!", out);
    }



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

    // Set the Atoms vector in Molecule (calls Molecule::update())
    mol.setAtoms(atoms);

    // Output Molecule data
    out << mol << std::endl;

    return mol; // Return Molecule object (no intermediates)

  }; // CQMoleculeOptions

}; // namespace ChronusQ

