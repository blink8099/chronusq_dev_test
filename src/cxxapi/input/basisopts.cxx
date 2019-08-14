/* 
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *  
 *  Copyright (C) 2014-2019 Li Research Group (University of Washington)
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
#include <util/mpi.hpp>
#include <cerr.hpp>

namespace ChronusQ {

  /**
   *
   *  Check valid keywords in the section.
   *
  */
  void CQBASIS_VALID( std::ostream &out, CQInputFile &input ) {

    // Allowed keywords
    std::vector<std::string> allowedKeywords = {
      "FORCECART",
      "BASIS",
      "BASISTYPE"
    };

    // Specified keywords
    std::vector<std::string> basisKeywords = input.getDataInSection("BASIS");

    // Make sure all of basisKeywords in allowedKeywords
    for( auto &keyword : basisKeywords ) {
      auto ipos = std::find(allowedKeywords.begin(),allowedKeywords.end(),keyword);
      if( ipos == allowedKeywords.end() ) 
        CErr("Keyword BASIS." + keyword + " is not recognized",std::cout);// Error
    }
    // Check for disallowed combinations (if any)
  }

  /**
   *  Construct a basis set object using the input file.
   *
   *  \param [in] out   Output device for data output
   *  \param [in] input Input file datastructure
   *  \param [in] mol   Molecule object from which to construct the BasisSet object
   *
   *  \returns Appropriate BasisSet object for the input parameters
   */
  BasisSet CQBasisSetOptions(std::ostream &out, CQInputFile &input,
    Molecule &mol) {

    // Determine if we're forcing cartesian functions
    bool forceCart(false);
    OPTOPT( forceCart = input.getData<bool>("BASIS.FORCECART"); );

    // Find the Basis File
    std::string basisName;
    try {
      basisName = input.getData<std::string>("BASIS.BASIS");

      if( std::find_if( basisKeyword.begin(), basisKeyword.end(),
            [&]( std::pair<std::string, std::string> x ) -> bool{
              return not x.first.compare(basisName); 
            } ) == basisKeyword.end() ) {


        CErr(basisName + " is not a supported Basis Set");

      }

    } catch(...) {
      CErr("BasisSet not specified!",out);
    }

    BASIS_FUNCTION_TYPE bType = REAL_GTO;
    try{
      std::string bTypeString = input.getData<std::string>("BASIS.BASISTYPE");
      if(not bTypeString.compare("GIAO")) bType = COMPLEX_GIAO;
      else if(not bTypeString.compare("GTO")) bType = REAL_GTO;
      else CErr("BASIS.BASISTYPE not valid.",out);
    } catch(...) {;}

    // Construct the BasisSet object
    BasisSet basis(basisName,mol,bType,forceCart,MPIRank() == 0);

    // Ouput BasisSet information
    out << basis << std::endl;

    return basis; // Return BasisSet object (no intermediates)

  }; // CQBasisSetOptions


}; // namespace ChronusQ
