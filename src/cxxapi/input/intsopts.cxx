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

#include <electronintegrals/print.hpp>
#include <electronintegrals/twoeints.hpp>
#include <electronintegrals/twoeints/incore4indexeri.hpp>
#include <electronintegrals/twoeints/gtodirecteri.hpp>
#include <electronintegrals/twoeints/giaodirecteri.hpp>
#include <electronintegrals/twoeints/incorerieri.hpp>

namespace ChronusQ {

  enum class CONTRACTION_ALGORITHM {
    DIRECT,
    INCORE,
    DENFIT
  }; ///< 2-e Integral Contraction Algorithm

  /**
   *
   *  Check valid keywords in the section.
   *
  */
  void CQINTS_VALID( std::ostream &out, CQInputFile &input ) {

    // Allowed keywords
    std::vector<std::string> allowedKeywords = {
      "ALG",          // Direct or Incore?
      "SCHWARTZ",     // double
      "RI",           // AUXBASIS or CHOLESKY or False
      "RITHRESHOLD",  // double
      "FINITENUCLEI", // True or False
      "NONRELCOULOMB",// True or False
      "DC",           // True or False
      "DIRACCOULOMB", // True or False
      "BREIT",        // True or False
      "GAUNT",        // True or False
      "SSSS",         // True or False
      "GAUGE",        // True or False
      "LIBCINT"       // Ture or False
    };

    // Specified keywords
    std::vector<std::string> intsKeywords = input.getDataInSection("INTS");

    // Make sure all of basisKeywords in allowedKeywords
    for( auto &keyword : intsKeywords ) {
      auto ipos = std::find(allowedKeywords.begin(),allowedKeywords.end(),keyword);
      if( ipos == allowedKeywords.end() ) 
        CErr("Keyword INTS." + keyword + " is not recognized",std::cout);// Error
    }
    // Check for disallowed combinations (if any)
  }

  /**
   *  \brief Optionally set the control parameters for an
   *  AOIntegrals object
   *
   *  \param [in] out    Output device for data / error output.
   *  \param [in] input  Input file datastructure
   *  \param [in] aoints AOIntegrals object 
   *
   *
   */ 
  std::shared_ptr<IntegralsBase> CQIntsOptions(std::ostream &out, 
      CQInputFile &input, CQMemManager &mem, Molecule &mol,
      std::shared_ptr<BasisSet> basis, std::shared_ptr<BasisSet> dfbasis) {

    // Parse integral algorithm
    std::string ALG = "DIRECT";
    OPTOPT( ALG = input.getData<std::string>("INTS.ALG"); )
    trim(ALG);

    // Control Variables
    CONTRACTION_ALGORITHM contrAlg = CONTRACTION_ALGORITHM::DIRECT; ///< Alg for 2-body contraction
    double threshSchwartz = 1e-12; ///< Schwartz screening threshold
    std::string RI = "FALSE"; ///< RI algorithm
    double CDRI_thresh = 1e-4; ///< Cholesky RI threshold

    if( not ALG.compare("DIRECT") )
      contrAlg = CONTRACTION_ALGORITHM::DIRECT;
    else if( not ALG.compare("INCORE") )
      contrAlg = CONTRACTION_ALGORITHM::INCORE;
    else
      CErr(ALG + "not a valid INTS.ALG",out);

    // Parse Schwartz threshold
    OPTOPT( threshSchwartz = input.getData<double>("INTS.SCHWARTZ"); )

    // Parse RI option
    OPTOPT( RI = input.getData<std::string>("INTS.RI");)
    trim(RI);

    if( not RI.compare("AUXBASIS") and not RI.compare("CHOLESKY") and not RI.compare("FALSE") )
      CErr(RI + " not a valid INTS.RI",out);

    if(RI.compare("FALSE") and contrAlg != CONTRACTION_ALGORITHM::INCORE) {
      contrAlg = CONTRACTION_ALGORITHM::INCORE;
      std::cout << "Incore ERI algorithm enforced by RI." << std::endl;
    }
    if(not RI.compare("AUXBASIS") and dfbasis->nBasis < 1)
      CErr("Keyword INTS.RI requires a non-empty DFbasis->",std::cout);
    if(not RI.compare("CHOLESKY"))
      OPTOPT( CDRI_thresh = input.getData<double>("INTS.CDRI_THRESHOLD"); )

    std::shared_ptr<IntegralsBase> aoi = nullptr;

    if(basis->basisType == REAL_GTO) {
      std::shared_ptr<Integrals<double>> aoint =
          std::make_shared<Integrals<double>>();
      if(not RI.compare("AUXBASIS"))
        aoint->ERI =
            std::make_shared<InCoreAuxBasisRIERI<double>>(mem,basis->nBasis,dfbasis);
      else if (not RI.compare("CHOLESKY"))
        aoint->ERI =
            std::make_shared<InCoreCholeskyRIERI<double>>(mem,basis->nBasis,CDRI_thresh);
      else if (contrAlg == CONTRACTION_ALGORITHM::INCORE)
        aoint->ERI =
            std::make_shared<InCore4indexERI<double>>(mem,basis->nBasis);
      else
        aoint->ERI =
            std::make_shared<DirectERI<double>>(mem,*basis,mol,threshSchwartz);

      aoi = std::dynamic_pointer_cast<IntegralsBase>(aoint);
    } else if(basis->basisType == COMPLEX_GIAO) {
      std::shared_ptr<Integrals<dcomplex>> giaoint =
          std::make_shared<Integrals<dcomplex>>();
      if(RI.compare("FALSE"))
        CErr("GIAO resolution of identity ERI NYI",std::cout);
      else if (contrAlg == CONTRACTION_ALGORITHM::INCORE)
        giaoint->ERI =
            std::make_shared<InCore4indexERI<dcomplex>>(mem,basis->nBasis);
      else
        giaoint->ERI =
            std::make_shared<DirectERI<dcomplex>>(mem,*basis,mol,threshSchwartz);

      aoi = std::dynamic_pointer_cast<IntegralsBase>(giaoint);
    }

    // Print
    out <<  *aoi << std::endl;

    return aoi;

  }; // CQIntsOptions


}; // namespace ChronusQ
