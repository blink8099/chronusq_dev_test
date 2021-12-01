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

#include <fockbuilder/fourcompfock.hpp>
#include <cqlinalg.hpp>
#include <physcon.hpp>
#include <util/matout.hpp>
#include <particleintegrals/twopints/incore4indexreleri.hpp>
#include <particleintegrals/twopints/gtodirectreleri.hpp>

//#define _PRINT_MATRICES

namespace ChronusQ {

  /**   
   *  \brief Forms the 4C GD in Bathes
   */
  template <typename MatsT, typename IntsT>
  void FourCompFock<MatsT,IntsT>::formRawGDInBatches(SingleSlater<MatsT,IntsT> &ss,
    EMPerturbation &pert, bool increment, double xHFX, bool HerDen,
    std::vector<std::shared_ptr<PauliSpinorSquareMatrices<MatsT>>> & onePDMs, 
    std::vector<std::shared_ptr<PauliSpinorSquareMatrices<MatsT>>> & coulombMatrices, 
    std::vector<std::shared_ptr<PauliSpinorSquareMatrices<MatsT>>> & exchangeMatrices,
    std::vector<std::shared_ptr<PauliSpinorSquareMatrices<MatsT>>> & twoeHs) {
    
      // for incore just wrap around the old loop for now
      //
      // coulombMatrices and exchangeMatices won't be used
      //
      // because coulombMatrix is SquareMatrix instead of PauliSpinorSquareMatrices
      // so the dividing is not accurate 
      if( std::dynamic_pointer_cast<InCore4indexRelERIContraction<MatsT,IntsT>>(ss.TPI) ) {
        
        // cache the ss pointers
        std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> ss1PDM 
          = increment ? ss.deltaOnePDM: ss.onePDM;
        std::shared_ptr<SquareMatrix<MatsT>> ssCoulombMatrix = ss.coulombMatrix;
        std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> ssExchangeMatrix = ss.exchangeMatrix;
        std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> ssTwoeH = ss.twoeH;
        
        // allocate scratch space for coulombMatrix and exchangeMatrix 
        ss.coulombMatrix = std::make_shared<SquareMatrix<MatsT>>(ss.memManager, ss.coulombMatrix->dimension());
        ss.exchangeMatrix = std::make_shared<PauliSpinorSquareMatrices<MatsT>>(ss.memManager, ss.exchangeMatrix->dimension());

        for (auto i = 0ul; i < onePDMs.size(); i++) {
          
          // update pointers
          if (increment) ss.deltaOnePDM = onePDMs[i];
          else ss.onePDM = onePDMs[i];
          
          ss.twoeH = twoeHs[i];
          formGDInCore(ss, pert, increment, xHFX, HerDen);
        
        }

        // revert ss pointers
        if (increment) ss.deltaOnePDM = ss1PDM;
        else ss.onePDM = ss1PDM;
        ss.coulombMatrix = ssCoulombMatrix;
        ss.exchangeMatrix = ssExchangeMatrix;
        ss.twoeH = ssTwoeH;
      
      } else if( std::dynamic_pointer_cast<GTODirectRelERIContraction<MatsT,IntsT>>(ss.TPI) ) {
        
        formRawGDInBatchesDirect(ss, pert, increment, xHFX, HerDen, 
          onePDMs, coulombMatrices, exchangeMatrices, twoeHs);
      
      } else
        CErr("Unsupported ERIContraction type.");

  };


  /**   
   *  \brief Forms the 4C Fock matrix using AO-direct
   */
  template <typename MatsT, typename IntsT>
  void FourCompFock<MatsT,IntsT>::formRawGDInBatchesDirect(SingleSlater<MatsT,IntsT> &ss,
    EMPerturbation &pert, bool increment, double xHFX, bool HerDen, 
    std::vector<std::shared_ptr<PauliSpinorSquareMatrices<MatsT>>> & onePDMs, 
    std::vector<std::shared_ptr<PauliSpinorSquareMatrices<MatsT>>> & coulombMatrices, 
    std::vector<std::shared_ptr<PauliSpinorSquareMatrices<MatsT>>> & exchangeMatrices,
    std::vector<std::shared_ptr<PauliSpinorSquareMatrices<MatsT>>> & twoeHs) {
    
    
    // disable libint2
    if (not this->hamiltonianOptions_.Libcint) CErr("4C Integrals Needs Libcint");
    CQMemManager &mem = ss.memManager;
    GTODirectRelERIContraction<MatsT,IntsT> &relERICon =
        *std::dynamic_pointer_cast<GTODirectRelERIContraction<MatsT,IntsT>>(ss.TPI);

    bool computeCoulomb  = coulombMatrices.size() > 0;
    bool computeExchange = (std::abs(xHFX) > 1e-12) and exchangeMatrices.size() > 0;
    bool computeTwoeHs   = twoeHs.size() > 0;
    
    if (not computeCoulomb and not computeExchange and not computeTwoeHs) {
     CErr("Nothing specified to compute in FockBuilder::formRawGDInBatches");
    } else if(not computeCoulomb and not computeTwoeHs) {
     CErr("Only computeExchange is not supported in FockBuilder::formRawGDInBatches");
    }
    
    auto & coulombContainers = computeCoulomb ? coulombMatrices: twoeHs;
    
    size_t mPDM  = onePDMs.size();
    size_t NB1C  = ss.basisSet().nBasis;
    size_t NB2C  = 2 * NB1C; 
    size_t NB4C  = 4 * NB1C;
    size_t NB1C2 = NB1C*NB1C;
    size_t NB1C4 = NB1C*NB1C*NB1C*NB1C;
    size_t NB1C3 = NB1C*NB1C*NB1C;

    size_t SS = NB2C*NB1C+NB1C;
    size_t LS = NB2C*NB1C;
    size_t SL = NB1C;

    auto MS = SCALAR;

    size_t mpiRank   = MPIRank(ss.comm);
    bool   isNotRoot = mpiRank != 0;
    
    // allocate scratch spaces
    
    std::vector<std::shared_ptr<PauliSpinorSquareMatrices<MatsT>>>
      contract1PDMLL, contract1PDMSS, contract1PDMLS, contract1PDMSL, 
      CScrLLMS, CScrSS, CScrLS, XScrLL, XScrSS, XScrLS; 
     
    #define ALLOCATE_PAULISPINOR_SCR(SCR, SCRSIZE, hasXYZ) \
       SCR.push_back(std::make_shared<PauliSpinorSquareMatrices<MatsT>>(mem, SCRSIZE, hasXYZ, hasXYZ)); 
       // no need to intailize those matrices as it will be initialize in twoBodyRelContract
       // SCR.back()->clear();
    
    /* 
     * SCR Usage for different hamiltonian Options:
     *   
     *   Coulomb Terms and eXchange Terms
     *   They won't be used at the same time!!!
     *
     * 1. Bare Coulomb:
     *    - C: contract1PDMLL(only MS), CScrLLMS
     *    - X: contract1PDMLL(only MS), XScrLL 
     * 2. Dirac Coulomb without SSSS:
     *    - C: contract1PDMSS, CScrLLMS 
     *         contract1PDMLL, CScrSS
     *    - X: contract1PDMLS, XScrLS  
     * 3. Dirac Coulomb SSSS:
     *    - C: contract1PDMSS, CScrSS   
     *    - X: contract1PDMSS, XScrSS  
     * 4. Gaunt:
     *    - C: contract1PDMLS, CScrLS 
     *         contract1PDMSL, CScrLS
     *    - X: contract1PDMLL, XScrLL 
     *         contract1PDMSS, XScrSS  
     *         contract1PDMLS, XScrLS
     *         contract1PDMSL, XScrLS
     * 5. Gauge:
     *    - C: contract1PDMLS, CScrLS
     *         contract1PDMSL, CScrLS
     *    - X: contract1PDMLL, XScrLL
     *         contract1PDMSS, XScrSS
     *         contract1PDMLS, XScrLS
     *         contract1PDMSL, XScrLS
     */
    
    
    bool allocate1PDMLLXYZ = this->hamiltonianOptions_.DiracCoulomb or
      this->hamiltonianOptions_.DiracCoulombSSSS or
      this->hamiltonianOptions_.Gaunt or
      this->hamiltonianOptions_.Gauge;
    
    bool allocate1PDMSS = allocate1PDMLLXYZ;
    bool allocate1PDMLS = (this->hamiltonianOptions_.DiracCoulomb and computeExchange) or
      this->hamiltonianOptions_.Gaunt or
      this->hamiltonianOptions_.Gauge;
    bool allocate1PDMSL = 
      this->hamiltonianOptions_.Gaunt or
      this->hamiltonianOptions_.Gauge;
    
    bool allocateCScrLLMS = this->hamiltonianOptions_.BareCoulomb or 
      this->hamiltonianOptions_.DiracCoulomb;
    bool allocateCScrSS = this->hamiltonianOptions_.DiracCoulomb or
      this->hamiltonianOptions_.DiracCoulombSSSS;
    bool allocateCScrLS = this->hamiltonianOptions_.Gaunt or 
      this->hamiltonianOptions_.Gauge;
    
    bool allocateXScrLL = this->hamiltonianOptions_.BareCoulomb or 
      this->hamiltonianOptions_.Gaunt or 
      this->hamiltonianOptions_.Gauge;
    bool allocateXScrSS = this->hamiltonianOptions_.DiracCoulombSSSS or 
      this->hamiltonianOptions_.Gaunt or 
      this->hamiltonianOptions_.Gauge;
    bool allocateXScrLS = this->hamiltonianOptions_.Gaunt or 
      this->hamiltonianOptions_.Gauge;
    
    for (auto i = 0ul; i < mPDM; i++) {
      // Allocate Density
      ALLOCATE_PAULISPINOR_SCR(contract1PDMLL, NB1C, allocate1PDMLLXYZ); 
      if (allocate1PDMSS) ALLOCATE_PAULISPINOR_SCR(contract1PDMSS, NB1C, true);    
      if (allocate1PDMLS) ALLOCATE_PAULISPINOR_SCR(contract1PDMLS, NB1C, true);    
      if (allocate1PDMSL) ALLOCATE_PAULISPINOR_SCR(contract1PDMSL, NB1C, true);    
      // allocate Coulomb SCR
      if (allocateCScrLLMS) ALLOCATE_PAULISPINOR_SCR(CScrLLMS, NB1C, false); 
      if (allocateCScrSS)   ALLOCATE_PAULISPINOR_SCR(CScrSS, NB1C, true);    
      if (allocateCScrLS)   ALLOCATE_PAULISPINOR_SCR(CScrLS, NB1C, true);    
      // if(allocateCScrSL)   ALLOCATE_PAULISPINOR_SCR(CScrSL, true);    
      // allocate Exchange SCR
      if (computeExchange) {
        if (allocateXScrLL) ALLOCATE_PAULISPINOR_SCR(XScrLL, NB1C, true); 
        if (allocateXScrSS) ALLOCATE_PAULISPINOR_SCR(XScrSS, NB1C, true);    
        if (allocateXScrLS) ALLOCATE_PAULISPINOR_SCR(XScrLS, NB1C, true);    
      }
    }

    // allocate dummies
    auto dummy_pauli = std::make_shared<PauliSpinorSquareMatrices<MatsT>>(mem, 0, false, false);
    
    // Compute 1/(2mc)^2
    MatsT C2 = 1./(4*SpeedOfLight*SpeedOfLight);
    
    // Component Scatter Density
    for (auto i = 0ul; i < mPDM; i++) {
      auto onePDMLL = contract1PDMLL.size() != 0 ? contract1PDMLL[i]: dummy_pauli;
      auto onePDMLS = contract1PDMLS.size() != 0 ? contract1PDMLS[i]: dummy_pauli;
      auto onePDMSL = contract1PDMSL.size() != 0 ? contract1PDMSL[i]: dummy_pauli;
      auto onePDMSS = contract1PDMSS.size() != 0 ? contract1PDMSS[i]: dummy_pauli;
      
      onePDMs[i]->componentScatter(*onePDMLL, *onePDMLS, *onePDMSL, *onePDMSS);  
    } 
    
#ifdef _PRINT_MATRICES
    prettyPrintSmart(std::cout, "1PDM[MS]", contract1PDM[0].S().pointer(), NB2C, NB2C, NB2C);
    prettyPrintSmart(std::cout, "1PDM[MX]", contract1PDM[0].X().pointer(), NB2C, NB2C, NB2C);
    prettyPrintSmart(std::cout, "1PDM[MY]", contract1PDM[0].Y().pointer(), NB2C, NB2C, NB2C);
    prettyPrintSmart(std::cout, "1PDM[MZ]", contract1PDM[0].Z().pointer(), NB2C, NB2C, NB2C);
#endif

    // Initialization 
    if(not increment) {
      for (auto i = 0ul; i < mPDM; i++) {
        if(computeCoulomb)  coulombMatrices[i]->clear(); 
        if(computeExchange) exchangeMatrices[i]->clear(); 
        if(computeTwoeHs)   twoeHs[i]->clear();
      }
    };


    /**********************************************/
    /*                                            */
    /*              DIRECT COULOMB     	          */
    /*                                            */
    /**********************************************/


    if(this->hamiltonianOptions_.BareCoulomb) { // DIRECT_COULOMB

      /*+++++++++++++++++++++++++++++++++++++++++++++*/
      /* Start of Direct Coulomb (LL|LL) Contraction */
      /*+++++++++++++++++++++++++++++++++++++++++++++*/

      std::vector<TwoBodyRelContraction<MatsT>> contractLL;

      for (auto i = 0ul; i < mPDM; i++) {
        contractLL.push_back({contract1PDMLL[i], CScrLLMS[i], HerDen, BARE_COULOMB});
        if (computeExchange) contractLL.push_back({contract1PDMLL[i], XScrLL[i]});
      } 
      
      // Call the contraction engine to do the assembly of Dirac-Coulomb LLLL
      relERICon.twoBodyRelContract(ss.comm, true, contractLL, pert, not computeExchange); 
      
      if (computeCoulomb or computeTwoeHs) { 
        for (auto i = 0ul; i < mPDM; i++) {
          coulombContainers[i]->componentAdd('N', MatsT(1.), "LL", *CScrLLMS[i]);
        }  
      }

      if (computeExchange) {
        for (auto i = 0ul; i < mPDM; i++) {
          exchangeMatrices[i]->componentAdd('N', MatsT(1.), "LL", *XScrLL[i]);
        }  
      }

 #ifdef _PRINT_MATRICES
      std::cout<<"After BARE COULOMB"<<std::endl;
      prettyPrintSmart(std::cout, "COULOMB-S",           twoeHs[0]->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "COULOMB-X",           twoeHs[0]->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "COULOMB-Y",           twoeHs[0]->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "COULOMB-Z",           twoeHs[0]->Z().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-S", exchangeMatrices[0]->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-X", exchangeMatrices[0]->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Y", exchangeMatrices[0]->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Z", exchangeMatrices[0]->Z().pointer(), NB2C, NB2C, NB2C);
#endif


      /*---------------------------------------------*/
      /*   End of Direct Coulomb (LL|LL) Contraction */
      /*---------------------------------------------*/

    } // DIRECT_COULOMB

    /**********************************************/
    /*                                            */
    /*              DIRAC-COULOMB                 */
    /*                                            */
    /**********************************************/

    if(this->hamiltonianOptions_.DiracCoulomb) { // DIRAC_COULOMB

  
      /*++++++++++++++++++++++++++++++++++++++++++++*/
      /* Start of Dirac-Coulomb (LL|LL) Contraction */
      /*++++++++++++++++++++++++++++++++++++++++++++*/
  
      if (computeCoulomb or computeTwoeHs) { 
        std::vector<TwoBodyRelContraction<MatsT>> contractDCLL;
        
        for (auto i = 0ul; i < mPDM; i++) {
          contractDCLL.push_back({contract1PDMSS[i], CScrLLMS[i], HerDen, LLLL});
          contractDCLL.push_back({contract1PDMLL[i], CScrSS[i]});
        } 

        // Call the contraction engine to do the assembly of Dirac-Coulomb LLLL
        relERICon.twoBodyRelContract(ss.comm, true, contractDCLL, pert, not computeExchange);

        // Add Dirac-Coulomb contributions to the LLLL block
        for (auto i = 0ul; i < mPDM; i++) {
          coulombContainers[i]->componentAdd('N', C2, "LL", *CScrLLMS[i]);
          coulombContainers[i]->componentAdd('N', C2, "SS", *CScrSS[i]);
        }  
      } 

      /*++++++++++++++++++++++++++++++++++++++++++++*/
      /* End of Dirac-Coulomb (LL|LL) Contraction   */
      /*++++++++++++++++++++++++++++++++++++++++++++*/

#ifdef _PRINT_MATRICES

      std::cout<<"After LLLL"<<std::endl;
      prettyPrintSmart(std::cout, "COULOMB-S",           twoeHs[0]->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "COULOMB-X",           twoeHs[0]->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "COULOMB-Y",           twoeHs[0]->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "COULOMB-Z",           twoeHs[0]->Z().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-S", exchangeMatrice[0]->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-X", exchangeMatrice[0]->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Y", exchangeMatrice[0]->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Z", exchangeMatrice[0]->Z().pointer(), NB2C, NB2C, NB2C);

#endif

      /*++++++++++++++++++++++++++++++++++++++++++++*/
      /* Start of Dirac-Coulomb (LS|SL) Contraction */
      /*++++++++++++++++++++++++++++++++++++++++++++*/

      if( computeExchange ) {
#if 0 
      std::vector<TwoBodyContraction<MatsT>> contractDCLS =
        { {contract1PDMLL.S().pointer(), CScrLLMS, HerDen, LLSS},
          {contract1PDMLL.S().pointer(), XScrLLMS},
          {contract1PDMLL.X().pointer(), XScrLLMX},
          {contract1PDMLL.Y().pointer(), XScrLLMY},
          {contract1PDMLL.Z().pointer(), XScrLLMZ},
          {contract1PDMSS.S().pointer(), CScrSSMS},
          {contract1PDMSS.X().pointer(), CScrSSMX},
          {contract1PDMSS.Y().pointer(), CScrSSMY},
          {contract1PDMSS.Z().pointer(), CScrSSMZ},
          {contract1PDMSS.S().pointer(), XScrSSMS},
          {contract1PDMSS.X().pointer(), XScrSSMX},
          {contract1PDMSS.Y().pointer(), XScrSSMY},
          {contract1PDMSS.Z().pointer(), XScrSSMZ},
          {contract1PDMLS.S().pointer(), XScrLSMS},
          {contract1PDMLS.X().pointer(), XScrLSMX},
          {contract1PDMLS.Y().pointer(), XScrLSMY},
          {contract1PDMLS.Z().pointer(), XScrLSMZ} };

      // Call the contraction engine to do the assembly of Dirac-Coulomb LLSS
      relERICon.twoBodyContract(ss.comm, true, contractDCLS, pert);

      // Add Dirac-Coulomb contributions to the LLSS block
      MatAdd('N','N', NB1C, NB1C, -C2, XScrLSMS, NB1C, MatsT(1.0), 
		      ss.exchangeMatrix->S().pointer()+LS, NB2C,
		      ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -C2, XScrLSMX, NB1C, MatsT(1.0), 
		      ss.exchangeMatrix->X().pointer()+LS, NB2C,
		      ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -C2, XScrLSMY, NB1C, MatsT(1.0), 
		      ss.exchangeMatrix->Y().pointer()+LS, NB2C,
		      ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -C2, XScrLSMZ, NB1C, MatsT(1.0), 
		      ss.exchangeMatrix->Z().pointer()+LS, NB2C,
		      ss.exchangeMatrix->Z().pointer()+LS, NB2C);
#endif


#ifdef _PRINT_MATRICES

      std::cout<<"After LLSS"<<std::endl;
      prettyPrintSmart(std::cout, "COULOMB-S",           ss.twoeH->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "COULOMB-X",           ss.twoeH->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "COULOMB-Y",           ss.twoeH->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "COULOMB-Z",           ss.twoeH->Z().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
    
#endif //_PRINT_MATRICES
      } 
    
    } //_DIRAC_COULOMB
//
//
//
//    /*************************************/
//    /*                                   */
//    /*              SSSS                 */
//    /*                                   */
//    /*************************************/
//
//    if(this->hamiltonianOptions_.DiracCoulombSSSS) { // SSSS
//
//      double C4 = 1./(16*SpeedOfLight*SpeedOfLight*SpeedOfLight*SpeedOfLight);
//  
//      /*++++++++++++++++++++++++++++++++++++++++++++*/
//      /* Start of Dirac-Coulomb (SS|SS) Contraction */
//      /*++++++++++++++++++++++++++++++++++++++++++++*/
//  
//      std::vector<TwoBodyContraction<MatsT>> contractDCSS =
//        { {contract1PDMLL.S().pointer(), CScrLLMS, HerDen, SSSS},
//          {contract1PDMLL.S().pointer(), XScrLLMS},
//          {contract1PDMLL.X().pointer(), XScrLLMX},
//          {contract1PDMLL.Y().pointer(), XScrLLMY},
//          {contract1PDMLL.Z().pointer(), XScrLLMZ},
//          {contract1PDMSS.S().pointer(), CScrSSMS},
//          {contract1PDMSS.X().pointer(), CScrSSMX},
//          {contract1PDMSS.Y().pointer(), CScrSSMY},
//          {contract1PDMSS.Z().pointer(), CScrSSMZ},
//          {contract1PDMSS.S().pointer(), XScrSSMS},
//          {contract1PDMSS.X().pointer(), XScrSSMX},
//          {contract1PDMSS.Y().pointer(), XScrSSMY},
//          {contract1PDMSS.Z().pointer(), XScrSSMZ},
//          {contract1PDMLS.S().pointer(), XScrLSMS},
//          {contract1PDMLS.X().pointer(), XScrLSMX},
//          {contract1PDMLS.Y().pointer(), XScrLSMY},
//          {contract1PDMLS.Z().pointer(), XScrLSMZ} };
//
//      // Call the contraction engine to do the assembly of Dirac-Coulomb LLLL
//      relERICon.twoBodyContract(ss.comm, true, contractDCSS,pert);
//
//      // Add (SS|SS) Coulomb contributions to the SSSS block
//      MatAdd('N','N', NB1C, NB1C, 2.0*C4, CScrSSMS, NB1C, MatsT(1.0), 
//                      ss.twoeH->S().pointer()+SS, NB2C,
//                      ss.twoeH->S().pointer()+SS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, 2.0*C4, CScrSSMX, NB1C, MatsT(1.0), 
//                      ss.twoeH->X().pointer()+SS, NB2C,
//                      ss.twoeH->X().pointer()+SS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, 2.0*C4, CScrSSMY, NB1C, MatsT(1.0), 
//                      ss.twoeH->Y().pointer()+SS, NB2C,
//                      ss.twoeH->Y().pointer()+SS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, 2.0*C4, CScrSSMZ, NB1C, MatsT(1.0), 
//                      ss.twoeH->Z().pointer()+SS, NB2C,
//                      ss.twoeH->Z().pointer()+SS, NB2C);
//
//      // Add (SS|SS) exchange contributions to the SSSS block
//      MatAdd('N','N', NB1C, NB1C, -C4, XScrSSMS, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->S().pointer()+SS, NB2C,
//                      ss.exchangeMatrix->S().pointer()+SS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C4, XScrSSMX, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->X().pointer()+SS, NB2C,
//                      ss.exchangeMatrix->X().pointer()+SS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C4, XScrSSMY, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->Y().pointer()+SS, NB2C,
//                      ss.exchangeMatrix->Y().pointer()+SS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C4, XScrSSMZ, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->Z().pointer()+SS, NB2C,
//                      ss.exchangeMatrix->Z().pointer()+SS, NB2C);
//
//
//#ifdef _PRINT_MATRICES
//      std::cout<<"After SSSS"<<std::endl;
//      prettyPrintSmart(std::cout, "COULOMB-S",           ss.twoeH->S().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "COULOMB-X",           ss.twoeH->X().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "COULOMB-Y",           ss.twoeH->Y().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "COULOMB-Z",           ss.twoeH->Z().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
//#endif
//    }
//
//
//    /*************************************/
//    /*                                   */
//    /*              GAUNT                */
//    /*                                   */
//    /*************************************/
//
//    // if the gauge term is included, the Gaunt term needs to be scaled by half
//    if(this->hamiltonianOptions_.Gauge) C2=C2/2.0;
//
//    if(this->hamiltonianOptions_.Gaunt) { // Gaunt
//
//      std::vector<TwoBodyContraction<MatsT>> contractDCGaunt =
//        { {contract1PDMLL.S().pointer(), CScrLLMS, HerDen, GAUNT},
//  	  //
//          {contract1PDMLL.S().pointer(), XScrLLMS},
//          {contract1PDMLL.X().pointer(), XScrLLMX},
//          {contract1PDMLL.Y().pointer(), XScrLLMY},
//          {contract1PDMLL.Z().pointer(), XScrLLMZ},
//	  //
//          {contract1PDMSS.S().pointer(), CScrSSMS},
//          {contract1PDMSS.X().pointer(), CScrSSMX},
//          {contract1PDMSS.Y().pointer(), CScrSSMY},
//          {contract1PDMSS.Z().pointer(), CScrSSMZ},
//	  //
//          {contract1PDMSS.S().pointer(), XScrSSMS},
//          {contract1PDMSS.X().pointer(), XScrSSMX},
//          {contract1PDMSS.Y().pointer(), XScrSSMY},
//          {contract1PDMSS.Z().pointer(), XScrSSMZ},
//	  //
//          {contract1PDMLS.S().pointer(), XScrLSMS},
//          {contract1PDMLS.X().pointer(), XScrLSMX},
//          {contract1PDMLS.Y().pointer(), XScrLSMY},
//          {contract1PDMLS.Z().pointer(), XScrLSMZ},
//	  //
//          {contract1PDMLS.S().pointer(), CScrLSMS},
//          {contract1PDMLS.X().pointer(), CScrLSMX},
//          {contract1PDMLS.Y().pointer(), CScrLSMY},
//          {contract1PDMLS.Z().pointer(), CScrLSMZ},
//	  //
//          {contract1PDMSL.S().pointer(), CScrLSMS},
//          {contract1PDMSL.X().pointer(), CScrLSMX},
//          {contract1PDMSL.Y().pointer(), CScrLSMY},
//          {contract1PDMSL.Z().pointer(), CScrLSMZ},
//	  //
//          {contract1PDMSL.S().pointer(), XScrLSMS},
//          {contract1PDMSL.X().pointer(), XScrLSMX},
//          {contract1PDMSL.Y().pointer(), XScrLSMY},
//          {contract1PDMSL.Z().pointer(), XScrLSMZ},
//	};
//
//      // Call the contraction engine to do the assembly of Gaunt
//      relERICon.twoBodyContract(ss.comm, true, contractDCGaunt,pert);
//
//      // Add (LL|SS) Coulomb contributions
//      MatAdd('N','N', NB1C, NB1C, 2.0*C2, CScrLSMS, NB1C, MatsT(1.0), 
//                      ss.twoeH->S().pointer()+LS, NB2C,
//                      ss.twoeH->S().pointer()+LS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, 2.0*C2, CScrLSMX, NB1C, MatsT(1.0), 
//                      ss.twoeH->X().pointer()+LS, NB2C,
//                      ss.twoeH->X().pointer()+LS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, 2.0*C2, CScrLSMY, NB1C, MatsT(1.0), 
//                      ss.twoeH->Y().pointer()+LS, NB2C,
//                      ss.twoeH->Y().pointer()+LS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, 2.0*C2, CScrLSMZ, NB1C, MatsT(1.0), 
//                      ss.twoeH->Z().pointer()+LS, NB2C,
//                      ss.twoeH->Z().pointer()+LS, NB2C);
//
//      // Add (LL|LL) exchange contributions
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrLLMS, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->S().pointer(), NB2C,
//                      ss.exchangeMatrix->S().pointer(), NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrLLMX, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->X().pointer(), NB2C,
//                      ss.exchangeMatrix->X().pointer(), NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrLLMY, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->Y().pointer(), NB2C,
//                      ss.exchangeMatrix->Y().pointer(), NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrLLMZ, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->Z().pointer(), NB2C,
//                      ss.exchangeMatrix->Z().pointer(), NB2C);
//
//
//      // Add (SS|SS) exchange contributions
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrSSMS, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->S().pointer()+SS, NB2C,
//                      ss.exchangeMatrix->S().pointer()+SS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrSSMX, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->X().pointer()+SS, NB2C,
//                      ss.exchangeMatrix->X().pointer()+SS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrSSMY, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->Y().pointer()+SS, NB2C,
//                      ss.exchangeMatrix->Y().pointer()+SS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrSSMZ, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->Z().pointer()+SS, NB2C,
//                      ss.exchangeMatrix->Z().pointer()+SS, NB2C);
//
//      // Add (LL|SS) exchange contributions
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrLSMS, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->S().pointer()+LS, NB2C,
//                      ss.exchangeMatrix->S().pointer()+LS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrLSMX, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->X().pointer()+LS, NB2C,
//                      ss.exchangeMatrix->X().pointer()+LS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrLSMY, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->Y().pointer()+LS, NB2C,
//                      ss.exchangeMatrix->Y().pointer()+LS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrLSMZ, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->Z().pointer()+LS, NB2C,
//                      ss.exchangeMatrix->Z().pointer()+LS, NB2C);
//
//
//
//#ifdef _PRINT_MATRICES
//      std::cout<<"After GAUNT"<<std::endl;
//      prettyPrintSmart(std::cout, "COULOMB-S",           ss.twoeH->S().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "COULOMB-X",           ss.twoeH->X().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "COULOMB-Y",           ss.twoeH->Y().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "COULOMB-Z",           ss.twoeH->Z().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
//#endif
//    }
//
//
//    /*************************************/
//    /*                                   */
//    /*              GAUGE                */
//    /*                                   */
//    /*************************************/
//
//    if(this->hamiltonianOptions_.Gauge) { // Gauge
//
//      std::vector<TwoBodyContraction<MatsT>> contractDCGauge =
//        { {contract1PDMLL.S().pointer(), CScrLLMS, HerDen, GAUGE},
//          //
//          {contract1PDMLL.S().pointer(), XScrLLMS},
//          {contract1PDMLL.X().pointer(), XScrLLMX},
//          {contract1PDMLL.Y().pointer(), XScrLLMY},
//          {contract1PDMLL.Z().pointer(), XScrLLMZ},
//          //
//          {contract1PDMSS.S().pointer(), CScrSSMS},
//          {contract1PDMSS.X().pointer(), CScrSSMX},
//          {contract1PDMSS.Y().pointer(), CScrSSMY},
//          {contract1PDMSS.Z().pointer(), CScrSSMZ},
//          //
//          {contract1PDMSS.S().pointer(), XScrSSMS},
//          {contract1PDMSS.X().pointer(), XScrSSMX},
//          {contract1PDMSS.Y().pointer(), XScrSSMY},
//          {contract1PDMSS.Z().pointer(), XScrSSMZ},
//          //
//          {contract1PDMLS.S().pointer(), XScrLSMS},
//          {contract1PDMLS.X().pointer(), XScrLSMX},
//          {contract1PDMLS.Y().pointer(), XScrLSMY},
//          {contract1PDMLS.Z().pointer(), XScrLSMZ},
//          //
//          {contract1PDMLS.S().pointer(), CScrLSMS},
//          {contract1PDMLS.X().pointer(), CScrLSMX},
//          {contract1PDMLS.Y().pointer(), CScrLSMY},
//          {contract1PDMLS.Z().pointer(), CScrLSMZ},
//          //
//          {contract1PDMSL.S().pointer(), CScrLSMS},
//          {contract1PDMSL.X().pointer(), CScrLSMX},
//          {contract1PDMSL.Y().pointer(), CScrLSMY},
//          {contract1PDMSL.Z().pointer(), CScrLSMZ},
//          //
//          {contract1PDMSL.S().pointer(), XScrLSMS},
//          {contract1PDMSL.X().pointer(), XScrLSMX},
//          {contract1PDMSL.Y().pointer(), XScrLSMY},
//          {contract1PDMSL.Z().pointer(), XScrLSMZ},
//        };
//
//      // Call the contraction engine to do the assembly of Gaunt
//      relERICon.twoBodyContract(ss.comm, true, contractDCGauge,pert);
//
//      // Add (LL|SS) Coulomb contributions
//      MatAdd('N','N', NB1C, NB1C, 2.0*C2, CScrLSMS, NB1C, MatsT(1.0), 
//                      ss.twoeH->S().pointer()+LS, NB2C,
//                      ss.twoeH->S().pointer()+LS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, 2.0*C2, CScrLSMX, NB1C, MatsT(1.0), 
//                      ss.twoeH->X().pointer()+LS, NB2C,
//                      ss.twoeH->X().pointer()+LS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, 2.0*C2, CScrLSMY, NB1C, MatsT(1.0), 
//                      ss.twoeH->Y().pointer()+LS, NB2C,
//                      ss.twoeH->Y().pointer()+LS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, 2.0*C2, CScrLSMZ, NB1C, MatsT(1.0), 
//                      ss.twoeH->Z().pointer()+LS, NB2C,
//                      ss.twoeH->Z().pointer()+LS, NB2C);
//
//      // Add (LL|LL) exchange contributions
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrLLMS, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->S().pointer(), NB2C,
//                      ss.exchangeMatrix->S().pointer(), NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrLLMX, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->X().pointer(), NB2C,
//                      ss.exchangeMatrix->X().pointer(), NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrLLMY, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->Y().pointer(), NB2C,
//                      ss.exchangeMatrix->Y().pointer(), NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrLLMZ, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->Z().pointer(), NB2C,
//                      ss.exchangeMatrix->Z().pointer(), NB2C);
//
//
//      // Add (SS|SS) exchange contributions
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrSSMS, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->S().pointer()+SS, NB2C,
//                      ss.exchangeMatrix->S().pointer()+SS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrSSMX, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->X().pointer()+SS, NB2C,
//                      ss.exchangeMatrix->X().pointer()+SS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrSSMY, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->Y().pointer()+SS, NB2C,
//                      ss.exchangeMatrix->Y().pointer()+SS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrSSMZ, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->Z().pointer()+SS, NB2C,
//                      ss.exchangeMatrix->Z().pointer()+SS, NB2C);
//
//      // Add (LL|SS) exchange contributions
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrLSMS, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->S().pointer()+LS, NB2C,
//                      ss.exchangeMatrix->S().pointer()+LS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrLSMX, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->X().pointer()+LS, NB2C,
//                      ss.exchangeMatrix->X().pointer()+LS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrLSMY, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->Y().pointer()+LS, NB2C,
//                      ss.exchangeMatrix->Y().pointer()+LS, NB2C);
//      MatAdd('N','N', NB1C, NB1C, -C2, XScrLSMZ, NB1C, MatsT(1.0), 
//                      ss.exchangeMatrix->Z().pointer()+LS, NB2C,
//                      ss.exchangeMatrix->Z().pointer()+LS, NB2C);
//
//
//
//#ifdef _PRINT_MATRICES
//      std::cout<<"After GAUGE"<<std::endl;
//      prettyPrintSmart(std::cout, "COULOMB-S",           ss.twoeH->S().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "COULOMB-X",           ss.twoeH->X().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "COULOMB-Y",           ss.twoeH->Y().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "COULOMB-Z",           ss.twoeH->Z().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
//      prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
//#endif
//    }



    /*******************************/
    /* Final Assembly of 4C Matrix */
    /*******************************/
    ROOT_ONLY(ss.comm);
    
    
    /****************************************/
    /* Hermitrize Coulomb and Exchange Part */
    /****************************************/

    if (computeCoulomb or computeTwoeHs) { 
      for (auto i = 0ul; i < mPDM; i++) {
        coulombContainers[i]->symmetrizeLSSL('C'); 
      }  
    }

    if (computeExchange) { 
      for (auto i = 0ul; i < mPDM; i++) {
        exchangeMatrices[i]->symmetrizeLSSL('C');
      }  
    }

    if (computeTwoeHs) {
      for (auto i = 0ul; i < mPDM; i++) { 
        // G[D] += 2*J[D]
        if (computeCoulomb) {
          *twoeHs[i] += 2.0 * *coulombMatrices[i];
        } else {
          *twoeHs[i] *= 2.0;
        }

        // Form GD: G[D] = 2.0*J[D] - K[D]
        if (computeExchange) {
          *twoeHs[i] -= xHFX * *exchangeMatrices[i];
        } 
      }
    }

#ifdef _PRINT_MATRICES

    prettyPrintSmart(std::cout,"twoeH MS",ss.twoeH->S().pointer(),NB2C,NB2C,NB2C);
    prettyPrintSmart(std::cout,"twoeH MX",ss.twoeH->X().pointer(),NB2C,NB2C,NB2C);
    prettyPrintSmart(std::cout,"twoeH MY",ss.twoeH->Y().pointer(),NB2C,NB2C,NB2C);
    prettyPrintSmart(std::cout,"twoeH MZ",ss.twoeH->Z().pointer(),NB2C,NB2C,NB2C);

    size_t NB4C2 = NB4C*NB4C;

    MatsT* TEMP_GATHER1 = mem.malloc<MatsT>(NB4C2);
    MatsT* TEMP_GATHER2 = mem.malloc<MatsT>(NB4C2);

    memset(TEMP_GATHER1,0.,NB4C2*sizeof(MatsT));
    memset(TEMP_GATHER2,0.,NB4C2*sizeof(MatsT));

    std::cout << std::scientific << std::setprecision(16);
    SpinGather(NB2C,TEMP_GATHER1,NB4C,contract1PDM.S().pointer(),NB2C,contract1PDM.Z().pointer(),NB2C,contract1PDM.Y().pointer(),NB2C,contract1PDM.X().pointer(),NB2C);
    prettyPrintSmart(std::cout,"density Gather",TEMP_GATHER1,NB4C,NB4C,NB4C,1,12,16);


    SpinGather(NB2C,TEMP_GATHER2,NB4C,ss.twoeH->S().pointer(),NB2C,ss.twoeH->Z().pointer(),NB2C,ss.twoeH->Y().pointer(),NB2C,ss.twoeH->X().pointer(),NB2C);
    prettyPrintSmart(std::cout,"twoeH Gather",TEMP_GATHER2,NB4C,NB4C,NB4C,1,12,16);

    SpinGather(NB2C,TEMP_GATHER1,NB4C,ss.coreH->S().pointer(),NB2C,ss.coreH->Z().pointer(),NB2C,ss.coreH->Y().pointer(),NB2C,ss.coreH->X().pointer(),NB2C);
    prettyPrintSmart(std::cout,"coreH Gather",TEMP_GATHER1,NB4C,NB4C,NB4C,1,12,16);
 
    mem.free(TEMP_GATHER1);
    mem.free(TEMP_GATHER2);

#endif //_PRINT_MATRICES


  }; // FourCompFock<MatsT, IntsT>::formRawGDInBatchesDirect

  /*******************************************************************************/
  /*                                                                             */
  /* Compute memory requirement for build 4C GD in Batches                       */
  /* Returns:                                                                    */
  /*   size_t SCR size needed for one batch                                      */
  /*   IMPORTANT HERE: size are all in MatsT (dcomplex)                          */
  /*******************************************************************************/
  template <typename MatsT, typename IntsT>
  size_t FourCompFock<MatsT,IntsT>::formRawGDSCRSizePerBatch(SingleSlater<MatsT,IntsT> &ss,
    bool CoulombOnly) const {
    
      size_t SCRSize  = 0ul;
      
      if( std::dynamic_pointer_cast<GTODirectRelERIContraction<MatsT,IntsT>>(ss.TPI) ) {
        
        GTODirectRelERIContraction<MatsT,IntsT> &relERICon =
            *std::dynamic_pointer_cast<GTODirectRelERIContraction<MatsT,IntsT>>(ss.TPI);
        
        // Update with contraction SCR 
        #define UPDATE_CONTRACTION_SCR_SIZE(CONTTYPE) \
          auto contSCR = relERICon.directRelScaffoldLibcintSCRSize(CONTTYPE, CoulombOnly); \
          SCRSize  = std::max(SCRSize, contSCR); 

        if (this->hamiltonianOptions_.BareCoulomb) {  
          UPDATE_CONTRACTION_SCR_SIZE(BARE_COULOMB);
        }
        if (this->hamiltonianOptions_.DiracCoulomb) { 
          UPDATE_CONTRACTION_SCR_SIZE(LLLL);
          if (not CoulombOnly) UPDATE_CONTRACTION_SCR_SIZE(LLSS);
        }
        if (this->hamiltonianOptions_.DiracCoulombSSSS) {
          UPDATE_CONTRACTION_SCR_SIZE(SSSS);
        }
        if (this->hamiltonianOptions_.Gaunt) {
          UPDATE_CONTRACTION_SCR_SIZE(GAUNT);
        }
        if (this->hamiltonianOptions_.Gauge) {
          UPDATE_CONTRACTION_SCR_SIZE(GAUGE);
        }
        
        // plus extra SCR to build component scattered X and AX
        // see line 155 for sepecific allocations
        bool allocate1PDMLLXYZ = this->hamiltonianOptions_.DiracCoulomb or
          this->hamiltonianOptions_.DiracCoulombSSSS or
          this->hamiltonianOptions_.Gaunt or
          this->hamiltonianOptions_.Gauge;
        
        bool allocate1PDMSS = allocate1PDMLLXYZ;
        bool allocate1PDMLS = (this->hamiltonianOptions_.DiracCoulomb and not CoulombOnly) or
          this->hamiltonianOptions_.Gaunt or
          this->hamiltonianOptions_.Gauge;
        bool allocate1PDMSL = 
          this->hamiltonianOptions_.Gaunt or
          this->hamiltonianOptions_.Gauge;
        
        bool allocateCScrLLMS = this->hamiltonianOptions_.BareCoulomb or 
          this->hamiltonianOptions_.DiracCoulomb;
        bool allocateCScrSS = this->hamiltonianOptions_.DiracCoulomb or
          this->hamiltonianOptions_.DiracCoulombSSSS;
        bool allocateCScrLS = this->hamiltonianOptions_.Gaunt or 
          this->hamiltonianOptions_.Gauge;
        
        size_t NB1C  = ss.basisSet().nBasis;
        size_t NB1C2 = NB1C*NB1C;
        // density requirements
        SCRSize += allocate1PDMLLXYZ ? NB1C2*4: NB1C2;
        SCRSize += allocate1PDMLS ? NB1C2*4: 0;
        SCRSize += allocate1PDMSL ? NB1C2*4: 0;
        SCRSize += allocate1PDMSS ? NB1C2*4: 0;
        
        // Coulomb SCR
        SCRSize += allocateCScrLLMS ? NB1C2: 0;
        SCRSize += allocateCScrSS ? NB1C2*4: 0;
        SCRSize += allocateCScrLS ? NB1C2*4: 0;
        
        // eXchange SCR
        if (not CoulombOnly) {
          bool allocateXScrLL = this->hamiltonianOptions_.BareCoulomb or 
            this->hamiltonianOptions_.Gaunt or 
            this->hamiltonianOptions_.Gauge;
          bool allocateXScrSS = this->hamiltonianOptions_.DiracCoulombSSSS or 
            this->hamiltonianOptions_.Gaunt or 
            this->hamiltonianOptions_.Gauge;
          bool allocateXScrLS = this->hamiltonianOptions_.Gaunt or 
            this->hamiltonianOptions_.Gauge;
          
          SCRSize += allocateXScrLL ? NB1C2*4: 0;
          SCRSize += allocateXScrSS ? NB1C2*4: 0;
          SCRSize += allocateXScrLS ? NB1C2*4: 0;
        }

      }
      
      return SCRSize;

  }; // FourCompFock<MatsT, IntsT>::formRawGDSCRSizePerBatch



}; // namespace ChronusQ
