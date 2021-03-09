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
#include <electronintegrals/twoeints/incore4indexreleri.hpp>
#include <electronintegrals/twoeints/gtodirectreleri.hpp>

//#define _PRINT_MATRICES

namespace ChronusQ {

  /**   
   *  \brief Forms the 4C Fock matrix
   */
  template <typename MatsT, typename IntsT>
  void FourCompFock<MatsT,IntsT>::formGD(SingleSlater<MatsT,IntsT> &ss,
    EMPerturbation &pert, bool increment, double xHFX) {

    if( std::dynamic_pointer_cast<InCore4indexRelERIContraction<MatsT,IntsT>>(ss.ERI) )
      formGDInCore(ss, pert, increment, xHFX);
    else if( std::dynamic_pointer_cast<GTODirectRelERIContraction<MatsT,IntsT>>(ss.ERI) )
      formGDDirect(ss, pert, increment, xHFX);
      //formGD3Index(ss, pert, increment, xHFX);
    else
      CErr("Unsupported ERIContraction type.");

  };


  template <typename MatsT, typename IntsT>
  void FourCompFock<MatsT,IntsT>::formGDInCore(SingleSlater<MatsT,IntsT> &ss,
    EMPerturbation &pert, bool increment, double xHFX) {

    InCore4indexRelERI<IntsT> &relERI =
        *std::dynamic_pointer_cast<InCore4indexRelERI<IntsT>>(ss.aoints.ERI);
    CQMemManager &mem = ss.memManager;

    // Decide list of onePDMs to use
    PauliSpinorSquareMatrices<MatsT> &contract1PDM
        = increment ? *ss.deltaOnePDM : *ss.onePDM;

    size_t NB1C  = ss.basisSet().nBasis;
    size_t NB2C  = 2 * ss.basisSet().nBasis;
    size_t NB4C  = 4 * ss.basisSet().nBasis;
    size_t NB1C2 = NB1C*NB1C;
    size_t NB1C4 = NB1C*NB1C*NB1C*NB1C;
    size_t NB2C2 = NB2C*NB2C;
    size_t NB4C2 = NB4C*NB4C;

    size_t SS = NB2C*NB1C+NB1C;
    size_t LS = NB2C*NB1C;
    size_t SL = NB1C;

    size_t mpiRank   = MPIRank(ss.comm);
    bool   isNotRoot = mpiRank != 0;

    PauliSpinorSquareMatrices<MatsT> exchangeMatrixLL(mem, NB1C);
    PauliSpinorSquareMatrices<MatsT> contract1PDMLL(mem, NB1C);
    PauliSpinorSquareMatrices<MatsT> contract1PDMSS(mem, NB1C);
    PauliSpinorSquareMatrices<MatsT> contract1PDMLS(mem, NB1C);
    PauliSpinorSquareMatrices<MatsT> contract1PDMSL(mem, NB1C);

    MatsT* Scr1 = mem.malloc<MatsT>(NB1C2);
    MatsT* Scr2 = mem.malloc<MatsT>(NB1C2);
    MatsT* Scr3 = mem.malloc<MatsT>(NB1C2);
    MatsT* Scr4 = mem.malloc<MatsT>(NB1C2);
    memset(Scr1,0.,NB1C2*sizeof(MatsT));
    memset(Scr2,0.,NB1C2*sizeof(MatsT));
    memset(Scr3,0.,NB1C2*sizeof(MatsT));
    memset(Scr4,0.,NB1C2*sizeof(MatsT));


    // Compute 1/(2mc)^2
    //dcomplex scale = 1.;
    //dcomplex iscale = dcomplex(0.0, 1.0);
    dcomplex scale = 1./(4*SpeedOfLight*SpeedOfLight);
    dcomplex iscale = dcomplex(0.0, 1./(4*SpeedOfLight*SpeedOfLight));



#if 0
    MatsT* DEN_GATHER = mem.malloc<MatsT>(NB4C2);
    MatsT* LS_GATHER = mem.malloc<MatsT>(NB2C2);

    for(auto i = 0; i< NB2C2; i++) {
      LS_GATHER[i] = dcomplex(std::cos(i), std::sin(i));
    }
    SetMat('N', NB2C, NB2C, MatsT(1.), LS_GATHER, NB2C, DEN_GATHER+NB4C*NB2C, NB4C);
    SetMat('C', NB2C, NB2C, MatsT(1.), LS_GATHER, NB2C, DEN_GATHER+NB2C, NB4C);
    SetMat('N', NB2C, NB2C, MatsT(1.), LS_GATHER, NB2C, DEN_GATHER, NB4C);
    SetMat('N', NB2C, NB2C, MatsT(1.), LS_GATHER, NB2C, DEN_GATHER+NB4C*NB2C+NB2C, NB4C);
    MatAdd('C','N', NB2C, NB2C, MatsT(1.0), LS_GATHER, 
		    NB2C, MatsT(1.0), DEN_GATHER, NB4C, DEN_GATHER, NB4C);
    MatAdd('C','N', NB2C, NB2C, MatsT(1.0), LS_GATHER, 
		    NB2C, MatsT(1.0), DEN_GATHER+NB4C*NB2C+NB2C, 
		    NB4C, DEN_GATHER+NB4C*NB2C+NB2C, NB4C);

    prettyPrintSmart(std::cout,"Initial 4C Density",DEN_GATHER,NB4C,NB4C,NB4C);

    std::fill_n(DEN_GATHER,NB4C2,1.0);
    SpinScatter(NB2C,DEN_GATHER, NB4C, contract1PDM.S().pointer(),
		    NB2C,contract1PDM.Z().pointer(), NB2C,contract1PDM.Y().pointer(),
		    NB2C,contract1PDM.X().pointer(), NB2C);

    mem.free(DEN_GATHER);
    mem.free(LS_GATHER);
#endif

    for(size_t i = 0; i < contract1PDM.nComponent(); i++) {
      PAULI_SPINOR_COMPS c = static_cast<PAULI_SPINOR_COMPS>(i);
      SetMat('N', NB1C, NB1C, MatsT(1.), contract1PDM[c].pointer(),    NB2C,
             contract1PDMLL[c].pointer(), NB1C);
      SetMat('N', NB1C, NB1C, MatsT(1.), contract1PDM[c].pointer()+SS, NB2C,
             contract1PDMSS[c].pointer(), NB1C);
      SetMat('N', NB1C, NB1C, MatsT(1.), contract1PDM[c].pointer()+LS, NB2C,
             contract1PDMLS[c].pointer(), NB1C);
      SetMat('N', NB1C, NB1C, MatsT(1.), contract1PDM[c].pointer()+SL, NB2C,
             contract1PDMSL[c].pointer(), NB1C);
    }

#ifdef _PRINT_MATRICES
    prettyPrintSmart(std::cout, "1PDM[MS]", contract1PDM.S().pointer(), NB2C, NB2C, NB2C);
    prettyPrintSmart(std::cout, "1PDM[MX]", contract1PDM.X().pointer(), NB2C, NB2C, NB2C);
    prettyPrintSmart(std::cout, "1PDM[MY]", contract1PDM.Y().pointer(), NB2C, NB2C, NB2C);
    prettyPrintSmart(std::cout, "1PDM[MZ]", contract1PDM.Z().pointer(), NB2C, NB2C, NB2C);
#endif

#if 0
    std::fill_n(contract1PDMLL.S().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMLL.X().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMLL.Y().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMLL.Z().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMSS.S().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMSS.X().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMSS.Y().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMSS.Z().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMLS.S().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMLS.X().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMLS.Y().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMLS.Z().pointer(),NB1C2,1.0);
#endif

    if(not increment) {
      ss.coulombMatrix->clear();
      ss.exchangeMatrix->clear();
    };



    /**********************************************/
    /*                                            */
    /*   NON-RELATIVISTIC DIRECT COULOMB          */
    /*                                            */
    /**********************************************/

    if(this->hamiltonianOptions_.BareCoulomb) { // DIRECT_COULOMB

      auto topBareCoulomb = tick();
  
      /*+++++++++++++++++++++++++++++++++++++++++++++*/
      /* Start of Direct Coulomb (LL|LL) Contraction */
      /*+++++++++++++++++++++++++++++++++++++++++++++*/
  
      std::vector<TwoBodyContraction<MatsT>> contractLL =
        { {contract1PDMLL.S().pointer(), Scr1, true, COULOMB} };
  
      // Determine how many (if any) exchange terms to calculate
      if( std::abs(xHFX) > 1e-12 )
      for(size_t i = 0; i < ss.exchangeMatrix->nComponent(); i++) {
  
        PAULI_SPINOR_COMPS c = static_cast<PAULI_SPINOR_COMPS>(i);
        contractLL.push_back(
          {contract1PDMLL[c].pointer(), exchangeMatrixLL[c].pointer(), true, EXCHANGE}
        );
      }
  
      // Zero out K[i]
      if(not increment) ss.exchangeMatrix->clear();
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractLL, pert);
  
      /* Store LL block into 2C spin scattered matrices */
      // Assemble 4C coulombMatrix
      SetMat('N', NB1C, NB1C, MatsT(1.), Scr1, NB1C, ss.coulombMatrix->pointer(), NB2C);
  
      // Assemble 4C exchangeMatrix 
      for(auto i = 0; i < ss.exchangeMatrix->nComponent();i++){
        PAULI_SPINOR_COMPS c = static_cast<PAULI_SPINOR_COMPS>(i);
        SetMat('N', NB1C, NB1C, MatsT(1.), exchangeMatrixLL[c].pointer(), NB1C,
               (*ss.exchangeMatrix)[c].pointer(), NB2C);
      }
 
      /*---------------------------------------------*/
      /*   End of Direct Coulomb (LL|LL) Contraction */
      /*---------------------------------------------*/

      // Print out BareCoulomb duration 
      auto durBareCoulomb = tock(topBareCoulomb);
//      std::cout << "Non-relativistic Coulomb duration = " << durBareCoulomb << std::endl;
 
    } // DIRECT_COULOMB





    /**********************************************/
    /*                                            */
    /*              DIRAC-COULOMB      	          */
    /*                                            */
    /**********************************************/

    // ERI: (ab|cd)
    // ERIDCB0: ∇A∙∇B(ab|cd)
    // ERIDCB1: ∇Ax∇B(ab|cd)-X
    // ERIDCB2: ∇Ax∇B(ab|cd)-Y
    // ERIDCB3: ∇Ax∇B(ab|cd)-Z

    if(this->hamiltonianOptions_.DiracCoulomb) { // DIRAC_COULOMB

      auto topERIDC = tick();

      /*++++++++++++++++++++++++++++++++++++++++++++*/
      /* Start of Dirac-Coulomb (LL|LL) Contraction */
      /*++++++++++++++++++++++++++++++++++++++++++++*/

      std::vector<TwoBodyContraction<MatsT>> contractDCLL =
        { {contract1PDMSS.S().pointer(), Scr1, true, COULOMB, relERI[0].pointer(), TRANS_MNKL},
          {contract1PDMSS.X().pointer(), Scr2, true, COULOMB, relERI[1].pointer(), TRANS_MNKL},
          {contract1PDMSS.Y().pointer(), Scr3, true, COULOMB, relERI[2].pointer(), TRANS_MNKL},
          {contract1PDMSS.Z().pointer(), Scr4, true, COULOMB, relERI[3].pointer(), TRANS_MNKL} };
  
      // Call the contraction engine to do the assembly of Dirac-Coulomb LLLL
      ss.ERI->twoBodyContract(ss.comm, contractDCLL);
  
      // Add Dirac-Coulomb contributions  to the LLLL block
      MatAdd('N','N', NB1C, NB1C,  scale, Scr1, NB1C, MatsT(1.0), 
		      ss.coulombMatrix->pointer(), NB2C, ss.coulombMatrix->pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C, iscale, Scr2, NB1C, MatsT(1.0), 
		      ss.coulombMatrix->pointer(), NB2C, ss.coulombMatrix->pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C, iscale, Scr3, NB1C, MatsT(1.0), 
		      ss.coulombMatrix->pointer(), NB2C, ss.coulombMatrix->pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C, iscale, Scr4, NB1C, MatsT(1.0), 
		      ss.coulombMatrix->pointer(), NB2C, ss.coulombMatrix->pointer(), NB2C);
  

#ifdef _PRINT_MATRICES

      std::cout<<"After LLLL"<<std::endl;
      prettyPrintSmart(std::cout, "COULOMB",    ss.coulombMatrix->pointer(),      NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);

#endif //_PRINT_MATRICES

      /*------------------------------------------*/
      /* End of Dirac-Coulomb (LL|LL) Contraction */
      /*------------------------------------------*/
  
  
  
  

      /*++++++++++++++++++++++++++++++++++++++++++++*/
      /* Start of Dirac-Coulomb (SS|SS) Contraction */
      /*++++++++++++++++++++++++++++++++++++++++++++*/
  
      std::vector<TwoBodyContraction<MatsT>> contractSS =
      { {contract1PDMLL.S().pointer(), Scr1, true, COULOMB, relERI[0].pointer()},
        {contract1PDMLL.S().pointer(), Scr2, true, COULOMB, relERI[1].pointer()},
        {contract1PDMLL.S().pointer(), Scr3, true, COULOMB, relERI[2].pointer()},
        {contract1PDMLL.S().pointer(), Scr4, true, COULOMB, relERI[3].pointer()} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractSS);
  
      // Store SS block into 2C spin scattered matrices 
      // These scaling factors were modified to take into account the issue of storing the 
      // Coulomb portion in the exchange matrix, this will be fixed later
      SetMat('N', NB1C, NB1C, MatsT(scale),       Scr1, NB1C, ss.coulombMatrix->pointer()+SS,      NB2C);
      SetMat('N', NB1C, NB1C, MatsT(-2.0*iscale), Scr2, NB1C, ss.exchangeMatrix->X().pointer()+SS, NB2C);
      SetMat('N', NB1C, NB1C, MatsT(-2.0*iscale), Scr3, NB1C, ss.exchangeMatrix->Y().pointer()+SS, NB2C);
      SetMat('N', NB1C, NB1C, MatsT(-2.0*iscale), Scr4, NB1C, ss.exchangeMatrix->Z().pointer()+SS, NB2C);
  
#ifdef _PRINT_MATRICES
  
      std::cout<<"After SSSS"<<std::endl;
      prettyPrintSmart(std::cout, "COULOMB",    ss.coulombMatrix->pointer(),      NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
  
#endif //_PRINT_MATRICES
  
  
      /*--------------------------------------------*/
      /* End of Dirac-Coulomb (SS|SS) Contraction */
      /*--------------------------------------------*/
  
  
  
  
  
      /*++++++++++++++++++++++++++++++++++++++++++*/
      /* Start of Dirac-Coulomb (LL|SS) / (SS|LL) */
      /*++++++++++++++++++++++++++++++++++++++++++*/
  
      std::vector<TwoBodyContraction<MatsT>> contractLSScalar =
        { {contract1PDMLS.S().pointer(), Scr1, true, EXCHANGE, relERI[0].pointer(), TRANS_MNKL},
          {contract1PDMLS.X().pointer(), Scr2, true, EXCHANGE, relERI[1].pointer(), TRANS_MNKL},
          {contract1PDMLS.Y().pointer(), Scr3, true, EXCHANGE, relERI[2].pointer(), TRANS_MNKL},
          {contract1PDMLS.Z().pointer(), Scr4, true, EXCHANGE, relERI[3].pointer(), TRANS_MNKL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractLSScalar);
  
      // Add to the LS part of the exchangeMatrix[MS]
      MatAdd('N','N', NB1C, NB1C,  scale, Scr1, NB1C, MatsT(1.0),
             ss.exchangeMatrix->S().pointer()+LS, NB2C,
             ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, iscale, Scr2, NB1C, MatsT(1.0),
             ss.exchangeMatrix->S().pointer()+LS, NB2C,
             ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, iscale, Scr3, NB1C, MatsT(1.0),
             ss.exchangeMatrix->S().pointer()+LS, NB2C,
             ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, iscale, Scr4, NB1C, MatsT(1.0),
             ss.exchangeMatrix->S().pointer()+LS, NB2C,
             ss.exchangeMatrix->S().pointer()+LS, NB2C);
  
  
  
      std::vector<TwoBodyContraction<MatsT>> contractLSMX =
        { {contract1PDMLS.X().pointer(), Scr1, true, EXCHANGE, relERI[0].pointer(), TRANS_MNKL},
          {contract1PDMLS.S().pointer(), Scr2, true, EXCHANGE, relERI[1].pointer(), TRANS_MNKL},
          {contract1PDMLS.Z().pointer(), Scr3, true, EXCHANGE, relERI[2].pointer(), TRANS_MNKL},
          {contract1PDMLS.Y().pointer(), Scr4, true, EXCHANGE, relERI[3].pointer(), TRANS_MNKL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractLSMX);
  
      // Add to the LS part of the exchangeMatrix[MX]
      MatAdd('N','N', NB1C, NB1C,  scale, Scr1, NB1C,
             MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C,
             ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, iscale, Scr2, NB1C,
             MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C,
             ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C,  scale, Scr3, NB1C,
             MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C,
             ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -scale, Scr4, NB1C,
             MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C,
             ss.exchangeMatrix->X().pointer()+LS, NB2C);
  
  
  
  
  
      std::vector<TwoBodyContraction<MatsT>> contractLSMY =
        { {contract1PDMLS.Y().pointer(), Scr1, true, EXCHANGE, relERI[0].pointer(), TRANS_MNKL},
          {contract1PDMLS.Z().pointer(), Scr2, true, EXCHANGE, relERI[1].pointer(), TRANS_MNKL},
          {contract1PDMLS.S().pointer(), Scr3, true, EXCHANGE, relERI[2].pointer(), TRANS_MNKL},
          {contract1PDMLS.X().pointer(), Scr4, true, EXCHANGE, relERI[3].pointer(), TRANS_MNKL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractLSMY);
  
      // Add to the LS part of the exchangeMatrix[MY]
      MatAdd('N','N', NB1C, NB1C,  scale, Scr1, NB1C,
             MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C,
             ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -scale, Scr2, NB1C,
             MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C,
             ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, iscale, Scr3, NB1C,
             MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C,
             ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C,  scale, Scr4, NB1C,
             MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C,
             ss.exchangeMatrix->Y().pointer()+LS, NB2C);
  
  
  
  
  
      std::vector<TwoBodyContraction<MatsT>> contractLSMZ =
        { {contract1PDMLS.Z().pointer(), Scr1, true, EXCHANGE, relERI[0].pointer(), TRANS_MNKL},
          {contract1PDMLS.Y().pointer(), Scr2, true, EXCHANGE, relERI[1].pointer(), TRANS_MNKL},
          {contract1PDMLS.X().pointer(), Scr3, true, EXCHANGE, relERI[2].pointer(), TRANS_MNKL},
          {contract1PDMLS.S().pointer(), Scr4, true, EXCHANGE, relERI[3].pointer(), TRANS_MNKL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractLSMZ);
  
      // Add to the LS part of the exchangeMatrix[MZ]
      MatAdd('N','N', NB1C, NB1C,  scale, Scr1, NB1C,
             MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C,
             ss.exchangeMatrix->Z().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C,  scale, Scr2, NB1C,
             MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C,
             ss.exchangeMatrix->Z().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -scale, Scr3, NB1C,
             MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C,
             ss.exchangeMatrix->Z().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, iscale, Scr4, NB1C,
             MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C,
             ss.exchangeMatrix->Z().pointer()+LS, NB2C);
  
  
  
#ifdef _PRINT_MATRICES
  
      std::cout<<"After Dirac-Coulomb"<<std::endl;
      prettyPrintSmart(std::cout, "COULOMB",    ss.coulombMatrix->pointer(),      NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
  
#endif //_PRINT_MATRICES
  
      /*------------------------------------------*/
      /*   End of Dirac-Coulomb (LL|SS) / (SS|LL) */
      /*------------------------------------------*/
  
      auto durERIDC = tock(topERIDC);
//      std::cout << "Dirac-Coulomb duration   = " << durERIDC << std::endl;
    }
  
  





    /**********************************************/
    /*                                            */
    /*              GAUNT                         */
    /*                                            */
    /**********************************************/


    //ERI4:    ∇B∙∇C(mn|kl)
    //ERI5 :   ∇Bx∇C(mn|kl)-X
    //ERI6 :   ∇Bx∇C(mn|kl)-Y
    //ERI7 :   ∇Bx∇C(mn|kl)-Z
    //ERI8 :   ∇B_x∇C_y(mn|kl) + ∇B_y∇C_x(mn|kl)
    //ERI9 :   ∇B_y∇C_x(mn|kl)
    //ERI10:   ∇B_x∇C_z(mn|kl) + ∇B_z∇C_x(mn|kl)
    //ERI11:   ∇B_z∇C_x(mn|kl)
    //ERI12:   ∇B_y∇C_z(mn|kl) + ∇B_z∇C_y(mn|kl)
    //ERI13:   ∇B_z∇C_y(mn|kl)
    //ERI14: - ∇B_x∇C_x(mn|kl) - ∇B_y∇C_y(mn|kl) + ∇B_z∇C_z(mn|kl)
    //ERI15:   ∇B_x∇C_x(mn|kl) - ∇B_y∇C_y(mn|kl) - ∇B_z∇C_z(mn|kl)
    //ERI16: - ∇B_x∇C_x(mn|kl) + ∇B_y∇C_y(mn|kl) - ∇B_z∇C_z(mn|kl)
    //ERI17:   ∇B_x∇C_x(mn|kl)
    //ERI18:   ∇B_x∇C_y(mn|kl)
    //ERI19:   ∇B_x∇C_z(mn|kl)
    //ERI20:   ∇B_y∇C_y(mn|kl)
    //ERI21:   ∇B_y∇C_z(mn|kl)
    //ERI22:   ∇B_z∇C_z(mn|kl)


    if(this->hamiltonianOptions_.Gaunt) {//GAUNT

      auto topERIDG = tick();
  
#if 0 // Gaunt LLLL Spin-Free
      /* Gaunt LLLL Spin-Free */
      std::vector<TwoBodyContraction<MatsT>> contractGLLSF94 =
        { {contract1PDMSS.S().pointer(), Scr1, true, EXCHANGE, relERI[4].pointer()},
          {contract1PDMSS.X().pointer(), Scr2, true, EXCHANGE, relERI[4].pointer()},
          {contract1PDMSS.Y().pointer(), Scr3, true, EXCHANGE, relERI[4].pointer()},
          {contract1PDMSS.Z().pointer(), Scr4, true, EXCHANGE, relERI[4].pointer()} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLLSF94);
  
      // Add to the LL part of 4C exchangeMatrix in Pauli matrix form
      MatAdd('N','N', NB1C, NB1C, -scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer(), NB2C, ss.exchangeMatrix->S().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C, -scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer(), NB2C, ss.exchangeMatrix->X().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C, -scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer(), NB2C, ss.exchangeMatrix->Y().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C, -scale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer(), NB2C, ss.exchangeMatrix->Z().pointer(), NB2C);
#endif // Gaunt LLLL Spin-Orbit
  
#if 0 // Gaunt LLLL Spin-Orbit
      /* Gaunt LLLL Spin-Orbit */
      /* Equation (103) */
      std::vector<TwoBodyContraction<MatsT>> contractGLLSO103 =
        { {contract1PDMSS.X().pointer(), Scr2, true, EXCHANGE, relERI[5].pointer()},
          {contract1PDMSS.Y().pointer(), Scr3, true, EXCHANGE, relERI[6].pointer()},
          {contract1PDMSS.Z().pointer(), Scr4, true, EXCHANGE, relERI[7].pointer()} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLLSO103);
  
      // Add to the LL part of 4C exchangeMatrix in Pauli matrix form
      MatAdd('N','N', NB1C, NB1C, 2.0*iscale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer(), NB2C, ss.exchangeMatrix->S().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C, 2.0*iscale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer(), NB2C, ss.exchangeMatrix->S().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C, 2.0*iscale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer(), NB2C, ss.exchangeMatrix->S().pointer(), NB2C);
  
  
      /* Equation (104)-(106) */
      std::vector<TwoBodyContraction<MatsT>> contractGLLSO104106 =
        { {contract1PDMSS.S().pointer(), Scr1, true, EXCHANGE, relERI[5].pointer()},
          {contract1PDMSS.S().pointer(), Scr2, true, EXCHANGE, relERI[6].pointer()},
          {contract1PDMSS.S().pointer(), Scr3, true, EXCHANGE, relERI[7].pointer()} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLLSO104106);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, 2.0*iscale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer(), NB2C, ss.exchangeMatrix->X().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C, 2.0*iscale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer(), NB2C, ss.exchangeMatrix->Y().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C, 2.0*iscale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer(), NB2C, ss.exchangeMatrix->Z().pointer(), NB2C);
  
#endif // Gaunt LLLL Spin-Orbit
  
#if 1 //Gaunt LLLL
      /*++++++++++++++++++++++++++++++++++++*/
      /* Start of Gaunt (LL|LL) Contraction */
      /*++++++++++++++++++++++++++++++++++++*/
  
      /* Equation (113) */
      std::vector<TwoBodyContraction<MatsT>> contractGLL113 =
        { {contract1PDMSS.S().pointer(), Scr1, true, EXCHANGE, relERI[4].pointer()},
          {contract1PDMSS.X().pointer(), Scr2, true, EXCHANGE, relERI[5].pointer()},
          {contract1PDMSS.Y().pointer(), Scr3, true, EXCHANGE, relERI[6].pointer()},
          {contract1PDMSS.Z().pointer(), Scr4, true, EXCHANGE, relERI[7].pointer()} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLL113);
  
      // Add to the LL part of 4C exchangeMatrix in Pauli matrix form
      MatAdd('N','N', NB1C, NB1C, -3.0*scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer(), NB2C, ss.exchangeMatrix->S().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C, 3.0*iscale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer(), NB2C, ss.exchangeMatrix->S().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C, 3.0*iscale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer(), NB2C, ss.exchangeMatrix->S().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C, 3.0*iscale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer(), NB2C, ss.exchangeMatrix->S().pointer(), NB2C);
  
  
      /* Equation (114) */
      std::vector<TwoBodyContraction<MatsT>> contractGLL114 =
        { {contract1PDMSS.Z().pointer(), Scr1, true, EXCHANGE, relERI[14].pointer()},
          {contract1PDMSS.S().pointer(), Scr2, true, EXCHANGE, relERI[7].pointer()},
          {contract1PDMSS.X().pointer(), Scr3, true, EXCHANGE, relERI[10].pointer()},
          {contract1PDMSS.Y().pointer(), Scr4, true, EXCHANGE, relERI[12].pointer()} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLL114);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C,  scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer(), NB2C, ss.exchangeMatrix->Z().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C, iscale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer(), NB2C, ss.exchangeMatrix->Z().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C,  scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer(), NB2C, ss.exchangeMatrix->Z().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C,  scale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer(), NB2C, ss.exchangeMatrix->Z().pointer(), NB2C);
  
  
  
      /* Equation (115) */
      std::vector<TwoBodyContraction<MatsT>> contractGLL115 =
        { {contract1PDMSS.X().pointer(), Scr1, true, EXCHANGE, relERI[15].pointer()},
          {contract1PDMSS.S().pointer(), Scr2, true, EXCHANGE, relERI[5].pointer()},
          {contract1PDMSS.Y().pointer(), Scr3, true, EXCHANGE, relERI[8].pointer()},
          {contract1PDMSS.Z().pointer(), Scr4, true, EXCHANGE, relERI[10].pointer()} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLL115);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C,  scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer(), NB2C, ss.exchangeMatrix->X().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C, iscale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer(), NB2C, ss.exchangeMatrix->X().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C,  scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer(), NB2C, ss.exchangeMatrix->X().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C,  scale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer(), NB2C, ss.exchangeMatrix->X().pointer(), NB2C);
  
  
  
      /* Equation (116) */
      std::vector<TwoBodyContraction<MatsT>> contractGLL116 =
        { {contract1PDMSS.Y().pointer(), Scr1, true, EXCHANGE, relERI[16].pointer()},
          {contract1PDMSS.S().pointer(), Scr2, true, EXCHANGE, relERI[6].pointer()},
          {contract1PDMSS.X().pointer(), Scr3, true, EXCHANGE, relERI[8].pointer()},
          {contract1PDMSS.Z().pointer(), Scr4, true, EXCHANGE, relERI[12].pointer()} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLL116);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C,  scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer(), NB2C, ss.exchangeMatrix->Y().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C, iscale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer(), NB2C, ss.exchangeMatrix->Y().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C,  scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer(), NB2C, ss.exchangeMatrix->Y().pointer(), NB2C);
      MatAdd('N','N', NB1C, NB1C,  scale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer(), NB2C, ss.exchangeMatrix->Y().pointer(), NB2C);
  
  
#endif //Gaunt LLLL
  
#ifdef _PRINT_MATRICES
  
      std::cout<<"After Gaunt LLLL"<<std::endl;
      prettyPrintSmart(std::cout, "COULOMB",    ss.coulombMatrix->pointer(),      NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
  
#endif //_PRINT_MATRICES
  
      auto RESET = false;
      if(RESET) {
        ss.coulombMatrix->clear();
        ss.exchangeMatrix->clear();
      }
  
  
  
      /*----------------------------------*/
      /* End of Gaunt (LL|LL) Contraction */
      /*----------------------------------*/
  
  
  
  
      /*++++++++++++++++++++++++++++++++++++*/
      /* Start of Gaunt (SS|SS) Contraction */
      /*++++++++++++++++++++++++++++++++++++*/
  
#if 0 // Gaunt SSSS Spin-Free
      /* Gaunt SSSS Spin-Free */
      /* Equation (118) */
      std::vector<TwoBodyContraction<MatsT>> contractGSSSF118 =
        { {contract1PDMLL.S().pointer(), Scr1, true, EXCHANGE, relERI[4].pointer(), 1},
          {contract1PDMLL.X().pointer(), Scr2, true, EXCHANGE, relERI[4].pointer(), 1},
          {contract1PDMLL.Y().pointer(), Scr3, true, EXCHANGE, relERI[4].pointer(), 1},
          {contract1PDMLL.Z().pointer(), Scr4, true, EXCHANGE, relERI[4].pointer(), 1} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGSSSF118);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, -scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+SS, NB2C, ss.exchangeMatrix->S().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+SS, NB2C, ss.exchangeMatrix->X().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+SS, NB2C, ss.exchangeMatrix->Y().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -scale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+SS, NB2C, ss.exchangeMatrix->Z().pointer()+SS, NB2C);
#endif // Gaunt SSSS Spin-Free
  
#if 0 // Gaunt SSSS Spin-Orbit
      /* Gaunt SSSS Spin-Orbit */
      /* Equation (119) */
      std::vector<TwoBodyContraction<MatsT>> contractGSSSO119 =
        { {contract1PDMLL.X().pointer(), Scr2, true, EXCHANGE, relERI[5].pointer(), 1},
          {contract1PDMLL.Y().pointer(), Scr3, true, EXCHANGE, relERI[6].pointer(), 1},
          {contract1PDMLL.Z().pointer(), Scr4, true, EXCHANGE, relERI[7].pointer(), 1} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGSSSO119);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, 2.0*iscale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+SS, NB2C, ss.exchangeMatrix->S().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C, 2.0*iscale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+SS, NB2C, ss.exchangeMatrix->S().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C, 2.0*iscale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+SS, NB2C, ss.exchangeMatrix->S().pointer()+SS, NB2C);
  
      /* Equation (120)-(122) */
      std::vector<TwoBodyContraction<MatsT>> contractGSSSO120122 =
        { {contract1PDMLL.S().pointer(), Scr1, true, EXCHANGE, relERI[5].pointer(), 1},
          {contract1PDMLL.S().pointer(), Scr2, true, EXCHANGE, relERI[6].pointer(), 1},
          {contract1PDMLL.S().pointer(), Scr3, true, EXCHANGE, relERI[7].pointer(), 1} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGSSSO120122);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, 2.0*iscale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+SS, NB2C, ss.exchangeMatrix->X().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C, 2.0*iscale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+SS, NB2C, ss.exchangeMatrix->Y().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C, 2.0*iscale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+SS, NB2C, ss.exchangeMatrix->Z().pointer()+SS, NB2C);
  
#endif // Gaunt SSSS Spin-Orbit
  
#if 1 //Gaunt SSSS
  
      /* Equation (129) */
      std::vector<TwoBodyContraction<MatsT>> contractGSS129 =
        { {contract1PDMLL.S().pointer(), Scr1, true, EXCHANGE, relERI[4].pointer(), TRANS_MNKL},
          {contract1PDMLL.X().pointer(), Scr2, true, EXCHANGE, relERI[5].pointer(), TRANS_MNKL},
          {contract1PDMLL.Y().pointer(), Scr3, true, EXCHANGE, relERI[6].pointer(), TRANS_MNKL},
          {contract1PDMLL.Z().pointer(), Scr4, true, EXCHANGE, relERI[7].pointer(), TRANS_MNKL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGSS129);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C,-3.0*scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+SS, NB2C, ss.exchangeMatrix->S().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C,    iscale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+SS, NB2C, ss.exchangeMatrix->S().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C,    iscale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+SS, NB2C, ss.exchangeMatrix->S().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C,    iscale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+SS, NB2C, ss.exchangeMatrix->S().pointer()+SS, NB2C);
  
      /* Equation (130) */
      std::vector<TwoBodyContraction<MatsT>> contractGSS130 =
        { {contract1PDMLL.Z().pointer(), Scr1, true, EXCHANGE, relERI[14].pointer(), TRANS_MNKL},
          {contract1PDMLL.S().pointer(), Scr2, true, EXCHANGE, relERI[7].pointer(),  TRANS_MNKL},
          {contract1PDMLL.X().pointer(), Scr3, true, EXCHANGE, relERI[10].pointer(), TRANS_MNKL},
          {contract1PDMLL.Y().pointer(), Scr4, true, EXCHANGE, relERI[12].pointer(), TRANS_MNKL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGSS130);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C,      scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+SS, NB2C, ss.exchangeMatrix->Z().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C, 3.0*iscale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+SS, NB2C, ss.exchangeMatrix->Z().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C,      scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+SS, NB2C, ss.exchangeMatrix->Z().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C,      scale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+SS, NB2C, ss.exchangeMatrix->Z().pointer()+SS, NB2C);
  
  
      /* Equation (131) */
      std::vector<TwoBodyContraction<MatsT>> contractGSS131 =
        { {contract1PDMLL.X().pointer(), Scr1, true, EXCHANGE, relERI[15].pointer(), TRANS_MNKL},
          {contract1PDMLL.S().pointer(), Scr2, true, EXCHANGE, relERI[5].pointer(),  TRANS_MNKL},
          {contract1PDMLL.Z().pointer(), Scr3, true, EXCHANGE, relERI[10].pointer(), TRANS_MNKL},
          {contract1PDMLL.Y().pointer(), Scr4, true, EXCHANGE, relERI[8].pointer(),  TRANS_MNKL}};
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGSS131);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C,      scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+SS, NB2C, ss.exchangeMatrix->X().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C, 3.0*iscale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+SS, NB2C, ss.exchangeMatrix->X().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C,      scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+SS, NB2C, ss.exchangeMatrix->X().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C,      scale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+SS, NB2C, ss.exchangeMatrix->X().pointer()+SS, NB2C);
  
  
      /* Equation (132) */
      std::vector<TwoBodyContraction<MatsT>> contractGSS132 =
        { {contract1PDMLL.Y().pointer(), Scr1, true, EXCHANGE, relERI[16].pointer(), TRANS_MNKL},
          {contract1PDMLL.S().pointer(), Scr2, true, EXCHANGE, relERI[6].pointer(),  TRANS_MNKL},
          {contract1PDMLL.X().pointer(), Scr3, true, EXCHANGE, relERI[8].pointer(),  TRANS_MNKL},
          {contract1PDMLL.Z().pointer(), Scr4, true, EXCHANGE, relERI[12].pointer(), TRANS_MNKL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGSS132);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C,      scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+SS, NB2C, ss.exchangeMatrix->Y().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C, 3.0*iscale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+SS, NB2C, ss.exchangeMatrix->Y().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C,      scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+SS, NB2C, ss.exchangeMatrix->Y().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C,      scale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+SS, NB2C, ss.exchangeMatrix->Y().pointer()+SS, NB2C);
  
#endif // Gaunt SSSS
  
#ifdef _PRINT_MATRICES
  
      std::cout<<"After Gaunt SSSS"<<std::endl;
      prettyPrintSmart(std::cout, "COULOMB",    ss.coulombMatrix->pointer(),      NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
  
#endif //_PRINT_MATRICES
  
      if(RESET) {
        ss.coulombMatrix->clear();
        ss.exchangeMatrix->clear();
      }
      /*------------------------------------*/
      /*   End of Gaunt (SS|SS) Contraction */
      /*------------------------------------*/
  
  
  
  
  
  
      /*++++++++++++++++++++++++++++++++++++*/
      /* Start of Gaunt (LL|SS) Contraction */
      /*++++++++++++++++++++++++++++++++++++*/
  
#if 0 // Gaunt LLSS Spin-Free
      /* Gaunt LLSS Spin-Free */
      /* First term in Equations (91) and (136) */
      std::vector<TwoBodyContraction<MatsT>> contractGLSSF91136 =
        { {contract1PDMLS.S().pointer(), Scr1, true, COULOMB, relERI[4].pointer()},
          {contract1PDMSL.S().pointer(), Scr2, true, COULOMB, relERI[4].pointer(), 2} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLSSF91136);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, 2.0*scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C,-2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
  
      /* Gaunt LLSS Spin-Free */
      std::vector<TwoBodyContraction<MatsT>> contractGLSSF140 =
        { {contract1PDMSL.S().pointer(), Scr1, true, EXCHANGE, relERI[4].pointer(), 2},
          {contract1PDMSL.X().pointer(), Scr2, true, EXCHANGE, relERI[4].pointer(), 2},
          {contract1PDMSL.Y().pointer(), Scr3, true, EXCHANGE, relERI[4].pointer(), 2},
          {contract1PDMSL.Z().pointer(), Scr4, true, EXCHANGE, relERI[4].pointer(), 2} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLSSF140);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, scale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
#endif // Gaunt LLSS Spin-Free
  
#if 0 // Gaunt LLSS Spin-Orbit
      /* Gaunt LLSS Spin-Orbit */
  
      /* Equation (91) second term */
      std::vector<TwoBodyContraction<MatsT>> contractGLSSO91 =
        { {contract1PDMLS.X().pointer(), Scr2, true, COULOMB, relERI[5].pointer()},
          {contract1PDMLS.Y().pointer(), Scr3, true, COULOMB, relERI[6].pointer()},
          {contract1PDMLS.Z().pointer(), Scr4, true, COULOMB, relERI[7].pointer()} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLSSO91);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, -2.0*iscale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*iscale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*iscale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
  
  
      /* Equation (92) first term */
      std::vector<TwoBodyContraction<MatsT>> contractGLSSO92 =
        { {contract1PDMLS.S().pointer(), Scr1, true, COULOMB, relERI[5].pointer()},
          {contract1PDMLS.S().pointer(), Scr2, true, COULOMB, relERI[6].pointer()},
          {contract1PDMLS.S().pointer(), Scr3, true, COULOMB, relERI[7].pointer()} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLSSO92);
  
       // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, -2.0*iscale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*iscale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*iscale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
  
  
      /* Equation (136) second term*/
      std::vector<TwoBodyContraction<MatsT>> contractGLSSO136 =
        { {contract1PDMSL.X().pointer(), Scr2, true, COULOMB, relERI[5].pointer(), 2},
          {contract1PDMSL.Y().pointer(), Scr3, true, COULOMB, relERI[6].pointer(), 2},
          {contract1PDMSL.Z().pointer(), Scr4, true, COULOMB, relERI[7].pointer(), 2} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLSSO136);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, -2.0*iscale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*iscale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*iscale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
  
      /* Equation (137) first term */
      std::vector<TwoBodyContraction<MatsT>> contractGLSSO137 =
        { {contract1PDMSL.S().pointer(), Scr1, true, COULOMB, relERI[5].pointer(), 2},
          {contract1PDMSL.S().pointer(), Scr1, true, COULOMB, relERI[6].pointer(), 2},
          {contract1PDMSL.S().pointer(), Scr1, true, COULOMB, relERI[7].pointer(), 2} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLSSO137);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, 2.0*iscale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, 2.0*iscale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, 2.0*iscale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
  
  
      /* Equation (150) */
      std::vector<TwoBodyContraction<MatsT>> contractGLSSO150 =
        { {contract1PDMSL.Y().pointer(), Scr2, true, EXCHANGE, relERI[5].pointer(), 2},
          {contract1PDMSL.X().pointer(), Scr3, true, EXCHANGE, relERI[6].pointer(), 2} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLSSO150);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, 2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C,-2.0*scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
  
      
  
      /* Equation (151) */
      std::vector<TwoBodyContraction<MatsT>> contractGLSSO151 =
        { {contract1PDMSL.Y().pointer(), Scr2, true, EXCHANGE, relERI[7].pointer(), 2},
          {contract1PDMSL.Z().pointer(), Scr3, true, EXCHANGE, relERI[6].pointer(), 2} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLSSO151);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C,-2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, 2.0*scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
  
  
  
      /* Equation (152) */
      std::vector<TwoBodyContraction<MatsT>> contractGLSSO152 =
        { {contract1PDMSL.X().pointer(), Scr2, true, EXCHANGE, relERI[7].pointer(), 2},
          {contract1PDMSL.Z().pointer(), Scr3, true, EXCHANGE, relERI[5].pointer(), 2} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLSSO152);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, 2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C,-2.0*scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
  
  
      std::cout<<"After Gaunt LLSS Spin-Orbit "<<std::endl;
      prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
  
#endif // Gaunt LLSS Spin-Orbit
  
  
#if 1 // Gaunt LLSS COULOMB
  
      /* Equation (91) */
      std::vector<TwoBodyContraction<MatsT>> contractGLS91 =
        { {contract1PDMLS.S().pointer(), Scr1, true, COULOMB, relERI[4].pointer()},
          {contract1PDMLS.X().pointer(), Scr2, true, COULOMB, relERI[5].pointer()},
          {contract1PDMLS.Y().pointer(), Scr3, true, COULOMB, relERI[6].pointer()},
          {contract1PDMLS.Z().pointer(), Scr4, true, COULOMB, relERI[7].pointer()} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS91);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C,   2.0*scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*iscale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*iscale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*iscale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
  
  
      /* Equation (92)X first two terms */
      std::vector<TwoBodyContraction<MatsT>> contractGLS92AX =
        { {contract1PDMLS.S().pointer(), Scr1, true, COULOMB, relERI[5].pointer()},
          {contract1PDMLS.X().pointer(), Scr2, true, COULOMB, relERI[4].pointer()} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS92AX);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, -2.0*iscale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C,   2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
  
  
      /* Equation (92)X last term */
      std::vector<TwoBodyContraction<MatsT>> contractGLS92BX =
        { {contract1PDMLS.X().pointer(), Scr1, true, COULOMB, relERI[17].pointer()},
          {contract1PDMLS.Y().pointer(), Scr2, true, COULOMB, relERI[9].pointer()},
          {contract1PDMLS.Z().pointer(), Scr3, true, COULOMB, relERI[11].pointer()} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS92BX);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, -2.0*scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
  
  
      /* Equation (92)Y first two terms */
      std::vector<TwoBodyContraction<MatsT>> contractGLS92AY =
        { {contract1PDMLS.S().pointer(), Scr1, true, COULOMB, relERI[6].pointer()},
          {contract1PDMLS.Y().pointer(), Scr2, true, COULOMB, relERI[4].pointer()} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS92AY);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, -2.0*iscale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C,   2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
  
  
  
      /* Equation (92)Y last term */
      std::vector<TwoBodyContraction<MatsT>> contractGLS92BY =
        { {contract1PDMLS.X().pointer(), Scr1, true, COULOMB, relERI[18].pointer()},
          {contract1PDMLS.Y().pointer(), Scr2, true, COULOMB, relERI[20].pointer()},
          {contract1PDMLS.Z().pointer(), Scr3, true, COULOMB, relERI[13].pointer()} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS92BY);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, -2.0*scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
  
  
      /* Equation (92)Z first two terms */
      std::vector<TwoBodyContraction<MatsT>> contractGLS92AZ =
        { {contract1PDMLS.S().pointer(), Scr1, true, COULOMB, relERI[7].pointer()},
          {contract1PDMLS.Z().pointer(), Scr2, true, COULOMB, relERI[4].pointer()} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS92AZ);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, -2.0*iscale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C,   2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
  
  
      /* Equation (92)Z last term */
      std::vector<TwoBodyContraction<MatsT>> contractGLS92BZ =
        { {contract1PDMLS.X().pointer(), Scr1, true, COULOMB, relERI[19].pointer()},
          {contract1PDMLS.Y().pointer(), Scr2, true, COULOMB, relERI[21].pointer()},
          {contract1PDMLS.Z().pointer(), Scr3, true, COULOMB, relERI[22].pointer()} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS92BZ);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, -2.0*scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
  
  
  
#ifdef _PRINT_MATRICES
      std::cout<<"After Gaunt 91-92"<<std::endl;
      prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
#endif
  
      if(RESET) {
        ss.coulombMatrix->clear();
        ss.exchangeMatrix->clear();
      }
  
  
      /* Equation (136) */
      std::vector<TwoBodyContraction<MatsT>> contractGLS136 =
        { {contract1PDMSL.S().pointer(), Scr1, true, COULOMB, relERI[4].pointer(), TRANS_KL},
          {contract1PDMSL.X().pointer(), Scr2, true, COULOMB, relERI[5].pointer(), TRANS_KL},
          {contract1PDMSL.Y().pointer(), Scr3, true, COULOMB, relERI[6].pointer(), TRANS_KL},
          {contract1PDMSL.Z().pointer(), Scr4, true, COULOMB, relERI[7].pointer(), TRANS_KL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS136);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C,  -2.0*scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*iscale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*iscale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*iscale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
  
  
      /* Equation (137)X first two terms */
      std::vector<TwoBodyContraction<MatsT>> contractGLS137AX =
        { {contract1PDMSL.S().pointer(), Scr1, true, COULOMB, relERI[5].pointer(), TRANS_KL},
          {contract1PDMSL.X().pointer(), Scr2, true, COULOMB, relERI[4].pointer(), TRANS_KL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS137AX);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, 2.0*iscale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C,  2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
  
  
      /* Equation (137)X last term */
      std::vector<TwoBodyContraction<MatsT>> contractGLS137BX =
        { {contract1PDMSL.X().pointer(), Scr1, true, COULOMB, relERI[17].pointer(), TRANS_KL},
          {contract1PDMSL.Y().pointer(), Scr2, true, COULOMB, relERI[9].pointer(),  TRANS_KL},
          {contract1PDMSL.Z().pointer(), Scr3, true, COULOMB, relERI[11].pointer(), TRANS_KL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS137BX);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, -2.0*scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
  
      /* Equation (137)Y first two terms */
      std::vector<TwoBodyContraction<MatsT>> contractGLS137AY =
        { {contract1PDMSL.S().pointer(), Scr1, true, COULOMB, relERI[6].pointer(), TRANS_KL},
          {contract1PDMSL.Y().pointer(), Scr2, true, COULOMB, relERI[4].pointer(), TRANS_KL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS137AY);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, 2.0*iscale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C,  2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
  
  
      /* Equation (137)Y last term */
      std::vector<TwoBodyContraction<MatsT>> contractGLS137BY =
        { {contract1PDMSL.X().pointer(), Scr1, true, COULOMB, relERI[18].pointer(), TRANS_KL},
          {contract1PDMSL.Y().pointer(), Scr2, true, COULOMB, relERI[20].pointer(), TRANS_KL},
          {contract1PDMSL.Z().pointer(), Scr3, true, COULOMB, relERI[13].pointer(), TRANS_KL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS137BY);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, -2.0*scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
  
      /* Equation (137)Z first two terms */
      std::vector<TwoBodyContraction<MatsT>> contractGLS137AZ =
        { {contract1PDMSL.S().pointer(), Scr1, true, COULOMB, relERI[7].pointer(), TRANS_KL},
          {contract1PDMSL.Z().pointer(), Scr2, true, COULOMB, relERI[4].pointer(), TRANS_KL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS137AZ);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, 2.0*iscale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C,  2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
  
  
      /* Equation (137)Z last term */
      std::vector<TwoBodyContraction<MatsT>> contractGLS137BZ =
        { {contract1PDMSL.X().pointer(), Scr1, true, COULOMB, relERI[19].pointer(), TRANS_KL},
          {contract1PDMSL.Y().pointer(), Scr2, true, COULOMB, relERI[21].pointer(), TRANS_KL},
          {contract1PDMSL.Z().pointer(), Scr3, true, COULOMB, relERI[22].pointer(), TRANS_KL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS137BZ);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, -2.0*scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -2.0*scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
  
  
  
  
#ifdef _PRINT_MATRICES
      std::cout<<"After Gaunt 136-137"<<std::endl;
      prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
#endif
      if(RESET) {
        ss.coulombMatrix->clear();
        ss.exchangeMatrix->clear();
      }
  
  
  
#endif  // Gaunt LLSS COULOMB
  
#if 1 // Gaunt LLSS EXCHANGE
      /* Equation (159) */
      std::vector<TwoBodyContraction<MatsT>> contractGLS159 =
        { {contract1PDMSL.S().pointer(), Scr1, true, EXCHANGE, relERI[4].pointer(), TRANS_KL},
          {contract1PDMSL.X().pointer(), Scr2, true, EXCHANGE, relERI[5].pointer(), TRANS_KL},
          {contract1PDMSL.Y().pointer(), Scr3, true, EXCHANGE, relERI[6].pointer(), TRANS_KL},
          {contract1PDMSL.Z().pointer(), Scr4, true, EXCHANGE, relERI[7].pointer(), TRANS_KL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS159);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, -scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, iscale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, iscale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, iscale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+LS, NB2C);
  
      /* Equation (160) first four terms */
      std::vector<TwoBodyContraction<MatsT>> contractGLS160A =
        { {contract1PDMSL.Z().pointer(), Scr1, true, EXCHANGE, relERI[4].pointer(), TRANS_KL},
          {contract1PDMSL.Y().pointer(), Scr2, true, EXCHANGE, relERI[5].pointer(), TRANS_KL},
          {contract1PDMSL.X().pointer(), Scr3, true, EXCHANGE, relERI[6].pointer(), TRANS_KL},
          {contract1PDMSL.S().pointer(), Scr4, true, EXCHANGE, relERI[7].pointer(), TRANS_KL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS160A);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, 2.0*scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, 2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C,-2.0*scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C,   -iscale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
  
  
      /* Equation (160) last three terms */
      std::vector<TwoBodyContraction<MatsT>> contractGLS160B =
        { {contract1PDMSL.Z().pointer(), Scr1, true, EXCHANGE, relERI[14].pointer(), TRANS_KL},
          {contract1PDMSL.X().pointer(), Scr2, true, EXCHANGE, relERI[10].pointer(), TRANS_KL},
          {contract1PDMSL.Y().pointer(), Scr3, true, EXCHANGE, relERI[12].pointer(), TRANS_KL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS160B);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+LS, NB2C);
  
  
  
      /* Equation (161) first four terms */
      std::vector<TwoBodyContraction<MatsT>> contractGLS161A =
        { {contract1PDMSL.X().pointer(), Scr1, true, EXCHANGE, relERI[4].pointer(), TRANS_KL},
          {contract1PDMSL.Y().pointer(), Scr2, true, EXCHANGE, relERI[7].pointer(), TRANS_KL},
          {contract1PDMSL.Z().pointer(), Scr3, true, EXCHANGE, relERI[6].pointer(), TRANS_KL},
          {contract1PDMSL.S().pointer(), Scr4, true, EXCHANGE, relERI[5].pointer(), TRANS_KL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS161A);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, 2.0*scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C,-2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, 2.0*scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C,   -iscale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
  
  
      /* Equation (161) last three terms */
      std::vector<TwoBodyContraction<MatsT>> contractGLS161B =
        { {contract1PDMSL.X().pointer(), Scr1, true, EXCHANGE, relERI[15].pointer(), TRANS_KL},
          {contract1PDMSL.Y().pointer(), Scr2, true, EXCHANGE, relERI[8].pointer(),  TRANS_KL},
          {contract1PDMSL.Z().pointer(), Scr3, true, EXCHANGE, relERI[10].pointer(), TRANS_KL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS161B);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+LS, NB2C);
  
  
      /* Equation (162) first four terms */
      std::vector<TwoBodyContraction<MatsT>> contractGLS162A =
        { {contract1PDMSL.Y().pointer(), Scr1, true, EXCHANGE, relERI[4].pointer(), TRANS_KL},
          {contract1PDMSL.X().pointer(), Scr2, true, EXCHANGE, relERI[7].pointer(), TRANS_KL},
          {contract1PDMSL.Z().pointer(), Scr3, true, EXCHANGE, relERI[5].pointer(), TRANS_KL},
          {contract1PDMSL.S().pointer(), Scr4, true, EXCHANGE, relERI[6].pointer(), TRANS_KL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS162A);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, 2.0*scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, 2.0*scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C,-2.0*scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C,   -iscale, Scr4, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
  
  
      /* Equation (162) last three terms */
      std::vector<TwoBodyContraction<MatsT>> contractGLS162B =
        { {contract1PDMSL.Y().pointer(), Scr1, true, EXCHANGE, relERI[16].pointer(), TRANS_KL},
          {contract1PDMSL.X().pointer(), Scr2, true, EXCHANGE, relERI[8].pointer(),  TRANS_KL},
          {contract1PDMSL.Z().pointer(), Scr3, true, EXCHANGE, relERI[12].pointer(), TRANS_KL} };
  
      // Call the contraction engine to do the assembly
      ss.ERI->twoBodyContract(ss.comm, contractGLS162B);
  
      // Assemble 4C exchangeMatrix 
      MatAdd('N','N', NB1C, NB1C, scale, Scr1, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, scale, Scr2, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, scale, Scr3, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+LS, NB2C);
  
  
#ifdef _PRINT_MATRICES
      std::cout<<"After Gaunt 159-162"<<std::endl;
      prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
#endif //_PRINT_MATRICES
  
#endif // Gaunt LLSS EXCHANGE
  
#ifdef _PRINT_MATRICES
  
      std::cout<<"After Gaunt LLSS|SSLL"<<std::endl;
      prettyPrintSmart(std::cout, "COULOMB",    ss.coulombMatrix->pointer(),      NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
  
#endif //_PRINT_MATRICES
  
  
      /*------------------------------------*/
      /*   End of Gaunt (LL|SS) Contraction */
      /*------------------------------------*/
  
  
      auto durERIDG = tock(topERIDG);
      //std::cout << "Gaunt duration   = " << durERIDG << std::endl;
  
    } //GAUNT





    /*******************************/
    /* Final Assembly of 4C Matrix */
    /*******************************/
    ROOT_ONLY(ss.comm);

    // Copy LS to SL part of the exchangeMatrix[MS]
    SetMat('C', NB1C, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+SL, NB2C);
    // Copy LS to SL part of the exchangeMatrix[MX]
    SetMat('C', NB1C, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+SL, NB2C);
    // Copy LS to SL part of the exchangeMatrix[MY]
    SetMat('C', NB1C, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+SL, NB2C);
    // Copy LS to SL part of the exchangeMatrix[MZ]
    SetMat('C', NB1C, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+SL, NB2C);

    // Form GD: G[D] = 2.0*J[D] - K[D]
    if( std::abs(xHFX) > 1e-12 ) {
      *ss.twoeH = -xHFX * *ss.exchangeMatrix;
    } else {
      ss.twoeH->clear();
    }
    // G[D] += 2*J[D]
    *ss.twoeH += 2.0 * *ss.coulombMatrix;

    mem.free(Scr1);
    mem.free(Scr2);
    mem.free(Scr3);
    mem.free(Scr4);


#ifdef _PRINT_MATRICES

    prettyPrintSmart(std::cout,"twoeH MS",ss.twoeH->S().pointer(),NB2C,NB2C,NB2C);
    prettyPrintSmart(std::cout,"twoeH MX",ss.twoeH->X().pointer(),NB2C,NB2C,NB2C);
    prettyPrintSmart(std::cout,"twoeH MY",ss.twoeH->Y().pointer(),NB2C,NB2C,NB2C);
    prettyPrintSmart(std::cout,"twoeH MZ",ss.twoeH->Z().pointer(),NB2C,NB2C,NB2C);


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


  }; // FourCompFock<MatsT, IntsT>::formGDInCore








  /**   
   *  \brief Forms the 4C Fock matrix using 3 Index ERI
   */
  template <typename MatsT, typename IntsT>
  void FourCompFock<MatsT,IntsT>::formGD3Index(SingleSlater<MatsT,IntsT> &ss,
    EMPerturbation &pert, bool increment, double xHFX) {

    CQMemManager &mem = ss.memManager;
    GTODirectRelERIContraction<MatsT,IntsT> &relERICon =
        *std::dynamic_pointer_cast<GTODirectRelERIContraction<MatsT,IntsT>>(ss.ERI);

    // Decide list of onePDMs to use
    PauliSpinorSquareMatrices<MatsT> &contract1PDM
        = increment ? *ss.deltaOnePDM : *ss.onePDM;

    size_t NB1C  = ss.basisSet().nBasis;
    size_t NB2C  = 2 * ss.basisSet().nBasis;
    size_t NB4C  = 4 * ss.basisSet().nBasis;
    size_t NB1C2 = NB1C*NB1C;
    size_t NB1C4 = NB1C*NB1C*NB1C*NB1C;
    size_t NB1C3 = NB1C*NB1C*NB1C;
    size_t NB2C2 = NB2C*NB2C;
    size_t NB4C2 = NB4C*NB4C;

    size_t SS = NB2C*NB1C+NB1C;
    size_t LS = NB2C*NB1C;
    size_t SL = NB1C;

    auto MS = SCALAR;

    size_t mpiRank   = MPIRank(ss.comm);
    bool   isNotRoot = mpiRank != 0;

    PauliSpinorSquareMatrices<MatsT> exchangeMatrixLL(mem, NB1C);
    PauliSpinorSquareMatrices<MatsT> contract1PDMLL(mem, NB1C);
    PauliSpinorSquareMatrices<MatsT> contract1PDMSS(mem, NB1C);
    PauliSpinorSquareMatrices<MatsT> contract1PDMLS(mem, NB1C);
    PauliSpinorSquareMatrices<MatsT> contract1PDMSL(mem, NB1C);

    MatsT* Scr1 = mem.malloc<MatsT>(NB1C2);
    MatsT* Scr2 = mem.malloc<MatsT>(NB1C2);
    MatsT* Scr3 = mem.malloc<MatsT>(NB1C2);
    MatsT* Scr4 = mem.malloc<MatsT>(NB1C2);
    memset(Scr1,0.,NB1C2*sizeof(MatsT));
    memset(Scr2,0.,NB1C2*sizeof(MatsT));
    memset(Scr3,0.,NB1C2*sizeof(MatsT));
    memset(Scr4,0.,NB1C2*sizeof(MatsT));


    // Compute 1/(2mc)^2
    //dcomplex scale = 1.;
    //dcomplex iscale = dcomplex(0.0, 1.0);
    dcomplex scale = 1./(4*SpeedOfLight*SpeedOfLight);
    dcomplex iscale = dcomplex(0.0, 1./(4*SpeedOfLight*SpeedOfLight));

    for(size_t i = 0; i < contract1PDM.nComponent(); i++) {
      PAULI_SPINOR_COMPS c = static_cast<PAULI_SPINOR_COMPS>(i);
      SetMat('N', NB1C, NB1C, MatsT(1.), contract1PDM[c].pointer(),    NB2C,
             contract1PDMLL[c].pointer(), NB1C);
      SetMat('N', NB1C, NB1C, MatsT(1.), contract1PDM[c].pointer()+SS, NB2C,
             contract1PDMSS[c].pointer(), NB1C);
      SetMat('N', NB1C, NB1C, MatsT(1.), contract1PDM[c].pointer()+LS, NB2C,
             contract1PDMLS[c].pointer(), NB1C);
      SetMat('N', NB1C, NB1C, MatsT(1.), contract1PDM[c].pointer()+SL, NB2C,
             contract1PDMSL[c].pointer(), NB1C);
    }

#ifdef _PRINT_MATRICES
    prettyPrintSmart(std::cout, "1PDM[MS]", contract1PDM.S().pointer(), NB2C, NB2C, NB2C);
    prettyPrintSmart(std::cout, "1PDM[MX]", contract1PDM.X().pointer(), NB2C, NB2C, NB2C);
    prettyPrintSmart(std::cout, "1PDM[MY]", contract1PDM.Y().pointer(), NB2C, NB2C, NB2C);
    prettyPrintSmart(std::cout, "1PDM[MZ]", contract1PDM.Z().pointer(), NB2C, NB2C, NB2C);
#endif

 
    if(not increment) {
      ss.coulombMatrix->clear();
      ss.exchangeMatrix->clear();
    };


    /**********************************************/
    /*                                            */
    /*              DIRECT COULOMB     	          */
    /*                                            */
    /**********************************************/


    if(this->hamiltonianOptions_.BareCoulomb) { // DIRECT_COULOMB

      auto topBareCoulomb = tick();

      /*+++++++++++++++++++++++++++++++++++++++++++++*/
      /* Start of Direct Coulomb (LL|LL) Contraction */
      /*+++++++++++++++++++++++++++++++++++++++++++++*/
  
      std::vector<TwoBodyContraction<MatsT>> contractLL =
        { {contract1PDMLL.S().pointer(), Scr1, true, COULOMB} };
  
      // Determine how many (if any) exchange terms to calculate
      if( std::abs(xHFX) > 1e-12 ) {
        exchangeMatrixLL.clear();
        for(size_t i = 0; i < ss.exchangeMatrix->nComponent(); i++) {
  
          PAULI_SPINOR_COMPS c = static_cast<PAULI_SPINOR_COMPS>(i);
          contractLL.push_back(
            {contract1PDMLL[c].pointer(), exchangeMatrixLL[c].pointer(), true, EXCHANGE}
          );
        }
      }
  
      // Zero out K[i]
      if(not increment) ss.exchangeMatrix->clear();
  
      // Call the contraction engine to do the assembly of direct Coulomb LLLL
      GTODirectERIContraction<MatsT,IntsT>(ss.ERI->ints()).twoBodyContract(ss.comm, true, contractLL, pert);
  
  
      /* Store LL block into 2C spin scattered matrices */
      // Assemble 4C coulombMatrix
      SetMat('N', NB1C, NB1C, MatsT(1.), Scr1, NB1C, ss.coulombMatrix->pointer(), NB2C);
  
      // Assemble 4C exchangeMatrix 
      for(auto i = 0; i < ss.exchangeMatrix->nComponent();i++){
        PAULI_SPINOR_COMPS c = static_cast<PAULI_SPINOR_COMPS>(i);
        SetMat('N', NB1C, NB1C, MatsT(1.), exchangeMatrixLL[c].pointer(), NB1C,
               (*ss.exchangeMatrix)[c].pointer(), NB2C);
      }

      /*---------------------------------------------*/
      /*   End of Direct Coulomb (LL|LL) Contraction */
      /*---------------------------------------------*/

      // Print out BareCoulomb duration 
      auto durBareCoulomb = tock(topBareCoulomb);
//      std::cout << "Non-relativistic Coulomb duration = " << durBareCoulomb << std::endl;

    } // DIRECT_COULOMB


    /* using 3-Index ERI */
    // Loop over first shell
    size_t n1, bf1, bf1_s;
    size_t nERI3 = 37;
    for(auto s1(0), bf1_s(0); s1< ss.basisSet().nShell; bf1_s+=n1, s1++) {

      n1 = ss.basisSet().shells[s1].size();

      relERICon.computeERI3Index(s1);

    // Loop over all basis in the first shell
    for(auto ibatch = 0 ; ibatch < n1; ibatch++){
      auto ERI4bf1 = relERICon.ERI4DCB+nERI3*NB1C3*ibatch;
      bf1 = bf1_s + ibatch;


      /**********************************************/
      /*                                            */
      /*              DIRAC-COULOMB    	            */
      /*                                            */
      /**********************************************/

      // ERI: (ab|cd)
      // ERIDCB0: ∇A∙∇B(ab|cd)
      // ERIDCB1: ∇Ax∇B(ab|cd)-X
      // ERIDCB2: ∇Ax∇B(ab|cd)-Y
      // ERIDCB3: ∇Ax∇B(ab|cd)-Z

      if(this->hamiltonianOptions_.DiracCoulomb) { // DIRAC_COULOMB
  
        /*++++++++++++++++++++++++++++++++++++++++++++*/
        /* Start of Dirac-Coulomb (LL|LL) Contraction */
        /*++++++++++++++++++++++++++++++++++++++++++++*/
    
        std::vector<TwoBodyContraction<MatsT>> contractDCLL =
          { {contract1PDMSS.S().pointer(), Scr1, true, COULOMB, ERI4bf1+4*NB1C3},
            {contract1PDMSS.X().pointer(), Scr2, true, COULOMB, ERI4bf1+5*NB1C3},
            {contract1PDMSS.Y().pointer(), Scr3, true, COULOMB, ERI4bf1+6*NB1C3},
            {contract1PDMSS.Z().pointer(), Scr4, true, COULOMB, ERI4bf1+7*NB1C3} };
    
        // Call the contraction engine to do the assembly of Dirac-Coulomb LLLL
        relERICon.twoBodyContract3Index(ss.comm, contractDCLL);
    
        // Add Dirac-Coulomb contributions  to the LLLL block
        for(auto i=0; i<NB1C; i++){
          ss.coulombMatrix->pointer()[bf1+i*NB2C] += scale*Scr1[i] + iscale*Scr2[i] + iscale*Scr3[i] + iscale*Scr4[i];
        }
    
#ifdef _PRINT_MATRICES
    
        std::cout<<"After LLLL Iteration #"<<bf1<<std::endl;
        prettyPrintSmart(std::cout, "COULOMB",    ss.coulombMatrix->pointer(),      NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
    
#endif //_PRINT_MATRICES
    
        /*------------------------------------------*/
        /* End of Dirac-Coulomb (LL|LL) Contraction */
        /*------------------------------------------*/
    
    
    
    
    
        /*++++++++++++++++++++++++++++++++++++++++++++*/
        /* Start of Dirac-Coulomb (SS|SS) Contraction */
        /*++++++++++++++++++++++++++++++++++++++++++++*/
    
        std::vector<TwoBodyContraction<MatsT>> contractSS =
          { {contract1PDMLL.S().pointer(), Scr1, true, COULOMB, ERI4bf1},
            {contract1PDMLL.S().pointer(), Scr2, true, COULOMB, ERI4bf1+NB1C3},
            {contract1PDMLL.S().pointer(), Scr3, true, COULOMB, ERI4bf1+2*NB1C3},
            {contract1PDMLL.S().pointer(), Scr4, true, COULOMB, ERI4bf1+3*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractSS);
    
        // Store SS block into 2C spin scattered matrices 
        // These scaling factors were modified to take into account the issue of storing the 
        // Coulomb portion in the exchange matrix, this will be fixed later
        for(auto i=0; i<NB1C; i++){
          ss.coulombMatrix->pointer()[SS+bf1+i*NB2C]      +=       scale*Scr1[i];
          ss.exchangeMatrix->X().pointer()[SS+bf1+i*NB2C] += -2.0*iscale*Scr2[i];
          ss.exchangeMatrix->Y().pointer()[SS+bf1+i*NB2C] += -2.0*iscale*Scr3[i];
          ss.exchangeMatrix->Z().pointer()[SS+bf1+i*NB2C] += -2.0*iscale*Scr4[i];
        }
    
#ifdef _PRINT_MATRICES
    
        std::cout<<"After SSSS Iteration #"<<bf1<<std::endl;
        prettyPrintSmart(std::cout, "COULOMB",    ss.coulombMatrix->pointer(),      NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
    
#endif //_PRINT_MATRICES
    
    
        /*--------------------------------------------*/
        /* End of Dirac-Coulomb (SS|SS) Contraction */
        /*--------------------------------------------*/
    
    
    
    
    
        /*++++++++++++++++++++++++++++++++++++++++++*/
        /* Start of Dirac-Coulomb (LL|SS) / (SS|LL) */
        /*++++++++++++++++++++++++++++++++++++++++++*/
    
        std::vector<TwoBodyContraction<MatsT>> contractLSScalar =
          { {contract1PDMLS.S().pointer(), Scr1, true, EXCHANGE, ERI4bf1+4*NB1C3},
            {contract1PDMLS.X().pointer(), Scr2, true, EXCHANGE, ERI4bf1+5*NB1C3},
            {contract1PDMLS.Y().pointer(), Scr3, true, EXCHANGE, ERI4bf1+6*NB1C3},
            {contract1PDMLS.Z().pointer(), Scr4, true, EXCHANGE, ERI4bf1+7*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractLSScalar);
    
        // Add to the LS part of the exchangeMatrix[MS]
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->S().pointer()[LS+bf1+i*NB2C] += scale*Scr1[i] +iscale*Scr2[i] +iscale*Scr3[i] +iscale*Scr4[i];
        }
    
    
    
        std::vector<TwoBodyContraction<MatsT>> contractLSMX =
          { {contract1PDMLS.X().pointer(), Scr1, true, EXCHANGE, ERI4bf1+4*NB1C3},
            {contract1PDMLS.S().pointer(), Scr2, true, EXCHANGE, ERI4bf1+5*NB1C3},
            {contract1PDMLS.Z().pointer(), Scr3, true, EXCHANGE, ERI4bf1+6*NB1C3},
            {contract1PDMLS.Y().pointer(), Scr4, true, EXCHANGE, ERI4bf1+7*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractLSMX);
    
        // Add to the LS part of the exchangeMatrix[MX]
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->X().pointer()[LS+bf1+i*NB2C] += scale*Scr1[i] +iscale*Scr2[i] +scale*Scr3[i] -scale*Scr4[i];
        }
    
    
    
    
        std::vector<TwoBodyContraction<MatsT>> contractLSMY =
          { {contract1PDMLS.Y().pointer(), Scr1, true, EXCHANGE, ERI4bf1+4*NB1C3},
            {contract1PDMLS.Z().pointer(), Scr2, true, EXCHANGE, ERI4bf1+5*NB1C3},
            {contract1PDMLS.S().pointer(), Scr3, true, EXCHANGE, ERI4bf1+6*NB1C3},
            {contract1PDMLS.X().pointer(), Scr4, true, EXCHANGE, ERI4bf1+7*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractLSMY);
    
        // Add to the LS part of the exchangeMatrix[MY]
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->Y().pointer()[LS+bf1+i*NB2C] += scale*Scr1[i] -scale*Scr2[i] +iscale*Scr3[i] +scale*Scr4[i];
        }
    
    
    
    
    
        std::vector<TwoBodyContraction<MatsT>> contractLSMZ =
          { {contract1PDMLS.Z().pointer(), Scr1, true, EXCHANGE, ERI4bf1+4*NB1C3},
            {contract1PDMLS.Y().pointer(), Scr2, true, EXCHANGE, ERI4bf1+5*NB1C3},
            {contract1PDMLS.X().pointer(), Scr3, true, EXCHANGE, ERI4bf1+6*NB1C3},
            {contract1PDMLS.S().pointer(), Scr4, true, EXCHANGE, ERI4bf1+7*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractLSMZ);
    
        // Add to the LS part of the exchangeMatrix[MZ]
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->Z().pointer()[LS+bf1+i*NB2C] += scale*Scr1[i] +scale*Scr2[i] -scale*Scr3[i] +iscale*Scr4[i];
        }
    
    
        /*------------------------------------------*/
        /*   End of Dirac-Coulomb (LL|SS) / (SS|LL) */
        /*------------------------------------------*/
    
    
#ifdef _PRINT_MATRICES
    
        std::cout<<"After LLSS Iteration #"<<bf1<<std::endl;
        prettyPrintSmart(std::cout, "COULOMB",    ss.coulombMatrix->pointer(),      NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
    
#endif //_PRINT_MATRICES
    
  
      } //_DIRAC_COULOMB
  



      /**********************************************/
      /*                                            */
      /*              GAUNT                         */
      /*                                            */
      /**********************************************/
    
    
      //ERI4:    ∇B∙∇C(mn|kl)
      //ERI5 :   ∇Bx∇C(mn|kl)-X
      //ERI6 :   ∇Bx∇C(mn|kl)-Y
      //ERI7 :   ∇Bx∇C(mn|kl)-Z
      //ERI8 :   ∇B_x∇C_y(mn|kl) + ∇B_y∇C_x(mn|kl)
      //ERI9 :   ∇B_y∇C_x(mn|kl)
      //ERI10:   ∇B_x∇C_z(mn|kl) + ∇B_z∇C_x(mn|kl)
      //ERI11:   ∇B_z∇C_x(mn|kl)
      //ERI12:   ∇B_y∇C_z(mn|kl) + ∇B_z∇C_y(mn|kl)
      //ERI13:   ∇B_z∇C_y(mn|kl)
      //ERI14: - ∇B_x∇C_x(mn|kl) - ∇B_y∇C_y(mn|kl) + ∇B_z∇C_z(mn|kl)
      //ERI15:   ∇B_x∇C_x(mn|kl) - ∇B_y∇C_y(mn|kl) - ∇B_z∇C_z(mn|kl)
      //ERI16: - ∇B_x∇C_x(mn|kl) + ∇B_y∇C_y(mn|kl) - ∇B_z∇C_z(mn|kl)
      //ERI17:   ∇B_x∇C_x(mn|kl)
      //ERI18:   ∇B_x∇C_y(mn|kl)
      //ERI19:   ∇B_x∇C_z(mn|kl)
      //ERI20:   ∇B_y∇C_y(mn|kl)
      //ERI21:   ∇B_y∇C_z(mn|kl)
      //ERI22:   ∇B_z∇C_z(mn|kl)
    
      if(this->hamiltonianOptions_.Gaunt) {//GAUNT
    
#if 1 //Gaunt LLLL
        /*++++++++++++++++++++++++++++++++++++*/
        /* Start of Gaunt (LL|LL) Contraction */
        /*++++++++++++++++++++++++++++++++++++*/
    
        /* Equation (113) */
        std::vector<TwoBodyContraction<MatsT>> contractGLL113 =
          { {contract1PDMSS.S().pointer(), Scr1, true, EXCHANGE, ERI4bf1+ 8*NB1C3},
            {contract1PDMSS.X().pointer(), Scr2, true, EXCHANGE, ERI4bf1+ 9*NB1C3},
            {contract1PDMSS.Y().pointer(), Scr3, true, EXCHANGE, ERI4bf1+10*NB1C3},
            {contract1PDMSS.Z().pointer(), Scr4, true, EXCHANGE, ERI4bf1+11*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLL113);
    
        // Add to the LL part of 4C exchangeMatrix in Pauli matrix form
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->S().pointer()[bf1+i*NB2C] += -3.0*scale*Scr1[i] +3.0*iscale*Scr2[i] +3.0*iscale*Scr3[i] +3.0*iscale*Scr4[i];
        }
    
    
        /* Equation (114) */
        std::vector<TwoBodyContraction<MatsT>> contractGLL114 =
          { {contract1PDMSS.Z().pointer(), Scr1, true, EXCHANGE, ERI4bf1+18*NB1C3},
            {contract1PDMSS.S().pointer(), Scr2, true, EXCHANGE, ERI4bf1+11*NB1C3},
            {contract1PDMSS.X().pointer(), Scr3, true, EXCHANGE, ERI4bf1+14*NB1C3},
            {contract1PDMSS.Y().pointer(), Scr4, true, EXCHANGE, ERI4bf1+16*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLL114);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->Z().pointer()[bf1+i*NB2C] += scale*Scr1[i] +iscale*Scr2[i] +scale*Scr3[i] +scale*Scr4[i];
        }
    
    
    
        /* Equation (115) */
        std::vector<TwoBodyContraction<MatsT>> contractGLL115 =
          { {contract1PDMSS.X().pointer(), Scr1, true, EXCHANGE, ERI4bf1+19*NB1C3},
            {contract1PDMSS.S().pointer(), Scr2, true, EXCHANGE, ERI4bf1+ 9*NB1C3},
            {contract1PDMSS.Y().pointer(), Scr3, true, EXCHANGE, ERI4bf1+12*NB1C3},
            {contract1PDMSS.Z().pointer(), Scr4, true, EXCHANGE, ERI4bf1+14*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLL115);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->X().pointer()[bf1+i*NB2C] += scale*Scr1[i] +iscale*Scr2[i] +scale*Scr3[i] +scale*Scr4[i];
        }
    
    
    
        /* Equation (116) */
        std::vector<TwoBodyContraction<MatsT>> contractGLL116 =
          { {contract1PDMSS.Y().pointer(), Scr1, true, EXCHANGE, ERI4bf1+20*NB1C3},
            {contract1PDMSS.S().pointer(), Scr2, true, EXCHANGE, ERI4bf1+10*NB1C3},
            {contract1PDMSS.X().pointer(), Scr3, true, EXCHANGE, ERI4bf1+12*NB1C3},
            {contract1PDMSS.Z().pointer(), Scr4, true, EXCHANGE, ERI4bf1+16*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLL116);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->Y().pointer()[bf1+i*NB2C] += scale*Scr1[i] +iscale*Scr2[i] +scale*Scr3[i] +scale*Scr4[i];
        }
    
    
    
#ifdef _PRINT_MATRICES
    
        std::cout<<"After Gaunt LLLL Iteration #"<<bf1<<std::endl;
        prettyPrintSmart(std::cout, "COULOMB",    ss.coulombMatrix->pointer(),      NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
    
#endif //_PRINT_MATRICES
    
#endif //Gaunt LLLL
    
        /*----------------------------------*/
        /* End of Gaunt (LL|LL) Contraction */
        /*----------------------------------*/
    
    
    
    
        /*++++++++++++++++++++++++++++++++++++*/
        /* Start of Gaunt (SS|SS) Contraction */
        /*++++++++++++++++++++++++++++++++++++*/
    
#if 1 //Gaunt SSSS
    
        /* Equation (129) */
        std::vector<TwoBodyContraction<MatsT>> contractGSS129 =
          { {contract1PDMLL.S().pointer(), Scr1, true, EXCHANGE, ERI4bf1+27*NB1C3},
            {contract1PDMLL.X().pointer(), Scr2, true, EXCHANGE, ERI4bf1+28*NB1C3},
            {contract1PDMLL.Y().pointer(), Scr3, true, EXCHANGE, ERI4bf1+29*NB1C3},
            {contract1PDMLL.Z().pointer(), Scr4, true, EXCHANGE, ERI4bf1+30*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGSS129);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->S().pointer()[SS+bf1+i*NB2C] += -3.0*scale*Scr1[i] - iscale*Scr2[i] - iscale*Scr3[i] - iscale*Scr4[i];
        }
        
    
        /* Equation (130) */
        std::vector<TwoBodyContraction<MatsT>> contractGSS130 =
          { {contract1PDMLL.Z().pointer(), Scr1, true, EXCHANGE, ERI4bf1+34*NB1C3},
            {contract1PDMLL.S().pointer(), Scr2, true, EXCHANGE, ERI4bf1+30*NB1C3},
            {contract1PDMLL.X().pointer(), Scr3, true, EXCHANGE, ERI4bf1+33*NB1C3},
            {contract1PDMLL.Y().pointer(), Scr4, true, EXCHANGE, ERI4bf1+32*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGSS130);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->Z().pointer()[SS+bf1+i*NB2C] += scale*Scr1[i] - 3.0*iscale*Scr2[i] + scale*Scr3[i] + scale*Scr4[i];
        }
    
    
        /* Equation (131) */
        std::vector<TwoBodyContraction<MatsT>> contractGSS131 =
          { {contract1PDMLL.X().pointer(), Scr1, true, EXCHANGE, ERI4bf1+35*NB1C3},
            {contract1PDMLL.S().pointer(), Scr2, true, EXCHANGE, ERI4bf1+28*NB1C3},
            {contract1PDMLL.Z().pointer(), Scr3, true, EXCHANGE, ERI4bf1+33*NB1C3},
            {contract1PDMLL.Y().pointer(), Scr4, true, EXCHANGE, ERI4bf1+31*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGSS131);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->X().pointer()[SS+bf1+i*NB2C] += scale*Scr1[i] - 3.0*iscale*Scr2[i] + scale*Scr3[i] + scale*Scr4[i];
        }
    
    
        /* Equation (132) */
        std::vector<TwoBodyContraction<MatsT>> contractGSS132 =
          { {contract1PDMLL.Y().pointer(), Scr1, true, EXCHANGE, ERI4bf1+36*NB1C3},
            {contract1PDMLL.S().pointer(), Scr2, true, EXCHANGE, ERI4bf1+29*NB1C3},
            {contract1PDMLL.X().pointer(), Scr3, true, EXCHANGE, ERI4bf1+31*NB1C3},
            {contract1PDMLL.Z().pointer(), Scr4, true, EXCHANGE, ERI4bf1+32*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGSS132);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->Y().pointer()[SS+bf1+i*NB2C] += scale*Scr1[i] - 3.0*iscale*Scr2[i] + scale*Scr3[i] + scale*Scr4[i];
        }
    
    
#ifdef _PRINT_MATRICES
    
        std::cout<<"After Gaunt SSSS Iteration #"<<bf1<<std::endl;
        prettyPrintSmart(std::cout, "COULOMB",    ss.coulombMatrix->pointer(),      NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
    
#endif //_PRINT_MATRICES
    
    
#endif // Gaunt SSSS
        /*------------------------------------*/
        /*   End of Gaunt (SS|SS) Contraction */
        /*------------------------------------*/
    
    
    
    
    
    
        /*++++++++++++++++++++++++++++++++++++*/
        /* Start of Gaunt (LL|SS) Contraction */
        /*++++++++++++++++++++++++++++++++++++*/
    
    
#if 1 // Gaunt LLSS COULOMB
    
        /* Equation (91) */
        std::vector<TwoBodyContraction<MatsT>> contractGLS91 =
          { {contract1PDMLS.S().pointer(), Scr1, true, COULOMB, ERI4bf1+ 8*NB1C3},
            {contract1PDMLS.X().pointer(), Scr2, true, COULOMB, ERI4bf1+ 9*NB1C3},
            {contract1PDMLS.Y().pointer(), Scr3, true, COULOMB, ERI4bf1+10*NB1C3},
            {contract1PDMLS.Z().pointer(), Scr4, true, COULOMB, ERI4bf1+11*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS91);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->S().pointer()[LS+bf1+i*NB2C] += 2.0*scale*Scr1[i] -2.0*iscale*Scr2[i] -2.0*iscale*Scr3[i] -2.0*iscale*Scr4[i];
        }
    
    
        /* Equation (92)X first two terms */
        std::vector<TwoBodyContraction<MatsT>> contractGLS92AX =
          { {contract1PDMLS.S().pointer(), Scr1, true, COULOMB, ERI4bf1+9*NB1C3},
            {contract1PDMLS.X().pointer(), Scr2, true, COULOMB, ERI4bf1+8*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS92AX);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->X().pointer()[LS+bf1+i*NB2C] += -2.0*iscale*Scr1[i] +2.0*scale*Scr2[i];
        }
    
    
        /* Equation (92)X last term */
        std::vector<TwoBodyContraction<MatsT>> contractGLS92BX =
          { {contract1PDMLS.X().pointer(), Scr1, true, COULOMB, ERI4bf1+21*NB1C3},
            {contract1PDMLS.Y().pointer(), Scr2, true, COULOMB, ERI4bf1+13*NB1C3},
            {contract1PDMLS.Z().pointer(), Scr3, true, COULOMB, ERI4bf1+15*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS92BX);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->X().pointer()[LS+bf1+i*NB2C] += -2.0*scale*Scr1[i] -2.0*scale*Scr2[i] -2.0*scale*Scr3[i];
        }
    
    
        /* Equation (92)Y first two terms */
        std::vector<TwoBodyContraction<MatsT>> contractGLS92AY =
          { {contract1PDMLS.S().pointer(), Scr1, true, COULOMB, ERI4bf1+10*NB1C3},
            {contract1PDMLS.Y().pointer(), Scr2, true, COULOMB, ERI4bf1+ 8*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS92AY);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->Y().pointer()[LS+bf1+i*NB2C] += -2.0*iscale*Scr1[i] +2.0*scale*Scr2[i];
        }
    
    
    
        /* Equation (92)Y last term */
        std::vector<TwoBodyContraction<MatsT>> contractGLS92BY =
          { {contract1PDMLS.X().pointer(), Scr1, true, COULOMB, ERI4bf1+22*NB1C3},
            {contract1PDMLS.Y().pointer(), Scr2, true, COULOMB, ERI4bf1+24*NB1C3},
            {contract1PDMLS.Z().pointer(), Scr3, true, COULOMB, ERI4bf1+17*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS92BY);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->Y().pointer()[LS+bf1+i*NB2C] += -2.0*scale*Scr1[i] -2.0*scale*Scr2[i] -2.0*scale*Scr3[i];
        }
    
    
        /* Equation (92)Z first two terms */
        std::vector<TwoBodyContraction<MatsT>> contractGLS92AZ =
          { {contract1PDMLS.S().pointer(), Scr1, true, COULOMB, ERI4bf1+11*NB1C3},
            {contract1PDMLS.Z().pointer(), Scr2, true, COULOMB, ERI4bf1+ 8*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS92AZ);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->Z().pointer()[LS+bf1+i*NB2C] += -2.0*iscale*Scr1[i] +2.0*scale*Scr2[i];
        }
    
    
        /* Equation (92)Z last term */
        std::vector<TwoBodyContraction<MatsT>> contractGLS92BZ =
          { {contract1PDMLS.X().pointer(), Scr1, true, COULOMB, ERI4bf1+23*NB1C3},
            {contract1PDMLS.Y().pointer(), Scr2, true, COULOMB, ERI4bf1+25*NB1C3},
            {contract1PDMLS.Z().pointer(), Scr3, true, COULOMB, ERI4bf1+26*NB1C3} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS92BZ);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->Z().pointer()[LS+bf1+i*NB2C] += -2.0*scale*Scr1[i] -2.0*scale*Scr2[i] -2.0*scale*Scr3[i];
        }
    
    
    
#ifdef _PRINT_MATRICES
        std::cout<<"After Gaunt 91-92 Iteration #"<<bf1<<std::endl;
        prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
#endif
    
    
        /* Equation (136) */
        std::vector<TwoBodyContraction<MatsT>> contractGLS136 =
          { {contract1PDMSL.S().pointer(), Scr1, true, COULOMB, ERI4bf1+ 8*NB1C3, TRANS_KL},
            {contract1PDMSL.X().pointer(), Scr2, true, COULOMB, ERI4bf1+ 9*NB1C3, TRANS_KL},
            {contract1PDMSL.Y().pointer(), Scr3, true, COULOMB, ERI4bf1+10*NB1C3, TRANS_KL},
            {contract1PDMSL.Z().pointer(), Scr4, true, COULOMB, ERI4bf1+11*NB1C3, TRANS_KL} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS136);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->S().pointer()[LS+bf1+i*NB2C] += -2.0*scale*Scr1[i] -2.0*iscale*Scr2[i] -2.0*iscale*Scr3[i] -2.0*iscale*Scr4[i];
        }
    
    
        /* Equation (137)X first two terms */
        std::vector<TwoBodyContraction<MatsT>> contractGLS137AX =
          { {contract1PDMSL.S().pointer(), Scr1, true, COULOMB, ERI4bf1+9*NB1C3, TRANS_KL},
            {contract1PDMSL.X().pointer(), Scr2, true, COULOMB, ERI4bf1+8*NB1C3, TRANS_KL} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS137AX);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->X().pointer()[LS+bf1+i*NB2C] += 2.0*iscale*Scr1[i] +2.0*scale*Scr2[i];
        }
    
    
        /* Equation (137)X last term */
        std::vector<TwoBodyContraction<MatsT>> contractGLS137BX =
          { {contract1PDMSL.X().pointer(), Scr1, true, COULOMB, ERI4bf1+21*NB1C3, TRANS_KL},
            {contract1PDMSL.Y().pointer(), Scr2, true, COULOMB, ERI4bf1+13*NB1C3, TRANS_KL},
            {contract1PDMSL.Z().pointer(), Scr3, true, COULOMB, ERI4bf1+15*NB1C3, TRANS_KL} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS137BX);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->X().pointer()[LS+bf1+i*NB2C] += -2.0*scale*Scr1[i] -2.0*scale*Scr2[i] -2.0*scale*Scr3[i];
        }
    
        /* Equation (137)Y first two terms */
        std::vector<TwoBodyContraction<MatsT>> contractGLS137AY =
          { {contract1PDMSL.S().pointer(), Scr1, true, COULOMB, ERI4bf1+10*NB1C3, TRANS_KL},
            {contract1PDMSL.Y().pointer(), Scr2, true, COULOMB, ERI4bf1+ 8*NB1C3, TRANS_KL} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS137AY);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->Y().pointer()[LS+bf1+i*NB2C] += 2.0*iscale*Scr1[i] +2.0*scale*Scr2[i];
        }
    
    
        /* Equation (137)Y last term */
        std::vector<TwoBodyContraction<MatsT>> contractGLS137BY =
          { {contract1PDMSL.X().pointer(), Scr1, true, COULOMB, ERI4bf1+22*NB1C3, TRANS_KL},
            {contract1PDMSL.Y().pointer(), Scr2, true, COULOMB, ERI4bf1+24*NB1C3, TRANS_KL},
            {contract1PDMSL.Z().pointer(), Scr3, true, COULOMB, ERI4bf1+17*NB1C3, TRANS_KL} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS137BY);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->Y().pointer()[LS+bf1+i*NB2C] += -2.0*scale*Scr1[i] -2.0*scale*Scr2[i] -2.0*scale*Scr3[i];
        }
    
        /* Equation (137)Z first two terms */
        std::vector<TwoBodyContraction<MatsT>> contractGLS137AZ =
          { {contract1PDMSL.S().pointer(), Scr1, true, COULOMB, ERI4bf1+11*NB1C3, TRANS_KL},
            {contract1PDMSL.Z().pointer(), Scr2, true, COULOMB, ERI4bf1+ 8*NB1C3, TRANS_KL} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS137AZ);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->Z().pointer()[LS+bf1+i*NB2C] += 2.0*iscale*Scr1[i] +2.0*scale*Scr2[i];
        }
    
    
        /* Equation (137)Z last term */
        std::vector<TwoBodyContraction<MatsT>> contractGLS137BZ =
          { {contract1PDMSL.X().pointer(), Scr1, true, COULOMB, ERI4bf1+23*NB1C3, TRANS_KL},
            {contract1PDMSL.Y().pointer(), Scr2, true, COULOMB, ERI4bf1+25*NB1C3, TRANS_KL},
            {contract1PDMSL.Z().pointer(), Scr3, true, COULOMB, ERI4bf1+26*NB1C3, TRANS_KL} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS137BZ);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->Z().pointer()[LS+bf1+i*NB2C] += -2.0*scale*Scr1[i] -2.0*scale*Scr2[i] -2.0*scale*Scr3[i];
        }
    
    
    
    
#ifdef _PRINT_MATRICES
        std::cout<<"After Gaunt 136-137 Iteration #"<<bf1<<std::endl;
        prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
#endif
    
    
#endif  // Gaunt LLSS COULOMB
    
    
    
    
#if 1 // Gaunt LLSS EXCHANGE
        /* Equation (159) */
        std::vector<TwoBodyContraction<MatsT>> contractGLS159 =
          { {contract1PDMSL.S().pointer(), Scr1, true, EXCHANGE, ERI4bf1+ 8*NB1C3, TRANS_KL},
            {contract1PDMSL.X().pointer(), Scr2, true, EXCHANGE, ERI4bf1+ 9*NB1C3, TRANS_KL},
            {contract1PDMSL.Y().pointer(), Scr3, true, EXCHANGE, ERI4bf1+10*NB1C3, TRANS_KL},
            {contract1PDMSL.Z().pointer(), Scr4, true, EXCHANGE, ERI4bf1+11*NB1C3, TRANS_KL} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS159);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->S().pointer()[LS+bf1+i*NB2C] += -scale*Scr1[i] +iscale*Scr2[i] +iscale*Scr3[i] +iscale*Scr4[i];
        }
    
        /* Equation (160) first four terms */
        std::vector<TwoBodyContraction<MatsT>> contractGLS160A =
          { {contract1PDMSL.Z().pointer(), Scr1, true, EXCHANGE, ERI4bf1+ 8*NB1C3, TRANS_KL},
            {contract1PDMSL.Y().pointer(), Scr2, true, EXCHANGE, ERI4bf1+ 9*NB1C3, TRANS_KL},
            {contract1PDMSL.X().pointer(), Scr3, true, EXCHANGE, ERI4bf1+10*NB1C3, TRANS_KL},
            {contract1PDMSL.S().pointer(), Scr4, true, EXCHANGE, ERI4bf1+11*NB1C3, TRANS_KL} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS160A);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->Z().pointer()[LS+bf1+i*NB2C] += 2.0*scale*Scr1[i] +2.0*scale*Scr2[i] -2.0*scale*Scr3[i] -iscale*Scr4[i];
        }
    
    
        /* Equation (160) last three terms */
        std::vector<TwoBodyContraction<MatsT>> contractGLS160B =
          { {contract1PDMSL.Z().pointer(), Scr1, true, EXCHANGE, ERI4bf1+18*NB1C3, TRANS_KL},
            {contract1PDMSL.X().pointer(), Scr2, true, EXCHANGE, ERI4bf1+14*NB1C3, TRANS_KL},
            {contract1PDMSL.Y().pointer(), Scr3, true, EXCHANGE, ERI4bf1+16*NB1C3, TRANS_KL} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS160B);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->Z().pointer()[LS+bf1+i*NB2C] += scale*Scr1[i] +scale*Scr2[i] +scale*Scr3[i];
        }
    
    
    
        /* Equation (161) first four terms */
        std::vector<TwoBodyContraction<MatsT>> contractGLS161A =
          { {contract1PDMSL.X().pointer(), Scr1, true, EXCHANGE, ERI4bf1+ 8*NB1C3, TRANS_KL},
            {contract1PDMSL.Y().pointer(), Scr2, true, EXCHANGE, ERI4bf1+11*NB1C3, TRANS_KL},
            {contract1PDMSL.Z().pointer(), Scr3, true, EXCHANGE, ERI4bf1+10*NB1C3, TRANS_KL},
            {contract1PDMSL.S().pointer(), Scr4, true, EXCHANGE, ERI4bf1+ 9*NB1C3, TRANS_KL} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS161A);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->X().pointer()[LS+bf1+i*NB2C] += 2.0*scale*Scr1[i] -2.0*scale*Scr2[i] +2.0*scale*Scr3[i] -iscale*Scr4[i];
        }
    
    
        /* Equation (161) last three terms */
        std::vector<TwoBodyContraction<MatsT>> contractGLS161B =
          { {contract1PDMSL.X().pointer(), Scr1, true, EXCHANGE, ERI4bf1+19*NB1C3, TRANS_KL},
            {contract1PDMSL.Y().pointer(), Scr2, true, EXCHANGE, ERI4bf1+12*NB1C3, TRANS_KL},
            {contract1PDMSL.Z().pointer(), Scr3, true, EXCHANGE, ERI4bf1+14*NB1C3, TRANS_KL} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS161B);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->X().pointer()[LS+bf1+i*NB2C] += scale*Scr1[i] +scale*Scr2[i] +scale*Scr3[i];
        }
    
    
        /* Equation (162) first four terms */
        std::vector<TwoBodyContraction<MatsT>> contractGLS162A =
          { {contract1PDMSL.Y().pointer(), Scr1, true, EXCHANGE, ERI4bf1+ 8*NB1C3, TRANS_KL},
            {contract1PDMSL.X().pointer(), Scr2, true, EXCHANGE, ERI4bf1+11*NB1C3, TRANS_KL},
            {contract1PDMSL.Z().pointer(), Scr3, true, EXCHANGE, ERI4bf1+ 9*NB1C3, TRANS_KL},
            {contract1PDMSL.S().pointer(), Scr4, true, EXCHANGE, ERI4bf1+10*NB1C3, TRANS_KL} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS162A);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->Y().pointer()[LS+bf1+i*NB2C] += 2.0*scale*Scr1[i] +2.0*scale*Scr2[i] -2.0*scale*Scr3[i] -iscale*Scr4[i];
        }
    
    
        /* Equation (162) last three terms */
        std::vector<TwoBodyContraction<MatsT>> contractGLS162B =
          { {contract1PDMSL.Y().pointer(), Scr1, true, EXCHANGE, ERI4bf1+20*NB1C3, TRANS_KL},
            {contract1PDMSL.X().pointer(), Scr2, true, EXCHANGE, ERI4bf1+12*NB1C3, TRANS_KL},
            {contract1PDMSL.Z().pointer(), Scr3, true, EXCHANGE, ERI4bf1+16*NB1C3, TRANS_KL} };
    
        // Call the contraction engine to do the assembly
        relERICon.twoBodyContract3Index(ss.comm, contractGLS162B);
    
        // Assemble 4C exchangeMatrix 
        for(auto i=0; i<NB1C; i++){
          ss.exchangeMatrix->Y().pointer()[LS+bf1+i*NB2C] += scale*Scr1[i] +scale*Scr2[i] +scale*Scr3[i];
        }
    
    
#ifdef _PRINT_MATRICES
        std::cout<<"After Gaunt 159-162 Iteration #"<<bf1<<std::endl;
        prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
#endif //_PRINT_MATRICES
    
    
#ifdef _PRINT_MATRICES
    
        std::cout<<"After Gaunt LLSS|SSLL"<<std::endl;
        prettyPrintSmart(std::cout, "COULOMB",    ss.coulombMatrix->pointer(),      NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
        prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
    
#endif //_PRINT_MATRICES
    
#endif // Gaunt LLSS EXCHANGE
    
        /*------------------------------------*/
        /*   End of Gaunt (LL|SS) Contraction */
        /*------------------------------------*/
    
      } //GAUNT
  
    } // Loop over bf1 using 3-Index ERI
    } // Loop over s1 using 3-Index ERI



    /*******************************/
    /* Final Assembly of 4C Matrix */
    /*******************************/
    ROOT_ONLY(ss.comm);

    // Copy LS to SL part of the exchangeMatrix[MS]
    SetMat('C', NB1C, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+SL, NB2C);
    // Copy LS to SL part of the exchangeMatrix[MX]
    SetMat('C', NB1C, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+SL, NB2C);
    // Copy LS to SL part of the exchangeMatrix[MY]
    SetMat('C', NB1C, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+SL, NB2C);
    // Copy LS to SL part of the exchangeMatrix[MZ]
    SetMat('C', NB1C, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+SL, NB2C);

    // Form GD: G[D] = 2.0*J[D] - K[D]
    if( std::abs(xHFX) > 1e-12 ) {
      *ss.twoeH = -xHFX * *ss.exchangeMatrix;
    } else {
      ss.twoeH->clear();
    }
    // G[D] += 2*J[D]
    *ss.twoeH += 2.0 * *ss.coulombMatrix;


    mem.free(Scr1);
    mem.free(Scr2);
    mem.free(Scr3);
    mem.free(Scr4);


#ifdef _PRINT_MATRICES

    prettyPrintSmart(std::cout,"twoeH MS",ss.twoeH->S().pointer(),NB2C,NB2C,NB2C);
    prettyPrintSmart(std::cout,"twoeH MZ",ss.twoeH->Z().pointer(),NB2C,NB2C,NB2C);
    prettyPrintSmart(std::cout,"twoeH MY",ss.twoeH->Y().pointer(),NB2C,NB2C,NB2C);
    prettyPrintSmart(std::cout,"twoeH MX",ss.twoeH->X().pointer(),NB2C,NB2C,NB2C);


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


  }; // FourCompFock<MatsT, IntsT>::formGD3Index



  /**   
   *  \brief Forms the 4C Fock matrix using AO-direct
   */
  template <typename MatsT, typename IntsT>
  void FourCompFock<MatsT,IntsT>::formGDDirect(SingleSlater<MatsT,IntsT> &ss,
    EMPerturbation &pert, bool increment, double xHFX) {

    CQMemManager &mem = ss.memManager;
    GTODirectRelERIContraction<MatsT,IntsT> &relERICon =
        *std::dynamic_pointer_cast<GTODirectRelERIContraction<MatsT,IntsT>>(ss.ERI);

    // Decide list of onePDMs to use
    PauliSpinorSquareMatrices<MatsT> &contract1PDM
        = increment ? *ss.deltaOnePDM : *ss.onePDM;

    size_t NB1C  = ss.basisSet().nBasis;
    size_t NB2C  = 2 * ss.basisSet().nBasis;
    size_t NB4C  = 4 * ss.basisSet().nBasis;
    size_t NB1C2 = NB1C*NB1C;
    size_t NB1C4 = NB1C*NB1C*NB1C*NB1C;
    size_t NB1C3 = NB1C*NB1C*NB1C;
    size_t NB2C2 = NB2C*NB2C;
    size_t NB4C2 = NB4C*NB4C;

    size_t SS = NB2C*NB1C+NB1C;
    size_t LS = NB2C*NB1C;
    size_t SL = NB1C;

    auto MS = SCALAR;

    size_t mpiRank   = MPIRank(ss.comm);
    bool   isNotRoot = mpiRank != 0;

    PauliSpinorSquareMatrices<MatsT> exchangeMatrixLL(mem, NB1C);
    PauliSpinorSquareMatrices<MatsT> contract1PDMLL(mem, NB1C);
    PauliSpinorSquareMatrices<MatsT> contract1PDMSS(mem, NB1C);
    PauliSpinorSquareMatrices<MatsT> contract1PDMLS(mem, NB1C);
    PauliSpinorSquareMatrices<MatsT> contract1PDMSL(mem, NB1C);

    MatsT* ScrLLMS = mem.malloc<MatsT>(NB1C2);
    MatsT* ScrLLMX = mem.malloc<MatsT>(NB1C2);
    MatsT* ScrLLMY = mem.malloc<MatsT>(NB1C2);
    MatsT* ScrLLMZ = mem.malloc<MatsT>(NB1C2);
    MatsT* ScrSSMS = mem.malloc<MatsT>(NB1C2);
    MatsT* ScrSSMX = mem.malloc<MatsT>(NB1C2);
    MatsT* ScrSSMY = mem.malloc<MatsT>(NB1C2);
    MatsT* ScrSSMZ = mem.malloc<MatsT>(NB1C2);
    MatsT* ScrLSMS = mem.malloc<MatsT>(NB1C2);
    MatsT* ScrLSMX = mem.malloc<MatsT>(NB1C2);
    MatsT* ScrLSMY = mem.malloc<MatsT>(NB1C2);
    MatsT* ScrLSMZ = mem.malloc<MatsT>(NB1C2);
    memset(ScrLLMS,0.,NB1C2*sizeof(MatsT));
    memset(ScrLLMX,0.,NB1C2*sizeof(MatsT));
    memset(ScrLLMY,0.,NB1C2*sizeof(MatsT));
    memset(ScrLLMZ,0.,NB1C2*sizeof(MatsT));
    memset(ScrSSMS,0.,NB1C2*sizeof(MatsT));
    memset(ScrSSMX,0.,NB1C2*sizeof(MatsT));
    memset(ScrSSMY,0.,NB1C2*sizeof(MatsT));
    memset(ScrSSMZ,0.,NB1C2*sizeof(MatsT));
    memset(ScrLSMS,0.,NB1C2*sizeof(MatsT));
    memset(ScrLSMX,0.,NB1C2*sizeof(MatsT));
    memset(ScrLSMY,0.,NB1C2*sizeof(MatsT));
    memset(ScrLSMZ,0.,NB1C2*sizeof(MatsT));


    // Compute 1/(2mc)^2
    //dcomplex scale = 1.;
    //dcomplex iscale = dcomplex(0.0, 1.0);
    dcomplex scale = 1./(4*SpeedOfLight*SpeedOfLight);
    dcomplex iscale = dcomplex(0.0, 1./(4*SpeedOfLight*SpeedOfLight));

    for(size_t i = 0; i < contract1PDM.nComponent(); i++) {
      PAULI_SPINOR_COMPS c = static_cast<PAULI_SPINOR_COMPS>(i);
      SetMat('N', NB1C, NB1C, MatsT(1.), contract1PDM[c].pointer(),    NB2C,
             contract1PDMLL[c].pointer(), NB1C);
      SetMat('N', NB1C, NB1C, MatsT(1.), contract1PDM[c].pointer()+SS, NB2C,
             contract1PDMSS[c].pointer(), NB1C);
      SetMat('N', NB1C, NB1C, MatsT(1.), contract1PDM[c].pointer()+LS, NB2C,
             contract1PDMLS[c].pointer(), NB1C);
      SetMat('N', NB1C, NB1C, MatsT(1.), contract1PDM[c].pointer()+SL, NB2C,
             contract1PDMSL[c].pointer(), NB1C);
    }

#ifdef _PRINT_MATRICES
    prettyPrintSmart(std::cout, "1PDM[MS]", contract1PDM.S().pointer(), NB2C, NB2C, NB2C);
    prettyPrintSmart(std::cout, "1PDM[MX]", contract1PDM.X().pointer(), NB2C, NB2C, NB2C);
    prettyPrintSmart(std::cout, "1PDM[MY]", contract1PDM.Y().pointer(), NB2C, NB2C, NB2C);
    prettyPrintSmart(std::cout, "1PDM[MZ]", contract1PDM.Z().pointer(), NB2C, NB2C, NB2C);
#endif

#if 0
    std::fill_n(contract1PDMLL.S().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMLL.X().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMLL.Y().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMLL.Z().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMSS.S().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMSS.X().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMSS.Y().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMSS.Z().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMLS.S().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMLS.X().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMLS.Y().pointer(),NB1C2,1.0);
    std::fill_n(contract1PDMLS.Z().pointer(),NB1C2,1.0);
#endif

    if(not increment) {
      ss.coulombMatrix->clear();
      ss.exchangeMatrix->clear();
    };


    /**********************************************/
    /*                                            */
    /*              DIRECT COULOMB     	          */
    /*                                            */
    /**********************************************/


    if(this->hamiltonianOptions_.BareCoulomb) { // DIRECT_COULOMB

      auto topBareCoulomb = tick();

      /*+++++++++++++++++++++++++++++++++++++++++++++*/
      /* Start of Direct Coulomb (LL|LL) Contraction */
      /*+++++++++++++++++++++++++++++++++++++++++++++*/
  
      std::vector<TwoBodyContraction<MatsT>> contractLL =
        { {contract1PDMLL.S().pointer(), ScrLLMS, true, COULOMB} };
  
      // Determine how many (if any) exchange terms to calculate
      if( std::abs(xHFX) > 1e-12 ) {
        exchangeMatrixLL.clear();
        for(size_t i = 0; i < ss.exchangeMatrix->nComponent(); i++) {
  
          PAULI_SPINOR_COMPS c = static_cast<PAULI_SPINOR_COMPS>(i);
          contractLL.push_back(
            {contract1PDMLL[c].pointer(), exchangeMatrixLL[c].pointer(), true, EXCHANGE}
          );
        }
      }
  
      // Zero out K[i]
      if(not increment) ss.exchangeMatrix->clear();
  
      // Call the contraction engine to do the assembly of direct Coulomb LLLL
      GTODirectERIContraction<MatsT,IntsT>(ss.ERI->ints()).twoBodyContract(ss.comm, true, contractLL, pert);
  
  
      /* Store LL block into 2C spin scattered matrices */
      // Assemble 4C coulombMatrix
      SetMat('N', NB1C, NB1C, MatsT(1.), ScrLLMS, NB1C, ss.coulombMatrix->pointer(), NB2C);
  
      // Assemble 4C exchangeMatrix 
      for(auto i = 0; i < ss.exchangeMatrix->nComponent();i++){
        PAULI_SPINOR_COMPS c = static_cast<PAULI_SPINOR_COMPS>(i);
        SetMat('N', NB1C, NB1C, MatsT(1.), exchangeMatrixLL[c].pointer(), NB1C,
               (*ss.exchangeMatrix)[c].pointer(), NB2C);
      }

      /*---------------------------------------------*/
      /*   End of Direct Coulomb (LL|LL) Contraction */
      /*---------------------------------------------*/

      // Print out BareCoulomb duration 
      auto durBareCoulomb = tock(topBareCoulomb);
//      std::cout << "Non-relativistic Coulomb duration = " << durBareCoulomb << std::endl;

    } // DIRECT_COULOMB


    /**********************************************/
    /*                                            */
    /*              DIRAC-COULOMB    	            */
    /*                                            */
    /**********************************************/

    if(this->hamiltonianOptions_.DiracCoulomb) { // DIRAC_COULOMB
  
      /*++++++++++++++++++++++++++++++++++++++++++++*/
      /* Start of Dirac-Coulomb (LL|LL) Contraction */
      /*++++++++++++++++++++++++++++++++++++++++++++*/

      // 12 density matrices upon input stored as
      // LL(MS,MX,MY,MZ), SS(MS,MX,MY,MZ), LS(MS,MX,MY,MZ)
      //
      // 12 contrated matrices upon output stored as
      // LL(MS,MX,MY,MZ), SS(MS,MX,MY,MZ), LS(MS,MX,MY,MZ)
      //
   
      std::vector<TwoBodyContraction<MatsT>> contractDCLL =
        { {contract1PDMLL.S().pointer(), ScrLLMS, true, COULOMB},
          {contract1PDMLL.X().pointer(), ScrLLMX, true, COULOMB},
          {contract1PDMLL.Y().pointer(), ScrLLMY, true, COULOMB},
          {contract1PDMLL.Z().pointer(), ScrLLMZ, true, COULOMB},
          {contract1PDMSS.S().pointer(), ScrSSMS, true, COULOMB},
          {contract1PDMSS.X().pointer(), ScrSSMX, true, COULOMB},
          {contract1PDMSS.Y().pointer(), ScrSSMY, true, COULOMB},
          {contract1PDMSS.Z().pointer(), ScrSSMZ, true, COULOMB},
          {contract1PDMLS.S().pointer(), ScrLSMS, true, COULOMB},
          {contract1PDMLS.X().pointer(), ScrLSMX, true, COULOMB},
          {contract1PDMLS.Y().pointer(), ScrLSMY, true, COULOMB},
          {contract1PDMLS.Z().pointer(), ScrLSMZ, true, COULOMB} };
    
      // Call the contraction engine to do the assembly of Dirac-Coulomb LLLL
      relERICon.twoBodyContract(ss.comm, true, contractDCLL,pert);

      // All terms go into ss.exchange
      // Add Dirac-Coulomb contributions to the LLLL block
      MatAdd('N','N', NB1C, NB1C, scale, ScrLLMS, NB1C, MatsT(1.0), 
//		      ss.exchangeMatrix->S().pointer(), NB2C,
//		      ss.exchangeMatrix->S().pointer(), NB2C);
		      ss.coulombMatrix->pointer(), NB2C,
		      ss.coulombMatrix->pointer(), NB2C);

#ifdef _PRINT_MATRICES

      std::cout<<"After LLLL"<<std::endl;
      prettyPrintSmart(std::cout, "COULOMB",    ss.coulombMatrix->pointer(),      NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);

#endif
 
      // Add Dirac-Coulomb contributions to the SSSS block
      MatAdd('N','N', NB1C, NB1C, scale, ScrSSMS, NB1C, MatsT(1.0), 
//		      ss.exchangeMatrix->S().pointer()+SS, NB2C,
//		      ss.exchangeMatrix->S().pointer()+SS, NB2C);
		      ss.coulombMatrix->pointer()+SS, NB2C,
		      ss.coulombMatrix->pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C, MatsT(-2.0*scale), ScrSSMX, NB1C, MatsT(1.0), 
		      ss.exchangeMatrix->X().pointer()+SS, NB2C,
		      ss.exchangeMatrix->X().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C, MatsT(-2.0*scale), ScrSSMY, NB1C, MatsT(1.0), 
		      ss.exchangeMatrix->Y().pointer()+SS, NB2C,
		      ss.exchangeMatrix->Y().pointer()+SS, NB2C);
      MatAdd('N','N', NB1C, NB1C, MatsT(-2.0*scale), ScrSSMZ, NB1C, MatsT(1.0), 
		      ss.exchangeMatrix->Z().pointer()+SS, NB2C,
		      ss.exchangeMatrix->Z().pointer()+SS, NB2C);

#ifdef _PRINT_MATRICES

      std::cout<<"After SSSS"<<std::endl;
      prettyPrintSmart(std::cout, "COULOMB",    ss.coulombMatrix->pointer(),      NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);

#endif

      // Add Dirac-Coulomb contributions to the LLSS block
      MatAdd('N','N', NB1C, NB1C, -scale, ScrLSMS, NB1C, MatsT(1.0), 
		      ss.exchangeMatrix->S().pointer()+LS, NB2C,
		      ss.exchangeMatrix->S().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -scale, ScrLSMX, NB1C, MatsT(1.0), 
		      ss.exchangeMatrix->X().pointer()+LS, NB2C,
		      ss.exchangeMatrix->X().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -scale, ScrLSMY, NB1C, MatsT(1.0), 
		      ss.exchangeMatrix->Y().pointer()+LS, NB2C,
		      ss.exchangeMatrix->Y().pointer()+LS, NB2C);
      MatAdd('N','N', NB1C, NB1C, -scale, ScrLSMZ, NB1C, MatsT(1.0), 
		      ss.exchangeMatrix->Z().pointer()+LS, NB2C,
		      ss.exchangeMatrix->Z().pointer()+LS, NB2C);
#ifdef _PRINT_MATRICES
    
      std::cout<<"After Dirac-Coulomb"<<std::endl;
      prettyPrintSmart(std::cout, "COULOMB",    ss.coulombMatrix->pointer(),      NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-S", ss.exchangeMatrix->S().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-X", ss.exchangeMatrix->X().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Y", ss.exchangeMatrix->Y().pointer(), NB2C, NB2C, NB2C);
      prettyPrintSmart(std::cout, "EXCHANGE-Z", ss.exchangeMatrix->Z().pointer(), NB2C, NB2C, NB2C);
    
#endif //_PRINT_MATRICES
    
    
    } //_DIRAC_COULOMB
  



    /*******************************/
    /* Final Assembly of 4C Matrix */
    /*******************************/
    ROOT_ONLY(ss.comm);

    // Copy LS to SL part of the exchangeMatrix[MS]
    SetMat('C', NB1C, NB1C, MatsT(1.0), ss.exchangeMatrix->S().pointer()+LS, NB2C, ss.exchangeMatrix->S().pointer()+SL, NB2C);
    // Copy LS to SL part of the exchangeMatrix[MX]
    SetMat('C', NB1C, NB1C, MatsT(1.0), ss.exchangeMatrix->X().pointer()+LS, NB2C, ss.exchangeMatrix->X().pointer()+SL, NB2C);
    // Copy LS to SL part of the exchangeMatrix[MY]
    SetMat('C', NB1C, NB1C, MatsT(1.0), ss.exchangeMatrix->Y().pointer()+LS, NB2C, ss.exchangeMatrix->Y().pointer()+SL, NB2C);
    // Copy LS to SL part of the exchangeMatrix[MZ]
    SetMat('C', NB1C, NB1C, MatsT(1.0), ss.exchangeMatrix->Z().pointer()+LS, NB2C, ss.exchangeMatrix->Z().pointer()+SL, NB2C);

    // Form GD: G[D] = 2.0*J[D] - K[D]
    if( std::abs(xHFX) > 1e-12 ) {
      *ss.twoeH = -xHFX * *ss.exchangeMatrix;
    } else {
      ss.twoeH->clear();
    }
    // G[D] += 2*J[D]
    *ss.twoeH += 2.0 * *ss.coulombMatrix;


    mem.free(ScrLLMS);
    mem.free(ScrLLMX);
    mem.free(ScrLLMY);
    mem.free(ScrLLMZ);
    mem.free(ScrSSMS);
    mem.free(ScrSSMX);
    mem.free(ScrSSMY);
    mem.free(ScrSSMZ);
    mem.free(ScrLSMS);
    mem.free(ScrLSMX);
    mem.free(ScrLSMY);
    mem.free(ScrLSMZ);


#ifdef _PRINT_MATRICES

    prettyPrintSmart(std::cout,"twoeH MS",ss.twoeH->S().pointer(),NB2C,NB2C,NB2C);
    prettyPrintSmart(std::cout,"twoeH MX",ss.twoeH->X().pointer(),NB2C,NB2C,NB2C);
    prettyPrintSmart(std::cout,"twoeH MY",ss.twoeH->Y().pointer(),NB2C,NB2C,NB2C);
    prettyPrintSmart(std::cout,"twoeH MZ",ss.twoeH->Z().pointer(),NB2C,NB2C,NB2C);


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


  }; // FourCompFock<MatsT, IntsT>::formGD3Direct


  template <typename MatsT, typename IntsT>
  void FourCompFock<MatsT,IntsT>::formFock(SingleSlater<MatsT,IntsT> &ss,
    EMPerturbation &pert, bool increment, double xHFX) {

    // General fock build
    FockBuilder<MatsT,IntsT>::formFock(ss, pert, increment, xHFX);


  } // ROFock<MatsT,IntsT>::formFock


}; // namespace ChronusQ

