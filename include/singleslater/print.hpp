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

#include <singleslater.hpp>
#include <util/matout.hpp>


namespace ChronusQ {

  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::printFock(std::ostream &out) {

    size_t NB = basisSet().nBasis;

    prettyPrintSmart(out,"Fock (AO) Scalar",fockMatrix->S().pointer(),NB,NB,NB);

    if( fockMatrix->hasZ() )
      prettyPrintSmart(out,"Fock (AO) MZ",fockMatrix->Z().pointer(),NB,NB,NB);

    if( fockMatrix->hasXY() ) {
      prettyPrintSmart(out,"Fock (AO) MY",fockMatrix->Y().pointer(),NB,NB,NB);
      prettyPrintSmart(out,"Fock (AO) MX",fockMatrix->X().pointer(),NB,NB,NB);
    }


  }; // SingleSlater<T>::printFock

  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::print1PDMOrtho(std::ostream &out) {

    size_t NB = basisSet().nBasis;

    prettyPrintSmart(out,"1PDM (Ortho) Scalar",onePDMOrtho->S().pointer(),NB,NB,NB);

    if( onePDMOrtho->hasZ() )
      prettyPrintSmart(out,"1PDM (Ortho) MZ",onePDMOrtho->Z().pointer(),NB,NB,NB);

    if( onePDMOrtho->hasXY() ) {
      prettyPrintSmart(out,"1PDM (Ortho) MY",onePDMOrtho->Y().pointer(),NB,NB,NB);
      prettyPrintSmart(out,"1PDM (Ortho) MX",onePDMOrtho->X().pointer(),NB,NB,NB);
    }


  }; // SingleSlater<T>::print1PDMOrtho

  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::printGD(std::ostream &out) {

    size_t NB = basisSet().nBasis;

    prettyPrintSmart(out,"GD (AO) Scalar",twoeH->S().pointer(),NB,NB,NB);
    if (twoeH->hasZ())
      prettyPrintSmart(out,"GD (AO) MZ",twoeH->Z().pointer(),NB,NB,NB);
    if (twoeH->hasXY()) {
      prettyPrintSmart(out,"GD (AO) MY",twoeH->Y().pointer(),NB,NB,NB);
      prettyPrintSmart(out,"GD (AO) MX",twoeH->X().pointer(),NB,NB,NB);
    }

  }; // SingleSlater<T>::printGD


  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::printJ(std::ostream &out) {

    size_t NB = basisSet().nBasis;

    prettyPrintSmart(out,"J (AO) Scalar",coulombMatrix->pointer(),NB,NB,NB);


  }; // SingleSlater<T>::printJ


  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::printK(std::ostream &out) {

    size_t NB = basisSet().nBasis;

    prettyPrintSmart(out,"K (AO) Scalar",exchangeMatrix->S().pointer(),NB,NB,NB);
    if (exchangeMatrix->hasZ())
      prettyPrintSmart(out,"K (AO) MZ",exchangeMatrix->Z().pointer(),NB,NB,NB);
    if (exchangeMatrix->hasXY()) {
      prettyPrintSmart(out,"K (AO) MY",exchangeMatrix->Y().pointer(),NB,NB,NB);
      prettyPrintSmart(out,"K (AO) MX",exchangeMatrix->X().pointer(),NB,NB,NB);
    }

  }; // SingleSlater<T>::printK


  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::printMiscProperties(std::ostream &out) {

    out << "\nCharge Analysis:\n" << bannerTop << "\n\n";

    out << std::setw(20) << std::left << "  Atom";
    out << std::setw(20) << std::right << "Mulliken Charges";
    out << std::setw(20) << std::right << "Lowdin Charges" << std::endl;

    out << std::right << bannerMid << std::endl;

    Molecule & mol = this->molecule();
    for(auto iAtm = 0; iAtm < mol.nAtoms; iAtm++) {

      // Get symbol
      std::map<std::string,Atom>::const_iterator it = 
      std::find_if(atomicReference.begin(),atomicReference.end(),
        [&](const std::pair<std::string,Atom> &st){ 
          return (st.second.atomicNumber == mol.atoms[iAtm].atomicNumber) and
                 (st.second.massNumber == mol.atoms[iAtm].massNumber);}
         );

      out << "  " << std::setw(18) << std::left <<
        (it == atomicReference.end() ? "X" : it->first);

      out << std::setprecision(5) << std::right;

      out << std::setw(20) << mullikenCharges[iAtm];
      out << std::setw(20) << lowdinCharges[iAtm];

      out << std::endl;
    }

    out << std::endl << bannerEnd << std::endl;

  }; // SingleSlater<T>::printMiscProperties




  template <typename T, typename ValManipOp>
  void prettyMOPrint(std::ostream &out, size_t NB, size_t NOrb, double *EPS, 
    T* MO, size_t LDM, Molecule &mol, BasisSet &basis, 
    ValManipOp op ){

    constexpr size_t maxLPrint = 6 + 1;

    std::array< std::vector<std::string>, maxLPrint > cartLabel, sphLabel;
    std::array< std::string, maxLPrint > angLabel = 
    { "S", "P", "D", "F", "G", "H", "I" };

    std::array< std::string, 3 > axes = { "X", "Y", "Z" };

    // S Functions
    cartLabel[0] = { angLabel[0] };

    for(auto i = 0; i < 3; i++) {

      // P Functions
      cartLabel[1].emplace_back( angLabel[1] + axes[i] );

    for(auto j = i; j < 3; j++) {

      // D Functions
      cartLabel[2].emplace_back( angLabel[2] + axes[i] + axes[j] );

    for(auto k = j; k < 3; k++) {

      // F Functions
      cartLabel[3].emplace_back( angLabel[3] + axes[i] + axes[j] + axes[k] );

    for(auto l = k; l < 3; l++) {

      // G Functions
      cartLabel[4].emplace_back( angLabel[4] + axes[i] + axes[j] + axes[k] 
          + axes[l] );

    for(auto m = l; m < 3; m++) {

      // H Functions
      cartLabel[5].emplace_back( angLabel[5] + axes[i] + axes[j] + axes[k] 
          + axes[l] + axes[m] );

    for(auto n = m; n < 3; n++) {

      // I Functions
      cartLabel[6].emplace_back( angLabel[6] + axes[i] + axes[j] + axes[k] 
          + axes[l] + axes[m] + axes[n] );

    }}}}}}

    // Spherical Labels

    auto pm_string = [](int x) -> std::string {

      if( x > 0 )      return "+" + std::to_string(x) ;
      else if (x < 0 ) return "" + std::to_string(x) ;
      else             return " " + std::to_string(x) ;

    };

    sphLabel[0] = cartLabel[0];
    sphLabel[1] = cartLabel[1];
    for(auto i = 2; i < maxLPrint; i++) {

      for(auto ml = -i; ml <= i; ml++)
        sphLabel[i].emplace_back( angLabel[i] + pm_string(ml) );

    }

    size_t list = 4;
    size_t printWidth = 14;
    size_t eValOff = 24;


    out << std::endl << bannerTop << std::endl;

    out << std::scientific << std::left << std::setprecision(5);
    for(size_t p = 0; p < NOrb; p += list) {

      out << std::left << std::endl;
      int end = list;

      if( (p + list) >= NOrb ) end = NOrb - p;

      out << std::left << std::setw(eValOff) << " ";
      out << std::right;
      for(size_t k = p; k < p + end; k++)
        out << std::setw(printWidth) << k + 1;
      out << std::endl;

      out << std::left << std::setw(eValOff) << " EigV --";
      out << std::right;
      for(size_t k = p; k < p + end; k++)
        out << std::setw(printWidth) << EPS[k];
      out << std::endl << std::endl;



      auto getAtmSymb = [&](size_t iAtm) -> std::string {

        std::map<std::string,Atom>::const_iterator it = 
        std::find_if(atomicReference.begin(),atomicReference.end(),
          [&](const std::pair<std::string,Atom> &st){ 
            return (st.second.atomicNumber == mol.atoms[iAtm].atomicNumber) and
                   (st.second.massNumber == mol.atoms[iAtm].massNumber);}
           );

        return (it == atomicReference.end() ? "X" : it->first);

      };

      size_t iAtm = 0;
      std::string atmSymb = getAtmSymb(iAtm);
      size_t iShellAtm = 0;

      size_t nShell = basis.shells.size();

      for(size_t iShell = 0; iShell < nShell ; iShell++, iShellAtm++) {

        size_t bfst = basis.mapSh2Bf[iShell];
        size_t sz   = basis.shells[iShell].size();
        size_t L    = basis.shells[iShell].contr[0].l;

        size_t curCen = basis.mapSh2Cen[iShell];
        bool newAtm = false;
        if( curCen != iAtm ) {
          newAtm = true;
          iAtm++;
          atmSymb = getAtmSymb(iAtm);
          iShellAtm = 0;
        }

        bool lastAtm  = iAtm == (mol.atoms.size() - 1);


        bool firstAtm = iShell == 0;

        for(size_t mu = bfst, ml = 0 ; ml < sz; mu++, ml++) {

          // BF number
          out << " " << std::setw(5) << std::left <<  mu + 1; // 6


          // 16
          if( (newAtm or firstAtm) ) {
            out << std::setw(3) << iAtm;
            out << std::setw(7) << atmSymb;
          } else out << std::setw(10) << " ";

          out << std::setw(2) << iShellAtm ;
          out << std::setw(4) << sphLabel[L][ml]; //22

          out << std::setw(2) << " "; // 24


          for(auto q = p; q < p + end; q++) {

            double VAL = op(MO[mu + q*LDM]); 
            out << std::right << std::setw(printWidth);

            if(std::abs(VAL) > PRINT_SMALL)    out << VAL; 
            else if(std::isnan(std::abs(VAL))) out << "NAN";
            else if(std::isinf(std::abs(VAL))) out << "INF";
            else                               out << 0.;

          }



          out << std::endl;

          bool nextAtmNew = lastAtm ? false : 
            (mu+1) >= basis.mapCen2BfSt[iAtm+1];

          if( nextAtmNew ) out<<std::endl;

        }

      }



      out << std::endl;

    }


    out << bannerEnd << std::endl;


  };

  template <typename T>
  void prettyMOPrint(std::ostream &out, size_t NB, size_t NOrb, double *EPS, 
    T* MO, size_t LDM, Molecule &mol, BasisSet &basis){

    prettyMOPrint(out,NB,NOrb,EPS,MO,LDM,mol,basis,
        [](T x){ return x; });

  }


  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::printMOInfo(std::ostream &out) {

    out << std::scientific << std::setprecision(4);

    out << "\n\n" << "SCF Results:\n" << BannerTop << "\n\n";



    // List MO eigenenergies


    out << "Orbital Eigenenergies " << (this->nC == 1 ? "(Alpha) " : "" )
        << "/ Eh\n" << bannerTop << "\n";

    size_t NO = (this->nC == 1 ? this->nOA : this->nO);
    for(auto i = 0ul; i < this->nC * this->nAlphaOrbital(); i++) {

      if( i == 0 )
        out << "Occupied:\n";
      else if( i == NO )
        out << "\n\nVirtual:\n";

      out << std::setw(13) << this->eps1[i];

      if( i < NO and (i + 1) % 5 == 0 )  out << "\n";
      else if( i >= NO and ((i - NO) + 1) % 5 == 0 ) out << "\n";
    }

     
    out << "\n" << bannerEnd << "\n";

    if( this->nC == 1 and not this->iCS ) {
      out << "\n\nOrbital Eigenenergies (Beta) / Eh\n" << bannerTop << "\n";

      for(auto i = 0ul; i < this->nBetaOrbital(); i++) {

        if( i == 0 )
          out << "Occupied:\n";
        if( i == this->nOB )
          out << "\n\nVirtual:\n";

        out << std::setw(13) << this->eps2[i];

        if( i < this->nOB and (i + 1) % 5 == 0 )  out << "\n";
        else if( i >= this->nOB and ((i - this->nOB) + 1) % 5 == 0 ) out << "\n";
      }

      out << "\n" << bannerEnd << "\n";
    }




    if( not scfControls.printMOCoeffs ) return;








    // Pretty MO print
    size_t NB = this->nAlphaOrbital();
    size_t NOrb = NB * this->nC;

    std::function<double(MatsT)> printOp1 = [](MatsT x) { return std::real(x); };
    std::function<double(MatsT)> printOp2 = printOp1;

    if( std::is_same<MatsT,dcomplex>::value ) {
      printOp1 = [](MatsT x) { return std::abs(x); };
      printOp2 = [](MatsT x) { return std::arg(x); };
    }

    if( this->nC == 2 )
    out << " *** NOTICE: Alpha and Beta Coefficients refer to the SAME "
      << "Canonical MOs ***\n";


    out << "\n\nCanonical Molecular Orbital Coefficients (Alpha)"; 
    if( std::is_same<MatsT,dcomplex>::value ) 
      out << " Magnitude";

    prettyMOPrint(out,NB,NOrb,this->eps1,this->mo[0].pointer(),NOrb,
        this->molecule(),basisSet(),printOp1);


    if( std::is_same<MatsT,dcomplex>::value ) {
      out << "\n\nCanonical Molecular Orbital Coefficients (Alpha) Phase"; 
      prettyMOPrint(out,NB,NOrb,this->eps1,this->mo[0].pointer(),NOrb,
          this->molecule(),basisSet(),printOp2);
    }







    if( this->nC == 2 or not this->iCS ) {

      out << "\n\nCanonical Molecular Orbital Coefficients (Beta)"; 
      if( std::is_same<MatsT,dcomplex>::value ) 
        out << " Magnitude";

      if( this->nC == 1 )
        prettyMOPrint(out,NB,NOrb,this->eps2,this->mo[1].pointer(),NOrb,
            this->molecule(),basisSet(),printOp1);
      else
        prettyMOPrint(out,NB,NOrb,this->eps1,this->mo[0].pointer() + NB,NOrb,
            this->molecule(),basisSet(),printOp1);

      if( std::is_same<MatsT,dcomplex>::value ) {
        out << "\n\nCanonical Molecular Orbital Coefficients (Beta) Phase"; 

        if( this->nC == 1 )
          prettyMOPrint(out,NB,NOrb,this->eps2,this->mo[1].pointer(),NOrb,
              this->molecule(),basisSet(),printOp2);
        else
          prettyMOPrint(out,NB,NOrb,this->eps1,this->mo[0].pointer() + NB,NOrb,
              this->molecule(),basisSet(),printOp2);
      }

    }



    out << "\n" << BannerEnd << "\n\n";
  }; // SingleSlater<T>::printMOInfo



  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT, IntsT>::printFockTimings(std::ostream &out) {

    out << "    Fock Timings:\n";
    out << "      Wall time G[D] = " << std::setw(8)
        << std::setprecision(5)  << std::scientific
        << GDDur << " s\n\n";


  }; // SingleSlater<T>::printFockTimings

}; // namespace ChronusQ

