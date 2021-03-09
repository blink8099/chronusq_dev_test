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
#include <cqlinalg.hpp>
#include <util/matout.hpp>

namespace ChronusQ {

  /**
   *  \brief Parses the fchk file and overwrites mo1(mo2) 
   *
   *  If using internally-stored correlation-consistent basis
   *  sets in Gaussian calculation, include IOp(3/60=-1).
   *
   **/
  template <typename MatsT, typename IntsT>
  std::vector<int> SingleSlater<MatsT,IntsT>::fchkToCQMO() {

    std::ifstream fchkFile;
    fchkFile.open(fchkFileName);
    std::vector<int> sl;

    if ( not fchkFile.good() ) CErr("Could not find fchkFile. Use -s flag.");

    // dimension of mo1 and mo2
    auto NB =  basisSet().nBasis;
    auto NBC = this->nC * NB;
    auto NB2 = NB*NB;
    auto NBC2 = NBC*NBC;

    // Boolean for if fchk entries are found
    bool isBeta=false, readAlpha=false, readBeta=false;
    bool readShell=false;
    // Various integers
    int mo1Counter = 0, mo2Counter = 0, compCounter = 0, maxLfchk;
    // Various double
    double prevValue = 0.0;
    // Double pointer for handling complex mo1
    double* dptr = reinterpret_cast<double*>(this->mo[0].pointer());

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

      // Some sanity checks on fchk data
      // Checking NB
      if ( tokens.size() > 5 ){
        if ( tokens[2] == "basis" and tokens[3] == "functions" ){
          if ( std::stoi(tokens[5]) != NB ){
            std::cout << "      NB from fchk is " << tokens[5] << " and NB in Chronus is " << NB << "\n";
            CErr("Basis functions do not agree between fchk and Chronus!");
          }
        }
      }

      // Checking highest angular momentum
      if ( tokens.size() > 4 ){
        if ( tokens[0] == "Highest" and tokens[1] == "angular" ){
          maxLfchk = std::stoi(tokens[4]);
          if( maxLfchk > 5 ){
            std::cout << "Max L in fchk is " << maxLfchk << ", but greater than " << 5 << " is NYI" << "\n";
            CErr("Angular momentum too high for fchk parser!");
          }
        }
      }

      // Checking size of mo1
      if ( tokens.size() > 5 ){
        if ( tokens[0] == "Alpha" and tokens[1] == "MO" ){
          if ( this->nC == 1 ){
            if ( std::stoi(tokens[5]) != NBC2 ){
              std::cout << "      MO size from fchk is " << tokens[5] << " and MO size in Chronus is " << NBC2 << "\n";
              CErr("MO coefficient size do not agree between fchk and ChronusQ!");
            }
          } else if ( this->nC == 2 ){
            // Factor of two here since real and imaginary are separate in fchk
            if ( std::stoi(tokens[5]) != 2*NBC2 ){
              std::cout << "      MO size from fchk is " << tokens[5] << " but expected size is " << 2*NBC2 << "\n";
              CErr("MO coefficient size on fchk is not compatible!");
            }
          } else{
            CErr("fchk functionality only implemented for 1c and 2c!");
          }
          // Found alpha MO block
          readAlpha=true;
          continue;
        }
      }

      // Check if unrestricted
      if ( tokens.size() > 5 ){
        if ( tokens[0] == "Beta" and tokens[1] == "MO" ){
          if ( this->nC == 2 or this->iCS ) CErr("Beta MOs present in fchk file but Chronus is 2c or restricted");
          isBeta = true;
          readBeta = true;
          readAlpha = false;
          continue;
        }
      }

      // Check if shell list started
      if ( tokens.size() > 4 ){
        if ( tokens[0] == "Shell" and tokens[1] == "types" ){
          readShell = true;
          continue;
        }
      }

      // Check if shell list ended
      if ( tokens.size() > 5 ){
        if ( tokens[2] == "primitives" and tokens[3] == "per" ){
          readShell = false;
          continue;
        }
      }

      // Check if alpha or beta block ended
      if ( tokens.size() == 5 ){
        if ( tokens[0] == "Orthonormal" and tokens[1] == "basis" ){
          if( this-> nC == 1){
            if( isBeta ) readBeta = false;
            else readAlpha = false;
            continue;
          }
          if( this-> nC == 2){
            readAlpha = false;
            continue;
          }
        }
      }

      // Read in Alpha MO coeffients
      if ( readAlpha ){
        for(int i=0; i<tokens.size(); i++){
          dptr[mo1Counter]=std::stod(tokens[i]);
          mo1Counter=mo1Counter+1;
        }
      }

      // Read in Beta MO coeffients
      if ( readBeta ){
        for(int i=0; i<tokens.size(); i++){
          this->mo[1].pointer()[mo2Counter]=std::stod(tokens[i]);
          mo2Counter=mo2Counter+1;
        }
      }

      // Read in shell types
      if ( readShell ){
        for(int i=0; i<tokens.size(); i++){
          sl.push_back(std::stoi(tokens[i]));
        }
      }

    }// end of eof loop

    if ( not isBeta and this->nC == 1 and not this->iCS ) CErr("Could not find beta MOs on fchk file!");

    fchkFile.close();

    return sl;

  } // SingleSlater<T>::fchkToCQMO()

  /**
   *  \brief Reorders basis functions for a given l to CQ storage  
   *  Example (d orbitals): 0,+1,-1,+2,-2 to -2,-1,0,+1,+2 
   *
   **/
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::reorderAngMO(std::vector<int> sl, MatsT* tmo, int sp) {

    // Dimension of mo1
    auto NB = basisSet().nBasis;
    auto NBC = this->nC * NB;
    auto NBC2 = NBC*NBC;

    // Angular momentum reordered by using tons of counters for now
    // Basic counters for shells and MOs
    auto shellcounter=0, moentry=0;
    // Lines skipped is different for 2c
    int skipl = this->nC;
    // Spherical d function reordering
    // 0,+1,-1,+2,-2 to -2,-1,0,+1,+2
    int dcounter=0,dzeromove=2*skipl,dponemove=2*skipl;
    int dmonemove=-skipl,dptwomove=skipl,dmtwomove=-4*skipl;
    // Cartesian d function reordering
    int cdcounter=0,cdxxmove=0,cdyymove=2*skipl,cdzzmove=3*skipl;
    int cdxymove=-2*skipl,cdxzmove=-2*skipl,cdyzmove=-skipl;
    // Spherical f function reordering
    // 0,+1,-1,+2,-2,+3,-3 to -3,-2,-1,0,+1,+2,+3
    int fcounter=0,fzeromove=3*skipl,fponemove=3*skipl,fmonemove=0;
    int fptwomove=2*skipl,fmtwomove=-3*skipl,fpthreemove=skipl;
    int fmthreemove=-6*skipl;
    // Spherical g function reordering
    // 0,+1,-1,+2,-2,+3,-3,+4,-4 to -4,-3,-2,-1,0,+1,+2,+3,+4
    int gcounter=0,gzeromove=4*skipl,gponemove=4*skipl,gmonemove=skipl;
    int gptwomove=3*skipl,gmtwomove=-2*skipl,gpthreemove=2*skipl;
    int gmthreemove=-5*skipl,gpfourmove=skipl,gmfourmove=-8*skipl;
    // Spherical h function reordering
    // 0,+1,-1,+2,-2,+3,-3,+4,-4,+5,-5 to -5,-4,-3,-2,-1,0,+1,+2,+3,+4,+5
    int hcounter=0,hzeromove=5*skipl,hponemove=5*skipl,hmonemove=2*skipl;
    int hptwomove=4*skipl,hmtwomove=-1*skipl,hpthreemove=3*skipl;
    int hmthreemove=-4*skipl,hpfourmove=2*skipl,hmfourmove=-7*skipl;
    int hpfivemove=1*skipl,hmfivemove=-10*skipl;

    // Loop over mo
    for( int i=0; i<NBC2; i++){

      // Reset shellcounter for each MO
      if( i % NBC == 0 ) shellcounter=0;

      // Sanity check on shellcounter
      if( shellcounter > sl.size() ){
        std::cout << "shellcounter > size of sl " << "\n";
        continue;
      }

      // Skipping through s and p and beta d, f, g
      if( i < moentry ) continue;

      // Conditional for shell types
      // s functions
      if( sl[shellcounter] == 0 ){
        moentry = moentry + skipl;
        shellcounter = shellcounter + 1;
        continue;
      // Spherical p functions
      }else if( sl[shellcounter] == 1 ){
        moentry = moentry + 3*skipl;
        shellcounter = shellcounter + 1;
        continue;
      // Cartesian p functions
      }else if( sl[shellcounter] == -1 ){
        // Factor of 4 instead of 3 because grouped as SP
        moentry = moentry + 4*skipl;
        shellcounter = shellcounter + 1;
        continue;
      // Spherical d functions
      }else if( sl[shellcounter] == -2 ){

        // d0
        if( dcounter == 0 ){
          this->mo[sp].pointer()[i + skipl - 1 + dzeromove] = tmo[i];
          moentry = moentry + skipl;
          dcounter = dcounter + 1;
          continue;
        }

        // d+1
        if( dcounter == 1 ){
          this->mo[sp].pointer()[i + skipl - 1 + dponemove] = tmo[i];
          moentry = moentry + skipl;
          dcounter = dcounter + 1;
          continue;
        }

        // d-1
        if( dcounter == 2 ){
          this->mo[sp].pointer()[i + skipl - 1 + dmonemove] = tmo[i];
          moentry = moentry + skipl;
          dcounter = dcounter + 1;
          continue;
        }

        // d+2
        if( dcounter == 3 ){
          this->mo[sp].pointer()[i + skipl - 1 + dptwomove] = tmo[i];
          moentry = moentry + skipl;
          dcounter = dcounter + 1;
          continue;
        }

        // d-2
        if( dcounter == 4 ){
          this->mo[sp].pointer()[i + skipl - 1 + dmtwomove] = tmo[i];
          moentry = moentry + skipl;
          // Reset d counter
          dcounter = 0;
          // Going to next shell
          shellcounter = shellcounter + 1;
          continue;
        }

      // Cartesian d functions
      }else if( sl[shellcounter] == 2 ){

        // dxx
        if( cdcounter == 0 ){
          this->mo[sp].pointer()[i + skipl - 1 + cdxxmove] = tmo[i];
          moentry = moentry + skipl;
          cdcounter = cdcounter + 1;
          continue;
        }

        // dyy
        if( cdcounter == 1 ){
          this->mo[sp].pointer()[i + skipl - 1 + cdyymove] = tmo[i];
          moentry = moentry + skipl;
          cdcounter = cdcounter + 1;
          continue;
        }

        // dzz
        if( cdcounter == 2 ){
          this->mo[sp].pointer()[i + skipl - 1 + cdzzmove] = tmo[i];
          moentry = moentry + skipl;
          cdcounter = cdcounter + 1;
          continue;
        }

        // dxy
        if( cdcounter == 3 ){
          this->mo[sp].pointer()[i + skipl - 1 + cdxymove] = tmo[i];
          moentry = moentry + skipl;
          cdcounter = cdcounter + 1;
          continue;
        }

        // dxz
        if( cdcounter == 4 ){
          this->mo[sp].pointer()[i + skipl - 1 + cdxzmove] = tmo[i];
          moentry = moentry + skipl;
          cdcounter = cdcounter + 1;
          continue;
        }

        // dyz
        if( cdcounter == 5 ){
          this->mo[sp].pointer()[i + skipl - 1 + cdyzmove] = tmo[i];
          moentry = moentry + skipl;
          // Reset d counter
          cdcounter = 0;
          // Going to next shell
          shellcounter = shellcounter + 1;
          continue;
        }

      // Spherical f functions
      }else if( sl[shellcounter] == -3 ){

        // f0
        if( fcounter == 0 ){
          this->mo[sp].pointer()[i + skipl - 1 + fzeromove] = tmo[i];
          moentry = moentry + skipl;
          fcounter = fcounter + 1;
          continue;
        }

        // f+1
        if( fcounter == 1 ){
          this->mo[sp].pointer()[i + skipl - 1 + fponemove] = tmo[i];
          moentry = moentry + skipl;
          fcounter = fcounter + 1;
          continue;
        }

        // f-1
        if( fcounter == 2 ){
          this->mo[sp].pointer()[i + skipl - 1 + fmonemove] = tmo[i];
          moentry = moentry + skipl;
          fcounter = fcounter + 1;
          continue;
        }

        // f+2
        if( fcounter == 3 ){
          this->mo[sp].pointer()[i + skipl - 1 + fptwomove] = tmo[i];
          moentry = moentry + skipl;
          fcounter = fcounter + 1;
          continue;
        }

        // f-2
        if( fcounter == 4 ){
          this->mo[sp].pointer()[i + skipl - 1 + fmtwomove] = tmo[i];
          moentry = moentry + skipl;
          fcounter = fcounter + 1;
          continue;
        }

        // f+3
        if( fcounter == 5 ){
          this->mo[sp].pointer()[i + skipl - 1 + fpthreemove] = tmo[i];
          moentry = moentry + skipl;
          fcounter = fcounter + 1;
          continue;
        }

        // f-3
        if( fcounter == 6 ){
          this->mo[sp].pointer()[i + skipl - 1 + fmthreemove] = tmo[i];
          moentry = moentry + skipl;
          // Reset f counter
          fcounter = 0;
          // Going to next shell
          shellcounter = shellcounter + 1;
          continue;
        }

      // Spherical g functions
      }else if( sl[shellcounter] == -4 ){

        // g0
        if( gcounter == 0 ){
          this->mo[sp].pointer()[i + skipl - 1 + gzeromove] = tmo[i];
          moentry = moentry + skipl;
          gcounter = gcounter + 1;
          continue;
        }

        // g+1
        if( gcounter == 1 ){
          this->mo[sp].pointer()[i + skipl - 1 + gponemove] = tmo[i];
          moentry = moentry + skipl;
          gcounter = gcounter + 1;
          continue;
        }

        // g-1
        if( gcounter == 2 ){
          this->mo[sp].pointer()[i + skipl - 1 + gmonemove] = tmo[i];
          moentry = moentry + skipl;
          gcounter = gcounter + 1;
          continue;
        }

        // g+2
        if( gcounter == 3 ){
          this->mo[sp].pointer()[i + skipl - 1 + gptwomove] = tmo[i];
          moentry = moentry + skipl;
          gcounter = gcounter + 1;
          continue;
        }

        // g-2
        if( gcounter == 4 ){
          this->mo[sp].pointer()[i + skipl - 1 + gmtwomove] = tmo[i];
          moentry = moentry + skipl;
          gcounter = gcounter + 1;
          continue;
        }

        // g+3
        if( gcounter == 5 ){
          this->mo[sp].pointer()[i + skipl - 1 + gpthreemove] = tmo[i];
          moentry = moentry + skipl;
          gcounter = gcounter + 1;
          continue;
        }

        // g-3
        if( gcounter == 6 ){
          this->mo[sp].pointer()[i + skipl - 1 + gmthreemove] = tmo[i];
          moentry = moentry + skipl;
          gcounter = gcounter + 1;
          continue;
        }

        // g+4
        if( gcounter == 7 ){
          this->mo[sp].pointer()[i + skipl - 1 + gpfourmove] = tmo[i];
          moentry = moentry + skipl;
          gcounter = gcounter + 1;
          continue;
        }

        // g-4
        if( gcounter == 8 ){
          this->mo[sp].pointer()[i + skipl - 1 + gmfourmove] = tmo[i];
          moentry = moentry + skipl;
          // Reset g counter
          gcounter = 0;
          // Going to next shell
          shellcounter = shellcounter + 1;
          continue;
        }

      // Spherical h functions
      }else if( sl[shellcounter] == -5 ){

        // h0
        if( hcounter == 0 ){
          this->mo[sp].pointer()[i + skipl - 1 + hzeromove] = tmo[i];
          moentry = moentry + skipl;
          hcounter = hcounter + 1;
          continue;
        }

        // h+1
        if( hcounter == 1 ){
          this->mo[sp].pointer()[i + skipl - 1 + hponemove] = tmo[i];
          moentry = moentry + skipl;
          hcounter = hcounter + 1;
          continue;
        }

        // h-1
        if( hcounter == 2 ){
          this->mo[sp].pointer()[i + skipl - 1 + hmonemove] = tmo[i];
          moentry = moentry + skipl;
          hcounter = hcounter + 1;
          continue;
        }

        // h+2
        if( hcounter == 3 ){
          this->mo[sp].pointer()[i + skipl - 1 + hptwomove] = tmo[i];
          moentry = moentry + skipl;
          hcounter = hcounter + 1;
          continue;
        }

        // h-2
        if( hcounter == 4 ){
          this->mo[sp].pointer()[i + skipl - 1 + hmtwomove] = tmo[i];
          moentry = moentry + skipl;
          hcounter = hcounter + 1;
          continue;
        }

        // h+3
        if( hcounter == 5 ){
          this->mo[sp].pointer()[i + skipl - 1 + hpthreemove] = tmo[i];
          moentry = moentry + skipl;
          hcounter = hcounter + 1;
          continue;
        }

        // h-3
        if( hcounter == 6 ){
          this->mo[sp].pointer()[i + skipl - 1 + hmthreemove] = tmo[i];
          moentry = moentry + skipl;
          hcounter = hcounter + 1;
          continue;
        }

        // h+4
        if( hcounter == 7 ){
          this->mo[sp].pointer()[i + skipl - 1 + hpfourmove] = tmo[i];
          moentry = moentry + skipl;
          hcounter = hcounter + 1;
          continue;
        }

        // h-4
        if( hcounter == 8 ){
          this->mo[sp].pointer()[i + skipl - 1 + hmfourmove] = tmo[i];
          moentry = moentry + skipl;
          hcounter = hcounter + 1;
          continue;
        }

        // h+5
        if( hcounter == 9 ){
          this->mo[sp].pointer()[i + skipl - 1 + hpfivemove] = tmo[i];
          moentry = moentry + skipl;
          hcounter = hcounter + 1;
          continue;
        }

        // h-5
        if( hcounter == 10 ){
          this->mo[sp].pointer()[i + skipl - 1 + hmfivemove] = tmo[i];
          moentry = moentry + skipl;
          // Reset h counter
          hcounter = 0;
          // Going to next shell
          shellcounter = shellcounter + 1;
          continue;
        }

      // If shell value is not yet implemented
      } else{
        std::cout << "Current shell entry: " << sl[shellcounter] << "\n";
        CErr("Current shell value NYI!");

      } // End of shell checking

    } // Loop over mo

  } // SingleSlater<T>::reorderAngMO()


  /**
   *  \brief Reorders spin components of basis functions (needed for 2c only) 
   *  Example (1 real 2c MO with 2 basis functions): 
   *  alpha1,beta1,alpha2,beta2 to alpha1,alpha2,beta1,beta2
   *
   **/
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::reorderSpinMO() {

    // Dimension of mo1
    auto NB = basisSet().nBasis;
    auto NBC = this->nC * NB;
    auto NBC2 = NBC*NBC;

    // Counters for spin reordering
    int acount = 0, bcount = NB - 1;

    // mo scratch
    MatsT* mo1tmp = memManager.malloc<MatsT>(NBC2);
    SetMat('N',NBC,NBC,MatsT(1.),this->mo[0].pointer(),NBC,mo1tmp,NBC);

    // Loop over mo1
    for( int i=0; i<NBC2; i++){

      // Reset acount and bcount for each MO
      if( i % NBC == 0 ){
        acount = 0;
        bcount = NB - 1;
      }

      // If alpha
      if( i % 2 == 0 ){
        this->mo[0].pointer()[i-acount] = mo1tmp[i];
        acount = acount + 1;
      }else{ // If beta
        this->mo[0].pointer()[i+bcount] = mo1tmp[i];
        bcount = bcount - 1;
      }
    }

    memManager.free(mo1tmp);

  } // SingleSlater<T>::reorderSpinMO()

}
