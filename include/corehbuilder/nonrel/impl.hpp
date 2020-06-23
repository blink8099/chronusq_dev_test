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

#include <corehbuilder/nonrel.hpp>

namespace ChronusQ {

  template <>
  void NRCoreH<dcomplex, dcomplex>::addMagPert(EMPerturbation &pert,
      std::vector<dcomplex*> &CH) {


    //Compute the GIAO non-relativistic core Hamiltonian in the CGTO basis
    //H(S) = 2(T + V) + B * L + sigma * B + 1/4 *(B\timesr)^2

    size_t NB = aoints_.basisSet().nBasis;
    dcomplex onei = dcomplex(0,1);
    auto magAmp = pert.getDipoleAmp(Magnetic);


    // this part add the angular momentum term
    for ( auto index = 0 ; index < 3 ; index++ ) {
      MatAdd('N','N',NB,NB,-magAmp[index]*onei,
        aoints_.magDipole[index],NB,dcomplex(1.),CH[0],NB,CH[0],NB);
    } // for ( auto inde = 0 ; inde < 3 ; inde++ )

    // this part add the length gauge electric quadrupole term
    int diagindex[3];
    diagindex[0] = 0;  // xx
    diagindex[1] = 3;  // yy
    diagindex[2] = 5;  // zz

    double diagcoeff[3];
    diagcoeff[0] = 1.0/8.0*(magAmp[1]*magAmp[1]+magAmp[2]*magAmp[2]);
    diagcoeff[1] = 1.0/8.0*(magAmp[0]*magAmp[0]+magAmp[2]*magAmp[2]);
    diagcoeff[2] = 1.0/8.0*(magAmp[0]*magAmp[0]+magAmp[1]*magAmp[1]);

    // add diagonal part
    for ( auto index = 0 ; index < 3 ; index++ ) {
      MatAdd('N','N',NB,NB,
        dcomplex(2.0*diagcoeff[index]),
        aoints_.lenElecQuadrupole[diagindex[index]],
        NB,dcomplex(1.),CH[0],NB,CH[0], NB);
    }

    int offindex[3];
    offindex[0] = 1;  // xy
    offindex[1] = 2;  // xz
    offindex[2] = 4;  // yz

    double offcoeff[3];
    offcoeff[0] = -1.0/4.0*magAmp[0]*magAmp[1];
    offcoeff[1] = -1.0/4.0*magAmp[0]*magAmp[2];
    offcoeff[2] = -1.0/4.0*magAmp[1]*magAmp[2];

    // add off diagonal part
    for ( auto index = 0 ; index < 3 ; index++ ) {
      MatAdd('N','N',NB,NB,
        dcomplex(2.0*offcoeff[index]),
        aoints_.lenElecQuadrupole[offindex[index]],
        NB,dcomplex(1.),CH[0],NB,CH[0], NB);
    }


    // finally spin Zeeman term
    if(CH.size() > 1) {
      // z component
      SetMat('N',NB,NB,dcomplex(magAmp[2]),aoints_.overlap,
        NB, CH[1],NB );

      if(CH.size() > 2) {
        // y component
        SetMat('N',NB,NB,dcomplex(magAmp[1]),aoints_.overlap,
          NB, CH[2],NB );

        // x coponent
        SetMat('N',NB,NB,dcomplex(magAmp[0]),aoints_.overlap,
          NB, CH[3],NB );
      }
    }


  }

  template <>
  void NRCoreH<dcomplex, double>::addMagPert(EMPerturbation &pert,
    std::vector<dcomplex*> &CH) {


    CErr("GIAO + Real integrals is not a valid option");

  }
  template <>
  void NRCoreH<double, double>::addMagPert(EMPerturbation &pert,
    std::vector<double*> &CH) {


    CErr("GIAO + Real integrals is not a valid option");

  }

  /**
   *  \brief Compute the non-relativistic Core Hamiltonian in the CGTO basis
   *
   *  \f[ H(S) = 2(T + V) \f]
   */
  template <typename MatsT, typename IntsT>
  void NRCoreH<MatsT,IntsT>::computeCoreH(EMPerturbation& emPert, std::vector<MatsT*> &CH) {

    this->aoints_.computeAOOneE(emPert,this->oneETerms_); // compute the necessary 1e ints

    computeNRCH(emPert, CH);

  };  // void NRCoreH::computeCoreH(std::vector<MatsT*> &CH)

  /**
   *  \brief Compute the non-relativistic Core Hamiltonian in the CGTO basis
   *
   *  \f[ H(S) = 2(T + V) \f]
   */
  template <typename MatsT, typename IntsT>
  void NRCoreH<MatsT,IntsT>::computeNRCH(EMPerturbation& emPert, std::vector<MatsT*> &CH) {

    size_t NB = this->aoints_.basisSet().nBasis;

    // MatAdd for Real + Real -> Complex does not make sense
    for(auto k = 0ul; k < NB*NB; k++)
      CH[0][k] = 2. * (this->aoints_.kinetic[k] + this->aoints_.potential[k]);

    if( this->aoints_.basisSet().basisType == COMPLEX_GIAO and pert_has_type(emPert,Magnetic) )
      addMagPert(emPert,CH);


  #ifdef _DEBUGGIAOONEE
      // prettyPrintSmart(std::cout,"Core H",CH[0],NB,NB,NB);
      for ( auto ii = 0 ; ii < CH.size() ; ii++ ) {
        std::cout<<"ii= "<<ii<<std::endl;
        prettyPrintSmart(std::cout,"Core H",CH[ii],NB,NB,NB);
      }
  #endif

  };  // void NRCoreH::computeCoreH(std::vector<MatsT*> &CH)

}
