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

#include <fockbuilder/rofock.hpp>
#include <util/time.hpp>
#include <cqlinalg.hpp>

namespace ChronusQ {

  /**   
   *  \brief Forms the Roothaan's effective Fock matrix from general fock
   *
   *  Reference: J. Phys. Chem. A, Vol. 114, No. 33, 2010;
   *             Mol. Phys. 1974, 28, 819–828.
   *
   */
  template <typename MatsT, typename IntsT>
  void ROFock<MatsT,IntsT>::rohfFock(SingleSlater<MatsT,IntsT> &ss) {

    typedef MatsT*                    oper_t;

    size_t NB = ss.aoints.basisSet().nBasis;
    size_t NB2 = NB*NB;

    ROOT_ONLY(ss.comm);

    //construct focka and fockb
    for(auto j = 0; j < NB2; j++) {
      ss.mo1[j] = 0.5 * (ss.fockMatrix[SCALAR][j] + ss.fockMatrix[MZ][j]);
      ss.mo2[j] = 0.5 * (ss.fockMatrix[SCALAR][j] - ss.fockMatrix[MZ][j]); 
    }

    //construct projectors for closed, open, virtual
    //pc = dmb * S
    //po = (dma - dmb) * S
    //pv = I - dma * S
    MatsT* pc  = ss.memManager.template malloc<MatsT>(NB*NB);
    MatsT* po  = ss.memManager.template malloc<MatsT>(NB*NB);
    MatsT* pv  = ss.memManager.template malloc<MatsT>(NB*NB);
    MatsT* tmp  = ss.memManager.template malloc<MatsT>(NB*NB);
    MatsT* tmp2 = ss.memManager.template malloc<MatsT>(NB*NB);
    MatsT* S = ss.memManager.template malloc<MatsT>(NB*NB);

    //overlap matrix
    std::transform(ss.aoints.overlap, ss.aoints.overlap+NB2, S,
                   [](MatsT a){return a;});
    for(auto j = 0; j < NB2; j++) tmp[j] = 0.5 * (ss.onePDM[0][j] - ss.onePDM[1][j]);
    Gemm('N','N',NB,NB,NB,MatsT(1.),tmp,NB,S,NB,MatsT(0.),pc,NB);
    for(auto j = 0; j < NB2; j++) tmp[j] = ss.onePDM[1][j];
    Gemm('N','N',NB,NB,NB,MatsT(1.),tmp,NB,S,NB,MatsT(0.),po,NB);
    for(auto j = 0; j < NB2; j++) tmp[j] = 0.5 * (ss.onePDM[0][j] + ss.onePDM[1][j]);
    Gemm('N','N',NB,NB,NB,MatsT(-1.),tmp,NB,S,NB,MatsT(0.),pv,NB);
    for(auto j = 0; j < NB; j++) pv[j*NB+j] = MatsT(1.) + pv[j*NB+j];
    /*
     * construct Roothaan's effective fock
        ======== ======== ====== =========
        space     closed   open   virtual
        ======== ======== ====== =========
        closed      Fc      Fb     Fc
        open        Fb      Fc     Fa
        virtual     Fc      Fa     Fc
        ======== ======== ====== =========
        where Fc = (Fa + Fb) / 2
     */
    Gemm('N','N',NB,NB,NB,MatsT(0.5),ss.fockMatrix[SCALAR],NB,pc,NB,MatsT(0.),tmp2,NB);
    Gemm('C','N',NB,NB,NB,MatsT(0.5),pc,NB,tmp2,NB,MatsT(0.),tmp,NB);
    Gemm('C','N',NB,NB,NB,MatsT(1.),pv,NB,tmp2,NB,MatsT(1.),tmp,NB);
    Gemm('N','N',NB,NB,NB,MatsT(0.5),ss.fockMatrix[SCALAR],NB,po,NB,MatsT(0.),tmp2,NB);
    Gemm('C','N',NB,NB,NB,MatsT(0.5),po,NB,tmp2,NB,MatsT(1.),tmp,NB);
    Gemm('N','N',NB,NB,NB,MatsT(0.5),ss.fockMatrix[SCALAR],NB,pv,NB,MatsT(0.),tmp2,NB);
    Gemm('C','N',NB,NB,NB,MatsT(0.5),pv,NB,tmp2,NB,MatsT(1.),tmp,NB);
    Gemm('N','N',NB,NB,NB,MatsT(1.),ss.mo2,NB,pc,NB,MatsT(0.),tmp2,NB);
    Gemm('C','N',NB,NB,NB,MatsT(1.),po,NB,tmp2,NB,MatsT(1.),tmp,NB);
    Gemm('N','N',NB,NB,NB,MatsT(1.),ss.mo1,NB,pv,NB,MatsT(0.),tmp2,NB);
    Gemm('C','N',NB,NB,NB,MatsT(1.),po,NB,tmp2,NB,MatsT(1.),tmp,NB);
    MatAdd('C','N',NB,NB,MatsT(1.),tmp,NB,MatsT(1.),tmp,NB,ss.fockMatrix[SCALAR],NB);

    ss.memManager.free(pc,po,pv,tmp,tmp2,S);
  }; // ROFock<MatsT, IntsT>::rohfFock

  /**
   *  \brief Forms the Roothaan's effective Fock matrix for a single slater 
   *  determinant using the 1PDM.
   *
   *  \param [in] increment Whether or not the Fock matrix is being
   *  incremented using a previous density
   *
   *  Populates / overwrites fock strorage in SingleSlater &ss
   */
  template <typename MatsT, typename IntsT>
  void ROFock<MatsT,IntsT>::formFock(SingleSlater<MatsT,IntsT> &ss,
    EMPerturbation &pert, bool increment, double xHFX) {

    size_t NB = ss.aoints.basisSet().nBasis;
    size_t NB2 = NB*NB;

    // General fock build
    FockBuilder<MatsT,IntsT>::formFock(ss, pert, increment, xHFX);

    // ROHF fock build
    rohfFock(ss);

  } // ROFock<MatsT,IntsT>::formFock


}; // namespace ChronusQ
