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

#include <singleslater/kohnsham.hpp>

#include <grid/integrator.hpp>
#include <basisset/basisset_util.hpp>
#include <cqlinalg/blasutil.hpp>
#include <cqlinalg/blasext.hpp>

#include <util/threads.hpp>

namespace ChronusQ {

  template <typename MatsT, typename IntsT>
  void KohnSham<MatsT,IntsT>::loadFXCder(size_t NPTS, double *Den, double *Gamma, double *EpsEval, double *VRhoEval, 
    double *V2RhoEval, double *VgammaEval, double *V2gammaEval, double *V2RhogammaEval, 
    double *EpsSCR, double *VRhoSCR, double *VgammaSCR, double *V2RhoEvalSCR, double *V2gammaEvalSCR,
    double *V2RhogammaEvalSCR) { 

    for(auto iF = 0; iF < functionals.size(); iF++) {

      double *ES = nullptr, *VR = nullptr, *VS = nullptr, *V2R = nullptr, 
        *V2S = nullptr, *V2RS = nullptr;

      if( iF == 0 ) {
        ES = EpsEval;
        VR = VRhoEval;
        VS = VgammaEval;

        V2R  = V2RhoEval;
        V2S  = V2gammaEval;
        V2RS = V2RhogammaEval;
      } else {
        ES = EpsSCR;
        VR = VRhoSCR;
        VS = VgammaSCR;

        V2R  = V2RhoEvalSCR;
        V2S  = V2gammaEvalSCR;
        V2RS = V2RhogammaEvalSCR;
      }

      if( functionals[iF]->isGGA() )
        functionals[iF]->evalEXC_VXC_FXC(NPTS,Den,Gamma,ES,VR,VS,V2R,V2RS,V2S);
      else
        functionals[iF]->evalEXC_VXC_FXC(NPTS,Den,ES,VR,V2R);

      if( std::any_of(V2R,V2R + 3*NPTS,[&](double x){ return std::isnan(x); }) ) std::cerr << "V2R NANS\n";
      if( functionals[iF]->isGGA() ) {
        if( std::any_of(V2S,V2S + 6*NPTS,[&](double x){ return std::isnan(x); }) ) std::cerr << "V2S NANS\n";
        if( std::any_of(V2RS,V2RS + 6*NPTS,[&](double x){ return std::isnan(x); }) ) std::cerr << "V2RS NANS\n";
      }
      
      if( iF != 0 ) {

        AXPY(NPTS  ,1.,ES ,1,EpsEval  ,1);
        AXPY(2*NPTS,1.,VR ,1,VRhoEval ,1);
        AXPY(3*NPTS,1.,V2R,1,V2RhoEval,1);

        if( functionals[iF]->isGGA() ) {
          AXPY(3*NPTS,1.,VS  ,1,VgammaEval    ,1);
          AXPY(6*NPTS,1.,V2S ,1,V2gammaEval   ,1);
          AXPY(6*NPTS,1.,V2RS,1,V2RhogammaEval,1);
        }

      }

    }

  }; // KohnSham<T>::loadFXCder







  template <typename MatsT, typename IntsT>
  template <typename U>
  void KohnSham<MatsT, IntsT>::constructZVarsFXC(DENSITY_TYPE denTyp, bool isGGA, size_t NPts, 
    double* GDenS, double* GDenZ, double* GDenY, double* GDenX,
    U* TS, U* TZ, U* TY, U* TX,
    U* GTS, U* GTZ, U* GTY, U* GTX,
    double *VR, double *VG, 
    double *V2R, double *V2G, double *V2RG, 
    U *ZrhoVar1, U *ZgammaVar1, U *ZgammaVar2, U *ZgammaVar3, U *ZgammaVar4){

    memset(ZrhoVar1,0,NPts);


    if( denTyp == SCALAR ) {

      for(auto iPt = 0ul; iPt < NPts; iPt++)
        ZrhoVar1[iPt] = TS[iPt] * ( V2R[3*iPt] + 2 * V2R[3*iPt + 1] + V2R[3*iPt + 2] ) +
                        TZ[iPt] * ( V2R[3*iPt] -     V2R[3*iPt + 2] );

      if( isGGA )
      for(auto iPt = 0ul; iPt < NPts; iPt++) {

        U gPTss = (GDenS[iPt]          * GTS[iPt]         ) +
                  (GDenS[iPt + NPts]   * GTS[iPt + NPts]  ) +
                  (GDenS[iPt + 2*NPts] * GTS[iPt + 2*NPts]);

        U gPTsz = (GDenS[iPt]          * GTZ[iPt]         ) +
                  (GDenS[iPt + NPts]   * GTZ[iPt + NPts]  ) +
                  (GDenS[iPt + 2*NPts] * GTZ[iPt + 2*NPts]);

        U gPTzz(0.);
        if( this->onePDM.size() > 1 ) {

          // Sum of both <SZ> and <ZS>
          gPTsz += (GDenZ[iPt]          * GTS[iPt]         ) +
                   (GDenZ[iPt + NPts]   * GTS[iPt + NPts]  ) +
                   (GDenZ[iPt + 2*NPts] * GTS[iPt + 2*NPts]);

          gPTzz = (GDenZ[iPt]          * GTZ[iPt]         ) +
                  (GDenZ[iPt + NPts]   * GTZ[iPt + NPts]  ) +
                  (GDenZ[iPt + 2*NPts] * GTZ[iPt + 2*NPts]);

        }


        ZrhoVar1[iPt] += ( V2RG[6*iPt    ] + V2RG[6*iPt + 1] + V2RG[6*iPt + 2] + 
                           V2RG[6*iPt + 3] + V2RG[6*iPt + 4] + V2RG[6*iPt + 5]   ) * gPTss +
                         ( V2RG[6*iPt    ] - V2RG[6*iPt + 2] + V2RG[6*iPt + 3] -
                           V2RG[6*iPt + 5]                                       ) * gPTsz +
                         ( V2RG[6*iPt    ] - V2RG[6*iPt + 1] + V2RG[6*iPt + 2] + 
                           V2RG[6*iPt + 3] - V2RG[6*iPt + 4] + V2RG[6*iPt + 5]   ) * gPTzz;



        ZgammaVar1[iPt] = 2 * ( VG[3*iPt] + VG[3*iPt + 1] + VG[3*iPt + 2] );
        ZgammaVar2[iPt] = 2 * ( VG[3*iPt] - VG[3*iPt + 2]                 );

        ZgammaVar3[iPt] = 
          ( V2G[6*iPt    ] + 2*V2G[6*iPt + 1] + 2*V2G[6*iPt + 2] + 
            V2G[6*iPt + 3] + 2*V2G[6*iPt + 4] +   V2G[6*iPt + 5]   ) * gPTss +
          ( V2G[6*iPt    ] +   V2G[6*iPt + 1] -   V2G[6*iPt + 4] -
            V2G[6*iPt + 5]                                         ) * gPTsz +
          ( V2G[6*iPt    ] + 2*V2G[6*iPt + 2] -   V2G[6*iPt + 3] +
            V2G[6*iPt + 5]                                         ) * gPTzz +

         ( V2RG[6*iPt    ] + V2RG[6*iPt + 1] + V2RG[6*iPt + 2] + 
           V2RG[6*iPt + 3] + V2RG[6*iPt + 4] + V2RG[6*iPt + 5]   ) * TS[iPt] +
         ( V2RG[6*iPt    ] + V2RG[6*iPt + 1] + V2RG[6*iPt + 2] - 
           V2RG[6*iPt + 3] - V2RG[6*iPt + 4] - V2RG[6*iPt + 5]   ) * TZ[iPt];

        if( this->onePDM.size() > 1 )
          ZgammaVar4[iPt] = 
            ( V2G[6*iPt    ] + 2*V2G[6*iPt + 1] -   V2G[6*iPt + 4] -
              V2G[6*iPt + 5]                                         ) * gPTss +
            ( V2G[6*iPt    ] - 2*V2G[6*iPt + 2] +   V2G[6*iPt + 5]   ) * gPTsz +
            ( V2G[6*iPt    ] -   V2G[6*iPt + 1] +   V2G[6*iPt + 4] -
              V2G[6*iPt + 5]                                         ) * gPTzz +
  
           ( V2RG[6*iPt    ] - V2RG[6*iPt + 2] + V2RG[6*iPt + 3] - V2RG[6*iPt + 5]   ) * TS[iPt] +
           ( V2RG[6*iPt    ] - V2RG[6*iPt + 2] - V2RG[6*iPt + 3] + V2RG[6*iPt + 5]   ) * TZ[iPt];

      }

    } else {

      for(auto iPt = 0ul; iPt < NPts; iPt++)
        ZrhoVar1[iPt] = TZ[iPt] * ( V2R[3*iPt] - 2 * V2R[3*iPt + 1] + V2R[3*iPt + 2] ) +
                        TS[iPt] * ( V2R[3*iPt] -     V2R[3*iPt + 2] );



      if( isGGA )
      for(auto iPt = 0ul; iPt < NPts; iPt++) {

        U gPTss = (GDenS[iPt]          * GTS[iPt]         ) +
                  (GDenS[iPt + NPts]   * GTS[iPt + NPts]  ) +
                  (GDenS[iPt + 2*NPts] * GTS[iPt + 2*NPts]);

        U gPTsz = (GDenS[iPt]          * GTZ[iPt]         ) +
                  (GDenS[iPt + NPts]   * GTZ[iPt + NPts]  ) +
                  (GDenS[iPt + 2*NPts] * GTZ[iPt + 2*NPts]);

        U gPTzz(0.);
        if( this->onePDM.size() > 1 ) {

          // Sum of both <SZ> and <ZS>
          gPTsz += (GDenZ[iPt]          * GTS[iPt]         ) +
                   (GDenZ[iPt + NPts]   * GTS[iPt + NPts]  ) +
                   (GDenZ[iPt + 2*NPts] * GTS[iPt + 2*NPts]);

          gPTzz = (GDenZ[iPt]          * GTZ[iPt]         ) +
                  (GDenZ[iPt + NPts]   * GTZ[iPt + NPts]  ) +
                  (GDenZ[iPt + 2*NPts] * GTZ[iPt + 2*NPts]);

        }



        ZrhoVar1[iPt] += ( V2RG[6*iPt    ] + V2RG[6*iPt + 1] + V2RG[6*iPt + 2] - 
                           V2RG[6*iPt + 3] - V2RG[6*iPt + 4] - V2RG[6*iPt + 5]   ) * gPTss +
                         ( V2RG[6*iPt    ] - V2RG[6*iPt + 2] - V2RG[6*iPt + 3] +
                           V2RG[6*iPt + 5]                                       ) * gPTsz + 
                         ( V2RG[6*iPt    ] - V2RG[6*iPt + 1] + V2RG[6*iPt + 2] - 
                           V2RG[6*iPt + 3] + V2RG[6*iPt + 4] - V2RG[6*iPt + 5]   ) * gPTzz;

        ZgammaVar1[iPt] = 2 * ( VG[3*iPt] - VG[3*iPt + 2]                 );
        ZgammaVar2[iPt] = 2 * ( VG[3*iPt] - VG[3*iPt + 1] + VG[3*iPt + 2] );

        ZgammaVar3[iPt] = 
          ( V2G[6*iPt    ] +   V2G[6*iPt + 1] - V2G[6*iPt + 4] - V2G[6*iPt + 5]   ) * gPTss +
          ( V2G[6*iPt    ] - 2*V2G[6*iPt + 2] + V2G[6*iPt + 5]                    ) * gPTsz +
          ( V2G[6*iPt    ] -   V2G[6*iPt + 1] + V2G[6*iPt + 4] - V2G[6*iPt + 5]   ) * gPTzz +

          ( V2RG[6*iPt    ] - V2RG[6*iPt + 2] + V2RG[6*iPt + 3] - V2RG[6*iPt + 5] ) * TS[iPt] +
          ( V2RG[6*iPt    ] - V2RG[6*iPt + 2] - V2RG[6*iPt + 3] + V2RG[6*iPt + 5] ) * TZ[iPt];

        if( this->onePDM.size() > 1 )
          ZgammaVar4[iPt] = 
            ( V2G[6*iPt    ] + 2*V2G[6*iPt + 2] -   V2G[6*iPt + 3] +
              V2G[6*iPt + 5]                                         ) * gPTss +
            ( V2G[6*iPt    ] -   V2G[6*iPt + 1] +   V2G[6*iPt + 4] -
              V2G[6*iPt + 5]                                         ) * gPTsz +
            ( V2G[6*iPt    ] - 2*V2G[6*iPt + 1] + 2*V2G[6*iPt + 2] + 
              V2G[6*iPt + 3] - 2*V2G[6*iPt + 4] +   V2G[6*iPt + 5]   ) * gPTzz +
  
           ( V2RG[6*iPt    ] - V2RG[6*iPt + 1] + V2RG[6*iPt + 2] + 
             V2RG[6*iPt + 3] - V2RG[6*iPt + 4] + V2RG[6*iPt + 5]   ) * TS[iPt] +
           ( V2RG[6*iPt    ] - V2RG[6*iPt + 1] + V2RG[6*iPt + 2] - 
             V2RG[6*iPt + 3] + V2RG[6*iPt + 4] - V2RG[6*iPt + 5]   ) * TZ[iPt];

      }

    }

  }




  template <typename MatsT, typename IntsT>
  template <typename U>
  void KohnSham<MatsT, IntsT>::formZ_fxc(DENSITY_TYPE denType, bool isGGA, size_t NPts, size_t NBE, size_t IOff,
    double epsScreen, std::vector<double> &weights,
    U *ZrhoVar1, U *ZgammaVar1, U *ZgammaVar2, U *ZgammaVar3, U *ZgammaVar4,
    double* GDenS, double* GDenZ, double* GDenY, double* GDenX, U* GTS, U* GTZ, U* GTY, U* GTX,
    double *BasisScr, U* ZMAT) {

    memset(ZMAT,0,IOff*sizeof(U));

    double* ZMAT_RE = reinterpret_cast<double*>(ZMAT);
    double* ZMAT_IM = ZMAT_RE + 1;
    int INCZMAT = sizeof(U)/sizeof(double);

    for(auto iPt = 0ul; iPt < NPts; iPt++) {

      U Fg = 0.5 * weights[iPt] * ZrhoVar1[iPt];

      double Fg_r = std::real(Fg);
      AXPY(NBE,Fg_r,BasisScr + iPt*NBE,1, ZMAT_RE + iPt*NBE*INCZMAT ,INCZMAT);

      if( std::is_same<U,dcomplex>::value ) {
        double Fg_i = std::imag(Fg);
        AXPY(NBE,Fg_i,BasisScr + iPt*NBE,1, ZMAT_IM + iPt*NBE*INCZMAT ,INCZMAT);
      }



      if( isGGA ) {

        U FgX = ZgammaVar1[iPt] * GTS[iPt         ] + ZgammaVar2[iPt] * GTZ[iPt         ] + ZgammaVar3[iPt] * GDenS[iPt         ];
        U FgY = ZgammaVar1[iPt] * GTS[iPt + NPts  ] + ZgammaVar2[iPt] * GTZ[iPt + NPts  ] + ZgammaVar3[iPt] * GDenS[iPt + NPts  ];
        U FgZ = ZgammaVar1[iPt] * GTS[iPt + 2*NPts] + ZgammaVar2[iPt] * GTZ[iPt + 2*NPts] + ZgammaVar3[iPt] * GDenS[iPt + 2*NPts];

        if( this->onePDM.size() > 1 ) {

          FgX += ZgammaVar4[iPt] * GDenZ[iPt         ];
          FgY += ZgammaVar4[iPt] * GDenZ[iPt + NPts  ];
          FgZ += ZgammaVar4[iPt] * GDenZ[iPt + 2*NPts];

        }


        FgX *= weights[iPt];
        FgY *= weights[iPt];
        FgZ *= weights[iPt];

        double FgX_r = std::real(FgX);
        double FgY_r = std::real(FgY);
        double FgZ_r = std::real(FgZ);

        AXPY(NBE,FgX_r,BasisScr + iPt*NBE +   IOff,1,ZMAT_RE + iPt*NBE*INCZMAT,INCZMAT);
        AXPY(NBE,FgY_r,BasisScr + iPt*NBE + 2*IOff,1,ZMAT_RE + iPt*NBE*INCZMAT,INCZMAT);
        AXPY(NBE,FgZ_r,BasisScr + iPt*NBE + 3*IOff,1,ZMAT_RE + iPt*NBE*INCZMAT,INCZMAT);

      }



    }

  }









  template <typename MatsT, typename IntsT>
  template <typename U>
  void KohnSham<MatsT, IntsT>::formFXC( MPI_Comm c,  
      std::vector<TwoBodyContraction<U>> &cList ) {


    size_t itOff = this->nC == 2 ? 5 : 3;
    size_t nVec = cList.size() / itOff;
    size_t NB     = this->aoints.basisSet().nBasis;
    size_t NB2    = NB*NB;
    size_t NPPB   = intParam.nRadPerBatch * intParam.nAng;
    size_t nAtoms = this->aoints.molecule().nAtoms;


    // Parallelism
    size_t NT = GetNumThreads();
    size_t LAThreads = GetLAThreads();
    size_t mpiRank = MPIRank(c);
    size_t mpiSize = MPISize(c);

    // Turn off LA threads
    SetLAThreads(1);

    // Split the MPI Comm
    int color = ((mpiSize < nAtoms) or 
                 (mpiRank < nAtoms)) ? 1 : MPI_UNDEFINED;
                                                            
                  
    MPI_Comm intComm = MPICommSplit(c,color,mpiRank);


    bool isGGA = std::any_of(functionals.begin(),functionals.end(),
                   [](std::shared_ptr<DFTFunctional> &x) {
                     return x->isGGA(); 
                   }); 

#ifdef CQ_ENABLE_MPI
    if( intComm != MPI_COMM_NULL ) {
#endif

    U* NBNBSCR = this->memManager.template malloc<U>(NB2 * NT);
    U* NBNPSCR = this->memManager.template malloc<U>(NB*NPPB * NT);


    std::vector<double*> Re1PDM;
    for(auto i = 0; i < this->onePDM.size(); i++) {
      if( std::is_same<MatsT,double>::value )
        Re1PDM.push_back(reinterpret_cast<double*>(this->onePDM[i]));
      else {
        Re1PDM.push_back(this->memManager.template malloc<double>(NB2));
        GetMatRE('N',NB,NB,1.,this->onePDM[i],NB,Re1PDM.back(),NB);
      }
    }



    std::vector<std::vector<double *>> ReTSymm, ImTSymm; 
    for(auto iVec = 0; iVec < nVec; iVec++) {
      ReTSymm.emplace_back();

      size_t indx = iVec * itOff;
      for(auto iS = 0; iS < 2*this->nC; iS++) {
        
         ReTSymm.back().emplace_back(
             this->memManager.template malloc<double>(NB2));
         MatAdd('N','C',NB,NB,U(0.5),cList[indx + iS + 1].X,NB,U(0.5),
             cList[indx + iS + 1].X,NB,NBNBSCR,NB);
         GetMatRE('N',NB,NB,1.,NBNBSCR,NB,ReTSymm.back().back(),NB);

      }

    }



    U* GxcT_raw = this->memManager.template malloc<U>(2*this->nC*nVec*NT*NB2);
    U* Gxc_first = GxcT_raw;
    std::vector<std::vector<std::vector<U*>>> GxcT;
    for(auto ithread = 0; ithread < NT; ithread++) {
      GxcT.emplace_back();
      for(auto iVec = 0; iVec < nVec; iVec++) { 
        GxcT.back().emplace_back();
        for(auto iS = 0; iS < 2*this->nC; iS++){
          GxcT.back().back().emplace_back(Gxc_first);
          memset(GxcT.back().back().back(),0,NB2*sizeof(U));
          Gxc_first += NB2;
        }
      }
    }






    // Allocation of V Vairables
    double *DenS(nullptr),  *DenZ(nullptr),  *DenY(nullptr),  *DenX(nullptr);
    double *GDenS(nullptr), *GDenZ(nullptr), *GDenY(nullptr), *GDenX(nullptr);

    // Density
    DenS = this->memManager.template malloc<double>(NPPB * NT);
    if( this->nC == 2 or not this->iCS )
      DenZ = this->memManager.template malloc<double>(NPPB * NT);
    if( this->nC == 2 ) {
      DenY = this->memManager.template malloc<double>(NPPB * NT);
      DenX = this->memManager.template malloc<double>(NPPB * NT);
    }


    // Density Gradient
    if( isGGA ) {

      GDenS = this->memManager.template malloc<double>(3*NPPB * NT);
      if( this->nC == 2 or not this->iCS )
        GDenZ = this->memManager.template malloc<double>(3*NPPB * NT);
      if( this->nC == 2 ) {
        GDenY = this->memManager.template malloc<double>(3*NPPB * NT);
        GDenX = this->memManager.template malloc<double>(3*NPPB * NT);
      }

    }

    // Allocation of T evaluation
    U *TS(nullptr),  *TZ(nullptr),  *TY(nullptr),  *TX(nullptr);
    U *GTS(nullptr), *GTZ(nullptr), *GTY(nullptr), *GTX(nullptr);

    // T
    TS = this->memManager.template malloc<U>(NPPB * NT);
    TZ = this->memManager.template malloc<U>(NPPB * NT);
    if( this->nC == 2 ) {
      TY = this->memManager.template malloc<U>(NPPB * NT);
      TX = this->memManager.template malloc<U>(NPPB * NT);
    }


    // T Gradient
    if( isGGA ) {

      GTS = this->memManager.template malloc<U>(3*NPPB * NT);
      GTZ = this->memManager.template malloc<U>(3*NPPB * NT);
      if( this->nC == 2 ) {
        GTY = this->memManager.template malloc<U>(3*NPPB * NT);
        GTX = this->memManager.template malloc<U>(3*NPPB * NT);
      }

    }





    // U Variables

    double * eps     = this->memManager.template malloc<double>(NPPB * NT);
    double * U_n     = this->memManager.template malloc<double>(2*NPPB * NT);
    double * U_gamma = isGGA ? 
      this->memManager.template malloc<double>(3*NPPB * NT) : nullptr;

    // U First Derivatives

    double * dVU_n     = this->memManager.template malloc<double>(2*NPPB * NT);
    double * dVU_gamma = isGGA ? 
      this->memManager.template malloc<double>(3*NPPB * NT) : nullptr;

    double * eps_SCR(nullptr), * dVU_n_SCR(nullptr), * dVU_gamma_SCR(nullptr);
    if( functionals.size() > 1 ) {

      eps_SCR       = this->memManager.template malloc<double>(NPPB * NT);
      dVU_n_SCR     = this->memManager.template malloc<double>(2*NPPB * NT);
      dVU_gamma_SCR = isGGA ? 
        this->memManager.template malloc<double>(3*NPPB * NT) : nullptr;

    }

    // U Second Derivatives

    double * d2VU_n = this->memManager.template malloc<double>(3*NPPB * NT);
    double * d2VU_gamma   = isGGA ? 
      this->memManager.template malloc<double>(6*NPPB * NT) : nullptr;
    double * d2VU_n_gamma = isGGA ? 
      this->memManager.template malloc<double>(6*NPPB * NT) : nullptr;

    double * d2VU_n_SCR(nullptr), * d2VU_gamma_SCR(nullptr), 
           * d2VU_n_gamma_SCR(nullptr); 

    if( functionals.size() > 1 ) {
      d2VU_n_SCR = this->memManager.template malloc<double>(3*NPPB * NT);
      d2VU_gamma_SCR   = isGGA ? 
        this->memManager.template malloc<double>(6*NPPB * NT) : nullptr;
      d2VU_n_gamma_SCR = isGGA ? 
        this->memManager.template malloc<double>(6*NPPB * NT) : nullptr;
    }


    // Z Vars
    U * ZrhoVar = this->memManager.template malloc<U>(NPPB * NT);

    U * ZgammaVar1 = isGGA ? 
      this->memManager.template malloc<U>(NPPB * NT) : nullptr;
    U * ZgammaVar2 = isGGA ? 
      this->memManager.template malloc<U>(NPPB * NT) : nullptr;
    U * ZgammaVar3 = isGGA ? 
      this->memManager.template malloc<U>(NPPB * NT) : nullptr;
    U * ZgammaVar4 = isGGA ? 
      this->memManager.template malloc<U>(NPPB * NT) : nullptr;

    U* ZMAT = this->memManager.template malloc<U>(NB*NPPB * NT);



    double intDen = 0.;


    auto fxcbuild = [&](size_t &res, std::vector<cart_t> &batch, 
      std::vector<double> &weights, size_t NBE, double *BasisEval, 
      std::vector<size_t> &batchEvalShells, 
      std::vector<std::pair<size_t,size_t>> &subMatCut) {

      double epsScreen = intParam.epsilon / nAtoms /
        intParam.nAng / intParam.nRad;

      epsScreen = std::max(epsScreen,
                           std::numeric_limits<double>::epsilon());



      size_t NPts = batch.size();
      size_t IOff = NBE*NPts;
      int tid = GetThreadID();
      size_t nPtOff = tid * NPPB;


      // Get thread Local storage
      

      // Density and T
      double *DenS_loc = DenS + nPtOff;
      double *DenZ_loc = DenZ + nPtOff;
      double *DenY_loc = DenY + nPtOff;
      double *DenX_loc = DenX + nPtOff;

      double *GDenS_loc = GDenS + 3 * nPtOff;
      double *GDenZ_loc = GDenZ + 3 * nPtOff;
      double *GDenY_loc = GDenY + 3 * nPtOff;
      double *GDenX_loc = GDenX + 3 * nPtOff;

      U *TS_loc = TS + nPtOff;
      U *TZ_loc = TZ + nPtOff;
      U *TY_loc = TY + nPtOff;
      U *TX_loc = TX + nPtOff;

      U *GTS_loc = GTS + 3 * nPtOff;
      U *GTZ_loc = GTZ + 3 * nPtOff;
      U *GTY_loc = GTY + 3 * nPtOff;
      U *GTX_loc = GTX + 3 * nPtOff;


      // U Vars

      double *U_n_loc        = U_n        + 2 * nPtOff;     
      double *U_gamma_loc    = U_gamma    + 3 * nPtOff;

      double *eps_loc        = eps        +     nPtOff;
      double *dVU_n_loc      = dVU_n      + 2 * nPtOff;     
      double *dVU_gamma_loc  = dVU_gamma  + 3 * nPtOff;

      double *d2VU_n_loc       = d2VU_n       + 3 * nPtOff;     
      double *d2VU_gamma_loc   = d2VU_gamma   + 6 * nPtOff;
      double *d2VU_n_gamma_loc = d2VU_n_gamma + 6 * nPtOff;

      double *eps_SCR_loc       = eps_SCR       +     nPtOff;
      double *dVU_n_SCR_loc     = dVU_n_SCR     + 2 * nPtOff;     
      double *dVU_gamma_SCR_loc = dVU_gamma_SCR + 3 * nPtOff;

      double *d2VU_n_SCR_loc       = d2VU_n_SCR       + 3 * nPtOff;     
      double *d2VU_gamma_SCR_loc   = d2VU_gamma_SCR   + 6 * nPtOff;
      double *d2VU_n_gamma_SCR_loc = d2VU_n_gamma_SCR + 6 * nPtOff;


      // Z Vars

      U* ZrhoVar_loc    = ZrhoVar    + nPtOff;
      U* ZgammaVar1_loc = ZgammaVar1 + nPtOff;
      U* ZgammaVar2_loc = ZgammaVar2 + nPtOff;
      U* ZgammaVar3_loc = ZgammaVar3 + nPtOff;
      U* ZgammaVar4_loc = ZgammaVar4 + nPtOff;

      U* ZMAT_loc = ZMAT + nPtOff*NB;


      U* NBNBSCR_loc = NBNBSCR + NB * NB   * tid;
      U* NBNPSCR_loc = NBNPSCR + NB * NPPB * tid;

      double * NBNBSCR_r = reinterpret_cast<double*>(NBNBSCR_loc);
      double * NBNPSCR_r = reinterpret_cast<double*>(NBNPSCR_loc);




      // This evaluates the V variables for all components 
      // (Scalar, MZ (UKS) and Mx, MY (2 Comp))
      evalDen((isGGA ? GRADIENT : NOGRAD), NPts, NBE, NB, subMatCut, 
        NBNBSCR_r, NBNPSCR_r, Re1PDM[SCALAR], DenS_loc , 
        GDenS_loc, GDenS_loc + NPts, GDenS_loc + 2*NPts, BasisEval);

      if( this->onePDM.size() > 1 )
        evalDen((isGGA ? GRADIENT : NOGRAD), NPts, NBE, NB, subMatCut, 
          NBNBSCR_r ,NBNPSCR_r, Re1PDM[MZ], DenZ_loc, 
          GDenZ_loc, GDenZ_loc + NPts, GDenZ_loc + 2*NPts, BasisEval);

      if( this->onePDM.size() > 2 ) {
        evalDen((isGGA ? GRADIENT : NOGRAD), NPts, NBE, NB, subMatCut, 
          NBNBSCR_r ,NBNPSCR_r, Re1PDM[MY], DenY_loc, 
          GDenY_loc, GDenY_loc + NPts, GDenY_loc + 2*NPts, BasisEval);
        evalDen((isGGA ? GRADIENT : NOGRAD), NPts, NBE, NB, subMatCut, 
          NBNBSCR_r ,NBNPSCR_r, Re1PDM[MX], DenX_loc, 
          GDenX_loc, GDenX_loc + NPts, GDenX_loc + 2*NPts, BasisEval);
      
      }

      // V -> U variables for evaluating the kernel derivatives.
      mkAuxVar(isGGA,epsScreen,NPts,
        DenS_loc,DenZ_loc,DenY_loc,DenX_loc,
        GDenS_loc,GDenS_loc + NPts,GDenS_loc + 2*NPts,
        GDenZ_loc,GDenZ_loc + NPts,GDenZ_loc + 2*NPts,
        GDenY_loc,GDenY_loc + NPts,GDenY_loc + 2*NPts,
        GDenX_loc,GDenX_loc + NPts,GDenX_loc + 2*NPts,
        nullptr, 
        nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr,
        nullptr,U_n_loc,U_gamma_loc
      );



      loadFXCder(NPts,U_n_loc,U_gamma_loc,eps_loc,dVU_n_loc,d2VU_n_loc,
        dVU_gamma_loc, d2VU_gamma_loc,d2VU_n_gamma_loc,eps_SCR_loc,
        dVU_n_SCR_loc,dVU_gamma_SCR_loc, d2VU_n_SCR_loc,d2VU_gamma_SCR_loc,
        d2VU_n_gamma_SCR_loc);


      for(auto iT = 0; iT < nVec; iT++) {

        if( std::is_same<U,dcomplex>::value ) CErr("NO COMPLEX YET!");

        size_t indx       = itOff * iT;
        double *TS_d      = reinterpret_cast<double*>(TS_loc);
        double *GTS_d     = reinterpret_cast<double*>(GTS_loc);
        double *TZ_d      = reinterpret_cast<double*>(TZ_loc);
        double *GTZ_d     = reinterpret_cast<double*>(GTZ_loc);

        // This evaluates the V variables for all components 
        // (Scalar, MZ (UKS) and Mx, MY (2 Comp))
        evalDen((isGGA ? GRADIENT : NOGRAD), NPts, NBE, NB, subMatCut, 
          NBNBSCR_r, NBNPSCR_r, ReTSymm[iT][SCALAR] , TS_d, 
          GTS_d, GTS_d + NPts, GTS_d + 2*NPts, BasisEval);

        evalDen((isGGA ? GRADIENT : NOGRAD), NPts, NBE, NB, subMatCut, 
          NBNBSCR_r ,NBNPSCR_r, ReTSymm[iT][MZ] , TZ_d, 
          GTZ_d, GTZ_d + NPts, GTZ_d + 2*NPts, BasisEval);

        if( this->onePDM.size() > 2 ) {

          double *TY_d      = reinterpret_cast<double*>(TY_loc);
          double *GTY_d     = reinterpret_cast<double*>(GTY_loc);
          double *TX_d      = reinterpret_cast<double*>(TX_loc);
          double *GTX_d     = reinterpret_cast<double*>(GTX_loc);

          evalDen((isGGA ? GRADIENT : NOGRAD), NPts, NBE, NB, subMatCut, 
            NBNBSCR_r ,NBNPSCR_r, ReTSymm[iT][MY], TY_d, 
            GTY_d, GTY_d + NPts, GTY_d + 2*NPts, BasisEval);

          evalDen((isGGA ? GRADIENT : NOGRAD), NPts, NBE, NB, subMatCut, 
            NBNBSCR_r ,NBNPSCR_r, ReTSymm[iT][MX], TX_d, 
            GTX_d, GTX_d + NPts, GTX_d + 2*NPts, BasisEval);

        }


        constructZVarsFXC(SCALAR,isGGA,NPts, GDenS_loc, GDenZ_loc, GDenY_loc, 
          GDenX_loc, TS_loc, TZ_loc, TY_loc, TX_loc, GTS_loc, GTZ_loc, GTY_loc, 
          GTX_loc, dVU_n_loc, dVU_gamma_loc, d2VU_n_loc, d2VU_gamma_loc, 
          d2VU_n_gamma_loc, ZrhoVar_loc, ZgammaVar1_loc, ZgammaVar2_loc,
          ZgammaVar3_loc, ZgammaVar4_loc);

        formZ_fxc(SCALAR,isGGA,NPts,NBE,IOff,epsScreen,weights,ZrhoVar_loc,
          ZgammaVar1_loc, ZgammaVar2_loc, ZgammaVar3_loc, ZgammaVar4_loc, 
          GDenS_loc, GDenZ_loc, GDenY_loc, GDenX_loc, GTS_loc, GTZ_loc, 
          GTY_loc, GTX_loc, BasisEval, ZMAT_loc);

        double *ZMAT_r = reinterpret_cast<double*>(ZMAT_loc);

        DSYR2K('L','N',NBE,NPts,0.5,BasisEval,NBE,ZMAT_r,NBE,0.,NBNBSCR_r,NBE);

        IncBySubMat(NB,NB,NBE,NBE,GxcT[tid][iT][SCALAR],NB,NBNBSCR_loc,NBE,
            subMatCut);






        constructZVarsFXC(MZ,isGGA,NPts, GDenS_loc, GDenZ_loc, GDenY_loc, 
          GDenX_loc, TS_loc, TZ_loc, TY_loc, TX_loc, GTS_loc, GTZ_loc, GTY_loc, 
          GTX_loc, dVU_n_loc, dVU_gamma_loc, d2VU_n_loc, d2VU_gamma_loc, 
          d2VU_n_gamma_loc, ZrhoVar_loc, ZgammaVar1_loc, ZgammaVar2_loc,
          ZgammaVar3_loc, ZgammaVar4_loc);

        formZ_fxc(MZ,isGGA,NPts,NBE,IOff,epsScreen,weights,ZrhoVar_loc,
          ZgammaVar1_loc, ZgammaVar2_loc, ZgammaVar3_loc, ZgammaVar4_loc, 
          GDenS_loc, GDenZ_loc, GDenY_loc, GDenX_loc, GTS_loc, GTZ_loc, 
          GTY_loc, GTX_loc, BasisEval, ZMAT_loc);

        DSYR2K('L','N',NBE,NPts,0.5,BasisEval,NBE,ZMAT_r,NBE,0.,NBNBSCR_r,NBE);

        IncBySubMat(NB,NB,NBE,NBE,GxcT[tid][iT][MZ],NB,NBNBSCR_loc,NBE,
            subMatCut);


      } // iT loop

    }; // VXC

    // Create the BeckeIntegrator object
    BeckeIntegrator<EulerMac> 
      integrator(intComm,this->memManager,this->aoints.molecule(),
        this->aoints.basisSet(), EulerMac(intParam.nRad), intParam.nAng, 
        intParam.nRadPerBatch, (isGGA ? GRADIENT : NOGRAD), intParam.epsilon);

    // Integrate the FXC
    integrator.integrate<size_t>(fxcbuild);

    //std::cerr << "NELEC " << 4.*M_PI*intDen << "\n";

    /*
    for(auto iT = 0; iT < nVec; iT++) {

      // Add -Gxc[T] into K[T]
      for(auto iS = 0; iS < 2*this->nC; iS++) {

        for(auto ithread = 0; ithread < NT; ithread++) {
          HerMat('L',NB,GxcT[ithread][iT][iS],NB);

          U fact = (ithread == 0)  ? functionals.back()->xHFX : 1.;
          MatAdd('N','N',NB,NB,fact,cList[iT*itOff + iS + 1].AX,NB,
            U(-4. * M_PI), GxcT[ithread][iT][iS],NB,
            cList[iT*itOff + iS + 1].AX,NB);
        }


      }

    }
    */

    U* mpiScr = nullptr;
#ifdef CQ_ENABLE_MPI
    if( MPIRank(intComm) == 0 and MPISize(intComm) > 1 )
      mpiScr = this->memManager.template malloc<U>(NB*NB);
#endif

    for(auto iT = 0; iT < nVec; iT++) {

      for(auto iS = 0; iS < 2*this->nC; iS++) {

        // Add -Gxc[T] thread contributions into single storage
        for(auto ithread = 0; ithread < NT; ithread++) {
          HerMat('L',NB,GxcT[ithread][iT][iS],NB);

          U fact = (ithread == 0)  ? 0 : 1.;
          MatAdd('N','N',NB,NB,
            fact         , GxcT[0][iT][iS]      ,NB,
            U(-4. * M_PI), GxcT[ithread][iT][iS],NB,
            GxcT[0][iT][iS],NB);
        }

#ifdef CQ_ENABLE_MPI
        // Add MPI Contributions together
        if( MPISize(intComm) > 1 )
          mxx::reduce(GxcT[0][iT][iS],NB2,mpiScr,0,std::plus<U>(),intComm);

        if( MPIRank(intComm) == 0 and MPISize(intComm) > 1 )
          std::copy_n(mpiScr,NB2,GxcT[0][iT][iS]);
#endif

        // Add GxcT to K
        if( MPIRank(intComm) == 0 )
          MatAdd('N','N',NB,NB,
            U(functionals.back()->xHFX), cList[iT*itOff + iS + 1].AX, NB,
            U(1.),                       GxcT[0][iT][iS],             NB,
            cList[iT*itOff + iS + 1].AX, NB);

      }

    }


    // Free up the memory
    if( mpiScr ) this->memManager.free(mpiScr);
    this->memManager.free( GxcT_raw, NBNBSCR, NBNPSCR );

    for(auto &Y : ReTSymm) for(auto &X : Y) this->memManager.free(X);
    if( not std::is_same<MatsT,double>::value )
      for(auto &X : Re1PDM) this->memManager.free(X);

    if( DenS ) this->memManager.free( DenS );
    if( DenZ ) this->memManager.free( DenZ );
    if( DenY ) this->memManager.free( DenY );
    if( DenX ) this->memManager.free( DenX );

    if( GDenS ) this->memManager.free( GDenS );
    if( GDenZ ) this->memManager.free( GDenZ );
    if( GDenY ) this->memManager.free( GDenY );
    if( GDenX ) this->memManager.free( GDenX );

    if( TS ) this->memManager.free( TS );
    if( TZ ) this->memManager.free( TZ );
    if( TY ) this->memManager.free( TY );
    if( TX ) this->memManager.free( TX );

    if( GTS ) this->memManager.free( GTS );
    if( GTZ ) this->memManager.free( GTZ );
    if( GTY ) this->memManager.free( GTY );
    if( GTX ) this->memManager.free( GTX );

    this->memManager.free( eps, U_n, dVU_n, d2VU_n );
    if( U_gamma )      this->memManager.free( U_gamma );
    if( dVU_gamma )    this->memManager.free( dVU_gamma );
    if( d2VU_gamma )   this->memManager.free( d2VU_gamma );
    if( d2VU_n_gamma ) this->memManager.free( d2VU_n_gamma );


    if( eps_SCR )       this->memManager.free( eps_SCR );
    if( dVU_n_SCR )     this->memManager.free( dVU_n_SCR );
    if( dVU_gamma_SCR ) this->memManager.free( dVU_gamma_SCR );
    if( d2VU_n_SCR )     this->memManager.free( d2VU_n_SCR );
    if( d2VU_gamma_SCR ) this->memManager.free( d2VU_gamma_SCR );
    if( d2VU_n_gamma_SCR ) this->memManager.free( d2VU_n_gamma_SCR );

    this->memManager.free( ZrhoVar, ZMAT );
    if( isGGA ) {
      this->memManager.free( ZgammaVar1, ZgammaVar2, ZgammaVar3, ZgammaVar4 );
    }

#ifdef CQ_ENABLE_MPI
    MPICommFree(intComm); // Free communicator

    } // End of the MPI

    MPI_Barrier(c);
#endif

    // Turn back on LA threads
    SetLAThreads(LAThreads);


  }



}; // namespace ChronusQ

