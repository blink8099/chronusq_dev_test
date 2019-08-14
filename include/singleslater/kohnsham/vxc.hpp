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
#ifndef __INCLUDED_SINGLESLATER_KOHNSHAM_VXC_HPP__
#define __INCLUDED_SINGLESLATER_KOHNSHAM_VXC_HPP__

#include <singleslater/kohnsham.hpp>

#include <grid/integrator.hpp>
#include <basisset/basisset_util.hpp>
#include <cqlinalg/blasutil.hpp>
#include <cqlinalg/blasext.hpp>

#include <util/threads.hpp>

// VXC_DEBUG_LEVEL == 1 - Timing
// VXC_DEBUG_LEVEL == 2 - VXC/rho/gamma + Timing
// VXC_DEBUG_LEVEL == 3 - Debug 2 + Overlap + no screening
// VXC_DEBUG_LEVEL  > 3 - Debug 3 + print everthing
#ifndef VXC_DEBUG_LEVEL
#  define VXC_DEBUG_LEVEL 0
#endif

namespace ChronusQ {

  /**
   *  \brief wrapper for evaluating and loading the DFT 
   *  kernel derivatives from libxc wrt to the U var.
   *
   *  Note. We compute all the quantities for all points in the batch.
   *
   *  \param [in]  NPts       Number of points in the batch
   *  \param [in]  Den        Pointer to the Uvar Density vector[+,-].
   *  \param [in]  Gamma      Pointer to the GGA Uvar vector[++,+-,--]. 
   *  \param [out] epsEval    Pointer to the energy per unit particle. 
   *
   *  \param [out] VRhoEval   Pointer to the first part der of 
   *                          the energy per unit volume in terms of 
   *                          the dens[+,-]. 
   *
   *  \param [out] VgammaEval Pointer to the first part der of 
   *                          the energy per unit volume in terms of 
   *                          the gammas[++,+-,--].
   *
   *  \param [out] EpsSCR     Pointer for multiple functional eval. See epsEval
   *  \param [out] VRhoSCR    Pointer for multiple functional eval. See VRhoEval
   *  \param [out] VgammaSCR  Pointer for multiple functional eval. See VgammaEval
   */  
  template <typename MatsT, typename IntsT>
  void KohnSham<MatsT,IntsT>::loadVXCder(size_t NPts, double *Den, double *Gamma,
    double *epsEval, double*VRhoEval, double *VgammaEval, double *EpsSCR, 
    double *VRhoSCR, double *VgammaSCR) { 

    for(auto iF = 0; iF < functionals.size(); iF++) {
      double *ES,*VR,*VS;
      if( iF == 0 ) {
        ES = epsEval;
        VR = VRhoEval;
        VS = VgammaEval;
      } else {
        ES = EpsSCR;
        VR = VRhoSCR;
        VS = VgammaSCR;
      }

      if( functionals[iF]->isGGA() )
        functionals[iF]->evalEXC_VXC(NPts,Den,Gamma,ES,VR,VS);
      else
        functionals[iF]->evalEXC_VXC(NPts,Den,ES,VR);
      
      if( iF != 0 ) {
        MatAdd('N','N',NPts,1,1.,epsEval,NPts,1.,EpsSCR,NPts,epsEval,NPts);
        MatAdd('N','N',2*NPts,1,1.,VRhoEval,2*NPts,1.,VRhoSCR,2*NPts,VRhoEval,2*NPts);
      if( functionals[iF]->isGGA() )
        MatAdd('N','N',3*NPts,1,1.,VgammaEval,3*NPts,1.,VgammaSCR,3*NPts,VgammaEval,3*NPts);
      }

    }

#if VXC_DEBUG_LEVEL > 3
    bool printGGA = false;
    for(auto iF = 0; iF < functionals.size(); iF++) {
       if( functionals[iF]->isGGA() ) printGGA = true;
    }
    prettyPrintSmart(std::cerr,"Den+   ",Den,NPts,1,2*NPts,2);
    prettyPrintSmart(std::cerr,"Den-   ",Den+1,NPts,1,2*NPts,2);
    if ( printGGA ) {
      prettyPrintSmart(std::cerr,"gamma++   ",Gamma,NPts,1,NPts,3);
      prettyPrintSmart(std::cerr,"gamma+-   ",Gamma+1,NPts,1,NPts,3);
      prettyPrintSmart(std::cerr,"gamma--   ",Gamma+2,NPts,1,NPts,3);
    }
    prettyPrintSmart(std::cerr,"epsEval   ",epsEval,NPts,1,NPts);
    prettyPrintSmart(std::cerr,"VRhoEval   ",VRhoEval,NPts,2,NPts);
    if ( printGGA ) 
      prettyPrintSmart(std::cerr,"VgammaEval   ",VgammaEval,NPts,3,NPts);
#endif

  }; // KohnSham<T,MatsT,IntsT>::loadVXCder




  /**
   *  \brief form the U variables given the V variables.
   *
   *  \param [in]  isGGA                Whether to include GGA contributions
   *  \param [in]  epsScreen            Screening tolerance
   *  \param [in]  Scalar               Scalar
   *  \param [in]  dndX, dndY, dndZ     Gradient components of n scalar
   *  \param [in]  Mz, My, Mx           Magnetization components
   *  \param [in]  dMkdX, dMkdY, dMkdZ  Gradient components of Mk component of the magnetization
   *  \param [out] ncoll                U variable for density (+,-) for NPts
   *  \param [out] gammaColl            U variable for Gdensity (++,+-,--) for NPts
   *
   */  
  template <typename MatsT, typename IntsT>
  void KohnSham<MatsT,IntsT>::mkAuxVar(bool isGGA, 
    double epsScreen, size_t NPts,
    double *Scalar, double *Mz, double *My, double *Mx,
    double *dndX, double *dndY, double *dndZ, 
    double *dMzdX, double *dMzdY, double *dMzdZ, 
    double *dMydX, double *dMydY, double *dMydZ, 
    double *dMxdX, double *dMxdY, double *dMxdZ, 
    double *Mnorm, double *Kx, double *Ky, double *Kz, 
    double *Hx, double *Hy, double *Hz,
    bool *Msmall, double *nColl, double *gammaColl){

#if VXC_DEBUG_LEVEL > 3
    prettyPrintSmart(std::cerr,"Scalar Den   ",Scalar,NPts,1, NPts);
    if( this->onePDM.size() > 1 ) 
      prettyPrintSmart(std::cerr,"Mz     Den   ",Mz,NPts,1, NPts);
    if( this->onePDM.size() > 2 ) { 
      prettyPrintSmart(std::cerr,"My     Den   ",My,NPts,1, NPts);
      prettyPrintSmart(std::cerr,"Mx     Den   ",Mx,NPts,1, NPts);
    }
    if ( isGGA ) {
      prettyPrintSmart(std::cerr,"Scalar DenX   ",dndX,NPts,1, NPts);
      prettyPrintSmart(std::cerr,"Scalar DenY   ",dndY,NPts,1, NPts);
      prettyPrintSmart(std::cerr,"Scalar DenZ   ",dndZ,NPts,1, NPts);
      if( this->onePDM.size() > 1 ) { 
        prettyPrintSmart(std::cerr,"Mz DenX   ",dMzdX,NPts,1, NPts);
        prettyPrintSmart(std::cerr,"Mz DenY   ",dMzdY,NPts,1, NPts);
        prettyPrintSmart(std::cerr,"Mz DenZ   ",dMzdZ,NPts,1, NPts);
      }
      if( this->onePDM.size() > 2 ) { 
        prettyPrintSmart(std::cerr,"My DenX   ",dMydX,NPts,1, NPts);
        prettyPrintSmart(std::cerr,"My DenY   ",dMydY,NPts,1, NPts);
        prettyPrintSmart(std::cerr,"My DenZ   ",dMydZ,NPts,1, NPts);
        prettyPrintSmart(std::cerr,"Mx DenX   ",dMxdX,NPts,1, NPts);
        prettyPrintSmart(std::cerr,"Mx DenY   ",dMxdY,NPts,1, NPts);
        prettyPrintSmart(std::cerr,"Mx DenZ   ",dMxdZ,NPts,1, NPts);
      }
    }
#endif

    double tmp = 0.;
    double *uPlus  = nColl;
    double *uMinus = nColl + 1;
    memset(nColl,0,2*NPts*sizeof(double));


    // LDA contributions
    // U(+) = 0.5 * (SCALAR + MZ)
    // U(-) = 0.5 * (SCALAR - MZ)
    // 2C See J. Chem. Theory Comput. 2017, 13, 2591-2603  
    // U(+) = 0.5 * (SCALAR + |M| )
    // U(-) = 0.5 * (SCALAR - |M| )
         
    AXPY(NPts,0.5,Scalar,1,uPlus,2);
    AXPY(NPts,0.5,Scalar,1,uMinus,2);

    bool skipMz = false;
    if( this->onePDM.size() == 2 ) {
    // UKS
#if VXC_DEBUG_LEVEL < 3
      double MaxDenZ = *std::max_element(Mz,Mz+NPts);
      if (MaxDenZ < epsScreen) skipMz = true;
#endif
        if(not skipMz){
          AXPY(NPts,0.5,Mz,1,uPlus,2);
          AXPY(NPts,-0.5,Mz,1,uMinus,2);
        }
#if VXC_DEBUG_LEVEL >= 3
      if(skipMz) std::cerr << "Skypped Mz " << std::endl;
#endif
    }  else if ( this->onePDM.size() > 2) {
      std::fill_n(Msmall,NPts,0.);
    // 2C
    // 2C See J. Chem. Theory Comput. 2017, 13, 2591-2603  
     //Compute and store Mtot
      for(auto iPt = 0; iPt < NPts; iPt++) {
        tmp =  Mx[iPt] * Mx[iPt];
        tmp += My[iPt] * My[iPt];
        tmp += Mz[iPt] * Mz[iPt];
        if (tmp > 1.e-24) {

          Mnorm[iPt] = std::sqrt(tmp);
          Kx[iPt]    = Mx[iPt] / Mnorm[iPt];
          Ky[iPt]    = My[iPt] / Mnorm[iPt];
          Kz[iPt]    = Mz[iPt] / Mnorm[iPt];

        } else {

          Msmall[iPt] = true; 
          Mnorm[iPt]  = (1./3.) * (Mx[iPt] + My[iPt] + Mz[iPt]);
          Kx[iPt]    = 1. / 3.;
          Ky[iPt]    = 1. / 3.;
          Kz[iPt]    = 1. / 3.;
        }

      }
          AXPY(NPts,0.5,Mnorm,1,uPlus,2);
          AXPY(NPts,-0.5,Mnorm,1,uMinus,2);

    }

    // GGA Contributions
    // GAMMA(++) = 0.25 * (GSCALAR.GSCALAR + GMZ.GMZ) + 0.5 * GSCALAR.GMZ
    // GAMMA(+-) = 0.25 * (GSCALAR.GSCALAR - GMZ.GMZ) 
    // GAMMA(--) = 0.25 * (GSCALAR.GSCALAR + GMZ.GMZ) - 0.5 * GSCALAR.GMZ
    //2C
    // 2C See J. Chem. Theory Comput. 2017, 13, 2591-2603  
    // GAMMA(++) = 0.25 * (GSCALAR.GSCALAR + SUM_K GMK.GMK) + 0.5 * SING * SQRT(SUM_K (GSCALAR.GMK)^2)
    // GAMMA(+-) = 0.25 * (GSCALAR.GSCALAR - SUM_K (GMZ.GMK) ) 
    // GAMMA(--) = 0.25 * (GSCALAR.GSCALAR + SUM_K GMK.GMK) - 0.5 * SIGN * SQRT(SUM_K (GSCALAR.GMK)^2)

    if(isGGA) {
      // RKS part
      for(auto iPt = 0; iPt < NPts; iPt++) {
        gammaColl[3*iPt] = 0.25 *  (dndX[iPt]*dndX[iPt] + dndY[iPt]*dndY[iPt] + dndZ[iPt]*dndZ[iPt]);
        gammaColl[3*iPt+1] = gammaColl[3*iPt]; 
        gammaColl[3*iPt+2] = gammaColl[3*iPt];
      }

      if( this->onePDM.size() == 2 ) {
      // UKS
        for(auto iPt = 0; iPt < NPts; iPt++) {
          if( not skipMz ) {
            double inner  = 0.25 * 
              (dMzdX[iPt]*dMzdX[iPt] + dMzdY[iPt]*dMzdY[iPt] + dMzdZ[iPt]*dMzdZ[iPt]);
            double inner2 = 0.5  * 
              (dMzdX[iPt]*dndX[iPt] + dMzdY[iPt]*dndY[iPt] + dMzdZ[iPt]*dndZ[iPt]);
      
            gammaColl[3*iPt]   += inner;
            gammaColl[3*iPt+1] -= inner;
            gammaColl[3*iPt+2] += inner;
      
            gammaColl[3*iPt]   += inner2;
            gammaColl[3*iPt+2] -= inner2;
          }
        } // loop pts

      }  else if ( this->onePDM.size() > 2) {
      // 2C See J. Chem. Theory Comput. 2017, 13, 2591-2603  
        double tmpnMx, tmpnMy, tmpnMz;
        double tmpSign, inner, inner2;

        for(auto iPt = 0; iPt < NPts; iPt++) {
          double signMD = 1. ;
          tmpnMx       = dMxdX[iPt]*dndX[iPt];
          tmpnMx      += dMxdY[iPt]*dndY[iPt];
          tmpnMx      += dMxdZ[iPt]*dndZ[iPt];
          
          tmpnMy       = dMydX[iPt]*dndX[iPt];
          tmpnMy      += dMydY[iPt]*dndY[iPt];
          tmpnMy      += dMydZ[iPt]*dndZ[iPt];
          
          tmpnMz       = dMzdX[iPt]*dndX[iPt];
          tmpnMz      += dMzdY[iPt]*dndY[iPt];
          tmpnMz      += dMzdZ[iPt]*dndZ[iPt];
          
          tmpSign   = tmpnMx * Mx[iPt];
          tmpSign  += tmpnMy * My[iPt];
          tmpSign  += tmpnMz * Mz[iPt];
          if ( std::signbit(tmpSign) ) signMD = -1.;
          //std::cerr <<"Sig " << tmpSign << " " << std::signbit(tmpSign) << " " << signMD <<std::endl;
          inner  = 
               (dMxdX[iPt]*dMxdX[iPt] + dMxdY[iPt]*dMxdY[iPt] + dMxdZ[iPt]*dMxdZ[iPt]);
          inner += 
               (dMydX[iPt]*dMydX[iPt] + dMydY[iPt]*dMydY[iPt] + dMydZ[iPt]*dMydZ[iPt]);
          inner += 
               (dMzdX[iPt]*dMzdX[iPt] + dMzdY[iPt]*dMzdY[iPt] + dMzdZ[iPt]*dMzdZ[iPt]);
          inner *= 0.25;
          
          double DSDMnorm = 0.0;
          if (Msmall[iPt]) {
            DSDMnorm  = (1./3.) * (tmpnMx + tmpnMy +tmpnMz);
            //DSDMnorm[iPt]  = std::sqrt(tmpnMx * tmpnMx + tmpnMy * tmpnMy + tmpnMz * tmpnMz);
            Hx[iPt]        = (1./3.) * signMD ;
            Hy[iPt]        = Hx[iPt] ;
            Hz[iPt]        = Hx[iPt] ;
          } else {
            DSDMnorm  = std::sqrt(tmpnMx * tmpnMx + tmpnMy * tmpnMy + tmpnMz * tmpnMz);
            Hx[iPt]        = (tmpnMx * signMD) / DSDMnorm;
            Hy[iPt]        = (tmpnMy * signMD) / DSDMnorm;
            Hz[iPt]        = (tmpnMz * signMD) / DSDMnorm;
          }
 
          inner2 = 0.5  *  DSDMnorm * signMD ;
          
          gammaColl[3*iPt]   += inner;
          gammaColl[3*iPt+1] -= inner;
          gammaColl[3*iPt+2] += inner;
      
          gammaColl[3*iPt]   += inner2;
          gammaColl[3*iPt+2] -= inner2;


        } // loop pts
      } // 2C 
    } //GGA 

  }; //KohnSham<MatsT,IntsT>::mkAuxVar


  /**
   *  \brief evaluates the V (Den, GDENX, GDENY and GDENZ)
   *  varibles for a given density in input in DENMAT 
   *  (scalar or magnetization).
   *
   *
   *  \param [in]  typ        Type of evaluation to perform (gradient, etc)
   *  \param [in]  NPts       Number of points in the batch
   *  \param [in]  NBE        Effective number of basis to be evalauted (only shell actives)
   *  \param [in]  NB         Total Number of basis.
   *  \param [in]  subMatCut  Pair to handle the cut of the shell submatrix to be evaluated.
   *  \param [in]  SCR1       Pointer to an NB*NB scratch.
   *  \param [in]  SCR2       Pointer to an NB*NPts scratch.
   *  \param [in]  DENMAT     Pointer to 1PDM (scalar, Mk).
   *  \param [out] Den        Pointer to the V variable - SCALAR/Mk
   *  \param [out] GDenX      Pointer to the V variable - Gradient X comp of SCALAR/Mk
   *  \param [out] GDenY      Pointer to the V variable - Gradient Y comp of SCALAR/Mk
   *  \param [out] GDenZ      Pointer to the V variable - Gradient Z comp of SCALAR/Mk
   *  \param [in]  BasisScr   Pointer to Basis set evaluated over batch of points.
   */  
  template <typename MatsT, typename IntsT>
  void KohnSham<MatsT,IntsT>::evalDen(SHELL_EVAL_TYPE typ, size_t NPts,size_t NBE, size_t NB, 
    std::vector<std::pair<size_t,size_t>> &subMatCut, double *SCR1,
    double *SCR2, double *DENMAT, double *Den, double *GDenX, double *GDenY, double *GDenZ,
    double *BasisScr){

    size_t IOff = NPts*NBE;

    SubMatSet(NB,NB,NBE,NBE,DENMAT,NB,SCR1,NBE,subMatCut);           

    // Obtain Sum_nu P_mu_nu Phi_nu
    Gemm('N','N',NBE,NPts,NBE,1.,SCR1,NBE,BasisScr,NBE,0.,SCR2,NBE);

    if( typ != GRADIENT )
      for(auto iPt = 0; iPt < NPts; iPt++) {
        Den[iPt] = 0.;
        double *SCR_cur = SCR2 + iPt*NBE;
        double *B_cur   = BasisScr + iPt*NBE;
  
        for (size_t j = 0; j < NBE; j++) 
          Den[iPt] += SCR_cur[j] * B_cur[j];

    } else {

      for(auto iPt = 0; iPt < NPts; iPt++) {
        Den[iPt] = 0.;
        GDenX[iPt] = 0.;
        GDenY[iPt] = 0.;
        GDenZ[iPt] = 0.;
        const size_t NBEiPt = iPt*NBE;
        const double *SCR_cur  = SCR2 + NBEiPt;
        const double *B_cur    = BasisScr + NBEiPt;
        const double *B_curX   = B_cur  + IOff;
        const double *B_curY   = B_curX + IOff;
        const double *B_curZ   = B_curY + IOff;
      
        for(size_t j = 0; j < NBE; j++) { 
          Den[iPt]   += SCR_cur[j] * B_cur[j];
          GDenX[iPt] += SCR_cur[j] * B_curX[j];
          GDenY[iPt] += SCR_cur[j] * B_curY[j];
          GDenZ[iPt] += SCR_cur[j] * B_curZ[j];
        }
        // Since we are summing over mu and nu
        // Del (mu nu) = 2 * Del(mu) nu 
        GDenX[iPt] *= 2.0;
        GDenY[iPt] *= 2.0;
        GDenZ[iPt] *= 2.0;
      }
    }
  }; // //KohnSham<MatsT,IntsT>::evalDen

  /**
   *  \brief Evaluate the EXC energy 
   *
   *  \param [in] NPts     Number of points in the batch.
   *  \param [in] weights  Quadrature weights.
   *  \param [in] epsEval  Pointer to the energy per unit particle. 
   *  \param [in] DenS     Pointer to the scalar density.
   */  
  template <typename MatsT, typename IntsT>
  double KohnSham<MatsT,IntsT>::energy_vxc(size_t NPts, std::vector<double> &weights, double *epsEval, double *DenS){

    double XCEnergy = 0.0;

    for(auto iPt = 0; iPt < NPts; iPt++)  
      XCEnergy += weights[iPt] * epsEval[iPt] * DenS[iPt];
     // std::cerr << "XCEnergy " << XCEnergy << std::endl;
    return XCEnergy;

  };// KohnSham<MatsT,IntsT>::energy_vxc


  
  /**
   *  \brief Construct the required quantities for the formation of the Z vector, 
   *  for a given density component, given the kernel derivatives wrt U variables. 
   *
   *  See J. Chem. Theory Comput. 2011, 7, 3097–3104 (modified e derived for Scalar and Magn).
   *
   *  \param [in] dentyp       Type of 1PDM (SCALAR, {M_k}).
   *  \param [in] isGGA        Whether to include GGA contributions.
   *  \param [in] NPts         Number of points in the batch
   *
   *  \param [in] VRhoEval     Pointer to the first partial derivative of 
   *                           the energy per unit volume in terms of the dens[+,-]. 
   *
   *  \param [in] VgammaEval   Pointer to the first partial derivative of 
   *                           the energy per unit volume in terms of the 
   *                           gammas[++,+-,--].
   *
   *  \param[out]  ZrhoVar1    Factors to multiply the scalar density for particular Z matrix
   *  \param[out]  ZgammaVar1  Factors to multiply the gradient scalar density for particular Z matrix
   *  \param[out]  ZgammaVar2  Factors to multiply the gradient particular mag 
   *                           density for particular Z matrix
   *  
   *  Note we build 2 * X   in Eq 12 and 13 in J. Chem. Theory Comput. 2011, 7, 3097–3104.
   *  Since ZMAT LDA part needs to factor 0.5 for the symmetrization procedure 
   *  (see Eq. 15 1st term on r.h.s) ZrhoVar1  part does not need this factor anymore.
   *
   *  The 0.5 factors come from the chain rules.
   *
   *  On the other hand since we are building 2 * X, we factor already in both
   *  ZgammaVar1 and ZgammaVar2 (since there is 0.5 coming from the chain rules
   *  as well for the GGA terms). Note there is still a factor of 2 that is included
   *  already in the Grad SCALAR/Mz (the one required in Eq 17).
   *
   *  Notes. The ZrhoVar1   multiply the LDA contribution to ZMAT 
   *                        
   *  Notes. The ZgammaVar1 multiply the GGA Del SCALAR
   *                        contribution to ZMAT 
   *  Notes. The ZgammaVar2 multiply the GGA Del Mk 
   *                        contribution to ZMAT 
   *  
   */  
  template <typename MatsT, typename IntsT>
  void KohnSham<MatsT,IntsT>::constructZVars(DENSITY_TYPE denTyp, bool isGGA, size_t NPts, 
    double *VrhoEval, double *VgammaEval, double *ZrhoVar1, 
    double *ZgammaVar1, double *ZgammaVar2) {


    // FIXME: Don't zero out, copy / use MKL VAdd
    memset(ZrhoVar1,0,NPts*sizeof(double));
    if (isGGA) { 
      memset(ZgammaVar1,0,NPts*sizeof(double));
      memset(ZgammaVar2,0,NPts*sizeof(double));
    }


    if( denTyp == SCALAR) {
      // (DE/DN+ DN+/DSCALAR + DE/DN- DN-/DSCALAR )
      // Where DN+/DSCALAR = DN+/DSCALAR = 0.5
      AXPY(NPts,0.5,VrhoEval,2,ZrhoVar1,1);
      AXPY(NPts,0.5 ,VrhoEval+1,2,ZrhoVar1,1);
    } else {
      // (DE/DN+ DN+/DMk + DE/DN- DN-/DMk )
      // Where DN+/DMk =  0.5
      // Where DN-/DMk = -0.5 (only for UKS) 
      AXPY(NPts,0.5,VrhoEval,2,ZrhoVar1,1);
      AXPY(NPts,-0.5,VrhoEval+1,2,ZrhoVar1,1);
    }


    if(isGGA) 
    if( denTyp == SCALAR ) {

      // ( DE/DGamma++ DGamma++/DSCALAR + 
      //   DE/DGamma+- DGamma+-/DSCALAR + 
      //   DE/DGamma-- DGamma--/DSCALAR   ) 

      //   Where DGamma++/DSCALAR = 0.5 * (Del SCAL + Del Mz --- only UKS)
      //   Where DGamma+-/DSCALAR = 0.5 * (Del SCAL)
      //   Where DGamma--/DSCALAR = 0.5 * (Del SCAL - Del Mz --- only UKS)
      //   The Del SCAL and Del Mz will be assembled later in formZ_vxc
      //   NOTE we are building 2 * Z. So 0.5 ---> 1.

      AXPY(NPts,1.,VgammaEval,3  ,ZgammaVar1,1);
      AXPY(NPts,1.,VgammaEval+1,3,ZgammaVar1,1);
      AXPY(NPts,1.,VgammaEval+2,3,ZgammaVar1,1);

      if( this->onePDM.size() > 1 ) {
        AXPY(NPts,1.,VgammaEval,3   ,ZgammaVar2,1);
        AXPY(NPts,-1.,VgammaEval+2,3,ZgammaVar2,1);
      }
    } else {

      // ( DE/DGamma++ DGamma++/DMz + 
      //   DE/DGamma+- DGamma+-/DMz + 
      //   DE/DGamma-- DGamma--/DMz   ) 

      //   Where DGamma++/DMz = 0.5 * (Del Mz + Del SCAL --- only UKS)
      //   Where DGamma+-/DMz = 0.5 * (- Del Mz)
      //   Where DGamma--/DMz = 0.5 * (Del Mz - Del SCAL --- only UKS)
      //   The Del SCAL and Del Mz will be assembled later in formZ_vxc
      //   NOTE we are building 2 * Z. So 0.5 ---> 1.

      AXPY(NPts,1.,VgammaEval,3  ,ZgammaVar2,1);
      AXPY(NPts,-1.,VgammaEval+1,3,ZgammaVar2,1);
      AXPY(NPts,1.,VgammaEval+2,3,ZgammaVar2,1);

      AXPY(NPts,1.,VgammaEval,3   ,ZgammaVar1,1);
      AXPY(NPts,-1.,VgammaEval+2,3,ZgammaVar1,1);
    }

  }; //KohnSham<MatsT,IntsT>::constructZVars


  /**
   *  \brief assemble the final Z vector -> ZMAT (for a given density component),
   *  given the precomputed required U dependent quantities 
   *  (ZUvar# from constructZVars) and the V variables (DenS/Z/X/Y 
   *  and  GDenS/Z/Y/X from evalDen) in input. It requires 
   *  in input also the pointer (BasisScratch) to all basis set (and their gradient)
   *
   *  See J. Chem. Theory Comput. 2011, 7, 3097–3104 (modified e derived for Total and Magn)
   *
   *  \param [in]  isGGA        Whether to include GGA contributions
   *  \param [in]  NPts         Number of points in the batch
   *  \param [in]  NBE          Effective number of basis to be evalauted (only shell actives)
   *  \param [in]  IOff         Offset used for accessing all basis set (and their gradient) 
   *  \param [in]  epsScreen    Screening tollerance
   *  \param [in]  weights      Quadrature weights
   *
   *  \param [in]  ZrhoVar1     Factors to multiply the scalar density for particular Z matrix
   *  \param [in]  ZgammaVar1   Factors to multiply the gradient scalar density for particular Z matrix
   *  \param [in]  ZgammaVar2   Factors to multiply the gradient particular mag 
   *                            density for particular Z matrix
   *
   *  \param [in]  DenS         Pointer to the V variable - SCALAR
   *  \param [in]  DenZ         Pointer to the V variable - Mz 
   *  \param [in]  DenY         Pointer to the V variable - My
   *  \param [in]  DenZ         Pointer to the V variable - Mx
   *  \param [in]  GDenS        Pointer to the V variable - Gradient X,Y,Z comp of SCALAR
   *  \param [in]  GDenZ        Pointer to the V variable - Gradient X,Y,Z comp of Mz 
   *  \param [in]  GDenY        Pointer to the V variable - Gradient X,Y,Z comp of My
   *  \param [in]  GDenZ        Pointer to the V variable - Gradient X,Y,Z comp of Mx
   *
   *  \param [in]  BasisScratch Pointer to Basis set evaluated over batch of points.
   *
   *  \param [out] Pointer to the ZMAT.
   *   
   *  Note. See Documentations of constructZVars.
   */  
  template <typename MatsT, typename IntsT>
  void KohnSham<MatsT,IntsT>::formZ_vxc(DENSITY_TYPE denTyp, bool isGGA, size_t NPts, size_t NBE, size_t IOff, 
    double epsScreen, std::vector<double> &weights, double *ZrhoVar1,
    double *ZgammaVar1, double *ZgammaVar2, 
    double *DenS, double *DenZ, double *DenY, double *DenX, 
    double *GDenS, double *GDenZ, double *GDenY, double *GDenX, 
    double *Kx, double *Ky, double *Kz, 
    double *Hx, double *Hy, double *Hz,
    double *BasisScratch, double *ZMAT){

    double Fg;
    // Fx,y,z  ^m(all batch) in J. Chem. Theory Comput. 2011, 7, 3097–3104 Eq. 17 (changed for Total and Magn)
    double FgX;
    double FgY;
    double FgZ;

    memset(ZMAT,0,IOff*sizeof(double));

    if( this->onePDM.size() <= 2 ) {
      for(auto iPt = 0; iPt < NPts; iPt++) { 
      // LDA part -> Eq. 15 and 16 (see constructZVars docs for the missing factor of 0.5)
        Fg = weights[iPt] * ZrhoVar1[iPt];

#if VXC_DEBUG_LEVEL < 3
        if(std::abs(Fg) > epsScreen)
#endif
          AXPY(NBE,Fg,BasisScratch + iPt*NBE,1,ZMAT+iPt*NBE,1);

      // GGA part -> Eq. 15 and 17 (see constructZVars docs for the missing factor of 2)
        if( isGGA ) {
          FgX = weights[iPt] * ZgammaVar1[iPt] * GDenS[iPt];
          if( this->onePDM.size() > 1 ) 
            FgX += weights[iPt] * ZgammaVar2[iPt] * GDenZ[iPt];

          FgY = weights[iPt] * ZgammaVar1[iPt] * GDenS[iPt + NPts];
          if( this->onePDM.size() > 1 ) 
            FgY += weights[iPt] * ZgammaVar2[iPt] * GDenZ[iPt + NPts];

          FgZ = weights[iPt] * ZgammaVar1[iPt] * GDenS[iPt + 2*NPts];
          if( this->onePDM.size() > 1 ) 
            FgZ += weights[iPt] * ZgammaVar2[iPt] * GDenZ[iPt + 2*NPts];

#if VXC_DEBUG_LEVEL < 3
          if(std::abs(FgX) > epsScreen)
#endif
            AXPY(NBE,FgX,BasisScratch + iPt*NBE + IOff,1,ZMAT+iPt*NBE,1);

#if VXC_DEBUG_LEVEL < 3
          if(std::abs(FgY) > epsScreen)
#endif
            AXPY(NBE,FgY,BasisScratch + iPt*NBE + 2*IOff,1,ZMAT+iPt*NBE,1);

#if VXC_DEBUG_LEVEL < 3
          if(std::abs(FgZ) > epsScreen)
#endif
            AXPY(NBE,FgZ,BasisScratch + iPt*NBE + 3*IOff,1,ZMAT+iPt*NBE,1);
        }
      }

    } else {
      //2C
      // 2C See J. Chem. Theory Comput. 2017, 13, 2591-2603  
      if( denTyp == SCALAR) {
        // SCALAR

        for(auto iPt = 0; iPt < NPts; iPt++) { 
      // LDA part -> Eq. 15 and 16 (see constructZVars docs for the missing factor of 0.5)
          Fg = weights[iPt] * ZrhoVar1[iPt];

#if VXC_DEBUG_LEVEL < 3
          if(std::abs(Fg) > epsScreen)
#endif
            AXPY(NBE,Fg,BasisScratch + iPt*NBE,1,ZMAT+iPt*NBE,1);

        // GGA part -> Eq. 15 and 17 (see constructZVars docs for the missing factor of 2)
          if( isGGA ) {
            FgX  = weights[iPt] * ZgammaVar1[iPt] * GDenS[iPt];
            FgX += weights[iPt] * ZgammaVar2[iPt] * Hx[iPt] * GDenX[iPt];
            FgX += weights[iPt] * ZgammaVar2[iPt] * Hy[iPt] * GDenY[iPt];
            FgX += weights[iPt] * ZgammaVar2[iPt] * Hz[iPt] * GDenZ[iPt];
  
            FgY  = weights[iPt] * ZgammaVar1[iPt] * GDenS[iPt + NPts];
            FgY += weights[iPt] * ZgammaVar2[iPt] * Hx[iPt] * GDenX[iPt + NPts];
            FgY += weights[iPt] * ZgammaVar2[iPt] * Hy[iPt] * GDenY[iPt + NPts];
            FgY += weights[iPt] * ZgammaVar2[iPt] * Hz[iPt] * GDenZ[iPt + NPts];
  
            FgZ  = weights[iPt] * ZgammaVar1[iPt] * GDenS[iPt + 2*NPts];
            FgZ += weights[iPt] * ZgammaVar2[iPt] * Hx[iPt] * GDenX[iPt + 2*NPts];
            FgZ += weights[iPt] * ZgammaVar2[iPt] * Hy[iPt] * GDenY[iPt + 2*NPts];
            FgZ += weights[iPt] * ZgammaVar2[iPt] * Hz[iPt] * GDenZ[iPt + 2*NPts];

#if VXC_DEBUG_LEVEL < 3
            if(std::abs(FgX) > epsScreen)
#endif
              AXPY(NBE,FgX,BasisScratch + iPt*NBE + IOff,1,ZMAT+iPt*NBE,1);

#if VXC_DEBUG_LEVEL < 3
            if(std::abs(FgY) > epsScreen)
#endif
              AXPY(NBE,FgY,BasisScratch + iPt*NBE + 2*IOff,1,ZMAT+iPt*NBE,1);

#if VXC_DEBUG_LEVEL < 3
            if(std::abs(FgZ) > epsScreen)
#endif
              AXPY(NBE,FgZ,BasisScratch + iPt*NBE + 3*IOff,1,ZMAT+iPt*NBE,1);
          }
        } // loop over Pts

      } else if (denTyp == MX) {

        // MX
        for(auto iPt = 0; iPt < NPts; iPt++) { 
      // LDA part -> Eq. 15 and 16 (see constructZVars docs for the missing factor of 0.5)
          Fg = weights[iPt] * ZrhoVar1[iPt] * Kx[iPt];

#if VXC_DEBUG_LEVEL < 3
          if(std::abs(Fg) > epsScreen)
#endif
            AXPY(NBE,Fg,BasisScratch + iPt*NBE,1,ZMAT+iPt*NBE,1);

        // GGA part -> Eq. 15 and 17 (see constructZVars docs for the missing factor of 2)
          if( isGGA ) {
            FgX  = weights[iPt] * ZgammaVar1[iPt] * Hx[iPt] * GDenS[iPt];
            FgX += weights[iPt] * ZgammaVar2[iPt] * GDenX[iPt];
  
            FgY  = weights[iPt] * ZgammaVar1[iPt] * Hx[iPt] * GDenS[iPt + NPts];
            FgY += weights[iPt] * ZgammaVar2[iPt] * GDenX[iPt + NPts];
  
            FgZ  = weights[iPt] * ZgammaVar1[iPt] * Hx[iPt] * GDenS[iPt + 2*NPts];
            FgZ += weights[iPt] * ZgammaVar2[iPt] * GDenX[iPt + 2*NPts];

#if VXC_DEBUG_LEVEL < 3
            if(std::abs(FgX) > epsScreen)
#endif
              AXPY(NBE,FgX,BasisScratch + iPt*NBE + IOff,1,ZMAT+iPt*NBE,1);

#if VXC_DEBUG_LEVEL < 3
            if(std::abs(FgY) > epsScreen)
#endif
              AXPY(NBE,FgY,BasisScratch + iPt*NBE + 2*IOff,1,ZMAT+iPt*NBE,1);

#if VXC_DEBUG_LEVEL < 3
            if(std::abs(FgZ) > epsScreen)
#endif
              AXPY(NBE,FgZ,BasisScratch + iPt*NBE + 3*IOff,1,ZMAT+iPt*NBE,1);
          }
        } // loop over Pts

      } else if (denTyp == MY) {

        // MY
        for(auto iPt = 0; iPt < NPts; iPt++) { 
      // LDA part -> Eq. 15 and 16 (see constructZVars docs for the missing factor of 0.5)
          Fg = weights[iPt] * ZrhoVar1[iPt] * Ky[iPt];

#if VXC_DEBUG_LEVEL < 3
          if(std::abs(Fg) > epsScreen)
#endif
            AXPY(NBE,Fg,BasisScratch + iPt*NBE,1,ZMAT+iPt*NBE,1);

        // GGA part -> Eq. 15 and 17 (see constructZVars docs for the missing factor of 2)
          if( isGGA ) {
            FgX  = weights[iPt] * ZgammaVar1[iPt] * Hy[iPt] * GDenS[iPt];
            FgX += weights[iPt] * ZgammaVar2[iPt] * GDenY[iPt];
  
            FgY  = weights[iPt] * ZgammaVar1[iPt] * Hy[iPt] * GDenS[iPt + NPts];
            FgY += weights[iPt] * ZgammaVar2[iPt] * GDenY[iPt + NPts];
  
            FgZ  = weights[iPt] * ZgammaVar1[iPt] * Hy[iPt] * GDenS[iPt + 2*NPts];
            FgZ += weights[iPt] * ZgammaVar2[iPt] * GDenY[iPt + 2*NPts];

#if VXC_DEBUG_LEVEL < 3
            if(std::abs(FgX) > epsScreen)
#endif
              AXPY(NBE,FgX,BasisScratch + iPt*NBE + IOff,1,ZMAT+iPt*NBE,1);

#if VXC_DEBUG_LEVEL < 3
            if(std::abs(FgY) > epsScreen)
#endif
              AXPY(NBE,FgY,BasisScratch + iPt*NBE + 2*IOff,1,ZMAT+iPt*NBE,1);

#if VXC_DEBUG_LEVEL < 3
            if(std::abs(FgZ) > epsScreen)
#endif
              AXPY(NBE,FgZ,BasisScratch + iPt*NBE + 3*IOff,1,ZMAT+iPt*NBE,1);
          }
        } // loop over Pts

      } else if (denTyp == MZ) {

        // MZ
        for(auto iPt = 0; iPt < NPts; iPt++) { 
      // LDA part -> Eq. 15 and 16 (see constructZVars docs for the missing factor of 0.5)
          Fg = weights[iPt] * ZrhoVar1[iPt] * Kz[iPt];

#if VXC_DEBUG_LEVEL < 3
          if(std::abs(Fg) > epsScreen)
#endif
            AXPY(NBE,Fg,BasisScratch + iPt*NBE,1,ZMAT+iPt*NBE,1);

        // GGA part -> Eq. 15 and 17 (see constructZVars docs for the missing factor of 2)
          if( isGGA ) {
            FgX  = weights[iPt] * ZgammaVar1[iPt] * Hz[iPt] * GDenS[iPt];
            FgX += weights[iPt] * ZgammaVar2[iPt] * GDenZ[iPt];
  
            FgY  = weights[iPt] * ZgammaVar1[iPt] * Hz[iPt] * GDenS[iPt + NPts];
            FgY += weights[iPt] * ZgammaVar2[iPt] * GDenZ[iPt + NPts];
  
            FgZ  = weights[iPt] * ZgammaVar1[iPt] * Hz[iPt] * GDenS[iPt + 2*NPts];
            FgZ += weights[iPt] * ZgammaVar2[iPt] * GDenZ[iPt + 2*NPts];

#if VXC_DEBUG_LEVEL < 3
            if(std::abs(FgX) > epsScreen)
#endif
              AXPY(NBE,FgX,BasisScratch + iPt*NBE + IOff,1,ZMAT+iPt*NBE,1);

#if VXC_DEBUG_LEVEL < 3
            if(std::abs(FgY) > epsScreen)
#endif
              AXPY(NBE,FgY,BasisScratch + iPt*NBE + 2*IOff,1,ZMAT+iPt*NBE,1);

#if VXC_DEBUG_LEVEL < 3
            if(std::abs(FgZ) > epsScreen)
#endif
              AXPY(NBE,FgZ,BasisScratch + iPt*NBE + 3*IOff,1,ZMAT+iPt*NBE,1);
          }  
        } // loop over Pts

      } //MZ

    } //end 2C

  }; // KohnSham<MatsT,IntsT>::formZ_vxc


  /**
   *  \brief assemble the VXC for all density componet over batch
   *  of points. 
   *
   *  It handles submatrix of the VXC (for a given 
   *  subset of shell to be evaluated for the provided batch) 
   *  and assemble them. 
   *
   *  This function is integrated by the BeckeIntegrator
   *  object.
   */  
  template <typename MatsT, typename IntsT>
  void KohnSham<MatsT,IntsT>::formVXC() {
#if VXC_DEBUG_LEVEL >= 1
    // TIMING 
    auto topMem = std::chrono::high_resolution_clock::now();
#endif

    assert( intParam.nRad % intParam.nRadPerBatch == 0 );




    // Parallelism

    size_t nthreads = GetNumThreads();
    size_t LAThreads = GetLAThreads();
    size_t mpiRank   = MPIRank(this->comm);
    size_t mpiSize   = MPISize(this->comm);

    // MPI Communicator for numerical integration
    // *** Assumes that MPI will be done only across atoms in integration

    size_t nAtoms = this->aoints.molecule().nAtoms;
    int color = ((mpiSize < nAtoms) or 
                 (mpiRank < nAtoms)) ? 1 : MPI_UNDEFINED;
                                                            
                  
    MPI_Comm intComm = MPICommSplit(this->comm,color,mpiRank);







#ifdef CQ_ENABLE_MPI
    if( intComm != MPI_COMM_NULL ) 
#endif
    {
  
  
  
  
  
  
      // Define several useful quantities for later on
      size_t NPtsMaxPerBatch = intParam.nRadPerBatch * intParam.nAng;
  
      bool isGGA = std::any_of(functionals.begin(),functionals.end(),
                     [](std::shared_ptr<DFTFunctional> &x) {
                       return x->isGGA(); 
                     }); 
  
      // Turn off LA threads
      SetLAThreads(1);
  
      BasisSet &basis = this->aoints.basisSet();
      size_t NB     = basis.nBasis;
      size_t NB2    = NB*NB;
      size_t NTNB2  = nthreads * NB2;
      size_t NTNPPB = nthreads*NPtsMaxPerBatch;

      // Clean up all VXC components for a the evaluation for a new 
      // batch of points
  
      std::vector<std::vector<double*>> integrateVXC;
      double* intVXC_RAW = (nthreads == 1) ? nullptr :
        this->memManager.template malloc<double>(VXC.size() * NTNB2);
  
      for(auto k = 0; k < VXC.size(); k++) {
        integrateVXC.emplace_back();

        if( nthreads != 1 ) 
          for(auto ith = 0; ith < nthreads; ith++)
            integrateVXC.back().emplace_back(
              intVXC_RAW + (ith + k*nthreads) * NB2
            );

        else 
          integrateVXC.back().emplace_back(VXC[k]);

      }
      
      for(auto &X : integrateVXC) for(auto &Y : X) std::fill_n(Y,NB2,0.);
  
      std::vector<double> integrateXCEnergy(nthreads,0.);
  
      // Start Debug quantities
#if VXC_DEBUG_LEVEL >= 3
      // tmp VXC submat
      double *tmpS      = this->memManager.template malloc<double>(NB2); 
      std::fill_n(tmpS,basis.nBasis*basis.nBasis,0.);
#endif

#if VXC_DEBUG_LEVEL >= 2
      double sumrho   = 0.;
      double sumspin  = 0.;
      double sumgamma = 0.;
#endif
      // END Debug quantities
 
      // Allocating Memory
      // ----------------------------------------------------------//
    
      XCEnergy = 0.;
      double *SCRATCHNBNB = 
        this->memManager.template malloc<double>(NTNB2); 
      double *SCRATCHNBNP = 
        this->memManager.template malloc<double>(NTNPPB*NB); 

      double *DenS, *DenZ, *DenX, *DenY, *Mnorm ;
      double *KScratch;
      double *HScratch;
      bool   *Msmall;
      DenS = this->memManager.template malloc<double>(NTNPPB);

      if( this->onePDM.size() > 1 )
        DenZ = this->memManager.template malloc<double>(NTNPPB);

      if( this->onePDM.size() > 2 ) {
        DenY = this->memManager.template malloc<double>(NTNPPB);
        DenX = this->memManager.template malloc<double>(NTNPPB);

        Mnorm    = this->memManager.template malloc<double>(NTNPPB);
        KScratch = this->memManager.template malloc<double>(3*NTNPPB);
        Msmall   = this->memManager.template malloc<bool>(NTNPPB);
      }

      double *epsEval = this->memManager.template malloc<double>(NTNPPB);

      // Density U-Variables
      double *U_n   = 
        this->memManager.template malloc<double>(2*NTNPPB); 
      double *dVU_n = 
        this->memManager.template malloc<double>(2*NTNPPB); 

      double *ZrhoVar1, *ZgammaVar1, *ZgammaVar2;

      ZrhoVar1 = this->memManager.template malloc<double>(NTNPPB);
      
      double *GDenS, *GDenZ, *GDenX, *GDenY, *U_gamma, *dVU_gamma;
      if( isGGA ) {
        GDenS = this->memManager.template malloc<double>(3*NTNPPB);
        ZgammaVar1 = this->memManager.template malloc<double>(NTNPPB);
        ZgammaVar2 = this->memManager.template malloc<double>(NTNPPB);

        if( this->onePDM.size() > 1 )
          GDenZ = this->memManager.template malloc<double>(3*NTNPPB);

        if( this->onePDM.size() > 2 ) {
          GDenY = this->memManager.template malloc<double>(3*NTNPPB);
          GDenX = this->memManager.template malloc<double>(3*NTNPPB);
          HScratch = this->memManager.template malloc<double>(3*NTNPPB);
        }

        // Gamma U-Variables
        U_gamma = this->memManager.template malloc<double>(3*NTNPPB); 
        dVU_gamma = this->memManager.template malloc<double>(3*NTNPPB); 
      }

      double *epsSCR, *dVU_n_SCR, *dVU_gamma_SCR;
      if( functionals.size() > 1 ) {
        epsSCR    = this->memManager.template malloc<double>(NTNPPB);
        dVU_n_SCR    = this->memManager.template malloc<double>(2*NTNPPB);
        if(isGGA) 
          dVU_gamma_SCR = 
            this->memManager.template malloc<double>(3*NTNPPB);
      }
 
      // ZMatrix
      double *ZMAT = this->memManager.template malloc<double>(NTNPPB*NB);
 
      // Decide if we need to allocate space for real part of the 
      // densities and copy over the real parts
      std::vector<double*> Re1PDM;
      for(auto i = 0; i < this->onePDM.size(); i++) {
        if( std::is_same<MatsT,double>::value )
          Re1PDM.push_back(reinterpret_cast<double*>(this->onePDM[i]));
        else {
          Re1PDM.push_back(this->memManager.template malloc<double>(NB2));
          GetMatRE('N',NB,NB,1.,this->onePDM[i],NB,Re1PDM.back(),NB);
        }
      }
 

      // -------------------------------------------------------------//
      // End allocating Memory

#if VXC_DEBUG_LEVEL >= 1
      // TIMING
      auto botMem = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> durevalDen(0.)  ;
      std::chrono::duration<double> durmkAuxVar(0.) ;
      std::chrono::duration<double> durloadVXCder(0.) ;
      std::chrono::duration<double> durenergy_vxc(0.) ;
      std::chrono::duration<double> durconstructZVars(0.) ;
      std::chrono::duration<double> durformZ_vxc(0.) ;
      std::chrono::duration<double> durDSYR2K(0.) ;
      std::chrono::duration<double> durIncBySubMat(0.) ;
#endif

      auto vxcbuild = [&](size_t &res, std::vector<cart_t> &batch, 
        std::vector<double> &weights, size_t NBE, double *BasisEval, 
        std::vector<size_t> &batchEvalShells, 
        std::vector<std::pair<size_t,size_t>> &subMatCut) {

#if VXC_DEBUG_LEVEL > 3
        prettyPrintSmart(std::cerr,"BASIS SCR",BasisEval,NBE,
          4*batch.size(),NBE);
#endif


        // intParam.epsilon / ntotalpts (NANG * NRAD * NATOMS)
        double epsScreen = intParam.epsilon / nAtoms /
          intParam.nAng / intParam.nRad;

        epsScreen = std::max(epsScreen,
                             std::numeric_limits<double>::epsilon());

        size_t NPts = batch.size();
        size_t IOff = NBE*NPts;

        size_t thread_id = GetThreadID();
        size_t TIDNPPB   = thread_id * NPtsMaxPerBatch;

#if VXC_DEBUG_LEVEL >= 1
        // TIMING
        auto topevalDen = std::chrono::high_resolution_clock::now();
#endif

        // Setup local pointers
        double * SCRATCHNBNB_loc = SCRATCHNBNB + thread_id * NB2;
        double * SCRATCHNBNP_loc = SCRATCHNBNP + thread_id * NB*NPtsMaxPerBatch;

        double * DenS_loc = DenS + TIDNPPB;
        double * DenZ_loc = DenZ + TIDNPPB;
        double * DenY_loc = DenY + TIDNPPB;
        double * DenX_loc = DenX + TIDNPPB;

        double * GDenS_loc = GDenS + 3*TIDNPPB;
        double * GDenZ_loc = GDenZ + 3*TIDNPPB;
        double * GDenY_loc = GDenY + 3*TIDNPPB;
        double * GDenX_loc = GDenX + 3*TIDNPPB;
        

        double * epsEval_loc   = epsEval   +  TIDNPPB;
        double * U_n_loc       = U_n       + 2*TIDNPPB;
        double * dVU_n_loc     = dVU_n     + 2*TIDNPPB;
        double * U_gamma_loc   = U_gamma   + 3*TIDNPPB;
        double * dVU_gamma_loc = dVU_gamma + 3*TIDNPPB;


        double * ZrhoVar1_loc   = ZrhoVar1   + TIDNPPB;
        double * ZgammaVar1_loc = ZgammaVar1 + TIDNPPB;
        double * ZgammaVar2_loc = ZgammaVar2 + TIDNPPB;

        double * epsSCR_loc        = epsSCR        +   TIDNPPB;
        double * dVU_n_SCR_loc     = dVU_n_SCR     + 2*TIDNPPB;
        double * dVU_gamma_SCR_loc = dVU_gamma_SCR + 3*TIDNPPB;

        double *ZMAT_loc = ZMAT + NB * TIDNPPB;

        //2C
        double * Mnorm_loc    = Mnorm        +   TIDNPPB;
        double * KScratch_loc = KScratch     + 3*TIDNPPB;
        bool   * Msmall_loc   = Msmall       +   TIDNPPB;
        double * HScratch_loc = HScratch     + 3*TIDNPPB;

        // This evaluates the V variables for all components 
        // (Scalar, MZ (UKS) and Mx, MY (2 Comp))
        evalDen((isGGA ? GRADIENT : NOGRAD), NPts, NBE, NB, subMatCut, 
          SCRATCHNBNB_loc, SCRATCHNBNP_loc, Re1PDM[SCALAR], DenS_loc, 
          GDenS_loc, GDenS_loc + NPts, GDenS_loc + 2*NPts, BasisEval);

#if VXC_DEBUG_LEVEL < 3
        // Coarse screen on Density
        double MaxDenS_loc = *std::max_element(DenS_loc,DenS_loc+NPts);
        if (MaxDenS_loc < epsScreen) { return; }
#endif

        if( this->onePDM.size() > 1 )
          evalDen((isGGA ? GRADIENT : NOGRAD), NPts, NBE, NB, subMatCut, 
            SCRATCHNBNB_loc ,SCRATCHNBNP_loc, Re1PDM[MZ], DenZ_loc, 
            GDenZ_loc, GDenZ_loc + NPts, GDenZ_loc + 2*NPts, BasisEval);

        if( this->onePDM.size() > 2 ) {
          evalDen((isGGA ? GRADIENT : NOGRAD), NPts, NBE, NB, subMatCut, 
            SCRATCHNBNB_loc ,SCRATCHNBNP_loc, Re1PDM[MY], DenY_loc, 
            GDenY_loc, GDenY_loc + NPts, GDenY_loc + 2*NPts, BasisEval);
          evalDen((isGGA ? GRADIENT : NOGRAD), NPts, NBE, NB, subMatCut, 
            SCRATCHNBNB_loc ,SCRATCHNBNP_loc, Re1PDM[MX], DenX_loc, 
            GDenX_loc, GDenX_loc + NPts, GDenX_loc + 2*NPts, BasisEval);
        }

#if VXC_DEBUG_LEVEL >= 1
        // TIMING
        auto botevalDen  = std::chrono::high_resolution_clock::now();
        auto topmkAuxVar = std::chrono::high_resolution_clock::now();
#endif

        // V -> U variables for evaluating the kernel derivatives.
        mkAuxVar(isGGA,epsScreen,NPts,
          DenS_loc,DenZ_loc,DenY_loc,DenX_loc,
          GDenS_loc,GDenS_loc + NPts,GDenS_loc + 2*NPts,
          GDenZ_loc,GDenZ_loc + NPts,GDenZ_loc + 2*NPts,
          GDenY_loc,GDenY_loc + NPts,GDenY_loc + 2*NPts,
          GDenX_loc,GDenX_loc + NPts,GDenX_loc + 2*NPts,
          Mnorm_loc, 
          KScratch_loc, KScratch_loc + NPts, KScratch_loc + 2* NPts,
          HScratch_loc, HScratch_loc + NPts, HScratch_loc + 2* NPts,
          Msmall_loc,U_n_loc,U_gamma_loc
        );

#if VXC_DEBUG_LEVEL >= 1
        // TIMING
        auto botmkAuxVar = std::chrono::high_resolution_clock::now();
#endif


#if VXC_DEBUG_LEVEL >= 2
        assert(nthreads == 1);
        // Debug int
        for(auto iPt = 0; iPt < NPts; iPt++) { 
          sumrho  += weights[iPt] * (U_n[2*iPt] + U_n[2*iPt + 1]);
          sumspin += weights[iPt] * (U_n[2*iPt] - U_n[2*iPt + 1]);
          if(isGGA) 
            sumgamma += weights[iPt] * 
              ( U_gamma[3*iPt] + U_gamma[3*iPt+1] + U_gamma[3*iPt+2]);
        };
        // end debug
#endif
      
#if VXC_DEBUG_LEVEL >= 1
        // TIMING
        auto toploadVXCder = std::chrono::high_resolution_clock::now();
#endif

        // Get DFT Energy derivatives wrt U variables
        loadVXCder(NPts, U_n_loc, U_gamma_loc, epsEval_loc, dVU_n_loc, 
          dVU_gamma_loc, epsSCR_loc, dVU_n_SCR_loc, dVU_gamma_SCR_loc); 
  
#if VXC_DEBUG_LEVEL >= 1
        // TIMING
        auto botloadVXCder = std::chrono::high_resolution_clock::now();
        auto topenergy_vxc = std::chrono::high_resolution_clock::now();
#endif

        // Compute for the current batch the XC energy and increment the 
        // total XC energy.
        integrateXCEnergy[thread_id] += 
          energy_vxc(NPts, weights, epsEval_loc, DenS_loc);

#if VXC_DEBUG_LEVEL >= 1
        // TIMING
        auto botenergy_vxc     = 
          std::chrono::high_resolution_clock::now();
        auto topconstructZVars = 
          std::chrono::high_resolution_clock::now();
#endif
   
        // Construct the required quantities for the formation of the Z 
        // vector (SCALAR) given the kernel derivatives wrt U variables. 

        constructZVars(SCALAR,isGGA,NPts,dVU_n_loc,dVU_gamma_loc,
          ZrhoVar1_loc, ZgammaVar1_loc, ZgammaVar2_loc);

#if VXC_DEBUG_LEVEL >= 1
        // TIMING
        auto botconstructZVars = 
          std::chrono::high_resolution_clock::now();
        auto topformZ_vxc      = 
          std::chrono::high_resolution_clock::now();
#endif

        // Creating ZMAT (SCALAR) according to 
        //   J. Chem. Theory Comput. 2011, 7, 3097–3104 Eq. 15 
        formZ_vxc(SCALAR,isGGA, NPts, NBE, IOff, epsScreen, weights, 
          ZrhoVar1_loc, ZgammaVar1_loc, ZgammaVar2_loc, DenS_loc, 
          DenZ_loc, DenY_loc, DenX_loc, GDenS_loc, GDenZ_loc, GDenY_loc, 
          GDenX_loc, KScratch_loc, KScratch_loc + NPts, 
          KScratch_loc + 2* NPts, HScratch_loc, HScratch_loc + NPts, 
          HScratch_loc + 2* NPts, BasisEval, ZMAT_loc);

#if VXC_DEBUG_LEVEL >= 1
        // TIMING
        auto botformZ_vxc = std::chrono::high_resolution_clock::now();
#endif

        bool evalZ = true;

#if VXC_DEBUG_LEVEL < 3
        // Coarse screen on ZMat
        double MaxBasis = *std::max_element(BasisEval,BasisEval+IOff);
        double MaxZ     = *std::max_element(ZMAT_loc,ZMAT_loc+IOff);
        evalZ = ( std::abs(2 * MaxBasis * MaxZ) > epsScreen); 
#endif

        if (evalZ) {

 #if VXC_DEBUG_LEVEL >= 1
          auto topDSYR2K    = std::chrono::high_resolution_clock::now();
 #endif
          // Creating according to 
          //   J. Chem. Theory Comput. 2011, 7, 3097–3104 Eq. 14 
          //
          // Z -> VXC (submat - SCALAR)
          DSYR2K('L','N',NBE,NPts,1.,BasisEval,NBE,ZMAT_loc,NBE,0.,
            SCRATCHNBNB_loc,NBE);

 #if VXC_DEBUG_LEVEL >= 1
          // TIMING
          auto botDSYR2K      = std::chrono::high_resolution_clock::now();
          durDSYR2K += botDSYR2K - topDSYR2K;
          auto topIncBySubMat = std::chrono::high_resolution_clock::now();
 #endif

          // Locating the submatrix in the right position given the 
          // subset of shells for the given batch.
          IncBySubMat(NB,NB,NBE,NBE,integrateVXC[SCALAR][thread_id],NB,
            SCRATCHNBNB_loc,NBE,subMatCut);

 #if VXC_DEBUG_LEVEL >= 1
          // TIMING
          auto botIncBySubMat = std::chrono::high_resolution_clock::now();
          durIncBySubMat += botIncBySubMat - topIncBySubMat;
 #endif
        }



#if VXC_DEBUG_LEVEL > 3
        prettyPrintSmart(std::cerr,"Basis   ",BasisEval,NBE,NPts,NBE);
        prettyPrintSmart(std::cerr,"BasisX  ",BasisEval+NBE*NPts,NBE,
          NPts,NBE);
        prettyPrintSmart(std::cerr,"BasisY  ",BasisEval+2*NBE*NPts,
          NBE,NPts,NBE);
        prettyPrintSmart(std::cerr,"BasisZ  ",BasisEval+3*NBE*NPts,
          NBE,NPts,NBE);
        prettyPrintSmart(std::cerr,"ZMAT  ",ZMAT_loc,NBE,NPts,NBE);
#endif

#if VXC_DEBUG_LEVEL >= 1
        // TIMING
        durevalDen += botevalDen - topevalDen;
        durmkAuxVar += botmkAuxVar - topmkAuxVar;
        durloadVXCder += botloadVXCder - toploadVXCder;
        durenergy_vxc += botenergy_vxc - topenergy_vxc;
        durconstructZVars += botconstructZVars - topconstructZVars;
        durformZ_vxc += botformZ_vxc - topformZ_vxc;
#endif

#if VXC_DEBUG_LEVEL >= 3
        // Create Numerical Overlap
        for(auto iPt = 0; iPt < NPts; iPt++)
          Gemm('N','C',NB,NB,1,weights[iPt],BasisEval + iPt*NB,NB, 
            BasisEval + iPt*NB,NB, 1.,tmpS,NB);
#endif

        if( this->onePDM.size() == 1 ) return;

        //
        // ---------------   UKS or 2C ------------- Mz ---------------
        //    See J. Chem. Theory Comput. 2017, 13, 2591-2603  
        //

        // Construct the required quantities for the formation of 
        // the Z vector (Mz) given the kernel derivatives wrt U 
        // variables.
        constructZVars(MZ,isGGA,NPts,dVU_n_loc,dVU_gamma_loc,ZrhoVar1_loc,
          ZgammaVar1_loc, ZgammaVar2_loc);

        // Creating ZMAT (Mz) according to 
        //   J. Chem. Theory Comput. 2011, 7, 3097–3104 Eq. 15 
        formZ_vxc(MZ,isGGA, NPts, NBE, IOff, epsScreen, weights, 
          ZrhoVar1_loc, ZgammaVar1_loc, ZgammaVar2_loc, DenS_loc, 
          DenZ_loc, DenY_loc, DenX_loc, GDenS_loc, GDenZ_loc, GDenY_loc, 
          GDenX_loc, KScratch_loc, KScratch_loc + NPts, 
          KScratch_loc + 2* NPts, HScratch_loc, HScratch_loc + NPts, 
          HScratch_loc + 2* NPts, BasisEval, ZMAT_loc);


#if VXC_DEBUG_LEVEL < 3
        MaxZ     = *std::max_element(ZMAT_loc,ZMAT_loc+IOff);
        evalZ = ( std::abs(2 * MaxBasis * MaxZ) > epsScreen); 
#endif
        // Coarse screen on ZMat
        if(evalZ) {
  
          // Creating according to 
          //   J. Chem. Theory Comput. 2011, 7, 3097–3104 Eq. 14 
          //
          // Z -> VXC (submat)
          DSYR2K('L','N',NBE,NPts,1.,BasisEval,NBE,ZMAT_loc,NBE,0.,
            SCRATCHNBNB_loc,NBE);
    
    
          // Locating the submatrix in the right position given 
          // the subset of shells for the given batch.
          IncBySubMat(NB,NB,NBE,NBE,integrateVXC[MZ][thread_id],NB,
            SCRATCHNBNB_loc,NBE,subMatCut);           

        }
 


        if( this->onePDM.size() <= 2 ) return;

        //
        // ---------------  2C ------------- My ----------------------
        //

        // Construct the required quantities for the formation of the 
        // Z vector (Mz) given the kernel derivatives wrt U variables. 
        constructZVars(MY,isGGA,NPts,dVU_n_loc,dVU_gamma_loc,ZrhoVar1_loc,
          ZgammaVar1_loc, ZgammaVar2_loc);

        // Creating ZMAT (My) according to 
        //   J. Chem. Theory Comput. 2011, 7, 3097–3104 Eq. 15 
        formZ_vxc(MY,isGGA, NPts, NBE, IOff, epsScreen, weights, 
          ZrhoVar1_loc, ZgammaVar1_loc, ZgammaVar2_loc, DenS_loc, 
          DenZ_loc, DenY_loc, DenX_loc, GDenS_loc, GDenZ_loc, GDenY_loc, 
          GDenX_loc, KScratch_loc, KScratch_loc + NPts, 
          KScratch_loc + 2* NPts, HScratch_loc, HScratch_loc + NPts, 
          HScratch_loc + 2* NPts, BasisEval, ZMAT_loc);


#if VXC_DEBUG_LEVEL < 3
        MaxZ     = *std::max_element(ZMAT_loc,ZMAT_loc+IOff);
        evalZ = ( std::abs(2 * MaxBasis * MaxZ) > epsScreen); 
#endif
        // Coarse screen on ZMat
        if(evalZ) {
  
          // Creating according to 
          //   J. Chem. Theory Comput. 2011, 7, 3097–3104 Eq. 14 
          //
          // Z -> VXC (submat)
          DSYR2K('L','N',NBE,NPts,1.,BasisEval,NBE,ZMAT_loc,NBE,0.,
            SCRATCHNBNB_loc,NBE);
    
    
          // Locating the submatrix in the right position given the 
          // subset of shells for the given batch.
          IncBySubMat(NB,NB,NBE,NBE,integrateVXC[MY][thread_id],NB,
            SCRATCHNBNB_loc,NBE,subMatCut);           

        }

        //
        // ---------------  2C ------------- Mx ----------------------
        //

        // Construct the required quantities for the formation of the 
        // Z vector (Mx) given the kernel derivatives wrt U variables. 
        constructZVars(MX,isGGA,NPts,dVU_n_loc,dVU_gamma_loc,ZrhoVar1_loc,
          ZgammaVar1_loc, ZgammaVar2_loc);

        // Creating ZMAT (Mz) according to 
        //   J. Chem. Theory Comput. 2011, 7, 3097–3104 Eq. 15 
        formZ_vxc(MX,isGGA, NPts, NBE, IOff, epsScreen, weights, 
          ZrhoVar1_loc, ZgammaVar1_loc, ZgammaVar2_loc, DenS_loc, 
          DenZ_loc, DenY_loc, DenX_loc, GDenS_loc, GDenZ_loc, GDenY_loc, 
          GDenX_loc, KScratch_loc, KScratch_loc + NPts, 
          KScratch_loc + 2* NPts, HScratch_loc, HScratch_loc + NPts, 
          HScratch_loc + 2* NPts, BasisEval, ZMAT_loc);


#if VXC_DEBUG_LEVEL < 3
        MaxZ     = *std::max_element(ZMAT_loc,ZMAT_loc+IOff);
        evalZ = ( std::abs(2 * MaxBasis * MaxZ) > epsScreen); 
#endif
        // Coarse screen on ZMat
        if(evalZ) {
  
          // Creating according to 
          //   J. Chem. Theory Comput. 2011, 7, 3097–3104 Eq. 14 
          //
          // Z -> VXC (submat)
          DSYR2K('L','N',NBE,NPts,1.,BasisEval,NBE,ZMAT_loc,NBE,0.,
            SCRATCHNBNB_loc,NBE);
    
    
          // Locating the submatrix in the right position given the 
          // subset of shells for the given batch.
          IncBySubMat(NB,NB,NBE,NBE,integrateVXC[MX][thread_id],NB,
            SCRATCHNBNB_loc,NBE,subMatCut);           
        }

      }; // VXC integrate


      // Create the BeckeIntegrator object
      BeckeIntegrator<EulerMac> 
        integrator(intComm,this->memManager,this->aoints.molecule(),basis,
        EulerMac(intParam.nRad), intParam.nAng, intParam.nRadPerBatch,
          (isGGA ? GRADIENT : NOGRAD), intParam.epsilon);

      // Integrate the VXC
      integrator.integrate<size_t>(vxcbuild);


#if VXC_DEBUG_LEVEL >= 1
      // TIMING
      auto toptransform    = std::chrono::high_resolution_clock::now();
#endif

      // Finishing up the VXC
      // factor in the 4 pi (Lebedev) and built the upper triagolar part
      // since we create only the lower triangular. For all components
      for(auto k = 0; k < VXC.size(); k++) {
        if( nthreads == 1 )
          Scale(NB2,4*M_PI,VXC[k],1);
        else
          for(auto ithread = 0; ithread < nthreads; ithread++)
            MatAdd('N','N',NB,NB,((ithread == 0) ? 0. : 1.),VXC[k],NB,
              4*M_PI,integrateVXC[k][ithread],NB, VXC[k],NB);
        
        HerMat('L',NB,VXC[k],NB);
      }

      for(auto &X : integrateXCEnergy)
        XCEnergy += 4*M_PI*X;



// Combine MPI results
#ifdef CQ_ENABLE_MPI

      double* mpiScr;
      if( mpiRank == 0 ) mpiScr = this->memManager.template malloc<double>(NB*NB);

      for(auto &V : VXC) {

        mxx::reduce(V,NB*NB,mpiScr,0,std::plus<double>(),intComm);

        if( mpiRank == 0 ) std::copy_n(mpiScr,NB*NB,V);

      }

      if( mpiRank == 0 ) this->memManager.free(mpiScr);

      XCEnergy = mxx::reduce(XCEnergy,0,std::plus<double>(),intComm);

#endif

#if VXC_DEBUG_LEVEL >= 1
      // TIMING
      auto bottransform = std::chrono::high_resolution_clock::now();
#endif

#if VXC_DEBUG_LEVEL >= 3
      // DebugPrint
      std::cerr << std::endl;
      Scale(NB2,4*M_PI,tmpS,1);
      prettyPrintSmart(std::cerr,"Analytic  Overlap",
        this->aoints.overlap,NB,NB,NB);
      prettyPrintSmart(std::cerr,"Numeric  Overlap",tmpS,NB,NB,NB);
      std::cerr << std::endl;
      std::cerr << std::endl;
      for(auto i = 0; i < NB2; i++)
        tmpS[i] = std::abs(tmpS[i] - this->aoints.overlap[i]);
      std::cerr << "MAX DIFF OVERLAP = " << 
        *std::max_element(tmpS,tmpS+NB2) << std::endl;
#endif



#if VXC_DEBUG_LEVEL >= 2
      // DEBUG
      std::cerr << std::scientific << "\n";
      std::cerr << "N     electrons      = " << 4*M_PI*sumrho << "\n";
      std::cerr << "N unp electrons      = " << 4*M_PI*sumspin << "\n";
      std::cerr << "sum gamma        = " << 4*M_PI*sumgamma << "\n";
      std::cerr << "EXC              = " << XCEnergy << "\n";

      prettyPrintSmart(std::cerr,"onePDM Scalar",this->onePDM[SCALAR],
        NB,NB,NB);
      prettyPrintSmart(std::cerr,"Numerical Scalar VXC ",
        integrateVXC[SCALAR][0],NB,NB,NB);

      if( not this->iCS or this->nC > 1 ) { 
        prettyPrintSmart(std::cerr,"onePDM Mz",this->onePDM[MZ],
          NB,NB,NB);
        prettyPrintSmart(std::cerr,"Numerical Mz VXC",
          integrateVXC[MZ][0],NB,NB,NB);

        if( this->onePDM.size() > 2 ) {
          prettyPrintSmart(std::cerr,"onePDM My",this->onePDM[MY],
            NB,NB,NB);
          prettyPrintSmart(std::cerr,"Numerical My VXC",
            integrateVXC[MY][0],NB,NB,NB);

          prettyPrintSmart(std::cerr,"onePDM Mx",this->onePDM[MX],
            NB,NB,NB);
          prettyPrintSmart(std::cerr,"Numerical Mx VXC",
            integrateVXC[MX][0],NB,NB,NB);
        }
      }
#endif

      // Freeing the memory
      // ------------------------------------------------------------ //
      this->memManager.free(SCRATCHNBNB,SCRATCHNBNP,DenS,epsEval,U_n,
        dVU_n, ZrhoVar1,ZMAT);
      if( isGGA )  
        this->memManager.free(ZgammaVar1,ZgammaVar2,GDenS,U_gamma,
          dVU_gamma);

      if( this->onePDM.size() > 1 ) {
        this->memManager.free(DenZ);
        if( isGGA )  this->memManager.free(GDenZ);
      }

      if( this->onePDM.size() > 2 ) {
        this->memManager.free(DenX,DenY,Mnorm,KScratch,Msmall);
        if( isGGA )  this->memManager.free(GDenX,GDenY,HScratch);
      }

      if( functionals.size() > 1 ) {
        this->memManager.free(epsSCR,dVU_n_SCR);
        if( isGGA ) this->memManager.free(dVU_gamma_SCR);
      }


      if( nthreads != 1 ) this->memManager.free(intVXC_RAW);

      if( not std::is_same<MatsT,double>::value )
        for(auto &X : Re1PDM) this->memManager.free(X);

      // ------------------------------------------------------------- //
      // End freeing the memory


#if VXC_DEBUG_LEVEL >= 1
     // TIMING
     double d_batch = this->aoints.molecule().nAtoms * 
                        intParam.nRad / intParam.nRadPerBatch;

     std::chrono::duration<double> durMem = botMem - topMem;
     std::chrono::duration<double> durtransform = 
       bottransform - toptransform;

     std::cerr << std::scientific << "\n";
     std::cerr << "Mem " << durMem.count()/d_batch << "\n";
     std::cerr << "transform " << durtransform.count()/d_batch << "\n";
     std::cerr << "evalDen " << durevalDen.count()/d_batch << "\n";
     std::cerr << "mkAuxVar " << durmkAuxVar.count()/d_batch << "\n";
     std::cerr << "loadVXCder " << durloadVXCder.count()/d_batch << "\n";
     std::cerr << "energy_vxc " << durenergy_vxc.count()/d_batch << "\n";
     std::cerr << "constructZVars " << durconstructZVars.count()/d_batch 
               << "\n";
     std::cerr << "formZ_vxc " << durformZ_vxc.count()/d_batch << "\n";
     std::cerr << "DSYR2K " << durDSYR2K.count()/d_batch << "\n";
     std::cerr << "IncBySubMat " << durIncBySubMat.count()/d_batch;
     std::cerr <<  "\n\n\n";
#endif


  
      // Turn back on LA threads
      SetLAThreads(LAThreads);

      MPICommFree(intComm); // Free communicator

    } // Valid intComm

    MPI_Barrier(this->comm); // Syncronize the MPI processes

  }; // KohnSham::formVXC

}; // namespace ChronusQ

#endif
