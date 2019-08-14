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
#include <singleslater.hpp>
#include <cqlinalg/blasext.hpp>

#include <cqlinalg.hpp>
#include <cqlinalg/svd.hpp>
#include <cqlinalg/blasutil.hpp>
#include <physcon.hpp>
#include <util/matout.hpp>
#include <util/threads.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>

// #define _DEBUGGIAOONEE //SS  

namespace ChronusQ {


  template <>
  void SingleSlater<dcomplex, dcomplex>::addMagPert(EMPerturbation &pert,
      std::vector<dcomplex*> &CH) {


    //Compute the GIAO non-relativistic core Hamiltonian in the CGTO basis
    //H(S) = 2(T + V) + B * L + sigma * B + 1/4 *(B\timesr)^2

    size_t NB = this->aoints.basisSet().nBasis;
    dcomplex onei = dcomplex(0,1);
    auto magAmp = pert.getDipoleAmp(Magnetic);


    // this part add the angular momentum term 
    for ( auto index = 0 ; index < 3 ; index++ ) {
      MatAdd('N','N',NB,NB,-magAmp[index]*onei,
        this->aoints.magDipole[index],NB,dcomplex(1.),CH[0],NB,CH[0],NB);
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
        this->aoints.lenElecQuadrupole[diagindex[index]],
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
        this->aoints.lenElecQuadrupole[offindex[index]],
        NB,dcomplex(1.),CH[0],NB,CH[0], NB);
    }   


    // finally spin Zeeman term
    if(CH.size() > 1) {
      // z component
      SetMat('N',NB,NB,dcomplex(magAmp[2]),this->aoints.overlap,
        NB, CH[1],NB );
    
      if(CH.size() > 2) {
        // y component 
        SetMat('N',NB,NB,dcomplex(magAmp[1]),this->aoints.overlap,
          NB, CH[2],NB );

        // x coponent
        SetMat('N',NB,NB,dcomplex(magAmp[0]),this->aoints.overlap,
          NB, CH[3],NB );
      }
    }


  }

  template <>
  void SingleSlater<dcomplex, double>::addMagPert(EMPerturbation &pert,
    std::vector<dcomplex*> &CH) {


    CErr("GIAO + Real integrals is not a valid option");

  }
  template <>
  void SingleSlater<double, double>::addMagPert(EMPerturbation &pert,
    std::vector<double*> &CH) {


    CErr("GIAO + Real integrals is not a valid option");

  }

  /**
   *  \brief Compute the non-relativistic Core Hamiltonian in the CGTO basis
   *
   *  \f[ H(S) = 2(T + V) \f]
   */ 
  template <typename MatsT, typename IntsT>
  void SingleSlater<MatsT,IntsT>::computeNRCH(EMPerturbation& emPert, std::vector<MatsT*> &CH) {

    size_t NB = this->aoints.basisSet().nBasis;

    //MatAdd('N','N',NB,NB,IntsT(2.),this->aoints.kinetic,NB,
    //  IntsT(2.),this->aoints.potential,NB,CH[0],NB);


    // MatAdd for Real + Real -> Complex does not make sense
    for(auto k = 0ul; k < NB*NB; k++)
      CH[0][k] = 2. * (this->aoints.kinetic[k] + this->aoints.potential[k]);

    if( this->aoints.basisSet().basisType == COMPLEX_GIAO and pert_has_type(emPert,Magnetic) ) 
      addMagPert(emPert,CH);


#ifdef _DEBUGGIAOONEE
      // prettyPrintSmart(std::cout,"Core H",CH[0],NB,NB,NB);
      for ( auto ii = 0 ; ii < CH.size() ; ii++ ) { 
        std::cout<<"ii= "<<ii<<std::endl;
        prettyPrintSmart(std::cout,"Core H",CH[ii],NB,NB,NB);
      }
#endif 

  };  // void SingleSlater::computeNRCH(std::vector<MatsT*> &CH)



  template void SingleSlater<double,double>::computeNRCH(EMPerturbation& emPert,std::vector<double*> &CH);
  template void SingleSlater<dcomplex,double>::computeNRCH(EMPerturbation& emPert,std::vector<dcomplex*> &CH);
  template void SingleSlater<dcomplex,dcomplex>::computeNRCH(EMPerturbation& emPert, std::vector<dcomplex*> &CH);







  template <typename T>
  void formW(size_t NP, dcomplex *W, size_t LDW, T* pVdotP, size_t LDD, T* pVxPZ, 
    size_t LDZ, T* pVxPY, size_t LDY, T* pVxPX, size_t LDX);

  template <>
  void formW(size_t NP, dcomplex *W, size_t LDW, dcomplex* pVdotP, size_t LDD, dcomplex* pVxPZ, 
    size_t LDZ, dcomplex* pVxPY, size_t LDY, dcomplex* pVxPX, size_t LDX) {

    // W = [ W1  W2 ]
    //     [ W3  W4 ]
    dcomplex *W1 = W;
    dcomplex *W2 = W1 + LDW*NP;
    dcomplex *W3 = W1 + NP;
    dcomplex *W4 = W2 + NP;

    // W1 = pV.p + i (pVxp)(Z)
    MatAdd('N','N',NP,NP,dcomplex(1.),pVdotP,LDD,dcomplex(0.,1.),pVxPZ,LDZ,W1,LDW);
    // W4 = pV.p - i (pVxp)(Z)
    MatAdd('N','N',NP,NP,dcomplex(1.),pVdotP,LDD,dcomplex(0.,-1.),pVxPZ,LDZ,W4,LDW);
    // W2 = (pVxp)(Y) + i (pVxp)(X)
    MatAdd('N','N',NP,NP,dcomplex(1.),pVxPY,LDY,dcomplex(0.,1.),pVxPX,LDX,W2,LDW);
    // W3 = (pVxp)(Y) - i (pVxp)(X)
    MatAdd('N','N',NP,NP,dcomplex(1.),pVxPY,LDY,dcomplex(0.,-1.),pVxPX,LDX,W3,LDW);
  };

  template <>
  void formW(size_t NP, dcomplex *W, size_t LDW, double* pVdotP, size_t LDD, double* pVxPZ, 
    size_t LDZ, double* pVxPY, size_t LDY, double* pVxPX, size_t LDX) {

    // W = [ W1  W2 ]
    //     [ W3  W4 ]
    dcomplex *W1 = W;
    dcomplex *W2 = W1 + LDW*NP;
    dcomplex *W3 = W1 + NP;
    dcomplex *W4 = W2 + NP;

    // W1 = pV.p + i (pVxp)(Z)
    SetMatRE('N',NP,NP,1.,pVdotP,LDD,W1,LDW);
    SetMatIM('N',NP,NP,1.,pVxPZ,LDZ,W1,LDW);

    // W4 = conj(W1)
    SetMatRE('N',NP,NP,1.,pVdotP,LDD,W4,LDW);
    SetMatIM('N',NP,NP,-1.,pVxPZ,LDZ,W4,LDW);
  
    // W2 = (pVxp)(Y) + i (pVxp)(X)
    SetMatRE('N',NP,NP,1.,pVxPY,LDY,W2,LDW);
    SetMatIM('N',NP,NP,1.,pVxPX,LDX,W2,LDW);

    // W3 = -conj(W2)
    SetMatRE('N',NP,NP,-1.,pVxPY,LDY,W3,LDW);
    SetMatIM('N',NP,NP,1., pVxPX,LDX,W3,LDW);
  };





  /**
   *  \brief Compute the X2C Core Hamiltonian
   */ 
  template <typename MatsT, typename IntsT> 
  void SingleSlater<MatsT,IntsT>::computeX2CCH(EMPerturbation& emPert, std::vector<MatsT*> &CH) {

    IntsT* XXX = reinterpret_cast<IntsT*>(NULL);



    auto uncontractedBasis = this->aoints.basisSet().uncontractBasis();
    AOIntegrals<IntsT> uncontractedInts(memManager,
        this->aoints.molecule(),*uncontractedBasis);


    size_t NP = uncontractedBasis->nPrimitive;
    size_t NB = this->aoints.basisSet().nBasis;

    uncontractedInts.computeAOOneE(emPert,oneETerms); // FIXME: need to compute SL

    // Compute the mappings from primitives to CGTOs
    IntsT * mapPrim2Cont = memManager.malloc<IntsT>(NP*NB);
    this->aoints.basisSet().makeMapPrim2Cont(uncontractedInts.overlap,
      mapPrim2Cont,memManager);

    // Transformation matrix
    IntsT *UK = memManager.malloc<IntsT>(NP*NP);

    // Allocate Scratch Space (enough for 2*NP x 2*NP complex matricies)
    IntsT   *SCR1  = memManager.malloc<IntsT>(8*NP*NP);
    dcomplex *CSCR1 = reinterpret_cast<dcomplex*>(SCR1);

    // Make a copy of the overlap for later
    IntsT* SCPY = memManager.malloc<IntsT>(NP*NP);
    memcpy(SCPY,uncontractedInts.overlap,NP*NP*sizeof(IntsT));

    // Singular value storage (initially S then T)
    double* SS   = memManager.malloc<double>(NP);
    
    // Get SVD of uncontracted overlap
    // Store the left singular vectors in S
    SVD('O','N',NP,NP,uncontractedInts.overlap,NP,SS,XXX,NP,
      XXX,NP,memManager);

    double minSS = *std::min_element(SS,SS+NP);

    if( minSS < 1e-10 ) CErr("Uncontracted Overlap is Singular");

    // Form orthonormal transformation matrix in S
    for(auto i = 0ul; i < NP; i++)
      Scale(NP,IntsT(1.)/std::sqrt(SS[i]),
          uncontractedInts.overlap + i*NP,1);

    // Transform T into the orthonormal basis
    // T -> TO
    Gemm('C','N',NP,NP,NP,IntsT(1.),uncontractedInts.overlap,NP,
        uncontractedInts.kinetic,NP,IntsT(0.),SCR1,NP);
    Gemm('N','N',NP,NP,NP,IntsT(1.),SCR1,NP,uncontractedInts.overlap,NP,
        IntsT(0.),uncontractedInts.kinetic,NP);

    // Get the SVD of TO
    // Store the left singular vectors in TO
    SVD('O','N',NP,NP,uncontractedInts.kinetic,NP,SS,XXX,NP,
      XXX,NP,memManager);

    minSS = *std::min_element(SS,SS+NP);
    if( minSS < 1e-10 ) CErr("Uncontracted Kinetic Energy Tensor is Singular");

    // Form UK = S * T
    Gemm('N','N',NP,NP,NP,IntsT(1.),uncontractedInts.overlap,NP,
      uncontractedInts.kinetic,NP,IntsT(0.),UK,NP);

    // Allocate and for "P^2" potential
    IntsT *P2P = memManager.malloc<IntsT>(NP*NP);

    // P2P = UK**T * V * UK
    Gemm('C','N',NP,NP,NP,IntsT(1.),UK,NP,uncontractedInts.potential,NP,
        IntsT(0.),SCR1,NP);
    Gemm('N','N',NP,NP,NP,IntsT(1.),SCR1,NP,UK,NP,IntsT(0.),P2P,NP);

    // Transform PVP into the "P^2" basis
    Gemm('C','N',NP,NP,NP,IntsT(1.),UK,NP,uncontractedInts.PVdotP,NP,
        IntsT(0.),SCR1,NP);
    Gemm('N','N',NP,NP,NP,IntsT(1.),SCR1,NP,UK,NP,
        IntsT(0.),uncontractedInts.PVdotP,NP);
    
    // Loop over PVxP terms
    for(auto & SL : uncontractedInts.PVcrossP ){
      Gemm('C','N',NP,NP,NP,IntsT(1.),UK,NP,SL,NP  ,IntsT(0.),SCR1,NP);
      Gemm('N','N',NP,NP,NP,IntsT(1.),SCR1,NP,UK,NP,IntsT(0.),SL,NP);
    }

    // P^2 -> P^-1
    for(auto i = 0; i < NP; i++) SS[i] = 1./std::sqrt(2*SS[i]);

    // Transform PVP into "P^-1" basis
    for(auto j = 0; j < NP; j++) 
    for(auto i = 0; i < NP; i++){
      uncontractedInts.PVdotP[i + j*NP] *= SS[i] * SS[j];
      for(auto &SL : uncontractedInts.PVcrossP)
        SL[i + j*NP] *= SS[i] * SS[j];
    } 

    // Allocate 4C CORE Hamiltonian

    // CH = [ V    cp       ]
    //      [ cp   W - 2mc^2]
    dcomplex *CH4C = memManager.malloc<dcomplex>(16*NP*NP);
    memset(CH4C,0,16*NP*NP*sizeof(dcomplex));

    // Allocate W separately  as it's needed later
    size_t LDW = 2*NP;
    dcomplex *W  = memManager.malloc<dcomplex>(LDW*LDW);
      
    formW(NP,W,LDW,uncontractedInts.PVdotP,NP,
        uncontractedInts.PVcrossP[2],NP,
        uncontractedInts.PVcrossP[1],NP,
        uncontractedInts.PVcrossP[0],NP);      

    // Subtract out 2mc^2 from W diagonals
    const double WFact = 2. * SpeedOfLight * SpeedOfLight;
    for(auto j = 0ul; j < 2*NP; j++) W[j + LDW*j] -= WFact;

    // Copy W into the 4C CH storage
    dcomplex *CHW = CH4C + 8*NP*NP + 2*NP;
    SetMat('N',2*NP,2*NP,dcomplex(1.),W,LDW,CHW,4*NP);
  
    // P^-1 -> P
    for(auto i = 0; i < NP; i++) SS[i] = 1./SS[i];

    // V = [ P2P  0   ]
    //     [ 0    P2P ]
    dcomplex * V1 = CH4C;
    dcomplex * V2 = V1 + 4*NP*NP + NP;

    if(std::is_same<IntsT,double>::value) {
      SetMatRE('N',NP,NP,1.,reinterpret_cast<double*>(P2P),NP,V1,4*NP);
      SetMatRE('N',NP,NP,1.,reinterpret_cast<double*>(P2P),NP,V2,4*NP);
    } else {
      SetMat('N',NP,NP,dcomplex(1.),reinterpret_cast<dcomplex*>(P2P),NP,V1,4*NP);
      SetMat('N',NP,NP,dcomplex(1.),reinterpret_cast<dcomplex*>(P2P),NP,V2,4*NP);
    }

    // Set the diagonal cp blocks of CH
    // CP = [cp 0  ]
    //      [0  cp ]
    dcomplex *CP11 = CH4C + 8*NP*NP;
    dcomplex *CP12 = CP11 + 4*NP*NP + NP;
    dcomplex *CP21 = CH4C + 2*NP;
    dcomplex *CP22 = CP21 + 4*NP*NP + NP;
   
    for(auto j = 0; j < NP; j++) {
      CP11[j + 4*NP*j] = SpeedOfLight * SS[j];
      CP12[j + 4*NP*j] = SpeedOfLight * SS[j];
      CP21[j + 4*NP*j] = SpeedOfLight * SS[j];
      CP22[j + 4*NP*j] = SpeedOfLight * SS[j];
    }

    // Diagonalize the 4C CH
    double *CHEV = memManager.malloc<double>(4*NP);

    HermetianEigen('V','U',4*NP,CH4C,4*NP,CHEV,memManager);


    // Get pointers to "L" and "S" components of eigenvectors
    dcomplex *L = CH4C + 8*NP*NP;
    dcomplex *S = L + 2*NP;


    // Invert "L"; L -> L^-1
    LUInv(2*NP,L,4*NP,memManager);


    // Reuse the charge conjugated space for X and Y
    dcomplex *X = CH4C;
    dcomplex *Y = X + 2*NP;

    // Form X = S * L^-1
    Gemm('N','N',2*NP,2*NP,2*NP,dcomplex(1.),S,4*NP,L,4*NP,
      dcomplex(0.),X,4*NP);

    // Form Y = sqrt(1 + X**H * H)
      
    // Y = X**H * H
    Gemm('C','N',2*NP,2*NP,2*NP,dcomplex(1.),X,4*NP,X,4*NP,
      dcomplex(0.),Y,4*NP);

    // Y = Y + I
    for(auto j = 0; j < 2*NP; j++) Y[j + 4*NP*j] += 1.0;

    // Y -> V * y * V**H 
    // XXX: Store the eigenvalues of Y in CHEV
    HermetianEigen('V','U',2*NP,Y,4*NP,CHEV,memManager);

    // SCR1 -> V * y^-0.25
    for(auto j = 0ul; j < 2*NP; j++)
    for(auto i = 0ul; i < 2*NP; i++)
      CSCR1[i + 2*NP*j] = Y[i + 4*NP*j] * std::pow(CHEV[j],-0.25);

    // Y = SCR1 * SCR1**H
    Gemm('N','C',2*NP,2*NP,2*NP,dcomplex(1.),CSCR1,2*NP,CSCR1,2*NP,
      dcomplex(0.),Y,4*NP);
    
    // Build the effective two component CH in "L"
    dcomplex *FullCH2C = L;
 
    // Zero it out
    for(auto j = 0; j < 2*NP; j++)
    for(auto i = 0; i < 2*NP; i++)
      FullCH2C[i + 4*NP*j] = 0.;

    // Copy P2P into spin diagonal blocks of 2C CH
    dcomplex *CH2C1 = FullCH2C;
    dcomplex *CH2C2 = CH2C1 + 4*NP*NP + NP;

    if(std::is_same<IntsT,double>::value) {
      SetMatRE('N',NP,NP,1.,reinterpret_cast<double*>(P2P),NP,CH2C1,4*NP);
      SetMatRE('N',NP,NP,1.,reinterpret_cast<double*>(P2P),NP,CH2C2,4*NP);
    } else {
      SetMat('N',NP,NP,dcomplex(1.),reinterpret_cast<dcomplex*>(P2P),NP,CH2C1,4*NP);
      SetMat('N',NP,NP,dcomplex(1.),reinterpret_cast<dcomplex*>(P2P),NP,CH2C2,4*NP);
    }

    // Construct 2C CH in the uncontracted basis
    // 2C CH = Y * (V' + cp * X + X**H * cp + X**H * W' * X) * Y

    // SCR1 = cp * X
    for(auto j = 0; j < 2*NP; j++)
    for(auto i = 0; i < NP; i++) {
      CSCR1[i + 2*NP*j] = SpeedOfLight * SS[i] * X[i + 4*NP*j];
      CSCR1[i + NP + 2*NP*j] = SpeedOfLight * SS[i] * X[i + NP + 4*NP*j];
    }

    // 2C CH += SCR1 + SCR1**H
    MatAdd('N','N',2*NP,2*NP,dcomplex(1.),FullCH2C,4*NP,dcomplex(1.),
      CSCR1,2*NP, FullCH2C,4*NP);
    MatAdd('N','C',2*NP,2*NP,dcomplex(1.),FullCH2C,4*NP,dcomplex(1.),
      CSCR1,2*NP, FullCH2C,4*NP);


    // SCR1 = X**H * W
    Gemm('C','N',2*NP,2*NP,2*NP,dcomplex(1.),X,4*NP,W,LDW,dcomplex(0.),
      CSCR1,2*NP);
    
    // 2C CH += SCR1 * X
    Gemm('N','N',2*NP,2*NP,2*NP,dcomplex(1.),CSCR1,2*NP,X,4*NP,
      dcomplex(1.),FullCH2C,4*NP);

    // SCR1 = CH2C * Y
    Gemm('C','N',2*NP,2*NP,2*NP,dcomplex(1.),FullCH2C,4*NP,Y,4*NP,dcomplex(0.),
      CSCR1,2*NP);


    // 2C CH = Y * SCR1
    Gemm('N','N',2*NP,2*NP,2*NP,dcomplex(1.),Y,4*NP,CSCR1,2*NP,
      dcomplex(0.),FullCH2C,4*NP);

    // Allocate memory for the uncontracted spin components 
    // of the 2C CH
    dcomplex *HUnS = memManager.malloc<dcomplex>(NP*NP);
    dcomplex *HUnZ = memManager.malloc<dcomplex>(NP*NP);
    dcomplex *HUnX = memManager.malloc<dcomplex>(NP*NP);
    dcomplex *HUnY = memManager.malloc<dcomplex>(NP*NP);

    SpinScatter(NP,FullCH2C,4*NP,HUnS,NP,HUnZ,NP,HUnY,NP,HUnX,NP);

    // Partition the scratch space into one complex and one real NP x NP 
    // matrix
    IntsT   * SUK   = SCR1;
    dcomplex * CSCR2 = reinterpret_cast<dcomplex*>(SUK + NP*NP);

    // Store the Product of S and UK
    Gemm('N','N',NP,NP,NP,IntsT(1.),SCPY,NP,UK,NP,IntsT(0.),SCR1,NP); 

    // Transform the spin components of the 2C CH into R-space
    //
    // H(k) -> SUK * H(k) * (SUK)**H
    //
    // ** Using the fact that H(k) is hermetian
    // CSCR2 = SUK * H(k) -> CSCR2**H = H(k) * (SUK)**H
    // H(k) -> SUK * CSCR2**H
    //

    // Transform H(S)
    Gemm('N','N',NP,NP,NP,dcomplex(1.),SUK,NP,HUnS,NP,dcomplex(0.),
      CSCR2,NP);
    Gemm('N','C',NP,NP,NP,dcomplex(1.),SUK,NP,CSCR2,NP,dcomplex(0.),
      HUnS,NP);

    // Transform H(Z)
    Gemm('N','N',NP,NP,NP,dcomplex(1.),SUK,NP,HUnZ,NP,dcomplex(0.),
      CSCR2,NP);
    Gemm('N','C',NP,NP,NP,dcomplex(1.),SUK,NP,CSCR2,NP,dcomplex(0.),
      HUnZ,NP);

    // Transform H(Y)
    Gemm('N','N',NP,NP,NP,dcomplex(1.),SUK,NP,HUnY,NP,dcomplex(0.),
      CSCR2,NP);
    Gemm('N','C',NP,NP,NP,dcomplex(1.),SUK,NP,CSCR2,NP,dcomplex(0.),
      HUnY,NP);

    // Transform H(X)
    Gemm('N','N',NP,NP,NP,dcomplex(1.),SUK,NP,HUnX,NP,dcomplex(0.),
      CSCR2,NP);
    Gemm('N','C',NP,NP,NP,dcomplex(1.),SUK,NP,CSCR2,NP,dcomplex(0.),
      HUnX,NP);

    // Transform H(k) into the contracted basis

    Gemm('N','N',NB,NP,NP,dcomplex(1.),mapPrim2Cont,NB,HUnS,
      NP,dcomplex(0.),CSCR1,NB);
    Gemm('N','C',NB,NB,NP,dcomplex(1.),mapPrim2Cont,NB,CSCR1,
      NB,dcomplex(0.),HUnS,NB);

    Gemm('N','N',NB,NP,NP,dcomplex(1.),mapPrim2Cont,NB,HUnZ,
      NP,dcomplex(0.),CSCR1,NB);
    Gemm('N','C',NB,NB,NP,dcomplex(1.),mapPrim2Cont,NB,CSCR1,
      NB,dcomplex(0.),HUnZ,NB);

    Gemm('N','N',NB,NP,NP,dcomplex(1.),mapPrim2Cont,NB,HUnY,
      NP,dcomplex(0.),CSCR1,NB);
    Gemm('N','C',NB,NB,NP,dcomplex(1.),mapPrim2Cont,NB,CSCR1,
      NB,dcomplex(0.),HUnY,NB);

    Gemm('N','N',NB,NP,NP,dcomplex(1.),mapPrim2Cont,NB,HUnX,
      NP,dcomplex(0.),CSCR1,NB);
    Gemm('N','C',NB,NB,NP,dcomplex(1.),mapPrim2Cont,NB,CSCR1,
      NB,dcomplex(0.),HUnX,NB);


    size_t n1, n2;
    std::array<double,4> Ql={0.,2.,10.,28.};

    if( this->aoints.basisSet().maxL > 3 ) CErr("Boettger scaling for L > 3 NYI");

    for(auto s1(0ul), i(0ul); s1 < this->aoints.basisSet().nShell; s1++, i+=n1) {
      n1 = this->aoints.basisSet().shells[s1].size();
      
      size_t L1 = this->aoints.basisSet().shells[s1].contr[0].l;
      if ( L1 == 0 ) continue;

      size_t Z1 =
        this->aoints.molecule().atoms[this->aoints.basisSet().mapSh2Cen[s1]].atomicNumber;


    for(auto s2(0ul), j(0ul); s2 < this->aoints.basisSet().nShell; s2++, j+=n2) {
      n2 = this->aoints.basisSet().shells[s2].size();
      
      size_t L2 = this->aoints.basisSet().shells[s2].contr[0].l;
      if ( L2 == 0 ) continue;

      size_t Z2 =
        this->aoints.molecule().atoms[this->aoints.basisSet().mapSh2Cen[s2]].atomicNumber;
    
      dcomplex fudgeFactor = std::sqrt(
        Ql[L1] * Ql[L2] /
        Z1 / Z2
      );

      MatAdd('N','N',n1,n2,dcomplex(-1.),HUnZ + i + j*NB,NB,
        fudgeFactor,HUnZ + i + j*NB,NB, HUnZ + i + j*NB,NB);

      MatAdd('N','N',n1,n2,dcomplex(-1.),HUnY + i + j*NB,NB,
        fudgeFactor,HUnY + i + j*NB,NB, HUnY + i + j*NB,NB);

      MatAdd('N','N',n1,n2,dcomplex(-1.),HUnX + i + j*NB,NB,
        fudgeFactor,HUnX + i + j*NB,NB, HUnX + i + j*NB,NB);

    } // loop s2
    } // loop s1


//    GetMatRE('N',NB,NB,1.,HUnS,NB,CH[0],NB);
//    GetMatIM('N',NB,NB,1.,HUnZ,NB,CH[1],NB);
//    GetMatIM('N',NB,NB,1.,HUnY,NB,CH[2],NB);
//    GetMatIM('N',NB,NB,1.,HUnX,NB,CH[3],NB);
    std::copy_n(HUnS,NB*NB,CH[0]);
    std::copy_n(HUnZ,NB*NB,CH[1]);
    std::copy_n(HUnY,NB*NB,CH[2]);
    std::copy_n(HUnX,NB*NB,CH[3]);

    memManager.free(UK,SCR1);
  };

  template<> void SingleSlater<double,double>::computeX2CCH(
      EMPerturbation& emPert, std::vector<double*> &CH){


    CErr("X2C + Real WFN is not a valid option",std::cout);

  }

  template<> void SingleSlater<dcomplex,dcomplex>::computeX2CCH(
      EMPerturbation& emPert, std::vector<dcomplex*> &CH){


    CErr("X2C + Complex Ints NYI",std::cout);

  }


  template void SingleSlater<dcomplex,double>::computeX2CCH(
    EMPerturbation& emPert, std::vector<dcomplex*> &CH);


  /**
   *  \brief Compute the 4C Dirac Hamiltonian
   */ 
  template <typename MatsT, typename IntsT> void SingleSlater<MatsT,IntsT>::compute4CCH(std::vector<libint2::Shell> &shells, dcomplex* ) {


    CErr("4C NYI",std::cout);

  };

  template void SingleSlater<double,double>::compute4CCH(std::vector<libint2::Shell> &shells    , dcomplex* CH);
  template void SingleSlater<dcomplex,double>::compute4CCH(std::vector<libint2::Shell> &shells  , dcomplex* CH);
  template void SingleSlater<dcomplex,dcomplex>::compute4CCH(std::vector<libint2::Shell> &shells, dcomplex* CH);

}; // namespace ChronusQ

