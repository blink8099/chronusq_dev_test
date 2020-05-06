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
#include <corehbuilder/x2c.hpp>
#include <physcon.hpp>
#include <cqlinalg.hpp>
#include <cqlinalg/svd.hpp>

namespace ChronusQ {

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
  }

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
  }

  /**
   *  \brief Compute one-electron integrals
   */
  template <typename MatsT, typename IntsT>
  void X2C<MatsT, IntsT>::computeAOOneE(EMPerturbation &emPert) {
    this->aoints_.computeAOOneE(emPert,this->oneETerms_); // compute the necessary 1e ints
  }

  template void X2C<double,double>::computeAOOneE(EMPerturbation&);
  template void X2C<dcomplex,double>::computeAOOneE(EMPerturbation&);
  template void X2C<dcomplex,dcomplex>::computeAOOneE(EMPerturbation&);

  /**
   *  \brief Compute the X2C Core Hamiltonian
   */
  template <typename MatsT, typename IntsT>
  void X2C<MatsT, IntsT>::computeX2C(EMPerturbation& emPert, std::vector<MatsT*> &CH) {
    IntsT* XXX = reinterpret_cast<IntsT*>(NULL);

    size_t NP = uncontractedBasis_.nPrimitive;
    size_t NB = basisSet_.nBasis;

    uncontractedInts_.computeAOOneE(emPert,this->oneETerms_); // FIXME: need to compute SL

    // Make copy of integrals
    IntsT *overlap   = memManager_.malloc<IntsT>(NP*NP);
    std::copy_n(uncontractedInts_.overlap, NP*NP, overlap);
    IntsT *kinetic   = memManager_.malloc<IntsT>(NP*NP);
    std::copy_n(uncontractedInts_.kinetic, NP*NP, kinetic);
    IntsT *potential = memManager_.malloc<IntsT>(NP*NP);
    std::copy_n(uncontractedInts_.potential, NP*NP, potential);
    IntsT *PVdotP    = memManager_.malloc<IntsT>(NP*NP);
    std::copy_n(uncontractedInts_.PVdotP, NP*NP, PVdotP);
    std::vector<IntsT *> PVcrossP(3);
    for(size_t i = 0; i < 3; i++) {
      PVcrossP[i] = memManager_.malloc<IntsT>(NP*NP);
      std::copy_n(uncontractedInts_.PVcrossP[i], NP*NP, PVcrossP[i]);
    }

    // Compute the mappings from primitives to CGTOs
    mapPrim2Cont = memManager_.malloc<IntsT>(NP*NB);
    basisSet_.makeMapPrim2Cont(overlap,
      mapPrim2Cont,memManager_);

    // Transformation matrix
    UK = memManager_.malloc<IntsT>(NP*NP);

    // Allocate Scratch Space (enough for 2*NP x 2*NP complex matricies)
    IntsT   *SCR1  = memManager_.malloc<IntsT>(8*NP*NP);
    dcomplex *CSCR1 = reinterpret_cast<dcomplex*>(SCR1);

    // Singular value storage (initially S then T)
    p = memManager_.malloc<double>(NP);
    IntsT* SS = p;

    // Get SVD of uncontracted overlap
    // Store the left singular vectors in S
    SVD('O','N',NP,NP,overlap,NP,SS,XXX,NP,
      XXX,NP,memManager_);

    double minSS = *std::min_element(SS,SS+NP);

    if( minSS < 1e-10 ) CErr("Uncontracted Overlap is Singular");

    // Form orthonormal transformation matrix in S
    for(auto i = 0ul; i < NP; i++)
      Scale(NP,IntsT(1.)/std::sqrt(SS[i]),
          overlap + i*NP,1);

    // Transform T into the orthonormal basis
    // T -> TO
    Gemm('C','N',NP,NP,NP,IntsT(1.),overlap,NP,
        kinetic,NP,IntsT(0.),SCR1,NP);
    Gemm('N','N',NP,NP,NP,IntsT(1.),SCR1,NP,overlap,NP,
        IntsT(0.),kinetic,NP);

    // Get the SVD of TO
    // Store the left singular vectors in TO
    SVD('O','N',NP,NP,kinetic,NP,SS,XXX,NP,
      XXX,NP,memManager_);

    minSS = *std::min_element(SS,SS+NP);
    if( minSS < 1e-10 ) CErr("Uncontracted Kinetic Energy Tensor is Singular");

    // Form UK = S * T
    Gemm('N','N',NP,NP,NP,IntsT(1.),overlap,NP,
      kinetic,NP,IntsT(0.),UK,NP);

    // Allocate and for "P^2" potential
    IntsT *P2P = memManager_.malloc<IntsT>(NP*NP);

    // P2P = UK**T * V * UK
    Gemm('C','N',NP,NP,NP,IntsT(1.),UK,NP,potential,NP,
        IntsT(0.),SCR1,NP);
    Gemm('N','N',NP,NP,NP,IntsT(1.),SCR1,NP,UK,NP,IntsT(0.),P2P,NP);

    // Transform PVP into the "P^2" basis
    Gemm('C','N',NP,NP,NP,IntsT(1.),UK,NP,PVdotP,NP,
        IntsT(0.),SCR1,NP);
    Gemm('N','N',NP,NP,NP,IntsT(1.),SCR1,NP,UK,NP,
        IntsT(0.),PVdotP,NP);

    // Loop over PVxP terms
    for(auto & SL : PVcrossP ){
      Gemm('C','N',NP,NP,NP,IntsT(1.),UK,NP,SL,NP  ,IntsT(0.),SCR1,NP);
      Gemm('N','N',NP,NP,NP,IntsT(1.),SCR1,NP,UK,NP,IntsT(0.),SL,NP);
    }

    // P^2 -> P^-1
    for(auto i = 0; i < NP; i++) SS[i] = 1./std::sqrt(2*SS[i]);

    // Transform PVP into "P^-1" basis
    for(auto j = 0; j < NP; j++)
    for(auto i = 0; i < NP; i++){
      PVdotP[i + j*NP] *= SS[i] * SS[j];
      for(auto &SL : PVcrossP)
        SL[i + j*NP] *= SS[i] * SS[j];
    }

    // Allocate 4C CORE Hamiltonian

    // CH = [ V    cp       ]
    //      [ cp   W - 2mc^2]
    dcomplex *CH4C = memManager_.malloc<dcomplex>(16*NP*NP);
    memset(CH4C,0,16*NP*NP*sizeof(dcomplex));

    // Allocate W separately  as it's needed later
    size_t LDW = 2*NP;
    dcomplex *Wp  = memManager_.malloc<dcomplex>(LDW*LDW);

    formW(NP,Wp,LDW,PVdotP,NP,
        PVcrossP[2],NP,
        PVcrossP[1],NP,
        PVcrossP[0],NP);

    // Subtract out 2mc^2 from W diagonals
    const double WFact = 2. * SpeedOfLight * SpeedOfLight;
    for(auto j = 0ul; j < 2*NP; j++) Wp[j + LDW*j] -= WFact;

    // Copy W into the 4C CH storage
    dcomplex *CHW = CH4C + 8*NP*NP + 2*NP;
    SetMat('N',2*NP,2*NP,dcomplex(1.),Wp,LDW,CHW,4*NP);

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
    double *CHEV = memManager_.malloc<double>(4*NP);

    HermetianEigen('V','U',4*NP,CH4C,4*NP,CHEV,memManager_);


    // Get pointers to "L" and "S" components of eigenvectors
    dcomplex *L = CH4C + 8*NP*NP;
    dcomplex *S = L + 2*NP;


    // Invert "L"; L -> L^-1
    LUInv(2*NP,L,4*NP,memManager_);


    // Reuse the charge conjugated space for X and Y
    X = memManager_.malloc<dcomplex>(4*NP*NP);
    memset(X,0,4*NP*NP*sizeof(dcomplex));
    Y = memManager_.malloc<dcomplex>(4*NP*NP);
    memset(Y,0,4*NP*NP*sizeof(dcomplex));

    // Form X = S * L^-1
    Gemm('N','N',2*NP,2*NP,2*NP,dcomplex(1.),S,4*NP,L,4*NP,
      dcomplex(0.),X,2*NP);

    // Form Y = sqrt(1 + X**H * X)

    // Y = X**H * X
    Gemm('C','N',2*NP,2*NP,2*NP,dcomplex(1.),X,2*NP,X,2*NP,
      dcomplex(0.),Y,2*NP);

    // Y = Y + I
    for(auto j = 0; j < 2*NP; j++) Y[j + 2*NP*j] += 1.0;

    // Y -> V * y * V**H
    // XXX: Store the eigenvalues of Y in CHEV
    HermetianEigen('V','U',2*NP,Y,2*NP,CHEV,memManager_);

    // SCR1 -> V * y^-0.25
    for(auto j = 0ul; j < 2*NP; j++)
    for(auto i = 0ul; i < 2*NP; i++)
      CSCR1[i + 2*NP*j] = Y[i + 2*NP*j] * std::pow(CHEV[j],-0.25);

    // Y = SCR1 * SCR1**H
    Gemm('N','C',2*NP,2*NP,2*NP,dcomplex(1.),CSCR1,2*NP,CSCR1,2*NP,
      dcomplex(0.),Y,2*NP);

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
      CSCR1[i + 2*NP*j] = SpeedOfLight * SS[i] * X[i + 2*NP*j];
      CSCR1[i + NP + 2*NP*j] = SpeedOfLight * SS[i] * X[i + NP + 2*NP*j];
    }

    // 2C CH += SCR1 + SCR1**H
    MatAdd('N','N',2*NP,2*NP,dcomplex(1.),FullCH2C,4*NP,dcomplex(1.),
      CSCR1,2*NP, FullCH2C,4*NP);
    MatAdd('N','C',2*NP,2*NP,dcomplex(1.),FullCH2C,4*NP,dcomplex(1.),
      CSCR1,2*NP, FullCH2C,4*NP);


    // SCR1 = X**H * W
    Gemm('C','N',2*NP,2*NP,2*NP,dcomplex(1.),X,2*NP,Wp,LDW,dcomplex(0.),
      CSCR1,2*NP);

    // 2C CH += SCR1 * X
    Gemm('N','N',2*NP,2*NP,2*NP,dcomplex(1.),CSCR1,2*NP,X,2*NP,
      dcomplex(1.),FullCH2C,4*NP);

    // SCR1 = CH2C * Y
    Gemm('C','N',2*NP,2*NP,2*NP,dcomplex(1.),FullCH2C,4*NP,Y,2*NP,dcomplex(0.),
      CSCR1,2*NP);


    // 2C CH = Y * SCR1
    Gemm('N','N',2*NP,2*NP,2*NP,dcomplex(1.),Y,2*NP,CSCR1,2*NP,
      dcomplex(0.),FullCH2C,4*NP);

    // Allocate memory for the uncontracted spin components
    // of the 2C CH
    dcomplex *HUnS = memManager_.malloc<dcomplex>(NP*NP);
    dcomplex *HUnZ = memManager_.malloc<dcomplex>(NP*NP);
    dcomplex *HUnX = memManager_.malloc<dcomplex>(NP*NP);
    dcomplex *HUnY = memManager_.malloc<dcomplex>(NP*NP);

    SpinScatter(NP,FullCH2C,4*NP,HUnS,NP,HUnZ,NP,HUnY,NP,HUnX,NP);

    // Partition the scratch space into one complex and one real NP x NP
    // matrix
    IntsT   * SUK   = SCR1;
    dcomplex * CSCR2 = reinterpret_cast<dcomplex*>(SUK + NP*NP);

    // Store the Product of S and UK
    Gemm('N','N',NP,NP,NP,IntsT(1.),uncontractedInts_.overlap,NP,UK,NP,IntsT(0.),SCR1,NP);

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
    std::array<double,6> Ql={0.,2.,10.,28.,60.,110.};

    if( basisSet_.maxL > 5 ) CErr("Boettger scaling for L > 5 NYI");

    for(auto s1(0ul), i(0ul); s1 < basisSet_.nShell; s1++, i+=n1) {
      n1 = basisSet_.shells[s1].size();

      size_t L1 = basisSet_.shells[s1].contr[0].l;
      if ( L1 == 0 ) continue;

      size_t Z1 = molecule_.atoms[basisSet_.mapSh2Cen[s1]].atomicNumber;


    for(auto s2(0ul), j(0ul); s2 < basisSet_.nShell; s2++, j+=n2) {
      n2 = basisSet_.shells[s2].size();

      size_t L2 = basisSet_.shells[s2].contr[0].l;
      if ( L2 == 0 ) continue;

      size_t Z2 = molecule_.atoms[basisSet_.mapSh2Cen[s2]].atomicNumber;

      dcomplex fudgeFactor = -1 * std::sqrt(
        Ql[L1] * Ql[L2] /
        Z1 / Z2
      );

      MatAdd('N','N',n1,n2,dcomplex(1.),HUnZ + i + j*NB,NB,
        fudgeFactor,HUnZ + i + j*NB,NB, HUnZ + i + j*NB,NB);

      MatAdd('N','N',n1,n2,dcomplex(1.),HUnY + i + j*NB,NB,
        fudgeFactor,HUnY + i + j*NB,NB, HUnY + i + j*NB,NB);

      MatAdd('N','N',n1,n2,dcomplex(1.),HUnX + i + j*NB,NB,
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

    memManager_.free(overlap, kinetic, potential,
      PVdotP, PVcrossP[0], PVcrossP[1], PVcrossP[2],
      SCR1, P2P, CH4C, Wp, CHEV, HUnS, HUnZ, HUnX, HUnY);
  }

  template void X2C<dcomplex,double>::computeX2C(EMPerturbation&, std::vector<dcomplex*>&);

  template<> void X2C<dcomplex,dcomplex>::computeX2C(EMPerturbation&, std::vector<dcomplex*>&) {
    CErr("X2C + Complex Ints NYI",std::cout);
  }

  template<> void X2C<double,double>::computeX2C(EMPerturbation&, std::vector<double*>&) {
    CErr("X2C + Real WFN is not a valid option",std::cout);
  }


  /**
   *  \brief Compute the X2C Core Hamiltonian
   */
  template <typename MatsT, typename IntsT>
  void X2C<MatsT, IntsT>::computeCoreH(EMPerturbation& emPert, std::vector<MatsT*> &CH) {
    computeAOOneE(emPert);
    computeX2C(emPert, CH);
  }

  template void X2C<dcomplex,double>::computeCoreH(EMPerturbation&, std::vector<dcomplex*>&);

  template<> void X2C<dcomplex,dcomplex>::computeCoreH(EMPerturbation&, std::vector<dcomplex*>&) {
    CErr("X2C + Complex Ints NYI",std::cout);
  }

  template<> void X2C<double,double>::computeCoreH(EMPerturbation&, std::vector<double*>&) {
    CErr("X2C + Real WFN is not a valid option",std::cout);
  }

  /**
   *  \brief Compute the picture change matrices UL, US
   */
  template <typename MatsT, typename IntsT>
  void X2C<MatsT, IntsT>::computeU() {

    size_t NP = uncontractedBasis_.nPrimitive;
    size_t NB = basisSet_.nBasis;

    // UL = UK * Y * UK^-1 * UP2C
    // US = 2 * SpeedOfLight * UK * p^-1 * X * Y * UK^-1 * UP2C

    // 1.  UP2CSUK = UP2C * S * UK
    IntsT *UP2CS = memManager_.malloc<IntsT>(NB*NP);
    Gemm('N','N',NB,NP,NP,IntsT(1.),mapPrim2Cont,NB,
      uncontractedInts_.overlap,NP,IntsT(0.),UP2CS,NB);
    IntsT *UP2CSUK = memManager_.malloc<IntsT>(4*NP*NP);
    memset(UP2CSUK,0,4*NB*NP*sizeof(IntsT));
    Gemm('N','N',NB,NP,NP,IntsT(1.),UP2CS,NB,UK,NP,IntsT(0.),UP2CSUK,2*NB);
    if(std::is_same<IntsT,double>::value) {
      SetMatRE('N',NB,NP,1.,reinterpret_cast<double*>(UP2CSUK),2*NB,
        reinterpret_cast<double*>(UP2CSUK + 2*NB*NP + NB),2*NB);
    } else {
      SetMat('N',NB,NP,dcomplex(1.),reinterpret_cast<dcomplex*>(UP2CSUK),2*NB,
        reinterpret_cast<dcomplex*>(UP2CSUK + 2*NB*NP + NB),2*NB);
    }

    // 2. R^T = UP2C * S * UK * Y^T
    dcomplex *RT = memManager_.malloc<dcomplex>(4*NB*NP);
    Gemm('N','C',2*NB,2*NP,2*NP,dcomplex(1.),UP2CSUK,2*NB,
      Y,2*NP,dcomplex(0.),RT,2*NB);

    // 3. Xp = 2 c p^-1 X
    double twoC = 2 * SpeedOfLight;
    double *twoCPinv = memManager_.malloc<double>(NP);
    for(size_t i = 0; i < NP; i++) twoCPinv[i] = twoC/p[i];
    dcomplex *twoCPinvX = memManager_.malloc<dcomplex>(4*NP*NP);
    for(size_t j = 0; j < 2*NP; j++)
    for(size_t i = 0; i < NP; i++) {
      twoCPinvX[i + 2*NP*j] = twoCPinv[i] * X[i + 2*NP*j];
      twoCPinvX[i + NP + 2*NP*j] = twoCPinv[i] * X[i + NP + 2*NP*j];
    }

    // 4. UK2c = [ UK  0  ]
    //           [ 0   UK ]
    IntsT *UK2c = UP2CSUK;
    memset(UP2CSUK,0,4*NP*NP*sizeof(IntsT));
    IntsT *UK2c2 = UK2c + 2*NP*NP + NP;
    if(std::is_same<IntsT,double>::value) {
      SetMatRE('N',NP,NP,1.,reinterpret_cast<double*>(UK),NP,UK2c,2*NP);
      SetMatRE('N',NP,NP,1.,reinterpret_cast<double*>(UK),NP,UK2c2,2*NP);
    } else {
      SetMat('N',NP,NP,dcomplex(1.),reinterpret_cast<dcomplex*>(UK),NP,UK2c,2*NP);
      SetMat('N',NP,NP,dcomplex(1.),reinterpret_cast<dcomplex*>(UK),NP,UK2c2,2*NP);
    }

    // 5. US = UK2c * Xp * RT^T
    UL = memManager_.malloc<dcomplex>(4*NP*NB);
    US = memManager_.malloc<dcomplex>(4*NP*NB);
    Gemm('N','C',2*NP,2*NB,2*NP,dcomplex(1.),twoCPinvX,2*NP,
      RT,2*NB,dcomplex(0.),UL,2*NP);
    Gemm('N','N',2*NP,2*NB,2*NP,dcomplex(1.),UK2c,2*NP,
      UL,2*NP,dcomplex(0.),US,2*NP);

    // 6. UL = UK * RT^T
    Gemm('N','C',2*NP,2*NB,2*NP,dcomplex(1.),UK2c,2*NP,
      RT,2*NB,dcomplex(0.),UL,2*NP);

    memManager_.free(UP2CS, UP2CSUK, RT, twoCPinv, twoCPinvX);

  }

  template void X2C<dcomplex,double>::computeU();

  template<> void X2C<dcomplex,dcomplex>::computeU() {
    CErr("X2C + Complex Ints NYI",std::cout);
  }

  template<> void X2C<double,double>::computeU() {
    CErr("X2C + Real WFN is not a valid option",std::cout);
  }

  /**
   *  \brief Compute the X2C Core Hamiltonian
   */
  template <typename MatsT, typename IntsT>
  void X2C<MatsT, IntsT>::computeX2C_UDU(EMPerturbation& emPert, std::vector<MatsT*> &CH) {
    IntsT* XXX = reinterpret_cast<IntsT*>(NULL);

    size_t NP = uncontractedBasis_.nPrimitive;
    size_t NB = basisSet_.nBasis;

    // Make copy of integrals
    IntsT *overlap   = memManager_.malloc<IntsT>(NP*NP);
    std::copy_n(uncontractedInts_.overlap, NP*NP, overlap);
    IntsT *kinetic   = memManager_.malloc<IntsT>(NP*NP);
    std::copy_n(uncontractedInts_.kinetic, NP*NP, kinetic);
    IntsT *potential = memManager_.malloc<IntsT>(NP*NP);
    std::copy_n(uncontractedInts_.potential, NP*NP, potential);
    IntsT *PVdotP    = memManager_.malloc<IntsT>(NP*NP);
    std::copy_n(uncontractedInts_.PVdotP, NP*NP, PVdotP);
    std::vector<IntsT *> PVcrossP(3);
    for(size_t i = 0; i < 3; i++) {
      PVcrossP[i] = memManager_.malloc<IntsT>(NP*NP);
      std::copy_n(uncontractedInts_.PVcrossP[i], NP*NP, PVcrossP[i]);
    }

    // Allocate W separately  as it's needed later
    size_t LDW = 2*NP;
    W = memManager_.malloc<dcomplex>(LDW*LDW);

    formW(NP,W,LDW,PVdotP,NP,
        PVcrossP[2],NP,
        PVcrossP[1],NP,
        PVcrossP[0],NP);

    // T2c = [ T  0 ]
    //       [ 0  T ]
    IntsT *T2c = memManager_.malloc<IntsT>(LDW*LDW);
    memset(T2c,0,4*NP*NP*sizeof(IntsT));
    IntsT *T2c2 = T2c + 2*NP*NP + NP;
    if(std::is_same<IntsT,double>::value) {
      SetMatRE('N',NP,NP,1.,reinterpret_cast<double*>(kinetic),NP,T2c,2*NP);
      SetMatRE('N',NP,NP,1.,reinterpret_cast<double*>(kinetic),NP,T2c2,2*NP);
    } else {
      SetMat('N',NP,NP,dcomplex(1.),reinterpret_cast<dcomplex*>(kinetic),NP,T2c,2*NP);
      SetMat('N',NP,NP,dcomplex(1.),reinterpret_cast<dcomplex*>(kinetic),NP,T2c2,2*NP);
    }

    // V2c = [ V  0 ]
    //       [ 0  V ]
    IntsT *V2c = memManager_.malloc<IntsT>(LDW*LDW);
    memset(V2c,0,4*NP*NP*sizeof(IntsT));
    IntsT *V2c2 = V2c + 2*NP*NP + NP;
    if(std::is_same<IntsT,double>::value) {
      SetMatRE('N',NP,NP,1.,reinterpret_cast<double*>(potential),NP,V2c,2*NP);
      SetMatRE('N',NP,NP,1.,reinterpret_cast<double*>(potential),NP,V2c2,2*NP);
    } else {
      SetMat('N',NP,NP,dcomplex(1.),reinterpret_cast<dcomplex*>(potential),NP,V2c,2*NP);
      SetMat('N',NP,NP,dcomplex(1.),reinterpret_cast<dcomplex*>(potential),NP,V2c2,2*NP);
    }

    dcomplex *Hx2c = memManager_.malloc<dcomplex>(4*NB*NB);
    memset(Hx2c,0,4*NB*NB*sizeof(dcomplex));

    dcomplex *SCR = memManager_.malloc<dcomplex>(4*NP*NB);

    // Hx2c += UL^H * T2c * US
    Gemm('N','N',2*NP,2*NB,2*NP,dcomplex(1.),T2c,2*NP,
      US,2*NP,dcomplex(0.),SCR,2*NP);
    Gemm('C','N',2*NB,2*NB,2*NP,dcomplex(1.),UL,2*NP,
      SCR,2*NP,dcomplex(1.),Hx2c,2*NB);
    // Hx2c += US^H * T2c * UL
    Gemm('N','N',2*NP,2*NB,2*NP,dcomplex(1.),T2c,2*NP,
      UL,2*NP,dcomplex(0.),SCR,2*NP);
    Gemm('C','N',2*NB,2*NB,2*NP,dcomplex(1.),US,2*NP,
      SCR,2*NP,dcomplex(1.),Hx2c,2*NB);
    // Hx2c -= US^H * T2c * US
    Gemm('N','N',2*NP,2*NB,2*NP,dcomplex(1.),T2c,2*NP,
      US,2*NP,dcomplex(0.),SCR,2*NP);
    Gemm('C','N',2*NB,2*NB,2*NP,dcomplex(-1.),US,2*NP,
      SCR,2*NP,dcomplex(1.),Hx2c,2*NB);
    // Hx2c += UL^H * V2c * UL
    Gemm('N','N',2*NP,2*NB,2*NP,dcomplex(1.),V2c,2*NP,
      UL,2*NP,dcomplex(0.),SCR,2*NP);
    Gemm('C','N',2*NB,2*NB,2*NP,dcomplex(1.),UL,2*NP,
      SCR,2*NP,dcomplex(1.),Hx2c,2*NB);
    // Hx2c += 1/(4*C**2) US^H * W * US
    Gemm('N','N',2*NP,2*NB,2*NP,
      dcomplex(0.25/SpeedOfLight/SpeedOfLight),W,2*NP,
      US,2*NP,dcomplex(0.),SCR,2*NP);
    Gemm('C','N',2*NB,2*NB,2*NP,dcomplex(1.),US,2*NP,
      SCR,2*NP,dcomplex(1.),Hx2c,2*NB);

    // Allocate memory for the uncontracted spin components
    // of the 2C CH
    dcomplex *HUnS = memManager_.malloc<dcomplex>(NB*NB);
    dcomplex *HUnZ = memManager_.malloc<dcomplex>(NB*NB);
    dcomplex *HUnX = memManager_.malloc<dcomplex>(NB*NB);
    dcomplex *HUnY = memManager_.malloc<dcomplex>(NB*NB);

    SpinScatter(NB,Hx2c,2*NB,HUnS,NB,HUnZ,NB,HUnY,NB,HUnX,NB);

    size_t n1, n2;
    std::array<double,6> Ql={0.,2.,10.,28.,60.,110.};

    if( basisSet_.maxL > 5 ) CErr("Boettger scaling for L > 5 NYI");

    for(auto s1(0ul), i(0ul); s1 < basisSet_.nShell; s1++, i+=n1) {
      n1 = basisSet_.shells[s1].size();

      size_t L1 = basisSet_.shells[s1].contr[0].l;
      if ( L1 == 0 ) continue;

      size_t Z1 = molecule_.atoms[basisSet_.mapSh2Cen[s1]].atomicNumber;


    for(auto s2(0ul), j(0ul); s2 < basisSet_.nShell; s2++, j+=n2) {
      n2 = basisSet_.shells[s2].size();

      size_t L2 = basisSet_.shells[s2].contr[0].l;
      if ( L2 == 0 ) continue;

      size_t Z2 = molecule_.atoms[basisSet_.mapSh2Cen[s2]].atomicNumber;

      dcomplex fudgeFactor = -1 * std::sqrt(
        Ql[L1] * Ql[L2] /
        Z1 / Z2
      );

      MatAdd('N','N',n1,n2,dcomplex(1.),HUnZ + i + j*NB,NB,
        fudgeFactor,HUnZ + i + j*NB,NB, HUnZ + i + j*NB,NB);

      MatAdd('N','N',n1,n2,dcomplex(1.),HUnY + i + j*NB,NB,
        fudgeFactor,HUnY + i + j*NB,NB, HUnY + i + j*NB,NB);

      MatAdd('N','N',n1,n2,dcomplex(1.),HUnX + i + j*NB,NB,
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

    memManager_.free(overlap, kinetic, potential,
      PVdotP, PVcrossP[0], PVcrossP[1], PVcrossP[2],
      T2c, V2c, Hx2c, SCR, HUnS, HUnZ, HUnX, HUnY);
  }

  template void X2C<dcomplex,double>::computeX2C_UDU(EMPerturbation&, std::vector<dcomplex*>&);

  template<> void X2C<dcomplex,dcomplex>::computeX2C_UDU(EMPerturbation&, std::vector<dcomplex*>&) {
    CErr("X2C + Complex Ints NYI",std::cout);
  }

  template<> void X2C<double,double>::computeX2C_UDU(EMPerturbation&, std::vector<double*>&) {
    CErr("X2C + Real WFN is not a valid option",std::cout);
  }

}; // namespace ChronusQ

