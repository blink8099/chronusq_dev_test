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
#include <corehbuilder/nonrel.hpp>
#include <physcon.hpp>
#include <cqlinalg.hpp>
#include <cqlinalg/svd.hpp>

namespace ChronusQ {

  template <>
  void formW(size_t NP, dcomplex *W, size_t LDW, dcomplex* pVdotP, size_t LDD, dcomplex* pVxPZ,
    size_t LDZ, dcomplex* pVxPY, size_t LDY, dcomplex* pVxPX, size_t LDX, bool scalarOnly) {

    // W = [ W1  W2 ]
    //     [ W3  W4 ]
    dcomplex *W1 = W;
    dcomplex *W2 = W1 + LDW*NP;
    dcomplex *W3 = W1 + NP;
    dcomplex *W4 = W2 + NP;

    if (scalarOnly) {
      SetMatDiag(NP,NP,pVdotP,LDD,W,LDW);
      return;
    }

    // W1 = pV.p + i (pVxp)(Z)
    MatAdd('N','N',NP,NP,dcomplex(1.),pVdotP,LDD,dcomplex(0.,1.),pVxPZ,LDZ,W1,LDW);
    // W4 = pV.p - i (pVxp)(Z)
    MatAdd('N','N',NP,NP,dcomplex(1.),pVdotP,LDD,dcomplex(0.,-1.),pVxPZ,LDZ,W4,LDW);
    // W2 = (pVxp)(Y) + i (pVxp)(X)
    MatAdd('N','N',NP,NP,dcomplex(1.),pVxPY,LDY,dcomplex(0.,1.),pVxPX,LDX,W2,LDW);
    // W3 = -(pVxp)(Y) + i (pVxp)(X)
    MatAdd('N','N',NP,NP,dcomplex(-1.),pVxPY,LDY,dcomplex(0.,1.),pVxPX,LDX,W3,LDW);
  }

  template <>
  void formW(size_t NP, dcomplex *W, size_t LDW, double* pVdotP, size_t LDD, double* pVxPZ,
    size_t LDZ, double* pVxPY, size_t LDY, double* pVxPX, size_t LDX, bool scalarOnly) {

    // W = [ W1  W2 ]
    //     [ W3  W4 ]
    dcomplex *W1 = W;
    dcomplex *W2 = W1 + LDW*NP;
    dcomplex *W3 = W1 + NP;
    dcomplex *W4 = W2 + NP;

    // Scalar part
    SetMatDiag(NP,NP,pVdotP,LDD,W,LDW);

    if (scalarOnly) return;

    // Spin-orbit part

    // W1 = i (pVxp)(Z)
    SetMatIM('N',NP,NP,1.,pVxPZ,LDZ,W1,LDW);

    // W4 = conj(W1)
    SetMatIM('N',NP,NP,-1.,pVxPZ,LDZ,W4,LDW);

    // W2 = (pVxp)(Y) + i (pVxp)(X)
    SetMatRE('N',NP,NP,1.,pVxPY,LDY,W2,LDW);
    SetMatIM('N',NP,NP,1.,pVxPX,LDX,W2,LDW);

    // W3 = -conj(W2)
    SetMatRE('N',NP,NP,-1.,pVxPY,LDY,W3,LDW);
    SetMatIM('N',NP,NP,1., pVxPX,LDX,W3,LDW);
  }

  template <>
  void formW(size_t NP, double *W, size_t LDW, double* pVdotP, size_t LDD, double* pVxPZ,
    size_t LDZ, double* pVxPY, size_t LDY, double* pVxPX, size_t LDX, bool scalarOnly) {

    if (not scalarOnly) CErr("SOX2C + Real WFN is not a valid option");

    // Scalar part
    SetMatDiag(NP,NP,pVdotP,LDD,W,LDW);
  }

  /**
   *  \brief Boettger scaling for spin-orbit operator
   */
  template <typename MatsT, typename IntsT>
  void X2C<MatsT, IntsT>::BoettgerScale(std::vector<MatsT*> &CH) {

    size_t NB = basisSet_.nBasis;

    size_t n1, n2;
    std::array<double,6> Ql={0.,2.,10.,28.,60.,110.};

    if( this->basisSet_.maxL > 5 ) CErr("Boettger scaling for L > 5 NYI");

    for(auto s1(0ul), i(0ul); s1 < this->basisSet_.nShell; s1++, i+=n1) {
      n1 = this->basisSet_.shells[s1].size();

      size_t L1 = this->basisSet_.shells[s1].contr[0].l;
      if ( L1 == 0 ) continue;

      size_t Z1 = this->molecule_.atoms[this->basisSet_.mapSh2Cen[s1]].atomicNumber;


    for(auto s2(0ul), j(0ul); s2 < this->basisSet_.nShell; s2++, j+=n2) {
      n2 = this->basisSet_.shells[s2].size();

      size_t L2 = this->basisSet_.shells[s2].contr[0].l;
      if ( L2 == 0 ) continue;

      size_t Z2 = this->molecule_.atoms[this->basisSet_.mapSh2Cen[s2]].atomicNumber;

      MatsT fudgeFactor = -1 * std::sqrt(
        Ql[L1] * Ql[L2] /
        Z1 / Z2
      );

      MatAdd('N','N',n1,n2,MatsT(1.),CH[1] + i + j*NB,NB,
        fudgeFactor,CH[1] + i + j*NB,NB, CH[1] + i + j*NB,NB);

      MatAdd('N','N',n1,n2,MatsT(1.),CH[2] + i + j*NB,NB,
        fudgeFactor,CH[2] + i + j*NB,NB, CH[2] + i + j*NB,NB);

      MatAdd('N','N',n1,n2,MatsT(1.),CH[3] + i + j*NB,NB,
        fudgeFactor,CH[3] + i + j*NB,NB, CH[3] + i + j*NB,NB);

    } // loop s2
    } // loop s1
  }

  /**
   *  \brief Compute the X2C Core Hamiltonian
   */
  template <typename MatsT, typename IntsT>
  void X2C<MatsT, IntsT>::computeX2C(EMPerturbation &emPert, std::vector<MatsT*> &CH) {
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
    if (this->oneETerms_.SORelativistic)
      for(size_t i = 0; i < 3; i++) {
        PVcrossP[i] = memManager_.malloc<IntsT>(NP*NP);
        std::copy_n(uncontractedInts_.PVcrossP[i], NP*NP, PVcrossP[i]);
      }

    // Compute the mappings from primitives to CGTOs
    mapPrim2Cont = memManager_.malloc<IntsT>(NP*NB);
    basisSet_.makeMapPrim2Cont(overlap,
      mapPrim2Cont,memManager_);

    // Allocate Scratch Space (enough for 2*NP x 2*NP complex matricies)
    IntsT   *SCR1  = memManager_.malloc<IntsT>(8*NP*NP);
    MatsT *CSCR1 = reinterpret_cast<MatsT*>(SCR1);

    // Singular value storage (initially S then T)
    p = memManager_.malloc<double>(NP);
    IntsT* SS = p;

    // Get SVD of uncontracted overlap
    // Store the left singular vectors in S
    nPrimUse_ = ORTH(NP,NP,overlap,NP,SS,XXX,NP,memManager_);

    size_t NPU = nPrimUse_;

    // Form orthonormal transformation matrix in S
    for(auto i = 0ul; i < NPU; i++)
      Scale(NP,IntsT(1.)/std::sqrt(SS[i]),
          overlap + i*NP,1);

    // Transform T into the orthonormal basis
    // T -> TO
    Gemm('C','N',NPU,NP,NP,IntsT(1.),overlap,NP,
        kinetic,NP,IntsT(0.),SCR1,NPU);
    Gemm('N','N',NPU,NPU,NP,IntsT(1.),SCR1,NPU,overlap,NP,
        IntsT(0.),kinetic,NPU);

    // Get the SVD of TO
    // Store the left singular vectors in TO
    SVD('O','N',NPU,NPU,kinetic,NPU,SS,XXX,NPU,
      XXX,NPU,memManager_);

    // Transformation matrix
    UK = memManager_.malloc<IntsT>(NP*NPU);

    // Form UK = S * T
    Gemm('N','N',NP,NPU,NPU,IntsT(1.),overlap,NP,
      kinetic,NPU,IntsT(0.),UK,NP);

    // Allocate and for "P^2" potential
    IntsT *P2P = memManager_.malloc<IntsT>(NPU*NPU);

    // P2P = UK**T * V * UK
    Gemm('C','N',NPU,NP,NP,IntsT(1.),UK,NP,potential,NP,
        IntsT(0.),SCR1,NPU);
    Gemm('N','N',NPU,NPU,NP,IntsT(1.),SCR1,NPU,UK,NP,IntsT(0.),P2P,NPU);

    // Transform PVP into the "P^2" basis
    Gemm('C','N',NPU,NP,NP,IntsT(1.),UK,NP,PVdotP,NP,
        IntsT(0.),SCR1,NPU);
    Gemm('N','N',NPU,NPU,NP,IntsT(1.),SCR1,NPU,UK,NP,
        IntsT(0.),PVdotP,NPU);

    // Loop over PVxP terms
    if (this->oneETerms_.SORelativistic)
      for(auto & SL : PVcrossP ){
        Gemm('C','N',NPU,NP,NP,IntsT(1.),UK,NP,SL,NP,IntsT(0.),SCR1,NPU);
        Gemm('N','N',NPU,NPU,NP,IntsT(1.),SCR1,NPU,UK,NP,IntsT(0.),SL,NPU);
      }

    // P^2 -> P^-1
    for(auto i = 0; i < NPU; i++) SS[i] = 1./std::sqrt(2*SS[i]);

    // Transform PVP into "P^-1" basis
    for(auto j = 0; j < NPU; j++)
    for(auto i = 0; i < NPU; i++){
      PVdotP[i + j*NPU] *= SS[i] * SS[j];
      if (this->oneETerms_.SORelativistic)
        for(auto &SL : PVcrossP)
          SL[i + j*NPU] *= SS[i] * SS[j];
    }

    // Allocate 4C CORE Hamiltonian

    // CH = [ V    cp       ]
    //      [ cp   W - 2mc^2]
    MatsT *CH4C = memManager_.malloc<MatsT>(16*NPU*NPU);
    memset(CH4C,0,16*NPU*NPU*sizeof(MatsT));

    // Allocate W separately  as it's needed later
    size_t LDW = 2*NPU;
    MatsT *Wp  = memManager_.malloc<MatsT>(LDW*LDW);

    formW(NPU,Wp,LDW,PVdotP,NPU,
        PVcrossP[2],NPU,
        PVcrossP[1],NPU,
        PVcrossP[0],NPU,
        not this->oneETerms_.SORelativistic);

    // Subtract out 2mc^2 from W diagonals
    const double WFact = 2. * SpeedOfLight * SpeedOfLight;
    for(auto j = 0ul; j < 2*NPU; j++) Wp[j + LDW*j] -= WFact;

    // Copy W into the 4C CH storage
    MatsT *CHW = CH4C + 8*NPU*NPU + 2*NPU;
    SetMat('N',2*NPU,2*NPU,MatsT(1.),Wp,LDW,CHW,4*NPU);

    // P^-1 -> P
    for(auto i = 0; i < NPU; i++) SS[i] = 1./SS[i];

    // V = [ P2P  0   ]
    //     [ 0    P2P ]
    MatsT * V = CH4C;
    SetMatDiag(NPU,NPU,P2P,NPU,V,4*NPU);

    // Set the diagonal cp blocks of CH
    // CP = [cp 0  ]
    //      [0  cp ]
    MatsT *CP11 = CH4C + 8*NPU*NPU;
    MatsT *CP12 = CP11 + 4*NPU*NPU + NPU;
    MatsT *CP21 = CH4C + 2*NPU;
    MatsT *CP22 = CP21 + 4*NPU*NPU + NPU;

    for(auto j = 0; j < NPU; j++) {
      CP11[j + 4*NPU*j] = SpeedOfLight * SS[j];
      CP12[j + 4*NPU*j] = SpeedOfLight * SS[j];
      CP21[j + 4*NPU*j] = SpeedOfLight * SS[j];
      CP22[j + 4*NPU*j] = SpeedOfLight * SS[j];
    }

    // Diagonalize the 4C CH
    double *CHEV = memManager_.malloc<double>(4*NPU);

    HermetianEigen('V','U',4*NPU,CH4C,4*NPU,CHEV,memManager_);


    // Get pointers to "L" and "S" components of eigenvectors
    MatsT *L = CH4C + 8*NPU*NPU;
    MatsT *S = L + 2*NPU;


    // Invert "L"; L -> L^-1
    LUInv(2*NPU,L,4*NPU,memManager_);


    // Reuse the charge conjugated space for X and Y
    X = memManager_.malloc<MatsT>(4*NPU*NPU);
    memset(X,0,4*NPU*NPU*sizeof(MatsT));
    Y = memManager_.malloc<MatsT>(4*NPU*NPU);
    memset(Y,0,4*NPU*NPU*sizeof(MatsT));

    // Form X = S * L^-1
    Gemm('N','N',2*NPU,2*NPU,2*NPU,MatsT(1.),S,4*NPU,L,4*NPU,
      MatsT(0.),X,2*NPU);

    // Form Y = sqrt(1 + X**H * X)

    // Y = X**H * X
    Gemm('C','N',2*NPU,2*NPU,2*NPU,MatsT(1.),X,2*NPU,X,2*NPU,
      MatsT(0.),Y,2*NPU);

    // Y = Y + I
    for(auto j = 0; j < 2*NPU; j++) Y[j + 2*NPU*j] += 1.0;

    // Y -> V * y * V**H
    // XXX: Store the eigenvalues of Y in CHEV
    HermetianEigen('V','U',2*NPU,Y,2*NPU,CHEV,memManager_);

    // SCR1 -> V * y^-0.25
    for(auto j = 0ul; j < 2*NPU; j++)
    for(auto i = 0ul; i < 2*NPU; i++)
      CSCR1[i + 2*NPU*j] = Y[i + 2*NPU*j] * std::pow(CHEV[j],-0.25);

    // Y = SCR1 * SCR1**H
    Gemm('N','C',2*NPU,2*NPU,2*NPU,MatsT(1.),CSCR1,2*NPU,CSCR1,2*NPU,
      MatsT(0.),Y,2*NPU);

    // Build the effective two component CH in "L"
    MatsT *FullCH2C = L;

    // Copy P2P into spin diagonal blocks of 2C CH
    MatsT *CH2C1 = FullCH2C;
    SetMatDiag(NPU,NPU,P2P,NPU,CH2C1,4*NPU);

    // Construct 2C CH in the uncontracted basis
    // 2C CH = Y * (V' + cp * X + X**H * cp + X**H * W' * X) * Y

    // SCR1 = cp * X
    for(auto j = 0; j < 2*NPU; j++)
    for(auto i = 0; i < NPU; i++) {
      CSCR1[i + 2*NPU*j] = SpeedOfLight * SS[i] * X[i + 2*NPU*j];
      CSCR1[i + NPU + 2*NPU*j] = SpeedOfLight * SS[i] * X[i + NPU + 2*NPU*j];
    }

    // 2C CH += SCR1 + SCR1**H
    MatAdd('N','N',2*NPU,2*NPU,MatsT(1.),FullCH2C,4*NPU,MatsT(1.),
      CSCR1,2*NPU, FullCH2C,4*NPU);
    MatAdd('N','C',2*NPU,2*NPU,MatsT(1.),FullCH2C,4*NPU,MatsT(1.),
      CSCR1,2*NPU, FullCH2C,4*NPU);


    // SCR1 = X**H * W
    Gemm('C','N',2*NPU,2*NPU,2*NPU,MatsT(1.),X,2*NPU,Wp,LDW,MatsT(0.),
      CSCR1,2*NPU);

    // 2C CH += SCR1 * X
    Gemm('N','N',2*NPU,2*NPU,2*NPU,MatsT(1.),CSCR1,2*NPU,X,2*NPU,
      MatsT(1.),FullCH2C,4*NPU);

    // SCR1 = CH2C * Y
    Gemm('C','N',2*NPU,2*NPU,2*NPU,MatsT(1.),FullCH2C,4*NPU,Y,2*NPU,MatsT(0.),
      CSCR1,2*NPU);


    // 2C CH = Y * SCR1
    Gemm('N','N',2*NPU,2*NPU,2*NPU,MatsT(1.),Y,2*NPU,CSCR1,2*NPU,
      MatsT(0.),FullCH2C,4*NPU);

    // Allocate memory for the uncontracted spin components
    // of the 2C CH
    MatsT *HUnS = memManager_.malloc<MatsT>(NP*NP);
    MatsT *HUnZ, *HUnX, *HUnY;

    if (this->oneETerms_.SORelativistic) {
      HUnZ = memManager_.malloc<MatsT>(NP*NP);
      HUnX = memManager_.malloc<MatsT>(NP*NP);
      HUnY = memManager_.malloc<MatsT>(NP*NP);
      SpinScatter(NPU,FullCH2C,4*NPU,HUnS,NPU,HUnZ,NPU,HUnY,NPU,HUnX,NPU);
    } else {
      MatAdd('N','N',NPU,NPU,MatsT(1.),FullCH2C,4*NPU,MatsT(1.),
        FullCH2C+NPU+4*NPU*NPU,4*NPU,HUnS,NPU);
    }

    // Partition the scratch space into one complex and one real NP x NP
    // matrix
    IntsT   * SUK   = SCR1;
    MatsT * CSCR2 = reinterpret_cast<MatsT*>(SUK + NP*NP);

    // Store the Product of S and UK
    Gemm('N','N',NP,NPU,NP,IntsT(1.),uncontractedInts_.overlap,NP,UK,NP,IntsT(0.),SCR1,NP);

    // Transform the spin components of the 2C CH into R-space
    //
    // H(k) -> SUK * H(k) * (SUK)**H
    //
    // ** Using the fact that H(k) is hermetian
    // CSCR2 = SUK * H(k) -> CSCR2**H = H(k) * (SUK)**H
    // H(k) -> SUK * CSCR2**H
    //

    // Transform H(S)
    Gemm('N','N',NP,NPU,NPU,MatsT(1.),SUK,NP,HUnS,NPU,MatsT(0.),
      CSCR2,NP);
    Gemm('N','C',NP,NP,NPU,MatsT(1.),SUK,NP,CSCR2,NP,MatsT(0.),
      HUnS,NP);

    if (this->oneETerms_.SORelativistic) {
      // Transform H(Z)
      Gemm('N','N',NP,NPU,NPU,MatsT(1.),SUK,NP,HUnZ,NPU,MatsT(0.),
        CSCR2,NP);
      Gemm('N','C',NP,NP,NPU,MatsT(1.),SUK,NP,CSCR2,NP,MatsT(0.),
        HUnZ,NP);

      // Transform H(Y)
      Gemm('N','N',NP,NPU,NPU,MatsT(1.),SUK,NP,HUnY,NPU,MatsT(0.),
        CSCR2,NP);
      Gemm('N','C',NP,NP,NPU,MatsT(1.),SUK,NP,CSCR2,NP,MatsT(0.),
        HUnY,NP);

      // Transform H(X)
      Gemm('N','N',NP,NPU,NPU,MatsT(1.),SUK,NP,HUnX,NPU,MatsT(0.),
        CSCR2,NP);
      Gemm('N','C',NP,NP,NPU,MatsT(1.),SUK,NP,CSCR2,NP,MatsT(0.),
        HUnX,NP);
    }

    // Transform H(k) into the contracted basis

    Gemm('N','N',NB,NP,NP,MatsT(1.),mapPrim2Cont,NB,HUnS,
      NP,MatsT(0.),CSCR1,NB);
    Gemm('N','C',NB,NB,NP,MatsT(1.),mapPrim2Cont,NB,CSCR1,
      NB,MatsT(0.),CH[0],NB);

    if (CH.size() > 1) {

      if (this->oneETerms_.SORelativistic) {
        Gemm('N','N',NB,NP,NP,MatsT(1.),mapPrim2Cont,NB,HUnZ,
          NP,MatsT(0.),CSCR1,NB);
        Gemm('N','C',NB,NB,NP,MatsT(1.),mapPrim2Cont,NB,CSCR1,
          NB,MatsT(0.),CH[1],NB);

        Gemm('N','N',NB,NP,NP,MatsT(1.),mapPrim2Cont,NB,HUnY,
          NP,MatsT(0.),CSCR1,NB);
        Gemm('N','C',NB,NB,NP,MatsT(1.),mapPrim2Cont,NB,CSCR1,
          NB,MatsT(0.),CH[2],NB);

        Gemm('N','N',NB,NP,NP,MatsT(1.),mapPrim2Cont,NB,HUnX,
          NP,MatsT(0.),CSCR1,NB);
        Gemm('N','C',NB,NB,NP,MatsT(1.),mapPrim2Cont,NB,CSCR1,
          NB,MatsT(0.),CH[3],NB);
      } else {
        memset(CH[1],0,NB*NB*sizeof(MatsT));
        memset(CH[2],0,NB*NB*sizeof(MatsT));
        memset(CH[3],0,NB*NB*sizeof(MatsT));
      }

    }

    memManager_.free(overlap, kinetic, potential, PVdotP,
      SCR1, P2P, CH4C, Wp, CHEV, HUnS);
    if (this->oneETerms_.SORelativistic)
      memManager_.free(PVcrossP[0], PVcrossP[1], PVcrossP[2],
        HUnZ, HUnX, HUnY);

  }

  template void X2C<dcomplex,double>::computeX2C(EMPerturbation&, std::vector<dcomplex*>&);

  template<> void X2C<dcomplex,dcomplex>::computeX2C(EMPerturbation&, std::vector<dcomplex*>&) {
    CErr("X2C + Complex Ints NYI",std::cout);
  }

  template void X2C<double,double>::computeX2C(EMPerturbation&, std::vector<double*>&);


  /**
   *  \brief Compute the X2C Core Hamiltonian
   */
  template <typename MatsT, typename IntsT>
  void X2C<MatsT, IntsT>::computeCoreH(EMPerturbation& emPert, std::vector<MatsT*> &CH) {
    this->aoints_.computeAOOneE(emPert,this->oneETerms_); // compute the necessary 1e ints
    computeX2C(emPert, CH);
    if (this->oneETerms_.SORelativistic) BoettgerScale(CH);
  }

  template void X2C<dcomplex,double>::computeCoreH(EMPerturbation&, std::vector<dcomplex*>&);

  template<> void X2C<dcomplex,dcomplex>::computeCoreH(EMPerturbation&, std::vector<dcomplex*>&) {
    CErr("X2C + Complex Ints NYI",std::cout);
  }

  template void X2C<double,double>::computeCoreH(EMPerturbation&, std::vector<double*>&);

  /**
   *  \brief Compute the picture change matrices UL, US
   */
  template <typename MatsT, typename IntsT>
  void X2C<MatsT, IntsT>::computeU() {

    size_t NP = uncontractedBasis_.nPrimitive;
    size_t NPU= nPrimUse_;
    size_t NB = basisSet_.nBasis;

    // UL = UK * Y * UK^-1 * UP2C
    // US = 2 * SpeedOfLight * UK * p^-1 * X * Y * UK^-1 * UP2C

    // 1.  UP2CSUK = UP2C * S * UK
    IntsT *UP2CS = memManager_.malloc<IntsT>(NB*NP);
    Gemm('N','N',NB,NP,NP,IntsT(1.),mapPrim2Cont,NB,
      uncontractedInts_.overlap,NP,IntsT(0.),UP2CS,NB);
    IntsT *UP2CSUK = memManager_.malloc<IntsT>(4*NP*NPU);
    Gemm('N','N',NB,NPU,NP,IntsT(1.),UP2CS,NB,UK,NP,IntsT(0.),UP2CSUK,2*NB);
    SetMatDiag(NB,NPU,UP2CSUK,2*NB,UP2CSUK,2*NB);

    // 2. R^T = UP2C * S * UK * Y^T
    MatsT *RT = memManager_.malloc<MatsT>(4*NB*NPU);
    Gemm('N','C',2*NB,2*NPU,2*NPU,MatsT(1.),UP2CSUK,2*NB,
      Y,2*NPU,MatsT(0.),RT,2*NB);

    // 3. Xp = 2 c p^-1 X
    double twoC = 2 * SpeedOfLight;
    double *twoCPinv = memManager_.malloc<double>(NPU);
    for(size_t i = 0; i < NPU; i++) twoCPinv[i] = twoC/p[i];
    MatsT *twoCPinvX = memManager_.malloc<MatsT>(4*NPU*NPU);
    for(size_t j = 0; j < 2*NPU; j++)
    for(size_t i = 0; i < NPU; i++) {
      twoCPinvX[i + 2*NPU*j] = twoCPinv[i] * X[i + 2*NPU*j];
      twoCPinvX[i + NPU + 2*NPU*j] = twoCPinv[i] * X[i + NPU + 2*NPU*j];
    }

    // 4. UK2c = [ UK  0  ]
    //           [ 0   UK ]
    IntsT *UK2c = UP2CSUK;
    SetMatDiag(NP,NPU,UK,NP,UK2c,2*NP);

    // 5. US = UK2c * Xp * RT^T
    UL = memManager_.malloc<MatsT>(4*NP*NB);
    US = memManager_.malloc<MatsT>(4*NP*NB);
    Gemm('N','C',2*NPU,2*NB,2*NPU,MatsT(1.),twoCPinvX,2*NPU,
      RT,2*NB,MatsT(0.),UL,2*NPU);
    Gemm('N','N',2*NP,2*NB,2*NPU,MatsT(1.),UK2c,2*NP,
      UL,2*NPU,MatsT(0.),US,2*NP);

    // 6. UL = UK2c * RT^T
    Gemm('N','C',2*NP,2*NB,2*NPU,MatsT(1.),UK2c,2*NP,
      RT,2*NB,MatsT(0.),UL,2*NP);

    memManager_.free(UP2CS, UP2CSUK, RT, twoCPinv, twoCPinvX);

  }

  template void X2C<dcomplex,double>::computeU();

  template<> void X2C<dcomplex,dcomplex>::computeU() {
    CErr("X2C + Complex Ints NYI",std::cout);
  }

  template void X2C<double,double>::computeU();

  /**
   *  \brief Compute the X2C Core Hamiltonian
   */
  template <typename MatsT, typename IntsT>
  void X2C<MatsT, IntsT>::computeX2C_UDU(EMPerturbation& emPert, std::vector<MatsT*> &CH) {

    size_t NP = uncontractedBasis_.nPrimitive;
    size_t NB = basisSet_.nBasis;

    // Make copy of integrals
    IntsT *kinetic   = memManager_.malloc<IntsT>(NP*NP);
    std::copy_n(uncontractedInts_.kinetic, NP*NP, kinetic);
    IntsT *potential = memManager_.malloc<IntsT>(NP*NP);
    std::copy_n(uncontractedInts_.potential, NP*NP, potential);
    IntsT *PVdotP    = memManager_.malloc<IntsT>(NP*NP);
    std::copy_n(uncontractedInts_.PVdotP, NP*NP, PVdotP);
    std::vector<IntsT *> PVcrossP(3);
    if (this->oneETerms_.SORelativistic)
      for(size_t i = 0; i < 3; i++) {
        PVcrossP[i] = memManager_.malloc<IntsT>(NP*NP);
        std::copy_n(uncontractedInts_.PVcrossP[i], NP*NP, PVcrossP[i]);
      }

    // Allocate W separately  as it's needed later
    size_t LDW = 2*NP;
    W = memManager_.malloc<MatsT>(LDW*LDW);

    formW(NP,W,LDW,PVdotP,NP,
      PVcrossP[2],NP,
      PVcrossP[1],NP,
      PVcrossP[0],NP,
      not this->oneETerms_.SORelativistic);

    // T2c = [ T  0 ]
    //       [ 0  T ]
    IntsT *T2c = memManager_.malloc<IntsT>(LDW*LDW);
    SetMatDiag(NP,NP,kinetic,NP,T2c,2*NP);

    // V2c = [ V  0 ]
    //       [ 0  V ]
    IntsT *V2c = memManager_.malloc<IntsT>(LDW*LDW);
    SetMatDiag(NP,NP,potential,NP,V2c,2*NP);

    MatsT *Hx2c = memManager_.malloc<MatsT>(4*NB*NB);
    MatsT *SCR = memManager_.malloc<MatsT>(4*NP*NB);

    // Hx2c = UL^H * T2c * US
    Gemm('N','N',2*NP,2*NB,2*NP,MatsT(1.),T2c,2*NP,
      US,2*NP,MatsT(0.),SCR,2*NP);
    Gemm('C','N',2*NB,2*NB,2*NP,MatsT(1.),UL,2*NP,
      SCR,2*NP,MatsT(0.),Hx2c,2*NB);
    // Hx2c += US^H * T2c * UL
    Gemm('N','N',2*NP,2*NB,2*NP,MatsT(1.),T2c,2*NP,
      UL,2*NP,MatsT(0.),SCR,2*NP);
    Gemm('C','N',2*NB,2*NB,2*NP,MatsT(1.),US,2*NP,
      SCR,2*NP,MatsT(1.),Hx2c,2*NB);
    // Hx2c -= US^H * T2c * US
    Gemm('N','N',2*NP,2*NB,2*NP,MatsT(1.),T2c,2*NP,
      US,2*NP,MatsT(0.),SCR,2*NP);
    Gemm('C','N',2*NB,2*NB,2*NP,MatsT(-1.),US,2*NP,
      SCR,2*NP,MatsT(1.),Hx2c,2*NB);
    // Hx2c += UL^H * V2c * UL
    Gemm('N','N',2*NP,2*NB,2*NP,MatsT(1.),V2c,2*NP,
      UL,2*NP,MatsT(0.),SCR,2*NP);
    Gemm('C','N',2*NB,2*NB,2*NP,MatsT(1.),UL,2*NP,
      SCR,2*NP,MatsT(1.),Hx2c,2*NB);
    // Hx2c += 1/(4*C**2) US^H * W * US
    Gemm('N','N',2*NP,2*NB,2*NP,
      MatsT(0.25/SpeedOfLight/SpeedOfLight),W,2*NP,
      US,2*NP,MatsT(0.),SCR,2*NP);
    Gemm('C','N',2*NB,2*NB,2*NP,MatsT(1.),US,2*NP,
      SCR,2*NP,MatsT(1.),Hx2c,2*NB);

    if (this->oneETerms_.SORelativistic)
      SpinScatter(NB,Hx2c,2*NB,CH[0],NB,CH[1],NB,CH[2],NB,CH[3],NB);
    else {
      MatAdd('N','N',NB,NB,MatsT(1.),Hx2c,2*NB,MatsT(1.),
        Hx2c+NB+2*NB*NB,2*NB,CH[0],NB);
      if (CH.size() > 1) {
        memset(CH[1],0,NB*NB*sizeof(MatsT));
        memset(CH[2],0,NB*NB*sizeof(MatsT));
        memset(CH[3],0,NB*NB*sizeof(MatsT));
      }
    }

    memManager_.free(kinetic, potential, PVdotP, T2c, V2c, Hx2c, SCR);
    if (this->oneETerms_.SORelativistic)
      memManager_.free(PVcrossP[0], PVcrossP[1], PVcrossP[2]);
  }

  template void X2C<dcomplex,double>::computeX2C_UDU(EMPerturbation&, std::vector<dcomplex*>&);

  template<> void X2C<dcomplex,dcomplex>::computeX2C_UDU(EMPerturbation&, std::vector<dcomplex*>&) {
    CErr("X2C + Complex Ints NYI",std::cout);
  }

  template void X2C<double,double>::computeX2C_UDU(EMPerturbation&, std::vector<double*>&);

  /**
   *  \brief Compute the X2C Core Hamiltonian correction to NR
   */
  template <typename MatsT, typename IntsT>
  void X2C<MatsT, IntsT>::computeX2C_corr(EMPerturbation &emPert, std::vector<MatsT*> &CH) {

    computeX2C(emPert, CH);

    size_t NP = uncontractedBasis_.nPrimitive;
    size_t NB = basisSet_.nBasis;

    NRCoreH<MatsT, IntsT> nr(uncontractedInts_);
    std::vector<MatsT*> NRCH(CH.size(), nullptr);
    MatsT* SCR = memManager_.malloc<MatsT>(NB*NP);
    for (auto &CHi : NRCH) {
      CHi = memManager_.malloc<MatsT>(NP*NP);
      memset(CHi,0,NP*NP*sizeof(MatsT));
    }
    nr.computeNRCH(emPert, NRCH);

    // Transform H(k) into the contracted basis
    for (size_t i=0; i<CH.size(); i++) {
      Gemm('N','N',NB,NP,NP,MatsT(1.),mapPrim2Cont,NB,NRCH[i],
        NP,MatsT(0.),SCR,NB);
      Gemm('N','C',NB,NB,NP,MatsT(-1.),mapPrim2Cont,NB,SCR,
        NB,MatsT(1.),CH[i],NB);
    }

    for (auto &CHi : NRCH)
      memManager_.free(CHi);
    memManager_.free(SCR);

  }

  template void X2C<dcomplex,double>::computeX2C_corr(EMPerturbation&, std::vector<dcomplex*>&);

  template<> void X2C<dcomplex,dcomplex>::computeX2C_corr(EMPerturbation&, std::vector<dcomplex*>&) {
    CErr("X2C + Complex Ints NYI",std::cout);
  }

  template void X2C<double,double>::computeX2C_corr(EMPerturbation&, std::vector<double*>&);

}; // namespace ChronusQ

