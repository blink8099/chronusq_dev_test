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
#include <electronintegrals/relativisticints.hpp>
#include <matrix.hpp>
#include <physcon.hpp>
#include <cqlinalg.hpp>
#include <cqlinalg/svd.hpp>

namespace ChronusQ {

  /**
   *  \brief Boettger scaling for spin-orbit operator
   */
  template <typename MatsT, typename IntsT>
  void X2C<MatsT, IntsT>::BoettgerScale(
      std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> coreH) {

    size_t NB = basisSet_.nBasis;

    size_t n1, n2;
    std::array<double,6> Ql={0.,2.,10.,28.,60.,110.};

    if( this->basisSet_.maxL > 5 ) CErr("Boettger scaling for L > 5 NYI");

    for(auto s1(0ul), i(0ul); s1 < this->basisSet_.nShell; s1++, i+=n1) {
      n1 = this->basisSet_.shells[s1].size();

      size_t L1 = this->basisSet_.shells[s1].contr[0].l;
      if ( L1 == 0 ) continue;

      auto Z1 = this->molecule_.atoms[this->basisSet_.mapSh2Cen[s1]].nucCharge;


    for(auto s2(0ul), j(0ul); s2 < this->basisSet_.nShell; s2++, j+=n2) {
      n2 = this->basisSet_.shells[s2].size();

      size_t L2 = this->basisSet_.shells[s2].contr[0].l;
      if ( L2 == 0 ) continue;

      auto Z2 = this->molecule_.atoms[this->basisSet_.mapSh2Cen[s2]].nucCharge;

      MatsT fudgeFactor = -1 * std::sqrt(
        Ql[L1] * Ql[L2] /
        Z1 / Z2
      );

      MatAdd('N','N',n1,n2,MatsT(1.),coreH->Z().pointer() + i + j*NB,NB,
          fudgeFactor,coreH->Z().pointer() + i + j*NB,NB,
          coreH->Z().pointer() + i + j*NB,NB);

      MatAdd('N','N',n1,n2,MatsT(1.),coreH->Y().pointer() + i + j*NB,NB,
          fudgeFactor,coreH->Y().pointer() + i + j*NB,NB,
          coreH->Y().pointer() + i + j*NB,NB);

      MatAdd('N','N',n1,n2,MatsT(1.),coreH->X().pointer() + i + j*NB,NB,
          fudgeFactor,coreH->X().pointer() + i + j*NB,NB,
          coreH->X().pointer() + i + j*NB,NB);

    } // loop s2
    } // loop s1
  }

  /**
   *  \brief Compute the X2C Core Hamiltonian
   */
  template <typename MatsT, typename IntsT>
  void X2C<MatsT, IntsT>::computeX2C(EMPerturbation &emPert,
      std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> coreH) {
    IntsT* XXX = reinterpret_cast<IntsT*>(NULL);

    size_t NP = uncontractedBasis_.nPrimitive;
    size_t NB = basisSet_.nBasis;

    uncontractedInts_.computeAOOneE(memManager_,
        molecule_, uncontractedBasis_, emPert,
        {{OVERLAP,0}, {KINETIC,0}, {NUCLEAR_POTENTIAL,0}},
        this->aoiOptions_);

    // Make copy of integrals
    IntsT *overlap   = memManager_.malloc<IntsT>(NP*NP);
    std::copy_n(uncontractedInts_.overlap->pointer(), NP*NP, overlap);

    // Compute the mappings from primitives to CGTOs
    mapPrim2Cont = memManager_.malloc<IntsT>(NP*NB);
    basisSet_.makeMapPrim2Cont(overlap,mapPrim2Cont,memManager_);

    // Allocate Scratch Space (enough for 2*NP x 2*NP complex matricies)
    IntsT *SCR1  = memManager_.malloc<IntsT>(8*NP*NP);
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
    std::shared_ptr<OneEInts<IntsT>> kinetic =
        std::dynamic_pointer_cast<OneEInts<IntsT>>(
            ElectronIntegrals::transform(
                *uncontractedInts_.kinetic, 'N', overlap, NPU, NP));

    // Get the SVD of TO
    // Store the left singular vectors in TO
    SVD('O','N',NPU,NPU,kinetic->pointer(),NPU,SS,XXX,NPU,
      XXX,NPU,memManager_);

    // Transformation matrix
    UK = memManager_.malloc<IntsT>(NP*NPU);

    // Form UK = S * T
    Gemm('N','N',NP,NPU,NPU,IntsT(1.),overlap,NP,
      kinetic->pointer(),NPU,IntsT(0.),UK,NP);

    // Allocate and for "P^2" potential
    std::shared_ptr<OneERelInts<IntsT>> potential =
        std::dynamic_pointer_cast<OneERelInts<IntsT>>(
            ElectronIntegrals::transform(
                *uncontractedInts_.potential, 'N', UK, NPU, NP));

    // P^2 -> P^-1
    for(auto i = 0; i < NPU; i++) SS[i] = 1./std::sqrt(2*SS[i]);

    // Transform PVP into "P^-1" basis
    for (auto &oei : potential->SZYX())
      for(auto j = 0; j < NPU; j++)
        for(auto i = 0; i < NPU; i++)
          oei(i,j) *= SS[i] * SS[j];

    // Allocate 4C CORE Hamiltonian

    // CH = [ V    cp       ]
    //      [ cp   W - 2mc^2]
    MatsT *CH4C = memManager_.malloc<MatsT>(16*NPU*NPU);
    std::fill_n(CH4C,16*NPU*NPU,MatsT(0.));

    // Allocate W separately  as it's needed later
    size_t LDW = 2*NPU;
    SquareMatrix<MatsT> Wp(
        std::dynamic_pointer_cast<OneERelInts<IntsT>>(potential)
            ->template formW<MatsT>());

    // Subtract out 2mc^2 from W diagonals
    const double WFact = 2. * SpeedOfLight * SpeedOfLight;
    for(auto j = 0ul; j < 2*NPU; j++) Wp(j,j) -= WFact;

    // Copy W into the 4C CH storage
    MatsT *CHW = CH4C + 8*NPU*NPU + 2*NPU;
    SetMat('N',2*NPU,2*NPU,MatsT(1.),Wp.pointer(),LDW,CHW,4*NPU);

    // P^-1 -> P
    for(auto i = 0; i < NPU; i++) SS[i] = 1./SS[i];

    // V = [ V  0 ]
    //     [ 0  V ]
    MatsT * CHV = CH4C;
    SetMatDiag(NPU,NPU,potential->pointer(),NPU,CHV,4*NPU);

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
    std::fill_n(X,4*NPU*NPU,MatsT(0.));
    Y = memManager_.malloc<MatsT>(4*NPU*NPU);
    std::fill_n(Y,4*NPU*NPU,MatsT(0.));

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
    SquareMatrix<MatsT> FullCH2C(memManager_, 2*NPU);

    // Copy potential into spin diagonal blocks of 2C CH
    SetMatDiag(NPU,NPU,potential->pointer(),NPU,FullCH2C.pointer(),2*NPU);

    // Construct 2C CH in the uncontracted basis
    // 2C CH = Y * (V' + cp * X + X**H * cp + X**H * W' * X) * Y

    // SCR1 = cp * X
    for(auto j = 0; j < 2*NPU; j++)
    for(auto i = 0; i < NPU; i++) {
      CSCR1[i + 2*NPU*j] = SpeedOfLight * SS[i] * X[i + 2*NPU*j];
      CSCR1[i + NPU + 2*NPU*j] = SpeedOfLight * SS[i] * X[i + NPU + 2*NPU*j];
    }

    // 2C CH += SCR1 + SCR1**H
    MatAdd('N','N',2*NPU,2*NPU,MatsT(1.),FullCH2C.pointer(),2*NPU,MatsT(1.),
      CSCR1,2*NPU, FullCH2C.pointer(),2*NPU);
    MatAdd('N','C',2*NPU,2*NPU,MatsT(1.),FullCH2C.pointer(),2*NPU,MatsT(1.),
      CSCR1,2*NPU, FullCH2C.pointer(),2*NPU);


    // SCR1 = X**H * W
    Gemm('C','N',2*NPU,2*NPU,2*NPU,MatsT(1.),X,2*NPU,
         Wp.pointer(),LDW,MatsT(0.),CSCR1,2*NPU);

    // 2C CH += SCR1 * X
    Gemm('N','N',2*NPU,2*NPU,2*NPU,MatsT(1.),CSCR1,2*NPU,
         X,2*NPU,MatsT(1.),FullCH2C.pointer(),2*NPU);

    // SCR1 = CH2C * Y
    Gemm('C','N',2*NPU,2*NPU,2*NPU,MatsT(1.),FullCH2C.pointer(),2*NPU,
         Y,2*NPU,MatsT(0.),CSCR1,2*NPU);


    // 2C CH = Y * SCR1
    Gemm('N','N',2*NPU,2*NPU,2*NPU,MatsT(1.),Y,2*NPU,CSCR1,2*NPU,
      MatsT(0.),FullCH2C.pointer(),2*NPU);

    // Allocate memory for the uncontracted spin components
    // of the 2C CH
    PauliSpinorSquareMatrices<MatsT> HUn(
        FullCH2C.template spinScatter<MatsT>(
            this->aoiOptions_.OneESpinOrbit,this->aoiOptions_.OneESpinOrbit));

    // Partition the scratch space into one complex and one real NP x NP
    // matrix
    IntsT *SUK   = SCR1;
    IntsT *CPSUK = SUK + NP*NPU;

    // Store the Product of S and UK
    Gemm('N','N',NP,NPU,NP,IntsT(1.),uncontractedInts_.overlap->pointer(),NP,
         UK,NP,IntsT(0.),SUK,NP);
    // Store the Product of mapPrim2Cont and SUK
    Gemm('N','N',NB,NPU,NP,IntsT(1.),mapPrim2Cont,NB,
         SUK,NP,IntsT(0.),CPSUK,NB);

    // Transform the spin components of the 2C CH into R-space
    *coreH = HUn.transform('C', CPSUK, NB, NB);

    memManager_.free(overlap, SCR1, CH4C, CHEV);


  }

  template void X2C<dcomplex,double>::computeX2C(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<dcomplex>>);

  template<> void X2C<dcomplex,dcomplex>::computeX2C(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<dcomplex>>) {
    CErr("X2C + Complex Ints NYI",std::cout);
  }

  template void X2C<double,double>::computeX2C(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<double>>);


  /**
   *  \brief Compute the X2C Core Hamiltonian
   */
  template <typename MatsT, typename IntsT>
  void X2C<MatsT, IntsT>::computeCoreH(EMPerturbation& emPert,
      std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> coreH) {
    computeX2C(emPert, coreH);
    if (this->aoiOptions_.OneESpinOrbit)
      BoettgerScale(coreH);
  }

  template void X2C<dcomplex,double>::computeCoreH(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<dcomplex>>);

  template<> void X2C<dcomplex,dcomplex>::computeCoreH(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<dcomplex>>) {
    CErr("X2C + Complex Ints NYI",std::cout);
  }

  template void X2C<double,double>::computeCoreH(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<double>>);

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
      uncontractedInts_.overlap->pointer(),NP,IntsT(0.),UP2CS,NB);
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
  void X2C<MatsT, IntsT>::computeX2C_UDU(EMPerturbation& emPert,
      std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> coreH) {

    size_t NP = uncontractedBasis_.nPrimitive;
    size_t NB = basisSet_.nBasis;

    // Allocate W separately  as it's needed later
    W = std::make_shared<SquareMatrix<MatsT>>(
        std::dynamic_pointer_cast<OneERelInts<IntsT>>(
            uncontractedInts_.potential)->template formW<MatsT>());

    // T2c = [ T  0 ]
    //       [ 0  T ]
    const OneEInts<IntsT> &T2c = uncontractedInts_.kinetic->
        template spatialToSpinBlock<IntsT>();

    // V2c = [ V  0 ]
    //       [ 0  V ]
    const OneEInts<IntsT> &V2c = uncontractedInts_.potential->
        template spatialToSpinBlock<IntsT>();

    SquareMatrix<MatsT> Hx2c(memManager_, 2*NB);
    MatsT *SCR = memManager_.malloc<MatsT>(4*NP*NB);

    // Hx2c = UL^H * T2c * US
    Gemm('N','N',2*NP,2*NB,2*NP,MatsT(1.),T2c.pointer(),2*NP,
      US,2*NP,MatsT(0.),SCR,2*NP);
    Gemm('C','N',2*NB,2*NB,2*NP,MatsT(1.),UL,2*NP,
      SCR,2*NP,MatsT(0.),Hx2c.pointer(),2*NB);
    // Hx2c += US^H * T2c * UL
    Gemm('N','N',2*NP,2*NB,2*NP,MatsT(1.),T2c.pointer(),2*NP,
      UL,2*NP,MatsT(0.),SCR,2*NP);
    Gemm('C','N',2*NB,2*NB,2*NP,MatsT(1.),US,2*NP,
      SCR,2*NP,MatsT(1.),Hx2c.pointer(),2*NB);
    // Hx2c -= US^H * T2c * US
    Gemm('N','N',2*NP,2*NB,2*NP,MatsT(1.),T2c.pointer(),2*NP,
      US,2*NP,MatsT(0.),SCR,2*NP);
    Gemm('C','N',2*NB,2*NB,2*NP,MatsT(-1.),US,2*NP,
      SCR,2*NP,MatsT(1.),Hx2c.pointer(),2*NB);
    // Hx2c += UL^H * V2c * UL
    Gemm('N','N',2*NP,2*NB,2*NP,MatsT(1.),V2c.pointer(),2*NP,
      UL,2*NP,MatsT(0.),SCR,2*NP);
    Gemm('C','N',2*NB,2*NB,2*NP,MatsT(1.),UL,2*NP,
      SCR,2*NP,MatsT(1.),Hx2c.pointer(),2*NB);
    // Hx2c += 1/(4*C**2) US^H * W * US
    Gemm('N','N',2*NP,2*NB,2*NP,
      MatsT(0.25/SpeedOfLight/SpeedOfLight),W->pointer(),2*NP,
      US,2*NP,MatsT(0.),SCR,2*NP);
    Gemm('C','N',2*NB,2*NB,2*NP,MatsT(1.),US,2*NP,
      SCR,2*NP,MatsT(1.),Hx2c.pointer(),2*NB);

    *coreH = Hx2c.template spinScatter<MatsT>(
        this->aoiOptions_.OneESpinOrbit, this->aoiOptions_.OneESpinOrbit);

    memManager_.free(SCR);
  }

  template void X2C<dcomplex,double>::computeX2C_UDU(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<dcomplex>>);

  template<> void X2C<dcomplex,dcomplex>::computeX2C_UDU(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<dcomplex>>) {
    CErr("X2C + Complex Ints NYI",std::cout);
  }

  template void X2C<double,double>::computeX2C_UDU(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<double>>);

  /**
   *  \brief Compute the X2C Core Hamiltonian correction to NR
   */
  template <typename MatsT, typename IntsT>
  void X2C<MatsT, IntsT>::computeX2C_corr(EMPerturbation &emPert,
      std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> coreH) {

    computeX2C(emPert, coreH);

    size_t NP = uncontractedBasis_.nPrimitive;
    size_t NB = basisSet_.nBasis;

    std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> NRcoreH =
        std::make_shared<PauliSpinorSquareMatrices<MatsT>>(memManager_, NP);
    NRcoreH->clear();

    NRCoreH<MatsT, IntsT>(uncontractedInts_, this->aoiOptions_)
        .computeNRCH(emPert, NRcoreH);

    *coreH -= NRcoreH->transform('C', mapPrim2Cont, NB, NB);

  }

  template void X2C<dcomplex,double>::computeX2C_corr(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<dcomplex>>);

  template<> void X2C<dcomplex,dcomplex>::computeX2C_corr(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<dcomplex>>) {
    CErr("X2C + Complex Ints NYI",std::cout);
  }

  template void X2C<double,double>::computeX2C_corr(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<double>>);

}; // namespace ChronusQ

