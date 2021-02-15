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


#include <integrals.hpp>
#include <electronintegrals/inhouseaointegral.hpp>
#include <util/matout.hpp>
#include <cqlinalg/blas1.hpp>
#include <cqlinalg/blasext.hpp>
#include <cqlinalg/blasutil.hpp>

#include <util/threads.hpp>
#include <util/mpi.hpp>
#include <util/time.hpp>
#include <util/math.hpp>

#include <electronintegrals/twoeints/gtodirectreleri.hpp>
#include <electronintegrals/contract/direct.hpp>

#include <chrono>

#define _PRECOMPUTE_SHELL_PAIRS

#define _SHZ_SCREEN_4C

//#define _KRAMERS_SYM_

#define _CONTRACTION_

#include <libcint.hpp>

namespace ChronusQ {

  // For DCB Hamitlonian, 
  // 12 density matrices upon input stored as
  // LL(MS,MX,MY,MZ), SS(MS,MX,MY,MZ), LS(MS,MX,MY,MZ)
  //
  // 12 contrated matrices upon output stored as
  // LL(MS,MX,MY,MZ), SS(MS,MX,MY,MZ), LS(MS,MX,MY,MZ)
  //
 
  template <typename MatsT, typename IntsT>
  void GTODirectRelERIContraction<MatsT,IntsT>::directScaffold(
    MPI_Comm comm, const bool screen,
    std::vector<TwoBodyContraction<MatsT>> &matList) const {

    DirectERI<IntsT> &eri =
        dynamic_cast<DirectERI<IntsT>&>(this->ints_);
    CQMemManager& memManager_ = eri.memManager();
    BasisSet& basisSet_ = eri.basisSet();

 
    // Determine the number of OpenMP threads
    size_t nThreads  = GetNumThreads();
    size_t LAThreads = GetLAThreads();
    SetLAThreads(1); // Turn off parallelism in LA functions

    // There are 78 2nd Derivatives per ERI 
    int AxBx = 3;
    int AxBy = 4;
    int AxBz = 5;
    int AyBx = 14;
    int AyBy = 15;
    int AyBz = 16;
    int AzBx = 24;
    int AzBy = 25;
    int AzBz = 26;

    int CxDx = 60;
    int CxDy = 61;
    int CxDz = 62;
    int CyDx = 65;
    int CyDy = 66;
    int CyDz = 67;
    int CzDx = 69;
    int CzDy = 70;
    int CzDz = 71;

    int AxCx = 6;
    int AxCy = 7;
    int AxCz = 8;
    int AyCx = 17;
    int AyCy = 18;
    int AyCz = 19;
    int AzCx = 27;
    int AzCy = 28;
    int AzCz = 29;

    int AxDx = 9;
    int AxDy = 10;
    int AxDz = 11;
    int AyDx = 20;
    int AyDy = 21;
    int AyDz = 22;
    int AzDx = 30;
    int AzDy = 31;
    int AzDz = 32;

    int BxCx = 36;
    int BxCy = 37;
    int BxCz = 38;
    int ByCx = 44;
    int ByCy = 45;
    int ByCz = 46;
    int BzCx = 51;
    int BzCy = 52;
    int BzCz = 53;

    int BxDx = 39;
    int BxDy = 40;
    int BxDz = 41;
    int ByDx = 47;
    int ByDy = 48;
    int ByDz = 49;
    int BzDx = 54;
    int BzDy = 55;
    int BzDz = 56;


    const size_t nBasis   = basisSet_.nBasis;
    const size_t nMat     = matList.size();
    const size_t nShell   = basisSet_.nShell;

    // Allocate scratch for raw integral batches
    //size_t maxShellSize = 
    //  std::max_element(basisSet_.shells.begin(),basisSet_.shells.end(),
    //    [](libint2::Shell &sh1, libint2::Shell &sh2) {
    //      return sh1.size() < sh2.size();
    //    })->size();

    size_t maxShellSize = (basisSet_.maxL+1)*(basisSet_.maxL+2)/2;
    size_t NB  = maxShellSize*4;
    size_t NB2 = NB*NB;
    size_t NB3 = NB2*NB;
    size_t NB4 = NB2*NB2;

    double *ERIBuffer = memManager_.malloc<double>(2*4*NB4*nThreads);
 
    // Create a vector of libint2::Engines for possible threading
    std::vector<libint2::Engine> engines(nThreads);

    // Initialize the first engine for the integral evaluation
    engines[0] = libint2::Engine(libint2::Operator::coulomb,
      basisSet_.maxPrim,basisSet_.maxL,2);

    // For DCB Hamitlonian, 
    // 12 density matrices upon input stored as
    // LL(MS,MX,MY,MZ), SS(MS,MX,MY,MZ), LS(MS,MX,MY,MZ)
    //
    // 12 contrated matrices upon output stored as
    // LL(MS,MX,MY,MZ), SS(MS,MX,MY,MZ), LS(MS,MX,MY,MZ)
    //
    // Allocate thread local storage to store integral contractions

    int LLMS = 0;
    int LLMX = 1;
    int LLMY = 2;
    int LLMZ = 3;
    int SSMS = 4;
    int SSMX = 5;
    int SSMY = 6;
    int SSMZ = 7;
    int LSMS = 8;
    int LSMX = 9;
    int LSMY = 10;
    int LSMZ = 11;

    for(auto iMat = 0; iMat < nMat; iMat++)
      memset(matList[iMat].AX,0.,nBasis*nBasis*sizeof(MatsT));

    std::vector<std::vector<MatsT*>> AXthreads;
    MatsT *AXRaw = nullptr;
    if(nThreads != 1) {
      AXRaw = memManager_.malloc<MatsT>(nThreads*nMat*nBasis*nBasis);    
      memset(AXRaw,0,nThreads*nMat*nBasis*nBasis*sizeof(MatsT));
    }

    if(nThreads == 1) {
      AXthreads.emplace_back();
      for(auto iMat = 0; iMat < nMat; iMat++)
        AXthreads.back().push_back(matList[iMat].AX);
    } else {
      for(auto iThread = 0; iThread < nThreads; iThread++) {
        AXthreads.emplace_back();
        for(auto iMat = 0; iMat < nMat; iMat++) 
          AXthreads.back().push_back(AXRaw + iThread*nMat*nBasis*nBasis + iMat*nBasis*nBasis);
      }
    }

#ifdef _SHZ_SCREEN_4C
    // Compute shell block norms (∞-norm) of matList.X
    if(eri.schwartz() == nullptr) eri.computeSchwartz();
    double *ShBlkNorms_raw = memManager_.malloc<double>(nMat*nShell*nShell);
    std::vector<double*> ShBlkNorms;
    for(auto iMat = 0, iOff = 0; iMat < nMat; iMat++, iOff += nShell*nShell ) {
      ShellBlockNorm(basisSet_.shells,matList[iMat].X,nBasis,ShBlkNorms_raw + iOff);
      ShBlkNorms.emplace_back(ShBlkNorms_raw + iOff);
    }

    // Find the max value of shell block ∞-norms of all matList.X
    double maxShBlkNorm = 0.;
    for(auto iMat = 0; iMat < nMat; iMat++)
      maxShBlkNorm = std::max(maxShBlkNorm,
        *std::max_element(ShBlkNorms[iMat],ShBlkNorms[iMat] + nShell*nShell) ); 

    size_t maxnPrim4 = 
      basisSet_.maxPrim * basisSet_.maxPrim * basisSet_.maxPrim * 
      basisSet_.maxPrim;

    // Set Libint precision
    engines[0].set_precision(
      std::min(
        std::numeric_limits<double>::epsilon(),
        eri.threshSchwartz()/maxShBlkNorm
      )/maxnPrim4
    );
//    engines[0].set_precision(0.);

    // Get the max over all the matricies for the shell block ∞-norms
    // OVERWRITES ShBlkNorms[0]
    for(auto k = 0; k < nShell*nShell; k++) {
      double mx = std::abs(ShBlkNorms[0][k]);
      for(auto iMat = 1; iMat < nMat; iMat++)
        mx = std::max(mx,std::abs(ShBlkNorms[iMat][k]));
      ShBlkNorms[0][k] = mx;
    }


#else
    // Set Libint precision
    engines[0].set_precision(0.);
#endif

    // Copy master thread engine to other threads
    for(size_t i = 1; i < nThreads; i++) engines[i] = engines[0];

    // Keeping track of number of integrals skipped
    std::vector<size_t> nSkip(nThreads,0);

    dcomplex iscale = dcomplex(0.0, 1.0);

    auto topDirect = tick();

    #pragma omp parallel
    {

      size_t thread_id = GetThreadID();

      auto &AX_loc = AXthreads[thread_id];

      double *ERIBuffAB = &ERIBuffer[thread_id*4*NB4];
      double *ERIBuffCD = &ERIBuffer[nThreads*4*NB4 + thread_id*4*NB4];

      size_t n1,n2,n3,n4,m,n,k,l,mnkl,bf1,bf2,bf3,bf4;
      size_t s4_max;

      for(size_t s1(0), bf1_s(0), s1234(0); s1 < basisSet_.nShell; 
          bf1_s+=n1, s1++) { 

        n1 = basisSet_.shells[s1].size(); // Size of Shell 1

      for(size_t s2(0), bf2_s(0); s2 <= s1; bf2_s+=n2, s2++) {

        n2 = basisSet_.shells[s2].size(); // Size of Shell 2

#ifdef _SHZ_SCREEN_4C
        double shz12 = 0, shMax12 = 0;
        if( screen ) {
          shz12 = eri.schwartz()[s1 + s2*nShell];
          shMax12 = ShBlkNorms[0][s1 + s2*nShell];
        }
#endif

      for(size_t s3(0), bf3_s(0); s3 <= s1; bf3_s+=n3, s3++) {

        n3 = basisSet_.shells[s3].size(); // Size of Shell 3
        s4_max = (s1 == s3) ? s2 : s3; // Determine the unique max of Shell 4

#ifdef _SHZ_SCREEN_4C

        double shMax123 = 0;
        if( screen ) {
          // Pre-calculate shell-block norm max's that only
          // depend on shells 1,2 and 3
          shMax123 = 
            std::max(ShBlkNorms[0][s1 + s3*nShell], 
                     ShBlkNorms[0][s2 + s3*nShell]);

          shMax123 = std::max(shMax123,shMax12);
        }

#endif
 
      for(size_t s4(0), bf4_s(0); s4 <= s4_max; bf4_s+=n4, s4++, s1234++) {

        n4 = basisSet_.shells[s4].size(); // Size of Shell 4

        // Round Robbin work distribution
        #ifdef _OPENMP
        if( s1234 % nThreads != thread_id ) continue;
        #endif


#ifdef _SHZ_SCREEN_4C

        double shMax = 0;

        if( screen ) {
          // Compute Shell norm max
          shMax = 
            std::max(ShBlkNorms[0][s1 + s4*nShell],
            std::max(ShBlkNorms[0][s2 + s4*nShell],
                     ShBlkNorms[0][s3 + s4*nShell]));

          shMax = std::max(shMax,shMax123);

          if((shMax *shMax * shz12 * eri.schwartz()[s3 + s4*nShell]) <
             eri.threshSchwartz()) { nSkip[thread_id]++; continue; }
        }
 
#endif

        // Evaluate ERI for shell quartet (s1 s2 | s3 s4)
        engines[thread_id].compute2<
          libint2::Operator::coulomb, libint2::BraKet::xx_xx, 2>(
          basisSet_.shells[s1],
          basisSet_.shells[s2],
          basisSet_.shells[s3],
          basisSet_.shells[s4]
        );


        const auto& buff = engines[thread_id].results();

        if(&buff == nullptr) continue;

#if 1 // Contraction

	memset(ERIBuffAB,0.0,4*NB4);
	memset(ERIBuffCD,0.0,4*NB4);

        for(m = 0, mnkl = 0ul  ; m <                  n1; ++m) 
        for(n =   maxShellSize ; n <   maxShellSize + n2; ++n)
        for(k = 2*maxShellSize ; k < 2*maxShellSize + n3; ++k) 
        for(l = 3*maxShellSize ; l < 3*maxShellSize + n4; ++l, ++mnkl) {

          auto MNKL = m + n*NB + k*NB2 + l*NB3;
          auto MNLK = m + n*NB + l*NB2 + k*NB3;
          auto NMKL = n + m*NB + k*NB2 + l*NB3;
          auto NMLK = n + m*NB + l*NB2 + k*NB3;
          auto KLMN = k + l*NB + m*NB2 + n*NB3;
          auto KLNM = k + l*NB + n*NB2 + m*NB3;
          auto LKMN = l + k*NB + m*NB2 + n*NB3;
          auto LKNM = l + k*NB + n*NB2 + m*NB3;

          /* Dirac-Coulomb */
          // ∇A∙∇B(mn|kl)
          auto dAdotdB = buff[AxBx][mnkl] + buff[AyBy][mnkl] + buff[AzBz][mnkl];
          // ∇Ax∇B(mn|kl)
          auto dAcrossdB_x =  buff[AyBz][mnkl] - buff[AzBy][mnkl];
          auto dAcrossdB_y = -buff[AxBz][mnkl] + buff[AzBx][mnkl];
          auto dAcrossdB_z =  buff[AxBy][mnkl] - buff[AyBx][mnkl];

          // ∇C∙∇D(mn|kl)
          auto dCdotdD = buff[CxDx][mnkl] + buff[CyDy][mnkl] + buff[CzDz][mnkl];
          // ∇Cx∇D(mn|kl)
          auto dCcrossdD_x =  buff[CyDz][mnkl] - buff[CzDy][mnkl];
          auto dCcrossdD_y = -buff[CxDz][mnkl] + buff[CzDx][mnkl];
          auto dCcrossdD_z =  buff[CxDy][mnkl] - buff[CyDx][mnkl];


          // ∇A∙∇B(mn|kl) followed by ∇Ax∇B(mn|kl) X, Y, and Z
          // (mn|kl)
	  ERIBuffAB[      MNKL] =  dAdotdB;
          ERIBuffAB[  NB4+MNKL] =  dAcrossdB_x;
          ERIBuffAB[2*NB4+MNKL] =  dAcrossdB_y;
          ERIBuffAB[3*NB4+MNKL] =  dAcrossdB_z;
          // (mn|lk)
	  ERIBuffAB[      MNLK] =  dAdotdB;
          ERIBuffAB[  NB4+MNLK] =  dAcrossdB_x;
          ERIBuffAB[2*NB4+MNLK] =  dAcrossdB_y;
          ERIBuffAB[3*NB4+MNLK] =  dAcrossdB_z;
          // (nm|kl)
	  ERIBuffAB[      NMKL] =  dAdotdB;
          ERIBuffAB[  NB4+NMKL] = -dAcrossdB_x;
          ERIBuffAB[2*NB4+NMKL] = -dAcrossdB_y;
          ERIBuffAB[3*NB4+NMKL] = -dAcrossdB_z;
          // (nm|lk)
	  ERIBuffAB[      NMLK] =  dAdotdB;
          ERIBuffAB[  NB4+NMLK] = -dAcrossdB_x;
          ERIBuffAB[2*NB4+NMLK] = -dAcrossdB_y;
          ERIBuffAB[3*NB4+NMLK] = -dAcrossdB_z;
          // (kl|mn)
	  ERIBuffAB[      KLMN] =  dCdotdD;
          ERIBuffAB[  NB4+KLMN] =  dCcrossdD_x;
          ERIBuffAB[2*NB4+KLMN] =  dCcrossdD_y;
          ERIBuffAB[3*NB4+KLMN] =  dCcrossdD_z;
          // (kl|nm)
	  ERIBuffAB[      KLNM] =  dCdotdD;
          ERIBuffAB[  NB4+KLNM] =  dCcrossdD_x;
          ERIBuffAB[2*NB4+KLNM] =  dCcrossdD_y;
          ERIBuffAB[3*NB4+KLNM] =  dCcrossdD_z;
          // (lk|mn)
	  ERIBuffAB[      LKMN] =  dCdotdD;
          ERIBuffAB[  NB4+LKMN] = -dCcrossdD_x;
          ERIBuffAB[2*NB4+LKMN] = -dCcrossdD_y;
          ERIBuffAB[3*NB4+LKMN] = -dCcrossdD_z;
          // (lk|nm)
	  ERIBuffAB[      LKNM] =  dCdotdD;
          ERIBuffAB[  NB4+LKNM] = -dCcrossdD_x;
          ERIBuffAB[2*NB4+LKNM] = -dCcrossdD_y;
          ERIBuffAB[3*NB4+LKNM] = -dCcrossdD_z;

          // ∇C∙∇D(mn|kl) followed by ∇Cx∇D(mn|kl) X, Y, and Z
          // (mn|kl)
	  ERIBuffCD[      MNKL] =  dCdotdD;
          ERIBuffCD[  NB4+MNKL] =  dCcrossdD_x;
          ERIBuffCD[2*NB4+MNKL] =  dCcrossdD_y;
          ERIBuffCD[3*NB4+MNKL] =  dCcrossdD_z;
          // (mn|lk)
	  ERIBuffCD[      MNLK] =  dCdotdD;
          ERIBuffCD[  NB4+MNLK] = -dCcrossdD_x;
          ERIBuffCD[2*NB4+MNLK] = -dCcrossdD_y;
          ERIBuffCD[3*NB4+MNLK] = -dCcrossdD_z;
          // (nm|kl)
	  ERIBuffCD[      NMKL] =  dCdotdD;
          ERIBuffCD[  NB4+NMKL] =  dCcrossdD_x;
          ERIBuffCD[2*NB4+NMKL] =  dCcrossdD_y;
          ERIBuffCD[3*NB4+NMKL] =  dCcrossdD_z;
          // (nm|lk)
	  ERIBuffCD[      NMLK] =  dCdotdD;
          ERIBuffCD[  NB4+NMLK] = -dCcrossdD_x;
          ERIBuffCD[2*NB4+NMLK] = -dCcrossdD_y;
          ERIBuffCD[3*NB4+NMLK] = -dCcrossdD_z;
          // (kl|mn)
	  ERIBuffCD[      KLMN] =  dAdotdB;
          ERIBuffCD[  NB4+KLMN] =  dAcrossdB_x;
          ERIBuffCD[2*NB4+KLMN] =  dAcrossdB_y;
          ERIBuffCD[3*NB4+KLMN] =  dAcrossdB_z;
          // (kl|nm)
	  ERIBuffCD[      KLNM] =  dAdotdB;
          ERIBuffCD[  NB4+KLNM] = -dAcrossdB_x;
          ERIBuffCD[2*NB4+KLNM] = -dAcrossdB_y;
          ERIBuffCD[3*NB4+KLNM] = -dAcrossdB_z;
          // (lk|mn)
	  ERIBuffCD[      LKMN] =  dAdotdB;
          ERIBuffCD[  NB4+LKMN] =  dAcrossdB_x;
          ERIBuffCD[2*NB4+LKMN] =  dAcrossdB_y;
          ERIBuffCD[3*NB4+LKMN] =  dAcrossdB_z;
          // (lk|nm)
	  ERIBuffCD[      LKNM] =  dAdotdB;
          ERIBuffCD[  NB4+LKNM] = -dAcrossdB_x;
          ERIBuffCD[2*NB4+LKNM] = -dAcrossdB_y;
          ERIBuffCD[3*NB4+LKNM] = -dAcrossdB_z;

	} // integral preparation loop

#if 0
        std::cout << "ERI00-03: ∇A∙∇B(ab|cd)  ∇Ax∇B(ab|cd)-X  ∇Ax∇B(ab|cd)-Y  ∇Ax∇B(ab|cd)-Z" << std::endl;
        for(m = 0ul,            bf1 = bf1_s; m <                  n1; ++m, bf1++) 
        for(n =   maxShellSize, bf2 = bf2_s; n <   maxShellSize + n2; ++n, bf2++) 
        for(k = 2*maxShellSize, bf3 = bf3_s; k < 2*maxShellSize + n3; ++k, bf3++) 
        for(l = 3*maxShellSize, bf4 = bf4_s; l < 3*maxShellSize + n4; ++l, bf4++) {

          auto MNKL = m + n*NB + k*NB2 + l*NB3;
          auto KLMN = k + l*NB + m*NB2 + n*NB3;
	  auto DotPrd = MNKL;
	  auto CrossX = MNKL+NB4;
	  auto CrossY = MNKL+2*NB4;
	  auto CrossZ = MNKL+3*NB4;

          std::cout << "(" << m << "," << n << "|" << k << "," << l << ")  ";
          std::cout << ERIBuffAB[DotPrd];
          std::cout << "   ";
          std::cout << ERIBuffAB[CrossX];
          std::cout << "   ";
          std::cout << ERIBuffAB[CrossY];
          std::cout << "   ";
          std::cout << ERIBuffAB[CrossZ] << std::endl;

        };
#endif

        for(m = 0ul,            bf1 = bf1_s; m <                  n1; ++m, bf1++) 
        for(n =   maxShellSize, bf2 = bf2_s; n <   maxShellSize + n2; ++n, bf2++) 
        for(k = 2*maxShellSize, bf3 = bf3_s; k < 2*maxShellSize + n3; ++k, bf3++) 
        for(l = 3*maxShellSize, bf4 = bf4_s; l < 3*maxShellSize + n4; ++l, bf4++) {


          auto MNKL = m + n*NB + k*NB2 + l*NB3;
          auto KLMN = k + l*NB + m*NB2 + n*NB3;
	  auto DotPrd = MNKL;
	  auto CrossX = MNKL+NB4;
	  auto CrossY = MNKL+2*NB4;
	  auto CrossZ = MNKL+3*NB4;



          /***********************************/
	  /* Dirac-Coulomb (LL|LL)           */
          /***********************************/
          auto AXMSTemp = AX_loc[LLMS];
          auto XMSTemp  = matList[SSMS].X;
          auto XMXTemp  = matList[SSMX].X;
          auto XMYTemp  = matList[SSMY].X;
          auto XMZTemp  = matList[SSMZ].X;

          //MNKL
          AXMSTemp[bf1 + bf2*nBasis] += ERIBuffCD[DotPrd]*XMSTemp[bf4 + bf3*nBasis];
          AXMSTemp[bf1 + bf2*nBasis] += ERIBuffCD[CrossX]*XMXTemp[bf4 + bf3*nBasis]*iscale;
          AXMSTemp[bf1 + bf2*nBasis] += ERIBuffCD[CrossY]*XMYTemp[bf4 + bf3*nBasis]*iscale;
          AXMSTemp[bf1 + bf2*nBasis] += ERIBuffCD[CrossZ]*XMZTemp[bf4 + bf3*nBasis]*iscale;

	  //NMKL
	  if(bf1_s!=bf2_s){
            AXMSTemp[bf2 + bf1*nBasis] += ERIBuffCD[DotPrd]*XMSTemp[bf4 + bf3*nBasis];
            AXMSTemp[bf2 + bf1*nBasis] += ERIBuffCD[CrossX]*XMXTemp[bf4 + bf3*nBasis]*iscale;
            AXMSTemp[bf2 + bf1*nBasis] += ERIBuffCD[CrossY]*XMYTemp[bf4 + bf3*nBasis]*iscale;
            AXMSTemp[bf2 + bf1*nBasis] += ERIBuffCD[CrossZ]*XMZTemp[bf4 + bf3*nBasis]*iscale;
	  }

	  //MNLK
	  if(bf3_s!=bf4_s){
            AXMSTemp[bf1 + bf2*nBasis] += ERIBuffCD[DotPrd]*XMSTemp[bf3 + bf4*nBasis];
            AXMSTemp[bf1 + bf2*nBasis] -= ERIBuffCD[CrossX]*XMXTemp[bf3 + bf4*nBasis]*iscale;
            AXMSTemp[bf1 + bf2*nBasis] -= ERIBuffCD[CrossY]*XMYTemp[bf3 + bf4*nBasis]*iscale;
            AXMSTemp[bf1 + bf2*nBasis] -= ERIBuffCD[CrossZ]*XMZTemp[bf3 + bf4*nBasis]*iscale;
	  }

	  //NMLK
	  if(bf1_s!=bf2_s and bf3_s!=bf4_s){
            AXMSTemp[bf2 + bf1*nBasis] += ERIBuffCD[DotPrd]*XMSTemp[bf3 + bf4*nBasis];
            AXMSTemp[bf2 + bf1*nBasis] -= ERIBuffCD[CrossX]*XMXTemp[bf3 + bf4*nBasis]*iscale;
            AXMSTemp[bf2 + bf1*nBasis] -= ERIBuffCD[CrossY]*XMYTemp[bf3 + bf4*nBasis]*iscale;
            AXMSTemp[bf2 + bf1*nBasis] -= ERIBuffCD[CrossZ]*XMZTemp[bf3 + bf4*nBasis]*iscale;
	  }

          //NMLK
          if(bf1_s!=bf3_s or bf2_s!=bf4_s){
 
            DotPrd = KLMN;
            CrossX = KLMN+NB4;
	    CrossY = KLMN+2*NB4;
	    CrossZ = KLMN+3*NB4;

            //KLMN
            AXMSTemp[bf3 + bf4*nBasis] += ERIBuffCD[DotPrd]*XMSTemp[bf2 + bf1*nBasis];
            AXMSTemp[bf3 + bf4*nBasis] += ERIBuffCD[CrossX]*XMXTemp[bf2 + bf1*nBasis]*iscale;
            AXMSTemp[bf3 + bf4*nBasis] += ERIBuffCD[CrossY]*XMYTemp[bf2 + bf1*nBasis]*iscale;
            AXMSTemp[bf3 + bf4*nBasis] += ERIBuffCD[CrossZ]*XMZTemp[bf2 + bf1*nBasis]*iscale;

	    //NMKL
	    if(bf3_s!=bf4_s){
              AXMSTemp[bf4 + bf3*nBasis] += ERIBuffCD[DotPrd]*XMSTemp[bf2 + bf1*nBasis];
              AXMSTemp[bf4 + bf3*nBasis] += ERIBuffCD[CrossX]*XMXTemp[bf2 + bf1*nBasis]*iscale;
              AXMSTemp[bf4 + bf3*nBasis] += ERIBuffCD[CrossY]*XMYTemp[bf2 + bf1*nBasis]*iscale;
              AXMSTemp[bf4 + bf3*nBasis] += ERIBuffCD[CrossZ]*XMZTemp[bf2 + bf1*nBasis]*iscale;
	    }

	    //MNLK
	    if(bf1_s!=bf2_s){
              AXMSTemp[bf3 + bf4*nBasis] += ERIBuffCD[DotPrd]*XMSTemp[bf1 + bf2*nBasis];
              AXMSTemp[bf3 + bf4*nBasis] -= ERIBuffCD[CrossX]*XMXTemp[bf1 + bf2*nBasis]*iscale;
              AXMSTemp[bf3 + bf4*nBasis] -= ERIBuffCD[CrossY]*XMYTemp[bf1 + bf2*nBasis]*iscale;
              AXMSTemp[bf3 + bf4*nBasis] -= ERIBuffCD[CrossZ]*XMZTemp[bf1 + bf2*nBasis]*iscale;
	    }

	    if(bf1_s!=bf2_s and bf3_s!=bf4_s){
              AXMSTemp[bf4 + bf3*nBasis] += ERIBuffCD[DotPrd]*XMSTemp[bf1 + bf2*nBasis];
              AXMSTemp[bf4 + bf3*nBasis] -= ERIBuffCD[CrossX]*XMXTemp[bf1 + bf2*nBasis]*iscale;
              AXMSTemp[bf4 + bf3*nBasis] -= ERIBuffCD[CrossY]*XMYTemp[bf1 + bf2*nBasis]*iscale;
              AXMSTemp[bf4 + bf3*nBasis] -= ERIBuffCD[CrossZ]*XMZTemp[bf1 + bf2*nBasis]*iscale;
	    }
	  }

          /***********************************/
	  /* Dirac-Coulomb (SS|SS)           */
          /***********************************/

	  DotPrd = MNKL;
	  CrossX = MNKL+NB4;
	  CrossY = MNKL+2*NB4;
	  CrossZ = MNKL+3*NB4;

          AXMSTemp  = AX_loc[SSMS];
          auto AXMXTemp  = AX_loc[SSMX];
          auto AXMYTemp  = AX_loc[SSMY];
          auto AXMZTemp  = AX_loc[SSMZ];
          XMSTemp = matList[LLMS].X;

	  //MNKL
          AXMSTemp[bf1 + bf2*nBasis] += ERIBuffAB[DotPrd]*XMSTemp[bf4 + bf3*nBasis];
          AXMXTemp[bf1 + bf2*nBasis] += ERIBuffAB[CrossX]*XMSTemp[bf4 + bf3*nBasis]*iscale;
          AXMYTemp[bf1 + bf2*nBasis] += ERIBuffAB[CrossY]*XMSTemp[bf4 + bf3*nBasis]*iscale;
          AXMZTemp[bf1 + bf2*nBasis] += ERIBuffAB[CrossZ]*XMSTemp[bf4 + bf3*nBasis]*iscale;

	  //NMKL
	  if(bf1_s!=bf2_s){
            AXMSTemp[bf2 + bf1*nBasis] += ERIBuffAB[DotPrd]*XMSTemp[bf4 + bf3*nBasis];
            AXMXTemp[bf2 + bf1*nBasis] -= ERIBuffAB[CrossX]*XMSTemp[bf4 + bf3*nBasis]*iscale;
            AXMYTemp[bf2 + bf1*nBasis] -= ERIBuffAB[CrossY]*XMSTemp[bf4 + bf3*nBasis]*iscale;
            AXMZTemp[bf2 + bf1*nBasis] -= ERIBuffAB[CrossZ]*XMSTemp[bf4 + bf3*nBasis]*iscale;
	  }

	  //MNLK
	  if(bf3_s!=bf4_s){
            AXMSTemp[bf1 + bf2*nBasis] += ERIBuffAB[DotPrd]*XMSTemp[bf3 + bf4*nBasis];
            AXMXTemp[bf1 + bf2*nBasis] += ERIBuffAB[CrossX]*XMSTemp[bf3 + bf4*nBasis]*iscale;
            AXMYTemp[bf1 + bf2*nBasis] += ERIBuffAB[CrossY]*XMSTemp[bf3 + bf4*nBasis]*iscale;
            AXMZTemp[bf1 + bf2*nBasis] += ERIBuffAB[CrossZ]*XMSTemp[bf3 + bf4*nBasis]*iscale;
	  }

	  //NMLK
	  if(bf1_s!=bf2_s and bf3_s!=bf4_s){
            AXMSTemp[bf2 + bf1*nBasis] += ERIBuffAB[DotPrd]*XMSTemp[bf3 + bf4*nBasis];
            AXMXTemp[bf2 + bf1*nBasis] -= ERIBuffAB[CrossX]*XMSTemp[bf3 + bf4*nBasis]*iscale;
            AXMYTemp[bf2 + bf1*nBasis] -= ERIBuffAB[CrossY]*XMSTemp[bf3 + bf4*nBasis]*iscale;
            AXMZTemp[bf2 + bf1*nBasis] -= ERIBuffAB[CrossZ]*XMSTemp[bf3 + bf4*nBasis]*iscale;
	  }

          if(bf1_s!=bf3_s or bf2_s!=bf4_s){
 
            DotPrd = KLMN;
            CrossX = KLMN+NB4;
	    CrossY = KLMN+2*NB4;
	    CrossZ = KLMN+3*NB4;

            //KLMN
            AXMSTemp[bf3 + bf4*nBasis] += ERIBuffAB[DotPrd]*XMSTemp[bf2 + bf1*nBasis];
            AXMXTemp[bf3 + bf4*nBasis] += ERIBuffAB[CrossX]*XMSTemp[bf2 + bf1*nBasis]*iscale;
            AXMYTemp[bf3 + bf4*nBasis] += ERIBuffAB[CrossY]*XMSTemp[bf2 + bf1*nBasis]*iscale;
            AXMZTemp[bf3 + bf4*nBasis] += ERIBuffAB[CrossZ]*XMSTemp[bf2 + bf1*nBasis]*iscale;

	    //NMKL
	    if(bf3_s!=bf4_s){
              AXMSTemp[bf4 + bf3*nBasis] += ERIBuffAB[DotPrd]*XMSTemp[bf2 + bf1*nBasis];
              AXMXTemp[bf4 + bf3*nBasis] -= ERIBuffAB[CrossX]*XMSTemp[bf2 + bf1*nBasis]*iscale;
              AXMYTemp[bf4 + bf3*nBasis] -= ERIBuffAB[CrossY]*XMSTemp[bf2 + bf1*nBasis]*iscale;
              AXMZTemp[bf4 + bf3*nBasis] -= ERIBuffAB[CrossZ]*XMSTemp[bf2 + bf1*nBasis]*iscale;
	    }

	    //MNLK
	    if(bf1_s!=bf2_s){
              AXMSTemp[bf3 + bf4*nBasis] += ERIBuffAB[DotPrd]*XMSTemp[bf1 + bf2*nBasis];
              AXMXTemp[bf3 + bf4*nBasis] += ERIBuffAB[CrossX]*XMSTemp[bf1 + bf2*nBasis]*iscale;
              AXMYTemp[bf3 + bf4*nBasis] += ERIBuffAB[CrossY]*XMSTemp[bf1 + bf2*nBasis]*iscale;
              AXMZTemp[bf3 + bf4*nBasis] += ERIBuffAB[CrossZ]*XMSTemp[bf1 + bf2*nBasis]*iscale;
	    }

	    if(bf1_s!=bf2_s and bf3_s!=bf4_s){
              AXMSTemp[bf4 + bf3*nBasis] += ERIBuffAB[DotPrd]*XMSTemp[bf1 + bf2*nBasis];
              AXMXTemp[bf4 + bf3*nBasis] -= ERIBuffAB[CrossX]*XMSTemp[bf1 + bf2*nBasis]*iscale;
              AXMYTemp[bf4 + bf3*nBasis] -= ERIBuffAB[CrossY]*XMSTemp[bf1 + bf2*nBasis]*iscale;
              AXMZTemp[bf4 + bf3*nBasis] -= ERIBuffAB[CrossZ]*XMSTemp[bf1 + bf2*nBasis]*iscale;
	    }
	  }


          /***********************************/
	  /* Dirac-Coulomb (LL|SS) / (SS|LL) */
          /***********************************/
	  DotPrd = MNKL;
	  CrossX = MNKL+NB4;
	  CrossY = MNKL+2*NB4;
	  CrossZ = MNKL+3*NB4;

          AXMSTemp  = AX_loc[LSMS];
          AXMXTemp  = AX_loc[LSMX];
          AXMYTemp  = AX_loc[LSMY];
          AXMZTemp  = AX_loc[LSMZ];
          XMSTemp = matList[LSMS].X;
          XMXTemp = matList[LSMX].X;
          XMYTemp = matList[LSMY].X;
          XMZTemp = matList[LSMZ].X;

          //MNKL
          AXMSTemp[bf1 + bf4*nBasis] -= ERIBuffCD[DotPrd]*XMSTemp[bf2 + bf3*nBasis];
          AXMSTemp[bf1 + bf4*nBasis] -= ERIBuffCD[CrossX]*XMXTemp[bf2 + bf3*nBasis]*iscale;
          AXMSTemp[bf1 + bf4*nBasis] -= ERIBuffCD[CrossY]*XMYTemp[bf2 + bf3*nBasis]*iscale;
          AXMSTemp[bf1 + bf4*nBasis] -= ERIBuffCD[CrossZ]*XMZTemp[bf2 + bf3*nBasis]*iscale;

          AXMXTemp[bf1 + bf4*nBasis] -= ERIBuffCD[DotPrd]*XMXTemp[bf2 + bf3*nBasis];
          AXMXTemp[bf1 + bf4*nBasis] -= ERIBuffCD[CrossX]*XMSTemp[bf2 + bf3*nBasis]*iscale;
          AXMXTemp[bf1 + bf4*nBasis] -= ERIBuffCD[CrossY]*XMZTemp[bf2 + bf3*nBasis];
          AXMXTemp[bf1 + bf4*nBasis] += ERIBuffCD[CrossZ]*XMYTemp[bf2 + bf3*nBasis];

          AXMYTemp[bf1 + bf4*nBasis] -= ERIBuffCD[DotPrd]*XMYTemp[bf2 + bf3*nBasis];
          AXMYTemp[bf1 + bf4*nBasis] += ERIBuffCD[CrossX]*XMZTemp[bf2 + bf3*nBasis];
          AXMYTemp[bf1 + bf4*nBasis] -= ERIBuffCD[CrossY]*XMSTemp[bf2 + bf3*nBasis]*iscale;
          AXMYTemp[bf1 + bf4*nBasis] -= ERIBuffCD[CrossZ]*XMXTemp[bf2 + bf3*nBasis];

          AXMZTemp[bf1 + bf4*nBasis] -= ERIBuffCD[DotPrd]*XMZTemp[bf2 + bf3*nBasis];
          AXMZTemp[bf1 + bf4*nBasis] -= ERIBuffCD[CrossX]*XMYTemp[bf2 + bf3*nBasis];
          AXMZTemp[bf1 + bf4*nBasis] += ERIBuffCD[CrossY]*XMXTemp[bf2 + bf3*nBasis];
          AXMZTemp[bf1 + bf4*nBasis] -= ERIBuffCD[CrossZ]*XMSTemp[bf2 + bf3*nBasis]*iscale;

          //MNLK
	  if(bf3_s!=bf4_s) {
            AXMSTemp[bf1 + bf3*nBasis] -= ERIBuffCD[DotPrd]*XMSTemp[bf2 + bf4*nBasis];
            AXMSTemp[bf1 + bf3*nBasis] += ERIBuffCD[CrossX]*XMXTemp[bf2 + bf4*nBasis]*iscale;
            AXMSTemp[bf1 + bf3*nBasis] += ERIBuffCD[CrossY]*XMYTemp[bf2 + bf4*nBasis]*iscale;
            AXMSTemp[bf1 + bf3*nBasis] += ERIBuffCD[CrossZ]*XMZTemp[bf2 + bf4*nBasis]*iscale;

            AXMXTemp[bf1 + bf3*nBasis] -= ERIBuffCD[DotPrd]*XMXTemp[bf2 + bf4*nBasis];
            AXMXTemp[bf1 + bf3*nBasis] += ERIBuffCD[CrossX]*XMSTemp[bf2 + bf4*nBasis]*iscale;
            AXMXTemp[bf1 + bf3*nBasis] += ERIBuffCD[CrossY]*XMZTemp[bf2 + bf4*nBasis];
            AXMXTemp[bf1 + bf3*nBasis] -= ERIBuffCD[CrossZ]*XMYTemp[bf2 + bf4*nBasis];

            AXMYTemp[bf1 + bf3*nBasis] -= ERIBuffCD[DotPrd]*XMYTemp[bf2 + bf4*nBasis];
            AXMYTemp[bf1 + bf3*nBasis] -= ERIBuffCD[CrossX]*XMZTemp[bf2 + bf4*nBasis];
            AXMYTemp[bf1 + bf3*nBasis] += ERIBuffCD[CrossY]*XMSTemp[bf2 + bf4*nBasis]*iscale;
            AXMYTemp[bf1 + bf3*nBasis] += ERIBuffCD[CrossZ]*XMXTemp[bf2 + bf4*nBasis];

            AXMZTemp[bf1 + bf3*nBasis] -= ERIBuffCD[DotPrd]*XMZTemp[bf2 + bf4*nBasis];
            AXMZTemp[bf1 + bf3*nBasis] += ERIBuffCD[CrossX]*XMYTemp[bf2 + bf4*nBasis];
            AXMZTemp[bf1 + bf3*nBasis] -= ERIBuffCD[CrossY]*XMXTemp[bf2 + bf4*nBasis];
            AXMZTemp[bf1 + bf3*nBasis] += ERIBuffCD[CrossZ]*XMSTemp[bf2 + bf4*nBasis]*iscale;
	  }

	  //NMKL
	  if(bf1_s!=bf2_s){
            AXMSTemp[bf2 + bf4*nBasis] -= ERIBuffCD[DotPrd]*XMSTemp[bf1 + bf3*nBasis];
            AXMSTemp[bf2 + bf4*nBasis] -= ERIBuffCD[CrossX]*XMXTemp[bf1 + bf3*nBasis]*iscale;
            AXMSTemp[bf2 + bf4*nBasis] -= ERIBuffCD[CrossY]*XMYTemp[bf1 + bf3*nBasis]*iscale;
            AXMSTemp[bf2 + bf4*nBasis] -= ERIBuffCD[CrossZ]*XMZTemp[bf1 + bf3*nBasis]*iscale;

            AXMXTemp[bf2 + bf4*nBasis] -= ERIBuffCD[DotPrd]*XMXTemp[bf1 + bf3*nBasis];
            AXMXTemp[bf2 + bf4*nBasis] -= ERIBuffCD[CrossX]*XMSTemp[bf1 + bf3*nBasis]*iscale;
            AXMXTemp[bf2 + bf4*nBasis] -= ERIBuffCD[CrossY]*XMZTemp[bf1 + bf3*nBasis];
            AXMXTemp[bf2 + bf4*nBasis] += ERIBuffCD[CrossZ]*XMYTemp[bf1 + bf3*nBasis];

            AXMYTemp[bf2 + bf4*nBasis] -= ERIBuffCD[DotPrd]*XMYTemp[bf1 + bf3*nBasis];
            AXMYTemp[bf2 + bf4*nBasis] += ERIBuffCD[CrossX]*XMZTemp[bf1 + bf3*nBasis];
            AXMYTemp[bf2 + bf4*nBasis] -= ERIBuffCD[CrossY]*XMSTemp[bf1 + bf3*nBasis]*iscale;
            AXMYTemp[bf2 + bf4*nBasis] -= ERIBuffCD[CrossZ]*XMXTemp[bf1 + bf3*nBasis];

            AXMZTemp[bf2 + bf4*nBasis] -= ERIBuffCD[DotPrd]*XMZTemp[bf1 + bf3*nBasis];
            AXMZTemp[bf2 + bf4*nBasis] -= ERIBuffCD[CrossX]*XMYTemp[bf1 + bf3*nBasis];
            AXMZTemp[bf2 + bf4*nBasis] += ERIBuffCD[CrossY]*XMXTemp[bf1 + bf3*nBasis];
            AXMZTemp[bf2 + bf4*nBasis] -= ERIBuffCD[CrossZ]*XMSTemp[bf1 + bf3*nBasis]*iscale;

	    if(bf3_s!=bf4_s) {
              AXMSTemp[bf2 + bf3*nBasis] -= ERIBuffCD[DotPrd]*XMSTemp[bf1 + bf4*nBasis];
              AXMSTemp[bf2 + bf3*nBasis] += ERIBuffCD[CrossX]*XMXTemp[bf1 + bf4*nBasis]*iscale;
              AXMSTemp[bf2 + bf3*nBasis] += ERIBuffCD[CrossY]*XMYTemp[bf1 + bf4*nBasis]*iscale;
              AXMSTemp[bf2 + bf3*nBasis] += ERIBuffCD[CrossZ]*XMZTemp[bf1 + bf4*nBasis]*iscale;

              AXMXTemp[bf2 + bf3*nBasis] -= ERIBuffCD[DotPrd]*XMXTemp[bf1 + bf4*nBasis];
              AXMXTemp[bf2 + bf3*nBasis] += ERIBuffCD[CrossX]*XMSTemp[bf1 + bf4*nBasis]*iscale;
              AXMXTemp[bf2 + bf3*nBasis] += ERIBuffCD[CrossY]*XMZTemp[bf1 + bf4*nBasis];
              AXMXTemp[bf2 + bf3*nBasis] -= ERIBuffCD[CrossZ]*XMYTemp[bf1 + bf4*nBasis];

              AXMYTemp[bf2 + bf3*nBasis] -= ERIBuffCD[DotPrd]*XMYTemp[bf1 + bf4*nBasis];
              AXMYTemp[bf2 + bf3*nBasis] -= ERIBuffCD[CrossX]*XMZTemp[bf1 + bf4*nBasis];
              AXMYTemp[bf2 + bf3*nBasis] += ERIBuffCD[CrossY]*XMSTemp[bf1 + bf4*nBasis]*iscale;
              AXMYTemp[bf2 + bf3*nBasis] += ERIBuffCD[CrossZ]*XMXTemp[bf1 + bf4*nBasis];

              AXMZTemp[bf2 + bf3*nBasis] -= ERIBuffCD[DotPrd]*XMZTemp[bf1 + bf4*nBasis];
              AXMZTemp[bf2 + bf3*nBasis] += ERIBuffCD[CrossX]*XMYTemp[bf1 + bf4*nBasis];
              AXMZTemp[bf2 + bf3*nBasis] -= ERIBuffCD[CrossY]*XMXTemp[bf1 + bf4*nBasis];
              AXMZTemp[bf2 + bf3*nBasis] += ERIBuffCD[CrossZ]*XMSTemp[bf1 + bf4*nBasis]*iscale;
	    }
	  }

          if(bf1_s!=bf3_s or bf2_s!=bf4_s){
 
            DotPrd = KLMN;
            CrossX = KLMN+NB4;
	    CrossY = KLMN+2*NB4;
	    CrossZ = KLMN+3*NB4;

            //KLMN
            AXMSTemp[bf3 + bf2*nBasis] -= ERIBuffCD[DotPrd]*XMSTemp[bf4 + bf1*nBasis];
            AXMSTemp[bf3 + bf2*nBasis] -= ERIBuffCD[CrossX]*XMXTemp[bf4 + bf1*nBasis]*iscale;
            AXMSTemp[bf3 + bf2*nBasis] -= ERIBuffCD[CrossY]*XMYTemp[bf4 + bf1*nBasis]*iscale;
            AXMSTemp[bf3 + bf2*nBasis] -= ERIBuffCD[CrossZ]*XMZTemp[bf4 + bf1*nBasis]*iscale;
  
            AXMXTemp[bf3 + bf2*nBasis] -= ERIBuffCD[DotPrd]*XMXTemp[bf4 + bf1*nBasis];
            AXMXTemp[bf3 + bf2*nBasis] -= ERIBuffCD[CrossX]*XMSTemp[bf4 + bf1*nBasis]*iscale;
            AXMXTemp[bf3 + bf2*nBasis] -= ERIBuffCD[CrossY]*XMZTemp[bf4 + bf1*nBasis];
            AXMXTemp[bf3 + bf2*nBasis] += ERIBuffCD[CrossZ]*XMYTemp[bf4 + bf1*nBasis];
  
            AXMYTemp[bf3 + bf2*nBasis] -= ERIBuffCD[DotPrd]*XMYTemp[bf4 + bf1*nBasis];
            AXMYTemp[bf3 + bf2*nBasis] += ERIBuffCD[CrossX]*XMZTemp[bf4 + bf1*nBasis];
            AXMYTemp[bf3 + bf2*nBasis] -= ERIBuffCD[CrossY]*XMSTemp[bf4 + bf1*nBasis]*iscale;
            AXMYTemp[bf3 + bf2*nBasis] -= ERIBuffCD[CrossZ]*XMXTemp[bf4 + bf1*nBasis];
  
            AXMZTemp[bf3 + bf2*nBasis] -= ERIBuffCD[DotPrd]*XMZTemp[bf4 + bf1*nBasis];
            AXMZTemp[bf3 + bf2*nBasis] -= ERIBuffCD[CrossX]*XMYTemp[bf4 + bf1*nBasis];
            AXMZTemp[bf3 + bf2*nBasis] += ERIBuffCD[CrossY]*XMXTemp[bf4 + bf1*nBasis];
            AXMZTemp[bf3 + bf2*nBasis] -= ERIBuffCD[CrossZ]*XMSTemp[bf4 + bf1*nBasis]*iscale;
  
  	    //KLNM
  	    if(bf3_s!=bf4_s){
              AXMSTemp[bf4 + bf2*nBasis] -= ERIBuffCD[DotPrd]*XMSTemp[bf3 + bf1*nBasis];
              AXMSTemp[bf4 + bf2*nBasis] -= ERIBuffCD[CrossX]*XMXTemp[bf3 + bf1*nBasis]*iscale;
              AXMSTemp[bf4 + bf2*nBasis] -= ERIBuffCD[CrossY]*XMYTemp[bf3 + bf1*nBasis]*iscale;
              AXMSTemp[bf4 + bf2*nBasis] -= ERIBuffCD[CrossZ]*XMZTemp[bf3 + bf1*nBasis]*iscale;
  
              AXMXTemp[bf4 + bf2*nBasis] -= ERIBuffCD[DotPrd]*XMXTemp[bf3 + bf1*nBasis];
              AXMXTemp[bf4 + bf2*nBasis] -= ERIBuffCD[CrossX]*XMSTemp[bf3 + bf1*nBasis]*iscale;
              AXMXTemp[bf4 + bf2*nBasis] -= ERIBuffCD[CrossY]*XMZTemp[bf3 + bf1*nBasis];
              AXMXTemp[bf4 + bf2*nBasis] += ERIBuffCD[CrossZ]*XMYTemp[bf3 + bf1*nBasis];
  
              AXMYTemp[bf4 + bf2*nBasis] -= ERIBuffCD[DotPrd]*XMYTemp[bf3 + bf1*nBasis];
              AXMYTemp[bf4 + bf2*nBasis] += ERIBuffCD[CrossX]*XMZTemp[bf3 + bf1*nBasis];
              AXMYTemp[bf4 + bf2*nBasis] -= ERIBuffCD[CrossY]*XMSTemp[bf3 + bf1*nBasis]*iscale;
              AXMYTemp[bf4 + bf2*nBasis] -= ERIBuffCD[CrossZ]*XMXTemp[bf3 + bf1*nBasis];
  
              AXMZTemp[bf4 + bf2*nBasis] -= ERIBuffCD[DotPrd]*XMZTemp[bf3 + bf1*nBasis];
              AXMZTemp[bf4 + bf2*nBasis] -= ERIBuffCD[CrossX]*XMYTemp[bf3 + bf1*nBasis];
              AXMZTemp[bf4 + bf2*nBasis] += ERIBuffCD[CrossY]*XMXTemp[bf3 + bf1*nBasis];
              AXMZTemp[bf4 + bf2*nBasis] -= ERIBuffCD[CrossZ]*XMSTemp[bf3 + bf1*nBasis]*iscale;
  	    }
  
  	    //LKMN
  	    if(bf1_s!=bf2_s) {
              AXMSTemp[bf3 + bf1*nBasis] -= ERIBuffCD[DotPrd]*XMSTemp[bf4 + bf2*nBasis];
              AXMSTemp[bf3 + bf1*nBasis] += ERIBuffCD[CrossX]*XMXTemp[bf4 + bf2*nBasis]*iscale;
              AXMSTemp[bf3 + bf1*nBasis] += ERIBuffCD[CrossY]*XMYTemp[bf4 + bf2*nBasis]*iscale;
              AXMSTemp[bf3 + bf1*nBasis] += ERIBuffCD[CrossZ]*XMZTemp[bf4 + bf2*nBasis]*iscale;
  
              AXMXTemp[bf3 + bf1*nBasis] -= ERIBuffCD[DotPrd]*XMXTemp[bf4 + bf2*nBasis];
              AXMXTemp[bf3 + bf1*nBasis] += ERIBuffCD[CrossX]*XMSTemp[bf4 + bf2*nBasis]*iscale;
              AXMXTemp[bf3 + bf1*nBasis] += ERIBuffCD[CrossY]*XMZTemp[bf4 + bf2*nBasis];
              AXMXTemp[bf3 + bf1*nBasis] -= ERIBuffCD[CrossZ]*XMYTemp[bf4 + bf2*nBasis];
  
              AXMYTemp[bf3 + bf1*nBasis] -= ERIBuffCD[DotPrd]*XMYTemp[bf4 + bf2*nBasis];
              AXMYTemp[bf3 + bf1*nBasis] -= ERIBuffCD[CrossX]*XMZTemp[bf4 + bf2*nBasis];
              AXMYTemp[bf3 + bf1*nBasis] += ERIBuffCD[CrossY]*XMSTemp[bf4 + bf2*nBasis]*iscale;
              AXMYTemp[bf3 + bf1*nBasis] += ERIBuffCD[CrossZ]*XMXTemp[bf4 + bf2*nBasis];
  
              AXMZTemp[bf3 + bf1*nBasis] -= ERIBuffCD[DotPrd]*XMZTemp[bf4 + bf2*nBasis];
              AXMZTemp[bf3 + bf1*nBasis] += ERIBuffCD[CrossX]*XMYTemp[bf4 + bf2*nBasis];
              AXMZTemp[bf3 + bf1*nBasis] -= ERIBuffCD[CrossY]*XMXTemp[bf4 + bf2*nBasis];
              AXMZTemp[bf3 + bf1*nBasis] += ERIBuffCD[CrossZ]*XMSTemp[bf4 + bf2*nBasis]*iscale;
  
  	      //LKNM
  	      if(bf4_s!=bf3_s) {
                AXMSTemp[bf4 + bf1*nBasis] -= ERIBuffCD[DotPrd]*XMSTemp[bf3 + bf2*nBasis];
                AXMSTemp[bf4 + bf1*nBasis] += ERIBuffCD[CrossX]*XMXTemp[bf3 + bf2*nBasis]*iscale;
                AXMSTemp[bf4 + bf1*nBasis] += ERIBuffCD[CrossY]*XMYTemp[bf3 + bf2*nBasis]*iscale;
                AXMSTemp[bf4 + bf1*nBasis] += ERIBuffCD[CrossZ]*XMZTemp[bf3 + bf2*nBasis]*iscale;
  
                AXMXTemp[bf4 + bf1*nBasis] -= ERIBuffCD[DotPrd]*XMXTemp[bf3 + bf2*nBasis];
                AXMXTemp[bf4 + bf1*nBasis] += ERIBuffCD[CrossX]*XMSTemp[bf3 + bf2*nBasis]*iscale;
                AXMXTemp[bf4 + bf1*nBasis] += ERIBuffCD[CrossY]*XMZTemp[bf3 + bf2*nBasis];
                AXMXTemp[bf4 + bf1*nBasis] -= ERIBuffCD[CrossZ]*XMYTemp[bf3 + bf2*nBasis];
  
                AXMYTemp[bf4 + bf1*nBasis] -= ERIBuffCD[DotPrd]*XMYTemp[bf3 + bf2*nBasis];
                AXMYTemp[bf4 + bf1*nBasis] -= ERIBuffCD[CrossX]*XMZTemp[bf3 + bf2*nBasis];
                AXMYTemp[bf4 + bf1*nBasis] += ERIBuffCD[CrossY]*XMSTemp[bf3 + bf2*nBasis]*iscale;
                AXMYTemp[bf4 + bf1*nBasis] += ERIBuffCD[CrossZ]*XMXTemp[bf3 + bf2*nBasis];
  
                AXMZTemp[bf4 + bf1*nBasis] -= ERIBuffCD[DotPrd]*XMZTemp[bf3 + bf2*nBasis];
                AXMZTemp[bf4 + bf1*nBasis] += ERIBuffCD[CrossX]*XMYTemp[bf3 + bf2*nBasis];
                AXMZTemp[bf4 + bf1*nBasis] -= ERIBuffCD[CrossY]*XMXTemp[bf3 + bf2*nBasis];
                AXMZTemp[bf4 + bf1*nBasis] += ERIBuffCD[CrossZ]*XMSTemp[bf3 + bf2*nBasis]*iscale;
  	      }
	    }

	  }

        } // contraction loop

#endif // Contraction


    }; // loop s4
    }; // loop s3
    }; // loop s2
    }; // loop s1


    }; // OpenMP context


    size_t nIntSkip = std::accumulate(nSkip.begin(),nSkip.end(),0);
    std::cout << "Screened " << nIntSkip << std::endl;

    auto durDirect = tock(topDirect);
    std::cout << "Direct Contraction took " <<  durDirect << " s\n"; 

    std::cout << std::endl;


    if (nThreads>1)
    for( auto iMat = 0; iMat < nMat;  iMat++ ) 
    for( auto iTh  = 0; iTh < nThreads; iTh++) {
 
      MatAdd('N','N',nBasis,nBasis,MatsT(1.0),AXthreads[iTh][iMat],nBasis,MatsT(1.0),
         matList[iMat].AX,nBasis,matList[iMat].AX,nBasis);

    };
    

#ifdef _SHZ_SCREEN_4C
    memManager_.free(ShBlkNorms_raw);
#endif

    if(AXRaw != nullptr) memManager_.free(AXRaw);
    if(ERIBuffer != nullptr) memManager_.free(ERIBuffer);

    // Turn threads for LA back on
    SetLAThreads(LAThreads);

  }

  template <>
  void GTODirectRelERIContraction<double,double>::directScaffold(
    MPI_Comm comm, const bool screen,
    std::vector<TwoBodyContraction<double>> &matList) const {
    CErr("Dirac-Coulomb + Real is an invalid option",std::cout);  
  }

  template <>
  void GTODirectRelERIContraction<dcomplex,dcomplex>::directScaffold(
    MPI_Comm comm, const bool screen,
    std::vector<TwoBodyContraction<dcomplex>> &matList) const {
    CErr("Complex integral is is an invalid option",std::cout);  
  }











  // For DCB Hamitlonian, 
  // 12 density matrices upon input stored as
  // LL(MS,MX,MY,MZ), SS(MS,MX,MY,MZ), LS(MS,MX,MY,MZ)
  //
  // 12 contrated matrices upon output stored as
  // LL(MS,MX,MY,MZ), SS(MS,MX,MY,MZ), LS(MS,MX,MY,MZ)
  //
 
  template <typename MatsT, typename IntsT>
  void GTODirectRelERIContraction<MatsT,IntsT>::directScaffoldLibcint(
    MPI_Comm comm, const bool screen,
    std::vector<TwoBodyContraction<MatsT>> &matList) const {

    DirectERI<IntsT> &eri =
        dynamic_cast<DirectERI<IntsT>&>(this->ints_);
    CQMemManager& memManager_ = eri.memManager();
    BasisSet& basisSet_ = eri.basisSet();
    Molecule& molecule_ = eri.molecule();

 
    // Determine the number of OpenMP threads
    size_t nThreads  = GetNumThreads();
    size_t LAThreads = GetLAThreads();
    SetLAThreads(1); // Turn off parallelism in LA functions



    /****************************/
    /* Format Basis for Libcint */
    /****************************/

    int nAtoms = molecule_.nAtoms;
    int nShells = basisSet_.nShell;
    int iAtom, iShell, off;

    // ATM_SLOTS = 6; BAS_SLOTS = 8;
    int *atm = memManager_.template malloc<int>(nAtoms * ATM_SLOTS);
    int *bas = memManager_.template malloc<int>(nShells * BAS_SLOTS);
    double *env = memManager_.template malloc<double>(nAtoms*3+nShells*basisSet_.maxPrim*2);
    double sNorm;

    off = PTR_ENV_START; // = 20

    for(iAtom = 0; iAtom < nAtoms; iAtom++) {

      atm[CHARGE_OF + ATM_SLOTS * iAtom] = molecule_.atoms[iAtom].atomicNumber;
      atm[PTR_COORD + ATM_SLOTS * iAtom] = off;
      env[off + 0] = molecule_.atoms[iAtom].coord[0]; // x (Bohr)
      env[off + 1] = molecule_.atoms[iAtom].coord[1]; // y (Bohr)
      env[off + 2] = molecule_.atoms[iAtom].coord[2]; // z (Bohr)
      off += 3;

    }

    for(iShell = 0; iShell < nShells; iShell++) {

      bas[ATOM_OF  + BAS_SLOTS * iShell]  = basisSet_.mapSh2Cen[iShell];
      bas[ANG_OF   + BAS_SLOTS * iShell]  = basisSet_.shells[iShell].contr[0].l;
      bas[NPRIM_OF + BAS_SLOTS * iShell]  = basisSet_.shells[iShell].alpha.size();
      bas[NCTR_OF  + BAS_SLOTS * iShell]  = 1;
      bas[PTR_EXP  + BAS_SLOTS * iShell]  = off;

      for(int iPrim=0; iPrim<basisSet_.shells[iShell].alpha.size(); iPrim++)
        env[off + iPrim] = basisSet_.shells[iShell].alpha[iPrim];

      off +=basisSet_.shells[iShell].alpha.size();

      bas[PTR_COEFF+ BAS_SLOTS * iShell] = off;

      // Spherical GTO normalization constant missing in Libcint
      sNorm = 2.0*std::sqrt(M_PI)/std::sqrt(2.0*basisSet_.shells[iShell].contr[0].l+1.0);
      for(int iCoeff=0; iCoeff<basisSet_.shells[iShell].alpha.size(); iCoeff++){
        env[off + iCoeff] = basisSet_.shells[iShell].contr[0].coeff[iCoeff]*sNorm;
     }

      off += basisSet_.shells[iShell].alpha.size();
    }

    int AxBx = 0;
    int AxBy = 1;
    int AxBz = 2;
    int AyBx = 3;
    int AyBy = 4;
    int AyBz = 5;
    int AzBx = 6;
    int AzBy = 7;
    int AzBz = 8;

    /*-------------------------------------*/
    /* End of Basis Formatting for Libcint */
    /*-------------------------------------*/



    // Get threads result buffer
    int buffSize = (basisSet_.maxL+1)*(basisSet_.maxL+2)/2;
    int buffN4 = buffSize*buffSize*buffSize*buffSize;
    double *buffAll = memManager_.malloc<double>(9*buffN4*nThreads);

    size_t maxShellSize = 
      std::max_element(basisSet_.shells.begin(),basisSet_.shells.end(),
        [](libint2::Shell &sh1, libint2::Shell &sh2) {
          return sh1.size() < sh2.size();
        })->size();
  
    //size_t maxShellSize = buffSize;
    size_t NB  = maxShellSize*4;
    size_t NB2 = NB*NB;
    size_t NB3 = NB2*NB;
    size_t NB4 = NB2*NB2;

    const size_t nBasis   = basisSet_.nBasis;
    const size_t nMat     = matList.size();
    const size_t nShell   = basisSet_.nShell;

    double *ERIBuffer = memManager_.malloc<double>(2*4*NB4*nThreads);

    // For DCB Hamitlonian, 
    // 12 density matrices upon input stored as
    // LL(MS,MX,MY,MZ), SS(MS,MX,MY,MZ), LS(MS,MX,MY,MZ)
    //
    // 12 contrated matrices upon output stored as
    // LL(MS,MX,MY,MZ), SS(MS,MX,MY,MZ), LS(MS,MX,MY,MZ)
    //
    // Allocate thread local storage to store integral contractions

    int LLMS = 0;
    int LLMX = 1;
    int LLMY = 2;
    int LLMZ = 3;
    int SSMS = 4;
    int SSMX = 5;
    int SSMY = 6;
    int SSMZ = 7;
    int LSMS = 8;
    int LSMX = 9;
    int LSMY = 10;
    int LSMZ = 11;


    for(auto iMat = 0; iMat < nMat; iMat++)
      memset(matList[iMat].AX,0.,nBasis*nBasis*sizeof(MatsT));

    std::vector<std::vector<MatsT*>> AXthreads;
    MatsT *AXRaw = nullptr;
    if(nThreads != 1) {
      AXRaw = memManager_.malloc<MatsT>(nThreads*nMat*nBasis*nBasis);    
      memset(AXRaw,0,nThreads*nMat*nBasis*nBasis*sizeof(MatsT));
    }

    if(nThreads == 1) {
      AXthreads.emplace_back();
      for(auto iMat = 0; iMat < nMat; iMat++)
        AXthreads.back().push_back(matList[iMat].AX);
    } else {
      for(auto iThread = 0; iThread < nThreads; iThread++) {
        AXthreads.emplace_back();
        for(auto iMat = 0; iMat < nMat; iMat++) 
          AXthreads.back().push_back(AXRaw + iThread*nMat*nBasis*nBasis + iMat*nBasis*nBasis);
      }
    }

#ifdef _SHZ_SCREEN_4C
    // Compute shell block norms (∞-norm) of matList.X
    if(eri.schwartz() == nullptr) eri.computeSchwartz();
    double *ShBlkNorms_raw = memManager_.malloc<double>(nMat*nShell*nShell);
    std::vector<double*> ShBlkNorms;
    for(auto iMat = 0, iOff = 0; iMat < nMat; iMat++, iOff += nShell*nShell ) {
      ShellBlockNorm(basisSet_.shells,matList[iMat].X,nBasis,ShBlkNorms_raw + iOff);
      ShBlkNorms.emplace_back(ShBlkNorms_raw + iOff);
    }

    // Find the max value of shell block ∞-norms of all matList.X
    double maxShBlkNorm = 0.;
    for(auto iMat = 0; iMat < nMat; iMat++)
      maxShBlkNorm = std::max(maxShBlkNorm,
        *std::max_element(ShBlkNorms[iMat],ShBlkNorms[iMat] + nShell*nShell) ); 

    size_t maxnPrim4 = 
      basisSet_.maxPrim * basisSet_.maxPrim * basisSet_.maxPrim * 
      basisSet_.maxPrim;

    // Get the max over all the matricies for the shell block ∞-norms
    // OVERWRITES ShBlkNorms[0]
    for(auto k = 0; k < nShell*nShell; k++) {
      double mx = std::abs(ShBlkNorms[0][k]);
      for(auto iMat = 1; iMat < nMat; iMat++)
        mx = std::max(mx,std::abs(ShBlkNorms[iMat][k]));
      ShBlkNorms[0][k] = mx;
    }

#endif

    // Keeping track of number of integrals skipped
    std::vector<size_t> nSkip(nThreads,0);

    dcomplex iscale = dcomplex(0.0, 1.0);

#ifdef _REPORT_INTEGRAL_TIMINGS
    std::cout<<"Using Libcint "<<std::endl;
    auto topDirect = tick();
#endif

    #pragma omp parallel
    {

      size_t thread_id = GetThreadID();

      auto &AX_loc = AXthreads[thread_id];

      double *ERIBuffAB = &ERIBuffer[thread_id*4*NB4];
      double *ERIBuffCD = &ERIBuffer[nThreads*4*NB4 + thread_id*4*NB4];

      size_t n1,n2,n3,n4,m,n,k,l,mnkl,bf1,bf2,bf3,bf4;
      size_t s4_max;

      int shls[4];
      double *buff = &buffAll[9*buffN4*thread_id];


      for(size_t s1(0), bf1_s(0), s1234(0); s1 < basisSet_.nShell; 
          bf1_s+=n1, s1++) { 

        n1 = basisSet_.shells[s1].size(); // Size of Shell 1

      for(size_t s2(0), bf2_s(0); s2 <= s1; bf2_s+=n2, s2++) {

        n2 = basisSet_.shells[s2].size(); // Size of Shell 2

#ifdef _SHZ_SCREEN_4C
        double shz12 = 0, shMax12 = 0;
        if( screen ) {
          shz12 = eri.schwartz()[s1 + s2*nShell];
          shMax12 = ShBlkNorms[0][s1 + s2*nShell];
        }
#endif

      for(size_t s3(0), bf3_s(0); s3 <= s1; bf3_s+=n3, s3++) {

        n3 = basisSet_.shells[s3].size(); // Size of Shell 3
        s4_max = (s1 == s3) ? s2 : s3; // Determine the unique max of Shell 4

#ifdef _SHZ_SCREEN_4C

        double shMax123 = 0;
        if( screen ) {
          // Pre-calculate shell-block norm max's that only
          // depend on shells 1,2 and 3
          shMax123 = 
            std::max(ShBlkNorms[0][s1 + s3*nShell], 
                     ShBlkNorms[0][s2 + s3*nShell]);

          shMax123 = std::max(shMax123,shMax12);
        }

#endif
 
      for(size_t s4(0), bf4_s(0); s4 <= s4_max; bf4_s+=n4, s4++, s1234++) {

        n4 = basisSet_.shells[s4].size(); // Size of Shell 4

        // Round Robbin work distribution
        #ifdef _OPENMP
        if( s1234 % nThreads != thread_id ) continue;
        #endif


#ifdef _SHZ_SCREEN_4C

        double shMax = 0;

        if( screen ) {
          // Compute Shell norm max
          shMax = 
            std::max(ShBlkNorms[0][s1 + s4*nShell],
            std::max(ShBlkNorms[0][s2 + s4*nShell],
                     ShBlkNorms[0][s3 + s4*nShell]));

          shMax = std::max(shMax,shMax123);

          if((shMax *shMax * shz12 * eri.schwartz()[s3 + s4*nShell]) <
             eri.threshSchwartz()) { nSkip[thread_id]++; continue; }
        }
 
#endif

	//memset(ERIBuffCD,0.0,4*NB4);
	//memset(ERIBuffAB,0.0,4*NB4);
  	auto nQuad = n1*n2*n3*n4;

        shls[0] = int(s1);
        shls[1] = int(s2);
        shls[2] = int(s3);
        shls[3] = int(s4);

#ifdef _CONTRACTION_ // Contraction
        //if(cint2e_sph(buff, shls, atm, nAtoms, bas, nShell, env, nullptr)!=0)
        if(cint2e_ipvip1_sph(buff, shls, atm, nAtoms, bas, nShells, env, nullptr)!=0)
        //if(cint2e_ip1ip2_sph(buff, shls, atm, nAtoms, bas, nShells, env, nullptr)!=0)
        for(l = 3*maxShellSize, mnkl = 0ul; l < 3*maxShellSize + n4; ++l)
        for(k = 2*maxShellSize            ; k < 2*maxShellSize + n3; ++k) 
        for(n =   maxShellSize            ; n <   maxShellSize + n2; ++n) 
        for(m = 0                         ; m <                  n1; ++m, ++mnkl) {
 
          auto MNKL = m + n*NB + k*NB2 + l*NB3;
          //auto MNLK = m + n*NB + l*NB2 + k*NB3;
          //auto NMKL = n + m*NB + k*NB2 + l*NB3;
          //auto NMLK = n + m*NB + l*NB2 + k*NB3;
          auto KLMN = k + l*NB + m*NB2 + n*NB3;
          //auto KLNM = k + l*NB + n*NB2 + m*NB3;
          //auto LKMN = l + k*NB + m*NB2 + n*NB3;
          //auto LKNM = l + k*NB + n*NB2 + m*NB3;

          /* Dirac-Coulomb */
          // ∇A∙∇B(mn|kl)
          auto dAdotdB = buff[AxBx*nQuad+mnkl] + buff[AyBy*nQuad+mnkl] + buff[AzBz*nQuad+mnkl];
          // ∇Ax∇B(mn|kl)
          auto dAcrossdB_x =  buff[AyBz*nQuad+mnkl] - buff[AzBy*nQuad+mnkl];
          auto dAcrossdB_y = -buff[AxBz*nQuad+mnkl] + buff[AzBx*nQuad+mnkl];
          auto dAcrossdB_z =  buff[AxBy*nQuad+mnkl] - buff[AyBx*nQuad+mnkl];


          // ∇A∙∇B(mn|kl) followed by ∇Ax∇B(mn|kl) X, Y, and Z
          // (mn|kl)
	  ERIBuffAB[      MNKL] =  dAdotdB;
          ERIBuffAB[  NB4+MNKL] =  dAcrossdB_x;
          ERIBuffAB[2*NB4+MNKL] =  dAcrossdB_y;
          ERIBuffAB[3*NB4+MNKL] =  dAcrossdB_z;
#if 0
          // (mn|lk)
	  ERIBuffAB[      MNLK] =  dAdotdB;
          ERIBuffAB[  NB4+MNLK] =  dAcrossdB_x;
          ERIBuffAB[2*NB4+MNLK] =  dAcrossdB_y;
          ERIBuffAB[3*NB4+MNLK] =  dAcrossdB_z;
          // (nm|kl)
	  ERIBuffAB[      NMKL] =  dAdotdB;
          ERIBuffAB[  NB4+NMKL] = -dAcrossdB_x;
          ERIBuffAB[2*NB4+NMKL] = -dAcrossdB_y;
          ERIBuffAB[3*NB4+NMKL] = -dAcrossdB_z;
          // (nm|lk)
	  ERIBuffAB[      NMLK] =  dAdotdB;
          ERIBuffAB[  NB4+NMLK] = -dAcrossdB_x;
          ERIBuffAB[2*NB4+NMLK] = -dAcrossdB_y;
          ERIBuffAB[3*NB4+NMLK] = -dAcrossdB_z;
#endif

          // ∇C∙∇D(mn|kl) followed by ∇Cx∇D(mn|kl) X, Y, and Z
          // (kl|mn)
	  ERIBuffCD[      KLMN] =  dAdotdB;
          ERIBuffCD[  NB4+KLMN] =  dAcrossdB_x;
          ERIBuffCD[2*NB4+KLMN] =  dAcrossdB_y;
          ERIBuffCD[3*NB4+KLMN] =  dAcrossdB_z;
#if 0
          // (kl|nm)
	  ERIBuffCD[      KLNM] =  dAdotdB;
          ERIBuffCD[  NB4+KLNM] = -dAcrossdB_x;
          ERIBuffCD[2*NB4+KLNM] = -dAcrossdB_y;
          ERIBuffCD[3*NB4+KLNM] = -dAcrossdB_z;
          // (lk|mn)
	  ERIBuffCD[      LKMN] =  dAdotdB;
          ERIBuffCD[  NB4+LKMN] =  dAcrossdB_x;
          ERIBuffCD[2*NB4+LKMN] =  dAcrossdB_y;
          ERIBuffCD[3*NB4+LKMN] =  dAcrossdB_z;
          // (lk|nm)
	  ERIBuffCD[      LKNM] =  dAdotdB;
          ERIBuffCD[  NB4+LKNM] = -dAcrossdB_x;
          ERIBuffCD[2*NB4+LKNM] = -dAcrossdB_y;
          ERIBuffCD[3*NB4+LKNM] = -dAcrossdB_z;
#endif
	} // AB integral preparation loop


        shls[0] = int(s3);
        shls[1] = int(s4);
        shls[2] = int(s1);
        shls[3] = int(s2);

        //if(cint2e_sph(buff, shls, atm, nAtoms, bas, nShell, env, nullptr)!=0)
        if(cint2e_ipvip1_sph(buff, shls, atm, nAtoms, bas, nShells, env, nullptr)!=0)
        //if(cint2e_ip1ip2_sph(buff, shls, atm, nAtoms, bas, nShells, env, nullptr)!=0)
        for(n =   maxShellSize, mnkl = 0ul; n <   maxShellSize + n2; ++n) 
        for(m = 0                         ; m <                  n1; ++m) 
        for(l = 3*maxShellSize            ; l < 3*maxShellSize + n4; ++l)
        for(k = 2*maxShellSize            ; k < 2*maxShellSize + n3; ++k, ++mnkl) {
 
          auto MNKL = m + n*NB + k*NB2 + l*NB3;
          //auto MNLK = m + n*NB + l*NB2 + k*NB3;
          //auto NMKL = n + m*NB + k*NB2 + l*NB3;
          //auto NMLK = n + m*NB + l*NB2 + k*NB3;
          auto KLMN = k + l*NB + m*NB2 + n*NB3;
          //auto KLNM = k + l*NB + n*NB2 + m*NB3;
          //auto LKMN = l + k*NB + m*NB2 + n*NB3;
          //auto LKNM = l + k*NB + n*NB2 + m*NB3;

          /* Dirac-Coulomb */
          // ∇C∙∇D(mn|kl)
          auto dCdotdD = buff[AxBx*nQuad+mnkl] + buff[AyBy*nQuad+mnkl] + buff[AzBz*nQuad+mnkl];
          // ∇Ax∇B(mn|kl)
          auto dCcrossdD_x =  buff[AyBz*nQuad+mnkl] - buff[AzBy*nQuad+mnkl];
          auto dCcrossdD_y = -buff[AxBz*nQuad+mnkl] + buff[AzBx*nQuad+mnkl];
          auto dCcrossdD_z =  buff[AxBy*nQuad+mnkl] - buff[AyBx*nQuad+mnkl];


          // ∇A∙∇B(mn|kl) followed by ∇Ax∇B(mn|kl) X, Y, and Z
         // (kl|mn)
	  ERIBuffAB[      KLMN] =  dCdotdD;
          ERIBuffAB[  NB4+KLMN] =  dCcrossdD_x;
          ERIBuffAB[2*NB4+KLMN] =  dCcrossdD_y;
          ERIBuffAB[3*NB4+KLMN] =  dCcrossdD_z;
#if 0
          // (kl|nm)
	  ERIBuffAB[      KLNM] =  dCdotdD;
          ERIBuffAB[  NB4+KLNM] =  dCcrossdD_x;
          ERIBuffAB[2*NB4+KLNM] =  dCcrossdD_y;
          ERIBuffAB[3*NB4+KLNM] =  dCcrossdD_z;
          // (lk|mn)
	  ERIBuffAB[      LKMN] =  dCdotdD;
          ERIBuffAB[  NB4+LKMN] = -dCcrossdD_x;
          ERIBuffAB[2*NB4+LKMN] = -dCcrossdD_y;
          ERIBuffAB[3*NB4+LKMN] = -dCcrossdD_z;
          // (lk|nm)
	  ERIBuffAB[      LKNM] =  dCdotdD;
          ERIBuffAB[  NB4+LKNM] = -dCcrossdD_x;
          ERIBuffAB[2*NB4+LKNM] = -dCcrossdD_y;
          ERIBuffAB[3*NB4+LKNM] = -dCcrossdD_z;
#endif
          // ∇C∙∇D(mn|kl) followed by ∇Cx∇D(mn|kl) X, Y, and Z
          // (mn|kl)
	  ERIBuffCD[      MNKL] =  dCdotdD;
          ERIBuffCD[  NB4+MNKL] =  dCcrossdD_x;
          ERIBuffCD[2*NB4+MNKL] =  dCcrossdD_y;
          ERIBuffCD[3*NB4+MNKL] =  dCcrossdD_z;
#if 0
          // (mn|lk)
	  ERIBuffCD[      MNLK] =  dCdotdD;
          ERIBuffCD[  NB4+MNLK] = -dCcrossdD_x;
          ERIBuffCD[2*NB4+MNLK] = -dCcrossdD_y;
          ERIBuffCD[3*NB4+MNLK] = -dCcrossdD_z;
          // (nm|kl)
	  ERIBuffCD[      NMKL] =  dCdotdD;
          ERIBuffCD[  NB4+NMKL] =  dCcrossdD_x;
          ERIBuffCD[2*NB4+NMKL] =  dCcrossdD_y;
          ERIBuffCD[3*NB4+NMKL] =  dCcrossdD_z;
          // (nm|lk)
	  ERIBuffCD[      NMLK] =  dCdotdD;
          ERIBuffCD[  NB4+NMLK] = -dCcrossdD_x;
          ERIBuffCD[2*NB4+NMLK] = -dCcrossdD_y;
          ERIBuffCD[3*NB4+NMLK] = -dCcrossdD_z;
#endif
	} // CD integral preparation loop


#if 0
        std::cout << "ERI00-03: ∇A∙∇B(ab|cd)  ∇Ax∇B(ab|cd)-X  ∇Ax∇B(ab|cd)-Y  ∇Ax∇B(ab|cd)-Z" << std::endl;
        for(m = 0ul,            bf1 = bf1_s; m <                  n1; ++m, bf1++) 
        for(n =   maxShellSize, bf2 = bf2_s; n <   maxShellSize + n2; ++n, bf2++) 
        for(k = 2*maxShellSize, bf3 = bf3_s; k < 2*maxShellSize + n3; ++k, bf3++) 
        for(l = 3*maxShellSize, bf4 = bf4_s; l < 3*maxShellSize + n4; ++l, bf4++) {

          auto MNKL = m + n*NB + k*NB2 + l*NB3;
          auto KLMN = k + l*NB + m*NB2 + n*NB3;
	  auto DotPrd = MNKL;
	  auto CrossX = MNKL+NB4;
	  auto CrossY = MNKL+2*NB4;
	  auto CrossZ = MNKL+3*NB4;

          std::cout << "(" << m << "," << n << "|" << k << "," << l << ")  ";
          std::cout << ERIBuffAB[DotPrd];
          std::cout << "   ";
          std::cout << ERIBuffAB[CrossX];
          std::cout << "   ";
          std::cout << ERIBuffAB[CrossY];
          std::cout << "   ";
          std::cout << ERIBuffAB[CrossZ] << std::endl;

        };

        std::cout << "ERI00-03: ∇C∙∇D(ab|cd)  ∇Cx∇D(ab|cd)-X  ∇Cx∇D(ab|cd)-Y  ∇Cx∇D(ab|cd)-Z" << std::endl;
        for(m = 0ul,            bf1 = bf1_s; m <                  n1; ++m, bf1++) 
        for(n =   maxShellSize, bf2 = bf2_s; n <   maxShellSize + n2; ++n, bf2++) 
        for(k = 2*maxShellSize, bf3 = bf3_s; k < 2*maxShellSize + n3; ++k, bf3++) 
        for(l = 3*maxShellSize, bf4 = bf4_s; l < 3*maxShellSize + n4; ++l, bf4++) {

          auto MNKL = m + n*NB + k*NB2 + l*NB3;
          auto KLMN = k + l*NB + m*NB2 + n*NB3;
	  auto DotPrd = MNKL;
	  auto CrossX = MNKL+NB4;
	  auto CrossY = MNKL+2*NB4;
	  auto CrossZ = MNKL+3*NB4;

          std::cout << "(" << m << "," << n << "|" << k << "," << l << ")  ";
          std::cout << ERIBuffCD[DotPrd];
          std::cout << "   ";
          std::cout << ERIBuffCD[CrossX];
          std::cout << "   ";
          std::cout << ERIBuffCD[CrossY];
          std::cout << "   ";
          std::cout << ERIBuffCD[CrossZ] << std::endl;

        };

#endif


        for(m = 0ul,            bf1 = bf1_s; m <                  n1; ++m, bf1++) 
        for(n =   maxShellSize, bf2 = bf2_s; n <   maxShellSize + n2; ++n, bf2++) 
        for(k = 2*maxShellSize, bf3 = bf3_s; k < 2*maxShellSize + n3; ++k, bf3++) 
        for(l = 3*maxShellSize, bf4 = bf4_s; l < 3*maxShellSize + n4; ++l, bf4++) {


          auto MNKL = m + n*NB + k*NB2 + l*NB3;
          auto KLMN = k + l*NB + m*NB2 + n*NB3;
	  auto DotPrd = MNKL;
	  auto CrossX = MNKL+NB4;
	  auto CrossY = MNKL+2*NB4;
	  auto CrossZ = MNKL+3*NB4;



          /***********************************/
	  /* Dirac-Coulomb (LL|LL)           */
          /***********************************/
          auto AXMSTemp = AX_loc[LLMS];
          auto XMSTemp  = matList[SSMS].X;
          auto XMXTemp  = matList[SSMX].X;
          auto XMYTemp  = matList[SSMY].X;
          auto XMZTemp  = matList[SSMZ].X;

	  /* Hermitian Density Matrices */
	  auto ScaleF = 1.0;
	  if(bf3_s!=bf4_s) ScaleF = 2.0;

          //MNKL
          AXMSTemp[bf1 + bf2*nBasis] += ERIBuffCD[DotPrd]*XMSTemp[bf4 + bf3*nBasis].real()*ScaleF;
          AXMSTemp[bf1 + bf2*nBasis] -= ERIBuffCD[CrossZ]*XMZTemp[bf4 + bf3*nBasis].imag()*ScaleF;
#ifndef _KRAMERS_SYM_
          AXMSTemp[bf1 + bf2*nBasis] -= ERIBuffCD[CrossX]*XMXTemp[bf4 + bf3*nBasis].imag()*ScaleF;
          AXMSTemp[bf1 + bf2*nBasis] -= ERIBuffCD[CrossY]*XMYTemp[bf4 + bf3*nBasis].imag()*ScaleF;
#endif

	  if(bf1_s!=bf2_s) AXMSTemp[bf2 + bf1*nBasis]  = AXMSTemp[bf1 + bf2*nBasis];

          if(bf1_s!=bf3_s or bf2_s!=bf4_s){
 
            DotPrd = KLMN;
            CrossX = KLMN+NB4;
	    CrossY = KLMN+2*NB4;
	    CrossZ = KLMN+3*NB4;

	    ScaleF = 1.0;
	    if(bf1_s!=bf2_s) ScaleF = 2.0;

            //KLMN
            AXMSTemp[bf3 + bf4*nBasis] += ERIBuffCD[DotPrd]*XMSTemp[bf2 + bf1*nBasis].real()*ScaleF;
            AXMSTemp[bf3 + bf4*nBasis] -= ERIBuffCD[CrossZ]*XMZTemp[bf2 + bf1*nBasis].imag()*ScaleF;
#ifndef _KRAMERS_SYM_
            AXMSTemp[bf3 + bf4*nBasis] -= ERIBuffCD[CrossX]*XMXTemp[bf2 + bf1*nBasis].imag()*ScaleF;
            AXMSTemp[bf3 + bf4*nBasis] -= ERIBuffCD[CrossY]*XMYTemp[bf2 + bf1*nBasis].imag()*ScaleF;
#endif

	    if(bf3_s!=bf4_s) AXMSTemp[bf4 + bf3*nBasis] = AXMSTemp[bf3 + bf4*nBasis];
	  }


#if 0 // Non-Hermitian Density Matrices LLLL

          //MNKL
          AXMSTemp[bf1 + bf2*nBasis] += ERIBuffCD[DotPrd]*XMSTemp[bf4 + bf3*nBasis];
          AXMSTemp[bf1 + bf2*nBasis] += ERIBuffCD[CrossX]*XMXTemp[bf4 + bf3*nBasis]*iscale;
          AXMSTemp[bf1 + bf2*nBasis] += ERIBuffCD[CrossY]*XMYTemp[bf4 + bf3*nBasis]*iscale;
          AXMSTemp[bf1 + bf2*nBasis] += ERIBuffCD[CrossZ]*XMZTemp[bf4 + bf3*nBasis]*iscale;

	  //NMKL 
	  if(bf1_s!=bf2_s){
            AXMSTemp[bf2 + bf1*nBasis] += ERIBuffCD[DotPrd]*XMSTemp[bf4 + bf3*nBasis];
            AXMSTemp[bf2 + bf1*nBasis] += ERIBuffCD[CrossX]*XMXTemp[bf4 + bf3*nBasis]*iscale;
            AXMSTemp[bf2 + bf1*nBasis] += ERIBuffCD[CrossY]*XMYTemp[bf4 + bf3*nBasis]*iscale;
            AXMSTemp[bf2 + bf1*nBasis] += ERIBuffCD[CrossZ]*XMZTemp[bf4 + bf3*nBasis]*iscale;
	  }

	  //MNLK 
	  if(bf3_s!=bf4_s){
            AXMSTemp[bf1 + bf2*nBasis] += ERIBuffCD[DotPrd]*XMSTemp[bf3 + bf4*nBasis];
            AXMSTemp[bf1 + bf2*nBasis] -= ERIBuffCD[CrossX]*XMXTemp[bf3 + bf4*nBasis]*iscale;
            AXMSTemp[bf1 + bf2*nBasis] -= ERIBuffCD[CrossY]*XMYTemp[bf3 + bf4*nBasis]*iscale;
            AXMSTemp[bf1 + bf2*nBasis] -= ERIBuffCD[CrossZ]*XMZTemp[bf3 + bf4*nBasis]*iscale;
	  }

	  //NMLK 
	  if(bf1_s!=bf2_s and bf3_s!=bf4_s){
            AXMSTemp[bf2 + bf1*nBasis] += ERIBuffCD[DotPrd]*XMSTemp[bf3 + bf4*nBasis];
            AXMSTemp[bf2 + bf1*nBasis] -= ERIBuffCD[CrossX]*XMXTemp[bf3 + bf4*nBasis]*iscale;
            AXMSTemp[bf2 + bf1*nBasis] -= ERIBuffCD[CrossY]*XMYTemp[bf3 + bf4*nBasis]*iscale;
            AXMSTemp[bf2 + bf1*nBasis] -= ERIBuffCD[CrossZ]*XMZTemp[bf3 + bf4*nBasis]*iscale;
	  }

          if(bf1_s!=bf3_s or bf2_s!=bf4_s){
 
            DotPrd = KLMN;
            CrossX = KLMN+NB4;
	    CrossY = KLMN+2*NB4;
	    CrossZ = KLMN+3*NB4;

            //KLMN
            AXMSTemp[bf3 + bf4*nBasis] += ERIBuffCD[DotPrd]*XMSTemp[bf2 + bf1*nBasis];
            AXMSTemp[bf3 + bf4*nBasis] += ERIBuffCD[CrossX]*XMXTemp[bf2 + bf1*nBasis]*iscale;
            AXMSTemp[bf3 + bf4*nBasis] += ERIBuffCD[CrossY]*XMYTemp[bf2 + bf1*nBasis]*iscale;
            AXMSTemp[bf3 + bf4*nBasis] += ERIBuffCD[CrossZ]*XMZTemp[bf2 + bf1*nBasis]*iscale;

	    //LKMN needed when D^{SS} is non-Hermitian
	    if(bf3_s!=bf4_s){
              AXMSTemp[bf4 + bf3*nBasis] += ERIBuffCD[DotPrd]*XMSTemp[bf2 + bf1*nBasis];
              AXMSTemp[bf4 + bf3*nBasis] += ERIBuffCD[CrossX]*XMXTemp[bf2 + bf1*nBasis]*iscale;
              AXMSTemp[bf4 + bf3*nBasis] += ERIBuffCD[CrossY]*XMYTemp[bf2 + bf1*nBasis]*iscale;
              AXMSTemp[bf4 + bf3*nBasis] += ERIBuffCD[CrossZ]*XMZTemp[bf2 + bf1*nBasis]*iscale;
	    }

	    //KLNM
	    if(bf1_s!=bf2_s){
              AXMSTemp[bf3 + bf4*nBasis] += ERIBuffCD[DotPrd]*XMSTemp[bf1 + bf2*nBasis];
              AXMSTemp[bf3 + bf4*nBasis] -= ERIBuffCD[CrossX]*XMXTemp[bf1 + bf2*nBasis]*iscale;
              AXMSTemp[bf3 + bf4*nBasis] -= ERIBuffCD[CrossY]*XMYTemp[bf1 + bf2*nBasis]*iscale;
              AXMSTemp[bf3 + bf4*nBasis] -= ERIBuffCD[CrossZ]*XMZTemp[bf1 + bf2*nBasis]*iscale;
	    }

	    //LKNM needed when D^{SS} is non-Hermitian
	    if(bf1_s!=bf2_s and bf3_s!=bf4_s){
              AXMSTemp[bf4 + bf3*nBasis] += ERIBuffCD[DotPrd]*XMSTemp[bf1 + bf2*nBasis];
              AXMSTemp[bf4 + bf3*nBasis] -= ERIBuffCD[CrossX]*XMXTemp[bf1 + bf2*nBasis]*iscale;
              AXMSTemp[bf4 + bf3*nBasis] -= ERIBuffCD[CrossY]*XMYTemp[bf1 + bf2*nBasis]*iscale;
              AXMSTemp[bf4 + bf3*nBasis] -= ERIBuffCD[CrossZ]*XMZTemp[bf1 + bf2*nBasis]*iscale;
	    }

	  }

#endif // Non-Hermitian Density Matrices LLLL




          /***********************************/
	  /* Dirac-Coulomb (SS|SS)           */
          /***********************************/

	  DotPrd = MNKL;
	  CrossX = MNKL+NB4;
	  CrossY = MNKL+2*NB4;
	  CrossZ = MNKL+3*NB4;

          AXMSTemp  = AX_loc[SSMS];
          auto AXMXTemp  = AX_loc[SSMX];
          auto AXMYTemp  = AX_loc[SSMY];
          auto AXMZTemp  = AX_loc[SSMZ];
          XMSTemp = matList[LLMS].X;

	  /* Hermitian Density Matrices */
	  ScaleF = 1.0;
	  dcomplex iScaleF = dcomplex(0.0, 1.0);

	  if(bf3_s!=bf4_s) {
	    ScaleF = 2.0;
	    iScaleF = dcomplex(0.0, 2.0);
	  }

	  //MNKL
          AXMSTemp[bf1 + bf2*nBasis] += ERIBuffAB[DotPrd]*XMSTemp[bf4 + bf3*nBasis].real()*ScaleF;
          AXMZTemp[bf1 + bf2*nBasis] += ERIBuffAB[CrossZ]*XMSTemp[bf4 + bf3*nBasis].real()*iScaleF;
          AXMXTemp[bf1 + bf2*nBasis] += ERIBuffAB[CrossX]*XMSTemp[bf4 + bf3*nBasis].real()*iScaleF;
          AXMYTemp[bf1 + bf2*nBasis] += ERIBuffAB[CrossY]*XMSTemp[bf4 + bf3*nBasis].real()*iScaleF;

	  if(bf1_s!=bf2_s) {
	    AXMSTemp[bf2 + bf1*nBasis]  =  AXMSTemp[bf1 + bf2*nBasis];
	    AXMXTemp[bf2 + bf1*nBasis]  = -AXMXTemp[bf1 + bf2*nBasis];
	    AXMYTemp[bf2 + bf1*nBasis]  = -AXMYTemp[bf1 + bf2*nBasis];
	    AXMZTemp[bf2 + bf1*nBasis]  = -AXMZTemp[bf1 + bf2*nBasis];
	  }

          if(bf1_s!=bf3_s or bf2_s!=bf4_s){
 
            DotPrd = KLMN;
            CrossX = KLMN+NB4;
	    CrossY = KLMN+2*NB4;
	    CrossZ = KLMN+3*NB4;

	    ScaleF = 1.0;
	    dcomplex iScaleF = dcomplex(0.0, 1.0);

	    if(bf1_s!=bf2_s) {
	      ScaleF = 2.0;
	      iScaleF = dcomplex(0.0, 2.0);
	    }

            //KLMN
            AXMSTemp[bf3 + bf4*nBasis] += ERIBuffAB[DotPrd]*XMSTemp[bf2 + bf1*nBasis].real()*ScaleF;
            AXMZTemp[bf3 + bf4*nBasis] += ERIBuffAB[CrossZ]*XMSTemp[bf2 + bf1*nBasis].real()*iScaleF;
            AXMXTemp[bf3 + bf4*nBasis] += ERIBuffAB[CrossX]*XMSTemp[bf2 + bf1*nBasis].real()*iScaleF;
            AXMYTemp[bf3 + bf4*nBasis] += ERIBuffAB[CrossY]*XMSTemp[bf2 + bf1*nBasis].real()*iScaleF;

	    if(bf3_s!=bf4_s) {
	      AXMSTemp[bf4 + bf3*nBasis] =  AXMSTemp[bf3 + bf4*nBasis];
	      AXMXTemp[bf4 + bf3*nBasis] = -AXMXTemp[bf3 + bf4*nBasis];
	      AXMYTemp[bf4 + bf3*nBasis] = -AXMYTemp[bf3 + bf4*nBasis];
	      AXMZTemp[bf4 + bf3*nBasis] = -AXMZTemp[bf3 + bf4*nBasis];
	    }

	  }


#if 0 // Non-Hermitian Density Matrices	SSSS

	  //MNKL
          AXMSTemp[bf1 + bf2*nBasis] += ERIBuffAB[DotPrd]*XMSTemp[bf4 + bf3*nBasis];
          AXMXTemp[bf1 + bf2*nBasis] += ERIBuffAB[CrossX]*XMSTemp[bf4 + bf3*nBasis]*iscale;
          AXMYTemp[bf1 + bf2*nBasis] += ERIBuffAB[CrossY]*XMSTemp[bf4 + bf3*nBasis]*iscale;
          AXMZTemp[bf1 + bf2*nBasis] += ERIBuffAB[CrossZ]*XMSTemp[bf4 + bf3*nBasis]*iscale;

	  //NMKL needed when D^{LL} is non-Hermitian
	  if(bf1_s!=bf2_s){
            AXMSTemp[bf2 + bf1*nBasis] += ERIBuffAB[DotPrd]*XMSTemp[bf4 + bf3*nBasis];
            AXMXTemp[bf2 + bf1*nBasis] -= ERIBuffAB[CrossX]*XMSTemp[bf4 + bf3*nBasis]*iscale;
            AXMYTemp[bf2 + bf1*nBasis] -= ERIBuffAB[CrossY]*XMSTemp[bf4 + bf3*nBasis]*iscale;
            AXMZTemp[bf2 + bf1*nBasis] -= ERIBuffAB[CrossZ]*XMSTemp[bf4 + bf3*nBasis]*iscale;
	  }

	  //MNLK
	  if(bf3_s!=bf4_s){
            AXMSTemp[bf1 + bf2*nBasis] += ERIBuffAB[DotPrd]*XMSTemp[bf3 + bf4*nBasis];
            AXMXTemp[bf1 + bf2*nBasis] += ERIBuffAB[CrossX]*XMSTemp[bf3 + bf4*nBasis]*iscale;
            AXMYTemp[bf1 + bf2*nBasis] += ERIBuffAB[CrossY]*XMSTemp[bf3 + bf4*nBasis]*iscale;
            AXMZTemp[bf1 + bf2*nBasis] += ERIBuffAB[CrossZ]*XMSTemp[bf3 + bf4*nBasis]*iscale;
	  }

	  //NMLK needed when D^{LL} is non-Hermitian
	  if(bf1_s!=bf2_s and bf3_s!=bf4_s){
            AXMSTemp[bf2 + bf1*nBasis] += ERIBuffAB[DotPrd]*XMSTemp[bf3 + bf4*nBasis];
            AXMXTemp[bf2 + bf1*nBasis] -= ERIBuffAB[CrossX]*XMSTemp[bf3 + bf4*nBasis]*iscale;
            AXMYTemp[bf2 + bf1*nBasis] -= ERIBuffAB[CrossY]*XMSTemp[bf3 + bf4*nBasis]*iscale;
            AXMZTemp[bf2 + bf1*nBasis] -= ERIBuffAB[CrossZ]*XMSTemp[bf3 + bf4*nBasis]*iscale;
	  }
	  
          // This assumes that D^{LL} is non-Hermitian
	  if(bf1_s!=bf2_s) {
            AXMSTemp[bf2 + bf1*nBasis] =  AXMSTemp[bf1 + bf2*nBasis];
	    AXMXTemp[bf2 + bf1*nBasis] = -AXMXTemp[bf1 + bf2*nBasis];
            AXMYTemp[bf2 + bf1*nBasis] = -AXMYTemp[bf1 + bf2*nBasis];
            AXMZTemp[bf2 + bf1*nBasis] = -AXMZTemp[bf1 + bf2*nBasis];
	  }

          if(bf1_s!=bf3_s or bf2_s!=bf4_s){
 
            DotPrd = KLMN;
            CrossX = KLMN+NB4;
	    CrossY = KLMN+2*NB4;
	    CrossZ = KLMN+3*NB4;

            //KLMN
            AXMSTemp[bf3 + bf4*nBasis] += ERIBuffAB[DotPrd]*XMSTemp[bf2 + bf1*nBasis];
            AXMXTemp[bf3 + bf4*nBasis] += ERIBuffAB[CrossX]*XMSTemp[bf2 + bf1*nBasis]*iscale;
            AXMYTemp[bf3 + bf4*nBasis] += ERIBuffAB[CrossY]*XMSTemp[bf2 + bf1*nBasis]*iscale;
            AXMZTemp[bf3 + bf4*nBasis] += ERIBuffAB[CrossZ]*XMSTemp[bf2 + bf1*nBasis]*iscale;

	    //LKMN needed when D^{LL} is non-Hermitian
	    if(bf3_s!=bf4_s){
              AXMSTemp[bf4 + bf3*nBasis] += ERIBuffAB[DotPrd]*XMSTemp[bf2 + bf1*nBasis];
              AXMXTemp[bf4 + bf3*nBasis] -= ERIBuffAB[CrossX]*XMSTemp[bf2 + bf1*nBasis]*iscale;
              AXMYTemp[bf4 + bf3*nBasis] -= ERIBuffAB[CrossY]*XMSTemp[bf2 + bf1*nBasis]*iscale;
              AXMZTemp[bf4 + bf3*nBasis] -= ERIBuffAB[CrossZ]*XMSTemp[bf2 + bf1*nBasis]*iscale;
	    }

	    //KLNM
	    if(bf1_s!=bf2_s){
              AXMSTemp[bf3 + bf4*nBasis] += ERIBuffAB[DotPrd]*XMSTemp[bf1 + bf2*nBasis];
              AXMXTemp[bf3 + bf4*nBasis] += ERIBuffAB[CrossX]*XMSTemp[bf1 + bf2*nBasis]*iscale;
              AXMYTemp[bf3 + bf4*nBasis] += ERIBuffAB[CrossY]*XMSTemp[bf1 + bf2*nBasis]*iscale;
              AXMZTemp[bf3 + bf4*nBasis] += ERIBuffAB[CrossZ]*XMSTemp[bf1 + bf2*nBasis]*iscale;
	    }

	    //LKNM needed when D^{LL} is non-Hermitian 
	    if(bf1_s!=bf2_s and bf3_s!=bf4_s){
              AXMSTemp[bf4 + bf3*nBasis] += ERIBuffAB[DotPrd]*XMSTemp[bf1 + bf2*nBasis];
              AXMXTemp[bf4 + bf3*nBasis] -= ERIBuffAB[CrossX]*XMSTemp[bf1 + bf2*nBasis]*iscale;
              AXMYTemp[bf4 + bf3*nBasis] -= ERIBuffAB[CrossY]*XMSTemp[bf1 + bf2*nBasis]*iscale;
              AXMZTemp[bf4 + bf3*nBasis] -= ERIBuffAB[CrossZ]*XMSTemp[bf1 + bf2*nBasis]*iscale;
	    }

            // This assumes that D^{SS} is non-Hermitian
	    if(bf3_s!=bf4_s) {
              AXMSTemp[bf4 + bf3*nBasis] =  AXMSTemp[bf3 + bf4*nBasis];
	      AXMXTemp[bf4 + bf3*nBasis] = -AXMXTemp[bf3 + bf4*nBasis];
              AXMYTemp[bf4 + bf3*nBasis] = -AXMYTemp[bf3 + bf4*nBasis];
              AXMZTemp[bf4 + bf3*nBasis] = -AXMZTemp[bf3 + bf4*nBasis];
	    }
	  }

#endif // Non-Hermitian Density Matrices SSSS

          /***********************************/
	  /* Dirac-Coulomb (LL|SS) / (SS|LL) */
          /***********************************/
	  DotPrd = MNKL;
	  CrossX = MNKL+NB4;
	  CrossY = MNKL+2*NB4;
	  CrossZ = MNKL+3*NB4;

          AXMSTemp  = AX_loc[LSMS];
          AXMXTemp  = AX_loc[LSMX];
          AXMYTemp  = AX_loc[LSMY];
          AXMZTemp  = AX_loc[LSMZ];
          XMSTemp = matList[LSMS].X;
          XMXTemp = matList[LSMX].X;
          XMYTemp = matList[LSMY].X;
          XMZTemp = matList[LSMZ].X;

          //MNKL
          AXMSTemp[bf1 + bf4*nBasis] -= ERIBuffCD[DotPrd]*XMSTemp[bf2 + bf3*nBasis];
          AXMSTemp[bf1 + bf4*nBasis] -= ERIBuffCD[CrossZ]*XMZTemp[bf2 + bf3*nBasis]*iscale;
#ifndef _KRAMERS_SYM_
          AXMSTemp[bf1 + bf4*nBasis] -= ERIBuffCD[CrossX]*XMXTemp[bf2 + bf3*nBasis]*iscale;
          AXMSTemp[bf1 + bf4*nBasis] -= ERIBuffCD[CrossY]*XMYTemp[bf2 + bf3*nBasis]*iscale;
#endif



          AXMXTemp[bf1 + bf4*nBasis] -= ERIBuffCD[CrossX]*XMSTemp[bf2 + bf3*nBasis]*iscale;
          AXMXTemp[bf1 + bf4*nBasis] -= ERIBuffCD[CrossY]*XMZTemp[bf2 + bf3*nBasis];
#ifndef _KRAMERS_SYM_
          AXMXTemp[bf1 + bf4*nBasis] -= ERIBuffCD[DotPrd]*XMXTemp[bf2 + bf3*nBasis];
          AXMXTemp[bf1 + bf4*nBasis] += ERIBuffCD[CrossZ]*XMYTemp[bf2 + bf3*nBasis];
#endif



          AXMYTemp[bf1 + bf4*nBasis] -= ERIBuffCD[CrossY]*XMSTemp[bf2 + bf3*nBasis]*iscale;
          AXMYTemp[bf1 + bf4*nBasis] += ERIBuffCD[CrossX]*XMZTemp[bf2 + bf3*nBasis];
#ifndef _KRAMERS_SYM_
          AXMYTemp[bf1 + bf4*nBasis] -= ERIBuffCD[DotPrd]*XMYTemp[bf2 + bf3*nBasis];
          AXMYTemp[bf1 + bf4*nBasis] -= ERIBuffCD[CrossZ]*XMXTemp[bf2 + bf3*nBasis];
#endif



          AXMZTemp[bf1 + bf4*nBasis] -= ERIBuffCD[CrossZ]*XMSTemp[bf2 + bf3*nBasis]*iscale;
          AXMZTemp[bf1 + bf4*nBasis] -= ERIBuffCD[DotPrd]*XMZTemp[bf2 + bf3*nBasis];
#ifndef _KRAMERS_SYM_
          AXMZTemp[bf1 + bf4*nBasis] += ERIBuffCD[CrossY]*XMXTemp[bf2 + bf3*nBasis];
          AXMZTemp[bf1 + bf4*nBasis] -= ERIBuffCD[CrossX]*XMYTemp[bf2 + bf3*nBasis];
#endif


          //MNLK
	  if(bf3_s!=bf4_s) {

            AXMSTemp[bf1 + bf3*nBasis] -= ERIBuffCD[DotPrd]*XMSTemp[bf2 + bf4*nBasis];
            AXMSTemp[bf1 + bf3*nBasis] += ERIBuffCD[CrossZ]*XMZTemp[bf2 + bf4*nBasis]*iscale;
#ifndef _KRAMERS_SYM_
            AXMSTemp[bf1 + bf3*nBasis] += ERIBuffCD[CrossX]*XMXTemp[bf2 + bf4*nBasis]*iscale;
            AXMSTemp[bf1 + bf3*nBasis] += ERIBuffCD[CrossY]*XMYTemp[bf2 + bf4*nBasis]*iscale;
#endif



            AXMXTemp[bf1 + bf3*nBasis] += ERIBuffCD[CrossX]*XMSTemp[bf2 + bf4*nBasis]*iscale;
            AXMXTemp[bf1 + bf3*nBasis] += ERIBuffCD[CrossY]*XMZTemp[bf2 + bf4*nBasis];
#ifndef _KRAMERS_SYM_
            AXMXTemp[bf1 + bf3*nBasis] -= ERIBuffCD[DotPrd]*XMXTemp[bf2 + bf4*nBasis];
            AXMXTemp[bf1 + bf3*nBasis] -= ERIBuffCD[CrossZ]*XMYTemp[bf2 + bf4*nBasis];
#endif



            AXMYTemp[bf1 + bf3*nBasis] += ERIBuffCD[CrossY]*XMSTemp[bf2 + bf4*nBasis]*iscale;
            AXMYTemp[bf1 + bf3*nBasis] -= ERIBuffCD[CrossX]*XMZTemp[bf2 + bf4*nBasis];
#ifndef _KRAMERS_SYM_
            AXMYTemp[bf1 + bf3*nBasis] += ERIBuffCD[CrossZ]*XMXTemp[bf2 + bf4*nBasis];
            AXMYTemp[bf1 + bf3*nBasis] -= ERIBuffCD[DotPrd]*XMYTemp[bf2 + bf4*nBasis];
#endif



            AXMZTemp[bf1 + bf3*nBasis] += ERIBuffCD[CrossZ]*XMSTemp[bf2 + bf4*nBasis]*iscale;
            AXMZTemp[bf1 + bf3*nBasis] -= ERIBuffCD[DotPrd]*XMZTemp[bf2 + bf4*nBasis];
#ifndef _KRAMERS_SYM_
            AXMZTemp[bf1 + bf3*nBasis] -= ERIBuffCD[CrossY]*XMXTemp[bf2 + bf4*nBasis];
            AXMZTemp[bf1 + bf3*nBasis] += ERIBuffCD[CrossX]*XMYTemp[bf2 + bf4*nBasis];
#endif

	  }

	  //NMKL
	  if(bf1_s!=bf2_s){
            AXMSTemp[bf2 + bf4*nBasis] -= ERIBuffCD[DotPrd]*XMSTemp[bf1 + bf3*nBasis];
            AXMSTemp[bf2 + bf4*nBasis] -= ERIBuffCD[CrossZ]*XMZTemp[bf1 + bf3*nBasis]*iscale;
#ifndef _KRAMERS_SYM_
            AXMSTemp[bf2 + bf4*nBasis] -= ERIBuffCD[CrossX]*XMXTemp[bf1 + bf3*nBasis]*iscale;
            AXMSTemp[bf2 + bf4*nBasis] -= ERIBuffCD[CrossY]*XMYTemp[bf1 + bf3*nBasis]*iscale;
#endif



            AXMXTemp[bf2 + bf4*nBasis] -= ERIBuffCD[CrossX]*XMSTemp[bf1 + bf3*nBasis]*iscale;
            AXMXTemp[bf2 + bf4*nBasis] -= ERIBuffCD[CrossY]*XMZTemp[bf1 + bf3*nBasis];
#ifndef _KRAMERS_SYM_
            AXMXTemp[bf2 + bf4*nBasis] -= ERIBuffCD[DotPrd]*XMXTemp[bf1 + bf3*nBasis];
            AXMXTemp[bf2 + bf4*nBasis] += ERIBuffCD[CrossZ]*XMYTemp[bf1 + bf3*nBasis];
#endif



            AXMYTemp[bf2 + bf4*nBasis] -= ERIBuffCD[CrossY]*XMSTemp[bf1 + bf3*nBasis]*iscale;
            AXMYTemp[bf2 + bf4*nBasis] += ERIBuffCD[CrossX]*XMZTemp[bf1 + bf3*nBasis];
#ifndef _KRAMERS_SYM_
            AXMYTemp[bf2 + bf4*nBasis] -= ERIBuffCD[CrossZ]*XMXTemp[bf1 + bf3*nBasis];
            AXMYTemp[bf2 + bf4*nBasis] -= ERIBuffCD[DotPrd]*XMYTemp[bf1 + bf3*nBasis];
#endif



            AXMZTemp[bf2 + bf4*nBasis] -= ERIBuffCD[CrossZ]*XMSTemp[bf1 + bf3*nBasis]*iscale;
            AXMZTemp[bf2 + bf4*nBasis] -= ERIBuffCD[DotPrd]*XMZTemp[bf1 + bf3*nBasis];
#ifndef _KRAMERS_SYM_
            AXMZTemp[bf2 + bf4*nBasis] += ERIBuffCD[CrossY]*XMXTemp[bf1 + bf3*nBasis];
            AXMZTemp[bf2 + bf4*nBasis] -= ERIBuffCD[CrossX]*XMYTemp[bf1 + bf3*nBasis];
#endif

	    if(bf3_s!=bf4_s) {

              AXMSTemp[bf2 + bf3*nBasis] -= ERIBuffCD[DotPrd]*XMSTemp[bf1 + bf4*nBasis];
              AXMSTemp[bf2 + bf3*nBasis] += ERIBuffCD[CrossZ]*XMZTemp[bf1 + bf4*nBasis]*iscale;
#ifndef _KRAMERS_SYM_
              AXMSTemp[bf2 + bf3*nBasis] += ERIBuffCD[CrossX]*XMXTemp[bf1 + bf4*nBasis]*iscale;
              AXMSTemp[bf2 + bf3*nBasis] += ERIBuffCD[CrossY]*XMYTemp[bf1 + bf4*nBasis]*iscale;
#endif



              AXMXTemp[bf2 + bf3*nBasis] += ERIBuffCD[CrossX]*XMSTemp[bf1 + bf4*nBasis]*iscale;
              AXMXTemp[bf2 + bf3*nBasis] += ERIBuffCD[CrossY]*XMZTemp[bf1 + bf4*nBasis];
#ifndef _KRAMERS_SYM_
              AXMXTemp[bf2 + bf3*nBasis] -= ERIBuffCD[DotPrd]*XMXTemp[bf1 + bf4*nBasis];
              AXMXTemp[bf2 + bf3*nBasis] -= ERIBuffCD[CrossZ]*XMYTemp[bf1 + bf4*nBasis];
#endif



              AXMYTemp[bf2 + bf3*nBasis] += ERIBuffCD[CrossY]*XMSTemp[bf1 + bf4*nBasis]*iscale;
              AXMYTemp[bf2 + bf3*nBasis] -= ERIBuffCD[CrossX]*XMZTemp[bf1 + bf4*nBasis];
#ifndef _KRAMERS_SYM_
              AXMYTemp[bf2 + bf3*nBasis] += ERIBuffCD[CrossZ]*XMXTemp[bf1 + bf4*nBasis];
              AXMYTemp[bf2 + bf3*nBasis] -= ERIBuffCD[DotPrd]*XMYTemp[bf1 + bf4*nBasis];
#endif



              AXMZTemp[bf2 + bf3*nBasis] += ERIBuffCD[CrossZ]*XMSTemp[bf1 + bf4*nBasis]*iscale;
              AXMZTemp[bf2 + bf3*nBasis] -= ERIBuffCD[DotPrd]*XMZTemp[bf1 + bf4*nBasis];
#ifndef _KRAMERS_SYM_
              AXMZTemp[bf2 + bf3*nBasis] -= ERIBuffCD[CrossY]*XMXTemp[bf1 + bf4*nBasis];
              AXMZTemp[bf2 + bf3*nBasis] += ERIBuffCD[CrossX]*XMYTemp[bf1 + bf4*nBasis];
#endif
	    }
	  }

          if(bf1_s!=bf3_s or bf2_s!=bf4_s){
 
            DotPrd = KLMN;
            CrossX = KLMN+NB4;
	    CrossY = KLMN+2*NB4;
	    CrossZ = KLMN+3*NB4;

            //KLMN
            AXMSTemp[bf3 + bf2*nBasis] -= ERIBuffCD[DotPrd]*XMSTemp[bf4 + bf1*nBasis];
            AXMSTemp[bf3 + bf2*nBasis] -= ERIBuffCD[CrossZ]*XMZTemp[bf4 + bf1*nBasis]*iscale;
#ifndef _KRAMERS_SYM_
            AXMSTemp[bf3 + bf2*nBasis] -= ERIBuffCD[CrossX]*XMXTemp[bf4 + bf1*nBasis]*iscale;
            AXMSTemp[bf3 + bf2*nBasis] -= ERIBuffCD[CrossY]*XMYTemp[bf4 + bf1*nBasis]*iscale;
#endif
 


            AXMXTemp[bf3 + bf2*nBasis] -= ERIBuffCD[CrossX]*XMSTemp[bf4 + bf1*nBasis]*iscale;
            AXMXTemp[bf3 + bf2*nBasis] -= ERIBuffCD[CrossY]*XMZTemp[bf4 + bf1*nBasis];
#ifndef _KRAMERS_SYM_
            AXMXTemp[bf3 + bf2*nBasis] -= ERIBuffCD[DotPrd]*XMXTemp[bf4 + bf1*nBasis];
            AXMXTemp[bf3 + bf2*nBasis] += ERIBuffCD[CrossZ]*XMYTemp[bf4 + bf1*nBasis];
#endif



  
            AXMYTemp[bf3 + bf2*nBasis] -= ERIBuffCD[CrossY]*XMSTemp[bf4 + bf1*nBasis]*iscale;
            AXMYTemp[bf3 + bf2*nBasis] += ERIBuffCD[CrossX]*XMZTemp[bf4 + bf1*nBasis];
#ifndef _KRAMERS_SYM_
            AXMYTemp[bf3 + bf2*nBasis] -= ERIBuffCD[CrossZ]*XMXTemp[bf4 + bf1*nBasis];
            AXMYTemp[bf3 + bf2*nBasis] -= ERIBuffCD[DotPrd]*XMYTemp[bf4 + bf1*nBasis];
#endif
 



            AXMZTemp[bf3 + bf2*nBasis] -= ERIBuffCD[CrossZ]*XMSTemp[bf4 + bf1*nBasis]*iscale;
            AXMZTemp[bf3 + bf2*nBasis] -= ERIBuffCD[DotPrd]*XMZTemp[bf4 + bf1*nBasis];
#ifndef _KRAMERS_SYM_
            AXMZTemp[bf3 + bf2*nBasis] += ERIBuffCD[CrossY]*XMXTemp[bf4 + bf1*nBasis];
            AXMZTemp[bf3 + bf2*nBasis] -= ERIBuffCD[CrossX]*XMYTemp[bf4 + bf1*nBasis];
#endif
  
  	    //KLNM
  	    if(bf3_s!=bf4_s){

              AXMSTemp[bf4 + bf2*nBasis] -= ERIBuffCD[DotPrd]*XMSTemp[bf3 + bf1*nBasis];
              AXMSTemp[bf4 + bf2*nBasis] -= ERIBuffCD[CrossZ]*XMZTemp[bf3 + bf1*nBasis]*iscale;
#ifndef _KRAMERS_SYM_
              AXMSTemp[bf4 + bf2*nBasis] -= ERIBuffCD[CrossX]*XMXTemp[bf3 + bf1*nBasis]*iscale;
              AXMSTemp[bf4 + bf2*nBasis] -= ERIBuffCD[CrossY]*XMYTemp[bf3 + bf1*nBasis]*iscale;
#endif
 



              AXMXTemp[bf4 + bf2*nBasis] -= ERIBuffCD[CrossX]*XMSTemp[bf3 + bf1*nBasis]*iscale;
              AXMXTemp[bf4 + bf2*nBasis] -= ERIBuffCD[CrossY]*XMZTemp[bf3 + bf1*nBasis];
#ifndef _KRAMERS_SYM_
              AXMXTemp[bf4 + bf2*nBasis] -= ERIBuffCD[DotPrd]*XMXTemp[bf3 + bf1*nBasis];
              AXMXTemp[bf4 + bf2*nBasis] += ERIBuffCD[CrossZ]*XMYTemp[bf3 + bf1*nBasis];
#endif
  



              AXMYTemp[bf4 + bf2*nBasis] -= ERIBuffCD[CrossY]*XMSTemp[bf3 + bf1*nBasis]*iscale;
              AXMYTemp[bf4 + bf2*nBasis] += ERIBuffCD[CrossX]*XMZTemp[bf3 + bf1*nBasis];
#ifndef _KRAMERS_SYM_
              AXMYTemp[bf4 + bf2*nBasis] -= ERIBuffCD[CrossZ]*XMXTemp[bf3 + bf1*nBasis];
              AXMYTemp[bf4 + bf2*nBasis] -= ERIBuffCD[DotPrd]*XMYTemp[bf3 + bf1*nBasis];
#endif
 



              AXMZTemp[bf4 + bf2*nBasis] -= ERIBuffCD[CrossZ]*XMSTemp[bf3 + bf1*nBasis]*iscale;
              AXMZTemp[bf4 + bf2*nBasis] -= ERIBuffCD[DotPrd]*XMZTemp[bf3 + bf1*nBasis];
#ifndef _KRAMERS_SYM_
              AXMZTemp[bf4 + bf2*nBasis] += ERIBuffCD[CrossY]*XMXTemp[bf3 + bf1*nBasis];
              AXMZTemp[bf4 + bf2*nBasis] -= ERIBuffCD[CrossX]*XMYTemp[bf3 + bf1*nBasis];
#endif

  	    }
  
  	    //LKMN
  	    if(bf1_s!=bf2_s) {

              AXMSTemp[bf3 + bf1*nBasis] -= ERIBuffCD[DotPrd]*XMSTemp[bf4 + bf2*nBasis];
              AXMSTemp[bf3 + bf1*nBasis] += ERIBuffCD[CrossZ]*XMZTemp[bf4 + bf2*nBasis]*iscale;
#ifndef _KRAMERS_SYM_
              AXMSTemp[bf3 + bf1*nBasis] += ERIBuffCD[CrossX]*XMXTemp[bf4 + bf2*nBasis]*iscale;
              AXMSTemp[bf3 + bf1*nBasis] += ERIBuffCD[CrossY]*XMYTemp[bf4 + bf2*nBasis]*iscale;
#endif
 



              AXMXTemp[bf3 + bf1*nBasis] += ERIBuffCD[CrossX]*XMSTemp[bf4 + bf2*nBasis]*iscale;
              AXMXTemp[bf3 + bf1*nBasis] += ERIBuffCD[CrossY]*XMZTemp[bf4 + bf2*nBasis];
#ifndef _KRAMERS_SYM_
              AXMXTemp[bf3 + bf1*nBasis] -= ERIBuffCD[DotPrd]*XMXTemp[bf4 + bf2*nBasis];
              AXMXTemp[bf3 + bf1*nBasis] -= ERIBuffCD[CrossZ]*XMYTemp[bf4 + bf2*nBasis];
#endif
  



              AXMYTemp[bf3 + bf1*nBasis] += ERIBuffCD[CrossY]*XMSTemp[bf4 + bf2*nBasis]*iscale;
              AXMYTemp[bf3 + bf1*nBasis] -= ERIBuffCD[CrossX]*XMZTemp[bf4 + bf2*nBasis];
#ifndef _KRAMERS_SYM_
              AXMYTemp[bf3 + bf1*nBasis] += ERIBuffCD[CrossZ]*XMXTemp[bf4 + bf2*nBasis];
              AXMYTemp[bf3 + bf1*nBasis] -= ERIBuffCD[DotPrd]*XMYTemp[bf4 + bf2*nBasis];
#endif
  



              AXMZTemp[bf3 + bf1*nBasis] += ERIBuffCD[CrossZ]*XMSTemp[bf4 + bf2*nBasis]*iscale;
              AXMZTemp[bf3 + bf1*nBasis] -= ERIBuffCD[DotPrd]*XMZTemp[bf4 + bf2*nBasis];
#ifndef _KRAMERS_SYM_
              AXMZTemp[bf3 + bf1*nBasis] -= ERIBuffCD[CrossY]*XMXTemp[bf4 + bf2*nBasis];
              AXMZTemp[bf3 + bf1*nBasis] += ERIBuffCD[CrossX]*XMYTemp[bf4 + bf2*nBasis];
#endif
  
  	      //LKNM
  	      if(bf4_s!=bf3_s) {

                AXMSTemp[bf4 + bf1*nBasis] -= ERIBuffCD[DotPrd]*XMSTemp[bf3 + bf2*nBasis];
                AXMSTemp[bf4 + bf1*nBasis] += ERIBuffCD[CrossZ]*XMZTemp[bf3 + bf2*nBasis]*iscale;
#ifndef _KRAMERS_SYM_
                AXMSTemp[bf4 + bf1*nBasis] += ERIBuffCD[CrossX]*XMXTemp[bf3 + bf2*nBasis]*iscale;
                AXMSTemp[bf4 + bf1*nBasis] += ERIBuffCD[CrossY]*XMYTemp[bf3 + bf2*nBasis]*iscale;
#endif
 



                AXMXTemp[bf4 + bf1*nBasis] += ERIBuffCD[CrossX]*XMSTemp[bf3 + bf2*nBasis]*iscale;
                AXMXTemp[bf4 + bf1*nBasis] += ERIBuffCD[CrossY]*XMZTemp[bf3 + bf2*nBasis];
#ifndef _KRAMERS_SYM_
                AXMXTemp[bf4 + bf1*nBasis] -= ERIBuffCD[DotPrd]*XMXTemp[bf3 + bf2*nBasis];
                AXMXTemp[bf4 + bf1*nBasis] -= ERIBuffCD[CrossZ]*XMYTemp[bf3 + bf2*nBasis];
#endif
  



                AXMYTemp[bf4 + bf1*nBasis] += ERIBuffCD[CrossY]*XMSTemp[bf3 + bf2*nBasis]*iscale;
                AXMYTemp[bf4 + bf1*nBasis] -= ERIBuffCD[CrossX]*XMZTemp[bf3 + bf2*nBasis];
#ifndef _KRAMERS_SYM_
                AXMYTemp[bf4 + bf1*nBasis] += ERIBuffCD[CrossZ]*XMXTemp[bf3 + bf2*nBasis];
                AXMYTemp[bf4 + bf1*nBasis] -= ERIBuffCD[DotPrd]*XMYTemp[bf3 + bf2*nBasis];
#endif
  



                AXMZTemp[bf4 + bf1*nBasis] += ERIBuffCD[CrossZ]*XMSTemp[bf3 + bf2*nBasis]*iscale;
                AXMZTemp[bf4 + bf1*nBasis] -= ERIBuffCD[DotPrd]*XMZTemp[bf3 + bf2*nBasis];
#ifndef _KRAMERS_SYM_
                AXMZTemp[bf4 + bf1*nBasis] -= ERIBuffCD[CrossY]*XMXTemp[bf3 + bf2*nBasis];
                AXMZTemp[bf4 + bf1*nBasis] += ERIBuffCD[CrossX]*XMYTemp[bf3 + bf2*nBasis];
#endif
  	      }
	    }

	  }

        } // contraction loop

#endif // Contraction


    }; // loop s4
    }; // loop s3
    }; // loop s2
    }; // loop s1


    }; // OpenMP context


#ifdef _REPORT_INTEGRAL_TIMINGS
    size_t nIntSkip = std::accumulate(nSkip.begin(),nSkip.end(),0);
    std::cout << "Screened " << nIntSkip << std::endl;

    auto durDirect = tock(topDirect);
    std::cout << "Dirac-Coulomb AO Direct Contraction took " <<  durDirect << " s\n"; 

    std::cout << std::endl;
#endif

    if (nThreads>1)
    for( auto iMat = 0; iMat < nMat;  iMat++ ) 
    for( auto iTh  = 0; iTh < nThreads; iTh++) {
 
      MatAdd('N','N',nBasis,nBasis,MatsT(1.0),AXthreads[iTh][iMat],nBasis,MatsT(1.0),
         matList[iMat].AX,nBasis,matList[iMat].AX,nBasis);

    };
    

#ifdef _SHZ_SCREEN_4C
    memManager_.free(ShBlkNorms_raw);
#endif

    if(AXRaw != nullptr) memManager_.free(AXRaw);
    if(ERIBuffer != nullptr) memManager_.free(ERIBuffer);

    memManager_.free(buffAll);
    memManager_.free(atm);
    memManager_.free(bas);
    memManager_.free(env);


    // Turn threads for LA back on
    SetLAThreads(LAThreads);

  }

  template <>
  void GTODirectRelERIContraction<double,double>::directScaffoldLibcint(
    MPI_Comm comm, const bool screen,
    std::vector<TwoBodyContraction<double>> &matList) const {
    CErr("Dirac-Coulomb + Real is an invalid option",std::cout);  
  }

  template <>
  void GTODirectRelERIContraction<dcomplex,dcomplex>::directScaffoldLibcint(
    MPI_Comm comm, const bool screen,
    std::vector<TwoBodyContraction<dcomplex>> &matList) const {
    CErr("Complex integral is is an invalid option",std::cout);  
  }

}; // namespace ChronusQ

