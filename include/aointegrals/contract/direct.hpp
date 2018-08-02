/* 
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *  
 *  Copyright (C) 2014-2018 Li Research Group (University of Washington)
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
#ifndef __INCLUDED_AOINTEGRALS_CONTRACT_DIRECT_HPP__
#define __INCLUDED_AOINTEGRALS_CONTRACT_DIRECT_HPP__


#include <aointegrals.hpp>
#include <util/matout.hpp>
#include <cqlinalg/blas1.hpp>
#include <cqlinalg/blasext.hpp>
#include <cqlinalg/blasutil.hpp>

#include <util/threads.hpp>
#include <util/mpi.hpp>
#include <util/time.hpp>
#include <util/math.hpp>

#define _FULL_DIRECT
//#define _SUB_TIMINGS
//#define _REPORT_INTEGRAL_TIMINGS

#define _PRECOMPUTE_SHELL_PAIRS

#define _SHZ_SCREEN

#ifndef _FULL_DIRECT
  #define _BATCH_DIRECT
#endif

#if defined(_FULL_DIRECT) 
  #define _USE_EIGHT_FOLD
#else
  #define _USE_FOUR_FOLD
#endif

#ifndef _FULL_DIRECT
  #warning "Batch Direct ERI contraction is broken for complex"
#endif

#define GetRealPtr(X,I,J,N) reinterpret_cast<double*>(X + I + J*N)

namespace ChronusQ {

  /**
   *  \brief Perform various tensor contractions of the ERI tensor
   *  directly. Wraps other helper functions and provides
   *  loop structure
   *
   *  Currently supports
   *    - Coulomb-type (34,12) contractions
   *    - Exchange-type (23,12) contractions
   *
   *  Works with both real and complex matricies
   *
   *  \param [in/out] list Contains information pertinent to the 
   *    matricies to be contracted with. See TwoBodyContraction
   *    for details
   */ 
  template <typename IntsT>
  template <typename TT>
  void AOIntegrals<IntsT>::twoBodyContractDirect(
    MPI_Comm c, const bool screen,
    std::vector<TwoBodyContraction<TT>> &list, EMPerturbation &pert) {

    // Only use GIAOs if GIAOs are selected and if
    // a Magnetic field is in the EMPerturbation
    bool useGIAO = 
      (basisSet_.basisType == COMPLEX_GIAO) and 
      pert_has_type(pert,Magnetic);

    if( not useGIAO ) directScaffold(c, screen, list);
    else              directScaffold(c, screen, list, pert);
    
  }; // AOIntegrals::twoBodyContractDirect


  template <typename T>
  void ShellBlockNorm(std::vector<libint2::Shell> &shSet, T *MAT, 
    size_t LDM, double *ShBlk) {

    size_t nShell = shSet.size();

    size_t n1,n2;
    for(auto s1(0ul), bf1(0ul); s1 < nShell; s1++, bf1 += n1) {
      n1 = shSet[s1].size();
    for(auto s2(0ul), bf2(0ul); s2 < nShell; s2++, bf2 += n2) {
      n2 = shSet[s2].size();

      T *block = MAT + bf1 + bf2*LDM;
      ShBlk[s1 + s2*nShell] = MatNorm<double>('I',n1,n2,block,LDM);

    }
    }

  };


  template <typename T>
  double * ShellBlockNorm(std::vector<libint2::Shell> &shSet, T *MAT, 
    size_t LDM, CQMemManager &mem) {

    size_t nShell = shSet.size();
    double *ShBlk = mem.template malloc<double>(nShell*nShell);

    ShellBlockNorm(shSet,MAT,LDM,ShBlk);

    return ShBlk;

  };


  template <typename IntsT>
  template <typename TT>
  void AOIntegrals<IntsT>::directScaffold(
    MPI_Comm comm, const bool screen,
    std::vector<TwoBodyContraction<TT>> &list) {

    size_t nthreads  = GetNumThreads();
    size_t LAThreads = GetLAThreads();
    size_t mpiRank   = MPIRank(comm);
    size_t mpiSize   = MPISize(comm);

    SetLAThreads(1); // Turn off parallelism in LA functions

    const size_t NB   = basisSet_.nBasis;
    const size_t NMat = list.size();
    const size_t NS   = basisSet_.nShell;


#ifdef _SHZ_SCREEN
    // Check whether any of the contractions are non-hermetian
    const bool AnyNonHer = std::any_of(list.begin(),list.end(),
      []( TwoBodyContraction<TT> & x ) -> bool { return not x.HER; });

    // Compute schwartz bounds if we haven't already
    if(schwartz == nullptr) computeSchwartz();
#endif


/*
    if( mpiSize > 1 )
      for(auto &C : list )
        prettyPrintSmart(std::cerr,"X in Direct",C.X,NB,NB,NB);
*/


    // Create thread-safe libint2::Engine's
      
    std::vector<libint2::Engine> engines(nthreads);

    // Construct engine for master thread
    engines[0] = libint2::Engine(libint2::Operator::coulomb,
      basisSet_.maxPrim, basisSet_.maxL, 0);





    // Allocate scratch for raw integral batches
    size_t maxShellSize = 
      std::max_element(basisSet_.shells.begin(),basisSet_.shells.end(),
        [](libint2::Shell &sh1, libint2::Shell &sh2) {
          return sh1.size() < sh2.size();
        })->size();

    size_t lenIntBuffer = 
      maxShellSize * maxShellSize * maxShellSize * maxShellSize; 

    lenIntBuffer *= sizeof(TT) / sizeof(double);

    size_t nBuffer = 2;

    size_t nAlloc = nBuffer*lenIntBuffer*nthreads*sizeof(double) + 
      nthreads*NMat*NB*NB*sizeof(TT) +
      list.size()*NS*NS*sizeof(double);
//  std::cerr << "DIRECT CONTRACTION " << nAlloc / 1e9 << std::endl;


    double * intBuffer = 
      memManager_.malloc<double>(nBuffer*lenIntBuffer*nthreads);
   
    double *intBuffer2 = intBuffer + nthreads*lenIntBuffer;


    // Allocate thread local storage to store integral contractions
    // XXX: Don't allocate anything if serial
    std::vector<std::vector<TT*>> AXthreads;
    TT *AXRaw = nullptr;
    if(nthreads != 1) {
      AXRaw = memManager_.malloc<TT>(nthreads*NMat*NB*NB);    
      memset(AXRaw,0,nthreads*NMat*NB*NB*sizeof(TT));
    }

    for(auto ithread = 0, iMat = 0; ithread < nthreads; ithread++) {
      AXthreads.emplace_back();
      for(auto jMat = 0; jMat < NMat; jMat++, iMat++) {
        if(nthreads == 1) {
          AXthreads.back().push_back(list[jMat].AX);
        } else {
          AXthreads.back().push_back(AXRaw + iMat*NB*NB);
        }
      }
    }


#ifdef _SHZ_SCREEN
    // Compute shell block norms
    double *ShBlkNorms_raw = 
      memManager_.malloc<double>(list.size()*NS*NS);

    std::vector<double*> ShBlkNorms;
    for(auto iMat = 0, iOff = 0; iMat < NMat; iMat++, 
      iOff += NS*NS ) {

      ShellBlockNorm(basisSet_.shells,list[iMat].X,NB,
        ShBlkNorms_raw + iOff);

      ShBlkNorms.emplace_back(ShBlkNorms_raw + iOff);

    }

    double maxShBlk = 0.;
    for(auto iMat = 0; iMat < NMat; iMat++)
      maxShBlk = std::max(maxShBlk,
        *std::max_element(ShBlkNorms[iMat],ShBlkNorms[iMat] + NS*NS) ); 


    size_t NP4 = 
      basisSet_.maxPrim * basisSet_.maxPrim * basisSet_.maxPrim * 
      basisSet_.maxPrim;

    engines[0].set_precision(
      std::min(
        std::numeric_limits<double>::epsilon(),
        threshSchwartz / maxShBlk
      ) / NP4
    );



    // Get the max over all the matricies for
    // the shell block norms
    // OVERWRITES ShBlkNorms[0]
    for(auto k = 0; k < NS*NS; k++) {

      double mx = std::abs(ShBlkNorms[0][k]);
      for(auto iMat = 1; iMat < NMat; iMat++)
        mx = std::max(mx,std::abs(ShBlkNorms[iMat][k]));
      ShBlkNorms[0][k] = mx;

    }


    if( AnyNonHer )
    for(auto i = 0; i < NS; i++)
    for(auto j = 0; j <= i; j++) {
      double mx = 
        std::max(std::abs(ShBlkNorms[0][i + j*NS]),
                 std::abs(ShBlkNorms[0][j + i*NS]));

      for(auto iMat = 1; iMat < NMat; iMat++)
        mx = std::max(mx,
          std::max(std::abs(ShBlkNorms[iMat][i + j*NS]),
                   std::abs(ShBlkNorms[iMat][j + i*NS])));

      ShBlkNorms[0][i + j*NS] = mx;
      ShBlkNorms[0][j + i*NS] = mx;

    }


#else
    // Set precision
    engines[0].set_precision(std::numeric_limits<double>::epsilon());
#endif


    // Copy master thread engine to other threads
    for(size_t i = 1; i < nthreads; i++) engines[i] = engines[0];

#ifdef _SUB_TIMINGS
    std::chrono::duration<double> durInner(0.), durCont(0.), durSymm(0.),
      durZero(0.);
#endif

    // Keeping track of number of integrals skipped
    std::vector<size_t> nSkip(nthreads,0);


    // MPI info
    size_t mpiChunks = (NS * (NS + 1) / 2) / mpiSize;
    size_t mpiS12St  = mpiRank * mpiChunks;
    size_t mpiS12End = (mpiRank + 1) * mpiChunks;
    if( mpiRank == (mpiSize - 1) ) mpiS12End = (NS * (NS + 1) / 2);

/*
    double t1 = MPI_Wtime();
    auto topDirect = std::chrono::high_resolution_clock::now();
*/
    auto topDirect = tick();
    #pragma omp parallel
    {

    // Set up thread local storage

    // SMP info
    size_t thread_id = GetThreadID();

    auto &engine = engines[thread_id];
    const auto& buf_vec = engine.results();
    
    auto &AX_loc = AXthreads[thread_id];


    double * intBuffer_loc  = intBuffer  + thread_id*lenIntBuffer;
    double * intBuffer2_loc = intBuffer2 + thread_id*lenIntBuffer;


    size_t n1,n2;

    // Always Loop over s2 <= s1
    for(size_t s1(0ul), bf1_s(0ul), s12(0ul); s1 < NS; bf1_s+=n1, s1++) { 
      n1 = basisSet_.shells[s1].size(); // Size of Shell 1

    auto sigPair12_it = basisSet_.shellData.shData.at(s1).begin();
    for( const size_t& s2 : basisSet_.shellData.sigShellPair[s1] ) {
      size_t bf2_s = basisSet_.mapSh2Bf[s2];
      n2 = basisSet_.shells[s2].size(); // Size of Shell 2

      const auto * sigPair12 = sigPair12_it->get();
      sigPair12_it++;

#ifdef CQ_ENABLE_MPI
      // MPI partition s12 blocks
      if( (s12 < mpiS12St) or (s12 >= mpiS12End) ) { s12++; continue; }
#endif

      // Round-Robbin work distribution
      if( (s12++) % nthreads != thread_id ) continue;


      // Cache variables for shells 1 and 2
        
#ifdef _FULL_DIRECT
      // Deneneracy factor for s1,s2 pair
      double s12_deg = (s1 == s2) ? 1.0 : 2.0;
#endif

#ifdef _SHZ_SCREEN
      double shz12 = 0, shMax12 = 0;
      if( screen ) {
        shz12 = schwartz[s1 + s2*NS];
        shMax12 = ShBlkNorms[0][s1 + s2*NS];
      }
#endif



#ifdef _BATCH_DIRECT

#ifdef _SUB_TIMINGS
      auto topZero = std::chrono::high_resolution_clock::now();
#endif

      // Zero out the integral buffer (hot spot)
      memset(intBuffer_loc,0,lenIntBuffer);

#ifdef _SUB_TIMINGS
      auto botZero = std::chrono::high_resolution_clock::now();
      durZero += botZero - topZero;
#endif

      double *intBuffCur = intBuffer_loc;

#endif


#ifdef _SUB_TIMINGS
      auto topInner = std::chrono::high_resolution_clock::now();
#endif


// The upper bound of s3 is s1 for the 8-fold symmetry and
// nShell for 4-fold.
#ifdef _USE_EIGHT_FOLD
  #define S3_MAX s1
#elif defined(_USE_FOUR_FOLD)
  // the "-" is for the <= in the loop
  #define S3_MAX NS - 1
#endif

      size_t n3,n4;

      for(size_t s3(0ul), bf3_s(0ul), s34(0ul); s3 <= S3_MAX; s3++, bf3_s += n3) { 
        n3 = basisSet_.shells[s3].size(); // Size of Shell 3

#ifdef _SHZ_SCREEN

        double shMax123 = 0;
        if( screen ) {
          // Pre-calculate shell-block norm max's that only
          // depend on shells 1,2 and 3
          shMax123 = 
            std::max(ShBlkNorms[0][s1 + s3*NS], 
                     ShBlkNorms[0][s2 + s3*NS]);

          shMax123 = std::max(shMax123,shMax12);
        }

#endif
        
// The upper bound of s4 is either s2 or s3 based on s1 and s3 for
// the 8-fold symmetry and s3 for the 4-fold symmetry
#ifdef _USE_EIGHT_FOLD
        size_t s4_max = (s1 == s3) ? s2 : s3;
#elif defined(_USE_FOUR_FOLD)
        size_t s4_max =  s3;
#endif

      auto sigPair34_it = basisSet_.shellData.shData.at(s3).begin();
      for( const size_t& s4 : basisSet_.shellData.sigShellPair[s3] ) {

        if (s4 > s4_max)
          break;  // for each s3, s4 are stored in monotonically increasing
                  // order

        const auto * sigPair34 = sigPair34_it->get();
        sigPair34_it++;
                    
        size_t bf4_s = basisSet_.mapSh2Bf[s4];
        n4 = basisSet_.shells[s4].size(); // Size of Shell 4

#ifdef _SHZ_SCREEN

        double shMax = 0;

        if( screen ) {
          // Compute Shell norm max
          shMax = 
            std::max(ShBlkNorms[0][s1 + s4*NS],
            std::max(ShBlkNorms[0][s2 + s4*NS],
                     ShBlkNorms[0][s3 + s4*NS]));

          shMax = std::max(shMax,shMax123);

          if((shMax * shz12 * schwartz[s3 + s4*NS]) < 
             threshSchwartz) { nSkip[thread_id]++; continue; }
        }
#endif
      

#ifdef _FULL_DIRECT

        // Degeneracy factor for s3,s4 pair
        double s34_deg = (s3 == s4) ? 1.0 : 2.0;

        // Degeneracy factor for s1, s2, s3, s4 quartet
        double s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1.0 : 2.0) : 2.0;

        // Total degeneracy factor
        double s1234_deg = s12_deg * s34_deg * s12_34_deg;

#endif

        // Evaluate ERI for shell quartet (s1 s2 | s3 s4)
        engine.compute2<
          libint2::Operator::coulomb, libint2::BraKet::xx_xx, 0>(
          basisSet_.shells[s1],
          basisSet_.shells[s2],
          basisSet_.shells[s3],
          basisSet_.shells[s4]
#ifdef _PRECOMPUTE_SHELL_PAIRS
          ,sigPair12,sigPair34
#endif
        );

        // Libint2 internal screening
        const double *buff = buf_vec[0];

        if(buff == nullptr) { nSkip[thread_id]++; continue; }

#ifdef _BATCH_DIRECT

        // Copy over buffer
        //std::copy_n(buff,n1*n2*n3*n4,intBuffCur);
        memcpy(intBuffCur,buff,n1*n2*n3*n4*sizeof(double));
        intBuffCur += n1*n2*n3*n4;

#elif defined(_FULL_DIRECT)

// Flag to turn contraction on and off
#if 1
        // Scale the buffer by the degeneracy factor and store
        // in infBuffer
        std::transform(buff,buff + n1*n2*n3*n4,intBuffer_loc,
          std::bind1st(std::multiplies<double>(),0.5*s1234_deg));

        size_t b1,b2,b3,b4;
        double *Xp1, *Xp2;
        double X1,X2;
        TT      T1,T2,T3,T4;
        TT      *Tp1,*Tp2;

        for(auto iMat = 0; iMat < NMat; iMat++) {
          
          // Hermetian contraction
          if( list[iMat].HER ) { 
            if( list[iMat].contType == COULOMB )
            for(auto i = 0ul, bf1 = bf1_s, ijkl(0ul); i < n1; i++, bf1++)      
            for(auto j = 0ul, bf2 = bf2_s; j < n2; j++, bf2++) { 
              // Cache i,j variables
              b1 = bf1 + NB*bf2; 
              X1 = *reinterpret_cast<double*>(list[iMat].X  + b1);
              Xp1 = reinterpret_cast<double*>(AX_loc[iMat] + b1);
            for(auto k = 0ul, bf3 = bf3_s; k < n3; k++, bf3++) 
            for(auto l = 0ul, bf4 = bf4_s; l < n4; l++, bf4++, ijkl++) { 

              // J(1,2) += I * X(4,3)
              *Xp1 += *GetRealPtr(list[iMat].X,bf4,bf3,NB) * intBuffer_loc[ijkl];

              // J(4,3) += I * X(1,2)
              *GetRealPtr(AX_loc[iMat],bf4,bf3,NB) +=  X1 * intBuffer_loc[ijkl];

              // J(2,1) and J(3,4) are handled on symmetrization after
              // contraction
            } // kl loop
            } // ij loop

            else if( list[iMat].contType == EXCHANGE )
            for(auto i = 0ul, bf1 = bf1_s, ijkl(0ul); i < n1; i++, bf1++)      
            for(auto j = 0ul, bf2 = bf2_s; j < n2; j++, bf2++)       
            for(auto k = 0ul, bf3 = bf3_s; k < n3; k++, bf3++) {

              // Cache i,j,k variables
              b1 = bf1 + bf3*NB;
              b2 = bf2 + bf3*NB;

              T1 = 0.5 * SmartConj(list[iMat].X[b1]);
              T2 = 0.5 * SmartConj(list[iMat].X[b2]);

            for(auto l = 0ul, bf4 = bf4_s; l < n4; l++, bf4++, ijkl++) { 

              // Indicies are swapped here to loop over contiguous memory
                
              // K(1,3) += 0.5 * I * X(2,4) = 0.5 * I * CONJ(X(4,2)) (**HER**)
              AX_loc[iMat][b1]           += 0.5 * SmartConj(list[iMat].X[bf4+NB*bf2]) * intBuffer_loc[ijkl];

              // K(4,2) += 0.5 * I * X(3,1) = 0.5 * I * CONJ(X(1,3)) (**HER**)
              AX_loc[iMat][bf4 + bf2*NB] += T1 * intBuffer_loc[ijkl];

              // K(4,1) += 0.5 * I * X(3,2) = 0.5 * I * CONJ(X(2,3)) (**HER**)
              AX_loc[iMat][bf4 + bf1*NB] += T2 * intBuffer_loc[ijkl];

              // K(2,3) += 0.5 * I * X(1,4) = 0.5 * I * CONJ(X(4,1)) (**HER**)
              AX_loc[iMat][b2]           += 0.5 * SmartConj(list[iMat].X[bf4+NB*bf1]) * intBuffer_loc[ijkl];

            } // l loop
            } // ijk

          // Nonhermetian contraction
          } else {

            if( list[iMat].contType == COULOMB )
            for(auto i = 0ul, bf1 = bf1_s, ijkl(0ul); i < n1; i++, bf1++)      
            for(auto j = 0ul, bf2 = bf2_s; j < n2; j++, bf2++) { 
              // Cache i,j variables
              b1 = bf1 + NB*bf2; 
              T1 = *(list[iMat].X  + b1);
              Tp1 = (AX_loc[iMat] + b1);

              b2 = bf2 + NB*bf1; 
              T2 = *(list[iMat].X  + b2);
              Tp2 = (AX_loc[iMat] + b2);
            for(auto k = 0ul, bf3 = bf3_s; k < n3; k++, bf3++) 
            for(auto l = 0ul, bf4 = bf4_s; l < n4; l++, bf4++, ijkl++) { 

              // J(1,2) += I * X(4,3)
              *Tp1 += 0.5*( list[iMat].X[bf4 + bf3*NB] + list[iMat].X[bf3 + bf4*NB]) * intBuffer_loc[ijkl];

              // J(3,4) += I * X(2,1)
              AX_loc[iMat][bf3 + bf4*NB] +=  0.5*(T2+T1) * intBuffer_loc[ijkl];

              // J(2,1) += I * X(3,4)
              *Tp2 += 0.5*( list[iMat].X[bf4 + bf3*NB] + list[iMat].X[bf3 + bf4*NB]) * intBuffer_loc[ijkl];

              // J(4,3) += I * X(1,2)
              AX_loc[iMat][bf4 + bf3*NB] +=  0.5*(T2+T1) * intBuffer_loc[ijkl];

            } // kl loop
            } // ij loop

            else if( list[iMat].contType == EXCHANGE )
            for(auto i = 0ul, bf1 = bf1_s, ijkl(0ul); i < n1; i++, bf1++)      
            for(auto j = 0ul, bf2 = bf2_s; j < n2; j++, bf2++)       
            for(auto k = 0ul, bf3 = bf3_s; k < n3; k++, bf3++) {

              // Cache i,j,k variables
              b1 = bf1 + bf3*NB;
              b2 = bf2 + bf3*NB;

              T1 = 0.5 * list[iMat].X[b1];
              T2 = 0.5 * list[iMat].X[b2];

              b3 = bf3 + bf1*NB;
              b4 = bf3 + bf2*NB;

              T3 = 0.5 * list[iMat].X[b3];
              T4 = 0.5 * list[iMat].X[b4];
            for(auto l = 0ul, bf4 = bf4_s; l < n4; l++, bf4++, ijkl++) { 

              // K(3,1) += 0.5 * I * X(4,2)
              AX_loc[iMat][b3]           += 0.5 * list[iMat].X[bf4+NB*bf2] * intBuffer_loc[ijkl];

              // K(4,2) += 0.5 * I * X(3,1)
              AX_loc[iMat][bf4 + bf2*NB] += T3 * intBuffer_loc[ijkl];
 
              // K(4,1) += 0.5 * I * X(3,2)
              AX_loc[iMat][bf4 + bf1*NB] += T4 * intBuffer_loc[ijkl];

              // K(3,2) += 0.5 * I * X(4,1)
              AX_loc[iMat][b4]           += 0.5 * list[iMat].X[bf4+NB*bf1] * intBuffer_loc[ijkl];

              // K(1,3) += 0.5 * I * X(2,4)
              AX_loc[iMat][b1]           += 0.5 * list[iMat].X[bf2+NB*bf4] * intBuffer_loc[ijkl];

              // K(2,4) += 0.5 * I * X(1,3)
              AX_loc[iMat][bf2 + bf4*NB] += T1 * intBuffer_loc[ijkl];
 
              // K(1,4) += 0.5 * I * X(2,3)
              AX_loc[iMat][bf1 + bf4*NB] += T2 * intBuffer_loc[ijkl];

              // K(2,3) += 0.5 * I * X(1,4)
              AX_loc[iMat][b2]           += 0.5 * list[iMat].X[bf1+NB*bf4] * intBuffer_loc[ijkl];

            } // l loop
            } // ijk

          } // Symmetry check

        } // iMat loop

#endif

#endif

      } // loop s4
      } // loop s3

#ifdef _SUB_TIMINGS
      auto botInner = std::chrono::high_resolution_clock::now();

      durInner += botInner - topInner;
#endif

#ifdef _BATCH_DIRECT
#if 0
      assert(nthreads == 1);

#ifdef _SUB_TIMINGS
      auto topSymm = std::chrono::high_resolution_clock::now();
#endif

      // Reorder and expand integrals into square matricies
      for(auto s3 = 0ul, bf3_s = 0ul, ijkl = 0ul; s3 < NS; s3++, 
        bf3_s += n3) { 
        n3 = basisSet_.shells[s3].size();

      for(auto s4 = 0ul, bf4_s = 0ul; s4 <= s3; s4++, bf4_s += n4) { 
        n4 = basisSet_.shells[s4].size();

        for(auto i = 0ul, bf1 = bf1_s; i < n1; i++, bf1++)      
        for(auto j = 0ul, bf2 = bf2_s; j < n2; j++, bf2++)       
        for(auto k = 0ul, bf3 = bf3_s; k < n3; k++, bf3++) 
        for(auto l = 0ul, bf4 = bf4_s; l < n4; l++, bf4++, ijkl++) { 

          intBuffer2_loc[bf4 + bf3*NB + j*nSQ_ + i*nSQ_*n2] = intBuffer_loc[ijkl];
          intBuffer2_loc[bf3 + bf4*NB + j*nSQ_ + i*nSQ_*n2] = intBuffer_loc[ijkl];

        }

      }
      }

#ifdef _SUB_TIMINGS
      auto botSymm = std::chrono::high_resolution_clock::now();
      durSymm += botSymm - topSymm;
#endif
      
     
#ifdef _SUB_TIMINGS
      auto topCont = std::chrono::high_resolution_clock::now();
#endif



      // Perform batched contractions
      for(auto &C : list ) {

        // J Contraction
        if( C.contType == COULOMB ) {

          Gemm('T','N',n1*n2,1,nSQ_,T(1.),intBuffer2_loc,nSQ_,C.X,nSQ_,
            T(0.),reinterpret_cast<G*>(intBuffer_loc),n1*n2);

          // Populate the lower triangle of J contraction storage
          for(auto i = 0ul, bf1 = bf1_s; i < n1; i++, bf1++)      
          for(auto j = 0ul, bf2 = bf2_s; j < n2; j++, bf2++)
            C.AX[bf1 + bf2*NB] = intBuffer_loc[j + i*n2]; 

        // K Contraction
        } else if( C.contType == EXCHANGE ) {

          for(auto i = 0ul, bf1 = bf1_s; i < n1; i++, bf1++)      
          for(auto j = 0ul, bf2 = bf2_s; j < n2; j++, bf2++) {

/*
            // T(m,n) = I(m,k) * X(n,k)
            Gemm('N','T',NB,NB,NB,T(1.),intBuffer2 +j*nSQ_ + i*n2*nSQ_,NB,
              C.X,NB,T(0.),reinterpret_cast<G*>(intBuffer),NB);

            for(auto nu = 0; nu < NB; nu++) {
              C.AX[nu + bf1*NB] += intBuffer[nu + bf2*NB];
              if(s1 != s2) { 
                C.AX[nu + bf2*NB] += intBuffer[nu + bf1*NB];
              }
            }
*/

            Gemm('N','T',NB,1,NB,T(1.),intBuffer2_loc +j*nSQ_ + i*n2*nSQ_,NB,
              C.X + bf2,NB,T(0.),reinterpret_cast<G*>(intBuffer_loc),NB);
            for(auto nu = 0; nu < NB; nu++) 
              C.AX[nu + bf1*NB] += intBuffer_loc[nu];

            if( s1 != s2 ) {

              Gemm('N','T',NB,1,NB,T(1.),intBuffer2_loc +j*nSQ_ + i*n2*nSQ_,NB,
                C.X + bf1,NB,T(0.),reinterpret_cast<G*>(intBuffer_loc),NB);
              for(auto nu = 0; nu < NB; nu++) 
                C.AX[nu + bf2*NB] += intBuffer_loc[nu];

            }

          } // ij loop

        } // Exchange check
      } // Loop over contractions

#ifdef _SUB_TIMINGS
      auto botCont = std::chrono::high_resolution_clock::now();
      durCont += botCont - topCont;
#endif

#endif
#endif

    }; // s2
    }; // s1


    }; // OpenMP context

/*
    auto botDirect = std::chrono::high_resolution_clock::now();

    double t2 = MPI_Wtime();
*/

    auto durDirect = tock(topDirect);

#ifdef _REPORT_INTEGRAL_TIMINGS
    size_t nIntSkip = std::accumulate(nSkip.begin(),nSkip.end(),0);
    std::cerr << "Screened " << nIntSkip << std::endl;

//  std::chrono::duration<double> durDirect = botDirect - topDirect;
    //std::cerr << "Direct Contraction took " << durDirect.count() << " s\n"; 
    std::cerr << "Direct Contraction took " <<  dureDirect << " s\n"; 

#ifdef _SUB_TIMINGS
    std::cerr << "  " << durInner.count() << " (" << durInner.count() / durDirect.count() * 100 
              << "%) Inner loop" << std::endl;
#ifndef _FULL_DIRECT
    std::cerr << "  " << durZero.count() << " (" << durZero.count() / durDirect.count() * 100 
              << "%) Zeroing buffer" << std::endl;
    std::cerr << "  " << durSymm.count() << " (" << durSymm.count() / durDirect.count() * 100 
              << "%) Symmetrization loop" << std::endl;
    std::cerr << "  " << durCont.count() << " (" << durCont.count() / durDirect.count() * 100 
              << "%) Contraction loop" << std::endl;
#endif
#endif
    std::cerr << std::endl;
#endif


#ifdef _FULL_DIRECT

    TT* SCR = this->memManager_.template malloc<TT>(nSQ_);
    for( auto iMat = 0; iMat < NMat;  iMat++ ) 
    for( auto iTh  = 0; iTh < nthreads; iTh++) {
  
    //prettyPrintSmart(std::cerr,"AX " + std::to_string(iMat) + " " + std::to_string(iTh),
    //  AXthreads[iTh][iMat],NB,NB,NB);

      if( list[iMat].HER ) {

        MatAdd('N','C',NB,NB,TT(0.5),AXthreads[iTh][iMat],NB,TT(0.5),
          AXthreads[iTh][iMat],NB,SCR,NB);

        if( nthreads != 1 )
          MatAdd('N','N',NB,NB,TT(1.),SCR,NB,TT(1.), list[iMat].AX,NB,list[iMat].AX,NB);
        else
          SetMat('N',NB,NB,TT(1.),SCR,NB,list[iMat].AX,NB);

      } else {

        if( nthreads != 1 )
          MatAdd('N','N',NB,NB,TT(0.5),AXthreads[iTh][iMat],NB,
            TT(1.), list[iMat].AX,NB,list[iMat].AX,NB);
        else 
          Scale(NB*NB,TT(0.5),list[iMat].AX,1);


      //std::transform(AXthreads[iTh][iMat], AXthreads[iTh][iMat] + NB*NB, 
      //  list[iMat].AX, []( G x ) -> G { return x / 4.; } );
      }

    };
    this->memManager_.free(SCR);
    
#else

    for( auto &C : list ) {

      // Symmetrize J contraction
      if( C.contType == COULOMB ) 
        HerMat('L',NB,C.AX,NB);
  
      // Inplace transpose of K contraction
      if( C.contType == EXCHANGE ) 
        IMatCopy('C',NB,NB,TT(1.),C.AX,NB,NB);

    } // Loop over contractions

#endif


#ifdef CQ_ENABLE_MPI
    // Combine all G[X] contributions onto Root process
    if( mpiSize > 1 ) {

      // FIXME: This should be able to be done with MPI_IN_PLACE for
      // the root process
        
      TT* mpiScr;
      if( mpiRank == 0 ) mpiScr = memManager_.malloc<TT>(NB*NB);

      for( auto &C : list ) {
//      prettyPrintSmart(std::cerr,"AX in Direct",C.AX,NB,NB,NB);

        mxx::reduce( C.AX, NB*NB, mpiScr, 0, std::plus<TT>(), comm );

        // Copy over the output buffer on root
        if( mpiRank == 0 ) std::copy_n(mpiScr,NB*NB,C.AX);

      }

      if( mpiRank == 0 ) memManager_.free(mpiScr);

    }

#endif




#ifdef _SUB_TIMINGS
    auto topFree = std::chrono::high_resolution_clock::now();
#endif

    // Free scratch space
    memManager_.free(intBuffer);
#ifdef _SHZ_SCREEN
    memManager_.free(ShBlkNorms_raw);
#endif
    if(AXRaw != nullptr) memManager_.free(AXRaw);

#ifdef _SUB_TIMINGS
    auto botFree = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> durFree = botFree - topFree;

    std::cerr << "Free took " << durFree.count() << "s" << std::endl;
#endif


    // Turn threads for LA back on
    SetLAThreads(LAThreads);

  };



  template <>
  template <>
  void AOIntegrals<dcomplex>::directScaffold( MPI_Comm comm, const bool screen,
    std::vector<TwoBodyContraction<dcomplex>> &list, EMPerturbation &pert) {
     

    if( MPISize(comm) > 1 ) CErr("GIAO Direct + MPI NYI",std::cout);

    const size_t NB   = basisSet_.nBasis;
    const size_t NMat = list.size();
    const size_t NS   = basisSet_.nShell;

    auto magAmp = pert.getDipoleAmp(Magnetic);  

    // Allocate scratch for raw integral batches
    size_t maxShellSize = 
      std::max_element(basisSet_.shells.begin(),basisSet_.shells.end(),
        [](libint2::Shell &sh1, libint2::Shell &sh2) {
          return sh1.size() < sh2.size();
        })->size();

    size_t lenIntBuffer = 
      maxShellSize * maxShellSize * nSQ_; 

    size_t nBuffer = 2;
    dcomplex * intBuffer = 
      memManager_.malloc<dcomplex>(nBuffer*lenIntBuffer);
   
    // double *intBuffer2 = intBuffer + nthreads*lenIntBuffer;

    dcomplex * alterintBuffer = 
      memManager_.malloc<dcomplex>(nBuffer*lenIntBuffer);


    size_t n1,n2;

    // Always Loop over s2 <= s1
    for(size_t s1(0ul), bf1_s(0ul), s12(0ul); s1 < NS; bf1_s+=n1, s1++) { 
      n1 = basisSet_.shells[s1].size(); // Size of Shell 1

#if 0
    auto sigPair12_it = basisSet_.shellData.shData.at(s1).begin();
    for( const size_t& s2 : basisSet_.shellData.sigShellPair[s1] ) {
#endif 

    for ( int s2 = 0 ; s2 <= s1 ; s2++ ) {
      size_t bf2_s = basisSet_.mapSh2Bf[s2];
      n2 = basisSet_.shells[s2].size(); // Size of Shell 2

#if 0
      const auto * sigPair12 = sigPair12_it->get();
      sigPair12_it++;
#endif 

#ifdef _FULL_DIRECT
      // Deneneracy factor for s1,s2 pair
      double s12_deg = (s1 == s2) ? 1.0 : 2.0;
#endif

/*
#ifdef _BATCH_DIRECT 


      // Zero out the integral buffer (hot spot)
      memset(intBuffer,0,lenIntBuffer);

      double *intBuffCur = intBuffer;

#endif
*/


      //SS Start generate shellpair1 

      libint2::ShellPair pair1_to_use;
      pair1_to_use.init( basisSet_.shells[s1],basisSet_.shells[s2],-1000);

      libint2::ShellPair pair1_to_use_switch;
      pair1_to_use_switch.init( basisSet_.shells[s2],basisSet_.shells[s1],-1000); 

// The upper bound of s3 is s1 for the 8-fold symmetry and
// nShell for 4-fold.
#ifdef _USE_EIGHT_FOLD
  #define S3_MAX s1
#elif defined(_USE_FOUR_FOLD)
  // the "-" is for the <= in the loop
  #define S3_MAX NS - 1
#endif

      size_t n3,n4;

      for(size_t s3(0), bf3_s(0), s34(0); s3 <= S3_MAX; s3++, bf3_s += n3) { 
        n3 = basisSet_.shells[s3].size(); // Size of Shell 3





// The upper bound of s4 is either s2 or s3 based on s1 and s3 for
// the 8-fold symmetry and s3 for the 4-fold symmetry
#ifdef _USE_EIGHT_FOLD
        size_t s4_max = (s1 == s3) ? s2 : s3;
#elif defined(_USE_FOUR_FOLD)
        size_t s4_max =  s3;
#endif

#if 0
      auto sigPair34_it = basisSet_.shellData.shData.at(s3).begin();
      for( const size_t& s4 : basisSet_.shellData.sigShellPair[s3] ) {
#endif 
      for ( int s4 = 0 ; s4 <=s4_max ; s4++ ){     
        if (s4 > s4_max)
          break;  // for each s3, s4 are stored in monotonically increasing
                  // order

#if 0
        const auto * sigPair34 = sigPair34_it->get();
        sigPair34_it++;
#endif 
                    
        size_t bf4_s = basisSet_.mapSh2Bf[s4];
        n4 = basisSet_.shells[s4].size(); // Size of Shell 4


        //SS start generate shellpair2 and calculate GIAO ERI

        libint2::ShellPair pair2_to_use;
        
        pair2_to_use.init( basisSet_.shells[s3],basisSet_.shells[s4],-1000);

/*
        libint2::ShellPair pair2_to_use_switch;
        // switch s3 and s4
        pair2_to_use_switch.init( basisSet_.shells[s4],basisSet_.shells[s3],-1000); 
*/

#ifdef _FULL_DIRECT

        // Degeneracy factor for s3,s4 pair
        double s34_deg = (s3 == s4) ? 1.0 : 2.0;

        // Degeneracy factor for s1, s2, s3, s4 quartet
        double s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1.0 : 2.0) : 2.0;

        // Total degeneracy factor
        double s1234_deg = s12_deg * s34_deg * s12_34_deg;

#endif

        // Evaluate ERI for shell quartet (s1 s2 | s3 s4)  

// std::cout<<"s1 "<<s1<<" s2 "<<s2<<" s3 "<<s3<<" s4 "<<s4<<std::endl;

        // calculate integral (s1,s2|s3,s4)
        auto two2buff = ComplexGIAOIntEngine::computeGIAOERIabcd(pair1_to_use,pair2_to_use,
          basisSet_.shells[s1],basisSet_.shells[s2],
          basisSet_.shells[s3],basisSet_.shells[s4],&magAmp[0]);


        // calculate integral (s1,s2|s4,s3)
        auto two2buff_switch = ComplexGIAOIntEngine::computeGIAOERIabcd(pair1_to_use_switch,pair2_to_use,
          basisSet_.shells[s2],basisSet_.shells[s1],
          basisSet_.shells[s3],basisSet_.shells[s4],&magAmp[0]);
        
        const dcomplex *buff = &(two2buff[0]); 
        const dcomplex *buffswitch = &(two2buff_switch[0]); 

#ifdef _FULL_DIRECT

// Flag to turn contraction on and off
#if 1
        // Scale the buffer by the degeneracy factor and store
        // in infBuffer

        std::transform(buff,buff + n1*n2*n3*n4 , intBuffer,
          std::bind1st(std::multiplies<dcomplex>(),0.5*s1234_deg));

        std::transform(buffswitch,buffswitch+n1*n2*n3*n4 ,alterintBuffer,
          std::bind1st(std::multiplies<dcomplex>(),0.5*s1234_deg));

        size_t b1,b2,b3,b4;
//        double *Xp1, *Xp2;   // what is this used for? 
        dcomplex X1,X2;

        for(auto &C : list ) {
          
          // Hermetian contraction
          if( C.HER ) { 

            if( C.contType == COULOMB )
            for(auto i = 0ul, bf1 = bf1_s, ijkl(0ul); i < n1; i++, bf1++)      
            for(auto j = 0ul, bf2 = bf2_s; j < n2; j++, bf2++) { 
              // bf1, bf2 are the index in the matrix
              // b1 is the index in the whole trunk of number 
              // Cache i,j variables
              b1 = bf1 + NB*bf2; 
              
              // in GIAO, J is complex. So X1 and Xp1 are not required 
              
              // X1 = *reinterpret_cast<double*>(list[iMat].X  + b1);
              // Xp1 = reinterpret_cast<double*>(AX_loc[iMat] + b1);
            for(auto k = 0ul, bf3 = bf3_s; k < n3; k++, bf3++) 
            for(auto l = 0ul, bf4 = bf4_s; l < n4; l++, bf4++, ijkl++) {

              int jikl;
              jikl = j*n1*n3*n4 + i*n3*n4 + k*n4 + l;  


              // J(1,2) += 1/2[I(1,2|3,4) * X(4,3) + I(1,2|4,3) * X(3,4)]
              C.AX[b1] += 0.5 * (C.X[bf4+bf3*NB] * intBuffer[ijkl]
                            + C.X[bf3+bf4*NB] * std::conj(alterintBuffer[jikl]));     

              // J(4,3) += 1/2[I(4,3|2,1) * X(1,2)+I(4,3|1,2) * X(2,1)
              //         = 1/2[I(1,2|3,4)* *X(1,2)+I(1,2|4,3) * X(2,1)
              C.AX[bf4+bf3*NB] +=  0.5 *( C.X[b1] 
                                * std::conj(intBuffer[ijkl])
                            + C.X[bf2+bf1*NB] * std::conj(alterintBuffer[jikl]));
                                                                             
              // J(2,1) and J(3,4) are handled on symmetrization after
              // contraction
                
            } // kl loop
            } // ij loop

            else if( C.contType == EXCHANGE )
            for(auto i = 0ul, bf1 = bf1_s, ijkl(0ul); i < n1; i++, bf1++)      
            for(auto j = 0ul, bf2 = bf2_s; j < n2; j++, bf2++)       
            for(auto k = 0ul, bf3 = bf3_s; k < n3; k++, bf3++) {

              // Cache i,j,k variables
              b1 = bf1 + bf3*NB;
              b2 = bf2 + bf3*NB;

              dcomplex T1 = 0.5 * SmartConj(C.X[b1]);
              dcomplex T2 = 0.5 * SmartConj(C.X[b2]);

            for(auto l = 0ul, bf4 = bf4_s; l < n4; l++, bf4++, ijkl++) { 

              int jikl;
              jikl = j*n1*n4*n3 + i*n4*n3 + k*n4 + l;  
              // Indicies are swapped here to loop over contiguous memory
                
              // K(1,3) += 0.5 * I(1,2|4,3) * X(2,4) = 0.5 * I * CONJ(X(4,2)) (**HER**)
              C.AX[b1]           += 0.5 * SmartConj(C.X[bf4+NB*bf2]) * std::conj(alterintBuffer[jikl]);

              // K(4,2) += 0.5 * I(4,3|1,2) * X(3,1) = 0.5 * I * CONJ(X(1,3)) (**HER**)
              C.AX[bf4 + bf2*NB] += T1 * std::conj(alterintBuffer[jikl]);

              // K(4,1) += 0.5 * I(4,3|2,1) * X(3,2) = 0.5 * I * CONJ(X(2,3)) (**HER**)
              C.AX[bf4 + bf1*NB] += T2 * std::conj( intBuffer[ijkl] );

              // K(2,3) += 0.5 * I(2,1|4,3) * X(1,4) = 0.5 * I * CONJ(X(4,1)) (**HER**)
              C.AX[b2]           += 0.5 * SmartConj(C.X[bf4+NB*bf1]) * std::conj( intBuffer[ijkl] );

            } // l loop
            } // ijk

          } else {    // here is non Hermitian contraction   


            if( C.contType == COULOMB )
            for(auto i = 0ul, bf1 = bf1_s, ijkl(0ul); i < n1; i++, bf1++)      
            for(auto j = 0ul, bf2 = bf2_s; j < n2; j++, bf2++) { 
              // Cache i,j variables
              b1 = bf1 + NB*bf2; 
              // T1 = *(C.X  + b1);
              // Tp1 = (AX_loc[iMat] + b1);

              b2 = bf2 + NB*bf1; 
              // T2 = *(list[iMat].X  + b2);
              // Tp2 = (AX_loc[iMat] + b2);
            for(auto k = 0ul, bf3 = bf3_s; k < n3; k++, bf3++) 
            for(auto l = 0ul, bf4 = bf4_s; l < n4; l++, bf4++, ijkl++) { 

              int jikl;
              jikl = j*n1*n3*n4 + i*n3*n4 + k*n4 + l;  

              // J(1,2) += 1/2*(I(1,2|3,4) * X(4,3) + I(1,2|4,3) * X(3,4) 
              C.AX[b1] += 0.5*( C.X[bf4 + bf3*NB]*intBuffer[ijkl] 
                          + C.X[bf3 + bf4*NB]) * std::conj(alterintBuffer[jikl]);

              // J(4,3) += 1/2[I(4,3|2,1) * X(1,2)+I(4,3|1,2) * X(2,1)
              //         = 1/2[I(1,2|3,4)* *X(1,2)+I(1,2|4,3) * X(2,1)
              C.AX[bf4+bf3*NB] +=  0.5 * (C.X[b1] 
                                * std::conj(intBuffer[ijkl])
                            + C.X[bf2+bf1*NB] * std::conj(alterintBuffer[jikl]));

              // J(2,1) += 1/2(I(2,1|3,4) * X(4,3) + I(2,1|4,3)* X(3,4))
              C.AX[bf2+bf1*NB] += 0.5*( C.X[bf4 + bf3*NB] * alterintBuffer[jikl] 
                                + C.X[bf3 + bf4*NB] * std::conj(intBuffer[ijkl]));

              // J(3,4) += 1/2(I(3,4|1,2) * X(2,1) + I(3,4|2,1) * X(1,2))
              C.AX[bf3 + bf4*NB] += 0.5*( C.X[bf2 + bf1*NB] * intBuffer[ijkl]
                                    +C.X[b1] * alterintBuffer[jikl]) ;

            } // kl loop
            } // ij loop

            else if( C.contType == EXCHANGE )
            for(auto i = 0ul, bf1 = bf1_s, ijkl(0ul); i < n1; i++, bf1++)      
            for(auto j = 0ul, bf2 = bf2_s; j < n2; j++, bf2++)       
            for(auto k = 0ul, bf3 = bf3_s; k < n3; k++, bf3++) {

              // Cache i,j,k variables
              b1 = bf1 + bf3*NB;
              b2 = bf2 + bf3*NB;

              // T1 = 0.5 * list[iMat].X[b1];
              // T2 = 0.5 * list[iMat].X[b2];

              b3 = bf3 + bf1*NB;
              b4 = bf3 + bf2*NB;

              // T3 = 0.5 * list[iMat].X[b3];
              // T4 = 0.5 * list[iMat].X[b4];
            for(auto l = 0ul, bf4 = bf4_s; l < n4; l++, bf4++, ijkl++) { 

              int jikl;
              jikl = j*n1*n3*n4 + i*n3*n4 + k*n4 + l;  

              // K(3,1) += 0.5 * I(3,4|2,1) * X(4,2)
              C.AX[b3]           += 0.5 * C.X[bf4+NB*bf2] * alterintBuffer[jikl];

              // K(4,2) += 0.5 * I(4,3|1,2) * X(3,1)
              C.AX[bf4 + bf2*NB] += 0.5 * C.X[b3] * std::conj(alterintBuffer[jikl]);
 
              // K(4,1) += 0.5 * I(4,3|2,1) * X(3,2)
              C.AX[bf4 + bf1*NB] += 0.5 * C.X[b4] * std::conj(intBuffer[ijkl]);

              // K(3,2) += 0.5 * I(3,4|1,2) * X(4,1)
              C.AX[b4]           += 0.5 * C.X[bf4+NB*bf1] * intBuffer[ijkl];

              // K(1,3) += 0.5 * I(1,2|4,3) * X(2,4)
              C.AX[b1]           += 0.5 * C.X[bf2+NB*bf4] * std::conj(alterintBuffer[jikl]);

              // K(2,4) += 0.5 * I(2,1|3,4) * X(1,3)
              C.AX[bf2 + bf4*NB] += 0.5 * C.X[b1] * alterintBuffer[jikl];
 
              // K(1,4) += 0.5 * I(1,2|3,4) * X(2,3)
              C.AX[bf1 + bf4*NB] += 0.5 * C.X[b2] * intBuffer[ijkl];

              // K(2,3) += 0.5 * I(2,1|4,3) * X(1,4)
              C.AX[b2]           += 0.5 * C.X[bf1+NB*bf4] * std::conj(intBuffer[ijkl]);

            } // l loop
            } // ijk

          } // non Hermitian finished 


        } // iMat loop

#endif
// this is for if 1

#endif
// this is for ifdef defined(_FULL_DIRECT) 


      } // loop s4
      } // loop s3



    }; // s2
    }; // s1


#ifdef _FULL_DIRECT

    for( auto &C : list ) {   
  
    //prettyPrintSmart(std::cerr,"AX " + std::to_string(iMat) + " " + std::to_string(iTh),
    //  AXthreads[iTh][iMat],NB,NB,NB);

      if( C.HER ) {

        MatAdd('N','C',NB,NB,dcomplex(0.5),C.AX,NB,dcomplex(0.5),
          C.AX,NB,intBuffer,NB);

/*
        if( nthreads != 1 )
          MatAdd('N','N',NB,NB,G(1.),reinterpret_cast<G*>(intBuffer),NB,
            G(1.), list[iMat].AX,NB,list[iMat].AX,NB);
        else
*/
          SetMat('N',NB,NB,dcomplex(1.),intBuffer,NB,
            C.AX,NB);

      } else {
/*
        if( nthreads != 1 )
          MatAdd('N','N',NB,NB,G(0.5),AXthreads[iTh][iMat],NB,
            G(1.), list[iMat].AX,NB,list[iMat].AX,NB);
        else 
*/
          Scale(NB*NB,dcomplex(0.5),C.AX,1);


      //std::transform(AXthreads[iTh][iMat], AXthreads[iTh][iMat] + NB*NB, 
      //  list[iMat].AX, []( G x ) -> G { return x / 4.; } );
      }  
    } // for( auto &C : list )
    

#endif



    // Free scratch space
    memManager_.free(intBuffer);

    // if(AXRaw != nullptr) memManager_.free(AXRaw);





  }

  template <>
  template <>
  void AOIntegrals<double>::directScaffold(
    MPI_Comm c, const bool b, 
    std::vector<TwoBodyContraction<double>> &list, EMPerturbation &pert) {
    CErr("GIAO + Real is an invalid option",std::cout);  
  }

  template <>
  template <>
  void AOIntegrals<double>::directScaffold(
    MPI_Comm c, const bool b, 
    std::vector<TwoBodyContraction<dcomplex>> &list, EMPerturbation &pert) {
    CErr("GIAO + Real is an invalid option",std::cout);  
  }



}; // namespace ChronusQ

#endif
