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

#include <cqlinalg.hpp>
#include <cqlinalg/blasutil.hpp>
#include <util/time.hpp>
#include <util/matout.hpp>
#include <electronintegrals/twoeints/incore4indexeri.hpp>
#include <electronintegrals/twoeints/incorerieri.hpp>
#include <electronintegrals/twoeints/gtodirecteri.hpp>

#include <util/threads.hpp>
#include <chrono>
// Debug directives
//#define _DEBUGORTHO
//#define __DEBUGERI__

//#define __IN_HOUSE_INT__

namespace ChronusQ {

  typedef std::vector<libint2::Shell> shell_set; 


  /**
   *  \brief Compute and store the full rank-4 ERI tensor using
   *  Libint2 over the CGTO basis.
   */ 
  template <>
  void InCore4indexERI<double>::computeAOInts(BasisSet &basisSet, Molecule&,
      EMPerturbation&, OPERATOR op, const AOIntsOptions &options) {

    if (op != ELECTRON_REPULSION)
      CErr("Only Electron repulsion integrals in InCore4indexERI<double>",std::cout);
    if (options.basisType != REAL_GTO)
      CErr("Only Real GTOs are allowed in InCore4indexERI<double>",std::cout);

    // Determine the number of OpenMP threads
    int nthreads = GetNumThreads();
 
    // Create a vector of libint2::Engines for possible threading
    std::vector<libint2::Engine> engines(nthreads);

    // Initialize the first engine for the integral evaluation
    engines[0] = libint2::Engine(libint2::Operator::coulomb,
      basisSet.maxPrim,basisSet.maxL,0);
    engines[0].set_precision(0.);


    // Copy over the engines to other threads if need be
    for(size_t i = 1; i < nthreads; i++) engines[i] = engines[0];

    std::fill_n(ERI,NB2*NB2,0.);
    InCore4indexERI<double> &eri4I = *this;

#if 0    
    // HBL 4C: populate ERISOC
    try { ERISOC = memManager_.malloc<double>(NB4); } 
    catch(...) {
      std::cout << std::fixed;
      std::cout << "Insufficient memory for the full ERI SOC tensor (" 
                << (NB4/1e9) * sizeof(double) << " GB)" << std::endl;
      std::cout << std::endl << memManager_ << std::endl;
      CErr();
    }
    std::fill_n(ERISOC,NB4,0.);
    //
#endif

#ifndef __IN_HOUSE_INT__
    std::cout<<"Using Libint "<<std::endl;
#else
    std::cout<<"Using In-house Integral Engine "<<std::endl;
#endif

    auto libint_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
      int thread_id = GetThreadID();

      // Get threads result buffer
      const auto& buf_vec = engines[thread_id].results();

      size_t n1,n2,n3,n4,i,j,k,l,ijkl,bf1,bf2,bf3,bf4;
      size_t s4_max;
      for(size_t s1(0), bf1_s(0), s1234(0); s1 < basisSet.nShell;
          bf1_s+=n1, s1++) { 

        n1 = basisSet.shells[s1].size(); // Size of Shell 1

      for(size_t s2(0), bf2_s(0); s2 <= s1; bf2_s+=n2, s2++) {

        n2 = basisSet.shells[s2].size(); // Size of Shell 2

#ifdef __IN_HOUSE_INT__
        libint2::ShellPair pair1_to_use;
        pair1_to_use.init( basisSet_.shells[s1],basisSet_.shells[s2],-2000);
#endif

      for(size_t s3(0), bf3_s(0); s3 <= s1; bf3_s+=n3, s3++) {

        n3 = basisSet.shells[s3].size(); // Size of Shell 3
        s4_max = (s1 == s3) ? s2 : s3; // Determine the unique max of Shell 4

      for(size_t s4(0), bf4_s(0); s4 <= s4_max; bf4_s+=n4, s4++, s1234++) {

        n4 = basisSet.shells[s4].size(); // Size of Shell 4

#ifdef __IN_HOUSE_INT__
        libint2::ShellPair pair2_to_use;
        pair2_to_use.init( basisSet_.shells[s3],basisSet_.shells[s4],-2000);

#endif

        // Round Robbin work distribution
        #ifdef _OPENMP
        if( s1234 % nthreads != thread_id ) continue;
        #endif

        // Evaluate ERI for shell quartet
#ifndef __IN_HOUSE_INT__
        engines[thread_id].compute2<
          libint2::Operator::coulomb, libint2::BraKet::xx_xx, 0>(
          basisSet.shells[s1],
          basisSet.shells[s2],
          basisSet.shells[s3],
          basisSet.shells[s4]
        );
        const auto *buff =  buf_vec[0] ;
        if(buff == nullptr) continue;
#else
        auto buff  = RealGTOIntEngine::BottomupHGP(pair1_to_use,pair2_to_use,
          basisSet_.shells[s1],
          basisSet_.shells[s2],
          basisSet_.shells[s3],
          basisSet_.shells[s4]
        );
#endif
// HBL 4C
//        std::cout << "Filter Ethan's (2NB)**4 intergal to (NB)**4 LLSS" << std::endl;
//        auto twoesp  = RealGTOIntEngine::BottomupHGP_TwoESP(pair1_to_use,pair2_to_use,
//          basisSet_.shells[s1],
//          basisSet_.shells[s2],
//          basisSet_.shells[s3],
//          basisSet_.shells[s4]
//        );
// HBL 4C

        // Libint2 internal screening

        // Place shell quartet into persistent storage with
        // permutational symmetry
        for(i = 0ul, bf1 = bf1_s, ijkl = 0ul ; i < n1; ++i, bf1++) 
        for(j = 0ul, bf2 = bf2_s             ; j < n2; ++j, bf2++) 
        for(k = 0ul, bf3 = bf3_s             ; k < n3; ++k, bf3++) 
        for(l = 0ul, bf4 = bf4_s             ; l < n4; ++l, bf4++, ++ijkl) {


// SS start compare the difference
/*
  std::cout<<"LA "<<basisSet_.shells[s1].contr[0].l
  <<" LB "<<basisSet_.shells[s2].contr[0].l
  <<" LC "<<basisSet_.shells[s3].contr[0].l 
  <<" LD "<<basisSet_.shells[s4].contr[0].l<<std::endl;
  std::cout<<"s1 "<<s1<<" s2 "<<s2<<" s3 "<<s3<<" s4 "<<s4<<std::endl;
  std::cout<<"  buff integral "<<std::setprecision(12)<<buff[ijkl];
  std::cout<<"  own integral  "<<std::setprecision(12)<<two2buff[ijkl]<<std::endl;

        std::cout << "bottomup duration = " << bottomup_duration.count() << std::endl;
        std::cout << "Time Ratio        = " << bottomup_duration.count() / libint_duration.count() << std::endl;
          */
// SS end


            // (12 | 34)
            eri4I(bf1, bf2, bf3, bf4) = buff[ijkl];
            // (12 | 43)
            eri4I(bf1, bf2, bf4, bf3) = buff[ijkl];
            // (21 | 34)
            eri4I(bf2, bf1, bf3, bf4) = buff[ijkl];
            // (21 | 43)
            eri4I(bf2, bf1, bf4, bf3) = buff[ijkl];
            // (34 | 12)
            eri4I(bf3, bf4, bf1, bf2) = buff[ijkl];
            // (43 | 12)
            eri4I(bf4, bf3, bf1, bf2) = buff[ijkl];
            // (34 | 21)
            eri4I(bf3, bf4, bf2, bf1) = buff[ijkl];
            // (43 | 21)
            eri4I(bf4, bf3, bf2, bf1) = buff[ijkl];


        }; // ijkl loop
      }; // s4
      }; // s3
      }; // s2
      }; // s1
    }; // omp region

    auto libint_stop = std::chrono::high_resolution_clock::now();
    auto libint_duration = std::chrono::duration_cast<std::chrono::milliseconds>(libint_stop - libint_start);
    std::cout << "Libint duration   = " << libint_duration.count() << std::endl;

    // Debug output of the ERIs
#ifdef __DEBUGERI__
    std::cout << "Two-Electron Integrals (ERIs)" << std::endl;
    for(auto i = 0ul; i < NB; i++)
    for(auto j = 0ul; j < NB; j++)
    for(auto k = 0ul; k < NB; k++)
    for(auto l = 0ul; l < NB; l++){
      std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
      std::cout << ERI[i + j*NB  + k*NB2 + l*NB3] << std::endl;
    };
#endif
  }; // InCore4indexERI<double>::computeERI


  /**
   *  \brief Allocate and evaluate the Schwartz bounds over the
   *  CGTO shell pairs.
   */ 
//  template <>
//  void AOIntegrals<dcomplex>::computeSchwartz() {
//    CErr("Only real GTOs are allowed",std::cout);
//  };
  template <typename IntsT>
  void DirectERI<IntsT>::computeSchwartz() {

    CQMemManager &memManager_ = this->memManager();

    if( schwartz() != nullptr ) memManager_.free(schwartz());

    // Allocate the schwartz tensor
    schwartz() = memManager_.malloc<double>(basisSet().nShell*basisSet().nShell);

    // Define the libint2 integral engine
    libint2::Engine engine(libint2::Operator::coulomb,
      basisSet().maxPrim,basisSet().maxL,0);

    engine.set_precision(0.); // Don't screen prims during evaluation

    const auto &buf_vec = engine.results();

    auto topSch = std::chrono::high_resolution_clock::now();
  
    size_t n1,n2;
    for(auto s1(0ul); s1 < basisSet().nShell; s1++) {
      n1 = basisSet().shells[s1].size(); // Size shell 1
    for(auto s2(0ul); s2 <= s1; s2++) {
      n2 = basisSet().shells[s2].size(); // Size shell 2



      // Evaluate the shell quartet (s1 s2 | s1 s2)
      engine.compute(
        basisSet().shells[s1],
        basisSet().shells[s2],
        basisSet().shells[s1],
        basisSet().shells[s2]
      );

      if(buf_vec[0] == nullptr) continue;

      // Allocate space to hold the diagonals
      double* diags = memManager_.malloc<double>(n1*n2);

      for(auto i(0), ij(0); i < n1; i++)
      for(auto j(0); j < n2; j++, ij++)
        diags[i + j*n1] = buf_vec[0][ij*n1*n2 + ij];


      schwartz()[s1 + s2*basisSet().nShell] =
        std::sqrt(MatNorm<double>('I',n1,n2,diags,n1));

      // Free up space
      memManager_.free(diags);

    } // loop s2
    } // loop s1

    auto botSch = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> durSch = botSch - topSch;

    HerMat('L',basisSet().nShell,schwartz(),basisSet().nShell);

#if 0
    prettyPrintSmart(std::cout,"Schwartz",schwartz,basisSet_.nShell,
      basisSet_.nShell,basisSet_.nShell);
#endif

  }; // DirectERI<double>::computeSchwartz
  template void DirectERI<double>::computeSchwartz();
  template void DirectERI<dcomplex>::computeSchwartz();

  /**
   *  \brief Compute and store the Auxiliary basis RI
   *  3-index ERI tensor using Libint2 over the CGTO basis.
   */ 
  template <>
  void InCoreAuxBasisRIERI<dcomplex>::computeAOInts(BasisSet&, Molecule&,
      EMPerturbation&, OPERATOR, const AOIntsOptions&) {
    CErr("Only real GTOs are allowed",std::cout);
  };
  template <>
  void InCoreAuxBasisRIERI<double>::computeAOInts(BasisSet &basisSet, Molecule&,
      EMPerturbation&, OPERATOR op, const AOIntsOptions &options) {

    if (op != ELECTRON_REPULSION)
      CErr("Only Electron repulsion integrals in InCoreAuxBasisRIERI<double>",std::cout);
    if (options.basisType != REAL_GTO)
      CErr("Only Real GTOs are allowed in InCoreAuxBasisRIERI<double>",std::cout);

    std::cout<<"Using Libint-RI "<<std::endl;
    auto topLibintRI = tick();

    // Determine the number of OpenMP threads
    int nthreads = GetNumThreads();
 
    // Create a vector of libint2::Engines for possible threading
    std::vector<libint2::Engine> engines(nthreads);

    // Initialize the first engine for the integral evaluation
    engines[0] = libint2::Engine(libint2::Operator::coulomb,
      std::max(basisSet.maxPrim, auxBasisSet_->maxPrim),
      std::max(basisSet.maxL, auxBasisSet_->maxL),0);
    engines[0].set_precision(0.);
    engines[0].set(libint2::BraKet::xs_xx);
    const auto& unitshell = libint2::Shell::unit();

    // Copy over the engines to other threads if need be
    for(size_t i = 1; i < nthreads; i++) engines[i] = engines[0];

    // Allocate and zero out ERIs
    size_t NB    = basisSet.nBasis;
    size_t NBRI  = auxBasisSet_->nBasis;
    size_t NB2   = NB*NB;
    size_t NBRI2 = NBRI*NBRI;
    size_t NBNBRI= NB*NBRI;
    size_t NB3   = NB2*NBRI;


    //********************************************************************//
    // Compute three-center ERI (Q|ij) where Q is the auxiliary basis,    //
    // and i,j are regular AO basis.                                      //
    //********************************************************************//
    auto topERI3 = tick();
    std::fill_n(ERI3J,NB3,0.);
    InCoreRIERI<double> &eri3j = *this;

//    try { ERI3K = memManager_.malloc<double>(NB3); } 
//    catch(...) {
//      std::cout << std::fixed;
//      std::cout << "Insufficient memory for the full RI-ERI tensor (" 
//                << (NB3/1e9) * sizeof(double) << " GB)" << std::endl;
//      std::cout << std::endl << memManager_ << std::endl;
//      CErr();
//    }
//    std::fill_n(ERI3K,NB3,0.);

    #pragma omp parallel
    {
      int thread_id = GetThreadID();

      // Get threads result buffer
      const auto& buf_vec = engines[thread_id].results();

      size_t n1,n2,n3,i,j,k,ijk,bf1,bf2,bf3;
      size_t nShellDF = auxBasisSet_->nShell;
      size_t nShell   = basisSet.nShell;
      // The outer loop runs over all DFBasis shells
      for(auto s1=0ul, bf1_s=0ul, s123=0ul; s1 < nShellDF; bf1_s+=n1, s1++) { 

        n1 = auxBasisSet_->shells[s1].size(); // Size of DFShell 1

      for(auto s2=0ul, bf2_s=0ul; s2 < nShell; bf2_s+=n2, s2++) {

        n2 = basisSet.shells[s2].size(); // Size of Shell 2

      for(auto s3=0ul, bf3_s=0ul; s3 < nShell; bf3_s+=n3, s3++, s123++) {

        n3 = basisSet.shells[s3].size(); // Size of Shell 3

        // Round Robbin work distribution
        #ifdef _OPENMP
        if( s123 % nthreads != thread_id ) continue;
        #endif

        // Evaluate ERI3 for shell quartet
        engines[thread_id].compute2<
          libint2::Operator::coulomb, libint2::BraKet::xs_xx, 0>(
          auxBasisSet_->shells[s1],
          unitshell,
          basisSet.shells[s2],
          basisSet.shells[s3]
        );
        const auto *buff =  buf_vec[0] ;
        if(buff == nullptr) continue;

        // Place shell triplet into persistent storage
        for(i = 0ul, bf1 = bf1_s, ijk = 0ul  ; i < n1; ++i, bf1++) 
        for(j = 0ul, bf2 = bf2_s             ; j < n2; ++j, bf2++) 
        for(k = 0ul, bf3 = bf3_s             ; k < n3; ++k, bf3++, ++ijk) {
          // (Q|12) -> RI-J
          eri3j(bf1, bf2, bf3) = buff[ijk];
        }; // ijk loop
      }; // s3
      }; // s2
      }; // s1
    }; // omp region

    auto durERI3 = tock(topERI3);
    std::cout << "Libint-RI-ERI3 duration   = " << durERI3 << " s " << std::endl;

    // reset integral engines to two-center integrals over DFBasis
    for(auto i = 0; i < nthreads; i++) engines[i].set(libint2::BraKet::xs_xs);

    auto ERI2PQ = memManager().malloc<double>(NBRI2);
    std::fill_n(ERI2PQ,NBRI2,0.);

    //********************************************************************//
    // Compute two-center ERI (P|Q) where Q is the auxiliary basis.       //
    //********************************************************************//
    auto topERI2 = tick();
    #pragma omp parallel
    {
      int thread_id = GetThreadID();

      // Get threads result buffer
      const auto& buf_vec = engines[thread_id].results();

      size_t n1,n2,i,j,ij,bf1,bf2;
      size_t nShellDF = auxBasisSet_->nShell;
      // The outer loop runs over all DFBasis shells
      for(auto s1=0ul, bf1_s=0ul, s12=0ul; s1 < nShellDF; bf1_s+=n1, s1++) { 

        n1 = auxBasisSet_->shells[s1].size(); // Size of DFShell 1

      for(auto s2=0ul, bf2_s=0ul; s2 < nShellDF; bf2_s+=n2, s2++, s12++) {

        n2 = auxBasisSet_->shells[s2].size(); // Size of DFShell 2

        // Round Robbin work distribution
        #ifdef _OPENMP
        if( s12 % nthreads != thread_id ) continue;
        #endif

        // Evaluate ERI2 for shell quartet
        engines[thread_id].compute2<
          libint2::Operator::coulomb, libint2::BraKet::xs_xs, 0>(
          auxBasisSet_->shells[s1],
          unitshell,
          auxBasisSet_->shells[s2],
          unitshell
        );
        const auto *buff =  buf_vec[0] ;
        if(buff == nullptr) continue;

        // Place shell doublet into persistent storage
        for(i = 0ul, bf1 = bf1_s, ij = 0ul ; i < n1; ++i, bf1++) 
        for(j = 0ul, bf2 = bf2_s           ; j < n2; ++j, bf2++, ++ij) {
          // (1 | 2)
          ERI2PQ[bf1 + bf2*NBRI ] = buff[ij];
        }; // ij loop
      }; // s2
      }; // s1
    }; // omp region
    auto durERI2 = tock(topERI2);
    std::cout << "Libint-RI-ERI2 duration   = " << durERI2 << " s " << std::endl;

    //********************************************************************//
    // Construct L=(P|Q)^{-1/2} and Contract with (Q|ij) to form L(Q|ij)  //
    // Save L(Q|ij) in ERI3J                                              //
    //********************************************************************//
    auto topERI3Trans = tick();

    // ERI2PQ = L L^H -> L^H
    int INFO = Cholesky('U',NBRI,ERI2PQ,NBRI);

    if (INFO)
      CErr("Error in Cholesky decomposition of auxiliary basis potential matrix.");

    // Zero out LowerTriangular
    #pragma omp parallel for
    for (size_t i = 0; i < NBRI; i++)
      std::fill_n(ERI2PQ + i*NBRI + i + 1, NBRI - i - 1, 0.0);

    // ERI2PQ = L^H^-1
    INFO = TriInv('U','N',NBRI,ERI2PQ,NBRI);

    if (INFO)
      CErr("Error in inverse of Cholesky decomposed auxiliary basis potential matrix.");

    // S^{-1/2}(Q|ij)
    auto ijK = memManager().malloc<double>(NB3);
    std::fill_n(ijK,NB3,0.);
    Gemm('T','N',NBRI,NB2,NBRI,double(1.),ERI2PQ,NBRI,eri3j.pointer(),NBRI,double(0.),ijK,NBRI);

    auto durERI3Trans = tock(topERI3Trans);
    std::cout << "Libint-RI-ERI3-Transformation duration   = " << durERI3Trans << " s " << std::endl;
    //----------------------------------------------------------------
    
#ifdef __DEBUGERI__
    // Debug output of the ERIs
    auto TempERI4 = memManager_.malloc<double>(NB2*NB2);
    Gemm('N','T',NB2,NB2,NBRI,double(1.),ijK,NB2,ijK,NB2,double(0.),TempERI4,NB2);
    std::cout << "Two-Electron Integrals (ERIs)" << std::endl;
    for(auto i = 0ul; i < NB; i++)
    for(auto j = 0ul; j < NB; j++)
    for(auto k = 0ul; k < NB; k++)
    for(auto l = 0ul; l < NB; l++){
      std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
      std::cout << TempERI4[i + j*NB  + k*NB2 + l*NB2*NB] << std::endl;
    };
    memManager_.free<double>(TempERI4);
#endif
    std::copy(ijK, ijK+NB3, eri3j.pointer());

//    auto topERI3K = tick();
//    std::copy(ijK, ijK+NB3, ERI3K);
//    IMatCopy('T',NB,NBNBRI,double(1.),ERI3K,NB,NBNBRI);
//    auto durERI3K = tock(topERI3K);
//    std::cout << "Libint-RI-ERI3K duration   = " << durERI3K << " s " << std::endl;

    auto durLibintRI = tock(topLibintRI);
    std::cout << "Libint-RI duration   = " << durLibintRI << " s " << std::endl;

    memManager().free<double>(ijK);
    memManager().free<double>(ERI2PQ);
  }; // InCoreAuxBasisRIERI<double>::computeERI

  /**
   *  \brief Allocate, compute and store the Cholesky RI
   *  3-index ERI tensor using Libint2 over the CGTO basis.
   */
  template <>
  void InCoreCholeskyRIERI<dcomplex>::computeAOInts(BasisSet&, Molecule&,
      EMPerturbation&, OPERATOR, const AOIntsOptions&) {
    CErr("Only real GTOs are allowed",std::cout);
  };
  template <>
  void InCoreCholeskyRIERI<double>::computeAOInts(BasisSet &basisSet, Molecule &mol,
      EMPerturbation &emPert, OPERATOR op, const AOIntsOptions &options) {

    if (op != ELECTRON_REPULSION)
      CErr("Only Electron repulsion integrals in InCoreCholeskyRIERI<double>",std::cout);
    if (options.basisType != REAL_GTO)
      CErr("Only Real GTOs are allowed in InCoreCholeskyRIERI<double>",std::cout);

    auto topCholeskyRI = tick();

    CQMemManager &mem = this->memManager();
    size_t NB = this->NB;
    size_t NB2 = NB*NB;

    double *diag = mem.malloc<double>(NB2);

    // Determine the number of OpenMP threads
    int nthreads = GetNumThreads();

    // Create a vector of libint2::Engines for possible threading
    std::vector<libint2::Engine> engines(nthreads);

    // Initialize the first engine for the integral evaluation
    engines[0] = libint2::Engine(libint2::Operator::coulomb,
      basisSet.maxPrim, basisSet.maxL,0);
    engines[0].set_precision(0.);

    // Copy over the engines to other threads if need be
    for(size_t i = 1; i < nthreads; i++) engines[i] = engines[0];

#ifndef __IN_HOUSE_INT__
    std::cout<<"Using Libint "<<std::endl;
#else
    std::cout<<"Using In-house Integral Engine "<<std::endl;
#endif

    auto topDiag = tick();
#ifdef CHOLESKY_BUILD_4INDEX
    InCore4indexERI<double> eri4I(mem, NB);
    eri4I.computeAOInts(basisSet, mol, emPert, op, options);
    #pragma omp parallel for
    for (size_t i = 0; i < NB2; i++) {
      diag[i] = eri4I(i,i);
    }
#else
    #pragma omp parallel
    {
      int thread_id = GetThreadID();

      // Get threads result buffer
      const auto& buf_vec = engines[thread_id].results();

      for (size_t P(0), PQ(0); P < basisSet.nShell; P++) {
        for (size_t Q = P; Q < basisSet.nShell; Q++, PQ++) {
          // Round Robbin work distribution
#ifdef _OPENMP
          if( PQ % nthreads != thread_id ) continue;
#endif

          // Evaluate ERI for shell quartet
#ifndef __IN_HOUSE_INT__
          engines[thread_id].compute2<
            libint2::Operator::coulomb, libint2::BraKet::xx_xx, 0>(
                basisSet.shells[P],
                basisSet.shells[Q],
                basisSet.shells[P],
                basisSet.shells[Q]
          );
          const auto *buff =  buf_vec[0] ;
          if(buff == nullptr) continue;
#else
          libint2::ShellPair pair1_to_use, pair2_to_use;
          pair1_to_use.init( basisSet_.shells[P],basisSet_.shells[Q],-2000);
          pair2_to_use.init( basisSet_.shells[P],basisSet_.shells[Q],-2000);
          auto buff  = RealGTOIntEngine::BottomupHGP(pair1_to_use,pair2_to_use,
              basisSet_.shells[P],
              basisSet_.shells[Q],
              basisSet_.shells[P],
              basisSet_.shells[Q]
          );
#endif

          for (size_t p(0), pBegin(basisSet.mapSh2Bf[P]),
               pSize(basisSet.shells[P].size());
               p < pSize; p++) {
            for (size_t q(0), qBegin(basisSet.mapSh2Bf[Q]),
                 qSize(basisSet.shells[Q].size());
                 q < qSize; q++) {
              size_t pqpq = (p * qSize + q) * (1 + pSize * qSize);
              diag[(pBegin + p) + (qBegin + q) * NB] = buff[pqpq];
              diag[(qBegin + q) + (pBegin + p) * NB] = buff[pqpq];
            }
          }
        }; // Q
      }; // P
    }; // omp region
#endif

    auto durDiag = tock(topDiag);
    std::cout << "Cholesky-RI-Diagonal duration   = " << durDiag << " s " << std::endl;

    // Temporary cholesky factor
    std::vector<double*> L;

    pivots.clear();
    size_t NBRI = 0;

    auto topCholesky = tick();
    while (NBRI < NB2) {
      // Select the pivot
      size_t pivot = 0;
      double Dmax = diag[0];
      for (size_t P = 0; P < NB2; P++) {
          if (Dmax < diag[P]) {
              Dmax = diag[P];
              pivot = P;
          }
      }

      // Check to see if convergence reached
      if (Dmax < delta_) break;

      // If here, we're trying to add this row
      pivots.push_back(pivot);
      double L_QQ = sqrt(Dmax);

      // If here, we're really going to add this row
      L.push_back(mem.malloc<double>(NB2));

#ifdef CHOLESKY_BUILD_4INDEX
      #pragma omp parallel for
      for (size_t i = 0; i < NB2; i++) {
        L[NBRI][i] = eri4I(i,pivot);
      }
#else
      size_t r = pivot % NB;
      size_t s = pivot / NB;
      size_t R = std::distance(basisSet.mapSh2Bf.begin(),
          std::upper_bound(basisSet.mapSh2Bf.begin(),
              basisSet.mapSh2Bf.end(), r)) - 1;
      size_t S = std::distance(basisSet.mapSh2Bf.begin(),
          std::upper_bound(basisSet.mapSh2Bf.begin(),
              basisSet.mapSh2Bf.end(), s)) - 1;
      size_t rBegin = basisSet.mapSh2Bf[R];
      size_t sBegin = basisSet.mapSh2Bf[S];
      size_t sSize = basisSet.shells[S].size();

      #pragma omp parallel
      {
        int thread_id = GetThreadID();

        // Get threads result buffer
        const auto& buf_vec = engines[thread_id].results();

        for (size_t P(0), PQ(0); P < basisSet.nShell; P++) {
          for (size_t Q = P; Q < basisSet.nShell; Q++, PQ++) {
            // Round Robbin work distribution
#ifdef _OPENMP
            if( PQ % nthreads != thread_id ) continue;
#endif

            // Evaluate ERI for shell quartet
#ifndef __IN_HOUSE_INT__
            engines[thread_id].compute2<
              libint2::Operator::coulomb, libint2::BraKet::xx_xx, 0>(
                  basisSet.shells[R],
                  basisSet.shells[S],
                  basisSet.shells[P],
                  basisSet.shells[Q]
            );
            const auto *buff =  buf_vec[0] ;
            if(buff == nullptr) continue;
#else
            libint2::ShellPair pair1_to_use, pair2_to_use;
            pair1_to_use.init( basisSet_.shells[P],basisSet_.shells[Q],-2000);
            pair2_to_use.init( basisSet_.shells[R],basisSet_.shells[S],-2000);
            auto buff  = RealGTOIntEngine::BottomupHGP(pair1_to_use,pair2_to_use,
                basisSet_.shells[R],
                basisSet_.shells[S],
                basisSet_.shells[P],
                basisSet_.shells[Q]
            );
#endif

            for (size_t p(0), pBegin(basisSet.mapSh2Bf[P]),
                 pSize(basisSet.shells[P].size());
                 p < pSize; p++) {
              for (size_t q(0), qBegin(basisSet.mapSh2Bf[Q]),
                   qSize(basisSet.shells[Q].size());
                   q < qSize; q++) {
                size_t rspq = ((r-rBegin) * sSize + (s-sBegin)) * pSize * qSize + p * qSize + q;
                L[NBRI][(pBegin + p) + (qBegin + q) * NB] = buff[rspq];
                L[NBRI][(qBegin + q) + (pBegin + p) * NB] = buff[rspq];
              }
            }
          }; // Q
        }; // P
      }; // omp region
#endif

      // [(m|Q) - L_m^P L_Q^P]
      for (size_t P = 0; P < NBRI; P++) {
          AXPY(NB2, -L[P][pivots[NBRI]], L[P], 1, L[NBRI], 1);
      }

      // 1/L_QQ [(m|Q) - L_m^P L_Q^P]
      Scale(NB2, 1.0 / L_QQ, L[NBRI], 1);

      // Zero the upper triangle
      #pragma omp parallel for
      for (size_t P = 0; P < pivots.size(); P++) {
          L[NBRI][pivots[P]] = 0.0;
      }

      // Set the pivot factor
      L[NBRI][pivot] = L_QQ;

      // Update the Schur complement diagonal
      #pragma omp parallel for
      for (size_t P = 0; P < NB2; P++) {
          diag[P] -= L[NBRI][P] * L[NBRI][P];
      }

      // Force truly zero elements to zero
      #pragma omp parallel for
      for (size_t P = 0; P < pivots.size(); P++) {
          diag[pivots[P]] = 0.0;
      }

      NBRI++;
    }

    auto durCholesky = tock(topCholesky);
    std::cout << "Cholesky-RI-Cholesky duration   = " << durCholesky << " s " << std::endl;

    this->setNRIBasis(NBRI);
    std::cout << "Cholesky-RI auxiliary dimension = " << NBRI << std::endl;

    #pragma omp parallel for
    for (size_t Q = 0; Q < NBRI; Q++) {
      for (size_t j = 0, ij = 0; j < NB; j++)
      for (size_t i = 0; i < NB; i++, ij++)
        (*this)(Q,i,j) = L[Q][ij];
    }

    auto durCholeskyRI = tock(topCholeskyRI);
    std::cout << "Cholesky-RI duration   = " << durCholeskyRI << " s " << std::endl;

    mem.free(diag);
    for (double *p : L) {
      mem.free(p);
    }

  }; // InCoreCholeskyRIERI<double>::computeERI

}; // namespace ChronusQ

//#endif
