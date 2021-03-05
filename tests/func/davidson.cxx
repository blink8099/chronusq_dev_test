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

#include <func.hpp>
#include <cerr.hpp>
#include <memmanager.hpp>
#include <itersolver.hpp>
#include <util/files.hpp>
#include <util/timer.hpp>

#include <util/matout.hpp>
#include <cqlinalg/factorization.hpp>
#include <cqlinalg/blas3.hpp>
#include <cqlinalg/eig.hpp>

using namespace ChronusQ;


template <typename ReadT, typename EigT>
void DAVIDSON_TEST(size_t nRoots, size_t m, size_t kG,
  std::string fname, bool doPre = false, double conver = 1e-10, 
  double etol = 8e-8, size_t block_size = 128) {
  
  MPI_Barrier(MPI_COMM_WORLD);

  std::string refName( FUNC_REFERENCE + fname );

  bool isRoot = MPIRank(MPI_COMM_WORLD) == 0;
  bool isMPI  = MPISize(MPI_COMM_WORLD) > 1;

  std::cout << "DAVIDSON_TEST with file:" << refName << std::endl;
  
  if( isRoot ) std::cout << "  * Will use " << MPISize(MPI_COMM_WORLD) << 
    " MPI Processes" << std::endl;

  SafeFile matFile(refName,true);

  CQMemManager mem(2e9,256);
  
  auto dims = matFile.getDims("/matrix");
  // Dummy checks on
  
  EXPECT_TRUE( dims.size() == 2   );
  EXPECT_TRUE( dims[0] == dims[1] );

  size_t N = dims[0];

  CB_INT MLoc = N, NLoc = N;

#ifdef CQ_ENABLE_MPI
  std::shared_ptr<CXXBLACS::BlacsGrid> grid = nullptr;
  CXXBLACS::ScaLAPACK_Desc_t descA;

  if( isMPI ) {
    grid = std::make_shared<CXXBLACS::BlacsGrid>(MPI_COMM_WORLD,
        block_size, block_size);

    std::tie(MLoc,NLoc) = grid->getLocalDims(N,N);
    descA = grid->descInit(N,N,0,0,MLoc);
  }
#endif

  // Allocate and read the matrix
  ReadT * A = (MLoc and NLoc) ? mem.malloc<ReadT>(MLoc * NLoc) : nullptr;

  ReadT *AREAD = nullptr;
  ReadT *DIAG  = nullptr;
  if( isRoot ) {

    AREAD = isMPI ? mem.malloc<ReadT>(N*N) : A;
    matFile.readData("/matrix",AREAD); 

    DIAG  = mem.malloc<ReadT>(N);
    for( size_t k = 0; k < N; k++ ) DIAG[k] = AREAD[k*(N+1)];

  }

  if( isRoot ) std::cout << "  * Load Data Successfully " << std::endl;

#ifdef CQ_ENABLE_MPI
  if( isMPI ) grid->Scatter(N,N,AREAD,N,A,MLoc,0,0);
#endif


  EigT *ALOC = reinterpret_cast<EigT*>(A);

#ifdef CQ_ENABLE_MPI
  if( isMPI ) {

    if( AREAD ) mem.free(AREAD);


    if( not std::is_same<EigT,ReadT>::value ) {
      ALOC = mem.malloc<EigT>(MLoc * NLoc);
      std::copy_n(A,MLoc*NLoc,ALOC);
      mem.free(A);
    }

  }
#endif


  typename Davidson<EigT>::LinearTrans_t func = [&]( size_t nVec, EigT *V, 
      EigT *AV) {

#ifdef CQ_ENABLE_MPI
    if( isMPI ) {

      CB_INT MLoc_V = N, NLoc_V = nVec;
      std::tie(MLoc_V, NLoc_V) = grid->getLocalDims(N,nVec);
      auto descV = grid->descInit(N,nVec,0,0,MLoc_V);

      bool alloc = MLoc_V and NLoc_V;


      EigT *VLOC = nullptr, *AVLOC = nullptr;

      if( alloc ) {

        VLOC  = mem.malloc<EigT>(MLoc_V * NLoc_V);
        AVLOC = mem.malloc<EigT>(MLoc_V * NLoc_V);

      }

      grid->Scatter(N,nVec,V,N,VLOC,MLoc_V,0,0);

      Gemm('N','N',N,nVec,N,EigT(1.),ALOC,1,1,descA,VLOC,1,1,descV,
          EigT(0.),AVLOC,1,1,descV);

      grid->Gather(N,nVec,AV,N,AVLOC,MLoc_V,0,0);

    } else 
#endif
      Gemm('N','N',N,nVec,N,EigT(1.),A,N,V,N,EigT(0.),AV,N);

  };

  typename Davidson<EigT>::LinearTrans_t PC = [&]( size_t nVec, EigT *V, 
      EigT *AV) {

    if( V != AV ) std::copy_n(V,nVec*N,AV);

    if( doPre )
    // Scale by inverse diagonals
    for( auto k = 0ul; k < N; k++ ) 
      Scale(nVec, EigT(1.) / DIAG[k], AV + k, N);
  };

#ifndef _CQ_GENERATE_TESTS

  size_t nThreads = omp_get_num_threads();
  ProgramTimer::initialize("Davidson test", nThreads);
  Davidson<EigT> davidson(MPI_COMM_WORLD,mem,N,5,128,conver,nRoots,
    func,PC);

  davidson.setM(m);
  davidson.setkG(kG);
  
  davidson.run();

  ROOT_ONLY(MPI_COMM_WORLD);

  dcomplex *refW = mem.malloc<dcomplex>(N);
  matFile.readData("/W",refW);

  dcomplex *W = davidson.eigVal();

  for(auto k = 0ul; k < nRoots; k++) {
    double diff1 = std::abs((W[k] - refW[k])/refW[k]);
    double diff2 = std::abs((W[k] - std::conj(refW[k]))/refW[k]);
    EXPECT_TRUE( diff1 < etol or diff2 < etol) <<
      "DIFF1 = " << diff1 << ", DIFF2 = " << diff2;
  }

#else

  dcomplex *ACMPLX = mem.malloc<dcomplex>(N*N);
  std::copy_n(A,N*N,ACMPLX);

  dcomplex *W = mem.malloc<dcomplex>(N);
  dcomplex *VR = mem.malloc<dcomplex>(N*N);
  dcomplex *VL = mem.malloc<dcomplex>(N*N);

  GeneralEigen('V','V',N,ACMPLX,N,W,VL,N,VR,N,mem);

  matFile.safeWriteData("/W",W,{N});

#endif

}

// 
// Hermittian Davidson
//
TEST(DAVIDSON, DAVIDSON_REAL_HERMITIAN) {
  
  DAVIDSON_TEST<double,double>(3,50,3,"real_Hermitian.hdf5");
  DAVIDSON_TEST<double,double>(3,50,3,"real_Hermitian.hdf5",true);

}

TEST(DAVIDSON, DAVIDSON_COMPLEX_HERMITIAN) {
  
  DAVIDSON_TEST<dcomplex,dcomplex>(3,50,3,"complex_Hermitian.hdf5");
  DAVIDSON_TEST<dcomplex,dcomplex>(3,50,3,"complex_Hermitian.hdf5",true);

}

// 
// Non-Hermittian
//
TEST(DAVIDSON, DAVIDSON_REAL_NONHERMITIAN) {
  
  DAVIDSON_TEST<double,double>(3,50,3,"real_nonHermitian.hdf5");
  DAVIDSON_TEST<double,double>(3,50,3,"real_nonHermitian.hdf5",true);

}

TEST(DAVIDSON, DAVIDSON_COMPLEX_NONHERMITIAN) {
  
  DAVIDSON_TEST<dcomplex,dcomplex>(3,50,3,"complex_nonHermitian.hdf5");
  DAVIDSON_TEST<dcomplex,dcomplex>(3,50,3,"complex_nonHermitian.hdf5",true);

}



