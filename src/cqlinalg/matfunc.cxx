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
#include <cqlinalg/matfunc.hpp>
#include <cqlinalg/blas3.hpp>
#include <cqlinalg/eig.hpp>

namespace ChronusQ {


  template <typename F, typename _F1, typename _F2>
  void MatDiagFunc(const F &func, size_t N, _F1 *A, size_t LDA, _F2 *B, 
    size_t LDB, CQMemManager &mem) {

    // Allocate space for eigenvalues and scratch
    double *W = mem.malloc<double>(N);
    _F1* SCR  = mem.malloc<_F1>(N*N);
    _F2* SCR2  = mem.malloc<_F1>(N*N);

    std::copy_n(A,N*N,SCR); // Copy A to SCR

    // A = V * a * V**H
    HermetianEigen('V','U',N,SCR,N,W,mem);

    // Compute X = V * func(a)
    for(auto j = 0; j < N; j++)
    for(auto i = 0; i < N; i++)
      SCR2[i + j*N] = SCR[i + j*N] * func(W[j]);
    
    // Compute B**H = V * X**H
    Gemm('N','C',N,N,N,_F2(1.),SCR,N,SCR2,N,_F2(0.),B,LDB);

    // FIXME: Use MKL for this transpose when direct is merged in
    Eigen::Map<Eigen::Matrix<_F2,Eigen::Dynamic,Eigen::Dynamic,
      Eigen::ColMajor>> BMap(B,LDB,N);

    BMap.adjointInPlace(); 

    mem.free(SCR,SCR2,W);

  };



  template <typename _FExp, typename _F1, typename _F2>
  void MatExp(char ALG, size_t N, _FExp ALPHA, _F1 *A, size_t LDA, 
    _F2 *ExpA, size_t LDEXPA, CQMemManager &mem) {

    assert(ALG == 'D');
    assert(std::real(ALPHA) < 1e-14);

    double AIM = std::is_same<_FExp,dcomplex>::value ? std::imag(ALPHA) : 0.;

    MatDiagFunc([&](double x) -> _F2 { 
        return dcomplex(std::cos(AIM*x),std::sin(AIM*x)); 
      }, N,A,LDA,ExpA,LDEXPA,mem);

  };


//template void MatExp(char,size_t,double,double*,size_t,double*,size_t,
//  CQMemManager&);

  template void MatExp(char,size_t,dcomplex,dcomplex*,size_t,dcomplex*,size_t,
    CQMemManager&);

}; // namespace ChronusQ
