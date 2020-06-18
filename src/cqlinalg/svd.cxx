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

#include <cqlinalg/svd.hpp>
#include <cqlinalg/util.hpp>
#include <cerr.hpp>
#include <limits>

namespace ChronusQ {

  template <>
  int SVD(char JOBU, char JOBVT, int M, int N, double *A, int LDA, double *S,
    double *U, int LDU, double *VT, int LDVT, CQMemManager &mem) {

    int INFO;
  
    auto test = std::bind(dgesvd_,&JOBU,&JOBVT,&M,&N,A,&LDA,S,U,&LDU,VT,&LDVT,
      std::placeholders::_1,std::placeholders::_2,&INFO);
  
    int LWORK = getLWork<double>(test);
    double *WORK = mem.malloc<double>(LWORK);
    
    dgesvd_(&JOBU,&JOBVT,&M,&N,A,&LDA,S,U,&LDU,VT,&LDVT,WORK,&LWORK,&INFO);

    mem.free(WORK);

    return INFO;

  }; // SVD (double)

  template <>
  int SVD(char JOBU, char JOBVT, int M, int N, dcomplex *A, int LDA, double *S,
    dcomplex *U, int LDU, dcomplex *VT, int LDVT, CQMemManager &mem) {

    int INFO;
  
    int LRWORK = 5*std::min(M,N);
    double   *RWORK = mem.malloc<double>(LRWORK);

    auto test = std::bind(zgesvd_,&JOBU,&JOBVT,&M,&N,A,&LDA,S,U,&LDU,VT,&LDVT,
      std::placeholders::_1,std::placeholders::_2,RWORK,&INFO);
  
    int LWORK  = getLWork<dcomplex>(test);
    dcomplex *WORK  = mem.malloc<dcomplex>(LWORK);

    
    zgesvd_(&JOBU,&JOBVT,&M,&N,A,&LDA,S,U,&LDU,VT,&LDVT,WORK,&LWORK,RWORK,
      &INFO);

    mem.free(WORK,RWORK);

    return INFO;

  }; // SVD (dcomplex)

  template <typename T>
  struct real_type {
    using type = T;
  };

  template <typename T>
  struct real_type<std::complex<T>> {
    using type = T;
  };

  template <typename _F>
  size_t ORTH(int M, int N, _F *A, int LDA, double *S,
    _F *U, int LDU, CQMemManager &mem) {

    if (not U) U = A;

    int INFO = SVD(A == U ? 'O' : 'S', 'N', M, N, A, LDA, S, U, LDU,
      reinterpret_cast<_F*>(NULL),LDA, mem);

    if (INFO < 0)
      CErr(std::to_string(-INFO) + "-th argument had an illegal value.");
    else if (INFO > 0)
      CErr("SVD did not converge.");

    double tol = std::max(M,N) * S[0]
      * std::numeric_limits<typename real_type<_F>::type>::epsilon();

    return std::distance(S, std::find_if(S,S+std::min(M,N),
      [tol](double x){ return x < tol; }));

  }; // ORTH

  template size_t ORTH(int, int, double*, int, double*, double*, int, CQMemManager&);
  template size_t ORTH(int, int, dcomplex*, int, double*, dcomplex*, int, CQMemManager&);

}; // namespace ChronusQ

