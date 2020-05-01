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

#include <cqlinalg/cqlinalg_config.hpp>

namespace ChronusQ {

  /**
   *  \brief Takes the inner product of two, possibly strided
   *  vectors
   *
   *  \f[
   *    a = \sum_i x_i^* y_i
   *  \f]
   *
   *  General to real, complex and mixed expressions.
   *
   *  \warning Real return involving complex vectors will
   *  return only the real part and discard the imaginary part.
   *
   *  Wraps BLAS functions. See
   *    http://www.netlib.org/lapack/lapack-3.1.1/html/ddot.f.html or
   *    http://www.netlib.org/lapack/lapack-3.1.1/html/zdotc.f.html for
   *  parameter documentation.
   */ 
  template <typename _F1, typename _F2, typename _F3>
  _F1 InnerProd(int N, _F2 *X, int INCX, _F3 *Y, int INCY);

  /**
   *  \brief Swaps the states of two vectors
   *
   *  Wraps BLAS functions. See
   *    http://www.netlib.org/lapack/lapack-3.1.1/html/dswap.f.html or
   *    http://www.netlib.org/lapack/lapack-3.1.1/html/zswap.f.html for
   *  parameter documentation.
   *    
   */
  template <typename _F1>
  void Swap(int N, _F1 *X, int INCX, _F1 *Y, int INCY);

  /**
   *  \brief Scales a vector in place
   *
   *  Wraps BLAS functions. See
   *    http://www.netlib.org/lapack/lapack-3.1.1/html/dscal.f.html or
   *    http://www.netlib.org/lapack/lapack-3.1.1/html/zscal.f.html for
   *  parameter documentation.
   *    
   */
  template <typename _F>
  void Scale(int N, _F ALPHA, _F* X, int INCX);

  /**
   *  \brief Returns the euclidian norm of a vector
   *
   *  Wraps BLAS functions. See
   *    http://www.netlib.org/lapack/lapack-3.1.1/html/dnrm2.f.html or
   *    http://www.netlib.org/lapack/lapack-3.1.1/html/dznrm2.f.html for
   *  parameter documentation.
   *    
   */
  template <typename _F1, typename _F2>
  _F1 TwoNorm(int N, _F2 *X, int INCX);

  
  /**
   *  \brief Returns constant time a vector plus a vector
   *
   *  Wraps BLAS functions. See
   *    http://www.netlib.org/lapack/lapack-3.1.1/html/daxpy.f.html 
   *    http://www.netlib.org/lapack/lapack-3.1.1/html/zaxpy.f.html 
   *  parameter documentation.
   *    
   */
  template <typename _F, typename _ShiftF>
  void AXPY(int N, _ShiftF alpha, _F *X, int INCX, _F *Y, int INCY);

  /**
   *  \brief Returns the specified norm of a matrix.
   *
   *  Wraps LAPACK functions. See
   *    http://www.netlib.org/lapack/lapack-3.1.1/html/dlange.f.html or
   *    http://www.netlib.org/lapack/lapack-3.1.1/html/zlange.f.html for
   *  parameter documentation.
   *
   */ 
  template <typename _F1, typename _F2>
  _F1 MatNorm(char NORM, int M, int N, _F2 *A, int LDA);
  


  /**
   *  \brief Normalizes a vector and returns the norm
   */
  template <typename _F>
  inline double Normalize(int N, _F *X, int INCX) {

    double norm = TwoNorm<double>(N,X,INCX);
    Scale(N,_F(1.)/norm,X,INCX);

    return norm;

  }
}; // namespace ChronusQ



