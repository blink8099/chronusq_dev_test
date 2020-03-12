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
#ifndef __INCLUDED_CQLINALG_CONFIG_HPP__
#define __INCLUDED_CQLINALG_CONFIG_HPP__

#include <chronusq_sys.hpp>

// Choose linear algebra headers
#ifdef _CQ_MKL
  #define MKL_Complex16 dcomplex // Redefine MKL complex type
  #define MKL_Complex8  std::complex<float> // Redefine MKL complex type
  #include <mkl.h> // MKL

  #ifdef CQ_ENABLE_MPI
    #include <mkl_blacs.h>  
    #include <mkl_scalapack.h>  
    #include <mkl_pblas.h>

    #define CXXBLACS_BLACS_Complex16 double
    #define CXXBLACS_BLACS_Complex8  float
    
    #define CXXBLACS_HAS_BLACS
    #define CXXBLACS_HAS_PBLAS
    #define CXXBLACS_HAS_SCALAPACK
  #endif
#else

  #ifdef CQ_ENABLE_MPI
//  #error CXXBLAS + nonMKL Not Tested!
  #endif

  #define CXXBLACS_HAS_BLAS
  #define CXXBLACS_BLAS_Complex16 double
  #define CXXBLACS_BLAS_Complex8  float
  //#define CXXBLACS_LAPACK_Complex16 double
  //#define CXXBLACS_LAPACK_Complex8  float

  // Redefine OpenBLAS complex type
  #define lapack_complex_float std::complex<float> 
  #define lapack_complex_double dcomplex 

  #include <f77blas.h>
  #include <lapacke.h> // OpenBLAS

  extern "C" {
    int openblas_get_num_threads();
  }

#endif

#ifdef CQ_ENABLE_MPI
  #define CXXBLACS_HAS_LAPACK
  #include <cxxblacs.hpp>
#else
  #define CB_INT size_t
#endif

#include <memmanager.hpp>
#include <Eigen/Core>

#endif
