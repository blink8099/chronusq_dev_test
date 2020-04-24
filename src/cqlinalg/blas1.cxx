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
#include <cqlinalg/blas1.hpp>

#ifndef _CQ_MKL
typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
#endif

namespace ChronusQ {

  template<>
  double InnerProd(int N, double *X, int INCX, double *Y, int INCY) {
    
#ifdef _CQ_MKL
    return ddot(&N,X,&INCX,Y,&INCY);
#else
    assert(INCX > 0 and INCY > 0);
    double res(0.);
    #pragma omp parallel for reduction(+:res)
    for(int j = 0; j < N; j++){
      res += X[j*INCX] * Y[j*INCY];
    }
    return res;
#endif 
        
  }; // InnerProd real = (real,real)


  template<>
  void AXPY(int N, double alpha, double *X, int INCX, double *Y, int INCY) {
#ifdef _CQ_MKL
    daxpy
#else
    daxpy_
#endif 
      (&N,&alpha,X,&INCX,Y,&INCY);
  }; // AXPY (real,real)

  template<>
  void AXPY(int N, dcomplex alpha, dcomplex *X, int INCX, dcomplex *Y, int INCY) {
#ifdef _CQ_MKL
    zaxpy(&N,&alpha,X,&INCX,Y,&INCY);
#else
    zaxpy_(&N,reinterpret_cast<double*>(&alpha),reinterpret_cast<double*>(X),
      &INCX,reinterpret_cast<double*>(Y),&INCY);
#endif 
      
  }; // AXPY (complex,complex)


  template<>
  void AXPY(int N, double alpha, dcomplex *X, int INCX, dcomplex *Y, int INCY) {
    AXPY(N,dcomplex(alpha),X,INCX,Y,INCY);
  }; // AXPY (complex,real)


  template<>
  dcomplex InnerProd(int N, double *X, int INCX, double *Y, int INCY) {
    return dcomplex(InnerProd<double>(N,X,INCX,Y,INCY));
  }; // InnerProd complex = (real,real)




  /**
   *  \warning Discards imaginary part of inner product
   */ 
  template<>
  double InnerProd(int N, dcomplex *X, int INCX, double *Y, int INCY) {
    INCX *= 2;
    return
#ifdef _CQ_MKL
      ddot
#else
      ddot_
#endif 
        (&N,reinterpret_cast<double*>(X),&INCX,Y,&INCY);
  }; // InnerProd real = (complex,real)




  template<>
  dcomplex InnerProd(int N, dcomplex *X, int INCX, double *Y, int INCY) {
    INCX *= 2;
    double re =
#ifdef _CQ_MKL
      ddot
#else
      ddot_
#endif 
        (&N,reinterpret_cast<double*>(X),&INCX,Y,&INCY);

    double im = -
#ifdef _CQ_MKL
      ddot
#else
      ddot_
#endif 
        (&N,reinterpret_cast<double*>(X) + 1,&INCX,Y,&INCY);

    return dcomplex(re,im);
  }; // InnerProd complex = (complex,real)


//SS start
/*
  template<>
  dcomplex InnerProd(int N, double *X, int INCX, dcomplex *Y, int INCY) {
    INCX *= 2;
    double re =
#ifdef _CQ_MKL
      ddot
#else
      ddot_
#endif 
        (&N,X,&INCX,reinterpret_cast<double*>(Y),&INCY);

    double im = -
#ifdef _CQ_MKL
      ddot
#else
      ddot_
#endif 
        (&N,X,&INCX,reinterpret_cast<double*>(Y) + 1,&INCY);

    return dcomplex(re,im);
  }; // InnerProd complex = (real,complex)
*/
//SS end


  template<>
  dcomplex InnerProd(int N, dcomplex *X, int INCX, dcomplex *Y, int INCY) {
#ifdef _CQ_MKL
      dcomplex res;
      zdotc(&res,&N,X,&INCX,Y,&INCY);
      return res;
#else
      auto res = zdotc_(&N,reinterpret_cast<double*>(X),&INCX,
                           reinterpret_cast<double*>(Y),&INCY);
      return *reinterpret_cast<dcomplex*>(&res);
#endif
  }; // InnerProd complex = (complex,complex)




  /**
   *  \warning Discards imaginary part of inner product
   */ 
  template<>
  double InnerProd(int N, dcomplex *X, int INCX, dcomplex *Y, int INCY) {
#ifdef _CQ_MKL
      return std::real(InnerProd<dcomplex>(N,X,INCX,Y,INCY));
#else
      auto res = zdotc_(&N,reinterpret_cast<double*>(X),&INCX,
        reinterpret_cast<double*>(Y),&INCY);
      return std::real(*reinterpret_cast<dcomplex*>(&res));
#endif
  }; // InnerProd real = (complex,complex)


  template <>
  void Swap(int N, double *X, int INCX, double *Y, int INCY) {
#ifdef _CQ_MKL
    dswap
#else
    dswap_
#endif
      (&N,X,&INCX,Y,&INCY);

  }; // Swap (double)

  template <>
  void Swap(int N, dcomplex *X, int INCX, dcomplex *Y, int INCY) {
#ifdef _CQ_MKL
    zswap(&N,X,&INCX,Y,&INCY);
#else
    zswap_(&N,reinterpret_cast<double*>(X),&INCX,
      reinterpret_cast<double*>(Y),&INCY);
#endif

  }; // Swap (complex)




  template<>
  double TwoNorm(int N, double *X, int INCX) {
    return
#ifdef _CQ_MKL
      dnrm2
#else
      dnrm2_
#endif
      (&N,X,&INCX);
  }; // TwoNorm double = (double)

  template<>
  double TwoNorm(int N, dcomplex *X, int INCX) {
    return
#ifdef _CQ_MKL
      dznrm2(&N,X,&INCX);
#else
      dznrm2_(&N,reinterpret_cast<double*>(X),&INCX);
#endif
      
  }; // TwoNorm double = (dcomplex)





  template<>
  double MatNorm(char NORM, int M, int N, double *A, int LDA) {

    return LAPACKE_dlange(CblasColMajor,NORM,M,N,A,LDA);

  }; // MatNorm (double)



  template<>
  double MatNorm(char NORM, int M, int N, dcomplex *A, int LDA) {

    return LAPACKE_zlange(CblasColMajor,NORM,M,N,A,LDA);

  }; // MatNorm (complex)




  template<>
  void Scale(int N, double ALPHA, double *X, int INCX) {

#ifdef _CQ_MKL
      dscal
#else
      dscal_
#endif
      (&N,&ALPHA,X,&INCX);

  }; // Scale (double)

  template<>
  void Scale(int N, dcomplex ALPHA, dcomplex *X, int INCX) {

#ifdef _CQ_MKL
      zscal(&N,&ALPHA,X,&INCX);
#else
      zscal_(&N,reinterpret_cast<double*>(&ALPHA),reinterpret_cast<double*>(X),
        &INCX);
#endif
      

  }; // Scale (dcomplex)

};
