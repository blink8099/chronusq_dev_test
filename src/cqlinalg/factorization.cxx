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
#include <cqlinalg/factorization.hpp>
#include <cqlinalg/util.hpp>

#include <cerr.hpp>
#include <util/matout.hpp>

namespace ChronusQ {

  // Cholesky specializations

  // Real wraps DPOTRF
  template <>
  int Cholesky(char UPLO, int N, double *A, int LDA) {
    int INFO;
    dpotrf_(&UPLO,&N,A,&LDA,&INFO);
    return INFO;
  }; // Cholesky (real)

  // Complex wraps ZPOTRF
  template <>
  int Cholesky(char UPLO, int N, dcomplex *A, int LDA) {
    int INFO;
    zpotrf_(&UPLO,&N,A,&LDA,&INFO);
    return INFO;
  }; // Cholesky (complex)


  // Real wraps DPOTRI
  template <>
  int CholeskyInv(char UPLO, int N, double *A, int LDA) {
    int INFO;
    dpotri_(&UPLO,&N,A,&LDA,&INFO);
    return INFO;
  }; // CholeskyInv (real)


  // Complex wraps ZPOTRI
  template <>
  int CholeskyInv(char UPLO, int N, dcomplex *A, int LDA) {
    int INFO;
    zpotri_(&UPLO,&N,A,&LDA,&INFO);
    return INFO;
  }; // CholeskyInv (complex)














  // Real wraps DGETRF
  template<>
  int LU(int M, int N, double *A, int LDA, int *IPIV) {

    int INFO;
    
    dgetrf_(&M,&N,A,&LDA,IPIV,&INFO);

    return INFO;

  }; // LU (real)


  // Complex wraps ZGETRF
  template<>
  int LU(int M, int N, dcomplex *A, int LDA, int *IPIV) {

    int INFO;
    
    zgetrf_(&M,&N,A,&LDA,IPIV,&INFO);

    return INFO;

  }; // LU (complex)


  // Real wraps DGETRF + DGETRI
  template <>
  int LUInv(int N, double *A, int LDA, CQMemManager &mem) {

    int INFO;

    int *IPIV = mem.malloc<int>(N);

    auto test =
      std::bind(dgetri_,&N,A,&LDA,IPIV,std::placeholders::_1,
        std::placeholders::_2,&INFO);


    INFO = LU(N,N,A,LDA,IPIV);

    if( INFO != 0 ) { mem.free(IPIV); return INFO; }

    int LWORK = getLWork<double>(test);
    double *WORK = mem.malloc<double>(LWORK);

    dgetri_(&N,A,&LDA,IPIV,WORK,&LWORK,&INFO);

    mem.free(IPIV,WORK);

    return INFO;

  }; // LUInv (real)


 
  // Complex wraps ZGETRF + ZGETRI
  template <>
  int LUInv(int N, dcomplex *A, int LDA, CQMemManager &mem) {

    int INFO;

    int *IPIV = mem.malloc<int>(N);

    auto test =
      std::bind(zgetri_,&N,A,&LDA,IPIV,std::placeholders::_1,
        std::placeholders::_2,&INFO);


    INFO = LU(N,N,A,LDA,IPIV);

    if( INFO != 0 ) { mem.free(IPIV); return INFO; }

    int LWORK = getLWork<dcomplex>(test);
    dcomplex *WORK = mem.malloc<dcomplex>(LWORK);

    zgetri_(&N,A,&LDA,IPIV,WORK,&LWORK,&INFO);

    mem.free(IPIV,WORK);

    return INFO;

  }; // LUInv (complex)







  template <>
  int BunchKaufman(char UPLO, int N, double *A, int LDA, int *IPIV, 
      CQMemManager &mem) {

    int INFO;
    auto test = 
      std::bind(dsytrf_,&UPLO,&N,A,&LDA,IPIV,std::placeholders::_1,
        std::placeholders::_2,&INFO);

    int LWORK = getLWork<double>(test);
    double *WORK = mem.malloc<double>(LWORK);

    dsytrf_(&UPLO,&N,A,&LDA,IPIV,WORK,&LWORK,&INFO);

    mem.free(WORK);

    return INFO;

  }

  template <>
  int BunchKaufman(char UPLO, int N, dcomplex *A, int LDA, int *IPIV,
      CQMemManager &mem) {

    int INFO;
    auto test = 
      std::bind(zhetrf_,&UPLO,&N,A,&LDA,IPIV,std::placeholders::_1,
        std::placeholders::_2,&INFO);

    int LWORK = getLWork<dcomplex>(test);
    dcomplex *WORK = mem.malloc<dcomplex>(LWORK);

    zhetrf_(&UPLO,&N,A,&LDA,IPIV,WORK,&LWORK,&INFO);

    mem.free(WORK);

    return INFO;

  }






  template<>
  int TriInv(char UPLO, char DIAG, int N, double *A, int LDA){
    
    int INFO;
    dtrtri_(&UPLO,&DIAG,&N,A,&LDA,&INFO);
    return INFO;

  };

  template<>
  int TriInv(char UPLO, char DIAG, int N, dcomplex *A, int LDA){
    
    int INFO;
    ztrtri_(&UPLO,&DIAG,&N,A,&LDA,&INFO);
    return INFO;

  };







  template <>
  int QR(int M, int N, double *A, int LDA, double *R, int LDR, 
    CQMemManager &mem) {

    int INFO;
    double *TAU = mem.malloc<double>(N);

    using namespace std::placeholders;
    auto qrf = std::bind(dgeqrf_,&M,&N,A,&LDA,TAU,_1,_2,&INFO);
    auto qrg = std::bind(dorgqr_,&M,&N,&N,A,&LDA,TAU,_1,_2,&INFO);

    int LWORK_QRF = getLWork<double>(qrf);
    int LWORK_QRG = getLWork<double>(qrg);

    int LWORK = std::max(LWORK_QRF,LWORK_QRG);
    double *WORK = mem.malloc<double>(LWORK);


    qrf(WORK,&LWORK);

    if( INFO != 0 ) { mem.free(TAU,WORK); return INFO; }

    int rCol = std::min(M,N);
    std::fill_n(R,rCol*LDR,0.);

    for(auto j = 0; j < N; j++) 
    for(auto i = 0; i <= j; i++)
      R[i + j*LDR] = A[i + j*LDA];


    qrg(WORK,&LWORK);

    mem.free(TAU,WORK); return INFO; 

  }


  template <>
  int QR(int M, int N, dcomplex *A, int LDA, dcomplex *R, int LDR, 
    CQMemManager &mem) {

    int INFO;
    dcomplex *TAU = mem.malloc<dcomplex>(N);

    using namespace std::placeholders;
    auto qrf = std::bind(zgeqrf_,&M,&N,A,&LDA,TAU,_1,_2,&INFO);
    auto qrg = std::bind(zungqr_,&M,&N,&N,A,&LDA,TAU,_1,_2,&INFO);

    int LWORK_QRF = getLWork<dcomplex>(qrf);
    int LWORK_QRG = getLWork<dcomplex>(qrg);

    int LWORK = std::max(LWORK_QRF,LWORK_QRG);
    dcomplex *WORK = mem.malloc<dcomplex>(LWORK);


    qrf(WORK,&LWORK);

    if( INFO != 0 ) { mem.free(TAU,WORK); return INFO; }

    int rCol = std::min(M,N);
    std::fill_n(R,rCol*LDR,0.);

    for(auto j = 0; j < N; j++) 
    for(auto i = 0; i <= j; i++)
      R[i + j*LDR] = A[i + j*LDA];


    qrg(WORK,&LWORK);

    mem.free(TAU,WORK); return INFO; 

  }
















  template <>
  int QZ(char JOBVSL, char JOBVSR, int N, double *A, int LDA, double *B, 
    int LDB, dcomplex *ALPHA, double *BETA, double *VSL, int LDVSL, 
    double *VSR, int LDVSR, CQMemManager &mem) {


    using SELECT_3 = int (*)(const double*, const double*, const double*);
    SELECT_3 SF = nullptr;
    int  SDIM = 0;
    char SORT_c   = 'N';


    int INFO;
    double *ALPHAR = mem.malloc<double>(N);
    double *ALPHAI = mem.malloc<double>(N);

    using namespace std::placeholders;
    auto gges = std::bind(dgges_,&JOBVSL,&JOBVSR,&SORT_c,SF,&N,A,&LDA,B,&LDB,
      &SDIM,ALPHAR,ALPHAI,BETA,VSL,&LDVSL,VSR,&LDVSR,_1,_2,&SDIM,&INFO);

    int LWORK = getLWork<double>(gges);

    double *WORK = mem.malloc<double>(LWORK);

    gges(WORK,&LWORK);

    for(auto k = 0; k < N; k++) ALPHA[k] = dcomplex(ALPHAR[k],ALPHAI[k]);

    mem.free(ALPHAR,ALPHAI,WORK);

    return INFO;

  } 

  template <>
  int QZ(char JOBVSL, char JOBVSR, int N, dcomplex *A, int LDA, dcomplex *B, 
    int LDB, dcomplex *ALPHA, dcomplex *BETA, dcomplex *VSL, int LDVSL, 
    dcomplex *VSR, int LDVSR, CQMemManager &mem) {


    using SELECT_2 = int (*)(const dcomplex*, const dcomplex*);
    SELECT_2 SF = nullptr;
    int SDIM = 0;
    char SORT_c   = 'N';


    int INFO;
    double *RWORK = mem.malloc<double>(8*N);

    using namespace std::placeholders;
    auto gges = std::bind(zgges_,&JOBVSL,&JOBVSR,&SORT_c,SF,&N,A,&LDA,B,&LDB,
      &SDIM,ALPHA,BETA,VSL,&LDVSL,VSR,&LDVSR,_1,_2,RWORK,&SDIM,&INFO);

    int LWORK = getLWork<dcomplex>(gges);

    dcomplex *WORK = mem.malloc<dcomplex>(LWORK);

    gges(WORK,&LWORK);

    mem.free(RWORK,WORK);

    return INFO;

  } 





  template <typename _F>
  int TGEXC(bool WANTQ, bool WANTZ, int N, _F *A, int LDA, _F *B, int LDB,
    _F *Q, int LDQ, _F *Z, int LDZ, int IFST, int ILST, CQMemManager &mem);

  template<>
  int TGEXC(bool WANTQ, bool WANTZ, int N, double *A, int LDA, double *B, 
    int LDB, double *Q, int LDQ, double *Z, int LDZ, int IFST, int ILST,
    CQMemManager &mem) {

    int WANTQ_i = int(WANTQ);
    int WANTZ_i = int(WANTZ);

    int INFO;

    using namespace std::placeholders;
    auto gexc = std::bind(dtgexc_,&WANTQ_i,&WANTZ_i,&N,A,&LDA,B,&LDA,
      Q,&LDQ,Z,&LDZ,&IFST,&ILST,_1,_2,&INFO);

    int LWORK = getLWork<double>(gexc);
    double *WORK = mem.malloc<double>(LWORK);

    gexc(WORK,&LWORK);

    mem.free(WORK);

    return INFO;
  }

  template<>
  int TGEXC(bool WANTQ, bool WANTZ, int N, dcomplex *A, int LDA, dcomplex *B, 
    int LDB, dcomplex *Q, int LDQ, dcomplex *Z, int LDZ, int IFST, int ILST,
    CQMemManager &mem) {

    int WANTQ_i = int(WANTQ);
    int WANTZ_i = int(WANTZ);

    int INFO;

    ztgexc_(&WANTQ_i,&WANTZ_i,&N,A,&LDA,B,&LDA,Q,&LDQ,Z,&LDZ,
      &IFST,&ILST,&INFO);

    return INFO;
  }




  struct TGSEN_OUT {

    int INFO;
    int M;

    double PL;
    double PR;
    double DIF;

  };


  template <typename _F>
  TGSEN_OUT TGSEN(int IJOB, bool WANTQ, bool WANTZ, int * SELECT, int N, _F *A, 
    int LDA, _F *B, int LDB, dcomplex *ALPHA, _F *BETA, _F *Q, int LDQ, _F *Z, 
    int LDZ, CQMemManager &mem);


  template <>
  TGSEN_OUT TGSEN(int IJOB, bool WANTQ, bool WANTZ, int * SELECT, int N, 
    double *A, int LDA, double *B, int LDB, dcomplex *ALPHA, double *BETA, 
    double *Q, int LDQ, double *Z, int LDZ, CQMemManager &mem) {

    if( IJOB != 0 ) CErr("No proper logic for TGSEN.IJOB != 0");

    int WANTQ_i = int(WANTQ);
    int WANTZ_i = int(WANTZ);

    int INFO;

    int LIWORK = 1;
    int IWORK  = 0;

    TGSEN_OUT out;

    double *ALPHAR = mem.malloc<double>(N); 
    double *ALPHAI = mem.malloc<double>(N); 

    using namespace std::placeholders;
    auto gsen = std::bind(dtgsen_,&IJOB,&WANTQ_i,&WANTZ_i,SELECT,&N,A,&LDA,
      B,&LDB,ALPHAR,ALPHAI,BETA,Q,&LDQ,Z,&LDZ,&out.M,&out.PL,&out.PR,
      &out.DIF,_1,_2,&IWORK,&LIWORK,&INFO);

    int LWORK = getLWork<double>(gsen);
    double *WORK = mem.malloc<double>(LWORK);

    gsen(WORK,&LWORK);


    for(auto k = 0; k < N; k++)
      ALPHA[k] = dcomplex(ALPHAR[k],ALPHAI[k]);

    mem.free(ALPHAR,ALPHAI,WORK);

    return out;
  }



  template <>
  TGSEN_OUT TGSEN(int IJOB, bool WANTQ, bool WANTZ, int * SELECT, int N, 
    dcomplex *A, int LDA, dcomplex *B, int LDB, dcomplex *ALPHA, 
    dcomplex *BETA, dcomplex *Q, int LDQ, dcomplex *Z, int LDZ, 
    CQMemManager &mem) {

    if( IJOB != 0 ) CErr("No proper logic for TGSEN.IJOB != 0");

    int WANTQ_i = int(WANTQ);
    int WANTZ_i = int(WANTZ);

    int INFO;

    int LIWORK = 1;
    int IWORK  = 0;

    TGSEN_OUT out;

    using namespace std::placeholders;
    auto gsen = std::bind(ztgsen_,&IJOB,&WANTQ_i,&WANTZ_i,SELECT,&N,A,&LDA,
      B,&LDB,ALPHA,BETA,Q,&LDQ,Z,&LDZ,&out.M,&out.PL,&out.PR,
      &out.DIF,_1,_2,&IWORK,&LIWORK,&INFO);

    int LWORK = getLWork<dcomplex>(gsen);
    dcomplex *WORK = mem.malloc<dcomplex>(LWORK);

    gsen(WORK,&LWORK);

    mem.free(WORK);

    return out;
  }

  template <typename _F>
  int OrdQZ2(char JOBVSL, char JOBVSR, int N, _F *A, int LDA, _F *B, 
    int LDB, dcomplex *ALPHA, _F *BETA, double hLim, double SIGMA, _F *VSL, int LDVSL, 
    _F *VSR, int LDVSR, CQMemManager &mem) {

    // Aux booleans for reordering functions
    bool WantQ = JOBVSL == 'V';
    bool WantZ = JOBVSR == 'V';

    double LARGE = 1.0e10;

    // Get the initial QZ factorization
    int INFO = 
      QZ(JOBVSL,JOBVSR,N,A,LDA,B,LDB,ALPHA,BETA,VSL,LDVSL,VSR,LDVSR,mem);

    if( INFO != 0 ) CErr("QZ Failed in OrdQZ");
    
    bool swap = true;

    int * SELECT = mem.malloc<int>(N);
    std::fill_n(SELECT,N,int(false));



    double betaTol = 1e-13;
    for(int i = 1; i < N; i++) 
    for(int j = i; j > 0; j--) {

        bool perfSwap = false;
        bool jm1Small = std::abs(BETA[j-1]) < betaTol;
        bool jSmall   = std::abs(BETA[j])   < betaTol;

        if( jSmall )        perfSwap = false; // if b[j] small, no swap
        else if( jm1Small ) perfSwap = true;  // swap if b[j-1] small
        else {

          // if (W[j-1] - SIGMA) > (W[j] - SIGMA), swap
          double Wjm1 = std::abs(ALPHA[j-1]/BETA[j-1] - SIGMA);
          double Wj   = std::abs(ALPHA[j]/BETA[j]     - SIGMA);
          if (std::real(ALPHA[j-1]/BETA[j-1]) < hLim) {
            Wjm1 += LARGE;
          }
          if (std::real(ALPHA[j]/BETA[j]) < hLim) {
            Wj += LARGE;
          }
          perfSwap = (Wjm1 > Wj) and (std::abs(Wjm1 - Wj) > 1e-10) ;

        }

        if( perfSwap ) {

          int ifst = j + 1;
          int ilst = j;

          TGEXC(WantQ,WantZ,N,A,LDA,B,LDA,VSL,LDVSL,VSR,LDVSR,ifst,
            ilst,mem);

          // Recompute the Generalized eigenvalues from the permuted
          // Shur form
          TGSEN(0,WantQ,WantZ,SELECT,N,A,LDA,B,LDB,ALPHA,BETA,VSL,LDVSL,VSR,
            LDVSR,mem);

        }

    }


    mem.free(SELECT);

    return INFO;

  }


  template <typename _F>
  int OrdQZ(char JOBVSL, char JOBVSR, int N, _F *A, int LDA, _F *B, 
    int LDB, dcomplex *ALPHA, _F *BETA, double SIGMA, _F *VSL, int LDVSL, 
    _F *VSR, int LDVSR, CQMemManager &mem) {

    // Aux booleans for reordering functions
    bool WantQ = JOBVSL == 'V';
    bool WantZ = JOBVSR == 'V';


    // Get the initial QZ factorization
    int INFO = 
      QZ(JOBVSL,JOBVSR,N,A,LDA,B,LDB,ALPHA,BETA,VSL,LDVSL,VSR,LDVSR,mem);

    if( INFO != 0 ) CErr("QZ Failed in OrdQZ");
    
    bool swap = true;

    int * SELECT = mem.malloc<int>(N);
    std::fill_n(SELECT,N,int(false));

#if 0
    int nCount = 0;
    do {

      // Catch if something stupid has happened
      if(nCount++ > N) CErr("OrdQZ Failed!");

      if( swap ) {

        swap = false;
        double betaTol = 1e-13;
        for(auto j = 0; j < (N-1); j++) {

          bool perfSwap = false;
          bool jSmall  = std::abs(BETA[j])   < betaTol;
          bool j1Small = std::abs(BETA[j+1]) < betaTol;

          if( j1Small )     perfSwap = false; // if b[j+1] small, no swap
          else if( jSmall ) perfSwap = true;  // swap if b[j] small
          else {

            // if (W[j] - SIGMA) > (W[j+1] - SIGMA), swap
            double Wj  = std::abs(ALPHA[j]/BETA[j]     - SIGMA);
            double Wj1 = std::abs(ALPHA[j+1]/BETA[j+1] - SIGMA);
            perfSwap = (Wj > Wj1) and (std::abs(Wj - Wj1) > 1e-10) ;

          }


          if( perfSwap ) {

            swap = true;
            int ifst = j + 2;
            int ilst = j + 1;

            TGEXC(WantQ,WantZ,N,A,LDA,B,LDA,VSL,LDVSL,VSR,LDVSR,ifst,
              ilst,mem);

          }
        }

        // Recompute the Generalized eigenvalues from the permuted
        // Shur form
        TGSEN(0,WantQ,WantZ,SELECT,N,A,LDA,B,LDB,ALPHA,BETA,VSL,LDVSL,VSR,
          LDVSR,mem);


      }


    } while(swap);
#else


    double betaTol = 1e-13;
    for(int i = 1; i < N; i++) 
    for(int j = i; j > 0; j--) {

        bool perfSwap = false;
        bool jm1Small = std::abs(BETA[j-1]) < betaTol;
        bool jSmall   = std::abs(BETA[j])   < betaTol;

        if( jSmall )        perfSwap = false; // if b[j] small, no swap
        else if( jm1Small ) perfSwap = true;  // swap if b[j-1] small
        else {

          // if (W[j-1] - SIGMA) > (W[j] - SIGMA), swap
          double Wjm1 = std::abs(ALPHA[j-1]/BETA[j-1] - SIGMA);
          double Wj   = std::abs(ALPHA[j]/BETA[j]     - SIGMA);
          perfSwap = (Wjm1 > Wj) and (std::abs(Wjm1 - Wj) > 1e-10) ;

        }

        if( perfSwap ) {

          int ifst = j + 1;
          int ilst = j;

          TGEXC(WantQ,WantZ,N,A,LDA,B,LDA,VSL,LDVSL,VSR,LDVSR,ifst,
            ilst,mem);

          // Recompute the Generalized eigenvalues from the permuted
          // Shur form
          TGSEN(0,WantQ,WantZ,SELECT,N,A,LDA,B,LDB,ALPHA,BETA,VSL,LDVSL,VSR,
            LDVSR,mem);

        }

    }

#endif

    mem.free(SELECT);

    return INFO;

  }

  template 
  int OrdQZ2(char JOBVSL, char JOBVSR, int N, double *A, int LDA, double *B, 
    int LDB, dcomplex *ALPHA, double *BETA, double hLim, double SIMGA, double *VSL, 
    int LDVSL, double *VSR, int LDVSR, CQMemManager &mem);

  template 
  int OrdQZ2(char JOBVSL, char JOBVSR, int N, dcomplex *A, int LDA, dcomplex *B, 
    int LDB, dcomplex *ALPHA, dcomplex *BETA, double hLim, double SIMGA, dcomplex *VSL, 
    int LDVSL, dcomplex *VSR, int LDVSR, CQMemManager &mem);


  template 
  int OrdQZ(char JOBVSL, char JOBVSR, int N, double *A, int LDA, double *B, 
    int LDB, dcomplex *ALPHA, double *BETA, double SIMGA, double *VSL, 
    int LDVSL, double *VSR, int LDVSR, CQMemManager &mem);

  template 
  int OrdQZ(char JOBVSL, char JOBVSR, int N, dcomplex *A, int LDA, dcomplex *B, 
    int LDB, dcomplex *ALPHA, dcomplex *BETA, double SIMGA, dcomplex *VSL, 
    int LDVSL, dcomplex *VSR, int LDVSR, CQMemManager &mem);


}; // namespace ChronusQ
