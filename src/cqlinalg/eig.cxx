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
#include <cqlinalg/eig.hpp>
#include <cqlinalg/util.hpp>

namespace ChronusQ {

  /**
   *  \brief Sorts a vector of eigenvalues in a standard way.
   *
   *  Assigns a standard ordering for both real and complex
   *  eigenvalues. Initially sorts by magnitude, then by the real
   *  part to ensure a consistant ordering between conjugate pairs
   *  if they exist.
   *
   *  \param [in]      N Number of eigenvalues to sort
   *  \param [in/out]  W Eigenvalues to sort. On exit contains sorted 
   *                     eigenvalues
   */ 
  template <typename _F>
  void EigenSort(int N, _F *W) {
  
    // Initially sort by magnitude
    std::stable_sort(W,W+N,
      [&](const _F& a, const _F& b){
        return std::abs(a) < std::abs(b);
      }
    );
  
    // Then sort by real part
    std::stable_sort(W,W+N,
      [&](const _F& a, const _F& b){
        return std::real(a) < std::real(b);
      }
    );
  }; // EigenSort (just values)
  
  /**
   *  \brief Sorts a vector of eigenvalues relative to a particular value
   *  in a standard way.
   *
   *  Assigns a standard ordering for both real and complex
   *  eigenvalues relative to their distance from a provided shift value. 
   *  Initially sorts by magnitude of the distance, then by the real
   *  part of the distance to ensure a consistant ordering between 
   *  conjugate pairs if they exist.
   *
   *  \param [in]      N Number of eigenvalues to sort
   *  \param [in/out]  W Eigenvalues to sort. On exit contains sorted 
   *                     eigenvalues
   *  \param [in]  shift Shift to sort around.
   */ 
  template <typename _F>
  void EigenSort(int N, _F *W, _F shift) {
  
    // Initially sort by magnitude
    std::stable_sort(W,W+N,
      [&](const _F& a, const _F& b){
        return std::abs(a) < std::abs(b);
      }
    );
  
    // Then sort by real part
    std::stable_sort(W,W+N,
      [&](const _F& a, const _F& b){
        return std::real(a) < std::real(b);
      }
    );
  
    // Then sort by madnitude of distance
    std::stable_sort(W,W+N,
      [&](const _F& a, const _F& b){
        return std::abs(a-shift) < std::abs(b-shift);
      }
    );
  }; // EigenSort (with shift)
  
  /**
   *  \brief Sorts a vector of eigenvalues and a corresponding matrix
   *  of eigenvectors in a standard way.
   *
   *  Assigns a standard ordering for both real and complex
   *  eigenvalues. Initially sorts by eigenvalue magnitude, then by 
   *  the real part to ensure a consistant ordering between conjugate 
   *  pairs if they exist. Also sorts the eigenvectors corresponding
   *  to the eigenvalues in place.
   *
   *  XXX: Requires Eigen
   *
   *  \param [in]      N Number of eigenvalues to sort
   *  \param [in]      M Leading dimension of the eigenvectors
   *  \param [in/out]  W Eigenvalues to sort. On exit contains sorted 
   *                     eigenvalues
   *  \param [in/out]  V Corresponding eigenvectors to sort. On exit
   *                     contains sorted eigenvectors
   */ 
  template <typename _F1, typename _F2>
  void EigenSort(int N, int M, _F1 *W, _F2 *V) {
  
    Eigen::Map<
      Eigen::Matrix<_F2,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>
    > VMap(V,N,M);
  
    std::vector<size_t> indx(M,0);
    std::iota(indx.begin(),indx.end(),0);
  
    // Initially sort by magnitude
    std::stable_sort(indx.begin(),indx.end(),
      [&](const size_t& a, const size_t& b){
        return std::abs(W[a]) < std::abs(W[b]);
      }
    );
  
    // Then sort by real part
    std::stable_sort(indx.begin(),indx.end(),
      [&](const size_t& a, const size_t& b){
        return std::real(W[a]) < std::real(W[b]);
      }
    );
  
    // O(N * LOG(N)) inplace permutation
    for(auto i = 0ul; i < (M-1); i++){
      size_t ind = indx[i];
      while(ind < i) ind = indx[ind];
  
      std::swap(W[i],W[ind]);
      VMap.col(i).swap(VMap.col(ind));
    }
  }; // EigenSort ( + 1 set of vectors)
  

  /**
   *  \brief Sorts a vector of eigenvalues and two corresponding matricies
   *  of eigenvectors (Left + Right) in a standard way.
   *
   *  Assigns a standard ordering for both real and complex
   *  eigenvalues. Initially sorts by eigenvalue magnitude, then by 
   *  the real part to ensure a consistant ordering between conjugate 
   *  pairs if they exist. Also sorts both sets of eigenvectors corresponding
   *  to the eigenvalues in place.
   *
   *  XXX: Requires Eigen
   *
   *  \param [in]      N Number of eigenvalues to sort
   *  \param [in]      M Leading dimension of the eigenvectors
   *  \param [in/out]  W Eigenvalues to sort. On exit contains sorted 
   *                     eigenvalues
   *  \param [in/out] V1 A set of corresponding eigenvectors to sort. On exit
   *                     contains sorted eigenvectors
   *  \param [in/out] V2 A set of corresponding eigenvectors to sort. On exit
   *                     contains sorted eigenvectors
   */ 
  template <typename _F1, typename _F2>
  void EigenSort(int N, int M, _F1 *W, _F2 *V1, _F2 *V2) {
  
    Eigen::Map<
      Eigen::Matrix<_F2,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>
    > V1Map(V1,N,M);

    Eigen::Map<
      Eigen::Matrix<_F2,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>
    > V2Map(V2,N,M);
  
    std::vector<size_t> indx(M,0);
    std::iota(indx.begin(),indx.end(),0);
  
    // Initially sort by magnitude
    std::stable_sort(indx.begin(),indx.end(),
      [&](const size_t& a, const size_t& b){
        return std::abs(W[a]) < std::abs(W[b]);
      }
    );
  
    // Then sort by real part
    std::stable_sort(indx.begin(),indx.end(),
      [&](const size_t& a, const size_t& b){
        return std::real(W[a]) < std::real(W[b]);
      }
    );
  
    // O(N * LOG(N)) inplace permutation
    for(auto i = 0ul; i < (M-1); i++){
      size_t ind = indx[i];
      while(ind < i) ind = indx[ind];
  
      std::swap(W[i],W[ind]);
      V1Map.col(i).swap(V1Map.col(ind));
      V2Map.col(i).swap(V2Map.col(ind));
    }
  }; // EigenSort (+ 2 sets of vectors)
  

  /**
   *  \brief Sorts a vector of eigenvalues and a corresponding matrix
   *  of eigenvectors relative to a particular value in a standard way.
   *
   *  Assigns a standard ordering for both real and complex
   *  eigenvalues relative to their distance from a provided shift value. 
   *  Initially sorts by the magnitude of the distance, then by 
   *  the real part to ensure a consistant ordering between conjugate 
   *  pairs if they exist. Also sorts the eigenvectors corresponding
   *  to the eigenvalues in place.
   *
   *  XXX: Requires Eigen
   *
   *  \param [in]      N Number of eigenvalues to sort
   *  \param [in]      M Leading dimension of the eigenvectors
   *  \param [in/out]  W Eigenvalues to sort. On exit contains sorted 
   *                     eigenvalues
   *  \param [in/out]  V Corresponding eigenvectors to sort. On exit
   *                     contains sorted eigenvectors
   *  \param [in]  shift Shift to sort around.
   */ 
  template <typename _F1, typename _F2>
  void EigenSort(int N, int M, _F1 *W, _F2 *V, _F1 shift) {
  
    Eigen::Map<
      Eigen::Matrix<_F2,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>
    > VMap(V,N,M);
  
    std::vector<size_t> indx(M,0);
    std::iota(indx.begin(),indx.end(),0);
  
    // Initially sort by magnitude
    std::stable_sort(indx.begin(),indx.end(),
      [&](const size_t& a, const size_t& b){
        return std::abs(W[a]) < std::abs(W[b]);
      }
    );
  
    // Then sort by real part
    std::stable_sort(indx.begin(),indx.end(),
      [&](const size_t& a, const size_t& b){
        return std::real(W[a]) < std::real(W[b]);
      }
    );
  
    // Initially sort by magnitude of distance
    std::stable_sort(indx.begin(),indx.end(),
      [&](const size_t& a, const size_t& b){
        return std::abs(W[a] - shift) < std::abs(W[b] - shift);
      }
    );
  
    // O(N * LOG(N)) inplace permutation
    for(auto i = 0ul; i < (M-1); i++){
      size_t ind = indx[i];
      while(ind < i) ind = indx[ind];
  
      std::swap(W[i],W[ind]);
      VMap.col(i).swap(VMap.col(ind));
    }
  }; // EigenSort (+ 1 set of vectors + shift )
  
  /**
   *  \brief Sorts a vector of eigenvalues and two corresponding matricies
   *  of eigenvectors (Left + Right) relative to a particular value 
   *  in a standard way.
   *
   *  Assigns a standard ordering for both real and complex
   *  eigenvalues relative to their distance from a provided shift value. 
   *  Initially sorts by the magnitude of the distance, then by 
   *  the real part to ensure a consistant ordering between conjugate 
   *  pairs if they exist. Also sorts both sets eigenvectors corresponding
   *  to the eigenvalues in place.
   *
   *  XXX: Requires Eigen
   *
   *  \param [in]      N Number of eigenvalues to sort
   *  \param [in]      M Leading dimension of the eigenvectors
   *  \param [in/out]  W Eigenvalues to sort. On exit contains sorted 
   *                     eigenvalues
   *  \param [in/out] V1 A set of corresponding eigenvectors to sort. On exit
   *                     contains sorted eigenvectors
   *  \param [in/out] V2 A set of corresponding eigenvectors to sort. On exit
   *                     contains sorted eigenvectors
   *  \param [in]  shift Shift to sort around.
   */ 
  template <typename _F1, typename _F2>
  void EigenSort(int N, int M, _F1 *W, _F2 *V1, _F2 *V2, _F1 shift) {
  
    Eigen::Map<
      Eigen::Matrix<_F2,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>
    > V1Map(V1,N,M);

    Eigen::Map<
      Eigen::Matrix<_F2,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>
    > V2Map(V2,N,M);
  
    std::vector<size_t> indx(M,0);
    std::iota(indx.begin(),indx.end(),0);
  
    // Initially sort by magnitude
    std::stable_sort(indx.begin(),indx.end(),
      [&](const size_t& a, const size_t& b){
        return std::abs(W[a]) < std::abs(W[b]);
      }
    );
  
    // Then sort by real part
    std::stable_sort(indx.begin(),indx.end(),
      [&](const size_t& a, const size_t& b){
        return std::real(W[a]) < std::real(W[b]);
      }
    );
  
    // Initially sort by magnitude of distance
    std::stable_sort(indx.begin(),indx.end(),
      [&](const size_t& a, const size_t& b){
        return std::abs(W[a] - shift) < std::abs(W[b] - shift);
      }
    );
  
    // O(N * LOG(N)) inplace permutation
    for(auto i = 0ul; i < (M-1); i++){
      size_t ind = indx[i];
      while(ind < i) ind = indx[ind];
  
      std::swap(W[i],W[ind]);
      V1Map.col(i).swap(V1Map.col(ind));
      V2Map.col(i).swap(V2Map.col(ind));
    }
  }; // EigenSort (+2 sets of vectors + shift)
  
  
  // Specializations for full spectral sort (N = M).
  template <typename _F1, typename _F2>
  void EigenSort(int N, _F1 *W, _F2 *V){ EigenSort(N,N,W,V); };
  template <typename _F1, typename _F2>
  void EigenSort(int N, _F1 *W, _F2 *V1, _F2 *V2){ EigenSort(N,N,W,V1,V2); };
  template <typename _F1, typename _F2>
  void EigenSort(int N, _F1 *W, _F2 *V, _F1 shift){ EigenSort(N,N,W,V,shift); };
  template <typename _F1, typename _F2>
  void EigenSort(int N, _F1 *W, _F2 *V1, _F2 *V2, _F1 shift) {
    EigenSort(N,N,W,V1,V2,shift);
  };
  
  
  // Explicit instantiations of EigenSort
  
  template void EigenSort<double>(  int,double*  );
  template void EigenSort<dcomplex>(int,dcomplex*);
  
  template void EigenSort<double>(  int,double*  ,double  );
  template void EigenSort<dcomplex>(int,dcomplex*,dcomplex);
  
  template void EigenSort<double,double>(    int,int,double*  ,double*  );
  template void EigenSort<double,dcomplex>(  int,int,double*  ,dcomplex*);
  template void EigenSort<dcomplex,double>(  int,int,dcomplex*,double*  );
  template void EigenSort<dcomplex,dcomplex>(int,int,dcomplex*,dcomplex*);
  
  template void EigenSort<double,double>(    int,int,double*  ,double*  ,
    double*  );
  template void EigenSort<double,dcomplex>(  int,int,double*  ,dcomplex*,
    dcomplex*);
  template void EigenSort<dcomplex,double>(  int,int,dcomplex*,double*  ,
    double*  );
  template void EigenSort<dcomplex,dcomplex>(int,int,dcomplex*,dcomplex*,
    dcomplex*);
  
  template void EigenSort<double,double>(    int,int,double*  ,double*  ,
    double  );
  template void EigenSort<double,dcomplex>(  int,int,double*  ,dcomplex*,
    double  );
  template void EigenSort<dcomplex,double>(  int,int,dcomplex*,double*  ,
    dcomplex);
  template void EigenSort<dcomplex,dcomplex>(int,int,dcomplex*,dcomplex*,
    dcomplex);
  
  template void EigenSort<double,double>(    int,int,double*  ,double*  ,
    double*, double);
  template void EigenSort<double,dcomplex>(  int,int,double*  ,dcomplex*,
    dcomplex*, double);
  template void EigenSort<dcomplex,double>(  int,int,dcomplex*,double*  ,
    double*, dcomplex);
  template void EigenSort<dcomplex,dcomplex>(int,int,dcomplex*,dcomplex*,
    dcomplex*, dcomplex);
  
  template void EigenSort<double,double>(    int,double*  ,double*  );
  template void EigenSort<double,dcomplex>(  int,double*  ,dcomplex*);
  template void EigenSort<dcomplex,double>(  int,dcomplex*,double*  );
  template void EigenSort<dcomplex,dcomplex>(int,dcomplex*,dcomplex*);
  
  template void EigenSort<double,double>(    int,double*  ,double*  ,
    double*  );
  template void EigenSort<double,dcomplex>(  int,double*  ,dcomplex*,
    dcomplex*);
  template void EigenSort<dcomplex,double>(  int,dcomplex*,double*  ,
    double*  );
  template void EigenSort<dcomplex,dcomplex>(int,dcomplex*,dcomplex*,
    dcomplex*);
  
  template void EigenSort<double,double>(    int,double*  ,double*  ,
    double  );
  template void EigenSort<double,dcomplex>(  int,double*  ,dcomplex*,
    double  );
  template void EigenSort<dcomplex,double>(  int,dcomplex*,double*  ,
    dcomplex);
  template void EigenSort<dcomplex,dcomplex>(int,dcomplex*,dcomplex*,
    dcomplex);
  
  template void EigenSort<double,double>(    int,double*  ,double*  ,
    double*, double);
  template void EigenSort<double,dcomplex>(  int,double*  ,dcomplex*,
    dcomplex*, double);
  template void EigenSort<dcomplex,double>(  int,dcomplex*,double*  ,
    double*, dcomplex);
  template void EigenSort<dcomplex,dcomplex>(int,dcomplex*,dcomplex*,
    dcomplex*, dcomplex);
  
  

  /** LAPACK Wrappers **/

  /**
   *  \brief Wrapper around LAPACK's DGEEV.
   *
   *  Wraps DGEEV to obtain the optimal workspace dimension and allocation.
   *  Passes CQMemManager object to handle memory allocation / deallocation.
   *
   *  See http://www.netlib.org/lapack/lapack-3.1.1/html/dgeev.f.html for
   *  parameter documentation.
   */ 
  int DGEEV(char JOBVL, char JOBVR, int N, double *A, int LDA, double *WR,
    double *WI, double *VL, int LDVL, double *VR, int LDVR, 
    CQMemManager &mem) {
  
    int INFO;
  
    auto test = 
      std::bind(dgeev_,&JOBVL,&JOBVR,&N,A,&LDA,WR,WI,VL,&LDVL,VR,&LDVR,
        std::placeholders::_1,std::placeholders::_2,&INFO);
  
    int LWORK = getLWork<double>(test);
    double *WORK = mem.malloc<double>(LWORK);
  
    dgeev_(&JOBVL,&JOBVR,&N,A,&LDA,WR,WI,VL,&LDVL,VR,&LDVR,
        WORK,&LWORK,&INFO);
  
    mem.free(WORK);
    
    return INFO;
  }; // DGEEV
  

  /**
   *  \brief Wrapper around LAPACK's DSYEV.
   *
   *  Wraps DSYEV to obtain the optimal workspace dimension and allocation.
   *  Passes CQMemManager object to handle memory allocation / deallocation.
   *
   *  See http://www.netlib.org/lapack/lapack-3.1.1/html/dsyev.f.html for
   *  parameter documentation.
   */ 
  int DSYEV(char JOBZ, char UPLO, int N, double *A, int LDA, double *W,
    CQMemManager &mem) {
  
    int INFO;
  
    auto test = std::bind(dsyev_,&JOBZ,&UPLO,&N,A,&LDA,W,std::placeholders::_1,
      std::placeholders::_2,&INFO);
  
    int LWORK = getLWork<double>(test);
    double *WORK = mem.malloc<double>(LWORK);
  
    dsyev_(&JOBZ,&UPLO,&N,A,&LDA,W,WORK,&LWORK,&INFO);
  
    mem.free(WORK);
    
    return INFO;
  }; // DSYEV
  
  
  /**
   *  \brief Wrapper around LAPACK's DGGEV.
   *
   *  Wraps DGGEV to obtain the optimal workspace dimension and allocation.
   *  Passes CQMemManager object to handle memory allocation / deallocation.
   *
   *  See http://www.netlib.org/lapack/lapack-3.1.1/html/dggev.f.html for
   *  parameter documentation.
   */ 
  int DGGEV(char JOBVL, char JOBVR, int N, double *A, int LDA, double *B,
    int LDB, double *ALPHAR, double *ALPHAI, double *BETA, double *VL,
    int LDVL, double *VR, int LDVR, CQMemManager &mem) {
  
    int INFO;
  
    auto test = std::bind(dggev_,&JOBVL,&JOBVR,&N,A,&LDA,B,&LDB,ALPHAR,
      ALPHAI,BETA,VL,&LDVL,VR,&LDVR,std::placeholders::_1,
      std::placeholders::_2,&INFO);
  
    int LWORK = getLWork<double>(test);
    double *WORK = mem.malloc<double>(LWORK);
  
    dggev_(&JOBVL,&JOBVR,&N,A,&LDA,B,&LDB,ALPHAR,ALPHAI,BETA,VL,&LDVL,
      VR,&LDVR,WORK,&LWORK,&INFO);
  
    mem.free(WORK);
    return INFO;
  }; // DGGEV
  
  
  /**
   *  \brief Wrapper around LAPACK's ZGEEV.
   *
   *  Wraps ZGEEV to obtain the optimal workspace dimension and allocation.
   *  Passes CQMemManager object to handle memory allocation / deallocation.
   *
   *  See http://www.netlib.org/lapack/lapack-3.1.1/html/zgeev.f.html for
   *  parameter documentation.
   */ 
  int ZGEEV(char JOBVL, char JOBVR, int N, dcomplex *A, int LDA, dcomplex *W,
    dcomplex *VL, int LDVL, dcomplex *VR, int LDVR, CQMemManager &mem) {
  
    int INFO;
  
    double * RWORK = mem.malloc<double>(2*N);
  
    auto test = std::bind(zgeev_,&JOBVL,&JOBVR,&N,A,&LDA,W,VL,&LDVL,VR,&LDVR,
      std::placeholders::_1,std::placeholders::_2,RWORK,&INFO);
  
    int LWORK = getLWork<dcomplex>(test);
    dcomplex *WORK = mem.malloc<dcomplex>(LWORK);
  
    zgeev_(&JOBVL,&JOBVR,&N,A,&LDA,W,VL,&LDVL,VR,&LDVR,WORK,&LWORK,RWORK,&INFO);
  
    mem.free(WORK);
    mem.free(RWORK);
  
    return INFO;
  
  }; // ZGEEV
  
  /**
   *  \brief Wrapper around LAPACK's ZHEEV.
   *
   *  Wraps ZHEEV to obtain the optimal workspace dimension and allocation.
   *  Passes CQMemManager object to handle memory allocation / deallocation.
   *
   *  See http://www.netlib.org/lapack/lapack-3.1.1/html/zheev.f.html for
   *  parameter documentation.
   */ 
  int ZHEEV(char JOBZ, char UPLO, int N, dcomplex *A, int LDA, double *W,
    CQMemManager &mem){
  
    int INFO;
  
    double * RWORK = mem.malloc<double>(3*N);
  
    auto test = std::bind(zheev_,&JOBZ,&UPLO,&N,A,&LDA,W,
      std::placeholders::_1,std::placeholders::_2,RWORK,&INFO);
  
    int LWORK = getLWork<dcomplex>(test);
    dcomplex *WORK = mem.malloc<dcomplex>(LWORK);
  
    zheev_(&JOBZ,&UPLO,&N,A,&LDA,W,WORK,&LWORK,RWORK,&INFO);
  
    mem.free(WORK);
    mem.free(RWORK);
  
    return INFO;
  }; // ZHEEV
  
  /**
   *  \brief Wrapper around LAPACK's ZGGEV.
   *
   *  Wraps ZGGEV to obtain the optimal workspace dimension and allocation.
   *  Passes CQMemManager object to handle memory allocation / deallocation.
   *
   *  See http://www.netlib.org/lapack/lapack-3.1.1/html/zggev.f.html for
   *  parameter documentation.
   */ 
  int ZGGEV(char JOBVL, char JOBVR, int N, dcomplex *A, int LDA, dcomplex *B,
    int LDB, dcomplex *ALPHA, dcomplex *BETA, dcomplex *VL, int LDVL, 
    dcomplex *VR, int LDVR, CQMemManager &mem) {
  
    int INFO;
  
    double * RWORK = mem.malloc<double>(8*N);
  
    auto test = std::bind(zggev_,&JOBVL,&JOBVR,&N,A,&LDA,B,&LDB,ALPHA,
      BETA,VL,&LDVL,VR,&LDVR,std::placeholders::_1,std::placeholders::_2,
      RWORK,&INFO);
  
    int LWORK = getLWork<dcomplex>(test);
    dcomplex *WORK = mem.malloc<dcomplex>(LWORK);
  
    zggev_(&JOBVL,&JOBVR,&N,A,&LDA,B,&LDB,ALPHA,BETA,VL,&LDVL,
      VR,&LDVR,WORK,&LWORK,RWORK,&INFO);
  
    mem.free(WORK);
    mem.free(RWORK);
  
    return INFO;
  }; // ZGGEV
  

  /** Specializations of smart linear algebra routines **/
  // see include/cqlinalg/eig.hpp for docs



  template<>
  int GeneralEigen(char JOBVL, char JOBVR, int N, double *A, int LDA, 
    dcomplex *W, double *VL, int LDVL, double *VR, int LDVR, 
    CQMemManager &mem){
  
    double *WR = mem.malloc<double>(N);
    double *WI = mem.malloc<double>(N);
  
    int INFO = DGEEV(JOBVL,JOBVR,N,A,LDA,WR,WI,VL,LDVL,VR,LDVR,mem);
  
    for(auto i = 0ul; i < N; i++) {
      W[i] = dcomplex(WR[i],WI[i]);
    }
  
    mem.free(WR);
    mem.free(WI);
  
    if(JOBVL == 'V' and JOBVR == 'V') EigenSort(N,W,VR,VL);
    else if(JOBVR == 'V')             EigenSort(N,W,VR);
    else if(JOBVL == 'V')             EigenSort(N,W,VL);
    else                              EigenSort(N,W);
  
    return INFO;
  }; // GeneralEigen (real)
  
  template<>
  int GeneralEigen(char JOBVL, char JOBVR, int N, dcomplex *A, int LDA, 
    dcomplex *W, dcomplex *VL, int LDVL, dcomplex *VR, int LDVR, 
    CQMemManager &mem){
  
    int INFO = ZGEEV(JOBVL,JOBVR,N,A,LDA,W,VL,LDVL,VR,LDVR,mem);
  
    if(JOBVL == 'V' and JOBVR == 'V') EigenSort(N,W,VR,VL);
    else if(JOBVR == 'V')             EigenSort(N,W,VR);
    else if(JOBVL == 'V')             EigenSort(N,W,VL);
    else                              EigenSort(N,W);
  
    return INFO;
  
  }; // GeneralEigen (complex)
  
  template<>
  int HermetianEigen(char JOBZ, char UPLO, int N, double *A, int LDA, double *W,
    CQMemManager &mem){
    
    return DSYEV(JOBZ,UPLO,N,A,LDA,W,mem);
  
  }; // HermetianEigen (real / real eigenvalues)
  
  template<>
  int HermetianEigen(char JOBZ, char UPLO, int N, dcomplex *A, int LDA, 
    double *W, CQMemManager &mem){
    
    return ZHEEV(JOBZ,UPLO,N,A,LDA,W,mem);
  
  }; // HermetianEigen (complex / real eigenvalues)
  
  template<>
  int HermetianEigen(char JOBZ, char UPLO, int N, double *A, int LDA, 
    dcomplex *W, CQMemManager &mem){
  
    int INFO;
    double *WReal = mem.malloc<double>(N);
  
    INFO = HermetianEigen(JOBZ,UPLO,N,A,LDA,WReal,mem);
    
    for(auto i = 0; i < N; i++) W[i] = WReal[i];

    mem.free(WReal);
  
    return INFO;
  
  }; // HermetianEigen ( real / complex eigenvalues)
  
  template<>
  int HermetianEigen(char JOBZ, char UPLO, int N, dcomplex *A, int LDA, 
    dcomplex *W, CQMemManager &mem){
  
    int INFO;
    double *WReal = mem.malloc<double>(N);
  
    INFO = HermetianEigen(JOBZ,UPLO,N,A,LDA,WReal,mem);
    
    for(auto i = 0; i < N; i++) W[i] = WReal[i];

    mem.free(WReal);
  
    return INFO;
  
  }; // HermetianEigen (complex / complex eigenvalues )
  
  template <>
  int GeneralizedEigen(char JOBVL, char JOBVR, int N, double *A, int LDA, 
    double *B, int LDB, dcomplex *ALPHA, double *BETA, double *VL, int LDVL, 
    double *VR, int LDVR, CQMemManager &mem){
  
    double *ALPHAR = mem.malloc<double>(N);
    double *ALPHAI = mem.malloc<double>(N);
  
    int INFO = DGGEV(JOBVL,JOBVR,N,A,LDA,B,LDB,ALPHAR,ALPHAI,BETA,VL,LDVL,
      VR,LDVR,mem);
  
    for(auto i = 0ul; i < N; i++) {
      ALPHA[i] = dcomplex(ALPHAR[i],ALPHAI[i]);
    }
  
    mem.free(ALPHAR);
    mem.free(ALPHAI);
  
    return INFO;
  
  }; // GeneralizedEigen (real / generalized eigenvalues)
  
  template <>
  int GeneralizedEigen(char JOBVL, char JOBVR, int N, double *A, int LDA, 
    double *B, int LDB, dcomplex *W, double *VL, int LDVL, double *VR, 
    int LDVR, CQMemManager &mem){
  
    double *ALPHAR = mem.malloc<double>(N);
    double *ALPHAI = mem.malloc<double>(N);
    double *BETA   = mem.malloc<double>(N);
  
    int INFO = DGGEV(JOBVL,JOBVR,N,A,LDA,B,LDB,ALPHAR,ALPHAI,BETA,VL,LDVL,
      VR,LDVR,mem);
  
    for(auto i = 0ul; i < N; i++) {
      W[i] = dcomplex(ALPHAR[i],ALPHAI[i])/BETA[i];
    }
  
    mem.free(ALPHAR);
    mem.free(ALPHAI);
    mem.free(BETA);
  
    if(JOBVL == 'V' and JOBVR == 'V') EigenSort(N,W,VR,VL);
    else if(JOBVR == 'V')             EigenSort(N,W,VR);
    else if(JOBVL == 'V')             EigenSort(N,W,VL);
    else                              EigenSort(N,W);
  
    return INFO;
  
  }; // GeneralizedEigen (real / actual eigenvalues)
  
  template <>
  int GeneralizedEigen(char JOBVL, char JOBVR, int N, dcomplex *A, int LDA, 
    dcomplex *B, int LDB, dcomplex *ALPHA, dcomplex *BETA, dcomplex *VL, 
    int LDVL, dcomplex *VR, int LDVR, CQMemManager &mem){
  
  
    return ZGGEV(JOBVL,JOBVR,N,A,LDA,B,LDB,ALPHA,BETA,VL,LDVL,VR,LDVR,mem);
  
  }; // GeneralizedEigen (complex / generalized eigenvalues)
  
  template <>
  int GeneralizedEigen(char JOBVL, char JOBVR, int N, dcomplex *A, int LDA, 
    dcomplex *B, int LDB, dcomplex *W, dcomplex *VL, int LDVL, dcomplex *VR, 
    int LDVR, CQMemManager &mem){
  
    dcomplex *ALPHA = mem.malloc<dcomplex>(N);
    dcomplex *BETA  = mem.malloc<dcomplex>(N);
  
    int INFO = ZGGEV(JOBVL,JOBVR,N,A,LDA,B,LDB,ALPHA,BETA,VL,LDVL,VR,LDVR,mem);
  
    for(auto i = 0ul; i < N; i++) {
      W[i] = ALPHA[i]/BETA[i];
    }
  
    mem.free(ALPHA);
    mem.free(BETA);
  
    if(JOBVL == 'V' and JOBVR == 'V') EigenSort(N,W,VR,VL);
    else if(JOBVR == 'V')             EigenSort(N,W,VR);
    else if(JOBVL == 'V')             EigenSort(N,W,VL);
    else                              EigenSort(N,W);
  
    return INFO;
  
  }; // GeneralizedEigen (complex / actual eigenvalues)

};
