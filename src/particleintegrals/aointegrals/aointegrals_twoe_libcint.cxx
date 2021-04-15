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
#include <particleintegrals/twopints/incore4indextpi.hpp>
#include <particleintegrals/twopints/incore4indexreleri.hpp>
#include <cqlinalg.hpp>
#include <cqlinalg/blasutil.hpp>
#include <util/timer.hpp>
#include <util/matout.hpp>

#include <util/threads.hpp>
#include <chrono>

#include <libcint.hpp>

//#define __DEBUGERI__


namespace ChronusQ {

  template <>
  void InCore4indexTPI<dcomplex>::computeERINRCINT(BasisSet&, Molecule&,
      EMPerturbation&, OPERATOR, const HamiltonianOptions&) {
    CErr("Only real GTOs are allowed",std::cout);
  };

  template <>
  void InCore4indexTPI<double>::computeERINRCINT(BasisSet &basisSet_, Molecule &molecule_,
      EMPerturbation&, OPERATOR, const HamiltonianOptions &hamiltonianOptions) {

    if (basisSet_.forceCart)
      CErr("Libcint + cartesian GTO NYI.");

    int nAtoms = molecule_.nAtoms;
    int nShells = basisSet_.nShell;
    int iAtom, iShell, off;

    // ATM_SLOTS = 6; BAS_SLOTS = 8;
    int *atm = memManager_.template malloc<int>(nAtoms * ATM_SLOTS);
    int *bas = memManager_.template malloc<int>(nShells * BAS_SLOTS);
    double *env = memManager_.template malloc<double>(PTR_ENV_START + nAtoms*3+nShells*basisSet_.maxPrim*2);
    double sNorm;

    off = PTR_ENV_START; // = 20

    for(iAtom = 0; iAtom < nAtoms; iAtom++) {

      atm[CHARGE_OF + ATM_SLOTS * iAtom] = molecule_.atoms[iAtom].atomicNumber;
      atm[PTR_COORD + ATM_SLOTS * iAtom] = off;
      env[off + 0] = molecule_.atoms[iAtom].coord[0]; // x (Bohr)
      env[off + 1] = molecule_.atoms[iAtom].coord[1]; // y (Bohr)
      env[off + 2] = molecule_.atoms[iAtom].coord[2]; // z (Bohr)
      off += 3;

    }

    for(iShell = 0; iShell < nShells; iShell++) {

      bas[ATOM_OF  + BAS_SLOTS * iShell]  = basisSet_.mapSh2Cen[iShell];
      bas[ANG_OF   + BAS_SLOTS * iShell]  = basisSet_.shells[iShell].contr[0].l;
      bas[NPRIM_OF + BAS_SLOTS * iShell]  = basisSet_.shells[iShell].alpha.size();
      bas[NCTR_OF  + BAS_SLOTS * iShell]  = 1;
      bas[PTR_EXP  + BAS_SLOTS * iShell]  = off;

      for(int iPrim=0; iPrim<basisSet_.shells[iShell].alpha.size(); iPrim++)
        env[off + iPrim] = basisSet_.shells[iShell].alpha[iPrim];

      off +=basisSet_.shells[iShell].alpha.size();

      bas[PTR_COEFF+ BAS_SLOTS * iShell] = off;

      // Spherical GTO normalization constant missing in Libcint
      sNorm = 2.0*std::sqrt(M_PI)/std::sqrt(2.0*basisSet_.shells[iShell].contr[0].l+1.0);
      for(int iCoeff=0; iCoeff<basisSet_.shells[iShell].alpha.size(); iCoeff++){
        env[off + iCoeff] = basisSet_.shells[iShell].contr[0].coeff[iCoeff]*sNorm;
     }

      off += basisSet_.shells[iShell].alpha.size();
    }


    // Determine the number of OpenMP threads
    int nthreads = GetNumThreads();
 
    // Allocate and zero out ERIs
    size_t NB  = basisSet_.nBasis;
    size_t NB2 = NB*NB;
    size_t NB3 = NB2*NB;
    size_t NB4 = NB2*NB2;

    InCore4indexTPI<double>::clear();


    // Get threads result buffer
    int buffSize = (basisSet_.maxL+1)*(basisSet_.maxL+2)/2;
    int buffN4 = buffSize*buffSize*buffSize*buffSize;
    double *buffAll = memManager_.malloc<double>(buffN4*nthreads);

    std::cout<<"Using Libcint "<<std::endl;

    auto topERI4 = tick();

    #pragma omp parallel
    {
      int thread_id = GetThreadID();

      size_t n1,n2,n3,n4,i,j,k,l,ijkl,bf1,bf2,bf3,bf4;
      size_t s4_max;
      int shls[4];
      double *buff = buffAll+buffN4*thread_id;

      for(size_t s1(0), bf1_s(0), s1234(0); s1 < nShells; 
          bf1_s+=n1, s1++) { 

        n1 = basisSet_.shells[s1].size(); // Size of Shell 1

      for(size_t s2(0), bf2_s(0); s2 <= s1; bf2_s+=n2, s2++) {

        n2 = basisSet_.shells[s2].size(); // Size of Shell 2

      for(size_t s3(0), bf3_s(0); s3 <= s1; bf3_s+=n3, s3++) {

        n3 = basisSet_.shells[s3].size(); // Size of Shell 3
        s4_max = (s1 == s3) ? s2 : s3; // Determine the unique max of Shell 4

      for(size_t s4(0), bf4_s(0); s4 <= s4_max; bf4_s+=n4, s4++, s1234++) {

        n4 = basisSet_.shells[s4].size(); // Size of Shell 4

        // Round Robbin work distribution
        #ifdef _OPENMP
        if( s1234 % nthreads != thread_id ) continue;
        #endif

        shls[0] = int(s1);
        shls[1] = int(s2);
        shls[2] = int(s3);
        shls[3] = int(s4);

        if (basisSet_.forceCart) {
          if(int2e_cart(buff, nullptr, shls, atm, nAtoms, bas, nShells, env, nullptr, nullptr)==0) continue;
        } else {
          if(int2e_sph(buff, nullptr, shls, atm, nAtoms, bas, nShells, env, nullptr, nullptr)==0) continue;
        }

        // permutational symmetry
	ijkl = 0ul;
        for(l = 0ul, bf4 = bf4_s ; l < n4; ++l, bf4++)
        for(k = 0ul, bf3 = bf3_s ; k < n3; ++k, bf3++) 
        for(j = 0ul, bf2 = bf2_s ; j < n2; ++j, bf2++) 
        for(i = 0ul, bf1 = bf1_s ; i < n1; ++i, bf1++) 
	{

            // (12 | 34)
            (*this)(bf1, bf2, bf3, bf4) = buff[ijkl];
            // (12 | 43)
            (*this)(bf1, bf2, bf4, bf3) = buff[ijkl];
            // (21 | 34)
            (*this)(bf2, bf1, bf3, bf4) = buff[ijkl];
            // (21 | 43)
            (*this)(bf2, bf1, bf4, bf3) = buff[ijkl];
            // (34 | 12)
            (*this)(bf3, bf4, bf1, bf2) = buff[ijkl];
            // (43 | 12)
            (*this)(bf4, bf3, bf1, bf2) = buff[ijkl];
            // (34 | 21)
            (*this)(bf3, bf4, bf2, bf1) = buff[ijkl];
            // (43 | 21)
            (*this)(bf4, bf3, bf2, bf1) = buff[ijkl];

	    ijkl++;

        }; // ijkl loop
      }; // s4
      }; // s3
      }; // s2
      }; // s1

    }; // omp region

    auto durERI4 = tock(topERI4);
    std::cout << "Libcint-ERI4 duration   = " << durERI4 << std::endl;

    memManager_.free(buffAll, atm, bas, env);

#ifdef __DEBUGERI__
    // Debug output of the ERIs
    std::cout << std::scientific << std::setprecision(16);
    std::cout << "Libcint ERI (ab|cd)" << std::endl;
    for(auto i = 0ul; i < NB; i++)
    for(auto j = 0ul; j < NB; j++)
    for(auto k = 0ul; k < NB; k++)
    for(auto l = 0ul; l < NB; l++){
      std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
      std::cout << (*this)(i, j, k, l) << std::endl;
    };
#endif // __DEBUGERI__ 


  } // computeERINRCINT




  template <>
  void InCore4indexRelERI<dcomplex>::computeERICINT(BasisSet&, Molecule&,
      EMPerturbation&, OPERATOR, const HamiltonianOptions&) {
    CErr("Only real GTOs are allowed",std::cout);
  };

  template <>
  void InCore4indexRelERI<double>::computeERICINT(BasisSet &originalBasisSet, Molecule &molecule_,
      EMPerturbation&, OPERATOR, const HamiltonianOptions &hamiltonianOptions) {

    if (originalBasisSet.forceCart)
      CErr("Libcint + cartesian GTO NYI.");

    BasisSet basisSet_(originalBasisSet);

    std::vector<libint2::Shell> shells;

    shells.push_back(*basisSet_.shells.begin());

    size_t buffSize = shells.back().size();
    size_t countExpCoef = shells.back().alpha.size() * 2;

    for (auto it = ++basisSet_.shells.begin(); it != basisSet_.shells.end(); it++) {

      if (shells.back().O == it->O and
          shells.back().alpha == it->alpha and
          shells.back().contr[0].l == it->contr[0].l) {

        shells.back().contr.push_back(it->contr[0]);
        countExpCoef += shells.back().alpha.size();

      } else {
        shells.push_back(*it);
        countExpCoef += shells.back().alpha.size() * 2;
      }

      buffSize = std::max(buffSize, shells.back().size());

    }

    basisSet_.shells = shells;

    basisSet_.update(false);

    int nAtoms = molecule_.nAtoms;
    int nShells = basisSet_.nShell;
    int iAtom, iShell, off;

    // ATM_SLOTS = 6; BAS_SLOTS = 8;
    int *atm = memManager_.template malloc<int>(nAtoms * ATM_SLOTS);
    int *bas = memManager_.template malloc<int>(nShells * BAS_SLOTS);
    double *env = memManager_.template malloc<double>(PTR_ENV_START + nAtoms*3 + countExpCoef);
    double sNorm;

    off = PTR_ENV_START; // = 20

    for(iAtom = 0; iAtom < nAtoms; iAtom++) {

      atm[CHARGE_OF + ATM_SLOTS * iAtom] = molecule_.atoms[iAtom].atomicNumber;
      atm[PTR_COORD + ATM_SLOTS * iAtom] = off;
      env[off + 0] = molecule_.atoms[iAtom].coord[0]; // x (Bohr)
      env[off + 1] = molecule_.atoms[iAtom].coord[1]; // y (Bohr)
      env[off + 2] = molecule_.atoms[iAtom].coord[2]; // z (Bohr)
      off += 3;

    }

    for(iShell = 0; iShell < nShells; iShell++) {

      int nContr = basisSet_.shells[iShell].contr.size();

      bas[ATOM_OF  + BAS_SLOTS * iShell]  = basisSet_.mapSh2Cen[iShell];
      bas[ANG_OF   + BAS_SLOTS * iShell]  = basisSet_.shells[iShell].contr[0].l;
      bas[NPRIM_OF + BAS_SLOTS * iShell]  = basisSet_.shells[iShell].alpha.size();
      bas[NCTR_OF  + BAS_SLOTS * iShell]  = nContr;
      bas[PTR_EXP  + BAS_SLOTS * iShell]  = off;

      for(int iPrim=0; iPrim<basisSet_.shells[iShell].alpha.size(); iPrim++)
        env[off + iPrim] = basisSet_.shells[iShell].alpha[iPrim];

      off +=basisSet_.shells[iShell].alpha.size();

      bas[PTR_COEFF+ BAS_SLOTS * iShell] = off;

      // Spherical GTO normalization constant missing in Libcint
      sNorm = 2.0*std::sqrt(M_PI)/std::sqrt(2.0*basisSet_.shells[iShell].contr[0].l+1.0);
      for (size_t i = 0; i < nContr; i++) {
        for(int iCoeff=0; iCoeff<basisSet_.shells[iShell].alpha.size(); iCoeff++){
          env[off + iCoeff] = basisSet_.shells[iShell].contr[i].coeff[iCoeff]*sNorm;
        }
        off += basisSet_.shells[iShell].alpha.size();
      }

    }

    int cache_size = 0;
    for (int i = 0; i < nShells; i++) {
      int n, shls[4]{i,i,i,i};
      if (basisSet_.forceCart) {
        n = int2e_cart(nullptr, nullptr, shls, atm, nAtoms, bas, nShells, env, nullptr, nullptr);
        cache_size = std::max(cache_size, n);
        if(hamiltonianOptions.DiracCoulomb) {
          n = int2e_ipvip1_cart(nullptr, nullptr, shls, atm, nAtoms, bas, nShells, env, nullptr, nullptr);
          cache_size = std::max(cache_size, n);
        }
        if(hamiltonianOptions.Gaunt) {
          n = int2e_ip1ip2_cart(nullptr, nullptr, shls, atm, nAtoms, bas, nShells, env, nullptr, nullptr);
          cache_size = std::max(cache_size, n);
        }
      } else {
        n = int2e_sph(nullptr, nullptr, shls, atm, nAtoms, bas, nShells, env, nullptr, nullptr);
        cache_size = std::max(cache_size, n);
        if(hamiltonianOptions.DiracCoulomb) {
          n = int2e_ipvip1_sph(nullptr, nullptr, shls, atm, nAtoms, bas, nShells, env, nullptr, nullptr);
          cache_size = std::max(cache_size, n);
        }
        if(hamiltonianOptions.Gaunt) {
          n = int2e_ip1ip2_sph(nullptr, nullptr, shls, atm, nAtoms, bas, nShells, env, nullptr, nullptr);
          cache_size = std::max(cache_size, n);
        }
      }
    }


    // Determine the number of OpenMP threads
    int nthreads = GetNumThreads();
 
    // Allocate and zero out ERIs
    size_t NB  = basisSet_.nBasis;
    size_t NB2 = NB*NB;
    size_t NB3 = NB2*NB;
    size_t NB4 = NB2*NB2;

    InCore4indexTPI<double>::clear();


    // Get threads result buffer
    int buffN4 = buffSize*buffSize*buffSize*buffSize;
    if (hamiltonianOptions.DiracCoulomb or hamiltonianOptions.Gaunt)
      buffN4 *= 9;
    double *buffAll = memManager_.malloc<double>(buffN4*nthreads);
    double *cacheAll = memManager_.malloc<double>(cache_size*nthreads);

    std::cout<<"Using Libcint "<<std::endl;

#if 1 // (ij|kl)
    auto topERI4 = tick();

    #pragma omp parallel
    {
      int thread_id = GetThreadID();

      size_t n1,n2,n3,n4,i,j,k,l,ijkl,bf1,bf2,bf3,bf4;
      size_t s4_max;
      int shls[4];
      double *buff = buffAll+buffN4*thread_id;
      double *cache = cacheAll+cache_size*thread_id;

      for(size_t s1(0), bf1_s(0), s1234(0); s1 < nShells; 
          bf1_s+=n1, s1++) { 

        n1 = basisSet_.shells[s1].size(); // Size of Shell 1

      for(size_t s2(0), bf2_s(0); s2 <= s1; bf2_s+=n2, s2++) {

        n2 = basisSet_.shells[s2].size(); // Size of Shell 2

      for(size_t s3(0), bf3_s(0); s3 <= s1; bf3_s+=n3, s3++) {

        n3 = basisSet_.shells[s3].size(); // Size of Shell 3
        s4_max = (s1 == s3) ? s2 : s3; // Determine the unique max of Shell 4

      for(size_t s4(0), bf4_s(0); s4 <= s4_max; bf4_s+=n4, s4++, s1234++) {

        n4 = basisSet_.shells[s4].size(); // Size of Shell 4

        // Round Robbin work distribution
        #ifdef _OPENMP
        if( s1234 % nthreads != thread_id ) continue;
        #endif

        shls[0] = int(s1);
        shls[1] = int(s2);
        shls[2] = int(s3);
        shls[3] = int(s4);

        if (basisSet_.forceCart) {
          if(int2e_cart(buff, nullptr, shls, atm, nAtoms, bas, nShells, env, nullptr, cache)==0) continue;
        } else {
          if(int2e_sph(buff, nullptr, shls, atm, nAtoms, bas, nShells, env, nullptr, cache)==0) continue;
        }

        // permutational symmetry
	ijkl = 0ul;
        for(l = 0ul, bf4 = bf4_s ; l < n4; ++l, bf4++)
        for(k = 0ul, bf3 = bf3_s ; k < n3; ++k, bf3++) 
        for(j = 0ul, bf2 = bf2_s ; j < n2; ++j, bf2++) 
        for(i = 0ul, bf1 = bf1_s ; i < n1; ++i, bf1++) 
	{

            // (12 | 34)
            (*this)(bf1, bf2, bf3, bf4) = buff[ijkl];
            // (12 | 43)
            (*this)(bf1, bf2, bf4, bf3) = buff[ijkl];
            // (21 | 34)
            (*this)(bf2, bf1, bf3, bf4) = buff[ijkl];
            // (21 | 43)
            (*this)(bf2, bf1, bf4, bf3) = buff[ijkl];
            // (34 | 12)
            (*this)(bf3, bf4, bf1, bf2) = buff[ijkl];
            // (43 | 12)
            (*this)(bf4, bf3, bf1, bf2) = buff[ijkl];
            // (34 | 21)
            (*this)(bf3, bf4, bf2, bf1) = buff[ijkl];
            // (43 | 21)
            (*this)(bf4, bf3, bf2, bf1) = buff[ijkl];

	    ijkl++;

        }; // ijkl loop
      }; // s4
      }; // s3
      }; // s2
      }; // s1

    }; // omp region

    auto durERI4 = tock(topERI4);
    //std::cout << "L = "<< basisSet_.shells[s1].contr[0].l<<" "<<basisSet_.shells[s2].contr[0].l<<" "
    //	               << basisSet_.shells[s3].contr[0].l<<" "<<basisSet_.shells[s4].contr[0].l<<std::endl;
    std::cout << "Libcint-ERI4 duration   = " << durERI4 << std::endl;

#ifdef __DEBUGERI__
    // Debug output of the ERIs
    std::cout << std::scientific << std::setprecision(16);
    std::cout << "Libcint ERI (ab|cd)" << std::endl;
    for(auto i = 0ul; i < NB; i++)
    for(auto j = 0ul; j < NB; j++)
    for(auto k = 0ul; k < NB; k++)
    for(auto l = 0ul; l < NB; l++){
      std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
      std::cout << (*this)(i, j, k, l) << std::endl;
    };
#endif // __DEBUGERI__ 

#endif // (ijkl)





    /* Dirac-Coulomb Integrals */
    if(hamiltonianOptions.DiracCoulomb) { // Dirac-Coulomb ∇_i∇_j(ij|kl)

      for (InCore4indexTPI<double>& c : components_) c.clear();
  
      int AxBx = 0;
      int AxBy = 1;
      int AxBz = 2;
      int AyBx = 3;
      int AyBy = 4;
      int AyBz = 5;
      int AzBx = 6;
      int AzBy = 7;
      int AzBz = 8;
  
      auto topERIDC = tick();
  
      #pragma omp parallel
      {
        int thread_id = GetThreadID();
  
        size_t n1,n2,n3,n4,i,j,k,l,ijkl,bf1,bf2,bf3,bf4;
        size_t s4_max;
        int shls[4];
        double *buff = buffAll + buffN4*thread_id;
        double *cache = cacheAll+cache_size*thread_id;
  
        for(size_t s1(0), bf1_s(0), s1234(0); s1 < nShells; bf1_s+=n1, s1++) { 
  
          n1 = basisSet_.shells[s1].size(); // Size of Shell 1
  
        for(size_t s2(0), bf2_s(0); s2 <= s1; bf2_s+=n2, s2++) {
  
          n2 = basisSet_.shells[s2].size(); // Size of Shell 2
  
        for(size_t s3(0), bf3_s(0); s3 < nShells ; bf3_s+=n3, s3++) {
  
          n3 = basisSet_.shells[s3].size(); // Size of Shell 3
  
        for(size_t s4(0), bf4_s(0); s4 <= s3; bf4_s+=n4, s4++, s1234++) {
  
          n4 = basisSet_.shells[s4].size(); // Size of Shell 4
  
          // Round Robbin work distribution
          #ifdef _OPENMP
          if( s1234 % nthreads != thread_id ) continue;
          #endif
  
          shls[0] = int(s1);
          shls[1] = int(s2);
          shls[2] = int(s3);
          shls[3] = int(s4);
  
          if (basisSet_.forceCart) {
            if(int2e_ipvip1_cart(buff, nullptr, shls, atm, nAtoms, bas, nShells, env, nullptr, cache)==0) continue;
          } else {
            if(int2e_ipvip1_sph(buff, nullptr, shls, atm, nAtoms, bas, nShells, env, nullptr, cache)==0) continue;
          }

          ijkl = 0ul;
  	  auto nQuad = n1*n2*n3*n4;
          for(l = 0ul, bf4 = bf4_s ; l < n4; ++l, bf4++)
          for(k = 0ul, bf3 = bf3_s ; k < n3; ++k, bf3++) 
          for(j = 0ul, bf2 = bf2_s ; j < n2; ++j, bf2++) 
          for(i = 0ul, bf1 = bf1_s ; i < n1; ++i, bf1++) {
  
#ifdef __DEBUGERI__
  
          std::cout << std::scientific << std::setprecision(16);
  	  std::cout <<"Libcint ∇A∙∇B(ij|kl)"<<std::endl;
  	  std::cout<<buff[AxBx*nQuad+ijkl]<<std::endl;
  	  std::cout<<buff[AxBy*nQuad+ijkl]<<std::endl;
  	  std::cout<<buff[AxBz*nQuad+ijkl]<<std::endl;
  	  std::cout<<buff[AyBx*nQuad+ijkl]<<std::endl;
  	  std::cout<<buff[AyBy*nQuad+ijkl]<<std::endl;
  	  std::cout<<buff[AyBz*nQuad+ijkl]<<std::endl;
  	  std::cout<<buff[AzBx*nQuad+ijkl]<<std::endl;
  	  std::cout<<buff[AzBy*nQuad+ijkl]<<std::endl;
  	  std::cout<<buff[AzBz*nQuad+ijkl]<<std::endl;
  
#endif
  
          // ∇A∙∇B(ij|kl)
          auto dAdotdB = buff[AxBx*nQuad+ijkl] + buff[AyBy*nQuad+ijkl] + buff[AzBz*nQuad+ijkl];
          // ∇Ax∇B(ijkl)
          auto dAcrossdB_x =  buff[AyBz*nQuad+ijkl] - buff[AzBy*nQuad+ijkl];
          auto dAcrossdB_y = -buff[AxBz*nQuad+ijkl] + buff[AzBx*nQuad+ijkl];
          auto dAcrossdB_z =  buff[AxBy*nQuad+ijkl] - buff[AyBx*nQuad+ijkl];
  
          auto IJKL = bf1 + bf2*NB + bf3*NB2 + bf4*NB3;
          auto IJLK = bf1 + bf2*NB + bf4*NB2 + bf3*NB3;
          auto JIKL = bf2 + bf1*NB + bf3*NB2 + bf4*NB3;
          auto JILK = bf2 + bf1*NB + bf4*NB2 + bf3*NB3;
  
          // ∇A∙∇B(ij|kl) followed by ∇Ax∇B(ij|kl) X, Y, and Z
          // (ij|kl)
          (*this)[0].pointer()[IJKL] = dAdotdB;
          (*this)[1].pointer()[IJKL] = dAcrossdB_x;
          (*this)[2].pointer()[IJKL] = dAcrossdB_y;
          (*this)[3].pointer()[IJKL] = dAcrossdB_z;
          // (ij|lk)
          (*this)[0].pointer()[IJLK] = dAdotdB;
          (*this)[1].pointer()[IJLK] = dAcrossdB_x;
          (*this)[2].pointer()[IJLK] = dAcrossdB_y;
          (*this)[3].pointer()[IJLK] = dAcrossdB_z;
          // (ji|kl)
          (*this)[0].pointer()[JIKL] = dAdotdB;
          (*this)[1].pointer()[JIKL] = -dAcrossdB_x;
          (*this)[2].pointer()[JIKL] = -dAcrossdB_y;
          (*this)[3].pointer()[JIKL] = -dAcrossdB_z;
          // (ji|lk)
          (*this)[0].pointer()[JILK] = dAdotdB;
          (*this)[1].pointer()[JILK] = -dAcrossdB_x;
          (*this)[2].pointer()[JILK] = -dAcrossdB_y;
          (*this)[3].pointer()[JILK] = -dAcrossdB_z;
  
  	  ijkl++;
  
          }; // ijkl loop
        }; // s4
        }; // s3
        }; // s2
        }; // s1
  
      }; // omp region
  
      auto durERIDC = tock(topERIDC);
      std::cout << "Libcint-ERI-Dirac-Coulomb duration   = " << durERIDC << std::endl;
    

#ifdef __DEBUGERI__
      std::cout << std::scientific << std::setprecision(16);
      std::cout << "ERI00-03: ∇A∙∇B(ab|cd)  ∇Ax∇B(ab|cd)-X  ∇Ax∇B(ab|cd)-Y  ∇Ax∇B(ab|cd)-Z" << std::endl;
      for(auto i = 0ul; i < NB; i++)
      for(auto j = 0ul; j < NB; j++)
      for(auto k = 0ul; k < NB; k++)
      for(auto l = 0ul; l < NB; l++){
        std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
        std::cout << (*this)[0](i, j, k, l);
        std::cout << "   ";
        std::cout << (*this)[1](i, j, k, l);
        std::cout << "   ";
        std::cout << (*this)[2](i, j, k, l);
        std::cout << "   ";
        std::cout << (*this)[3](i, j, k, l) << std::endl;
      };
#endif

    } // Dirac-Coulomb ∇_i∇_j(ij|kl)




    /* Gaunt Integrals */
    // ∇_j∇_k(ij|kl)
    if(hamiltonianOptions.Gaunt) {

      int AxCx = 0;
      int AxCy = 1;
      int AxCz = 2;
      int AyCx = 3;
      int AyCy = 4;
      int AyCz = 5;
      int AzCx = 6;
      int AzCy = 7;
      int AzCz = 8;
  
      int BxCx = 0;
      int BxCy = 1;
      int BxCz = 2;
      int ByCx = 3;
      int ByCy = 4;
      int ByCz = 5;
      int BzCx = 6;
      int BzCy = 7;
      int BzCz = 8;
  
      auto topERIGaunt = tick();
  
      #pragma omp parallel
      {
        int thread_id = GetThreadID();

        size_t n1,n2,n3,n4,i,j,k,l,ijkl,bf1,bf2,bf3,bf4;
        size_t s4_max;
        int shls[4];
        double *buff = buffAll + buffN4*thread_id;
        double *cache = cacheAll+cache_size*thread_id;
  
        for(size_t s1(0), bf1_s(0), s1234(0); s1 < nShells; 
            bf1_s+=n1, s1++) { 
  
          n1 = basisSet_.shells[s1].size(); // Size of Shell 1
  
        for(size_t s2(0), bf2_s(0); s2 < nShells; bf2_s+=n2, s2++) {
  
          n2 = basisSet_.shells[s2].size(); // Size of Shell 2
  
        for(size_t s3(0), bf3_s(0); s3 <= s1 ; bf3_s+=n3, s3++) {
  
          n3 = basisSet_.shells[s3].size(); // Size of Shell 3
  
        for(size_t s4(0), bf4_s(0); s4 < nShells ; bf4_s+=n4, s4++, s1234++) {
  
          n4 = basisSet_.shells[s4].size(); // Size of Shell 4
  
          // Round Robbin work distribution
          #ifdef _OPENMP
          if( s1234 % nthreads != thread_id ) continue;
          #endif
  
          shls[0] = int(s1);
          shls[1] = int(s2);
          shls[2] = int(s3);
          shls[3] = int(s4);

          if (basisSet_.forceCart) {
            if(int2e_ip1ip2_cart(buff, nullptr, shls, atm, nAtoms, bas, nShells, env, nullptr, cache)==0) continue;
          } else {
            if(int2e_ip1ip2_sph(buff, nullptr, shls, atm, nAtoms, bas, nShells, env, nullptr, cache)==0) continue;
          }

  
  	  ijkl = 0ul;
  	  auto nQuad = n1*n2*n3*n4;
          for(l = 0ul, bf4 = bf4_s ; l < n4; ++l, bf4++)
          for(k = 0ul, bf3 = bf3_s ; k < n3; ++k, bf3++) 
          for(j = 0ul, bf2 = bf2_s ; j < n2; ++j, bf2++) 
          for(i = 0ul, bf1 = bf1_s ; i < n1; ++i, bf1++) {
  
#ifdef __DEBUGERI__
            std::cout << std::scientific << std::setprecision(16);
            std::cout <<"Libcint ∇A∙∇C(ij|kl)"<<std::endl;
  	    std::cout<<buff[AxCx*nQuad+ijkl]<<std::endl;
  	    std::cout<<buff[AxCy*nQuad+ijkl]<<std::endl;
  	    std::cout<<buff[AxCz*nQuad+ijkl]<<std::endl;
  	    std::cout<<buff[AyCx*nQuad+ijkl]<<std::endl;
  	    std::cout<<buff[AyCy*nQuad+ijkl]<<std::endl;
  	    std::cout<<buff[AyCz*nQuad+ijkl]<<std::endl;
  	    std::cout<<buff[AzCx*nQuad+ijkl]<<std::endl;
  	    std::cout<<buff[AzCy*nQuad+ijkl]<<std::endl;
  	    std::cout<<buff[AzCz*nQuad+ijkl]<<std::endl;
#endif
  
            // ∇A∙∇C(ij|kl)
            auto dAdotdC = buff[AxCx*nQuad+ijkl] + buff[AyCy*nQuad+ijkl] + buff[AzCz*nQuad+ijkl];
            // ∇Ax∇C(ijkl)
            auto dAcrossdC_x =  buff[AyCz*nQuad+ijkl] - buff[AzCy*nQuad+ijkl];
            auto dAcrossdC_y = -buff[AxCz*nQuad+ijkl] + buff[AzCx*nQuad+ijkl];
            auto dAcrossdC_z =  buff[AxCy*nQuad+ijkl] - buff[AyCx*nQuad+ijkl];
  
  	    // Change the index so that we do ∇B∙∇C(ij|kl) using the ∇A∙∇C engine
            auto IJKL = bf2 + bf1*NB + bf3*NB2 + bf4*NB3;
            auto LKJI = bf4 + bf3*NB + bf1*NB2 + bf2*NB3;
  
            // ∇B∙∇C(ij|kl) followed by ∇Bx∇C(ij|kl) X, Y, and Z
            // (ij|kl)
            (*this)[4].pointer()[IJKL] = dAdotdC;
            (*this)[5].pointer()[IJKL] = dAcrossdC_x;
            (*this)[6].pointer()[IJKL] = dAcrossdC_y;
            (*this)[7].pointer()[IJKL] = dAcrossdC_z;
  	    // (lk|ji)
            (*this)[4].pointer()[LKJI] = dAdotdC;
            (*this)[5].pointer()[LKJI] = -dAcrossdC_x;
            (*this)[6].pointer()[LKJI] = -dAcrossdC_y;
            (*this)[7].pointer()[LKJI] = -dAcrossdC_z;
  
  
            // ∇B_x∇C_y(ij|kl) + ∇B_y∇C_x(ij|kl)
            // (ij|kl)
            (*this)[8].pointer()[IJKL] = buff[BxCy*nQuad+ijkl] + buff[ByCx*nQuad+ijkl];
            // (lk|ji)
            (*this)[8].pointer()[LKJI] = buff[BxCy*nQuad+ijkl] + buff[ByCx*nQuad+ijkl];
  
  
            // ∇B_y∇C_x(ij|kl)
            // (ij|kl)
            (*this)[9].pointer()[IJKL] = buff[ByCx*nQuad+ijkl];
            // (lk|ji)
            (*this)[9].pointer()[LKJI] = buff[BxCy*nQuad+ijkl];
  
  
            // ∇B_x∇C_z(ij|kl) + ∇B_z∇C_x(ij|kl)
            // (ij|kl)
            (*this)[10].pointer()[IJKL] = buff[BxCz*nQuad+ijkl] + buff[BzCx*nQuad+ijkl];
            // (lk|ji)
            (*this)[10].pointer()[LKJI] = buff[BxCz*nQuad+ijkl] + buff[BzCx*nQuad+ijkl];
  
  
            // ∇B_z∇C_x(ij|kl)
            // (ij|kl)
            (*this)[11].pointer()[IJKL] = buff[BzCx*nQuad+ijkl];
            // (lk|ji)
            (*this)[11].pointer()[LKJI] = buff[BxCz*nQuad+ijkl];
  
  
            // ∇B_y∇C_z(ij|kl) + ∇B_z∇C_y(ij|kl)
            // (ij|kl)
            (*this)[12].pointer()[IJKL] = buff[ByCz*nQuad+ijkl] + buff[BzCy*nQuad+ijkl];
            // (lk|ji)
            (*this)[12].pointer()[LKJI] = buff[ByCz*nQuad+ijkl] + buff[BzCy*nQuad+ijkl];
  
  
            // ∇B_z∇C_y(ij|kl)
            // (ij|kl)
            (*this)[13].pointer()[IJKL] = buff[BzCy*nQuad+ijkl];
            // (lk|ji)
            (*this)[13].pointer()[LKJI] = buff[ByCz*nQuad+ijkl];
  
  
            // - ∇B_x∇C_x(ij|kl) - ∇B_y∇C_y(ij|kl) + ∇B_z∇C_z(ij|kl)
            // (ij|kl)
            (*this)[14].pointer()[IJKL] = - buff[BxCx*nQuad+ijkl] - buff[ByCy*nQuad+ijkl] + buff[BzCz*nQuad+ijkl];
            // (lk|ji)
            (*this)[14].pointer()[LKJI] = - buff[BxCx*nQuad+ijkl] - buff[ByCy*nQuad+ijkl] + buff[BzCz*nQuad+ijkl];
  
  
            // ∇B_x∇C_x(ij|kl) - ∇B_y∇C_y(ij|kl) - ∇B_z∇C_z(ij|kl)
            // (ij|kl)
            (*this)[15].pointer()[IJKL] = buff[BxCx*nQuad+ijkl] - buff[ByCy*nQuad+ijkl] - buff[BzCz*nQuad+ijkl];
            // (lk|ji)
            (*this)[15].pointer()[LKJI] = buff[BxCx*nQuad+ijkl] - buff[ByCy*nQuad+ijkl] - buff[BzCz*nQuad+ijkl];
  
  
            // - ∇B_x∇C_x(ij|kl) + ∇B_y∇C_y(ij|kl) - ∇B_z∇C_z(ij|kl)
            // (ij|kl)
            (*this)[16].pointer()[IJKL] = - buff[BxCx*nQuad+ijkl] + buff[ByCy*nQuad+ijkl] - buff[BzCz*nQuad+ijkl];
            // (lk|ji)
            (*this)[16].pointer()[LKJI] = - buff[BxCx*nQuad+ijkl] + buff[ByCy*nQuad+ijkl] - buff[BzCz*nQuad+ijkl];
  
  
            // ∇B_x∇C_x(ij|kl)
            // (ij|kl)
            (*this)[17].pointer()[IJKL] = buff[BxCx*nQuad+ijkl];
            // (lk|ji)
            (*this)[17].pointer()[LKJI] = buff[BxCx*nQuad+ijkl];
  
  
            // ∇B_x∇C_y(ij|kl)
            // (ij|kl)
            (*this)[18].pointer()[IJKL] = buff[BxCy*nQuad+ijkl];
            // (lk|ji)
            (*this)[18].pointer()[LKJI] = buff[ByCx*nQuad+ijkl];
  
  
            // ∇B_x∇C_z(ij|kl)
            // (ij|kl)
            (*this)[19].pointer()[IJKL] = buff[BxCz*nQuad+ijkl];
            // (lk|ji)
            (*this)[19].pointer()[LKJI] = buff[BzCx*nQuad+ijkl];
  
  
  	    // ∇B_y∇C_y(ij|kl)
            // (ij|kl)
            (*this)[20].pointer()[IJKL] = buff[ByCy*nQuad+ijkl];
            // (lk|ji)
            (*this)[20].pointer()[LKJI] = buff[ByCy*nQuad+ijkl];
  
  
  	    // ∇B_y∇C_z(ij|kl)
            // (ij|kl)
            (*this)[21].pointer()[IJKL] = buff[ByCz*nQuad+ijkl];
            // (lk|ji)
            (*this)[21].pointer()[LKJI] = buff[BzCy*nQuad+ijkl];
  
  
            // ∇B_z∇C_z(ij|kl)
            // (ij|kl)
            (*this)[22].pointer()[IJKL] = buff[BzCz*nQuad+ijkl];
            // (lk|ji)
            (*this)[22].pointer()[LKJI] = buff[BzCz*nQuad+ijkl];
  
  
  	  ijkl++;
          }; // ijkl loop
        }; // s4
        }; // s3
        }; // s2
        }; // s1
  
      }; // omp region
  
      auto durERIGaunt = tock(topERIGaunt);
      //std::cout << "Libcint-ERI-Gaunt duration   = " << durERIGaunt << std::endl;
  
      memManager_.free(cacheAll, buffAll, env, bas, atm);
  
  
#ifdef __DEBUGERI__
  
      std::cout << std::scientific << std::setprecision(16);
  
      std::cout << "ERI04-07: ∇B∙∇C(ab|cd)  ∇Bx∇C(ab|cd)-X  ∇Bx∇C(ab|cd)-Y  ∇Bx∇C(ab|cd)-Z" << std::endl;
      for(auto i = 0ul; i < NB; i++)
      for(auto j = 0ul; j < NB; j++)
      for(auto k = 0ul; k < NB; k++)
      for(auto l = 0ul; l < NB; l++){
        std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
        std::cout << (*this)[4](i, j, k, l);
        std::cout << "   ";
        std::cout << (*this)[5](i, j, k, l);
        std::cout << "   ";
        std::cout << (*this)[6](i, j, k, l);
        std::cout << "   ";
        std::cout << (*this)[7](i, j, k, l) << std::endl;
      };
  
      std::cout << "ERI08: ∇B_x∇C_y(ij|kl) + ∇B_y∇C_x(ij|kl)" << std::endl;
      for(auto i = 0ul; i < NB; i++)
      for(auto j = 0ul; j < NB; j++)
      for(auto k = 0ul; k < NB; k++)
      for(auto l = 0ul; l < NB; l++){
        std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
        std::cout << (*this)[8](i, j, k, l) << std::endl;
      };
  
      std::cout << "ERI09: ∇B_y∇C_x(ij|kl)" << std::endl;
      for(auto i = 0ul; i < NB; i++)
      for(auto j = 0ul; j < NB; j++)
      for(auto k = 0ul; k < NB; k++)
      for(auto l = 0ul; l < NB; l++){
        std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
        std::cout << (*this)[9](i, j, k, l) << std::endl;
      };
  
      std::cout << "ERI10: ∇B_x∇C_z(ij|kl) + ∇B_z∇C_x(ij|kl)" << std::endl;
      for(auto i = 0ul; i < NB; i++)
      for(auto j = 0ul; j < NB; j++)
      for(auto k = 0ul; k < NB; k++)
      for(auto l = 0ul; l < NB; l++){
        std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
        std::cout << (*this)[10](i, j, k, l) << std::endl;
      };
  
      std::cout << "ERI11: ∇B_z∇C_x(ij|kl)" << std::endl;
      for(auto i = 0ul; i < NB; i++)
      for(auto j = 0ul; j < NB; j++)
      for(auto k = 0ul; k < NB; k++)
      for(auto l = 0ul; l < NB; l++){
        std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
        std::cout << (*this)[11](i, j, k, l) << std::endl;
      };
  
      std::cout << "ERI12: ∇B_y∇C_z(ij|kl) + ∇B_z∇C_y(ij|kl)" << std::endl;
      for(auto i = 0ul; i < NB; i++)
      for(auto j = 0ul; j < NB; j++)
      for(auto k = 0ul; k < NB; k++)
      for(auto l = 0ul; l < NB; l++){
        std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
        std::cout << (*this)[12](i, j, k, l) << std::endl;
      };
  
      std::cout << "ERI13: ∇B_z∇C_y(ij|kl)" << std::endl;
      for(auto i = 0ul; i < NB; i++)
      for(auto j = 0ul; j < NB; j++)
      for(auto k = 0ul; k < NB; k++)
      for(auto l = 0ul; l < NB; l++){
        std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
        std::cout << (*this)[13](i, j, k, l) << std::endl;
      };
  
      std::cout << "ERI14: - ∇B_x∇C_x(ij|kl) - ∇B_y∇C_y(ij|kl) + ∇B_z∇C_z(ij|kl)" << std::endl;
      for(auto i = 0ul; i < NB; i++)
      for(auto j = 0ul; j < NB; j++)
      for(auto k = 0ul; k < NB; k++)
      for(auto l = 0ul; l < NB; l++){
        std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
        std::cout << (*this)[14](i, j, k, l) << std::endl;
      };
  
      std::cout << "ERI15: ∇B_x∇C_x(ij|kl) - ∇B_y∇C_y(ij|kl) - ∇B_z∇C_z(ij|kl)" << std::endl;
      for(auto i = 0ul; i < NB; i++)
      for(auto j = 0ul; j < NB; j++)
      for(auto k = 0ul; k < NB; k++)
      for(auto l = 0ul; l < NB; l++){
        std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
        std::cout << (*this)[15](i, j, k, l) << std::endl;
      };
  
      std::cout << "ERI16: - ∇B_x∇C_x(ij|kl) + ∇B_y∇C_y(ij|kl) - ∇B_z∇C_z(ij|kl)" << std::endl;
      for(auto i = 0ul; i < NB; i++)
      for(auto j = 0ul; j < NB; j++)
      for(auto k = 0ul; k < NB; k++)
      for(auto l = 0ul; l < NB; l++){
        std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
        std::cout << (*this)[16](i, j, k, l) << std::endl;
      };
  
      std::cout << "ERI17: ∇B_x∇C_x(ij|kl)" << std::endl;
      for(auto i = 0ul; i < NB; i++)
      for(auto j = 0ul; j < NB; j++)
      for(auto k = 0ul; k < NB; k++)
      for(auto l = 0ul; l < NB; l++){
        std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
        std::cout << (*this)[17](i, j, k, l) << std::endl;
      };
  
      std::cout << "ERI18: ∇B_x∇C_y(ij|kl)" << std::endl;
      for(auto i = 0ul; i < NB; i++)
      for(auto j = 0ul; j < NB; j++)
      for(auto k = 0ul; k < NB; k++)
      for(auto l = 0ul; l < NB; l++){
        std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
        std::cout << (*this)[18](i, j, k, l) << std::endl;
      };
  
      std::cout << "ERI19: ∇B_x∇C_z(ij|kl)" << std::endl;
      for(auto i = 0ul; i < NB; i++)
      for(auto j = 0ul; j < NB; j++)
      for(auto k = 0ul; k < NB; k++)
      for(auto l = 0ul; l < NB; l++){
        std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
        std::cout << (*this)[19](i, j, k, l) << std::endl;
      };
  
      std::cout << "ERI20: ∇B_y∇C_y(ij|kl)" << std::endl;
      for(auto i = 0ul; i < NB; i++)
      for(auto j = 0ul; j < NB; j++)
      for(auto k = 0ul; k < NB; k++)
      for(auto l = 0ul; l < NB; l++){
        std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
        std::cout << (*this)[20](i, j, k, l) << std::endl;
      };
  
      std::cout << "ERI21: ∇B_y∇C_z(ij|kl)" << std::endl;
      for(auto i = 0ul; i < NB; i++)
      for(auto j = 0ul; j < NB; j++)
      for(auto k = 0ul; k < NB; k++)
      for(auto l = 0ul; l < NB; l++){
        std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
        std::cout << (*this)[21](i, j, k, l) << std::endl;
      };
  
      std::cout << "ERI22: ∇B_z∇C_z(ij|kl)" << std::endl;
      for(auto i = 0ul; i < NB; i++)
      for(auto j = 0ul; j < NB; j++)
      for(auto k = 0ul; k < NB; k++)
      for(auto l = 0ul; l < NB; l++){
        std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
        std::cout << (*this)[22](i, j, k, l) << std::endl;
      };

#endif
  
    } // Gaunt

#if 0
    prettyPrintSmart(std::cout,"Rank-2 ERI00 ∇∇(ab|cd)",(*this)[0].pointer(),    NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI01 ∇∇(ab|cd)",(*this)[1].pointer(),       NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI02 ∇∇(ab|cd)",(*this)[2].pointer(),       NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI03 ∇∇(ab|cd)",(*this)[3].pointer(),       NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI04 ∇∇(ab|cd)",(*this)[4].pointer(),  NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI05 ∇∇(ab|cd)",(*this)[5].pointer(),  NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI06 ∇∇(ab|cd)",(*this)[6].pointer(),  NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI07 ∇∇(ab|cd)",(*this)[7].pointer(),  NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI08 ∇∇(ab|cd)",(*this)[8].pointer(),  NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI09 ∇∇(ab|cd)",(*this)[9].pointer(),  NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI10 ∇∇(ab|cd)",(*this)[10].pointer(), NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI11 ∇∇(ab|cd)",(*this)[11].pointer(), NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI12 ∇∇(ab|cd)",(*this)[12].pointer(), NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI13 ∇∇(ab|cd)",(*this)[13].pointer(), NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI14 ∇∇(ab|cd)",(*this)[14].pointer(), NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI15 ∇∇(ab|cd)",(*this)[15].pointer(), NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI16 ∇∇(ab|cd)",(*this)[16].pointer(), NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI17 ∇∇(ab|cd)",(*this)[17].pointer(), NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI18 ∇∇(ab|cd)",(*this)[18].pointer(), NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI19 ∇∇(ab|cd)",(*this)[19].pointer(), NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI20 ∇∇(ab|cd)",(*this)[20].pointer(), NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI21 ∇∇(ab|cd)",(*this)[21].pointer(), NB*NB,NB*NB,NB*NB);
    prettyPrintSmart(std::cout,"Rank-2 ERI22 ∇∇(ab|cd)",(*this)[22].pointer(), NB*NB,NB*NB,NB*NB);
#endif

  }; // InCore4indexRelERI<double>::computeERICINT





  template <>
  void InCore4indexTPI<dcomplex>::computeERIGCCINT(BasisSet&, Molecule&,
      EMPerturbation&, OPERATOR, const HamiltonianOptions&) {
    CErr("Only real GTOs are allowed",std::cout);
  };

  template <>
  void InCore4indexTPI<double>::computeERIGCCINT(BasisSet &originalBasisSet, Molecule &molecule_,
      EMPerturbation&, OPERATOR, const HamiltonianOptions&) {

    if (originalBasisSet.forceCart)
      CErr("Libcint + cartesian GTO NYI.");

    BasisSet basisSet_(originalBasisSet);

    std::vector<libint2::Shell> shells;

    shells.push_back(*basisSet_.shells.begin());

    size_t buffSize = shells.back().size();
    size_t countExpCoef = shells.back().alpha.size() * 2;

    for (auto it = ++basisSet_.shells.begin(); it != basisSet_.shells.end(); it++) {

      if (shells.back().O == it->O and
          shells.back().alpha == it->alpha and
          shells.back().contr[0].l == it->contr[0].l) {

        shells.back().contr.push_back(it->contr[0]);
        countExpCoef += shells.back().alpha.size();

      } else {
        shells.push_back(*it);
        countExpCoef += shells.back().alpha.size() * 2;
      }

      buffSize = std::max(buffSize, shells.back().size());

    }

    basisSet_.shells = shells;

    basisSet_.update(false);

    int nAtoms = molecule_.nAtoms;
    int nShells = basisSet_.nShell;
    int iAtom, iShell, off;

    // ATM_SLOTS = 6; BAS_SLOTS = 8;
    int *atm = memManager_.template malloc<int>(nAtoms * ATM_SLOTS);
    int *bas = memManager_.template malloc<int>(nShells * BAS_SLOTS);
    double *env = memManager_.template malloc<double>(PTR_ENV_START + nAtoms*3 + countExpCoef);
    double sNorm;

    off = PTR_ENV_START; // = 20

    for(iAtom = 0; iAtom < nAtoms; iAtom++) {

      atm[CHARGE_OF + ATM_SLOTS * iAtom] = molecule_.atoms[iAtom].atomicNumber;
      atm[PTR_COORD + ATM_SLOTS * iAtom] = off;
      env[off + 0] = molecule_.atoms[iAtom].coord[0]; // x (Bohr)
      env[off + 1] = molecule_.atoms[iAtom].coord[1]; // y (Bohr)
      env[off + 2] = molecule_.atoms[iAtom].coord[2]; // z (Bohr)
      off += 3;

    }

    for(iShell = 0; iShell < nShells; iShell++) {

      int nContr = basisSet_.shells[iShell].contr.size();

      bas[ATOM_OF  + BAS_SLOTS * iShell]  = basisSet_.mapSh2Cen[iShell];
      bas[ANG_OF   + BAS_SLOTS * iShell]  = basisSet_.shells[iShell].contr[0].l;
      bas[NPRIM_OF + BAS_SLOTS * iShell]  = basisSet_.shells[iShell].alpha.size();
      bas[NCTR_OF  + BAS_SLOTS * iShell]  = nContr;
      bas[PTR_EXP  + BAS_SLOTS * iShell]  = off;

      for(int iPrim=0; iPrim<basisSet_.shells[iShell].alpha.size(); iPrim++)
        env[off + iPrim] = basisSet_.shells[iShell].alpha[iPrim];

      off +=basisSet_.shells[iShell].alpha.size();

      bas[PTR_COEFF+ BAS_SLOTS * iShell] = off;

      // Spherical GTO normalization constant missing in Libcint
      sNorm = 2.0*std::sqrt(M_PI)/std::sqrt(2.0*basisSet_.shells[iShell].contr[0].l+1.0);
      for (size_t i = 0; i < nContr; i++) {
        for(int iCoeff=0; iCoeff<basisSet_.shells[iShell].alpha.size(); iCoeff++){
          env[off + iCoeff] = basisSet_.shells[iShell].contr[i].coeff[iCoeff]*sNorm;
        }
        off += basisSet_.shells[iShell].alpha.size();
      }

    }

    int cache_size = 0;
    for (int i = 0; i < nShells; i++) {
      int n, shls[4]{i,i,i,i};
      if (basisSet_.forceCart) {
        n = int2e_cart(nullptr, nullptr, shls, atm, nAtoms, bas, nShells, env, nullptr, nullptr);
      } else {
        n = int2e_sph(nullptr, nullptr, shls, atm, nAtoms, bas, nShells, env, nullptr, nullptr);
      }
      cache_size = std::max(cache_size, n);
    }

    // Determine the number of OpenMP threads
    int nthreads = GetNumThreads();

    // Allocate and zero out ERIs
    size_t NB  = basisSet_.nBasis;
    size_t NB2 = NB*NB;
    size_t NB3 = NB2*NB;
    size_t NB4 = NB2*NB2;

    InCore4indexTPI<double>::clear();


    // Get threads result buffer
    int buffN4 = buffSize*buffSize*buffSize*buffSize;
    double *buffAll = memManager_.malloc<double>(buffN4*nthreads);
    double *cacheAll = memManager_.malloc<double>(cache_size*nthreads);

    std::cout<<"Using Libcint "<<std::endl;

    auto topERI4 = tick();

    #pragma omp parallel
    {
      int thread_id = GetThreadID();

      size_t n1,n2,n3,n4,i,j,k,l,ijkl,bf1,bf2,bf3,bf4;
      size_t s4_max;
      int shls[4];
      double *buff = buffAll+buffN4*thread_id;
      double *cache = cacheAll+cache_size*thread_id;

      for(size_t s1(0), bf1_s(0), s1234(0); s1 < nShells;
          bf1_s+=n1, s1++) {

        n1 = basisSet_.shells[s1].size(); // Size of Shell 1

      for(size_t s2(0), bf2_s(0); s2 <= s1; bf2_s+=n2, s2++) {

        n2 = basisSet_.shells[s2].size(); // Size of Shell 2

      for(size_t s3(0), bf3_s(0); s3 <= s1; bf3_s+=n3, s3++) {

        n3 = basisSet_.shells[s3].size(); // Size of Shell 3
        s4_max = (s1 == s3) ? s2 : s3; // Determine the unique max of Shell 4

      for(size_t s4(0), bf4_s(0); s4 <= s4_max; bf4_s+=n4, s4++, s1234++) {

        n4 = basisSet_.shells[s4].size(); // Size of Shell 4

        // Round Robbin work distribution
        #ifdef _OPENMP
        if( s1234 % nthreads != thread_id ) continue;
        #endif

        shls[0] = int(s1);
        shls[1] = int(s2);
        shls[2] = int(s3);
        shls[3] = int(s4);

        if (basisSet_.forceCart) {
          if(int2e_cart(buff, nullptr, shls, atm, nAtoms, bas, nShells, env, nullptr, cache)==0) continue;
        } else {
          if(int2e_sph(buff, nullptr, shls, atm, nAtoms, bas, nShells, env, nullptr, cache)==0) continue;
        }

        // permutational symmetry
  ijkl = 0ul;
        for(l = 0ul, bf4 = bf4_s ; l < n4; ++l, bf4++)
        for(k = 0ul, bf3 = bf3_s ; k < n3; ++k, bf3++)
        for(j = 0ul, bf2 = bf2_s ; j < n2; ++j, bf2++)
        for(i = 0ul, bf1 = bf1_s ; i < n1; ++i, bf1++)
  {

            // (12 | 34)
            (*this)(bf1, bf2, bf3, bf4) = buff[ijkl];
            // (12 | 43)
            (*this)(bf1, bf2, bf4, bf3) = buff[ijkl];
            // (21 | 34)
            (*this)(bf2, bf1, bf3, bf4) = buff[ijkl];
            // (21 | 43)
            (*this)(bf2, bf1, bf4, bf3) = buff[ijkl];
            // (34 | 12)
            (*this)(bf3, bf4, bf1, bf2) = buff[ijkl];
            // (43 | 12)
            (*this)(bf4, bf3, bf1, bf2) = buff[ijkl];
            // (34 | 21)
            (*this)(bf3, bf4, bf2, bf1) = buff[ijkl];
            // (43 | 21)
            (*this)(bf4, bf3, bf2, bf1) = buff[ijkl];

      ijkl++;

        }; // ijkl loop

      }; // s4
      }; // s3
      }; // s2
      }; // s1
    }; // omp region

    auto durERI4 = tock(topERI4);
    std::cout << "Libcint-ERI4 duration   = " << durERI4 << std::endl;

    memManager_.free(cacheAll, buffAll, env, bas, atm);

#ifdef __DEBUGERI__
    // Debug output of the ERIs
    std::cout << std::scientific << std::setprecision(16);
    std::cout << "Libcint ERI (ab|cd)" << std::endl;
    for(auto i = 0ul; i < NB; i++)
    for(auto j = 0ul; j < NB; j++)
    for(auto k = 0ul; k < NB; k++)
    for(auto l = 0ul; l < NB; l++){
      std::cout << "(" << i << "," << j << "|" << k << "," << l << ")  ";
      std::cout << (*this)(i, j, k, l) << std::endl;
    };
#endif // __DEBUGERI__

  }; // InCore4indexRelERI<double>::computeERIGCCINT



}; // namespace ChronusQ

//#endif
