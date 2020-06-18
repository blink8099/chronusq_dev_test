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

#include <aointegrals.hpp>
#include <aointegrals/giaointegrals.hpp>
#include <cqlinalg.hpp>
#include <cqlinalg/svd.hpp>
#include <cqlinalg/blasutil.hpp>
#include <physcon.hpp>
#include <util/matout.hpp>
#include <util/threads.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>

// Debug directives
//#define _DEBUGORTHO
//#define _DEBUGERI


namespace ChronusQ {

  typedef std::vector<libint2::Shell> shell_set; 

  /**
   *  \brief A general wrapper for 1-e (2 index) integral evaluation.
   *
   *  Currently computes 1-e integrals using Libint2. Shells sets are
   *  passed in order to be possibly general to the uncontracted basis.
   *  Handles all internal memory allocation including the evaluated matricies
   *  themselves
   *
   *  \param [in] op     Operator for which to calculate the 1-e integrals
   *  \param [in] shells Shell set for the integral evaluation
   *
   *  \returns    A vector of properly allocated pointers which store the
   *              1-e evaluations.
   *
   *  This function returns a vector of pointers as it sometimes makes sense
   *  to evaluate several matricies together if they are inimately related,
   *  namely the length gauge electric multipoles and the overlap.
   *
   *  z.B. op == libint2::Operator::emultipole3
   *
   *  The function will return a vector of 20 pointers in the following order
   *  { overlap, 
   *    dipole_x, dipole_y, dipole_z, 
   *    quadrupole_xx, quadrupole_xy, quadrupole_xz, quadrupole_yy,
   *      quadrupole_yz, quadrupole_zz,
   *    octupole_xxx, octupole_xxy, octupole_xxz, octupole_xyy,
   *      octupole_xyz, octupole_xzz, octupole_yyy, octupole_yyz,
   *      octupole_yzz, octupole_zzz
   *  }
   *
   *  z.B. op == libint2::Operator::kinetic
   *
   *  The function will return a vector of 1 pointer
   *
   *  { kinetic }
   */ 
  template <>
  std::vector<dcomplex*> 
  AOIntegrals<dcomplex>::OneEDriverLibint(
    libint2::Operator op, shell_set& shells) {
    CErr("Only real GTOs are allowed",std::cout);
  };

  template <>
  std::vector<double*> 
  AOIntegrals<double>::OneEDriverLibint(
    libint2::Operator op, shell_set& shells) {

    // Determine the number of basis functions for the passed shell set
    size_t NB = std::accumulate(shells.begin(),shells.end(),0,
      [](size_t init, libint2::Shell &sh) -> size_t {
        return init + sh.size();
      }
    );

    size_t NBSQ = NB*NB;


    // Determine the maximum angular momentum of the passed shell set
    int maxL = std::max_element(shells.begin(), shells.end(),
      [](libint2::Shell &sh1, libint2::Shell &sh2){
        return sh1.contr[0].l < sh2.contr[0].l;
      }
    )->contr[0].l;

    // Determine the maximum contraction depth of the passed shell set
    int maxPrim = std::max_element(shells.begin(), shells.end(),
      [](libint2::Shell &sh1, libint2::Shell &sh2){
        return sh1.alpha.size() < sh2.alpha.size();
      }
    )->alpha.size();

    // Determine the number of OpenMP threads
    int nthreads = GetNumThreads();

    // Create a vector of libint2::Engines for possible threading
    std::vector<libint2::Engine> engines(nthreads);

    // Initialize the first engine for the integral evaluation
    engines[0] = libint2::Engine(op,maxPrim,maxL,0);
    engines[0].set_precision(0.0);


    // If engine is V, define nuclear charges
    if(op == libint2::Operator::nuclear){
      std::vector<std::pair<double,std::array<double,3>>> q;
      for(auto &atom : molecule_.atoms)
        q.push_back( { static_cast<double>(atom.atomicNumber), atom.coord } );

      engines[0].set_params(q);
    }

    // Copy over the engines to other threads if need be
    for(size_t i = 1; i < nthreads; i++) engines[i] = engines[0];


    // Determine the number of operators
    std::vector<double*> mats( engines[0].results().size() );

    std::vector<
      Eigen::Map<
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>
      > 
    > matMaps;
    for( auto i = 0; i < mats.size(); i++ ) {
      mats[i] = memManager_.template malloc<double>(NBSQ);
      std::fill_n(mats[i],NBSQ,0.);
      matMaps.emplace_back(mats[i],NB,NB);
    }


    #pragma omp parallel
    {
      int thread_id = GetThreadID();

      const auto& buf_vec = engines[thread_id].results();
      size_t n1,n2;

      // Loop over unique shell pairs
      for(size_t s1(0), bf1_s(0), s12(0); s1 < shells.size(); bf1_s+=n1, s1++){ 
        n1 = shells[s1].size(); // Size of Shell 1
      for(size_t s2(0), bf2_s(0); s2 <= s1; bf2_s+=n2, s2++, s12++) {
        n2 = shells[s2].size(); // Size of Shell 2

        // Round Robbin work distribution
        #ifdef _OPENMP
        if( s12 % nthreads != thread_id ) continue;
        #endif

        // Compute the integrals       
        engines[thread_id].compute(shells[s1],shells[s2]);

        // If the integrals were screened, move on to the next batch
        if(buf_vec[0] == nullptr) continue;

        // Place integral blocks into their respective matricies
        for(auto iMat = 0; iMat < buf_vec.size(); iMat++){
          Eigen::Map<
            const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,
              Eigen::RowMajor>>
            bufMat(buf_vec[iMat],n1,n2);

          matMaps[iMat].block(bf1_s,bf2_s,n1,n2) = bufMat.template cast<double>();
        }

      } // Loop over s2 <= s1
      } // Loop over s1

    } // end OpenMP context


    // Symmetrize the matricies 
    for(auto nMat = 0; nMat < matMaps.size(); nMat++) 
      matMaps[nMat] = matMaps[nMat].template selfadjointView<Eigen::Lower>();


    return mats;

  }; // AOIntegrals::OneEDriver


  template <>
  template <size_t NOPER, bool SYMM, typename F>
  std::vector<dcomplex*>
  AOIntegrals<dcomplex>::OneEDriverLocalGTO(const F &obFunc, shell_set& shells) {
    CErr("Only real GTOs are allowed",std::cout);
  };

  template <>
  template <size_t NOPER, bool SYMM, typename F>
  std::vector<double*>
  AOIntegrals<double>::OneEDriverLocalGTO(const F &obFunc, shell_set& shells) {

    // Determine the number of basis functions for the passed shell set
    size_t NB = std::accumulate(shells.begin(),shells.end(),0,
      [](size_t init, libint2::Shell &sh) -> size_t {
        return init + sh.size();
      }
    );

    size_t NBSQ = NB*NB;

    // Determine the maximum angular momentum of the passed shell set
    int maxL = std::max_element(shells.begin(), shells.end(),
      [](libint2::Shell &sh1, libint2::Shell &sh2){
        return sh1.contr[0].l < sh2.contr[0].l;
      }
    )->contr[0].l;

    // Determine the maximum contraction depth of the passed shell set
    int maxPrim = std::max_element(shells.begin(), shells.end(),
      [](libint2::Shell &sh1, libint2::Shell &sh2){
        return sh1.alpha.size() < sh2.alpha.size();
      }
    )->alpha.size();


    // Determine the number of operators
    std::vector<double*> mats(NOPER); 

    std::vector<
      Eigen::Map<
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> 
      > 
    > matMaps;

    for( auto i = 0; i < mats.size(); i++ ) {
      mats[i] = memManager_.template malloc<double>(NBSQ); 
      std::fill_n(mats[i],NBSQ,0.);  
      matMaps.emplace_back(mats[i],NB,NB);
    }

//    if(basisType == REAL_GTO)
      // pre compute all the shellpair data
//      auto pair_to_use = genShellPairs(shells,std::log(std::numeric_limits<double>::lowest()));
    
    size_t n1,n2;
    // Loop over unique shell pairs
    for(size_t s1(0), bf1_s(0), s12(0); s1 < shells.size(); bf1_s+=n1, s1++){ 
      n1 = shells[s1].size(); // Size of Shell 1
    for(size_t s2(0), bf2_s(0); s2 <= s1; bf2_s+=n2, s2++, s12++) {
      n2 = shells[s2].size(); // Size of Shell 2

      libint2::ShellPair pair_to_use;
      pair_to_use.init(shells[s1],shells[s2],-1000);

      auto buff = obFunc(pair_to_use, shells[s1],shells[s2]);

      assert(buff.size() == NOPER);

      // Place integral blocks into their respective matricies
      for(auto iMat = 0; iMat < buff.size(); iMat++){
        Eigen::Map<
          const Eigen::Matrix<
            double,
            Eigen::Dynamic,Eigen::Dynamic,  
            Eigen::RowMajor>>
          bufMat(&buff[iMat][0],n1,n2);

        matMaps[iMat].block(bf1_s,bf2_s,n1,n2) = bufMat.template cast<double>();
      }

    } // Loop over s2 <= s1
    } // Loop over s1



    // Symmetrize the matricies 
    // XXX: USES EIGEN
    // FIXME: not SYMM -> creates a temporary
    for(auto nMat = 0; nMat < matMaps.size(); nMat++) {
      if(SYMM) matMaps[nMat] = matMaps[nMat].template selfadjointView<Eigen::Lower>();
      else {
        for(auto i = 0  ; i < NB; ++i)
        for(auto j = i+1; j < NB; ++j)
          matMaps[nMat](i,j) = - matMaps[nMat](j,i);
      }
    }

    // here the symmetrization is still real 
    return mats;

  }; // AOIntegrals::OneEDriverLocalGTO

  /**
   *  \brief Allocate, compute  and store the 1-e integrals + 
   *  orthonormalization matricies over the given CGTO basis.
   *
   *  Computes:
   *    Overlap + length gauge Electric Multipoles
   *    Kinetic energy matrix
   *    Nuclear potential energy matrix
   *    Core Hamiltonian (T + V)
   *    Orthonormalization matricies (Lowdin / Cholesky)
   *
   */ 
  template <>
  void AOIntegrals<dcomplex>::computeAOOneEGTO(OneETerms &oneETerms) {
    CErr("Only real GTOs are allowed",std::cout);
  };
  template <>
  void AOIntegrals<double>::computeAOOneEGTO(OneETerms &oneETerms) {

    size_t NB = basisSet_.nBasis;

    if(oneETerms.coreH) {
      // Compute base 1-e integrals
      auto _multipole = 
        OneEDriverLibint(libint2::Operator::emultipole3,basisSet_.shells);

      auto _kinetic = OneEDriverLibint(libint2::Operator::kinetic,basisSet_.shells);

      // Use Libint for point nuclei, in-house for Gaussian nuclei
      auto _potential = not oneETerms.finiteWidthNuc ? 
        OneEDriverLibint(libint2::Operator::nuclear,basisSet_.shells) :
        OneEDriverLocalGTO<1,true>(
            [&](libint2::ShellPair& pair, libint2::Shell& sh1, 
                libint2::Shell& sh2) -> std::vector<std::vector<double>> { 
              return RealGTOIntEngine::computePotentialV(molecule_.chargeDist,
                  pair,sh1,sh2,molecule_);
              }, basisSet_.shells);


      auto _L = OneEDriverLocalGTO<3,false>(
                  std::bind(&RealGTOIntEngine::computeAngularL,
                            std::placeholders::_1, std::placeholders::_2,
                            std::placeholders::_3),
                  basisSet_.shells);

      auto _E1V = OneEDriverLocalGTO<3,false>(
                  std::bind(&RealGTOIntEngine::computeEDipoleE1_vel,
                            std::placeholders::_1, std::placeholders::_2,
                            std::placeholders::_3),
                  basisSet_.shells);

      auto _E2V = OneEDriverLocalGTO<6,false>(
                  std::bind(&RealGTOIntEngine::computeEQuadrupoleE2_vel,
                            std::placeholders::_1, std::placeholders::_2,
                            std::placeholders::_3),
                  basisSet_.shells);

      auto _E3V = OneEDriverLocalGTO<10,false>(
                  std::bind(&RealGTOIntEngine::computeEOctupoleE3_vel,
                            std::placeholders::_1, std::placeholders::_2,
                            std::placeholders::_3),
                  basisSet_.shells);

      auto _M2  = OneEDriverLocalGTO<9,false>(
                  std::bind(&RealGTOIntEngine::computeMQuadrupoleM2_vel,
                            std::placeholders::_1, std::placeholders::_2,
                            std::placeholders::_3),
                  basisSet_.shells);


      overlap = reinterpret_cast<double*>(_multipole[0]);
      kinetic   = reinterpret_cast<double*>(_kinetic[0]);
      potential = reinterpret_cast<double*>(_potential[0]);

      std::copy_n(_multipole.begin()+1, 3, std::back_inserter(lenElecDipole));
      std::copy_n(_multipole.begin()+4, 6, std::back_inserter(lenElecQuadrupole));
      std::copy_n(_multipole.begin()+10,10,std::back_inserter(lenElecOctupole));

      std::copy(_L.begin() ,_L.end() ,std::back_inserter(magDipole));
      std::copy(_M2.begin(),_M2.end(),std::back_inserter(magQuadrupole));

      std::copy(_E1V.begin(),_E1V.end(),std::back_inserter(velElecDipole));
      std::copy(_E2V.begin(),_E2V.end(),std::back_inserter(velElecQuadrupole));
      std::copy(_E3V.begin(),_E3V.end(),std::back_inserter(velElecOctupole));

    }

    if( oneETerms.relativistic ) {

      auto _SL = OneEDriverLocalGTO<3,false>(
            [&](libint2::ShellPair& pair, libint2::Shell& sh1, 
                libint2::Shell& sh2) -> std::vector<std::vector<double>> { 
              return RealGTOIntEngine::computeSL(molecule_.chargeDist,
                  pair,sh1,sh2,molecule_);
              }, basisSet_.shells);

      auto _PVdP = OneEDriverLocalGTO<1,true>(
            [&](libint2::ShellPair& pair, libint2::Shell& sh1, 
                libint2::Shell& sh2) -> std::vector<std::vector<double>> { 
              return RealGTOIntEngine::computepVdotp(molecule_.chargeDist,
                  pair,sh1,sh2,molecule_);
              }, basisSet_.shells);

      PVdotP    = reinterpret_cast<double*>(_PVdP[0]);
      PVcrossP  = {reinterpret_cast<double*>(_SL[0]),
                   reinterpret_cast<double*>(_SL[1]),
                   reinterpret_cast<double*>(_SL[2])};
    }

#if X2C_DEBUG_LEVEL >= 3
    prettyPrintSmart(std::cout,"Overlap",_overlap[0],NB,NB,NB);
    prettyPrintSmart(std::cout,"Kinetic",_kinetic[0],NB,NB,NB);
    prettyPrintSmart(std::cout,"Potential",_potential[0],NB,NB,NB);

    prettyPrintSmart(std::cout,"PV.P",_PVdP[0],NB,NB,NB);
    prettyPrintSmart(std::cout,"(PVxP) X",_SL[0],NB,NB,NB);
    prettyPrintSmart(std::cout,"(PVxP) Y",_SL[1],NB,NB,NB);
    prettyPrintSmart(std::cout,"(PVxP) Z",_SL[2],NB,NB,NB);
#endif


    // Save Integrals to disk
    if( savFile.exists() ) {

      std::string potentialTag = oneETerms.finiteWidthNuc ? "_FINITE_WIDTH" : "";

      savFile.safeWriteData("INTS/OVERLAP", overlap, {NB,NB});
      savFile.safeWriteData("INTS/KINETIC", kinetic, {NB,NB});
      savFile.safeWriteData("INTS/POTENTIAL" + potentialTag,potential,{NB,NB});
  
      const std::array<std::string,3> dipoleList =
        { "X","Y","Z" };
      const std::array<std::string,6> quadrupoleList =
        { "XX","XY","XZ","YY","YZ","ZZ" };
      const std::array<std::string,10> octupoleList =
        { "XXX","XXY","XXZ","XYY","XYZ","XZZ","YYY",
          "YYZ","YZZ","ZZZ" };

      // Length Gauge electric dipole
      for(auto i = 0; i < 3; i++)
        savFile.safeWriteData("INTS/ELEC_DIPOLE_LEN_" + 
          dipoleList[i], lenElecDipole[i], {NB,NB} );

      // Length Gauge electric quadrupole
      for(auto i = 0; i < 6; i++)
        savFile.safeWriteData("INTS/ELEC_QUADRUPOLE_LEN_" + 
          quadrupoleList[i], lenElecQuadrupole[i], {NB,NB} );

      // Length Gauge electric octupole
      for(auto i = 0; i < 10; i++)
        savFile.safeWriteData("INTS/ELEC_OCTUPOLE_LEN_" + 
          octupoleList[i], lenElecOctupole[i], {NB,NB} );


      // Magnetic Dipole
      for(auto i = 0; i < 3; i++)
        savFile.safeWriteData("INTS/MAG_DIPOLE_" + 
          dipoleList[i], magDipole[i], {NB,NB} );

      // Magnetic Quadrupole
      for(auto i = 0; i < 6; i++)
        savFile.safeWriteData("INTS/MAG_QUADRUPOLE_" + 
          quadrupoleList[i], magQuadrupole[i], {NB,NB} );


      // Velocity Gauge electric dipole
      for(auto i = 0; i < 3; i++)
        savFile.safeWriteData("INTS/ELEC_DIPOLE_VEL_" + 
          dipoleList[i], velElecDipole[i], {NB,NB} );

      // Velocity Gauge electric quadrupole
      for(auto i = 0; i < 6; i++)
        savFile.safeWriteData("INTS/ELEC_QUADRUPOLE_VEL_" + 
          quadrupoleList[i], velElecQuadrupole[i], {NB,NB} );

      // Velocity Gauge electric octupole
      for(auto i = 0; i < 10; i++)
        savFile.safeWriteData("INTS/ELEC_OCTUPOLE_VEL_" + 
          octupoleList[i], velElecOctupole[i], {NB,NB} );

      // FIXME: Write PVP integrals
    }

  }; // AOIntegrals<double>::computeAOOneE


  template <>
  void AOIntegrals<double>::computeAOOneE(EMPerturbation &emPert, 
      OneETerms &oneETerms) {
    computeAOOneEGTO(oneETerms);
  };

  template <>
  void AOIntegrals<dcomplex>::computeAOOneE(EMPerturbation &emPert, 
      OneETerms &oneETerms) {
    CErr("Only real GTOs are allowed",std::cout);
  };


}; // namespace ChronusQ

