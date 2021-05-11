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

#include <integrals.hpp>
#include <particleintegrals/inhouseaointegral.hpp>
#include <cqlinalg.hpp>
#include <cqlinalg/svd.hpp>
#include <cqlinalg/blasutil.hpp>
#include <physcon.hpp>
#include <util/matout.hpp>
#include <util/timer.hpp>
#include <util/threads.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <particleintegrals/onepints/aoonepints.hpp>
#include <libcint.hpp>

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
  void OnePInts<dcomplex>::OnePDriverLibint(libint2::Operator op,
      Molecule &mol, shell_set& shells, std::vector<dcomplex*> mats,
      Particle p) {
    CErr("Only real GTOs are allowed",std::cout);
  };

  template <>
  void OnePInts<double>::OnePDriverLibint(libint2::Operator op,
      Molecule &mol, shell_set& shells, std::vector<double*> mats, 
      Particle p) {

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

    // If engine is K, prescale it by 1/m
    if (op == libint2::Operator::kinetic)
      engines[0].prescale_by(1.0 / p.mass);  

    // If engine is V, define nuclear charges (pseudo molecule is used for NEO)
    if(op == libint2::Operator::nuclear){
      std::vector<std::pair<double,std::array<double,3>>> q;
      for (auto ind : mol.atomsC) // loop over classical atoms
        q.push_back( { -1.0 * p.charge * mol.atoms[ind].nucCharge, mol.atoms[ind].coord } );

      engines[0].set_params(q);
      
    }

    // for multipoles, prescale it by charge
    if (op == libint2::Operator::emultipole1 or op == libint2::Operator::emultipole2 or op == libint2::Operator::emultipole3)
      engines[0].prescale_by(-1.0 * p.charge);

    // Copy over the engines to other threads if need be
    for(size_t i = 1; i < nthreads; i++) engines[i] = engines[0];


    std::vector<
      Eigen::Map<
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>
      > 
    > matMaps;
    for( auto i = 0; i < mats.size(); i++ ) {
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

  }; // OnePInts::OnePDriver


  /**
   *  \brief A general wrapper for 1-e (2 index) integral evaluation by libcint
   *
   *
   *  \param [in] op     Operator for which to calculate the 1-e integrals
   *
   */
  template <>
  void OnePInts<dcomplex>::OnePDriverLibcint(OPERATOR, const Molecule&,
      const BasisSet&, const Particle&, bool finiteWidthNuc) {
    CErr("Only real GTOs are allowed",std::cout);
  };

  template <>
  void OnePInts<double>::OnePDriverLibcint(OPERATOR op,
      const Molecule &molecule_, const BasisSet &originalBasisSet,
      const Particle &p, bool finiteWidthNuc) {

    if (originalBasisSet.forceCart)
      CErr("Libcint + cartesian GTO NYI.");

    BasisSet basisSet_ = originalBasisSet.groupGeneralContractionBasis();

    size_t buffSize = std::max_element(basisSet_.shells.begin(),
                                       basisSet_.shells.end(),
                                       [](libint2::Shell &a, libint2::Shell &b) {
                                         return a.size() < b.size();
                                       })->size();
    buffSize *= buffSize;

    int nAtoms = molecule_.nAtoms;
    int nShells = basisSet_.nShell;

    // ATM_SLOTS = 6; BAS_SLOTS = 8;
    int *atm = memManager_.template malloc<int>(nAtoms * ATM_SLOTS);
    int *bas = memManager_.template malloc<int>(nShells * BAS_SLOTS);
    double *env = memManager_.template malloc<double>(basisSet_.getLibcintEnvLength(molecule_));


    basisSet_.setLibcintEnv(molecule_, atm, bas, env);

    // Determine the number of OpenMP threads
    int nthreads = GetNumThreads();

    double *buffAll = memManager_.template malloc<double>(buffSize*nthreads);

    clear();
    Eigen::Map<
      Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>
    > matMap(mat_.pointer(), NB, NB);


    #pragma omp parallel
    {
      int thread_id = GetThreadID();
      size_t n1,n2;
      int shls[2];
      double *buff = buffAll + buffSize * thread_id;

      // Loop over unique shell pairs
      for(size_t s1(0), bf1_s(0), s12(0); s1 < basisSet_.nShell; bf1_s+=n1, s1++){
        n1 = basisSet_.shells[s1].size(); // Size of Shell 1
      for(size_t s2(0), bf2_s(0); s2 <= s1; bf2_s+=n2, s2++, s12++) {
        n2 = basisSet_.shells[s2].size(); // Size of Shell 2

        // Round Robbin work distribution
        #ifdef _OPENMP
        if( s12 % nthreads != thread_id ) continue;
        #endif

        shls[0] = int(s2);
        shls[1] = int(s1);

        // Compute the integrals
        if(cint1e_kin_sph(buff, shls, atm, nAtoms, bas, nShells, env)==0) continue;

        // Place integral blocks into their respective matricies
        Eigen::Map<
          const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,
            Eigen::RowMajor>
        > bufMat(buff, n1, n2);

        matMap.block(bf1_s,bf2_s,n1,n2) = bufMat.template cast<double>();

      } // Loop over s2 <= s1
      } // Loop over s1

    } // end OpenMP context


    // Symmetrize the matricies
    matMap = matMap.template selfadjointView<Eigen::Lower>();


    // If engine is K, prescale it by 1/m
    if (op == KINETIC)
      mat_ *= 1.0 / p.mass;

//    // If engine is V, define nuclear charges (pseudo molecule is used for NEO)
//    if(op == NUCLEAR_POTENTIAL){
//      std::vector<std::pair<double,std::array<double,3>>> q;
//      for (auto ind : mol.atomsC) // loop over classical atoms
//        q.push_back( { -1.0 * p.charge * mol.atoms[ind].nucCharge, mol.atoms[ind].coord } );

//      engines[0].set_params(q);

//    }

    // for multipoles, prescale it by charge
//    if (op == libint2::Operator::emultipole1 or op == libint2::Operator::emultipole2 or op == libint2::Operator::emultipole3)
//      engines[0].prescale_by(-1.0 * p.charge);

  }; // OnePInts::OnePDriver


  template <>
  template <size_t NOPER, bool SYMM, typename F>
  void OnePInts<double>::OnePDriverLocal(
      const F &obFunc, shell_set& shells, std::vector<double*> mats) {

    // Determine the number of basis functions for the passed shell set
    size_t NB = std::accumulate(shells.begin(),shells.end(),0,
      [](size_t init, libint2::Shell &sh) -> size_t {
        return init + sh.size();
      }
    );

    size_t NBSQ = NB*NB;

    std::vector<
      Eigen::Map<
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> 
      > 
    > matMaps;

    for( auto i = 0; i < mats.size(); i++ ) {
      std::fill_n(mats[i],NBSQ,0.);  
      matMaps.emplace_back(mats[i],NB,NB);
    }

//    if(basisType == REAL_GTO)
      // pre compute all the shellpair data
//      auto pair_to_use = genShellPairs(shells,std::log(std::numeric_limits<double>::lowest()));
    
    // Loop over unique shell pairs



    auto start  = tick();

    // Determine the number of OpenMP threads
    int nthreads = GetNumThreads();
    #pragma omp parallel  
    {
    int thread_id = GetThreadID();

    size_t n1,n2;
    for(size_t s1(0), bf1_s(0), s12(0); s1 < shells.size(); bf1_s+=n1, s1++){ 
      n1 = shells[s1].size(); // Size of Shell 1
    for(size_t s2(0), bf2_s(0); s2 <= s1; bf2_s+=n2, s2++, s12++) {
      n2 = shells[s2].size(); // Size of Shell 2


        // Round Robbin work distribution
        #ifdef _OPENMP
        if( s12 % nthreads != thread_id ) continue;
        #endif


      libint2::ShellPair pair_to_use;
      pair_to_use.init(shells[s1],shells[s2],-1000);

      auto buff = obFunc(pair_to_use, shells[s1],shells[s2]);

/*
#pragma omp critical
{
      std::cout<<"s1= "<<s1<<" s2 = "<<s2<<std::endl;
      for ( int elements = 0 ; elements < buff[0].size() ; elements++ ) {
        std::cout<<"buff["<<elements<<"]= "<<buff[0][elements]<<std::endl;
      } 
}  // critical 
*/
      assert(buff.size() == NOPER);

 /*     
      // Place integral blocks into their respective matricies
      for ( int iidx = 0 ; iidx < n1 ; iidx++ ) {
        for ( int jidx = 0 ; jidx < n2 ; jidx++ ) {
          for ( int icomp = 0 ; icomp < NOPER ; icomp++ ) {
            mats[icomp][(iidx+bf1_s)*NB+bf2_s+jidx] = buff[icomp][iidx*n2+jidx];
            std::cout<<"iidx+bf1_s= "<<iidx+bf1_s<<"  bf2_s+jidx= "<<bf2_s+jidx<<" elements = "<<iidx*n2+jidx<<" value "<<buff[icomp][iidx*n2+jidx]<<mats[icomp][(iidx+bf1_s)*NB+bf2_s+jidx]<<std::endl;
          }
        }
      }
*/

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
    }   // omp
 
    double end = tock(start);
    //std::cout<<"onee driver time= "<<end<<std::endl;

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

  }; // OnePInts::OnePDriverLocal

  template <>
  void OnePInts<double>::computeAOInts(BasisSet &basis, Molecule &mol,
      EMPerturbation&, OPERATOR op, const HamiltonianOptions &options) {

    if (options.basisType != REAL_GTO)
      CErr("Only Real GTOs are allowed in OnePInts<double>",std::cout);
    if (op == NUCLEAR_POTENTIAL and
        (options.OneEScalarRelativity or options.OneESpinOrbit))
      CErr("Relativistic nuclear potential is not implemented in OnePInts,"
           " they are implemented in OnePRelInts",std::cout);

    std::vector<double*> tmp(1, pointer());

    switch (op) {
    case OVERLAP:
      OnePDriverLibint(libint2::Operator::overlap,mol,basis.shells,tmp,options.particle);
      break;
    case KINETIC:
//      OnePDriverLibint(libint2::Operator::kinetic,mol,basis.shells,tmp,options.particle);
      OnePDriverLibcint(op, mol, basis, options.particle);
      //output(std::cout,"",true);
      break;
    case NUCLEAR_POTENTIAL:
      if (options.finiteWidthNuc) {
        OnePDriverLocal<1,true>(
            [&](libint2::ShellPair& pair, libint2::Shell& sh1,
                libint2::Shell& sh2) -> std::vector<std::vector<double>> {
              return RealGTOIntEngine::computePotentialV(mol.chargeDist,
                  pair,sh1,sh2,mol);
              }, basis.shells,tmp);
      }
      else
        OnePDriverLibint(libint2::Operator::nuclear,mol,basis.shells,tmp,options.particle);
      break;
    case ELECTRON_REPULSION:
      CErr("Electron repulsion integrals are not implemented in OnePInts,"
           " they are implemented in TwoEInts",std::cout);
      break;
    case LEN_ELECTRIC_MULTIPOLE:
    case VEL_ELECTRIC_MULTIPOLE:
    case MAGNETIC_MULTIPOLE:
      CErr("Requested operator is not implemented in OnePInts,"
           " it is implemented in MultipoleInts",std::cout);
      break;
    default:
      CErr("Requested operator is not implemented in OneEInts.");
      break;
    }

  };

  template <>
  void VectorInts<double>::computeAOInts(BasisSet &basis, Molecule &mol,
      EMPerturbation&, OPERATOR op, const HamiltonianOptions &options) {
    if (options.basisType != REAL_GTO)
      CErr("Only Real GTOs are allowed in VectorInts<double>",std::cout);
    if (options.OneEScalarRelativity or options.OneESpinOrbit)
      CErr("Relativistic multipole integrals are implemented in OnePRelInts",std::cout);

    switch (op) {
    case OVERLAP:
    case KINETIC:
    case NUCLEAR_POTENTIAL:
      CErr("Requested operator is not implemented in VectorInts,"
           " it is implemented in OneEInts",std::cout);
      break;
    case ELECTRON_REPULSION:
      CErr("Electron repulsion integrals are not implemented in VectorInts,"
           " they are implemented in TwoEInts",std::cout);
      break;
    case LEN_ELECTRIC_MULTIPOLE:
      CErr("Len Electric multipole integrals are not implemented in VectorInts,"
           " they are implemented in MultipoleInts",std::cout);
      break;
    case VEL_ELECTRIC_MULTIPOLE:
      switch (order()) {
      case 1:
        OnePInts<double>::OnePDriverLocal<3,false>(
            std::bind(&RealGTOIntEngine::computeEDipoleE1_vel,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3),
            basis.shells, pointers());
        break;
      case 2:
        OnePInts<double>::OnePDriverLocal<6,false>(
            std::bind(&RealGTOIntEngine::computeEQuadrupoleE2_vel,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3),
            basis.shells, pointers());
        break;
      case 3:
        OnePInts<double>::OnePDriverLocal<10,false>(
            std::bind(&RealGTOIntEngine::computeEOctupoleE3_vel,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3),
            basis.shells, pointers());
        break;
      default:
        CErr("Requested operator is NYI in VectorInts.",std::cout);
        break;
      }
      break;
    case MAGNETIC_MULTIPOLE:
      switch (order()) {
      case 1:
        OnePInts<double>::OnePDriverLocal<3,false>(
            std::bind(&RealGTOIntEngine::computeAngularL,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3),
            basis.shells, pointers());
        break;
      case 2:
        OnePInts<double>::OnePDriverLocal<9,false>(
            std::bind(&RealGTOIntEngine::computeMQuadrupoleM2_vel,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3),
            basis.shells, pointers());
        break;
      default:
        CErr("Requested operator is NYI in VectorInts.",std::cout);
        break;
      }
      break;
    default:
      CErr("Requested operator is not implemented in VectorInts.");
      break;
    }

  };

  template <>
  void MultipoleInts<double>::computeAOInts(BasisSet &basis, Molecule &mol,
      EMPerturbation &emPert, OPERATOR op, const HamiltonianOptions &options) {
    if (options.basisType != REAL_GTO)
      CErr("Only Real GTOs are allowed in MultipoleInts<double>",std::cout);
    if (options.OneEScalarRelativity or options.OneESpinOrbit)
      CErr("Relativistic multipole integrals are implemented in OnePRelInts",std::cout);

    std::vector<double*> _multipole(1, nullptr);
    libint2::Operator libOp;

    switch (op) {
    case OVERLAP:
    case KINETIC:
    case NUCLEAR_POTENTIAL:
      CErr("Requested operator is not implemented in MultipoleInts,"
           " it is implemented in OnePInts",std::cout);
      break;
    case ELECTRON_REPULSION:
      CErr("Electron repulsion integrals are not implemented in MultipoleInts,"
           " they are implemented in TwoEInts",std::cout);
      break;
    case LEN_ELECTRIC_MULTIPOLE:
      try { _multipole[0] = memManager().malloc<double>(nBasis()*nBasis()); }
      catch(...) {
        std::cout << std::fixed;
        std::cout << "Insufficient memory for the full INTS tensor ("
                  << (nBasis()*nBasis()/1e9) * sizeof(double) << " GB)" << std::endl;
        std::cout << std::endl << this->memManager() << std::endl;
        CErr();
      }
      if (highOrder() >= 1) {
        std::copy_n(dipolePointers().begin(), 3, std::back_inserter(_multipole));
        libOp = libint2::Operator::emultipole1;
        if (highOrder() >= 2) {
          std::copy_n(quadrupolePointers().begin(), 6, std::back_inserter(_multipole));
          libOp = libint2::Operator::emultipole2;
          if (highOrder() == 3) {
            std::copy_n(octupolePointers().begin(), 10, std::back_inserter(_multipole));
            libOp = libint2::Operator::emultipole3;
          } else
            CErr("Requested operator is NYI in MultipoleInts.",std::cout);
        }
      }
      OnePInts<double>::OnePDriverLibint(libOp,mol,basis.shells,_multipole,options.particle);
      memManager().free(_multipole[0]);
      break;
    case VEL_ELECTRIC_MULTIPOLE:
    case MAGNETIC_MULTIPOLE:
      for (VectorInts<double> &vInts: components_) {
        vInts.computeAOInts(basis, mol, emPert, op, options);
      }
      break;
    default:
      CErr("Requested operator is not implemented in MultipoleInts.");
      break;
    }

  };

  template <>
  void OnePRelInts<double>::computeAOInts(BasisSet &basis, Molecule &mol,
      EMPerturbation&, OPERATOR op, const HamiltonianOptions &options) {
    if (options.basisType != REAL_GTO)
      CErr("Only Real GTOs are allowed in OnePRelInts<double>",std::cout);
    if (not options.OneEScalarRelativity or op != NUCLEAR_POTENTIAL)
      CErr("Only relativistic nuclear potential is implemented in OnePRelInts.",std::cout);

    std::vector<double*> _potential(1, pointer());
    if (options.finiteWidthNuc)
      OnePDriverLocal<1,true>(
          [&](libint2::ShellPair& pair, libint2::Shell& sh1,
              libint2::Shell& sh2) -> std::vector<std::vector<double>> {
            return RealGTOIntEngine::computePotentialV(mol.chargeDist,
                pair,sh1,sh2,mol);
            }, basis.shells,_potential);
    else
      OnePDriverLibint(libint2::Operator::nuclear,mol,basis.shells,_potential,options.particle);

    // Point nuclei is used when chargeDist is empty
    const std::vector<libint2::Shell> &chargeDist = options.finiteWidthNuc ?
        mol.chargeDist : std::vector<libint2::Shell>();

    std::vector<double*> _PVdP(1, scalar().pointer());
    OnePInts<double>::OnePDriverLocal<1,true>(
          [&](libint2::ShellPair& pair, libint2::Shell& sh1,
              libint2::Shell& sh2) -> std::vector<std::vector<double>> {
            return RealGTOIntEngine::computepVdotp(chargeDist,
                pair,sh1,sh2,mol);
            }, basis.shells, _PVdP);

    if (options.OneESpinOrbit and components_.size() >=4) {

      OnePInts<double>::OnePDriverLocal<3,false>(
            [&](libint2::ShellPair& pair, libint2::Shell& sh1,
                libint2::Shell& sh2) -> std::vector<std::vector<double>> {
              return RealGTOIntEngine::computeSL(chargeDist,
                  pair,sh1,sh2,mol);
              }, basis.shells, SOXYZPointers());
    }

  };

  template void Integrals<double>::computeAOOneP(
      CQMemManager&, Molecule&, BasisSet&, EMPerturbation&,
      const std::vector<std::pair<OPERATOR,size_t>>&,
      const HamiltonianOptions&);

}; // namespace ChronusQ

