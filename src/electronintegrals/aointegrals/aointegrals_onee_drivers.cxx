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

#include <integrals/impl.hpp>
#include <electronintegrals/inhouseaointegral.hpp>
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
  void OneEInts<dcomplex>::OneEDriverLibint(libint2::Operator op,
      Molecule &mol, BasisSet& basis, std::vector<dcomplex*> mats, size_t deriv) {
    CErr("Only real GTOs are allowed",std::cout);
  };

  template <>
  void OneEInts<double>::OneEDriverLibint(libint2::Operator op,
      Molecule &mol, BasisSet& basis, std::vector<double*> mats, size_t deriv) {

    shell_set& shells = basis.shells;

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
    engines[0] = libint2::Engine(op,maxPrim,maxL,deriv);
    engines[0].set_precision(0.0);


    // If engine is V, define nuclear charges
    if(op == libint2::Operator::nuclear){
      std::vector<std::pair<double,std::array<double,3>>> q;
      for(auto &atom : mol.atoms)
        q.push_back( { static_cast<double>(atom.atomicNumber), atom.coord } );

      engines[0].set_params(q);
    }

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
      size_t n1,n2,atom1,atom2;

      // Loop over unique shell pairs
      for(size_t s1(0), bf1_s(0), s12(0); s1 < shells.size(); bf1_s+=n1, s1++){ 

        n1 = shells[s1].size(); // Size of Shell 1
        atom1 = basis.mapSh2Cen[s1]; // Index of atom for Shell 1

      for(size_t s2(0), bf2_s(0); s2 <= s1; bf2_s+=n2, s2++, s12++) {

        n2 = shells[s2].size(); // Size of Shell 2
        atom2 = basis.mapSh2Cen[s2]; // Index of atom for Shell 2

        // Round Robbin work distribution
        #ifdef _OPENMP
        if( s12 % nthreads != thread_id ) continue;
        #endif

        // Compute the integrals
        engines[thread_id].compute(shells[s1],shells[s2]);

        // If the integrals were screened, move on to the next batch
        if(buf_vec[0] == nullptr) continue;

        // adds the iOp result of the engine to the iMat matrix 
        //   For non-gradients, iOp and iMat should be the same
        //   For gradients, they can differ
        auto add_shellset_to_mat = [&](size_t iOp, size_t iMat) {


          // std::cout << "iOp: " << iOp << " iMat: " << iMat << std::endl;
          Eigen::Map<
            const Eigen::Matrix<
              double,
              Eigen::Dynamic,
              Eigen::Dynamic,
              Eigen::RowMajor
            >
          > bufMat(buf_vec[iOp],n1,n2);

          size_t _idx = 0;
          for ( auto r_idx = 0; r_idx < n1; r_idx++) {
            for (auto c_idx = 0; c_idx < n2; c_idx++, _idx++) {
              // std::cout << buf_vec[iOp][_idx] << " ";
            }
            // std::cout << std::endl;
          }

          matMaps[iMat].block(bf1_s, bf2_s, n1, n2) += bufMat;

        };

        // Place integral blocks into their respective matricies
        switch (deriv) {

          case 0:
            for(auto iMat = 0; iMat < buf_vec.size(); iMat++){
              add_shellset_to_mat(iMat, iMat);
            }
            break; // case deriv == 0

          case 1:
            // For gradients, libint returns first the gradients of the
            //   bra/ket, and then the gradients of the operator. We handle
            //   these separately.
            // e.g.
            //   For the (O1s|V|H1s) nuclear attraction gradients in H2O with
            //   atom indices: O:0, H:1, H:2, libint will return 15 derivative
            //   integrals.
            //   (3 cartesian indices * (2 shell centers + 3 nuclear centers))
            //   There are only 9 gradient integrals
            //   (3 cartesian indices * 3 nuclear centers)
            //
            //   The results will be mapped to their respective gradient
            //   integrals by:
            //
            //   | ======================================================== |
            //   |   Engine result    | Gradient integral |  iOps   | iMats |
            //   | ------------------ + ----------------- + ------- + ----- |
            //   | (d/dR0 O1s|V| H1s) | d/dR0 (O1s|V|H1s) | [0,2]   | [0,2] |
            //   | (O1s|V| d/dR1 H1s) | d/dR1 (O1s|V|H1s) | [3,5]   | [3,5] |
            //   | (O1s|d/dR0 V| H1s) | d/dR0 (O1s|V|H1s) | [6,8]   | [0,2] |
            //   | (O1s|d/dR1 V| H1s) | d/dR1 (O1s|V|H1s) | [9,11]  | [3,5] |
            //   | (O1s|d/dR2 V| H1s) | d/dR2 (O1s|V|H1s) | [12,14] | [6,8] |
            //   | ======================================================== |
            //
            // For geometry independent operators, libint will only return 6
            //   derivative integrals. (bra then ket)
            // std::cout << "(" << s1 << "," << s2 << ")" << std::endl;
            // for (auto& x: buf_vec) {
            //   std::cout << "**************************************" << std::endl;
            //   for ( auto i = 0 ; i < n1*n2 ; i++ ) {
            //     std::cout << i << ": " << x[i] << std::endl;
            //   }
            // }

            size_t result_idx = 0;
            
            // First the bra and ket
            for (auto xyz = 0; xyz < 3; xyz++, result_idx++)
              add_shellset_to_mat(result_idx, 3*atom1 + xyz);

            for (auto xyz = 0; xyz < 3; xyz++, result_idx++)
              add_shellset_to_mat(result_idx, 3*atom2 + xyz);

            // Gradient of operator
            auto nAtoms = mol.atoms.size();
            if (op == libint2::Operator::nuclear) {
              for (auto iAt = 0; iAt < nAtoms; iAt++) {
                for ( auto xyz = 0; xyz < 3; xyz++, result_idx++) {
                  add_shellset_to_mat(result_idx, 3*iAt + xyz);
                }
              }
            }
            break; // case deriv == 1
        } // switch deriv

      } // Loop over s2 <= s1
      } // Loop over s1

    } // end OpenMP context


    // Symmetrize the matricies 
    for(auto nMat = 0; nMat < matMaps.size(); nMat++) 
      matMaps[nMat] = matMaps[nMat].template selfadjointView<Eigen::Lower>();

    // std::cout << std::endl;

  }; // OneEInts::OneEDriver


  template <>
  template <size_t NOPER, bool SYMM, typename F>
  void OneEInts<double>::OneEDriverLocal(
      const F &obFunc, shell_set& shells, std::vector<double*> mats) {

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

  }; // OneEInts::OneEDriverLocal

  template <>
  void OneEInts<double>::computeAOInts(BasisSet &basis, Molecule &mol,
      EMPerturbation&, OPERATOR op, const AOIntsOptions &options) {

    if (options.basisType != REAL_GTO)
      CErr("Only Real GTOs are allowed in OneEInts<double>",std::cout);
    if (op == NUCLEAR_POTENTIAL and
        (options.OneEScalarRelativity or options.OneESpinOrbit))
      CErr("Relativistic nuclear potential is not implemented in OneEInts,"
           " they are implemented in OneERelInts",std::cout);

    std::vector<double*> tmp(1, pointer());

    switch (op) {
    case OVERLAP:
      OneEDriverLibint(libint2::Operator::overlap,mol,basis,tmp);
      break;
    case KINETIC:
      OneEDriverLibint(libint2::Operator::kinetic,mol,basis,tmp);
      break;
    case NUCLEAR_POTENTIAL:
      if (options.finiteWidthNuc)
        OneEDriverLocal<1,true>(
            [&](libint2::ShellPair& pair, libint2::Shell& sh1,
                libint2::Shell& sh2) -> std::vector<std::vector<double>> {
              return RealGTOIntEngine::computePotentialV(mol.chargeDist,
                  pair,sh1,sh2,mol);
              }, basis.shells,tmp);
      else
        OneEDriverLibint(libint2::Operator::nuclear,mol,basis,tmp);
      break;
    case ELECTRON_REPULSION:
      CErr("Electron repulsion integrals are not implemented in OneEInts,"
           " they are implemented in TwoEInts",std::cout);
      break;
    case LEN_ELECTRIC_MULTIPOLE:
    case VEL_ELECTRIC_MULTIPOLE:
    case MAGNETIC_MULTIPOLE:
      CErr("Requested operator is not implemented in OneEInts,"
           " it is implemented in MultipoleInts",std::cout);
      break;
    }

  };

  template <>
  void MultipoleInts<double>::computeAOInts(BasisSet &basis, Molecule &mol,
      EMPerturbation&, OPERATOR op, const AOIntsOptions &options) {
    if (options.basisType != REAL_GTO)
      CErr("Only Real GTOs are allowed in MultipoleInts<double>",std::cout);
    if (options.OneEScalarRelativity or options.OneESpinOrbit)
      CErr("Relativistic multipole integrals are implemented in OneERelInts",std::cout);

    std::vector<double*> _multipole(1, nullptr);
    libint2::Operator libOp;

    switch (op) {
    case OVERLAP:
    case KINETIC:
    case NUCLEAR_POTENTIAL:
      CErr("Requested operator is not implemented in MultipoleInts,"
           " it is implemented in OneEInts",std::cout);
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
      OneEInts<double>::OneEDriverLibint(libOp,mol,basis,_multipole);
      memManager().free(_multipole[0]);
      break;
    case VEL_ELECTRIC_MULTIPOLE:
      switch (highOrder()) {
      case 3:
        OneEInts<double>::OneEDriverLocal<10,false>(
            std::bind(&RealGTOIntEngine::computeEOctupoleE3_vel,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3),
            basis.shells, octupolePointers());
      case 2:
        OneEInts<double>::OneEDriverLocal<6,false>(
            std::bind(&RealGTOIntEngine::computeEQuadrupoleE2_vel,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3),
            basis.shells, quadrupolePointers());
      case 1:
        OneEInts<double>::OneEDriverLocal<3,false>(
            std::bind(&RealGTOIntEngine::computeEDipoleE1_vel,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3),
            basis.shells, dipolePointers());
        break;
      default:
        CErr("Requested operator is NYI in MultipoleInts.",std::cout);
        break;
      }
      break;
    case MAGNETIC_MULTIPOLE:
      switch (highOrder()) {
      case 2:
        OneEInts<double>::OneEDriverLocal<9,false>(
            std::bind(&RealGTOIntEngine::computeMQuadrupoleM2_vel,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3),
            basis.shells, quadrupolePointers());
      case 1:
        OneEInts<double>::OneEDriverLocal<3,false>(
            std::bind(&RealGTOIntEngine::computeAngularL,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3),
            basis.shells, dipolePointers());
        break;
      default:
        CErr("Requested operator is NYI in MultipoleInts.",std::cout);
        break;
      }
      break;
    }

  };

  template <>
  void OneERelInts<double>::computeAOInts(BasisSet &basis, Molecule &mol,
      EMPerturbation&, OPERATOR op, const AOIntsOptions &options) {
    if (options.basisType != REAL_GTO)
      CErr("Only Real GTOs are allowed in OneERelInts<double>",std::cout);
    if (not options.OneEScalarRelativity or op != NUCLEAR_POTENTIAL)
      CErr("Only relativistic nuclear potential is implemented in OneERelInts.",std::cout);
    if (not options.finiteWidthNuc)
      CErr("Relativistic nuclear potential requires finite width nuclei.",std::cout);

    std::vector<double*> _potential(1, pointer());
    OneEDriverLocal<1,true>(
        [&](libint2::ShellPair& pair, libint2::Shell& sh1,
            libint2::Shell& sh2) -> std::vector<std::vector<double>> {
          return RealGTOIntEngine::computePotentialV(mol.chargeDist,
              pair,sh1,sh2,mol);
          }, basis.shells,_potential);

    std::vector<double*> _PVdP(1, scalar().pointer());
    OneEInts<double>::OneEDriverLocal<1,true>(
          [&](libint2::ShellPair& pair, libint2::Shell& sh1,
              libint2::Shell& sh2) -> std::vector<std::vector<double>> {
            return RealGTOIntEngine::computepVdotp(mol.chargeDist,
                pair,sh1,sh2,mol);
            }, basis.shells, _PVdP);

    if (options.OneESpinOrbit and components_.size() >=4) {

      OneEInts<double>::OneEDriverLocal<3,false>(
            [&](libint2::ShellPair& pair, libint2::Shell& sh1,
                libint2::Shell& sh2) -> std::vector<std::vector<double>> {
              return RealGTOIntEngine::computeSL(mol.chargeDist,
                  pair,sh1,sh2,mol);
              }, basis.shells, SOXYZPointers());
    }

  };

  template<>
  void GradInts<OneEInts,double>::computeAOInts(BasisSet& basis,
    Molecule& mol, EMPerturbation&, OPERATOR op, const AOIntsOptions& options)
  {

    std::vector<double*> gradPtrs(3*nAtoms_, nullptr);

    for (auto i = 0; i < 3*nAtoms_; i++) {
      gradPtrs[i] = components_[i].pointer();
    }


    switch (op) {
    case OVERLAP:
      OneEInts<double>::OneEDriverLibint(
        libint2::Operator::overlap, mol, basis, gradPtrs, 1
      );
      break;
    case KINETIC:
      OneEInts<double>::OneEDriverLibint(
        libint2::Operator::kinetic, mol, basis, gradPtrs, 1
      );
      break;
    case NUCLEAR_POTENTIAL:
      if (options.finiteWidthNuc)
        CErr("Finite width nuclei potential gradients not yet implemented!");
      else
        OneEInts<double>::OneEDriverLibint(
          libint2::Operator::nuclear, mol, basis, gradPtrs, 1
        );
      break;
    case ELECTRON_REPULSION:
      CErr("Electron repulsion integrals are not implemented in OneEInts,"
           " they are implemented in TwoEInts",std::cout);
      break;
    case LEN_ELECTRIC_MULTIPOLE:
    case VEL_ELECTRIC_MULTIPOLE:
    case MAGNETIC_MULTIPOLE:
      CErr("Requested operator is not implemented in OneEInts,"
           " it is implemented in MultipoleInts",std::cout);
      break;
    }


  };

  template<>
  void GradInts<MultipoleInts, double>::computeAOInts(BasisSet&,
    Molecule&, EMPerturbation&, OPERATOR, const AOIntsOptions&) {

    CErr("Gradient integrals for multipole operators not yet implemented!");

  };

  template<>
  void GradInts<OneERelInts, double>::computeAOInts(BasisSet&,
    Molecule&, EMPerturbation&, OPERATOR, const AOIntsOptions&) {

    CErr("Gradient integrals for relativistic operators not yet implemented!");

  };

  template void Integrals<double>::computeAOOneE(
      CQMemManager&, Molecule&, BasisSet&, EMPerturbation&,
      const std::vector<std::pair<OPERATOR,size_t>>&,
      const AOIntsOptions&);

  template void Integrals<double>::computeGradInts(
      CQMemManager&, Molecule&, BasisSet&, EMPerturbation&,
      const std::vector<std::pair<OPERATOR,size_t>>&,
      const AOIntsOptions&);


}; // namespace ChronusQ

