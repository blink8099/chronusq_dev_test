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

#include <electronintegrals/oneeints/aooneeints.hpp>

// Debug directives
//#define _DEBUGORTHO
//#define _DEBUGERI
//#define _DEBUGGIAOERI //SS
//#define _DEBUGGIAOONEE //SS 


namespace ChronusQ {

  typedef std::vector<libint2::Shell> shell_set;

  template <>
  template <size_t NOPER, bool SYMM, typename F>
  void OneEInts<dcomplex>::OneEDriverLocal(
      const F &obFunc, shell_set& shells, std::vector<dcomplex*> mats) {

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
        Eigen::Matrix<dcomplex,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> 
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
            dcomplex,
            Eigen::Dynamic,Eigen::Dynamic,  
            Eigen::RowMajor>>
          bufMat(&buff[iMat][0],n1,n2);

        matMaps[iMat].block(bf1_s,bf2_s,n1,n2) = bufMat.template cast<dcomplex>();
      }

    } // Loop over s2 <= s1
    } // Loop over s1


    // Symmetrize the matricies 
    // XXX: USES EIGEN
    // FIXME: not SYMM -> creates a temporary
    for(auto nMat = 0; nMat < matMaps.size(); nMat++) {
      if(SYMM) {
        for(auto i = 0  ; i < NB; ++i)
        for(auto j = i+1; j < NB; ++j)
          matMaps[nMat](i,j) = std::conj(matMaps[nMat](j,i));
      } else {
        for(auto i = 0  ; i < NB; ++i)
        for(auto j = i+1; j < NB; ++j)
          matMaps[nMat](i,j) = - std::conj(matMaps[nMat](j,i));
      }
    }

  }; // OneEInts::OneEDriverLocal


  template <>
  void OneEInts<dcomplex>::computeAOInts(BasisSet &basis, Molecule &mol,
      EMPerturbation &emPert, OPERATOR op, const HamiltonianOptions &options) {
    if (options.basisType == REAL_GTO)
      CErr("Real GTOs are not allowed in OneEInts<dcomplex>",std::cout);
    if (options.basisType == COMPLEX_GTO)
      CErr("Complex GTOs NYI in OneEInts<dcomplex>",std::cout);
    if (options.OneEScalarRelativity or options.OneESpinOrbit)
      CErr("Relativistic integrals are implemented in OneERelInts",std::cout);

    auto magAmp = emPert.getDipoleAmp(Magnetic);
    std::vector<dcomplex*> tmp(1, pointer());

    switch (op) {
    case OVERLAP:
      OneEInts<dcomplex>::OneEDriverLocal<1,true>(
          std::bind(&ComplexGIAOIntEngine::computeGIAOOverlapS,
                    std::placeholders::_1, std::placeholders::_2,
                    std::placeholders::_3, &magAmp[0]),
          basis.shells, tmp);
      break;
    case KINETIC:
      OneEInts<dcomplex>::OneEDriverLocal<1,true>(
          std::bind(&ComplexGIAOIntEngine::computeGIAOKineticT,
                    std::placeholders::_1, std::placeholders::_2,
                    std::placeholders::_3, &magAmp[0]),
          basis.shells, tmp);
      break;
    case NUCLEAR_POTENTIAL:
      if (options.finiteWidthNuc)
        CErr("Finite nuclei NYI for GIAO",std::cout);
      OneEInts<dcomplex>::OneEDriverLocal<1,true>(
          [&](libint2::ShellPair& pair, libint2::Shell& sh1,
              libint2::Shell& sh2) -> std::vector<std::vector<dcomplex>> {
            return ComplexGIAOIntEngine::computeGIAOPotentialV(
                pair,sh1,sh2,&magAmp[0],mol);
            }, basis.shells, tmp);
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
  void MultipoleInts<dcomplex>::computeAOInts(BasisSet &basis, Molecule&,
      EMPerturbation &emPert, OPERATOR op, const HamiltonianOptions &options) {
    if (options.basisType == REAL_GTO)
      CErr("Real GTOs are not allowed in MultipoleInts<dcomplex>",std::cout);
    if (options.basisType == COMPLEX_GTO)
      CErr("Complex GTOs NYI in MultipoleInts<dcomplex>",std::cout);
    if (options.OneEScalarRelativity or options.OneESpinOrbit)
      CErr("Relativistic integrals are implemented in OneERelInts",std::cout);

    auto magAmp = emPert.getDipoleAmp(Magnetic);

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
      switch (highOrder()) {
      case 3:
        OneEInts<dcomplex>::OneEDriverLocal<10,true>(
            std::bind(&ComplexGIAOIntEngine::computeGIAOEOctupoleE3_len,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, &magAmp[0]),
            basis.shells, octupolePointers());
      case 2:
        OneEInts<dcomplex>::OneEDriverLocal<6,true>(
            std::bind(&ComplexGIAOIntEngine::computeGIAOEQuadrupoleE2_len,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, &magAmp[0]),
            basis.shells, quadrupolePointers());
      case 1:
        OneEInts<dcomplex>::OneEDriverLocal<3,true>(
            std::bind(&ComplexGIAOIntEngine::computeGIAOEDipoleE1_len,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, &magAmp[0]),
            basis.shells, dipolePointers());
        break;
      default:
        CErr("Requested operator is NYI in MultipoleInts.",std::cout);
        break;
      }
      break;
    case VEL_ELECTRIC_MULTIPOLE:
      CErr("Requested operator is NYI in MultipoleInts.",std::cout);
      break;
    case MAGNETIC_MULTIPOLE:
      switch (highOrder()) {
      case 1:
        OneEInts<dcomplex>::OneEDriverLocal<3,false>(
            std::bind(&ComplexGIAOIntEngine::computeGIAOAngularL,
                      std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3, &magAmp[0]),
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
  void OneERelInts<dcomplex>::computeAOInts(BasisSet&, Molecule&,
      EMPerturbation&, OPERATOR, const HamiltonianOptions &options) {
    if (options.basisType == REAL_GTO)
      CErr("Real GTOs are not allowed in OneERelInts<dcomplex>",std::cout);
    if (options.basisType == COMPLEX_GTO)
      CErr("Complex GTOs NYI in OneERelInts<dcomplex>",std::cout);

    CErr("Relativistic GIAO one electron integrals NYI",std::cout);
  };

  template void Integrals<dcomplex>::computeAOOneE(
      CQMemManager&, Molecule&, BasisSet&, EMPerturbation&,
      const std::vector<std::pair<OPERATOR,size_t>>&,
      const HamiltonianOptions&);


}; // namespace ChronusQ

