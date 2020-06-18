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
//#define _DEBUGGIAOERI //SS
//#define _DEBUGGIAOONEE //SS 


namespace ChronusQ {

  typedef std::vector<libint2::Shell> shell_set; 

  template <size_t NOPER, bool SYMM, typename F>
  std::vector<dcomplex*>
  GIAOIntegrals::OneEDriverLocalGIAO(const F &obFunc, shell_set& shells) {

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
    std::vector<dcomplex*> mats(NOPER); 

    std::vector<
      Eigen::Map<
        Eigen::Matrix<dcomplex,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> 
      > 
    > matMaps;

    for( auto i = 0; i < mats.size(); i++ ) {
      mats[i] = memManager_.template malloc<dcomplex>(NBSQ); 
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

    // here the symmetrization is still real 
    return mats;

  }; // GIAOIntegrals::OneEDriverLocalGIAO


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

  void GIAOIntegrals::computeAOOneEGIAO(EMPerturbation &emPert, OneETerms &oneETerms) {

    size_t NB = basisSet_.nBasis;

    if(oneETerms.coreH) {

      auto magAmp = emPert.getDipoleAmp(Magnetic);

      auto _GIAOS = OneEDriverLocalGIAO<1,true>( 
                  std::bind(&ComplexGIAOIntEngine::computeGIAOOverlapS,
                            std::placeholders::_1, std::placeholders::_2,
                            std::placeholders::_3, &magAmp[0]),
                  basisSet_.shells);

      auto _GIAOT = OneEDriverLocalGIAO<1,true>( 
                  std::bind(&ComplexGIAOIntEngine::computeGIAOKineticT,
                            std::placeholders::_1, std::placeholders::_2,
                            std::placeholders::_3, &magAmp[0]),
                  basisSet_.shells);

      auto _GIAOL = OneEDriverLocalGIAO<3,false>( 
                  std::bind(&ComplexGIAOIntEngine::computeGIAOAngularL,
                            std::placeholders::_1, std::placeholders::_2,
                            std::placeholders::_3, &magAmp[0]),
                  basisSet_.shells);

      auto _GIAOED1 = OneEDriverLocalGIAO<3,true>( 
                  std::bind(&ComplexGIAOIntEngine::computeGIAOEDipoleE1_len,
                            std::placeholders::_1, std::placeholders::_2,
                            std::placeholders::_3, &magAmp[0]),
                  basisSet_.shells);

      auto _GIAOEQ2 = OneEDriverLocalGIAO<6,true>( 
                  std::bind(&ComplexGIAOIntEngine::computeGIAOEQuadrupoleE2_len,
                            std::placeholders::_1, std::placeholders::_2,
                            std::placeholders::_3, &magAmp[0]),
                  basisSet_.shells);

      auto _GIAOEO3 = OneEDriverLocalGIAO<10,true>( 
                  std::bind(&ComplexGIAOIntEngine::computeGIAOEOctupoleE3_len,
                            std::placeholders::_1, std::placeholders::_2,
                            std::placeholders::_3, &magAmp[0]),
                  basisSet_.shells);

/*
//XSLIC: finite nuclei?
//SS: finite nuclei doesn't work for now 
      auto _GIAOV = OneEDriverLocalGIAO<1,true>(
            [&](libint2::ShellPair& pair, libint2::Shell& sh1, 
                libint2::Shell& sh2) -> std::vector<std::vector<dcomplex>> { 
              return ComplexGIAOIntEngine::computeGIAOPotentialV(
                  molecule_.chargeDist, pair,sh1,sh2,&magAmp[0],molecule_);
              }, basisSet_.shells);
*/
  
      auto _GIAOV = OneEDriverLocalGIAO<1,true>(
            [&](libint2::ShellPair& pair, libint2::Shell& sh1, 
                libint2::Shell& sh2) -> std::vector<std::vector<dcomplex>> { 
              return ComplexGIAOIntEngine::computeGIAOPotentialV(
                  pair,sh1,sh2,&magAmp[0],molecule_);
              }, basisSet_.shells);



      overlap   = reinterpret_cast<dcomplex*>(_GIAOS[0]);
      kinetic   = reinterpret_cast<dcomplex*>(_GIAOT[0]);
      potential = reinterpret_cast<dcomplex*>(_GIAOV[0]);
      magDipole = {reinterpret_cast<dcomplex*>(_GIAOL[0]), 
                   reinterpret_cast<dcomplex*>(_GIAOL[1]),
                   reinterpret_cast<dcomplex*>(_GIAOL[2])};

      lenElecDipole = { reinterpret_cast<dcomplex*>(_GIAOED1[0]),
                        reinterpret_cast<dcomplex*>(_GIAOED1[1]),
                        reinterpret_cast<dcomplex*>(_GIAOED1[2])};

      lenElecQuadrupole = {reinterpret_cast<dcomplex*>(_GIAOEQ2[0]),
                           reinterpret_cast<dcomplex*>(_GIAOEQ2[1]),
                           reinterpret_cast<dcomplex*>(_GIAOEQ2[2]),
                           reinterpret_cast<dcomplex*>(_GIAOEQ2[3]),
                           reinterpret_cast<dcomplex*>(_GIAOEQ2[4]),
                           reinterpret_cast<dcomplex*>(_GIAOEQ2[5])}; 

      lenElecOctupole = {reinterpret_cast<dcomplex*>(_GIAOEO3[0]),
                         reinterpret_cast<dcomplex*>(_GIAOEO3[1]),
                         reinterpret_cast<dcomplex*>(_GIAOEO3[2]),
                         reinterpret_cast<dcomplex*>(_GIAOEO3[3]),
                         reinterpret_cast<dcomplex*>(_GIAOEO3[4]),
                         reinterpret_cast<dcomplex*>(_GIAOEO3[5]),
                         reinterpret_cast<dcomplex*>(_GIAOEO3[6]),
                         reinterpret_cast<dcomplex*>(_GIAOEO3[7]),
                         reinterpret_cast<dcomplex*>(_GIAOEO3[8]),
                         reinterpret_cast<dcomplex*>(_GIAOEO3[9])};  
    } 

#ifdef _DEBUGGIAOONEE

    prettyPrintSmart(std::cout,"GIAO S",overlap,basisSet_.nBasis,basisSet_.nBasis,basisSet_.nBasis);

    prettyPrintSmart(std::cout,"GIAO T",kinetic,basisSet_.nBasis,basisSet_.nBasis,basisSet_.nBasis);

    for ( int ii = 0 ; ii < 3 ; ii++ ) {
      std::cout<<"ii = "<<ii<<std::endl;
      prettyPrintSmart(std::cout,"GIAO L",magDipole[ii],
        basisSet_.nBasis,basisSet_.nBasis,basisSet_.nBasis);
    } // for ( int ii = 0 )

    // print length gauge electric dipole  
    for ( int ii = 0 ; ii < 3 ; ii++ ) {
      std::cout<<"ii = "<<ii<<std::endl;
      prettyPrintSmart(std::cout,"GIAO electric Dipole length gauge",lenElecDipole[ii],
        basisSet_.nBasis,basisSet_.nBasis,basisSet_.nBasis);
    } // for ( int ii = 0 )
    
    // print out GIAO electric quadrupole in length gauge
    for ( int ii = 0 ; ii < 6 ; ii++ ) {
      std::cout<<"ii = "<<ii<<std::endl;
      prettyPrintSmart(std::cout,"GIAO EQ length gauge",lenElecQuadrupole[ii],
        basisSet_.nBasis,basisSet_.nBasis,basisSet_.nBasis);
    } // for ( int ii = 0 )
    
    prettyPrintSmart(std::cout,"GIAO V",potential,basisSet_.nBasis,basisSet_.nBasis,basisSet_.nBasis);
                                                                      

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
 /*
      for(auto i = 0; i < 3; i++)
        savFile.safeWriteData("INTS/ELEC_DIPOLE_LEN_" + 
          dipoleList[i], lenElecDipole[i], {NB,NB} );
*/

      // Length Gauge electric quadrupole
      for(auto i = 0; i < 6; i++)
        savFile.safeWriteData("INTS/ELEC_QUADRUPOLE_LEN_" + 
          quadrupoleList[i], lenElecQuadrupole[i], {NB,NB} );
/*
      // Length Gauge electric octupole
      for(auto i = 0; i < 10; i++)
        savFile.safeWriteData("INTS/ELEC_OCTUPOLE_LEN_" + 
          octupoleList[i], lenElecOctupole[i], {NB,NB} );
*/

      // Magnetic Dipole
      for(auto i = 0; i < 3; i++)
        savFile.safeWriteData("INTS/MAG_DIPOLE_" + 
          dipoleList[i], magDipole[i], {NB,NB} );
      // FIXME: Write valocity gauge integrals!
      // FIXME: Write PVP integrals
    }

  }; // GIAOIntegrals::computeAOOneEGIAO

  void GIAOIntegrals::computeAOOneE(EMPerturbation &emPert, 
      OneETerms &oneETerms) {

//    bool useGIAO = 
//      (basisSet_.basisType == COMPLEX_GIAO) and 
//      pert_has_type(emPert,Magnetic);
    GIAOIntegrals::computeAOOneEGIAO(emPert,oneETerms);

  };


}; // namespace ChronusQ

