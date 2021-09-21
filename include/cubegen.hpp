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
#pragma once


#include <chronusq_sys.hpp>
#include <quantum.hpp>
#include <singleslater.hpp>

#include <cqlinalg/blas3.hpp>
#include <cqlinalg/blasutil.hpp>
#include <physcon.hpp>

namespace ChronusQ {



  enum CUBE_TYPE {
    
    _CHARGE_DENSITY,
    _SPIN_DENSITY,
    _ELECTROSTATIC_POTENTIAL

  };




  struct CubeGenBase {

  protected:

    virtual void evalCDCube() = 0;
    virtual void evalSDCube() = 0;
    virtual void evalEPCube() = 0;

    virtual void writePreamble() {

      *cubeFile << "OUTER LOOP: X, MIDDLE LOOP: Y, INNERLOOP: Z\n";
      *cubeFile << std::fixed;
      *cubeFile << cubeName << ": ";
      if( cubeType == _CHARGE_DENSITY ) 
        *cubeFile << "CHARGE DENSITY";
      if( cubeType == _SPIN_DENSITY ) 
        *cubeFile << "SPIN DENSITY";
      if( cubeType == _ELECTROSTATIC_POTENTIAL ) 
        *cubeFile << "ELECTROSTATIC POTENTIAL";

      *cubeFile << " : GENERATED BY CHRONUSQ\n";


    }


  public:



    std::string cubeName;
    std::shared_ptr<std::ofstream> cubeFile;
    CUBE_TYPE cubeType;

    std::array<size_t,3> nPts;
    std::array<double,3> L;


    CubeGenBase() = delete;
    CubeGenBase(std::string name, CUBE_TYPE ct, std::array<size_t,3> npts) :
      cubeName(name), cubeType(ct), nPts(npts), 
      cubeFile(std::make_shared<std::ofstream>(name)){ } 
    CubeGenBase(std::string name, CUBE_TYPE ct, size_t nx, size_t ny, 
      size_t nz) : CubeGenBase(name, ct, {nx,ny,nz}){ }


    void evalCube() {

      writePreamble();

      if( cubeType == _CHARGE_DENSITY )
        evalCDCube();
      else if( cubeType == _SPIN_DENSITY )
        evalSDCube();
      else if( cubeType == _ELECTROSTATIC_POTENTIAL )
        evalEPCube();

    };

  };


  template <typename T>
  class WaveFunctionCubeGenBase : public CubeGenBase {

  protected:

    std::shared_ptr<WaveFunction<T>> ref_;

    void evalCDCube(){

      size_t NB = ref_->nAlphaOrbital();

      std::vector<libint2::Shell> &shells = ref_->basisSet().shells;
      std::vector<double> BASIS(NB,0.);
      std::vector<T> SCR(NB,0.);

      for(auto ix = 0l; ix < nPts[0]; ix++)
      for(auto iy = 0l; iy < nPts[1]; iy++) {
      for(auto iz = 0l; iz < nPts[2]; iz++) {

        std::array<double,3> pt = 
          {(ix-(int)nPts[0]/2) * L[0],
           (iy-(int)nPts[1]/2) * L[1],
           (iz-(int)nPts[2]/2) * L[2]};

        evalShellSet(ref_->memManager,NOGRAD,shells,&pt[0],1,&BASIS[0],false);

        blas::gemm(blas::Layout::ColMajor,'T',blas::Op::NoTrans,1,NB,NB,T(1.),&BASIS[0],NB,ref_->onePDM->S().pointer(),NB,
            T(0.),&SCR[0],1);

        double val = blas::dot(NB,&SCR[0],1,&BASIS[0],1);

        *cubeFile << std::right << std::setw(15) << std::setprecision(5)
          << std::scientific << val;

        if( iz % 6 == 5 ) *cubeFile << "\n";


      }
        *cubeFile << "\n";
      }

    };


    void evalSDCube() {

      size_t NB = ref_->nAlphaOrbital();

      std::vector<libint2::Shell> &shells = ref_->basisSet().shells;
      std::vector<double> BASIS(NB,0.);
      std::vector<T> SCR(NB,0.);

      for(auto ix = 0l; ix < nPts[0]; ix++)
      for(auto iy = 0l; iy < nPts[1]; iy++) {
      for(auto iz = 0l; iz < nPts[2]; iz++) {

        std::array<double,3> pt = 
          {(ix-(int)nPts[0]/2) * L[0],
           (iy-(int)nPts[1]/2) * L[1],
           (iz-(int)nPts[2]/2) * L[2]};

        double val = 0;

        if( ref_->onePDM.hasZ() ) {

          evalShellSet(ref_->memManager,NOGRAD,shells,&pt[0],1,&BASIS[0],false);

          for(size_t i = 1; i < ref_->onePDM->nComponent(); i++) {
            blas::gemm(blas::Layout::ColMajor,'T',blas::Op::NoTrans,1,NB,NB,T(1.),&BASIS[0],NB,
                (*ref_->onePDM)[static_cast<PAULI_SPINOR_COMPS>(i)].pointer()],NB,
                T(0.),&SCR[0],1);
  
            double tmp = blas::dot(NB,&SCR[0],1,&BASIS[0],1);
            val += tmp*tmp;
          }

          val = std::sqrt(val);

        }
        *cubeFile << std::right << std::setw(15) << std::setprecision(5)
          << std::scientific << val;

        if( iz % 6 == 5 ) *cubeFile << "\n";


      }
        *cubeFile << "\n";
      }

    };



    void evalEPCube(){
    
    
      size_t NB = ref_->nAlphaOrbital();

      std::vector<libint2::Shell> &shells = ref_->basisSet().shells;
    
      std::vector<T> SCR(NB*NB,0.);
    
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
    
      libint2::Engine engine(libint2::Operator::nuclear,maxPrim,maxL,0);
      engine.set_precision(0.);

      for(auto ix = 0l; ix < nPts[0]; ix++)
      for(auto iy = 0l; iy < nPts[1]; iy++) {
      for(auto iz = 0l; iz < nPts[2]; iz++) {

        std::array<double,3> pt = 
          {(ix-(int)nPts[0]/2) * L[0],
           (iy-(int)nPts[1]/2) * L[1],
           (iz-(int)nPts[2]/2) * L[2]};
    
        std::vector<std::pair<double,std::array<double,3>>> q;
        q.push_back( {1., pt} );
        engine.set_params(q);
    
        const auto& buf_vec = engine.results();
        size_t n1,n2;

        // Loop over unique shell pairs
        for(size_t s1(0), bf1_s(0), s12(0); s1 < shells.size(); 
            bf1_s+=n1, s1++){ 
          n1 = shells[s1].size(); // Size of Shell 1
        for(size_t s2(0), bf2_s(0); s2 <= s1; bf2_s+=n2, s2++, s12++) {
          n2 = shells[s2].size(); // Size of Shell 2


          engine.compute(shells[s1],shells[s2]);

          // If the integrals were screened, move on to the next batch
          if(buf_vec[0] == nullptr) continue;

          SetMat('N',n1,n2,1.,const_cast<double*>(buf_vec[0]),n1,
              &SCR[bf1_s + bf2_s*NB],NB);

        }
        }

        HerMat('L',NB,&SCR[0],NB);
        prettyPrintSmart(std::cout,"V",&SCR[0],NB,NB,NB);

        double val = blas::dot(NB*NB,&SCR[0],1,ref_->onePDM->S().pointer(),1);

        for( auto &atom : ref_->molecule().atoms ) {

          auto &c = atom.coord;
          val += atom.atomicNumber / 
            std::sqrt( std::pow(pt[0] - c[0],2.) + 
                       std::pow(pt[1] - c[1],2.) + 
                       std::pow(pt[2] - c[2],2.) ); 

        }


        *cubeFile << std::right << std::setw(15) << std::setprecision(5)
          << std::scientific << val;

        if( iz % 6 == 5 ) *cubeFile << "\n";


      }
        *cubeFile << "\n";
      }
    
    };

    void writePreamble() {

      CubeGenBase::writePreamble();

      size_t nAtoms = ref_->molecule().nAtoms;
      Molecule &mol = ref_->molecule();

      *cubeFile << std::setprecision(6);

      L = {0.283459,0.283459,0.283459};
      *cubeFile << std::setw(6) << std::right << nAtoms;
      *cubeFile << std::setw(15) << -L[0]*(double)nPts[0]/2;
      *cubeFile << std::setw(15) << -L[1]*(double)nPts[1]/2;
      *cubeFile << std::setw(15) << -L[2]*(double)nPts[2]/2;
      *cubeFile << "\n";


      *cubeFile << std::setw(6) << std::right << nPts[0];
      *cubeFile << std::setw(15) << L[0];
      *cubeFile << std::setw(15) << 0.;
      *cubeFile << std::setw(15) << 0.;
      *cubeFile << "\n";

      *cubeFile << std::setw(6) << std::right << nPts[1];
      *cubeFile << std::setw(15) << 0.;
      *cubeFile << std::setw(15) << L[1];
      *cubeFile << std::setw(15) << 0.;
      *cubeFile << "\n";

      *cubeFile << std::setw(6) << std::right << nPts[2];
      *cubeFile << std::setw(15) << 0.;
      *cubeFile << std::setw(15) << 0.;
      *cubeFile << std::setw(15) << L[2];
      *cubeFile << "\n";

      for( auto &atom : ref_->molecule().atoms ) {

        *cubeFile << std::setw(6) << atom.atomicNumber;
        *cubeFile << std::setw(15) << 0.;

        *cubeFile << std::setw(15) << atom.coord[0];
        *cubeFile << std::setw(15) << atom.coord[1];
        *cubeFile << std::setw(15) << atom.coord[2];

        *cubeFile << "\n";

      }

    }


  public:

    using ref_type = WaveFunction<T>;

    WaveFunctionCubeGenBase() = delete;

    WaveFunctionCubeGenBase(std::string name, CUBE_TYPE ct, 
      std::array<size_t,3> npts, std::shared_ptr<WaveFunction<T>> ref) :
      CubeGenBase(name,ct,npts), ref_(ref){ }

    WaveFunctionCubeGenBase(std::string name, CUBE_TYPE ct, 
      size_t nx, size_t ny, size_t nz, std::shared_ptr<WaveFunction<T>> ref) :
      WaveFunctionCubeGenBase(name,ct,{nx,ny,nz},ref){ }

  };



  namespace detail {

    template <typename R, typename = void>
    struct cube_gen_base_identity{ };

    template <typename R>
    struct cube_gen_base_identity< R,
      typename std::enable_if< 
        std::is_base_of<WaveFunction<typename R::value_type>,R>::value >::type
      > {

        using base_type = WaveFunctionCubeGenBase<typename R::value_type>;

    };

  };

  template < typename Reference, 
             typename Base = 
               typename detail::cube_gen_base_identity<Reference>::base_type >
  class CubeGen : public Base {


    using value_type = typename Reference::value_type;

    public:

      CubeGen( std::string name, CUBE_TYPE ct, std::array<size_t,3> npts,
        std::shared_ptr<Reference> ref) :
        Base(name,ct,npts,std::dynamic_pointer_cast<WaveFunction<value_type>>(ref)){ }

  };

};

