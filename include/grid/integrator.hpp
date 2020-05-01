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

// TODO: Increment docs on quadrature / integrators


#include <chronusq_sys.hpp>
#include <grid/quadrature.hpp>
#include <basisset.hpp>
#include <molecule.hpp>
#include <physcon.hpp>

#include <util/threads.hpp>
#include <util/mpi.hpp>

#include <cqlinalg/cqlinalg_config.hpp>

// INT_DEBUG_LEVEL == 1 - Timing
// INT_DEBUG_LEVEL >= 3 - Print EVERYTHING and turn off screening
#ifndef INT_DEBUG_LEVEL
#  define INT_DEBUG_LEVEL 0
#endif

using cart_t = std::array<double,3>;
namespace ChronusQ {

  /**
   *  Public function used in the SphereINtegrator (and so in the BeckeIntegrator) 
   *  the function to be integrated needs to get as arguments the res, the batch of points, 
   *  batch of weights, and an arbitrary number of arguments
   */
  template <typename _RetType, typename _PtTyp, typename _Func, typename... Args>
  void BatchIntegrator(_RetType &res, const _Func &F, std::vector<_PtTyp> &batch, 
    std::vector<double> &weights, Args... args) {
    F(res,batch,weights,args...);
  }

  
  /** 
   *  \brief 1D Quadrature integrator
   *
   *  Templated over Quadrature class. To be used as 
   *  Integrator1<_QTyp> obj(QTyp_Constr(arg...));
   */
  template <class _QTyp>
  class Integrator1D {
  protected:

    _QTyp q; ///< Quadrature

  public:

    // Defaulted / Deleted ctors
    Integrator1D()                     = delete;
    Integrator1D(const Integrator1D &) = default;
    Integrator1D(Integrator1D &&)      = default;

    /**
     *  \brief Integrator1D constructor.
     *
     *  Constructs a Integrator1D object from a Quadrature object
     */ 
    Integrator1D( _QTyp g ) : q(std::move(g)) {
      if(q.pts.size() == 0) q.generateQuadrature();
    };

    /**
     *  \brief Integrator1D constructor.
     *
     *  Constructs the member Quadrature object in place using variadic 
     *  parameter pack.
     */ 
    template<typename... Args>
    Integrator1D(size_t N, Args... args) :
      q(N,args...){ assert(N != 0); q.generateQuadrature(); }

 
    /**
     *  \brief Integrate function
     *
     *  Integrates a passed 1D function and returns a value
     */ 
    template <typename T, class F>
    T integrate(const F &func) {

#if INT_DEBUG_LEVEL >= 3
      std::cerr << "1D" << std::endl;
#endif

      T res(0.);
      for(size_t i = 0; i < q.nPts; i++) {
        res += q.weights[i] * func(q.pts[i]);
      }
      return res;
    }; // integrate

  }; // class Integrator1D

  /** 
   *  \brief 2D Quadrature integrator
   *  
   *  Templated over Quadrature classes. To be used as 
   *  Integrator2<_QTyp1,_QTyp2> obj(QTyp1_Constr(arg1,...),QTyp2_Constr(arg2,...);
   *
   */
  template <class _QTyp1, class _QTyp2>
  class Integrator2D {

  protected:
 
     _QTyp1 q1; ///< Quadrature over dimension 1
     _QTyp2 q2; ///< Quadrature over dimension 2

  public:

    // Defaulted / Deleted ctors
    Integrator2D()                     = delete;
    Integrator2D(const Integrator2D &) = default;
    Integrator2D(Integrator2D &&)      = default;

    /**
     *  \brief Integrator2D constructor.
     *
     *  Constructs a Integrator2D object from two Quadrature objects
     */ 
    Integrator2D(_QTyp1 g1, _QTyp2 g2) : 
      q1(std::move(g1)), q2(std::move(g2)) {

      if (q1.pts.size() == 0) q1.generateQuadrature();
      if (q2.pts.size() == 0) q2.generateQuadrature();

    };
    
    /**
     *  \brief Integrate function
     *
     *  Integrates a passed 2D function and returns a value
     */ 
    template <typename T, class F>
    T integrate(const F &func) {

#if INT_DEBUG_LEVEL >= 3
      std::cerr << "2D" << std::endl;
#endif
      T res(0.);

      for(size_t i = 0; i < q1.nPts; i++) 
      for(size_t j = 0; j < q2.nPts; j++) {
        res += q1.weights[i] * q2.weights[j] * 
          func(q1.pts[i],q2.pts[j]);
      }

      return res;

    }; // integrate

  }; // class Integrator2D

  /** 
   *  \brief 3D Quadrature integrator
   *  
   *  Templated over Quadrature classes. To be used as 
   *  Integrator3<_QTyp1,_QTyp2,_QTyp3> obj(QTyp1_Constr(arg1,...),
   *                                        QTyp2_Constr(arg2,...),
   *                                        QTyp3_Constr(arg3,...);
   */
  template <class _QTyp1, class _QTyp2, class _QTyp3>
  class Integrator3D {

  protected:
 
     _QTyp1 q1; ///< Quadrature over dimension 1
     _QTyp2 q2; ///< Quadrature over dimension 2
     _QTyp3 q3; ///< Quadrature over dimension 3

  public:

    // Defaulted / Deleted ctors
    Integrator3D()                     = delete;
    Integrator3D(const Integrator3D &) = default;
    Integrator3D(Integrator3D &&)      = default;

    /**
     *  \brief Integrator3D constructor.
     *
     *  Constructs a Integrator3D object from three Quadrature objects
     */ 
    Integrator3D(_QTyp1 g1, _QTyp2 g2, _QTyp3 g3) : 
      q1(std::move(g1)), q2(std::move(g2)), q3(std::move(g3)) {

      if (q1.pts.size() == 0) q1.generateQuadrature();
      if (q2.pts.size() == 0) q2.generateQuadrature();
      if (q3.pts.size() == 0) q3.generateQuadrature();

    };
     

    /**
     *  \brief Generates a 3D batch of points via the 
     *  cartesian product of the 1D quadratures.
     *
     *  For a select number of points in the first quadrature
     *  (q1), this function returns the cartesian product over
     *  all of the points in the second (q2) and third (q3)
     *  quadratures
     */ 
    void generateBatch(size_t ist, size_t iend, 
      std::vector<cart_t> &batchPt, std::vector<double> &batchW) {

      for(size_t i = ist; i <= iend; i++) 
      for(size_t j = 0; j < q2.nPts; j++) 
      for(size_t k = 0; k < q3.nPts; k++) {
        cart_t pt{q1.pts[i],q2.pts[j],q3.pts[k]}; 
        double weight = q1.weights[i] * q2.weights[j] * q3.weights[k] ; 
        batchPt[(i - ist)*q2.nPts*q3.nPts + j*q3.nPts + k] = pt;
        batchW [(i - ist)*q2.nPts*q3.nPts + j*q3.nPts + k] = weight;
      }

    }; // generateBatch


    /**
     *  \brief Integrate function
     *
     *  Integrates a passed 3D function and increments the result
     *  which is to be passed by reference. The result is to be scaled
     *  by a passed parameter.
     */ 
    template <typename T, class F, typename... Args>
    void integrate(double SCALE, T &res, const F &func, Args... args) {

#if INT_DEBUG_LEVEL >= 3
      std::cerr << "3D" << std::endl;
#endif

      size_t n1DPerMacroBatch = 4;
      size_t nMicroBatches = 1;
      size_t nMacroBatches   = this->q1.nPts / n1DPerMacroBatch;
      size_t nBatches = nMacroBatches * nMicroBatches;
      for(size_t iBatch = 0; iBatch < nBatches; iBatch++){
        size_t Jst = iBatch * n1DPerMacroBatch;
        size_t Jend = Jst + n1DPerMacroBatch - 1;
        std::vector<cart_t> batchPt((Jend - Jst + 1) * this->q2.nPts*this->q3.nPts,cart_t({0,0,0}));
        std::vector<double> batchW((Jend - Jst + 1) * this->q2.nPts*this->q3.nPts,0);
        generateBatch(Jst,Jend,batchPt,batchW);
        BatchIntegrator(res,func,batchPt,batchW,args...);
      }
      res *= SCALE;

    }; // integrate


    /**
     *  \brief Specialization of integrate function which leaves
     *  the result unscaled.
     */ 
    template <typename T, class F, typename... Args>
    inline void integrate(T &res, const F &func, Args... args) {
      integrate<T>(1.0,res,func,args...);
    } // integrate


    /**
     *  \brief Generalization of the integrate function which
     *  constructs the result internally and passes to the
     *  integrate function to be incremented by reference.
     *
     */ 
    template <typename T, class F, typename... Args>
    inline T integrate(const F &func, Args... args) {
      T res(0.);
      integrate<T>(res,func,args...);
      return res;
    } // integrate

  }; // class Integrator3D


  /**
   *  \brief Sphere Integrator
   *
   *  Handles the integration of a 3D (cartesian) function over a ball of
   *  a specified radius.
   *
   *  \f[
   *    \int_0^R r^2 \mathrm{d}r \int_0^\pi \sin\pi \mathrm{d}\phi \int_0^{2\pi}\ mathrm{d}\theta
   *      f(r,\theta,\pi)
   *  \f]
   *
   *  Templated over radial quadrature scheme, angular quadrature taken to be Lebedev. Specializes
   *  Integrator2D.
   *
   *  Coordinate transforms:
   *
   *  \f$ x = r \sin\theta \cos\phi \f$
   *  \f$ y = r \sin\theta \sin\phi \f$
   *  \f$ z = r \cos\theta \f$
   */ 
  template <class _QTyp1>
  class SphereIntegrator : public Integrator2D<_QTyp1,Lebedev> {

  protected:
    double Scale;             ///< Radial scaling parameter
    cart_t Center;            ///< Origin of the ball
    size_t nRadPerMacroBatch; ///< # Radial points per macro batch

  public:

    // Defaulted / Deleted ctors
    SphereIntegrator()                         = delete;
    SphereIntegrator(const SphereIntegrator &) = default;
    SphereIntegrator(SphereIntegrator &&)      = default;

    /**
     *  \brief SphereIntegrator constructor.
     *
     *  Constructs a SphereIntegrator object from a Quadrature object for the
     *  radial quadrature
     */ 
    SphereIntegrator( _QTyp1 g, size_t NAng , cart_t center, double scale,
      size_t nRadPerMacroBatch_ = 4) :
      Scale(scale),Center(center), nRadPerMacroBatch(nRadPerMacroBatch_),
      Integrator2D<_QTyp1,Lebedev> (g,Lebedev(NAng)){ 

      assert(g.nPts % nRadPerMacroBatch == 0);
      
     };


    /**
     *  \brief Generates a 3D batch of points via the 
     *  cartesian product of the radial and angular quadrature
     *  schemes.
     *
     *  For a select number of points in the radial quadrature,
     *  this function returns the cartesian points for an interior
     *  region of the ball.
     */ 
    void generateBatch(size_t Jst, size_t Jend, 
      std::vector<cart_t> &batchPt, std::vector<double> &batchW) {

      for(size_t J = Jst; J <= Jend; J++) {
        // FIXME: Screen on maximal spatial extent
        double R = this->q1.pts[J] * Scale; 
      for(size_t i = 0; i < this->q2.nPts; i++) {

        // INT = W1(i) * W2(j) * R(i) * R(i) * 
        //       func(R(i)*x(j),R(i)*y(j),R(i)*z(j))
          
        cart_t pt;
        pt[0] = (R*this->q2.pts[i][0] + Center[0]);
        pt[1] = (R*this->q2.pts[i][1] + Center[1]);
        pt[2] = (R*this->q2.pts[i][2] + Center[2]);
        double weight = this->q1.weights[J] * this->q2.weights[i] * R * R * Scale; 
        batchPt[(J - Jst)*this->q2.nPts + i] = pt;
        batchW [(J - Jst)*this->q2.nPts + i] = weight;

      } // i loop
      } // J loop

    }; // generateBatch



    /**
     *  \brief Integrate function
     *
     *  Integrates a passed 3D function and increments the result
     *  which is to be passed by reference. The result is to be scaled
     *  by a passed parameter.
     */ 
    template <typename T, class F, typename... Args>
    void integrate(double SCALE, T &res, const F &func, Args... args) {
      size_t nMicroBatches = 1;
      size_t nMacroBatches   = this->q1.nPts / nRadPerMacroBatch;
      size_t nBatches = nMacroBatches * nMicroBatches;

#if INT_DEBUG_LEVEL >= 3
      std::cerr << "NBatch " << nBatches << std::endl;
      std::cerr << "SPHERE OF NRAD " <<  this->q1.nPts << " AND NANG " <<  this->q2.nPts << std::endl;
      std::cerr << "SCALE " <<  Scale << std::endl;
      std::cerr << "CENTER { "<< Center[0] <<" , " << Center[1] <<" , " << Center[2] <<" }, " << std::endl; 
#endif

      size_t nthreads = GetNumThreads();

      #pragma omp parallel
      {
      
      size_t thread_id = GetThreadID();

      for(size_t iBatch = 0; iBatch < nBatches; iBatch++){

        if( iBatch % nthreads != thread_id ) continue;

        size_t Jst = iBatch * nRadPerMacroBatch;
        size_t Jend = Jst + nRadPerMacroBatch - 1;
        std::vector<cart_t> batchPt((Jend - Jst + 1) * this->q2.nPts,cart_t({0,0,0}));
        std::vector<double> batchW((Jend - Jst + 1) * this->q2.nPts,0);
        generateBatch(Jst,Jend,batchPt,batchW);

        std::pair<double,double> rBounds{Scale*this->q1.pts[Jst], Scale*this->q1.pts[Jend]};

        // FIXME: This will cause a race condition for scalar integration
        BatchIntegrator(res,func,batchPt,batchW,rBounds,args...);

      }
      }
      res *= SCALE;
    }; // Integrate

    /**
     *  \brief Specialization of integrate function which leaves
     *  the result scaled by 4*PI (the cannonical choice)
     */ 
    template <typename T, class F, typename... Args>
    inline void integrate(T &res, const F &func, Args... args) {
      integrate<T>(4.*M_PI,res,func,args...);
    }; // integrate

    /**
     *  \brief Generalization of the integrate function which
     *  constructs the result internally and passes to the
     *  integrate function to be incremented by reference.
     *
     */ 
    template <typename T, class F, typename... Args>
    T integrate(const F &func, Args... args) {
      T res(0.);
      integrate<T>(res,func,args...);
      return res;
    }; // integrate

  }; // class SphereIntegrator



  /**
   *  \brief Becke Molecular Integrator
   *
   *  Integrates a molecular integrand using the Becke sphere integration
   *  scheme. Specializes SphereIntegrator
   *
   *  See J. Chem. Phys. 88, 2547 (1988)
   */ 
  template <class _QTyp1>
  class BeckeIntegrator : public SphereIntegrator<_QTyp1> {

    size_t iAtm; ///< Current center for integral

  protected:
 
    MPI_Comm         comm;

    CQMemManager     &memManager_; ///< Memory managment
    Molecule         &molecule_;   ///< Molecule object for nuclear potential
    BasisSet         &basisSet_;   ///< BasisSet for the GTO basis defintion
    SHELL_EVAL_TYPE  typ_      ;   ///< Specification of basis evaluatio requirements
    double           epsScreen_;   ///< Raw screening tolerance
    size_t           NDer;         ///< Number of required basis set derivatives

  public:

    // Defaulted / Deleted ctors
    BeckeIntegrator()                     = delete;
    BeckeIntegrator(const BeckeIntegrator &) = default;
    BeckeIntegrator(BeckeIntegrator &&)      = default;

    /**
     *  \brief BeckeIntegrator constructor.
     *
     *  Constructs a BeckeIntegrator object from a Quadrature scheme for the
     *  radial integration
     */ 
    BeckeIntegrator(MPI_Comm c, CQMemManager &mem, Molecule &mol,
      BasisSet &basis, _QTyp1 g, size_t NAng, size_t NRadPerMacroBatch, 
      SHELL_EVAL_TYPE typ, double epsScreen) :
      comm(c), memManager_(mem),molecule_(mol),basisSet_(basis),typ_(typ),
      epsScreen_(epsScreen),
      SphereIntegrator<_QTyp1>(g,NAng,{0.,0.,0.},1.,NRadPerMacroBatch),
      NDer((typ_ == GRADIENT) ? 4:1){ };

    /**
     *  Functions for the multicenter numerical integration from
     *  J. Chem. Phys. 88, 2547(1988). The following hBecke, gBecke and 
     *  evalPartitionWeights are base on the Ref above.
     */

    inline double hBecke(double x) {return 1.5 * x - 0.5 * x * x * x;}; // Eq. 19
    inline double gBecke(double x) {return hBecke(hBecke(hBecke(x)));}; // Eq. 20 f_3

    // TODO: Put member functions in CXX file
          

    // TODO: revist use of raw memory for batchW
    // FIXME: critical for NAtoms>40
  /**
   *  \brief Evaluates the form the U variables given the V variables.
   *
   *
   *  Function for the multicenter numerical integration from
   *  J. Chem. Phys. 88, 2547(1988).
   *
   *  \param [in]  iCurrent index for the current atomic center on which the shere is centered
   *  \param [in]  rSq Pointer to the square values of the distances of all points to the centers
   *  \param [in/out]  Raw/Updated grid weights
   *
   *
   *  \returns  the max weight for the batch. 
   */  
    double evalPartitionWeights(size_t iCurrent, 
      double *R, std::vector<double> &batchW){

      std::vector<double> partitionScratch(this->molecule_.nAtoms);

      size_t nPointsPerBatch = batchW.size();

      //double curRMin, curRMax;
      double weightMax = 0.0;
      for(size_t iPt = 0; iPt < nPointsPerBatch; iPt++){ 

        std::fill(partitionScratch.begin(),partitionScratch.end(),1.);
        for(size_t iCenter = 0; iCenter < molecule_.nAtoms; iCenter++) {

          double RA = R[iCenter + iPt*molecule_.nAtoms];

          for(size_t jCenter = 0; jCenter < molecule_.nAtoms; jCenter++){
            if(iCenter == jCenter) continue;
            double RB = R[jCenter + iPt*molecule_.nAtoms];
            double mu = (RA - RB) / (molecule_.RIJ[iCenter][jCenter]);
            partitionScratch[iCenter] *= 0.5 * (1.0 - gBecke(mu)); //Eq. 21
          }

        }

        // Normalization
        double sum = 0.0;
        for(size_t iCenter = 0; iCenter < molecule_.nAtoms; iCenter++)
          sum += partitionScratch[iCenter];

        // Updating weights
        batchW[iPt] *= (partitionScratch[iCurrent]/sum);  //Based on Eq. 22
        weightMax = std::max(weightMax, batchW[iPt]);

      } // loop iPt

      return weightMax;

    }; // evalPartitionWeights

  /**
   *  \brief Evaluate the squared distances for all cart_t points in the batchPt 
   *  from all atoms in the molecule and their cartisian compoents.
   *
   *  \param [in]  batchPt Pointer to the x,y,z coordinated for all points in the batch
   *  \param [out] rSq     Pointer to the square values of the distances from all points to the centers
   *  \param [out] cenXYZ  Pointer to the components of the distances from all points to the centers
   *
   */  
    void calcCenDist(std::vector<cart_t> &batchPt, double *rSq, double *R,  
      double* cenXYZ) {
  
      size_t nPointsPerBatch = batchPt.size();

      cart_t rA({0.,0.,0.});
      for(size_t i = 0;       i < nPointsPerBatch;        i++      ) 
      for(size_t iCenter = 0; iCenter < molecule_.nAtoms; iCenter++) {

        rA[0] = batchPt[i][0]  - molecule_.atoms[iCenter].coord[0];
        rA[1] = batchPt[i][1]  - molecule_.atoms[iCenter].coord[1];
        rA[2] = batchPt[i][2]  - molecule_.atoms[iCenter].coord[2];

        rSq[iCenter + i*molecule_.nAtoms ] = 
          rA[0]*rA[0] +rA[1]*rA[1] + rA[2]*rA[2];
        R[iCenter + i*molecule_.nAtoms] = std::sqrt(rSq[iCenter + i*molecule_.nAtoms ]);

        cenXYZ[3*iCenter + 3*i*molecule_.nAtoms + 0] = rA[0];
        cenXYZ[3*iCenter + 3*i*molecule_.nAtoms + 1] = rA[1];
        cenXYZ[3*iCenter + 3*i*molecule_.nAtoms + 2] = rA[2];

      } // loop over points + centers

    }; // calcCenDist

  /**
   *  \brief Integration function according the Becke scheme 
   *
   *  See J. Chem. Phys. 88, 2547(1988).  
   *
   *  \param [in]  res     VXC submatrix for the current batch
   *  \param [in] func    funtion to be integrated (formVXC)
   *  \param [in] arg     several arguments to be passed in 
   */  
    template <typename T, class F, typename... Args>
    void integrate(T &res, const F &func, Args... args) {

#if INT_DEBUG_LEVEL >= 1
    //TIMING
      std::chrono::duration<double> durDist(0.)  ;
      std::chrono::duration<double> durWeight(0.)  ;
      std::chrono::duration<double> durBasis(0.) ;
      std::chrono::duration<double> durFunc(0.)  ;
#endif

      size_t nthreads = GetNumThreads();
      size_t mpiRank  = MPIRank(comm);
      size_t mpiSize  = MPISize(comm);

      assert( mpiSize <= molecule_.nAtoms );

      size_t maxBatchSize      = this->nRadPerMacroBatch * this->q2.nPts;
      size_t maxBatchSizeAtoms = maxBatchSize * molecule_.nAtoms;

      // Allocate Basis scratch
      double *BasisEval = 
        memManager_.template malloc<double>(nthreads*NDer*maxBatchSize*basisSet_.nBasis);

      // Allocate Basis scratch (shell cartisian)
      int LMax = 0;
      for(auto iSh = 0; iSh < basisSet_.nShell; iSh++)
        LMax = std::max(basisSet_.shells[iSh].contr[0].l,LMax);

      size_t shSizeCar = ((LMax+1)*(LMax+2))/2; 
      double * SCR_Car = 
        memManager_.template malloc<double>(nthreads*NDer*shSizeCar);

      // Allocate scratch for Point Distances (squared and components) from each atomic center

      double * cenRSq = memManager_.template malloc<double>(nthreads * maxBatchSizeAtoms);
      double * cenR   = memManager_.template malloc<double>(nthreads * maxBatchSizeAtoms);
      double * cenXYZ = memManager_.template malloc<double>(nthreads * 3*maxBatchSizeAtoms);


      // Populate cutoff Map (for each shell the distance beyond which the shell contribution is negligeble
      std::vector<double> mapSh2Cut;

      double epsilon = 
        std::max((epsScreen_/maxBatchSizeAtoms),std::numeric_limits<double>::epsilon()); 

      // Cutoff radius accordin Eq.20 in J. Chem. Theory Comput. 2011, 7, 3097-3104
      auto cutFunc = [&] (double alpha) -> double{
        return std::sqrt((-std::log(epsilon) + 0.5 * std::log(alpha))/alpha);
      };

      //Evaluate the Max for  each primitive in the shell and than push back in the mapSh2Cut vector
      for(auto iSh = 0; iSh < basisSet_.nShell; iSh++)
        mapSh2Cut.emplace_back(
          cutFunc(
            *std::max_element(
              basisSet_.shells[iSh].alpha.begin(),
              basisSet_.shells[iSh].alpha.end(),[&](double x, double y){
                return cutFunc(x) < cutFunc(y);
              }
            )
          )
        );


      // Function that it is finally integrated
      auto g = [&](T &res, std::vector<cart_t> &batch, std::vector<double> &weights, 
        const std::pair<double,double> &rBounds, Args... args) -> void {

#if INT_DEBUG_LEVEL >= 1
        // Timing
        auto topDist = std::chrono::high_resolution_clock::now();
#endif

        size_t thread_id = GetThreadID();       

        double * BasisEval_loc = BasisEval + thread_id * NDer * maxBatchSize * basisSet_.nBasis;
        double * SCR_Car_loc   = SCR_Car   + thread_id * NDer * shSizeCar;

        double * cenRSq_loc = cenRSq + thread_id * maxBatchSizeAtoms;
        double * cenR_loc   = cenR   + thread_id * maxBatchSizeAtoms;
        double * cenXYZ_loc = cenXYZ + thread_id * 3*maxBatchSizeAtoms;

        // Populate for each batch the distances vectors 
        calcCenDist(batch,cenRSq_loc,cenR_loc,cenXYZ_loc);



        double maxR = rBounds.second;
        double minR = rBounds.first;

        double epsilon = 
          std::max((epsScreen_/maxBatchSizeAtoms),std::numeric_limits<double>::epsilon()); 

#if INT_DEBUG_LEVEL >= 1
        // Timing
        auto botDist = std::chrono::high_resolution_clock::now();
        durDist += botDist - topDist;
#endif
        

        // Populating a vector of bool to know which shell need to 
        // be evaluated for the current batch of points according to 
        // the cutoff distances 
        std::vector<bool> evalShell;
        size_t basisEvalDim(0);
        std::vector<size_t> batchEvalShells;

        // SubMat Vector of pairs specifing the blocks of the super matrix to be used
        std::vector<std::pair<size_t,size_t>> batchSubMat; 

#if INT_DEBUG_LEVEL > 0
           std::cerr << "Screen ON eps= " << epsilon<<std::endl;
        
#endif
        for(auto iSh = 0; iSh < basisSet_.nShell; iSh++) {

          double RAS = molecule_.RIJ[iAtm][basisSet_.mapSh2Cen[iSh]];
          evalShell.emplace_back(
#if INT_DEBUG_LEVEL < 3
           // Note. the spherical shell of point has to be within the shell cutoff 
           // if is on the center or inside the other shell cutoff 
            not (
              (RAS >= (minR + mapSh2Cut[iSh])) or
              (RAS <  (maxR - mapSh2Cut[iSh]))  
            )
#else
            true
#endif
          );

          if(evalShell.back()) {
            basisEvalDim += basisSet_.shells[iSh].size();
            batchEvalShells.emplace_back(iSh);
          }

        }
 
#if INT_DEBUG_LEVEL >= 3
        std::cerr << "BASIS DIM " << basisEvalDim << std::endl;
#endif

        // Skyp the entire batch;
        if(basisEvalDim == 0){ return; }

        //FIXME (Write a function to handle this)
        batchSubMat.emplace_back(
          basisSet_.mapSh2Bf[batchEvalShells[0]],
          basisSet_.mapSh2Bf[batchEvalShells.back()] +
            basisSet_.shells[batchEvalShells.back()].size()
        );

        for(auto iShell = batchEvalShells.begin(); 
            iShell != batchEvalShells.end()-1;
            ++iShell){

          if(*(iShell+1) - (*iShell) != 1){
            batchSubMat.back().second = 
              basisSet_.mapSh2Bf[*iShell] + basisSet_.shells[*iShell].size();

            batchSubMat.emplace_back(
              basisSet_.mapSh2Bf[*(iShell+1)],
              basisSet_.mapSh2Bf[batchEvalShells.back()] +
                basisSet_.shells[batchEvalShells.back()].size()
            );
          }
       
        }

        if(batchEvalShells.size() == 1) 
          batchSubMat[0].second = 
            batchSubMat[0].first + basisSet_.shells[batchEvalShells[0]].size();

#if INT_DEBUG_LEVEL >= 1
        // TIMING
        auto topBasis = std::chrono::high_resolution_clock::now();
#endif
        
        evalShellSet(typ_,basisSet_.shells,evalShell,cenRSq_loc,cenXYZ_loc,batch.size(),molecule_.nAtoms,
          basisSet_.mapSh2Cen,basisEvalDim,BasisEval_loc,SCR_Car_loc,shSizeCar,basisSet_.forceCart);

#if INT_DEBUG_LEVEL >= 1
        // TIMNG
        auto botBasis = std::chrono::high_resolution_clock::now();
        durBasis += botBasis - topBasis;
#endif
        
#if INT_DEBUG_LEVEL >= 1
        // Timing
        auto topWeight = std::chrono::high_resolution_clock::now();
#endif

        // Modify weight according Becke scheme, get max weight
        auto maxWeight = evalPartitionWeights(iAtm,cenR_loc,weights); 

#if INT_DEBUG_LEVEL >= 1
        // Timing
        auto botWeight = std::chrono::high_resolution_clock::now();
        durWeight += botWeight - topWeight;
#endif

#if INT_DEBUG_LEVEL < 3
         if (std::abs(maxWeight) < epsilon) {
           //std::cerr << "batch screened" << std::endl;
           return;
         }
#endif


#if INT_DEBUG_LEVEL >= 1
        auto topFunc = std::chrono::high_resolution_clock::now();
#endif


        // Final call to be resambled ba the lambda function
        func(res,batch,weights,basisEvalDim,BasisEval_loc,batchEvalShells,batchSubMat,args...);

#if INT_DEBUG_LEVEL >= 1
        auto botFunc = std::chrono::high_resolution_clock::now();
        // TIMNG
        durFunc += botFunc - topFunc;
#endif
        
      }; // End g function

      // Perform integration over atomic centers, by using spherical Integrators
      for(iAtm = 0; iAtm < molecule_.nAtoms; iAtm++) {


        // Round robin on mpi processes
        if( iAtm % mpiSize != mpiRank ) continue;

        this->Center = {molecule_.atoms[iAtm].coord[0],
                        molecule_.atoms[iAtm].coord[1],
                        molecule_.atoms[iAtm].coord[2]};

        // The effective radius is chosen as half of the Bragg-Slater radius of the 
        // respective atom (stored in the slaterRadius in Ang), except for
        // hydrogen in which case the factor of 0.5 is not applied (the stored value
        // for hydrogen is pre scaled by 2 to prevent scaling).
        // Procedure according J. Chem. Phys. 88, 2547(1988). pg 2550 
          
        this->Scale = 0.5*molecule_.atoms[iAtm].slaterRadius/AngPerBohr;
        SphereIntegrator<_QTyp1>::template integrate<T>(1.,res,g,args...);

      } // loop over atoms
      res *= 4.* M_PI;

      // clean memory
      memManager_.free(BasisEval,cenRSq,cenXYZ,cenR,SCR_Car);

#if INT_DEBUG_LEVEL >= 1
      //TIMING
      double d_batch =  molecule_.nAtoms *  this->q1.nPts /  this->nRadPerMacroBatch;
      std::cerr << std::scientific << std::endl;
      std::cerr << "Total # of Batch  " << d_batch << std::endl;
      std::cerr << "Total Dist " << durDist.count() << std::endl;
      std::cerr << "Total Weight " << durWeight.count() << std::endl;
      std::cerr << "Total Basis " << durBasis.count() << std::endl;
      std::cerr << "Total Func " << durFunc.count() << std::endl;
      std::cerr << "Total int the gfun " <<  durDist.count()+ durBasis.count() +durFunc.count() 
                << std::endl;

      std::cerr << "Dist " << durDist.count()/d_batch << std::endl;
      std::cerr << "Weight " << durWeight.count()/d_batch << std::endl;
      std::cerr << "Basis " << durBasis.count()/d_batch << std::endl;
      std::cerr << "Func " << durFunc.count()/d_batch << std::endl;
#endif

    };// integrate

    /**
     *  \brief Generalization of the integrate function which
     *  constructs the result internally and passes to the
     *  integrate function to be incremented by reference.
     *
     */ 
    template <typename T, class F, typename... Args>
    T integrate(const F &func, Args... args) {
      T res(0.);
      integrate<T>(res,func,args...);
      return res;
    }

  };// class BeckeIntegrator 


}; // namespace ChronusQ

