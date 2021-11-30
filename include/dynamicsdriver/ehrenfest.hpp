/* 
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *  
 *  Copyright (C) 2014-2019 Li Research Group (University of Washington)
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
#ifndef __INCLUDED_EHRENFEST_HPP__
#define __INCLUDED_EHRENFEST_HPP__

#include <chronusq_sys.hpp>
#include <cerr.hpp>
#include <memmanager.hpp>
#include <realtime.hpp>


// Ehrenfest Headers


namespace ChronusQ {

  /**
   *  \brief A struct to store information pertinent to the time
   *  propagation procedure.
   */ 
  struct NucIntegrationScheme {

    double MaxTN    = 0.1;  ///< Max simulation time in AU
    double dTN      = 0.01; ///< Time-step in AU
    size_t N        = 5;    ///< N parameter
    size_t M        = 4;    ///< M paramater

  }; // struct IntegrationScheme

  struct EhrenfestBase {

    typedef std::vector<std::array<double, 3>> NucVec;

    SafeFile savFile;  ///< Data File

    NucIntegrationScheme intScheme; ///< Nuclear integration scheme

    size_t nAtoms;     ///< Number of atoms
    std::vector<std::array<double, 3>> current_x;  ///< Nuclear position
    std::vector<std::array<double, 3>> current_p;  ///< Nuclear momentum
    std::vector<double> Mass; ///< Nuclear mass in atomic unit 
    std::vector<bool> isQuantum; ///< Quantum nuclei or not
    std::vector<bool> isGhost;   

    std::array<double, 3> proton_velocity; ///< Proton velocity

    double kineticEng = 0.0; ///< Nuclear Kinetic Energy
    double basisCenterKE = 0.0; ///< Basis Center Kinetic Energy

    bool init_pert = false; ///< whether to perturb the geometry at time zero

    double pert_val_x = 1e-5; ///< perturbation value
    double pert_val_y = 1e-5; ///< perturbation value
    double pert_val_z = 1e-5; ///< perturbation value

    bool move_pb = false; ///< whether to move the proton basis

    bool velocity_matching = false;

    bool fix_first = false;
    bool fix_second = false;
    bool fix_third = false;

    bool read_prev = false;

    double mass_scale = 1.0;

    double hist_time = 0.0;

    std::shared_ptr<RealTimeBase> rt; ///< Pointer to the real-time object
    std::shared_ptr<RealTimeBase> aux_rt=nullptr; ///< Pointer to the auxiliary real-time object

    //EhrenfestBase()                      = delete;
    EhrenfestBase(const EhrenfestBase &) = delete;
    EhrenfestBase(EhrenfestBase &&)      = delete;

    //EhrenfestBase( CQMemManager &memManager): memManager_(memManager) { }
    EhrenfestBase() { }

    
    // EhrenfestBase procedural functions
    virtual void doSimulation()      = 0;
    //virtual void doSimulation2()     = 0;
    //virtual void bomd()  = 0;

    // get RT pointer 
    void getRT(std::shared_ptr<RealTimeBase> _rt, std::shared_ptr<RealTimeBase> _aux_rt=nullptr)
    {
      // main
      rt = _rt;

      // aux
      if(_aux_rt)
        aux_rt = _aux_rt;

    };

    // Verlet update position 
    virtual void updateX(NucVec& input_x, const NucVec& input_p, const double dt) = 0;

    // Verlet update momentum 
    virtual void updateP(NucVec& output_p, const NucVec& intput_p, const NucVec& g, const double dt, bool updateQPvelocity = false) = 0; 

    // Update integrals
    virtual void updateInt(NucVec& input_x, double time) = 0;

    // compute proton velocity
    virtual void computeProtonVelocity(const NucVec& intput_p, const NucVec& g, const double dt) = 0;

    // save the current state of the wave function and coordinates
    virtual void saveCurrentState() = 0;

    // read the saved state 
    virtual void readSavedState() = 0;

    // Compute nuclear kinetic energy
    void computeKinetic() {

      // reset the kinetic energy
      kineticEng = 0.;

      // reset the basis center kinetic energy
      basisCenterKE = 0.;

      for(size_t ic = 0; ic < nAtoms; ic++) {
        if (isQuantum[ic])
          continue;
        kineticEng += 0.5 * current_p[ic][0]*current_p[ic][0] / Mass[ic]; 
        kineticEng += 0.5 * current_p[ic][1]*current_p[ic][1] / Mass[ic]; 
        kineticEng += 0.5 * current_p[ic][2]*current_p[ic][2] / Mass[ic]; 
        if (isQuantum[ic]) {
          basisCenterKE += 0.5 * current_p[ic][0]*current_p[ic][0] / Mass[ic];
          basisCenterKE += 0.5 * current_p[ic][1]*current_p[ic][1] / Mass[ic];
          basisCenterKE += 0.5 * current_p[ic][2]*current_p[ic][2] / Mass[ic];
        }
      }

    };

    // Progress functions
    void printEFHeader();
    void printEFStep(double Time);

  //protected:
  //  
  //  CQMemManager      &memManager_;  ///< Memory manager

  };

  template<template <typename, typename> class _SSTyp, typename IntsT>
  class Ehrenfest : public EhrenfestBase {

    typedef dcomplex*              oper_t;
    typedef std::vector<oper_t>    oper_t_coll;

  public: 

    
    // Constructors

    // Disable default, copy and move constructors
    Ehrenfest()                  = delete;
    Ehrenfest(const Ehrenfest &) = delete;
    Ehrenfest(Ehrenfest &&)      = delete;

    /**
     *  \brief Ehrenfest Constructor
     *  
     * 
     */
    template <typename RefMatsT>
    Ehrenfest(_SSTyp<RefMatsT,IntsT> &ss) : 
      EhrenfestBase() {
      
      //rt = _rt;
      //aux_rt = _aux_rt;

      //std::cout << "constructor of EF class" << std::endl;

      // electron slater determinant
      //std::shared_ptr<_SSTyp<dcomplex, IntsT>> elec_propagator_ptr = std::dynamic_pointer_cast<_SSTyp<dcomplex, IntsT>>(rt->propagator_);
      //auto elec_propagator_ptr = rt->get_propagator_ptr();

      // number of atoms 
      nAtoms = ss.aoints.molecule_.atoms.size();

      // initialize x and p
      current_x.resize(nAtoms);
      current_p.resize(nAtoms);

      // get the mass and isquantum for nuclei
      for(size_t ic = 0; ic < nAtoms; ic++) {
        // nuclear mass
        Mass.emplace_back(1836.15267343 * ss.aoints.molecule_.atoms[ic].atomicMass);

        //std::cout << "Mass is " << std::endl;
        //std::cout << Mass.back() << std::endl;

        // quantum nuclei or not
        //isQuantum.emplace_back( (ss.aoints.molecule_.atoms[ic].quantum or ss.aoints.molecule_.atoms[ic].atomicNumber == 0) );
        isQuantum.emplace_back( (ss.aoints.molecule_.atoms[ic].quantum ) );

        // scale the quantum proton's mass
        if(isQuantum.back())
          Mass.back() *= mass_scale;

        isGhost.emplace_back( ss.aoints.molecule_.atoms[ic].atomicNumber == 0 );


        // nuclei position 
        current_x[ic][0] = ss.aoints.molecule_.atoms[ic].coord[0]; // x 
        current_x[ic][1] = ss.aoints.molecule_.atoms[ic].coord[1]; // y
        current_x[ic][2] = ss.aoints.molecule_.atoms[ic].coord[2]; // z
        
        // nuclear momentum 
        current_p[ic][0] = 0.0; // x
        current_p[ic][1] = 0.0; // y
        current_p[ic][2] = 0.0; // z

        if(isQuantum.back())
          proton_velocity = {0.0, 0.0, 0.0};

      }
    }; // Ehrenfest constructor

    //~Ehrenfest(){ dealloc(); }

    // Ehrenfest procedural functions
    void doSimulation();
    void doSimulation2();
    //void bomd();

    // Verlet update position 
    void updateX(NucVec& input_x, const NucVec& input_p, const double dt);

    // Verlet update position but the quantum proton
    void updateXNoQP(NucVec& input_x, const NucVec& input_p, const double dt);

    // Verlet update momentum 
    void updateP(NucVec& output_p, const NucVec& input_p, const NucVec& g, const double dt, bool updateQPvelocity = false); 

    // Update integrals
    void updateInt(NucVec& input_x, double time);

    // compute proton velocity
    void computeProtonVelocity(const NucVec& intput_p, const NucVec& g, const double dt);

    // Scale mass of quantum proton
    void scaleProtonM();

    // Perturb the position of the first atom
    void pertFirst();

    // Progress functions
    void printEFHeader();
    void printEFStep(double Time);

    // save the current state of the wave function and coordinates
    virtual void saveCurrentState();

    // read the saved state 
    virtual void readSavedState();

    // Memory functions
    //void alloc();
    //void dealloc();

  }; // class Ehrenfest


}; // namespace ChronusQ

#endif
