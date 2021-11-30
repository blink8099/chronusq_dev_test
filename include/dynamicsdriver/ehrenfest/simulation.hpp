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
#ifndef __INCLUDED_EHRENFEST_SIMULATION_HPP__
#define __INCLUDED_EHRENFEST_SIMULATION_HPP__

#include <ehrenfest.hpp>
#include <physcon.hpp>
#include <util/time.hpp>
#include <chrono>


namespace ChronusQ {

  template<template <typename, typename> class _SSTyp, typename IntsT>
  void Ehrenfest<_SSTyp,IntsT>::doSimulation() {

    printEFHeader();

    // current time 
    double Time;

    // TNe 
    double dTNe = intScheme.dTN / intScheme.N;

    // Te
    double dTe  = dTNe / intScheme.M;

    // p(t + 0.5 * dTN)
    NucVec p_half(nAtoms);

    // p used to determine midpoint fock update
    NucVec p_fock(nAtoms);

    // x of the next time-step 
    NucVec x_next(nAtoms);

    // x used in midpoint fock update
    NucVec x_fock(nAtoms);

    // initialize tmp x and p as zeros
    for(size_t ic = 0; ic < nAtoms; ic++) {
      p_half[ic] = {0.0,0.0,0.0};
      p_fock[ic] = {0.0,0.0,0.0};
      x_next[ic] = {0.0,0.0,0.0};
      x_fock[ic] = {0.0,0.0,0.0};
    }

    std::shared_ptr<_SSTyp<dcomplex, IntsT>> elec_propagator_ptr = std::dynamic_pointer_cast<_SSTyp<dcomplex, IntsT>>(rt->propagator_);
    std::shared_ptr<_SSTyp<dcomplex, IntsT>> prot_propagator_ptr = nullptr;
    if(aux_rt) 
      prot_propagator_ptr = std::dynamic_pointer_cast<_SSTyp<dcomplex, IntsT>>(aux_rt->propagator_);

    // scale proton mass
    if(aux_rt)
      scaleProtonM();

    if(aux_rt and init_pert) {
      pertFirst();
      // compute unorthogonalized density matrix
      elec_propagator_ptr->ortho2aoDen(); 
      if (prot_propagator_ptr)
        prot_propagator_ptr->ortho2aoDen();
    }

    // Gradient
    NucVec Grad(nAtoms);

    // Proton Kinetic energy when trans_pb is true
    double PKine = 0.;

    // Previous step energy 
    double ETot_prev = 0.;

    // proton expectation value
    std::array<double, 3> cur_prot_pos, old_prot_pos, diff_prot_pos;

    // read in previous state 
    if (read_prev)
      readSavedState(); 

    // loop over times
    for( Time = 0.; Time <= intScheme.MaxTN; Time += intScheme.dTN ) {
     
      std::cout << "hist time is " << hist_time << std::endl;

      // check idenpotency
      if(aux_rt)
        prot_propagator_ptr->checkIdempotency();

      // perturbation 
      EMPerturbation pert_t = rt->pert.getPert(Time + hist_time);

      // compute one-body contribution to gradient at current time
      if( Time == 0. )
        rt->propagator_->formCoreH(pert_t);

      //auto GradStart = tick();

      // compute the two-body contribution to the gradient
      // !!! This assumes that the Fock matrix has already been
      // calculated
      std::dynamic_pointer_cast<RealTime<_SSTyp,IntsT>>(rt)->formFockGrad(Time + hist_time);

      // compute gradient
      rt->propagator_->computeGradients();

      // add contribution from auxiliary system if doing NEO
      if(aux_rt) {
        if( Time == 0. )
          aux_rt->propagator_->formCoreH(pert_t);
        std::dynamic_pointer_cast<RealTime<_SSTyp,IntsT>>(aux_rt)->formFockGrad(Time + hist_time);
        aux_rt->propagator_->computeGradients();
        rt->propagator_->addAuxGradients();

        // print out total gradients
        std::cout << "total Energy Gradient (electron and proton) is " << std::endl;
        for (size_t ic = 0; ic < rt->propagator_->GradtotalEnergy.size(); ic++) {
          for (size_t XYZ = 0; XYZ != 3; XYZ++) {
            std::cout << std::setprecision(8);
            std::cout << std::setw(16) << rt->propagator_->GradtotalEnergy[ic][XYZ] << " ";
          }
          std::cout << std::endl;
        }
      }

      // disable the calculation of gradients 
      elec_propagator_ptr->aoints.doGrad = false;
      if(aux_rt)
        prot_propagator_ptr->aoints.doGrad = false;

      // get the gradients
      for(size_t ic = 0; ic < nAtoms; ic++) {
        Grad[ic][0] = rt->propagator_->GradtotalEnergy[ic][0];
        Grad[ic][1] = rt->propagator_->GradtotalEnergy[ic][1];
        Grad[ic][2] = rt->propagator_->GradtotalEnergy[ic][2];
      }

      // if this is not the first step
      if (Time != 0.) 
        // update from p(t+0.5*dTN) -> p(t+dTN)
        updateP(current_p,p_half,Grad,intScheme.dTN);

      // save the current state
      //saveCurrentState();

      // compute nuclear kinetic energy 
      this->computeKinetic();

      // compute energy
      elec_propagator_ptr->computeProperties(pert_t);
      if(aux_rt)
        prot_propagator_ptr->computeProperties(pert_t);

      
      // print information for this step 
      //printEFStep(Time);
      // print out energy
      double totalEnergy = this->rt->propagator_->totalEnergy; 

      // add nuclear kinetic energy 
      //totalEnergy += kineticEng;

      if(aux_rt) {
        totalEnergy += this->aux_rt->propagator_->OBEnergy;
        totalEnergy += this->aux_rt->propagator_->PPEnergy;
        totalEnergy += this->aux_rt->propagator_->FieldEnergy;
      }

      std::cout << "Ehrenfest Step " << int(Time / intScheme.dTN) << " Time " << (Time + hist_time) * FSPerAUTime << " fs " << std::endl; 

      std::cout << "EPot " << std::setw(16) << std::setprecision(12) << totalEnergy << " EKin " << std::setw(16) << kineticEng << " ETot " << std::setw(16) << totalEnergy + kineticEng << std::endl; 
      std::cout << "EProt " << std::setprecision(12) << PKine << std::endl;

      // print out nulear position
      std::cout << "Nuclear Position (Angstrom):" << std::endl;
      for(size_t ic = 0; ic < this->nAtoms; ic++) {
        std::cout << std::setw(16) << std::setprecision(8) << this->current_x[ic][0] * AngPerBohr;
        std::cout << std::setw(16) << std::setprecision(8) << this->current_x[ic][1] * AngPerBohr;
        std::cout << std::setw(16) << std::setprecision(8) << this->current_x[ic][2] * AngPerBohr;
        std::cout << std::endl;
      }

      // print out nulear momentum
      std::cout << "Nuclear Momentum:" << std::endl;
      for(size_t ic = 0; ic < this->nAtoms; ic++) {
        std::cout << std::setw(16) << std::setprecision(8) << this->current_p[ic][0];
        std::cout << std::setw(16) << std::setprecision(8) << this->current_p[ic][1];
        std::cout << std::setw(16) << std::setprecision(8) << this->current_p[ic][2];
        std::cout << std::endl;
      }

      // compute p(t+0.5*dTN) -> p_half
      updateP(p_half,current_p,Grad,intScheme.dTN,true);

      // compute x(t+dTN)
      updateX(x_next,p_half,intScheme.dTN);

      // j loop
      double Tp = Time;
      for(size_t j = 1; j <= intScheme.N; j++) {

        // compute t'
        Tp = Time + (j - 1) * dTNe;
        
        // compute x(t'+0.5*deltaT_Ne)
        updateX(x_fock,p_half,Tp-Time+0.5*dTNe);

        // Trial: compute velocity(t'+0.5*deltaT_Ne)
        //if (not velocity_matching)
        computeProtonVelocity(current_p,Grad,Tp-Time+0.5*dTNe);

        std::cout << "Midpoint Fock Step " << j << " Time " << (Tp + hist_time) * FSPerAUTime << " fs " << std::endl; 

        // print out nulear position
        std::cout << "Midpoint Fock is Computed with Nuclear Positions (Angstrom):" << std::endl;
        for(size_t ic = 0; ic < this->nAtoms; ic++) {
          std::cout << std::setw(16) << std::setprecision(8) << x_fock[ic][0] * AngPerBohr;
          std::cout << std::setw(16) << std::setprecision(8) << x_fock[ic][1] * AngPerBohr;
          std::cout << std::setw(16) << std::setprecision(8) << x_fock[ic][2] * AngPerBohr;
          std::cout << std::endl;
        }

        // update one- and two-electron integrals
        updateInt(x_fock,Tp);

        // compute unorthogonalized density matrix
        elec_propagator_ptr->ortho2aoDen(); 
        if (prot_propagator_ptr)
          prot_propagator_ptr->ortho2aoDen();

        // set time step and propagation length
        rt->intScheme.tMax = (intScheme.M - 1) * dTe;
        if (Time == 0.)
          rt->intScheme.deltaT = dTe;

        // get the quantum proton's velocity for NEO calculation
        if (prot_propagator_ptr and move_pb)
          rt->aux_rt->proton_velocity = proton_velocity;
        else if (prot_propagator_ptr and not move_pb)
          rt->aux_rt->proton_velocity = {0.0, 0.0, 0.0};         

        //rt->aux_rt->proton_velocity = {0.0, 0.0, 0.0};

        if (aux_rt) {
          std::cout << "proton velocity" << std::endl;
          std::cout << proton_velocity[0] << " " << proton_velocity[1] << " " << proton_velocity[2] << std::endl;
        }

        // propagate density matrices
        //auto PropStart = tick();
        rt->doPropagation(Tp+hist_time);
        //double PropDur = tock(PropStart);
        //std::cout << "Propagation time " << PropDur << std::endl; 

        // additional step
        if (j == intScheme.N) {
        
          updateX(x_fock,p_half,intScheme.dTN);
          //updateXNoQP(x_fock,p_half,intScheme.dTN);

          std::cout << "Midpoint Fock Step " << j+1 << " Time " << (Time+intScheme.dTN+hist_time)*FSPerAUTime << " fs " << std::endl; 

          // print out nulear position
          std::cout << "Midpoint Fock is Computed with Nuclear Positions (Angstrom):" << std::endl;
          for(size_t ic = 0; ic < this->nAtoms; ic++) {
            std::cout << std::setw(16) << std::setprecision(8) << x_fock[ic][0] * AngPerBohr;
            std::cout << std::setw(16) << std::setprecision(8) << x_fock[ic][1] * AngPerBohr;
            std::cout << std::setw(16) << std::setprecision(8) << x_fock[ic][2] * AngPerBohr;
            std::cout << std::endl;
          }

          // enable the calculation of gradients 
          elec_propagator_ptr->aoints.doGrad = true;
          if(aux_rt)
            prot_propagator_ptr->aoints.doGrad = true;

          // update integrals and its gradients
          // this also recomputes gradient of S^{-1/2}
          updateInt(x_fock,Time+intScheme.dTN+hist_time);

          // compute unorthogonalized density matrix
          // based on the orthogonalized density matrix 
          // and updated S^{1/2}
          elec_propagator_ptr->ortho2aoDen(); 
          if (prot_propagator_ptr)
            prot_propagator_ptr->ortho2aoDen();

          // form Fock matrix at x(t+dtN) with updated integrals and i
          // un-orthogonalized density matrix
          // so that gradient could be computed correctly
          std::dynamic_pointer_cast<RealTime<_SSTyp,IntsT>>(rt)->formFock(false, Tp+dTNe);
          if(aux_rt)
            std::dynamic_pointer_cast<RealTime<_SSTyp,IntsT>>(aux_rt)->formFock(false, Tp+dTNe);
        }

      } // j loop
      
      // update position x(t+dt)
      updateX(current_x, p_half, intScheme.dTN);

    } // outer loop

    // compute the two-body contribution to the gradient
    std::dynamic_pointer_cast<RealTime<_SSTyp,IntsT>>(rt)->formFockGrad(intScheme.MaxTN);

    // compute gradient
    rt->propagator_->computeGradients();

    // add contribution from auxiliary system if doing NEO
    if(aux_rt) {
      std::dynamic_pointer_cast<RealTime<_SSTyp,IntsT>>(aux_rt)->formFockGrad(intScheme.MaxTN);
      aux_rt->propagator_->computeGradients();
      rt->propagator_->addAuxGradients();

      // print out total gradients
      std::cout << "total Energy Gradient (electron and proton) is " << std::endl;
      for (size_t ic = 0; ic < rt->propagator_->GradtotalEnergy.size(); ic++) {
        for (size_t XYZ = 0; XYZ != 3; XYZ++) {
          std::cout << std::setprecision(8);
          std::cout << std::setw(16) << rt->propagator_->GradtotalEnergy[ic][XYZ] << " ";
        }
        std::cout << std::endl;
      }
    }

    // get the gradients
    for(size_t ic = 0; ic < nAtoms; ic++) {
      Grad[ic][0] = rt->propagator_->GradtotalEnergy[ic][0];
      Grad[ic][1] = rt->propagator_->GradtotalEnergy[ic][1];
      Grad[ic][2] = rt->propagator_->GradtotalEnergy[ic][2];
    }

    // update from p(t+0.5*dTN) -> p(t+dTN)
    updateP(current_p,p_half,Grad,intScheme.dTN);

    // print out nulear position
    std::cout << "Nuclear Position (Angstrom):" << std::endl;
    for(size_t ic = 0; ic < this->nAtoms; ic++) {
      std::cout << std::setw(16) << std::setprecision(8) << this->current_x[ic][0] * AngPerBohr;
      std::cout << std::setw(16) << std::setprecision(8) << this->current_x[ic][1] * AngPerBohr;
      std::cout << std::setw(16) << std::setprecision(8) << this->current_x[ic][2] * AngPerBohr;
      std::cout << std::endl;
    }

    // print out nulear momentum
    std::cout << "Nuclear Momentum:" << std::endl;
    for(size_t ic = 0; ic < this->nAtoms; ic++) {
      std::cout << std::setw(16) << std::setprecision(8) << this->current_p[ic][0];
      std::cout << std::setw(16) << std::setprecision(8) << this->current_p[ic][1];
      std::cout << std::setw(16) << std::setprecision(8) << this->current_p[ic][2];
      std::cout << std::endl;
    }

    // save the current state
    saveCurrentState();

  }; // doSimulation

  template<template <typename, typename> class _SSTyp, typename IntsT>
  void Ehrenfest<_SSTyp, IntsT>::updateX(NucVec& input_x, const NucVec& input_p,
                                  const double dt) 
  {
    size_t quantumIndex = 0;

    // update position based on momentum 
    for(size_t ic = 0; ic < nAtoms; ic++) {
      if(fix_first and ic == 0) {
        input_x[ic][0] = current_x[ic][0];
        input_x[ic][1] = current_x[ic][1];
        input_x[ic][2] = current_x[ic][2];
        continue;
      }
      if(fix_second and ic == 1) {
        input_x[ic][0] = current_x[ic][0];
        input_x[ic][1] = current_x[ic][1];
        input_x[ic][2] = current_x[ic][2];
        continue;
      }
      if(fix_third and ic == 2) {
        input_x[ic][0] = current_x[ic][0];
        input_x[ic][1] = current_x[ic][1];
        input_x[ic][2] = current_x[ic][2];
        continue;
      }
      // skip quantum nuclei
      if(not isQuantum[ic] or move_pb) {
        input_x[ic][0] = current_x[ic][0] + (input_p[ic][0] * dt / Mass[ic]);
        input_x[ic][1] = current_x[ic][1] + (input_p[ic][1] * dt / Mass[ic]);
        input_x[ic][2] = current_x[ic][2] + (input_p[ic][2] * dt / Mass[ic]);
        if (isQuantum[ic])
          quantumIndex = ic;          
      }
      else {
        input_x[ic][0] = current_x[ic][0];
        input_x[ic][1] = current_x[ic][1];
        input_x[ic][2] = current_x[ic][2];
      }
      //if(isGhost[ic] and move_pb) {
      //  input_x[ic][0] = current_x[ic][0] + (input_p[quantumIndex][0] * dt / Mass[ic]);
      //  input_x[ic][1] = current_x[ic][1] + (input_p[quantumIndex][1] * dt / Mass[ic]);
      //  input_x[ic][2] = current_x[ic][2] + (input_p[quantumIndex][2] * dt / Mass[ic]);
      //}
    }

  }; // updateX

  template<template <typename, typename> class _SSTyp, typename IntsT>
  void Ehrenfest<_SSTyp, IntsT>::updateXNoQP(NucVec& input_x, const NucVec& input_p,
                                             const double dt) 
  {

    // update position based on momentum 
    for(size_t ic = 0; ic < nAtoms; ic++) {
      // skip quantum nuclei
      if(not isQuantum[ic]) {
        input_x[ic][0] = current_x[ic][0] + (input_p[ic][0] * dt / Mass[ic]);
        input_x[ic][1] = current_x[ic][1] + (input_p[ic][1] * dt / Mass[ic]);
        input_x[ic][2] = current_x[ic][2] + (input_p[ic][2] * dt / Mass[ic]);
      }
      else {
        input_x[ic][0] = current_x[ic][0];
        input_x[ic][1] = current_x[ic][1];
        input_x[ic][2] = current_x[ic][2];
      }
    }

  }; // updateX

  template<template <typename, typename> class _SSTyp, typename IntsT>
  void Ehrenfest<_SSTyp, IntsT>::updateP(NucVec& output_p, const NucVec& input_p, 
                                         const NucVec& g, const double dt, 
                                         bool updateQPvelocity) 
  {

    // update position based on momentum 
    for(size_t ic = 0; ic < nAtoms; ic++) {
      if(fix_first and ic == 0) {
        output_p[ic][0] = current_p[ic][0];
        output_p[ic][1] = current_p[ic][1];
        output_p[ic][2] = current_p[ic][2];
        continue;
      }
      if(fix_second and ic == 1) {
        output_p[ic][0] = current_p[ic][0];
        output_p[ic][1] = current_p[ic][1];
        output_p[ic][2] = current_p[ic][2];
        continue;
      }
      if(fix_third and ic == 2) {
        output_p[ic][0] = current_p[ic][0];
        output_p[ic][1] = current_p[ic][1];
        output_p[ic][2] = current_p[ic][2];
        continue;
      }
      // skip quantum nuclei
      if(not isQuantum[ic] or move_pb) {
        output_p[ic][0] = input_p[ic][0] - 0.5 * g[ic][0] * dt;
        output_p[ic][1] = input_p[ic][1] - 0.5 * g[ic][1] * dt;
        output_p[ic][2] = input_p[ic][2] - 0.5 * g[ic][2] * dt;
      }
      else {
        output_p[ic][0] = 0.0;
        output_p[ic][1] = 0.0;
        output_p[ic][2] = 0.0;
      }
      if (isGhost[ic]) {
        output_p[ic][0] = 0.0;
        output_p[ic][1] = 0.0;
        output_p[ic][2] = 0.0;
      }

      if ( (isQuantum[ic] or isGhost[ic]) and updateQPvelocity) {
        // velocity matching algorithn
        if (this->velocity_matching) {
          // compute R dot
          std::dynamic_pointer_cast<RealTime<_SSTyp,IntsT>>(aux_rt)->formRdot(); 

          // get proton velocity from RT
          proton_velocity = aux_rt->proton_velocity;

          // output p
          output_p[ic][0] = proton_velocity[0] * Mass[ic];
          output_p[ic][1] = proton_velocity[1] * Mass[ic];
          output_p[ic][2] = proton_velocity[2] * Mass[ic];

          std::cout << "vx is " << proton_velocity[0] << std::endl;
        }
        else 
          proton_velocity = {output_p[ic][0] / Mass[ic], output_p[ic][1] / Mass[ic], output_p[ic][2] / Mass[ic]};
      }
    }

  }; // updateP


  template<template <typename, typename> class _SSTyp, typename IntsT>
  void Ehrenfest<_SSTyp, IntsT>::computeProtonVelocity(const NucVec& input_p, 
                                                       const NucVec& g, 
                                                       const double dt)
  {

    if (not this->velocity_matching) {
      // update position based on momentum 
      for(size_t ic = 0; ic < nAtoms; ic++) {
        // skip quantum nuclei
        if(isQuantum[ic]) {
          proton_velocity[0] = (input_p[ic][0] - g[ic][0] * dt) / Mass[ic];
          proton_velocity[1] = (input_p[ic][1] - g[ic][1] * dt) / Mass[ic];
          proton_velocity[2] = (input_p[ic][2] - g[ic][2] * dt) / Mass[ic];
        }
      }
    }
    else {
      // compute R dot
      std::dynamic_pointer_cast<RealTime<_SSTyp,IntsT>>(aux_rt)->formRdot(); 

      // get proton velocity from RT
      proton_velocity = aux_rt->proton_velocity;
    }

  }; // updateP


  template<template <typename, typename> class _SSTyp, typename IntsT>
  void Ehrenfest<_SSTyp, IntsT>::scaleProtonM() 
  {

    // loop over atoms
    for(size_t ic = 0; ic < nAtoms; ic++) {
      // zero out quantum nuclei momentum
      if(isQuantum[ic]) {
        Mass[ic] *= mass_scale;
      }
    }

  }; // scaleProtonM

  template<template <typename, typename> class _SSTyp, typename IntsT>
  void Ehrenfest<_SSTyp,IntsT>::updateInt(NucVec& input_x, double time) 
  {
    // get pointers for determinant objects
    std::shared_ptr<_SSTyp<dcomplex, IntsT>> elec_propagator_ptr = std::dynamic_pointer_cast<_SSTyp<dcomplex, IntsT>>(rt->propagator_);
    std::shared_ptr<_SSTyp<dcomplex, IntsT>> prot_propagator_ptr = nullptr;
    if(aux_rt) 
      prot_propagator_ptr = std::dynamic_pointer_cast<_SSTyp<dcomplex, IntsT>>(aux_rt->propagator_);

    // update molecule object 
    elec_propagator_ptr->aoints.molecule_.updateClassicalNucPos(input_x);

    if(move_pb)
     elec_propagator_ptr->aoints.molecule_.updateQNpos(input_x);     

    // update basis object for the main system 
    elec_propagator_ptr->aoints.basisSet_.updateForMoveNuclei(elec_propagator_ptr->aoints.molecule_);

    // update basis object for the proton system 
    if(prot_propagator_ptr)
      prot_propagator_ptr->aoints.basisSet_.updateForMoveNuclei(prot_propagator_ptr->aoints.molecule_);         

    // perturbation 
    EMPerturbation pert_t = rt->pert.getPert(time);

    // recompute one-electron integrals 
    elec_propagator_ptr->formCoreH(pert_t);

    // recompute one-electron integrals for proton system
    if(prot_propagator_ptr)
      prot_propagator_ptr->formCoreH(pert_t);

    // Recompute two-electron integrals if using INCORE algorithm
    if(elec_propagator_ptr->aoints.contrAlg == INCORE )
      elec_propagator_ptr->aoints.computeERI(pert_t);

    if(prot_propagator_ptr) {
      if(elec_propagator_ptr->aoints.contrAlg == INCORE ) {
        elec_propagator_ptr->aoints.computeEPAI(pert_t, dynamic_cast<AOIntegralsBase&>(prot_propagator_ptr->aoints), true);
      }

      if(prot_propagator_ptr->aoints.contrAlg == INCORE ) {
        prot_propagator_ptr->aoints.computeEPAI(pert_t, dynamic_cast<AOIntegralsBase&>(elec_propagator_ptr->aoints), false);

        if(move_pb)
          prot_propagator_ptr->aoints.computeERI(pert_t);
      }
    }
  }; // updateP

  template<template <typename, typename> class _SSTyp, typename IntsT>
  void Ehrenfest<_SSTyp,IntsT>::pertFirst() {

    current_x[0][0] += pert_val_x;
    current_x[0][1] += pert_val_y;
    current_x[0][2] += pert_val_z;
    updateInt(current_x, 0.0);

  };

  /**
   *  \brief Saves the current state of wave function and nuclear coordinates
   *
   */
  template<template <typename, typename> class _SSTyp, typename IntsT>
  void Ehrenfest<_SSTyp,IntsT>::saveCurrentState() {
 
    // Checkpoint if file exists
    if ( savFile.exists() ) {

      // get pointers for determinant objects
      std::shared_ptr<_SSTyp<dcomplex, IntsT>> elec_propagator_ptr = std::dynamic_pointer_cast<_SSTyp<dcomplex, IntsT>>(rt->propagator_);
      std::shared_ptr<_SSTyp<dcomplex, IntsT>> prot_propagator_ptr = nullptr;
      if(aux_rt) 
        prot_propagator_ptr = std::dynamic_pointer_cast<_SSTyp<dcomplex, IntsT>>(aux_rt->propagator_);

      // get size of the electronic density matrix 
      size_t NB = elec_propagator_ptr->aoints.basisSet().nBasis;

      // get size of the protonic density matrix 
      size_t auxNB;
      if (aux_rt)
        auxNB = prot_propagator_ptr->aoints.basisSet().nBasis;

      const std::array<std::string,4> spinLabel = 
        { "SCALAR", "MZ", "MY", "MX" };

      // Save Electronic Matrices 
      for (auto i = 0; i < elec_propagator_ptr->fockMatrix.size(); i++) {
        
        savFile.safeWriteData("EF/EPDM_" + spinLabel[i],
          elec_propagator_ptr->onePDM[i], {NB,NB});
      }
      auto DSdims = savFile.getDims( "EF/EPDM_SCALAR" );

      // Save Protonic Matrices
      if (aux_rt) {
        for (auto i = 0; i < prot_propagator_ptr->fockMatrix.size(); i++) {
          
          savFile.safeWriteData("EF/PPDM_" + spinLabel[i],
            prot_propagator_ptr->onePDM[i], {auxNB,auxNB});
        }
      }

      // Save Nuclear Coordiantes
      savFile.safeWriteData("EF/NPOS",&this->current_x[0][0], {nAtoms, 3});

      // Save Nuclear Momentum
      savFile.safeWriteData("EF/NMOM",&this->current_p[0][0], {nAtoms, 3});

      // Save Time
      double time = intScheme.MaxTN+intScheme.dTN;
      savFile.safeWriteData("EF/TIME",&time,{1});

    }
  };

  /**
   *  \brief Read in the save state
   */
  template<template <typename, typename> class _SSTyp, typename IntsT>
  void Ehrenfest<_SSTyp,IntsT>::readSavedState() {

    // get pointers for determinant objects
    std::shared_ptr<_SSTyp<dcomplex, IntsT>> elec_propagator_ptr = std::dynamic_pointer_cast<_SSTyp<dcomplex, IntsT>>(rt->propagator_);
    std::shared_ptr<_SSTyp<dcomplex, IntsT>> prot_propagator_ptr = nullptr;
    if(aux_rt) 
      prot_propagator_ptr = std::dynamic_pointer_cast<_SSTyp<dcomplex, IntsT>>(aux_rt->propagator_);

    // get size of the electronic density matrix 
    size_t NB = elec_propagator_ptr->aoints.basisSet().nBasis;

    // get size of the protonic density matrix 
    size_t auxNB;
    if (aux_rt)
      auxNB = prot_propagator_ptr->aoints.basisSet().nBasis;

    // readin electronic density matrix
    auto DSdims = savFile.getDims( "EF/EPDM_SCALAR" );
    auto DZdims = savFile.getDims( "EF/EPDM_MZ" );
    //auto testdims = savFile.getDims( "SCF/1PDM_SCALAR" );
    //std::cout << "size of dim " << DSdims.size() << std::endl;

    // Errors in 1PDM Scalar
    if (DSdims[0] != NB or DSdims[1] != NB) {
      
      std::cout << "   * Incompatible EF/EPDM_SCALAR:";
      std::cout << "  Received (" << DSdims[0] << "," << DSdims[1] << ")"
        << " :";
      std::cout << "  Expected (" << NB << "," << NB << ")";
      CErr("Wrong dimension of 1PDM SCALAR!", std::cout);

    }

    // Read in electronic 1PDM SCALAR
    std::cout << "   * Found EF/EPDM_SCALAR !" << std::endl;
    savFile.readData("EF/EPDM_SCALAR",elec_propagator_ptr->onePDM[SCALAR]);
    //elec_propagator_ptr->print1PDM(std::cout);
    


    if (aux_rt) {
      auto auxDSdims = savFile.getDims( "EF/PPDM_SCALAR" );
      // Errors in 1PDM Scalar
      if (auxDSdims[0] != auxNB or auxDSdims[1] != auxNB) {
        
        std::cout << "   * Incompatible EF/PPDM_SCALAR:";
        std::cout << "  Received (" << auxDSdims[0] << "," << auxDSdims[1] << ")"
          << " :";
        std::cout << "  Expected (" << auxNB << "," << auxNB << ")";
        CErr("Wrong dimension of 1PDM SCALAR!", std::cout);

      }

      // Read in protonic 1PDM SCALAR
      std::cout << "   * Found EF/PPDM_SCALAR !" << std::endl;
      savFile.readData("EF/PPDM_SCALAR",prot_propagator_ptr->onePDM[SCALAR]);
    }

    // MZ
    if (not elec_propagator_ptr->iCS) {

      // Errors in 1PDM Mz
      if (DZdims[0] != NB or DZdims[1] != NB) {
        
        std::cout << "   * Incompatible EF/EPDM_MZ:";
        std::cout << "  Received (" << DZdims[0] << "," << DZdims[1] << ")"
          << " :";
        std::cout << "  Expected (" << NB << "," << NB << ")";
        CErr("Wrong dimension of 1PDM MZ!", std::cout);

      }

      // Read in electronic 1PDM SCALAR
      std::cout << "   * Found EF/EPDM_MZ !" << std::endl;
      savFile.readData("/EF/EPDM_MZ",elec_propagator_ptr->onePDM[MZ]);
    }

    if (aux_rt) {
      auto auxDZdims = savFile.getDims( "EF/PPDM_MZ" );
      // Errors in 1PDM Mz
      if (auxDZdims[0] != auxNB or auxDZdims[1] != auxNB) {
        
        std::cout << "   * Incompatible EF/PPDM_MZ:";
        std::cout << "  Received (" << auxDZdims[0] << "," << auxDZdims[1] << ")"
          << " :";
        std::cout << "  Expected (" << auxNB << "," << auxNB << ")";
        CErr("Wrong dimension of 1PDM MZ!", std::cout);

      }

      // Read in protonic 1PDM SCALAR
      std::cout << "   * Found EF/PPDM_MZ !" << std::endl;
      savFile.readData("/EF/PPDM_MZ",prot_propagator_ptr->onePDM[MZ]);
    }

    // read in nuclear coordinates
    savFile.readData("/EF/NPOS",&this->current_x[0][0]);

    // read in nuclear momentums
    savFile.readData("/EF/NMOM",&this->current_p[0][0]);

    // read in history time 
    savFile.readData("/EF/TIME",&this->hist_time);

    // update integrals
    this->updateInt(this->current_x,0.);

    // form Fock matrix
    std::dynamic_pointer_cast<RealTime<_SSTyp,IntsT>>(rt)->formFock(false, 0.);
    if (aux_rt)
      std::dynamic_pointer_cast<RealTime<_SSTyp,IntsT>>(aux_rt)->formFock(false,0.);

    // P (AO -> Ortho)
    elec_propagator_ptr->ao2orthoDen();
    if (aux_rt)
      prot_propagator_ptr->ao2orthoDen();

    // F (AO -> Ortho)
    elec_propagator_ptr->ao2orthoFock();
    if (aux_rt)
      prot_propagator_ptr->ao2orthoFock();
    
  };

}; // namespace ChronusQ

#endif
