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

#include <singleslater.hpp>
#include <singleslater/neo_singleslater.hpp>
#include <util/matout.hpp>

namespace ChronusQ {

  /**
   *  \brief Save the current state for the NEO-SCF calculation
   *
   *  Allocate memory for extrapolation and compute the energy
   */
  template <typename MatsT, typename IntsT>
  void NEOSingleSlater<MatsT,IntsT>::saveCurrentState() {

    // main system
    this->SingleSlater<MatsT,IntsT>::saveCurrentState();

    // auxiliary system
    this->aux_neoss->SingleSlater<MatsT,IntsT>::saveCurrentState();

  }; // NEOSingleSlater<MatsT>::saveCurrentState

  /*
  *   Brief: Function for the ModifyOrbitals object to get the vector of
  *          Fock matrices in the alpha/beta basis
  */
  template <typename MatsT, typename IntsT>
  std::vector<std::shared_ptr<SquareMatrix<MatsT>>> NEOSingleSlater<MatsT,IntsT>::getFock() {

    std::vector<std::shared_ptr<SquareMatrix<MatsT>>> fock;

    // Handle this set of fock Matrices
    if( this->nC == 1 and this->iCS) {
      fock.emplace_back(std::make_shared<SquareMatrix<MatsT>>(MatsT(0.5)*this->fockMatrix->S()));
    } else if( this->nC == 1 ) {
      fock.emplace_back(std::make_shared<SquareMatrix<MatsT>>(MatsT(0.5)*this->fockMatrix->S()+MatsT(0.5)*this->fockMatrix->Z()));
      fock.emplace_back(std::make_shared<SquareMatrix<MatsT>>(MatsT(0.5)*this->fockMatrix->S()-MatsT(0.5)*this->fockMatrix->Z()));
    } else {
      fock.emplace_back(std::make_shared<SquareMatrix<MatsT>>(this->fockMatrix->template spinGather<MatsT>()));
    }

    // Handle Auxiliary system
    if( this->aux_neoss->nC == 1 and this->aux_neoss->iCS) {
      fock.emplace_back(std::make_shared<SquareMatrix<MatsT>>(MatsT(0.5)*this->aux_neoss->fockMatrix->S()));
    } else if( this->aux_neoss->nC == 1 ) {
      fock.emplace_back(std::make_shared<SquareMatrix<MatsT>>(MatsT(0.5)*this->aux_neoss->fockMatrix->S()+MatsT(0.5)*this->aux_neoss->fockMatrix->Z()));
      fock.emplace_back(std::make_shared<SquareMatrix<MatsT>>(MatsT(0.5)*this->aux_neoss->fockMatrix->S()-MatsT(0.5)*this->aux_neoss->fockMatrix->Z()));
    } else {
      fock.emplace_back(std::make_shared<SquareMatrix<MatsT>>(this->aux_neoss->fockMatrix->template spinGather<MatsT>()));
    }
    return fock;
  };

  /*
  *   Brief: Function for the ModifyOrbitals object to get the vector of
  *          onePDM matrices in the alpha/beta basis
  */
  template <typename MatsT, typename IntsT>
  std::vector<std::shared_ptr<SquareMatrix<MatsT>>> NEOSingleSlater<MatsT,IntsT>::getOnePDM() {

    std::vector<std::shared_ptr<SquareMatrix<MatsT>>> den;

    // Handle this set of fock Matrices
    if( this->nC == 1 and this->iCS) {
      den.emplace_back(std::make_shared<SquareMatrix<MatsT>>(MatsT(0.5)*this->onePDM->S()));
    } else if( this->nC == 1 ) {
      den.emplace_back(std::make_shared<SquareMatrix<MatsT>>(MatsT(0.5)*this->onePDM->S()+MatsT(0.5)*this->onePDM->Z()));
      den.emplace_back(std::make_shared<SquareMatrix<MatsT>>(MatsT(0.5)*this->onePDM->S()-MatsT(0.5)*this->onePDM->Z()));
    } else {
      den.emplace_back(std::make_shared<SquareMatrix<MatsT>>(this->onePDM->template spinGather<MatsT>()));
    }

    // Handle Auxiliary system
    if( this->aux_neoss->nC == 1 and this->aux_neoss->iCS) {
      den.emplace_back(std::make_shared<SquareMatrix<MatsT>>(MatsT(0.5)*this->aux_neoss->onePDM->S()));
    } else if( this->aux_neoss->nC == 1 ) {
      den.emplace_back(std::make_shared<SquareMatrix<MatsT>>(MatsT(0.5)*this->aux_neoss->onePDM->S()+MatsT(0.5)*this->aux_neoss->onePDM->Z()));
      den.emplace_back(std::make_shared<SquareMatrix<MatsT>>(MatsT(0.5)*this->aux_neoss->onePDM->S()-MatsT(0.5)*this->aux_neoss->onePDM->Z()));
    } else {
      den.emplace_back(std::make_shared<SquareMatrix<MatsT>>(this->aux_neoss->onePDM->template spinGather<MatsT>()));
    }
    return den;
  };

  /*
  *     Brief: Returns the orthogonalization objects for both this and the auxiliary
  *            to be used for SCF
  *
  */
  template<typename MatsT, typename IntsT>
  std::vector<std::shared_ptr<Orthogonalization<MatsT>>> NEOSingleSlater<MatsT,IntsT>::getOrtho(){
    std::vector<std::shared_ptr<Orthogonalization<MatsT>>> ortho;
    
    // Handle this orthogonalization
    if( this->nC == 1 and this->iCS) {
      ortho.push_back(this->orthoAB);
    } else if( this->nC == 1) {
      ortho.push_back(this->orthoAB);
      ortho.push_back(this->orthoAB);
    } else {
      ortho.push_back(this->orthoAB);
    }

    // Handle aux
    if( this->aux_neoss->nC == 1 and this->aux_neoss->iCS) {
      ortho.push_back(this->aux_neoss->orthoAB);
    } else if( this->aux_neoss->nC == 1) {
      ortho.push_back(this->aux_neoss->orthoAB);
      ortho.push_back(this->aux_neoss->orthoAB);
    } else {
      ortho.push_back(this->aux_neoss->orthoAB);
    }
    return ortho;
  }

  template<typename MatsT, typename IntsT>
  void NEOSingleSlater<MatsT, IntsT>::runModifyOrbitals(EMPerturbation& pert) {

    bool iRO = false;

    std::vector<std::reference_wrapper<SquareMatrix<MatsT>>> moRefs;
    for( auto& m : this->mo )
      moRefs.emplace_back(m);
    for( auto& m : this->aux_neoss->mo)
      moRefs.emplace_back(m);

    std::vector<double*> epsVec;

    // This set of eigenvalues
    if( this->mo.size() == 2 ) {
      epsVec = {this->eps1,this->eps2};
    } else {
      epsVec = {this->eps1};
    }
    // Aux set of eigenvalues
    if( this->aux_neoss->mo.size() == 2 ){
      epsVec.emplace_back(this->aux_neoss->eps1);
      epsVec.emplace_back(this->aux_neoss->eps2);
    } else {
      epsVec.emplace_back(this->aux_neoss->eps1);
    }

    // Run modify orbitals
    this->modifyOrbitals->runModifyOrbitals(pert, moRefs, epsVec);

    saveCurrentState();
    this->ao2orthoFock();   // SCF Does not update the fockMatrixOrtho
    this->MOFOCK();
    this->aux_neoss->ao2orthoFock();
    this->aux_neoss->MOFOCK();
  };   // NEOSingleSlater<MatsT,IntsT> :: runModifyOrbitals

  template<typename MatsT, typename IntsT>
  std::vector<NRRotOptions> NEOSingleSlater<MatsT,IntsT>:: buildRotOpt(){
    
    std::vector<NRRotOptions> rotOpt;

    // Generate rotation information for this object
    size_t NB = this->nC*this->basisSet().nBasis;
    int cnt = 0;
    if( this->nC == 1 and this->iCS ){
      rotOpt.emplace_back();
      rotOpt[cnt].spaceIndex = 0;
      rotOpt[cnt].rotIndices = ssNRRotIndices(this->nOA, NB);
      ++cnt;
    } else if( this->nC == 1 ) {
      rotOpt.emplace_back();
      rotOpt[cnt].spaceIndex = 0;
      rotOpt[cnt].rotIndices = ssNRRotIndices(this->nOA, NB);
      ++cnt;
      rotOpt.emplace_back();
      rotOpt[cnt].spaceIndex = 1;
      rotOpt[cnt].rotIndices = ssNRRotIndices(this->nOB, NB);
      ++cnt;
    } else if( this->nC == 2 ){
      rotOpt.emplace_back();
      rotOpt[cnt].spaceIndex = 0;
      rotOpt[cnt].rotIndices = ssNRRotIndices(this->nO, NB);
      cnt++;
    } else if( this->nC == 4 ){
      size_t N = NB/2;
      rotOpt.emplace_back();
      rotOpt[cnt].spaceIndex = 0;
      rotOpt[cnt].rotIndices = ssNRRotIndices(this->nO, N, N); // only rotate the positive energy orbitals
      ++cnt;
    }

    // Generate Rotation information for the auxilliary object
    NB = aux_neoss->nC*aux_neoss->basisSet().nBasis;
    if( this->nC == 1 and this->iCS ){
      rotOpt.emplace_back();
      rotOpt[cnt].spaceIndex = 0;
      rotOpt[cnt].rotIndices = ssNRRotIndices(aux_neoss->nOA, NB);
      ++cnt;
    } else if( this->nC == 1 ) {
      rotOpt.emplace_back();
      rotOpt[cnt].spaceIndex = 0;
      rotOpt[cnt].rotIndices = ssNRRotIndices(aux_neoss->nOA, NB);
      ++cnt;
      rotOpt.emplace_back();
      rotOpt[cnt].spaceIndex = 1;
      rotOpt[cnt].rotIndices = ssNRRotIndices(aux_neoss->nOB, NB);
      ++cnt;
    } else if( this->nC == 2 ){
      rotOpt.emplace_back();
      rotOpt[cnt].spaceIndex = 0;
      rotOpt[cnt].rotIndices = ssNRRotIndices(aux_neoss->nO, NB);
      cnt++;
    } else if( this->nC == 4 ){
      size_t N = NB/2;
      rotOpt.emplace_back();
      rotOpt[cnt].spaceIndex = 0;
      rotOpt[cnt].rotIndices = ssNRRotIndices(aux_neoss->nO, N, N); // only rotate the positive energy orbitals
      ++cnt;
    }
    return rotOpt;
  }

  template<typename MatsT, typename IntsT>
  void NEOSingleSlater<MatsT,IntsT>::formDensity(bool computeAuxDen){

    SingleSlater<MatsT,IntsT>::formDensity();
    if( computeAuxDen )
      this->aux_neoss->formDensity(false);

  }

  template<typename MatsT, typename IntsT>
  void NEOSingleSlater<MatsT,IntsT>::printProperties(){

    this->SingleSlater<MatsT,IntsT>::printProperties();
    //aux_neoss->SingleSlater<MatsT,IntsT>>::printProperties();

  }

}; // namespace ChronusQ
