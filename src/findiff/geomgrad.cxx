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

#include <algorithm>

#include <cxxapi/options.hpp>
#include <findiff/geomgrad.hpp>
#include <grid/integrator.hpp>
#include <singleslater.hpp>
#include <electronintegrals/twoeints/incore4indexeri.hpp>

namespace ChronusQ {

  template <typename MatsT, typename IntsT>
  void NumGradient::moveMol(size_t iAtm, size_t iXYZ, double diff) {

    auto ref_t = std::dynamic_pointer_cast<SingleSlater<MatsT,IntsT>>(ref_);
    curr_ = nullptr;

    mol = std::move(Molecule(ref_t->molecule()));
    mol.atoms[iAtm].coord[iXYZ] += diff;
    mol.update();

    basis = CQBasisSetOptions(std::cout,input_,mol,"BASIS");
    dfbasis = CQBasisSetOptions(std::cout,input_,mol,"DFBASIS");

    aoints = CQIntsOptions(std::cout,input_,ref_t->memManager,basis,dfbasis);
    curr_ = CQSingleSlaterOptions(std::cout,input_,ref_t->memManager,mol,*basis,aoints);

    CQSCFOptions(std::cout,input_,*curr_,emPert);

    curr_->scfControls.guess = READMO;

    curr_->savFile = ref_t->savFile;

  };

  template void NumGradient::moveMol<double,double>(size_t, size_t, double);
  template void NumGradient::moveMol<dcomplex,double>(size_t, size_t, double);
  template void NumGradient::moveMol<dcomplex,dcomplex>(size_t, size_t, double);


  // Gets energy contributions at each geometry
  std::vector<double> NumGradient::getEnergies() {

    curr_->formCoreH(emPert);
 
    // If INCORE, compute and store the ERIs
    if(auto p = std::dynamic_pointer_cast<Integrals<double>>(aoints))
      p->ERI->computeAOInts(*basis, mol, emPert, ELECTRON_REPULSION,
                            {basis->basisType, false, false, false});
    else if(auto p = std::dynamic_pointer_cast<Integrals<dcomplex>>(aoints))
      p->ERI->computeAOInts(*basis, mol, emPert, ELECTRON_REPULSION,
                              {basis->basisType, false, false, false});
 
    curr_->formGuess();
    curr_->SCF(emPert);
 
    return curr_->getEnergySummary();

  };


  // Gets DFT grid weights at each geometry
  //   *Currently disabled*
  std::vector<double> NumGradient::getGridVals(bool doGrad, size_t iA,
    size_t iXYZ) {

    size_t nrad = 4 * 1;
    size_t nang = 14;

    SetLAThreads(1);

    if ( GetNumThreads() != 1 )
      CErr("Cannot run numerical grid gradient in parallel!");

    std::vector<double> resweights(nrad*nang, 0.);

    /*
    auto weightf = [&](size_t &res, std::vector<cart_t> &batch,
      std::vector<double> &weights, size_t NBE, double* BasisEval,
      std::vector<size_t> &batchEvalShells,
      std::vector<std::pair<size_t,size_t>> &subMatCut, size_t iAtm) {

      if (resweights.size() != weights.size()) {
        std::cout << "ResWeight size: " << resweights.size() << '\n';
        std::cout << "Weights size: " << weights.size() << std::endl;
        CErr("Can't do this with unmatching dims");
      }

      std::copy(weights.begin(), weights.end(), resweights.begin());

    };

    BeckeIntegrator<EulerMac> 
      integrator(MPI_COMM_NULL,ref_->memManager,mol,basis,
      EulerMac(nrad), nang, nrad, NOGRAD, 1e-14);

    if (doGrad)
      integrator.integrateGradient<size_t>(iA,iXYZ,weightf);
    else
      integrator.integrate<size_t>(weightf);
    */

    return resweights;

  };


  // Gets one electron integrals at each geometry
  template <typename IntsT>
  std::vector<IntsT> NumGradient::getOneEInts() {

    curr_->formCoreH(emPert);

    auto NB = basis->nBasis;
    auto NSq = NB*NB;

    std::vector<IntsT> results;

    auto copier = [&](IntsT* pointer) {
      std::vector<IntsT> temp;
      std::copy(pointer, pointer + NSq, back_inserter(results));
    };

    auto int_cast = std::dynamic_pointer_cast<Integrals<IntsT>>(aoints);

    copier(int_cast->overlap->pointer());
    copier(int_cast->kinetic->pointer());
    copier(int_cast->potential->pointer());

    std::cout << "result size: " << results.size() << std::endl;

    return results;

  };

  // Gets eris at each geometry
  template <typename IntsT>
  std::vector<IntsT> NumGradient::getERI() {

    curr_->formCoreH(emPert);

    if(auto p = std::dynamic_pointer_cast<Integrals<IntsT>>(aoints))
      p->ERI->computeAOInts(*basis, mol, emPert, ELECTRON_REPULSION,
                            {basis->basisType, false, false, false});

    auto NB = basis->nBasis;
    auto NSq = NB*NB;
    auto NB4 = NSq*NSq;

    std::vector<IntsT> results;

    auto copier = [&](IntsT* pointer) {
      std::copy(pointer, pointer + NB4, back_inserter(results));
    };

    auto int_cast = std::dynamic_pointer_cast<Integrals<IntsT>>(aoints);
    auto eri = std::dynamic_pointer_cast<InCore4indexERI<IntsT>>(int_cast->ERI);
    if ( eri == nullptr )
      CErr("Only incore eris in finite difference eri gradients");

    copier(eri->pointer());

    return results;

  };

  template std::vector<double> NumGradient::getERI();
  template std::vector<dcomplex> NumGradient::getERI();

}

