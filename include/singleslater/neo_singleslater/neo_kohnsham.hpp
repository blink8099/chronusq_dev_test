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
#include <singleslater.hpp>
#include <basisset/basisset_util.hpp>
#include <cqlinalg/blasext.hpp>
#include <util/timer.hpp>
#include <dft.hpp>
#include <singleslater/neo_singleslater.hpp>
#include <singleslater/kohnsham.hpp>

namespace ChronusQ {

  /**
   *  \brief The NEO-Kohn-Sham class.
   *
   *  Specializes the NEOSingleSlater class for a Kohn-Sham description of 
   *  the electron-proton wave function
   */
  template <typename MatsT, typename IntsT>
  class NEOKohnSham : public NEOSingleSlater<MatsT,IntsT>, public KohnSham<MatsT,IntsT> {

  template <typename MatsU, typename IntsU>
  friend class NEOKohnSham;

  protected:
    
    // pointer that points to the auxiliary class object
    std::shared_ptr<NEOKohnSham<MatsT,IntsT>> aux_neoks;

  public:
    
    std::vector<std::shared_ptr<DFTFunctional>> epc_functionals; ///< EPC kernels

    // EPC energy
    double EPCEnergy = 0.;

    template <typename... Args>
    NEOKohnSham(std::string epcfuncname, 
      std::vector<std::shared_ptr<DFTFunctional>> epc_funclist,
      std::string funcName, std::vector<std::shared_ptr<DFTFunctional>> funclist,
      MPI_Comm c, IntegrationParam ip, CQMemManager &mem, Molecule &mol, 
      BasisSet &basis, Integrals<IntsT> &aoi, Args... args) :
      WaveFunctionBase(c,mem,mol,basis,args...),
      QuantumBase(c,mem,args...),
      NEOSingleSlater<MatsT,IntsT>(c,mem,mol,basis,aoi,args...),
      KohnSham<MatsT,IntsT>(funcName,funclist,c,ip,mem,mol,basis,aoi,args...),
      SingleSlater<MatsT,IntsT>(c,mem,mol,basis,aoi,args...),
      epc_functionals(std::move(epc_funclist)) {};  // NEOKohnSham constructor


    template <typename... Args>
    NEOKohnSham(std::string epcfuncname, 
      std::vector<std::shared_ptr<DFTFunctional>> epc_funclist,
      std::string funcName, std::vector<std::shared_ptr<DFTFunctional>> funclist,
      std::string rL, std::string rS, MPI_Comm c, IntegrationParam ip, 
      CQMemManager &mem, Molecule &mol, 
      BasisSet &basis, Integrals<IntsT> &aoi, Args... args) :
      WaveFunctionBase(c,mem,mol,basis,args...),
      QuantumBase(c,mem,args...),
      NEOSingleSlater<MatsT,IntsT>(c,mem,mol,basis,aoi,args...),
      KohnSham<MatsT,IntsT>(rL,rS,funcName,funclist,c,ip,mem,mol,basis,aoi,args...),
      SingleSlater<MatsT,IntsT>(c,mem,mol,basis,aoi,args...),
      epc_functionals(std::move(epc_funclist)) {};  // NEOKohnSham constructor

    // Copy and Move ctors

    template <typename MatsU>
      NEOKohnSham(const NEOKohnSham<MatsU,IntsT> &other, int dummy = 0);
    template <typename MatsU>
       NEOKohnSham(NEOKohnSham<MatsU,IntsT> &&other, int dummy = 0);     
    NEOKohnSham(const NEOKohnSham<MatsT,IntsT> &other);
    NEOKohnSham(NEOKohnSham<MatsT,IntsT> &&other);
    
    /**
     *  \brief NEO-Kohn-Sham specification of formFock
     *  
     *  Compute EPC contribution to VXC and increment the fock matrix
     */
    virtual void formFock(EMPerturbation &pert, bool increment = false, double HFX = 0.) {

      NEOSingleSlater<MatsT,IntsT>::formFock(pert,increment,this->functionals.back()->xHFX);

      // form VXC matrix
      formVXC();

      ROOT_ONLY(this->comm);

      // Add VXC in Fock matrix
      //this->fockMatrix->output(std::cout, "F not VXC", true);
      *this->fockMatrix += *this->VXC;
      //this->VXC->output(std::cout, "VXC", true);
      //std::cout << "EXC is " << this->XCEnergy << std::endl;

    }; // formFock

    virtual void computeTotalProperties(EMPerturbation& emPert) {

      NEOSingleSlater<MatsT, IntsT>::computeTotalProperties(emPert);
      
      // Add EXC in the total energy
      this->totalEnergy += this->XCEnergy;
      //std::cout << "EXC is " << this->XCEnergy << std::endl;

      // Add EXC in the macro total energy
      this->totalMacroEnergy += this->XCEnergy;
      
    }; 

    void getAux(std::shared_ptr<NEOKohnSham<MatsT,IntsT>> neo_ks) 
    { 
      aux_neoks = neo_ks; 
      NEOSingleSlater<MatsT,IntsT>::getAux(neo_ks);
    };

    // VXC
    void formVXC();

    void mkAuxVar(bool isGGA, 
      double epsScreen, size_t NPts_Batch, 
      double *n, double *mx, double *my, double *mz,
      double *dndX, double *dndY, double *dndZ, 
      double *dmxdX, double *dmxdY, double *dmxdZ, 
      double *dmydX, double *dmydY, double *dmydZ, 
      double *dmzdX, double *dmzdY, double *dmzdZ, 
      double *Mnorm, double *Kx, double *Ky, double *Kz, 
      double *Hx, double *Hy, double *Hz,
      double *DSDMnorm, double *signMD,  
      bool* Msmall, double *nColl, double *gammaColl );

    void loadVXCderWithEPC(size_t NPts, double *Den, double *Gamma,
      double *aux_Den, double *aux_Gamma, double *cGamma, 
      double *epsEval, double *VRhoEval, double *VgammaEval,
      double *CVgammaEval, double *EpsSCR, double *VRhoSCR, 
      double *VgammaSCR, double *CVgammaSCR);

    void constructEPCZVars(DENSITY_TYPE denTyp, size_t NPts, 
      double *CVsigmaEval, double *ZsigmaVar3);

    void mkCrossAuxVar(bool check_aux, 
      double epsScreen, size_t NPts,
      double *dndX, double *dndY, double *dndZ, 
      double *dMxdX, double *dMxdY, double *dMxdZ, 
      double *dMydX, double *dMydY, double *dMydZ, 
      double *dMzdX, double *dMzdY, double *dMzdZ, 
      double *aux_dndX, double  *aux_dndY, double *aux_dndZ, 
      double *aux_dMxdX, double *aux_dMxdY, double *aux_dMxdZ, 
      double *aux_dMydX, double *aux_dMydY, double *aux_dMydZ, 
      double *aux_dMzdX, double *aux_dMzdY, double *aux_dMzdZ, 
      double *gammaColl);

    void formZ_vxc_epc(DENSITY_TYPE denTyp, 
      bool isGGA, size_t NPts, size_t NBE, size_t IOff, 
      double epsScreen, std::vector<double> &weights, double *ZrhoVar1,
      double *ZgammaVar1, double *ZgammaVar2, double *ZgammaVar3,
      double *DenS, double *DenZ, double *DenY, double *DenX, 
      double *GDenS, double *GDenZ, double *GDenY, double *GDenX, 
      double *aux_DenS,  double *aux_DenZ,  double *aux_DenY,  double *aux_DenX, 
      double *aux_GDenS, double *aux_GDenZ, double *aux_GDenY, double *aux_GDenX, 
      double *BasisScratch, double *ZMAT);

    // SCF Functions
    void buildModifyOrbitals();

  }; // NEOKohnSham class


}; // namespace ChronusQ
