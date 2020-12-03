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
#include <util/time.hpp>
#include <dft.hpp>

// KS_DEBUG_LEVEL == 1 - Timing
#ifndef KS_DEBUG_LEVEL
#  define KS_DEBUG_LEVEL 0
#endif

namespace ChronusQ {


  /**
   *  \brief A datastructure to hold the information
   *  pertaining to the control of the Kohn--Sham
   *  numerical integration.
   */ 
  struct IntegrationParam {
    double epsilon      = 1e-12; ///< Screening parameter
    size_t nAng         = 302;   ///< # Angular points
    size_t nRad         = 100;   ///< # Radial points
    size_t nRadPerBatch = 4;     ///< # Radial points / macro batch
  };


  /**
   *  \breif The Kohn--Sham class.
   *
   *  Specializes the SingleSlater class for a Kohn--Sham description of the
   *  many-body wave function
   */ 
  template <typename MatsT, typename IntsT>
  class KohnSham : public SingleSlater<MatsT,IntsT>,
    public std::enable_shared_from_this<KohnSham<MatsT,IntsT>> {


  protected:

    // Useful typedefs
    typedef MatsT*                   oper_t;
    typedef std::vector<oper_t>       oper_t_coll;
    typedef std::vector<oper_t_coll>  oper_t_coll2;

  public:

    std::shared_ptr<KohnSham<MatsT,IntsT>> getPtr(){ return this->shared_from_this(); }


    std::vector<std::shared_ptr<DFTFunctional>> functionals; ///< XC kernels
    IntegrationParam intParam; ///< Numerical integration controls

    bool isGGA_; ///< Whether or not the XC kernel is within the GGA
    double XCEnergy; ///< Exchange-correlation energy

    std::shared_ptr<PauliSpinorSquareMatrices<double>> VXC; ///< VXC terms

    // Current Timings
    double VXCDur;

    // Inherit ctors from SingleSlater<T>

    template <typename... Args>
    KohnSham(std::string funcName,
      std::vector<std::shared_ptr<DFTFunctional>> funclist,
      MPI_Comm c, IntegrationParam ip,
      CQMemManager &mem, Molecule &mol, BasisSet &basis,
      Integrals<IntsT> &aoi, Args... args) : 
      SingleSlater<MatsT,IntsT>(c,mem,mol,basis,aoi,args...),
      WaveFunctionBase(c,mem,args...),
      QuantumBase(c,mem,args...), isGGA_(false),
      functionals(std::move(funclist)),intParam(ip){ 

      // Append HF tags to reference names
      if(this->nC == 1) {
        if(this->iCS) {
          this->refLongName_  += "Restricted " + funcName;
          this->refShortName_ += "R" + funcName;
        } else {
          this->refLongName_  += "Unrestricted " + funcName;
          this->refShortName_ += "U" + funcName;
        }
      } else {
        this->refLongName_  += "Generalized " + funcName;
        this->refShortName_ += "G" + funcName;
      }

      size_t NB = this->basisSet().nBasis;
      if(this->nC > 1)
        VXC = std::make_shared<PauliSpinorSquareMatrices<double>>(this->memManager, NB, true);
      else if (not this->iCS)
        VXC = std::make_shared<PauliSpinorSquareMatrices<double>>(this->memManager, NB, false);
      else
        VXC = std::make_shared<PauliSpinorSquareMatrices<double>>(this->memManager, NB, false, false);


    }; // KohnSham constructor


    template <typename... Args>
    KohnSham(std::string rL, std::string rS, std::string funcName,
      std::vector<std::shared_ptr<DFTFunctional>> funclist,
      MPI_Comm c, IntegrationParam ip, 
      CQMemManager &mem, Molecule &mol, BasisSet &basis,
      Integrals<IntsT> &aoi, Args... args) : 
      SingleSlater<MatsT,IntsT>(c,mem,mol,basis,aoi,args...),
      WaveFunctionBase(c,mem,args...),
      QuantumBase(c,mem,args...), isGGA_(false),
      functionals(std::move(funclist)),intParam(ip) { 

      this->refLongName_  += rL + " " + funcName;
      this->refShortName_ += rS + funcName;

      size_t NB = this->basisSet().nBasis;
      if(this->nC > 1)
        VXC = std::make_shared<PauliSpinorSquareMatrices<double>>(this->memManager, NB, true);
      else if (not this->iCS)
        VXC = std::make_shared<PauliSpinorSquareMatrices<double>>(this->memManager, NB, false);
      else
        VXC = std::make_shared<PauliSpinorSquareMatrices<double>>(this->memManager, NB, false, false);


    }; // KohnSham constructor


    // Copy and Move ctors
      
    template <typename MatsU> 
      KohnSham(const KohnSham<MatsU,IntsT> &other, int dummy = 0); 
    template <typename MatsU> 
      KohnSham(KohnSham<MatsU,IntsT> &&other, int dummy = 0);
    KohnSham(const KohnSham<MatsT,IntsT> &other);
    KohnSham(KohnSham<MatsT,IntsT> &&other);


    /**
     *  \brief Kohn-Sham specialization of formFock
     *
     *  Compute VXC and increment the fock matrix
     */  
    virtual void formFock(EMPerturbation &pert, bool increment = false, double HFX = 0.) {

      SingleSlater<MatsT,IntsT>::formFock(pert,increment,functionals.back()->xHFX);

      auto VXCStart = tick();
      formVXC();
      VXCDur = tock(VXCStart);

      ROOT_ONLY(this->comm);

      // Add VXC in Fock matrix
      *this->fockMatrix += *VXC;

    }; // formFock

    /**
     *  \brief Kohn-Sham specialization of computeEnergy
     *
     *  Compute EXC and add it to the HF energy 
     */  
    virtual void computeEnergy() {

      SingleSlater<MatsT,IntsT>::computeEnergy();
      // Add EXC in the total energy
      this->totalEnergy += XCEnergy;
        
    }; // computeEnergy



    virtual void printFockTimings(std::ostream &out) {
  
      out << "    Fock Timings:\n";
      out << "      Wall time G[D] = " << std::setw(8)
          << std::setprecision(5)  << std::scientific
          << this->GDDur << " s\n";
      out << "      Wall time VXC  = " << std::setw(8)
          << std::setprecision(5)  << std::scientific
          << VXCDur << " s\n\n";
  
  
    }; // SingleSlater<T>::printFockTimings





    // KS specific functions
    // See include/singleslater/kohnsham/vxc.hpp for docs.

    // VXC
    void formVXC(); 

    void evalDen(SHELL_EVAL_TYPE typ, size_t NPts,size_t NBE, size_t NB, 
      std::vector<std::pair<size_t,size_t>> &subMatCut, double *SCR1,
      double *SCR2, double *DENMAT, double *Den, double *GDenX, double *GDenY, double *GDenZ,
      double *BasisScr);

    void formZ_vxc(DENSITY_TYPE denTyp, bool isGGA, size_t NPts, size_t NBE, size_t IOff, 
      double epsScreen, std::vector<double> &weights, double *ZrhoVar1,
      double *ZsigmaVar1, double *ZsigmaVar2, 
      double *DenS, double *DenZ, double *DenY, double *DenX, 
      double *GDenS, double *GDenZ, double *GDenY, double *GDenX, 
      double *Kx, double *Ky, double *Kz, 
      double *Hx, double *Hy, double *Hz,
      double *BasisScratch, double *ZMAT);

    double energy_vxc(size_t NPts, std::vector<double> &weights, double *EpsEval, double *Den);

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

    void loadVXCder(size_t NPts, double *Den, double *sigma, double *EpsEval, double*VRhoEval, 
      double *VsigmaEval, double *EpsSCR, double *VRhoSCR, double *VsigmaSCR); 

    void constructZVars(DENSITY_TYPE denTyp, bool isGGA, size_t NPts, 
      double *VrhoEval, double *VsigmaEval, double *ZrhoVar1, 
      double *ZsigmaVar1, double *ZsigmaVar2);

    // FXC Terms
      
    template <typename U>
    void evalTransDen(SHELL_EVAL_TYPE typ, size_t NPts,size_t NBE, size_t NB, 
      std::vector<std::pair<size_t,size_t>> &subMatCut, U *SCR1,
      U *SCR2, U *DENMAT, U *Den, U *GDenX, U *GDenY, U *GDenZ,
      U *BasisScr);

    void loadFXCder(size_t NPts, double *Den, double *sigma, double *EpsEval, double *VRhoEval, 
      double *V2RhoEval, double *VsigmaEval, double *V2sigmaEval, double *V2RhosigmaEval, 
      double *EpsSCR, double *VRhoSCR, double *VsigmaSCR, double *V2RhoEvalSCR, double *V2sigmaEvalSCR,
      double *V2RhosigmaEvalSCR); 

    template <typename U>
    void constructZVarsFXC(DENSITY_TYPE denTyp, bool isGGA, size_t NPts, 
      double* GDenS, double* GDenZ, double* GDenY, double* GDenX,
      U* TS, U* TZ, U* TY, U* TX,
      U* GTS, U* GTZ, U* GTY, U* GTX,
      double *VrhoEval, double *VsigmaEval, 
      double *V2rhoEval, double *V2sigmaEval, double *V2RhosigmaEval, 
      U *ZrhoVar1, U *ZgammaVar1, U *ZgammaVar2, U *ZgammaVar3, U *ZgammaVar4);

    template <typename U>
    void constructZVarsFXC(DENSITY_TYPE denTyp, bool isGGA, size_t NPts, 
      double* GDenS, double* GDenZ, double* GDenY, double* GDenX,
      bool * Msmall, double *Mnorm, 
      double *Kx, double *Ky, double *Kz, 
      double *Hx, double *Hy, double *Hz,
      U* TS, U* TZ, U* TY, U* TX,
      U* GTS, U* GTZ, U* GTY, U* GTX,
      U* gPTss, U* gPTsz, U* gPTsy, U* gPTsx, U* gPTzz, 
      U* gPTyy, U* gPTxx,  
      double *VrhoEval, double *VsigmaEval, 
      double *V2rhoEval, double *V2sigmaEval, double *V2RhosigmaEval, 
      U *ZrhoVar1, U *ZgammaVar1, U *ZgammaVar2, U *ZgammaVar3, U *ZgammaVar4);

    template <typename U>
    void formZ_fxc(DENSITY_TYPE denType, bool isGGA, size_t NPts, size_t NBE, size_t IOff,
      double epsScreen, std::vector<double> &weights,
      U *ZrhoVar1, U *ZgammaVar1, U *ZgammaVar2, U *ZgammaVar3, U *ZgammaVar4,
      double* GDenS, double* GDenZ, double* GDenY, double* GDenX, U* GTS, U* GTZ, U* GTY, U* GTX,
      double *BasisScr, U* ZMAT);

    // GTO-based TDDFT
    template <typename U>
    void formZ_fxc(DENSITY_TYPE denType, bool isGGA, size_t NPts, size_t NBE, size_t IOff,
      double epsScreen, std::vector<double> &weights,
      U *ZrhoVar1, U *ZgammaVar1, U *ZgammaVar2, U *ZgammaVar3, U *ZgammaVar4,
      bool * Msmall, double *Mnorm, 
      double* DSDMnorm, double* signMD, 
      double* GDenS, double* GDenZ, double* GDenY, double* GDenX, 
      double *Kx, double *Ky, double *Kz, 
      double *Hx, double *Hy, double *Hz,
      U* GTS, U* GTZ, U* GTY, U* GTX,
      U* gPTss, U* gPTsz, U* gPTsy, U* gPTsx, U* gPTzz, 
      U* gPTyy, U* gPTxx,  
      double *BasisScr, U* ZMAT);

    // Calculate gPTss,sx,sy,sz 
    template <typename U>
    void mkgPTVar( 
      size_t NPts, 
      double* GDenS, double* GDenZ, double* GDenY, double* GDenX, 
      U* GTS, U* GTZ, U* GTY, U* GTX,
      U* gPTss, U* gPTsz, U* gPTsy, U* gPTsx, U* gPTzz, 
      U* gPTyy, U* gPTxx  
      );


    template <typename U>
    void formFXC(MPI_Comm c,  std::vector<TwoBodyContraction<U>> &cList );





    MatsT* getNRCoeffs();
    std::pair<double,MatsT*> getStab();

  }; // class KohnSham


}; // namespace ChronusQ

