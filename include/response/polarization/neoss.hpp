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

#include <response/polarization.hpp>
#include <singleslater/print.hpp>
#include <cqlinalg/blas1.hpp>
#include <cqlinalg/factorization.hpp>
#include <particleintegrals.hpp>
#include <util/threads.hpp>
#include <util/timer.hpp>

namespace ChronusQ {
  
  template <typename MatsT, typename IntsT>
  std::shared_ptr<TPIContractions<MatsT,IntsT>> retrieveTPI(std::string l1, std::string l2,NEOSS<MatsT, IntsT>& neoss,std::shared_ptr<SingleSlater<MatsT,IntsT>> ss1 ){
    std::shared_ptr<TPIContractions<MatsT, IntsT>> ssTPI;
    if (l1 == l2){
      ssTPI = ss1->TPI;
    }
    else{
      // Reversed because l1 corresponds to X and l2 corresponds to AX
      std::pair<bool,std::shared_ptr<TwoPInts<IntsT>>> ssTPItuple = neoss.getCrossTPIs(l2,l1);
      if (auto tpi_t = std::dynamic_pointer_cast<InCore4indexTPI<IntsT>>(ssTPItuple.second) ) {
        ssTPI = std::make_unique<InCore4indexTPIContraction<MatsT,IntsT>>(*tpi_t);
      }
      else if (auto tpi_t = std::dynamic_pointer_cast<DirectTPI<IntsT>>(ssTPItuple.second) ) {
        ssTPI = std::make_unique<GTODirectTPIContraction<MatsT, IntsT>>(*tpi_t);
      }
      ssTPI->contractSecond = ssTPItuple.first;
    }
    return ssTPI;
  };

  template <typename MatsT, typename IntsT>
  size_t PolarizationPropagator<NEOSS<MatsT,IntsT>>::getNSingleDim( const bool doTDA){

    size_t N = 0;

    NEOSS<MatsT, IntsT>& neoss = dynamic_cast<NEOSS<MatsT, IntsT>&>(*this->ref_);

    auto labels = neoss.getLabels();
    //Loops over the subsystems in our NEOSS object,
    for(auto label:labels){
      auto ssbase = neoss.getSubSSBase(label);
      SingleSlater<MatsT,IntsT>& ss = dynamic_cast<SingleSlater<MatsT,IntsT>&> (*ssbase); 
      N += getNSingleSSDim(ss,doTDA);
			std::cout << "Inside NEOSS getNSingleDim: " << N << std::endl;
    }
    return N;

  }

  template<typename MatsT, typename IntsT>
  size_t PolarizationPropagator<NEOSS<MatsT,IntsT>>::getNSingleSSDim(SingleSlater<MatsT,IntsT>& ss, const bool doTDA){

    size_t nOAVA = ss.nOA * ss.nVA;
    size_t nOBVB = ss.nOB * ss.nVB;
    size_t nOV   = ss.nO  * ss.nV;
    size_t N_n = (ss.nC == 1) ? nOAVA + nOBVB : nOV;
    
		if( not doTDA and not this->doReduced ) N_n *= 2;
	  assert(N_n != 0);
    return N_n; 

  }
 
  template <typename MatsT, typename IntsT>
  template <typename U>
  void PolarizationPropagator<NEOSS<MatsT, IntsT>>::formLinearTrans_direct_impl(
    MPI_Comm c, RC_coll<U> x, SINGLESLATER_POLAR_COPT op,
    bool noTrans){

    if( op != FULL ) CErr("Direct + non-FULL NYI");
      
    NEOSS<MatsT, IntsT>& neoss = dynamic_cast<NEOSS<MatsT, IntsT>&>(*this->ref_);
    size_t VBase = 0;
    auto labels = neoss.getLabels();
    bool zeroed = false;

    const size_t N = this->getNSingleDim(this->genSettings.doTDA) * (this->doReduced ? 2 : 1);
    const size_t tdOffSet = N / 2;
    const size_t chunk = 600;
		std::cout << "getNSingleDim: " << this->getNSingleDim(this->genSettings.doTDA) * (this->doReduced ? 2 : 1);
    //Loops over the subsystems in our NEOSS object,
    for(auto label1:labels){
    
      auto ssbase1 =  neoss.getSubSSBase(label1);
      SingleSlater<MatsT, IntsT>& ss1 = dynamic_cast<SingleSlater<MatsT, IntsT>&>((*ssbase1));
      size_t HVBase = 0;
      std::shared_ptr<SingleSlater<MatsT, IntsT>> ss1int = std::dynamic_pointer_cast<SingleSlater<MatsT, IntsT>>((ssbase1));
      
      for (auto label2:labels){
        auto ssbase2 =  neoss.getSubSSBase(label2);
        SingleSlater<MatsT, IntsT>& ss2 = dynamic_cast<SingleSlater<MatsT, IntsT>&>((*ssbase2));
       
        auto TPIcast = retrieveTPI(label1, label2,neoss, ss1int);
        ProgramTimer::tick("Direct Hessian Contract");
/*
        std::shared_ptr<InCore4indexTPI<double>> p = std::dynamic_pointer_cast<InCore4indexTPI<double>> (ss1.aoints.TPI);
        if( label1 == label2 ){       
          std::shared_ptr<InCore4indexTPI<double>> moTPI;
          size_t nMO = ss1.basisSet().nBasis;
          std::cout <<"SizeMO: "<< nMO << std::endl;
          p->output(std::cout,"pre-transform",true);
          //auto moTPIstore =p->template spatialToSpinBlock<double>();
          auto moTPIstore2 = p->transform('N',ss1.mo[0].pointer(),nMO,nMO);
          
          moTPIstore2.output(std::cout,"Direct MO Transform of TPI",true);
        }
*/        
        //Ensures correct typing of TPI for the contraction call in 131 
        std::shared_ptr<TPIContractions<U,IntsT>> TPI =
           TPIContractions<MatsT,IntsT>::template convert<U>(TPIcast);
  
        for(auto &X : x) {

          const size_t nVec = X.nVec;

          if( not zeroed ){
            if( X.AX ) std::fill_n(X.AX,N*nVec,0.);
          }

          for(size_t k = 0; k < nVec; k += chunk) {
            MPI_Barrier(c); // Sync MPI Processes at begining of each batch of 
                        // vectors

            const size_t nDo = std::min( chunk, nVec - k);

            auto *V_c  = X.X  + k*N;
            auto *HV_c = X.AX + k*N;

            bool scatter = not bool(X.X);
#ifdef CQ_ENABLE_MPI
            scatter = mxx::any_of(scatter,c);
#endif

            // Transform ph vector MO -> AO
            auto cList = 
              this->template phTransitionVecMO2AO<U>(c,scatter,nDo,N,ss1,ss2,
                label1==label2, V_c+VBase, V_c + tdOffSet+VBase);
            TPI->twoBodyContract(c,cList); // form G[V]
            // Only finish transformation on root process
            if( MPIRank(c) == 0 ) {
              
              // Transform ph vector AO -> MO
              this->phTransitionVecAO2MO(nDo,N,cList,ss2,label1==label2,HV_c + HVBase,HV_c + tdOffSet + HVBase);
              // Scale by diagonals
              if(label1 == label2){
                this->phEpsilonScale(true,false,nDo,N,ss1,V_c+VBase,HV_c+HVBase);
                this->phEpsilonScale(true,false,nDo,N,ss1,V_c+tdOffSet+VBase,
                  HV_c+tdOffSet+HVBase);
              }
            }
            
            // Free up transformation memory
            this->memManager_.free(cList[0].X);

         }    

        } // loop over groups of vectors
        zeroed = true;
        ProgramTimer::tock("Direct Hessian Contract");
        HVBase += getNSingleSSDim(ss2,this->genSettings.doTDA)/2;   
      }

      VBase += getNSingleSSDim(ss1,this->genSettings.doTDA)/2;

    }

    // Now that we've done all subsystems, add the metric
    for(auto& X: x)
    if( X.AX and this->incMet and not this->doAPB_AMB )
      SetMat('N', N/2, X.nVec, U(-1.), X.AX + (N/2), N, X.AX + (N/2), N);
  };

  
/*
  template<typename MatsT, typename IntsT>
  std::pair<size_t,MatsT*> PolarizationPropagator<NEOSS<MatsT, IntsT>>::formPropGrad( ResponseOperator op ){
    
    NEOSS<MatsT, IntsT>& neoss = dynamic_cast<NEOSS<MatsT, IntsT>&>(*this->ref_);
    auto labels = neoss.getLabels();
  
    for (auto label:labels){  
      auto ssbase =  neoss.getSubSSBase(label);
      SingleSlater<MatsT, IntsT>& ss = dynamic_cast<SingleSlater<MatsT, IntsT>&>((*ssbase));
      Integrals<IntsT>& aoi = ss.aoints;
    }

  };
*/
  template <>
  void PolarizationPropagator<NEOSS<double,double>>::formLinearTrans_direct(
    MPI_Comm c, RC_coll<double> x, SINGLESLATER_POLAR_COPT op,
    bool noTrans){
  
    formLinearTrans_direct_impl(c,x,op,noTrans);
    
  };

  template<>
  void PolarizationPropagator<NEOSS<dcomplex,double>>::formLinearTrans_direct(
    MPI_Comm c, RC_coll<double> x, SINGLESLATER_POLAR_COPT op,
    bool noTrans){
  
    CErr("How did I get in here?: complex,double NEOSS formLinearTrans_direct");

  };

  template<>
  void PolarizationPropagator<NEOSS<dcomplex,dcomplex>>::formLinearTrans_direct(
    MPI_Comm c, RC_coll<double> x, SINGLESLATER_POLAR_COPT op,
    bool noTrans){
  
    CErr("How did I get in here?: complex,complex NEOSS formLinearTrans_direct");

  };

  template<typename MatsT, typename IntsT>
  template<typename U>
  void PolarizationPropagator<NEOSS<MatsT,IntsT>>::neoPreConditioner(size_t nVec,
    U shift, U* V, U* AV) {

    auto& neoss = dynamic_cast<NEOSS<MatsT, IntsT>&>(*this->ref_);

    size_t NS = this->nSingleDim_;
    size_t hNS = NS / 2;

    std::function< U(MatsT) > diag = 
      [&]( MatsT d ) {
        return d - shift;
      };

    if( this->doReduced ) 
      diag = [&]( MatsT d ) {
        return d*d - shift*shift;
      };

    auto labels = neoss.getLabels();
    std::vector<std::shared_ptr<SingleSlater<MatsT,IntsT>>> subsystems;
    for(const auto& label: labels) {
      subsystems.push_back(
        std::dynamic_pointer_cast<SingleSlater<MatsT,IntsT>>(
          neoss.getSubSSBase(label)
        )
      );
    }

    for(auto iVec = 0ul; iVec < nVec; iVec++) {
      U* AVk = AV + iVec * NS;
      U* Vk  = V  + iVec * NS;
      size_t off = 0;
      for(const auto& ss: subsystems) {
        double* eps = ss->eps1;

        size_t nOAVA = ss->nOA * ss->nVA;
        size_t nOBVB = ss->nOB * ss->nVB;
        size_t nOV = ss->nO * ss->nV;

        size_t N  = (ss->nC == 1) ? nOAVA  : nOV;
        size_t NV = (ss->nC == 1) ? ss->nVA : ss->nV;
        size_t NO = (ss->nC == 1) ? ss->nOA : ss->nO;

        size_t NNext = (ss->nC == 1) ? nOAVA + nOBVB : nOV;

        const bool doBeta = ss->nC == 1;

        // TODO: Profile; are the if blocks in the hot loop significant?
        // Alpha / full
        for(auto k = 0; k < N; k++) {
          size_t i = k / NV;
          size_t a = (k % NV) + NO;
          U scale = diagonals.size() != 0 ? 
                      diag(diagonals[k+off]) :
                      diag(eps[a]-eps[i]);

          // X update
          AVk[k] = Vk[k] / scale;
          // Y update
          if( not this->doReduced )
            AVk[k + hNS] = Vk[k + hNS] / scale;
        }

        // Beta
        if( doBeta ) {

          eps = ss->iCS ? eps : ss->eps2;
          N  = nOBVB;
          NV = ss->nVB;
          NO = ss->nOB;

          for(auto k = 0; k < N; k++) {
            size_t i = k / NV;
            size_t a = (k % NV) + NO;
            U scale = diagonals.size() != 0 ? 
                        diag(diagonals[k+off+nOAVA]) :
                        diag(eps[a]-eps[i]);

            AVk[k + nOAVA] = Vk[k + nOAVA] / scale;
            if( not this->doReduced )
              AVk[k + hNS + nOAVA] = Vk[k + hNS + nOAVA] / scale;
          }
        }

        // Update subblocks
        AVk += NNext;
        Vk += NNext;
        off += NNext;

      } // Subsytem loop
    } // Vector loop
  }

  template <typename MatsT, typename IntsT>
  void PolarizationPropagator<NEOSS<MatsT, IntsT>>::resGuess(
    size_t nGuess, MatsT *G, size_t LDG) {

    NEOSS<MatsT, IntsT>& neoss = dynamic_cast<NEOSS<MatsT,IntsT>&>(*this->ref_);
    size_t NS = getNSingleDim(true);

    // Build a vector of the absolute difference between orbital energy 
    //   differences and the shift that keeps track of compound index
    size_t offset = 0;
    auto labels = neoss.getLabels();
    std::vector<std::pair<double, size_t>> shift_energies;
    shift_energies.reserve(NS);
    diagonals.reserve(NS);
    const double deMin = this->resSettings.deMin;

    for(const auto& label: labels) {

      auto ss = std::dynamic_pointer_cast<SingleSlater<MatsT,IntsT>>(
        neoss.getSubSSBase(label)
      );

      double* eps = ss->eps1;

      size_t nOAVA = ss->nOA * ss->nVA;
      size_t nOBVB = ss->nOB * ss->nVB;
      size_t nOV = ss->nO * ss->nV;

      size_t N  = (ss->nC == 1) ? nOAVA  : nOV;
      size_t NV = (ss->nC == 1) ? ss->nVA : ss->nV;
      size_t NO = (ss->nC == 1) ? ss->nOA : ss->nO;

      size_t NNext = (ss->nC == 1) ? nOAVA + nOBVB : nOV;

      size_t NB = ss->nAlphaOrbital();
      size_t XSize = NB*NB*2*ss->nC;
      size_t AXSize = NB*NB*(2*ss->nC+1);

      // Allocate scratch for exact diagonal contributions
      MatsT* XSCR = this->memManager_.template malloc<MatsT>(XSize);
      MatsT* AXSCR = this->memManager_.template malloc<MatsT>(AXSize);
      std::fill_n(XSCR, XSize, MatsT(0.));
      std::fill_n(AXSCR, AXSize, MatsT(0.));

      // Alpha/full
      for(auto k = 0; k < N; k++) {
        size_t i = k / NV;
        size_t a = (k % NV) + NO;
        size_t idx = k + offset;
        MatsT ediff = (eps[a] - eps[i]) + 
          this->getGDiag(i, a, false, *ss, XSCR, AXSCR);
        diagonals.push_back(ediff);
        ediff = std::abs(ediff - deMin);

        shift_energies.push_back({std::real(ediff), idx});
      }

      // Beta
      if( ss->nC == 1 ) {
        eps = ss->iCS ? eps : ss->eps2;
        N  = nOBVB;
        NV = ss->nVB;
        NO = ss->nOB;
        for(auto k = 0; k < N; k++) {
          size_t i = k / NV;
          size_t a = (k % NV) + NO;
          size_t idx = k + nOAVA + offset;
          MatsT ediff = (eps[a] - eps[i]) + 
            this->getGDiag(i, a, true, *ss, XSCR, AXSCR);
          diagonals.push_back(ediff);
          ediff = std::abs(ediff - deMin);
          shift_energies.push_back({std::real(ediff), idx});
        }
      }

      offset += NNext;
      this->memManager_.free(XSCR, AXSCR);
    } // Subsystem loop

    // Sort in ascending order
    std::sort(shift_energies.begin(), shift_energies.end());

    // Set guess
    std::fill_n(G, nGuess*LDG, 0.);
    for(size_t iG = 0; iG < nGuess; iG++)
      G[iG * LDG + shift_energies[iG].second] = 1.;
  }

	
	//Computing the total property gradient by subsystem
	//FIXME: allow for subsystems to be printed as well 
  template<typename MatsT, typename IntsT>
  std::pair<size_t,MatsT*> PolarizationPropagator<NEOSS<MatsT, IntsT>>::formPropGrad( ResponseOperator op ){
		size_t nVec = OperatorSize[op];
		size_t offSet = 0; 

    NEOSS<MatsT, IntsT>& neoss = dynamic_cast<NEOSS<MatsT, IntsT>&>(*this->ref_);
    auto labels = neoss.getLabels();
		MatsT* grad = neoss.memManager.template malloc<MatsT>(this->nSingleDim_*nVec);

    for (auto label:labels){  
      auto ssbase =  neoss.getSubSSBase(label);
      SingleSlater<MatsT, IntsT>& ss = dynamic_cast<SingleSlater<MatsT, IntsT>&>((*ssbase));
      Integrals<IntsT>& aoi = ss.aoints;
   		std::vector<IntsT*> opS;
    	std::vector<MatsT*> opT;
	    bool needTrans = true;
	    switch (op) {
	
	      case LenElectricDipole: 
	        opS = aoi.lenElectric->dipolePointers();
	        break;
	
	      case LenElectricQuadrupole: 
	        opS = aoi.lenElectric->quadrupolePointers();
	        break;
	
	      case LenElectricOctupole: 
	        opS = aoi.lenElectric->octupolePointers();
	        break;
	
	      case VelElectricDipole: 
	        opS = aoi.velElectric->dipolePointers();
	        break;
	
	      case VelElectricQuadrupole: 
	        opS = aoi.velElectric->quadrupolePointers();
	        break;
	
	      case VelElectricOctupole: 
	        opS = aoi.velElectric->octupolePointers();
	        break;
	
	      case MagneticDipole: 
	        opS = aoi.magnetic->dipolePointers();
	        break;
	
	      case MagneticQuadrupole: 
	        opS = aoi.magnetic->quadrupolePointers();
	        break;
	
	      case Brillouin:
	        needTrans = false;
	        opT = ss.fockMO.size() > 1 ?
	              std::vector<MatsT*>{ss.fockMO[0].pointer(), ss.fockMO[1].pointer()}
	              : std::vector<MatsT*>{ss.fockMO[0].pointer()};
	        break;
	
	    }
	
	    /*
	    if(needTrans and (this->ref_->nC == 2 or not this->ref_->iCS)) 
	      CErr("NO 2C or U");
	      */
	
	    size_t NB = ss.nAlphaOrbital();
	    size_t NBC = ss.nC * NB;
	
	    int nOAVA = ss.nOA * ss.nVA;
	    int nOBVB = ss.nOB * ss.nVB;
	    int nOV   = ss.nO  * ss.nV;
	
	    int N  = (ss.nC == 1) ? nOAVA  : nOV;
	    int NV = (ss.nC == 1) ? ss.nVA : ss.nV;
	    int NO = (ss.nC == 1) ? ss.nOA : ss.nO;
	    MatsT* SCR(nullptr);

	    if( needTrans ) {
	      SCR   = ss.memManager.template malloc<MatsT>(NBC*NBC);
	      opT.emplace_back(ss.memManager.template malloc<MatsT>(NBC*NBC));
	      if( ss.nC == 1 and not ss.iCS)
	        opT.emplace_back(ss.memManager.template malloc<MatsT>(NBC*NBC));
	    }
	

	    for(auto iVec = 0; iVec < nVec; iVec++) {
				std::cout << "offSet: " << offSet << std::endl;
	      MatsT* CMO = ss.mo[0].pointer();
	      MatsT* CMOB = (ss.nC == 1) ? ss.mo[1].pointer() : ss.mo[0].pointer();
	
	      MatsT* V = grad + iVec*this->nSingleDim_;
	
	      if( needTrans ) {
	
	        Gemm('N','N',NB,NBC,NB,MatsT(1.),opS[iVec],NB ,CMO,NBC,MatsT(0.),SCR   ,NB);
	        Gemm('C','N',NBC,NBC,NB,MatsT(1.),CMO     ,NBC,SCR,NB ,MatsT(0.),opT[0],NBC);
	
	        if( ss.nC == 1 and not ss.iCS ) {
	
	          Gemm('N','N',NB,NB,NB,MatsT(1.),opS[iVec],NB,CMOB,NB,MatsT(0.),SCR ,NB);
	          Gemm('C','N',NB,NB,NB,MatsT(1.),CMOB     ,NB,SCR,NB,MatsT(0.),opT[1],NB);
	
	        } else if( ss.nC == 2 ) {
	
	          Gemm('N','N',NB,NBC,NB,MatsT(1.),opS[iVec],NB ,CMOB,NBC,MatsT(0.),SCR  ,NB);
	          Gemm('C','N',NBC,NBC,NB,MatsT(1.),CMOB    ,NBC,SCR,NB ,MatsT(1.),opT[0],NBC);
	
	        }
	
	      }
	
	      MatsT* BOP = (ss.nC == 1 and not ss.iCS) ? opT[1] : opT[0];
	
	      SetMat('N',NV,NO,MatsT(1.),opT[0] + NO,NBC,V+offSet,NV);
				
	      if( ss.nC == 1 )
	        SetMat('N',ss.nVB,ss.nOB,MatsT(1.),BOP + ss.nOB,NB,V + nOAVA + offSet,ss.nVB);
	
	      //for(auto i = 0 , ai = 0; i < NO; i++)
	      //for(auto a = NO        ; a < NB; a++, ai++) {
	
	      //  V[ai]                               = opT[0][a + i*NB];
	      //  V[ai+nOAVA]                         = opT[0][a + i*NB];
	
	      //}

	      if(this -> doReduced ) 
	
	
	        if( isHerOp(op) )
	          for(auto i = 0 , ai = 0; i < NO; i++)
	          for(auto a = NO        ; a < NB; a++, ai++) {
	
	            V[ai]         = std::sqrt(0.5)*(V[ai]         + opT[0][i + a*NB]);
	            V[ai + nOAVA] = std::sqrt(0.5)*(V[ai + nOAVA] + opT[0][i + a*NB]);
	
	          }
	        else
	          for(auto i = 0 , ai = 0; i < NO; i++)
	          for(auto a = NO        ; a < NB; a++, ai++) {
	
	            V[ai]         = std::sqrt(0.5)*(V[ai]         - opT[0][i + a*NB]);
	            V[ai + nOAVA] = std::sqrt(0.5)*(V[ai + nOAVA] - opT[0][i + a*NB]);
	
	          }
	
	      else {
				
	        SetMat('N',NO,NV,MatsT(1.),opT[0] + NO*NBC,NBC,V + this->nSingleDim_/2+offSet,NO);
	        IMatCopy('T',NO,NV,MatsT(1.),V + this->nSingleDim_/2+offSet,NO,NV);
	        if( ss.nC == 1 ) {
	          SetMat('N',ss.nOB,ss.nVB,MatsT(1.),BOP + ss.nOB*NB,NB,
	            V + nOAVA + this->nSingleDim_/2+offSet,ss.nOB);
	          IMatCopy('T',ss.nOB,ss.nVB,MatsT(1.),
	            V + this->nSingleDim_/2 + nOAVA+offSet,ss.nOB,ss.nVB);
	        	}
					}
	        //for(auto i = 0 , ai = 0; i < NO; i++)
	        //for(auto a = NO        ; a < NB; a++, ai++) {
	
	        //  V[ai + this->nSingleDim_/2]         = opT[0][i + a*NB];
	        //  V[ai + nOAVA + this->nSingleDim_/2] = opT[0][i + a*NB];
	
	        //}
	
	      }
	
	
	    
	
	    if(SCR) ss.memManager.free(SCR);
	    if( needTrans )
	      for(auto &X : opT ) ss.memManager.free(X);
	
			//FIXME: Implementation needed for APB/AMB	
	    // Transform to proper form form
	    /*
	    if( doAPB_AMB and not doReduced ) 
	      blockTransform(this->nSingleDim_/2,nVec,std::sqrt(0.5),
	        grad,this->nSingleDim_,grad+this->nSingleDim_/2,this->nSingleDim_);
			*/
//			std::cout << "ss.iCS: " << ss.iCS << std::endl;
//			std::cout << "ss.nC: " << ss.nC << std::endl;
//	    offSet += (ss.nC == 1 and not ss.iCS) ? nOV : nOAVA + nOBVB;
				offSet += N;
		}
  	

#if 0
    if( this->savFile.exists() ) {

      if( op == LenElectricDipole )
        this->savFile.safeWriteData( "/RESP/PROPGRAD/LEN_ELEC_DIPOLE",grad,
          { nVec, this->nSingleDim_ } );

      if( op == LenElectricQuadrupole )
        this->savFile.safeWriteData( "/RESP/PROPGRAD/LEN_ELEC_QUADRUPOLE",grad,
          { nVec, this->nSingleDim_ } );

      if( op == LenElectricOctupole )
        this->savFile.safeWriteData( "/RESP/PROPGRAD/LEN_ELEC_OCTUPOLE",grad,
          { nVec, this->nSingleDim_ } );

      if( op == VelElectricDipole )
        this->savFile.safeWriteData( "/RESP/PROPGRAD/VEL_ELEC_DIPOLE",grad,
          { nVec, this->nSingleDim_ } );

      if( op == VelElectricQuadrupole )
        this->savFile.safeWriteData( "/RESP/PROPGRAD/VEL_ELEC_QUADRUPOLE",grad,
          { nVec, this->nSingleDim_ } );

      if( op == VelElectricOctupole )
        this->savFile.safeWriteData( "/RESP/PROPGRAD/VEL_ELEC_OCTUPOLE",grad,
          { nVec, this->nSingleDim_ } );

      if( op == MagneticDipole )
        this->savFile.safeWriteData( "/RESP/PROPGRAD/MAG_DIPOLE",grad,
          { nVec, this->nSingleDim_ } );

      if( op == MagneticQuadrupole )
        this->savFile.safeWriteData( "/RESP/PROPGRAD/MAG_QUADRUPOLE",grad,
          { nVec, this->nSingleDim_ } );
    }
#endif
		prettyPrintSmart(std::cout,"grad",grad,this->nSingleDim_,nVec,nVec);	
    return {nVec, grad};
	}

}  // namespace ChronusQ
