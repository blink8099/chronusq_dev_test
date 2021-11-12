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
			std::pair<bool,std::shared_ptr<TwoPInts<IntsT>>> ssTPItuple = neoss.getCrossTPIs(l1,l2);
			//ssTPI->output(std::cout,"crossTPI",true);
			std::cout << "ssTPI bool: " << ssTPItuple.first << std::endl;
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

		//Loops over the subsystems in our NEOSS object,
		for(auto label1:labels){
  	
			auto ssbase1 =  neoss.getSubSSBase(label1);
			SingleSlater<MatsT, IntsT>& ss1 = dynamic_cast<SingleSlater<MatsT, IntsT>&>((*ssbase1));
			size_t HVBase = 0;
			std::shared_ptr<SingleSlater<MatsT, IntsT>> ss1int = std::dynamic_pointer_cast<SingleSlater<MatsT, IntsT>>((ssbase1));
			
			for (auto label2:labels){
				auto ssbase2 =  neoss.getSubSSBase(label2);
				std::cout << "HVBase: " << HVBase << std::endl; 	
				std::cout << "VBase: " << VBase << std::endl; 	
				SingleSlater<MatsT, IntsT>& ss2 = dynamic_cast<SingleSlater<MatsT, IntsT>&>((*ssbase2));
 			 
    		const size_t N  = this->getNSingleDim(this->genSettings.doTDA) * (this->doReduced ? 2 : 1);  
    		const size_t tdOffSet = N / 2;
   		 	const size_t chunk    = 600;
				std::cout << "N Value: " << N << std::endl;	
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

					if( not zeroed){
      			if( X.AX ) std::fill_n(X.AX,N*nVec,0.);
					}
      		for(size_t k = 0; k < nVec; k += chunk) {
						std::cout << "k: " <<  k << std::endl;
        		MPI_Barrier(c); // Sync MPI Processes at begining of each batch of 
                        // vectors

        		const size_t nDo = std::min( chunk, nVec - k);

        		auto *V_c  = X.X  + k*N;
       		  auto *HV_c = X.AX + k*N;

						std::cout << "nVec: " << nVec << std::endl;
						std::cout << "nDo: " << nDo << std::endl;
						std::cout << "tdOffSet: " << tdOffSet << std::endl;
        bool scatter = not bool(X.X);
#ifdef CQ_ENABLE_MPI
        scatter = mxx::any_of(scatter,c);
#endif

						std::cout << label1 << std::endl;
						std::cout << label2 << std::endl;

        		// Transform ph vector MO -> AO
        		auto cList = 
          		this->template phTransitionVecMO2AO<U>(c,scatter,nDo,N,ss1, label1==label2, V_c+VBase,
            		  V_c + tdOffSet+VBase);
        		TPI->twoBodyContract(c,cList); // form G[V]
        		// Only finish transformation on root process
        		if( MPIRank(c) == 0 ) {
							
          		// Transform ph vector AO -> MO
          		std::cout << "HV_c + HVBase" << HV_c + HVBase << std::endl;
          		this->phTransitionVecAO2MO(nDo,N,cList,ss2,label1==label2,HV_c + HVBase,HV_c + tdOffSet + HVBase);
          		// Scale by diagonals
          		if(label1 == label2){
          			this->phEpsilonScale(true,false,nDo,N,ss1,V_c+VBase,HV_c+HVBase);
          			this->phEpsilonScale(true,false,nDo,N,ss1,V_c+tdOffSet+VBase,
            			HV_c+tdOffSet+HVBase);
							}
							std::cout << "Finished Epsilon Scaling" << std::endl;
        		}
						
        		// Free up transformation memory
        		this->memManager_.free(cList[0].X);

     		 }		

				//TODO: Test incMet to see if this works with NEOSS
      		if( X.AX and this->incMet and not this->doAPB_AMB )
        		SetMat('N', N/2, nVec, U(-1.), X.AX + (N/2), N, X.AX + (N/2), N);

    		} // loop over groups of vectors
				zeroed = true;
      	ProgramTimer::tock("Direct Hessian Contract");
				HVBase += getNSingleSSDim(ss2,this->genSettings.doTDA)/2;		
			}

			VBase += getNSingleSSDim(ss1,this->genSettings.doTDA)/2;

		}
	};

/*
  template <typename MatsT, typename IntsT>
  template <typename U>
  void printResMO_impl(
    std::ostream &out, size_t nRoots, double *W_print,
    std::vector<std::pair<std::string,double *>> data, U* VL, U* VR) {

		NEOSS<MatsT, IntsT>& neoss = dynamic_cast<NEOSS<MatsT, IntsT>&>(*this->ref_);
		auto labels = neoss.getLabels();
		//Loops over the subsystems in our NEOSS object,
		for(auto label1:labels){
			auto ssbase1 =  neoss.getSubSSBase(label1);
			SingleSlater<MatsT, IntsT>& ss = dynamic_cast<SingleSlater<MatsT, IntsT>&>((*ssbase1));
	
	    this->nSingleDim_ = getNSingleSSDim(ss,this->genSettings.doTDA);
	
	    out << "\n\n\n* RESIDUE EIGENMODES\n\n\n";
	
	    for(auto iRt = 0; iRt < nRoots; iRt++) {
	
	      out << "  Root " << std::setw(7) << std::right << iRt+1 << ":";
	
	      // Energy eigenvalues in various unit systems
	      out << std::setw(15) << std::right << "W(Eh) = " 
	          << std::setprecision(8) << std::fixed << W_print[iRt];
	
	      out << std::setw(15) << std::right << "W(eV) = " 
	          << std::setprecision(8) << std::fixed 
	          << W_print[iRt]*EVPerHartree;
	
	      out << "\n";
	
	
	      if( this->genSettings.evalProp ) {
	        for(auto &d : data) {
	        out << "       " << std::setw(7) << " " << " ";
	        out << std::setw(15) << std::right << d.first 
	            << std::setprecision(8) << std::fixed 
	            << d.second[iRt];
	
	        out << "\n";
	        }
	      }
	      auto xCont = getMOContributions(VR+iRt*this->nSingleDim_,1e-1);
	      decltype(xCont) yCont;
	      if( not this->genSettings.doTDA ) 
	        yCont = getMOContributions(VL+iRt*this->nSingleDim_,1e-1);
	
	      // MO contributions
	      out << "    MO Contributions:\n";
	      for(auto &c : xCont) {
	
	        char spinLabel = (c.first.first > 0) ? 
	                           ((this->ref_->nC == 1) ? 'A' : ' ') : 'B';
	      
	        out << "      ";
	        out << std::setw(4) << std::right 
	            << std::abs(c.first.second) + 1 << spinLabel << " -> ";
	        out << std::setw(4) << std::right 
	            << std::abs(c.first.first) + 1<< spinLabel;
	
	        if(std::is_same<U,double>::value)
	          out << "  " << std::fixed << std::setprecision(5) 
	                      << std::setw(10) << std::right << c.second << "\n";
	        else {
	          out << "  " << std::fixed << std::setprecision(5) 
	                      << std::setw(10) << std::right << std::abs(c.second);
	          out << "  " << std::fixed << std::setprecision(5) 
	                      << std::setw(10) << std::right << std::arg(c.second) 
	                      << "\n";
	        }
	
	
	
	      }
	      for(auto &c : yCont) {
	
	        char spinLabel = (c.first.first > 0) ? 
	                           ((this->ref_->nC == 1) ? 'A' : ' ') : 'B';
	      
	        out << "      ";
	        out << std::setw(4) << std::right 
	            << std::abs(c.first.second) + 1 << spinLabel << " <- ";
	        out << std::setw(4) << std::right 
	            << std::abs(c.first.first) + 1 << spinLabel;
	
	        if(std::is_same<U,double>::value)
	          out << "  " << std::fixed << std::setprecision(5) 
	                      << std::setw(10) << std::right << c.second << "\n";
	        else {
	          out << "  " << std::fixed << std::setprecision(5) 
	                      << std::setw(10) << std::right << std::abs(c.second);
	          out << "  " << std::fixed << std::setprecision(5) 
	                      << std::setw(10) << std::right << std::arg(c.second) 
	                      << "\n";
	        }
	
	
	
	      }
	
	
	      out << "\n\n";
	    }

 		}
	};
*/
  
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

}  // namespace ChronusQ
