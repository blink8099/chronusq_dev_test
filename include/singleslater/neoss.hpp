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
#include <matrix.hpp>


namespace ChronusQ {

  // Pure virtual class for only interface functions
  struct NEOBase {
    virtual std::shared_ptr<SingleSlaterBase> getSubSSBase(std::string label) = 0;
  };

  template <typename MatsT, typename IntsT>
  class NEOSS: virtual public SingleSlater<MatsT,IntsT>, public NEOBase {

    template <typename MatsU, typename IntsU>
    friend class NEOSS;

    template <typename F>
    void applyToEach(F func) {
      if(order_.size() == subsystems.size()) {
        for(auto& label: order_)
          func(subsystems.at(label));
      }
      else {
        for(auto& system: subsystems)
          func(system.second);
      }
    }

    protected:

      using SubSSPtr = std::shared_ptr<SingleSlater<MatsT,IntsT>>;

      std::unordered_map<std::string,SubSSPtr> subsystems;

      // Optional order
      std::vector<std::string> order_;

      // Subsystem coulomb interactions
      // First map is the "external" particle and second map is the integrated
      //   particle.
      // EXAMPLE: interCoulomb["electron"]["proton"] is the coulomb matrix for
      //   the electronic subsystem (in the electronic basis) coming from the
      //   protonic coulomb potential.
      std::unordered_map<std::string, std::unordered_map<std::string, SquareMatrix<MatsT>>> interCoulomb;

    public:

      // Main constructor
      // XXX: Does NOT construct electronic or protonic wavefunctions
      template <typename... Args>
      NEOSS(MPI_Comm c, CQMemManager &mem, Molecule &mol, BasisSet &basis,
                  Integrals<IntsT> &aoi, Args... args) :
        SingleSlater<MatsT,IntsT>(c,mem,mol,basis,aoi,args...),
        WaveFunctionBase(c,mem,args...),
        QuantumBase(c,mem,args...) { };

      // Copy/move constructors
      template <typename MatsU>
        NEOSS(const NEOSS<MatsU,IntsT>&, int dummy = 0);
      template <typename MatsU>
        NEOSS(NEOSS<MatsU,IntsT>&&, int dummy = 0);
      NEOSS(const NEOSS<MatsT,IntsT>&);
      NEOSS(NEOSS<MatsT,IntsT>&&);


      void addSubsystem(std::string label, std::shared_ptr<SingleSlater<MatsT,IntsT>> ss) {

        auto NB = ss->basisSet().nBasis;

        std::unordered_map<std::string, SquareMatrix<MatsT>> newCoulombs;
        for( auto& x: subsystems ) {
          newCoulombs.insert({x.first, SquareMatrix<MatsT>(ss->memManager, NB)});
        }

        subsystems[label] = ss;
      }

      void setOrder(std::vector<std::string> labels) {
        order_ = labels;
      }

      // Getters
      template <template <typename, typename> class T>
      std::shared_ptr<T<MatsT,IntsT>> getSubsystem(std::string label) {
        return std::dynamic_pointer_cast<T<MatsT,IntsT>>(subsystems.at(label));
      }
      std::shared_ptr<SingleSlaterBase> getSubSSBase(std::string label) {
        return std::dynamic_pointer_cast<SingleSlaterBase>(subsystems.at(label));
      }

      // Pass-through to each functions
      void SCFInit() {
        applyToEach([](SubSSPtr& ss){ ss->SCFInit(); });
      }

      void SCFFin() {
        applyToEach([](SubSSPtr& ss){ ss->SCFFin(); });
      }

      void saveCurrentState() {
        applyToEach([](SubSSPtr& ss){ ss->saveCurrentState(); });
      }

      void formGuess() {
        applyToEach([](SubSSPtr& ss){ ss->formGuess(); });
      }

      void formCoreH(EMPerturbation& emPert) {
        applyToEach([&](SubSSPtr& ss){ ss->formCoreH(emPert); });
      }

      // Properties
      void computeEnergy() {

        applyToEach([](SubSSPtr& ss){ ss->computeEnergy(); });

        this->totalEnergy = 0.;
        applyToEach([&](SubSSPtr& ss){ 
            std::cout << "Sub energy: " << ss->totalEnergy << std::endl;
            this->totalEnergy += ss->totalEnergy;
        });

      }

      void computeMultipole(EMPerturbation&) { }
      void computeSpin() { }
      void methodSpecificProperties() { }

      // Overrides specific to a NEO-SCF
      void printSCFProg(std::ostream& out = std::cout, bool printDiff = true) {
        out << "  Macro SCFIt: " << std::setw(6) << std::left;

        if( printDiff ) out << this->scfConv.nSCFMacroIter + 1;
        else            out << 0;

        // Current Total Energy
        out << std::setw(18) << std::fixed << std::setprecision(10)
                             << std::left << this->totalEnergy;

        if( printDiff ) {
          out << std::scientific << std::setprecision(7);
          out << std::setw(14) << std::right << this->scfConv.deltaEnergy;
          out << "   ";
        }
  
        out << std::endl;
      }

      bool evalConver(EMPerturbation& pert) {

        bool isConverged;

        // Compute all SCF convergence information on root process
        if( MPIRank(this->comm) == 0 ) {
          
          // Save copy of old energy
          double oldEnergy = this->totalEnergy;

          // Compute new energy
          this->computeProperties(pert);

          // Compute the difference between current and old energy
          this->scfConv.deltaEnergy = this->totalEnergy - oldEnergy;

          bool energyConv = std::abs(this->scfConv.deltaEnergy) < 
                            this->scfControls.eneConvTol;

          isConverged = energyConv;
        }

#ifdef CQ_ENABLE_MPI
        // Broadcast whether or not we're converged to ensure that all
        // MPI processes exit the NEO-SCF simultaneously
        if( MPISize(this->comm) > 1 ) MPIBCast(isConverged,0,this->comm);
#endif
        
        return isConverged;
      }

      void SCF(EMPerturbation& pert);

      // Disable NR/stability for now
      MatsT* getNRCoeffs() {
        CErr("NR NYI for NEO!");
        return nullptr;
      }

      std::pair<double,MatsT*> getStab() {
        CErr("NR NYI for NEO!");
        return {0., nullptr};
      }
  };

}

#include <singleslater/neoss/scf.hpp>

