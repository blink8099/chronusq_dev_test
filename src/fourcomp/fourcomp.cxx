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
#include <corehbuilder/fourcomp.hpp>
#include <corehbuilder/nonrel.hpp>
#include <physcon.hpp>
#include <cqlinalg.hpp>
#include <cqlinalg/svd.hpp>
#include <util/matout.hpp>
#include <electronintegrals/oneeints/relativisticints.hpp>

namespace ChronusQ {

  /**
   *  \brief Compute the 4C Relativistic Core Hamiltonian
   *
   *  Hamiltonian format:
   *         [ V               T ]
   *         [ T     1/(4c^2)W-T ]
   */
  template <typename MatsT, typename IntsT>
  void FourComponent<MatsT,IntsT>::compute4CCH(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> coreH) {

    size_t NB = basisSet_.nBasis;

    // Form 1/(4c^2)*W-T
    SquareMatrix<dcomplex> W_spinBlock(1./(4. * SpeedOfLight * SpeedOfLight)
        * std::dynamic_pointer_cast<OneERelInts<IntsT>>(
              this->aoints_.potential)->template formW<dcomplex>()
        - this->aoints_.kinetic->matrix()
              .template spatialToSpinBlock<dcomplex>());

    // Spin Scatter 
    PauliSpinorSquareMatrices<dcomplex> W(W_spinBlock.spinScatter<dcomplex>());

    // Set the V block in the first diagonal of CH
    // V = [ V   0 ]
    //     [ 0   V ] 
    //
    //      [ V   |    ]    [ V1    |      ]
    //      [   V |    ] => [   V2  |      ]
    //      [     |    ]    [       |      ]
    //      [     |    ]    [       |      ]
    
    // Spin Scatter
    PauliSpinorSquareMatrices<dcomplex> V2C(
        PauliSpinorSquareMatrices<dcomplex>::spinBlockScatterBuild(
            this->aoints_.potential->matrix(),
            this->aoints_.potential->matrix()));


    // Set the T blocks in the off-diagonal of CH
    //  [     | T   ]    [          |  CP11    ]
    //  [     |   T ] => [          |     CP12 ]
    //  [ T   |     ]    [ CP21     |          ]
    //  [   T |     ]    [     CP22 |          ]
    
    // Spin Scatter
    PauliSpinorSquareMatrices<dcomplex> T2C(
        PauliSpinorSquareMatrices<dcomplex>::spinBlockScatterBuild(
            this->aoints_.kinetic->matrix(),
            this->aoints_.kinetic->matrix()));

    // Build 4C coreH
    // Scalar
    SetMat('N',NB,NB,dcomplex(1.),V2C.S().pointer(),NB,coreH->S().pointer(),2*NB);
    SetMat('N',NB,NB,dcomplex(1.),T2C.S().pointer(),NB,coreH->S().pointer()+NB,2*NB);
    SetMat('N',NB,NB,dcomplex(1.),T2C.S().pointer(),NB,coreH->S().pointer()+2*NB*NB,2*NB);
    SetMat('N',NB,NB,dcomplex(1.),W.S().pointer(),NB,coreH->S().pointer()+2*NB*NB+NB,2*NB);

    // MZ
    SetMat('N',NB,NB,dcomplex(1.),V2C.Z().pointer(),NB,coreH->Z().pointer(),2*NB);
    SetMat('N',NB,NB,dcomplex(1.),T2C.Z().pointer(),NB,coreH->Z().pointer()+NB,2*NB);
    SetMat('N',NB,NB,dcomplex(1.),T2C.Z().pointer(),NB,coreH->Z().pointer()+2*NB*NB,2*NB);
    SetMat('N',NB,NB,dcomplex(1.),W.Z().pointer(),NB,coreH->Z().pointer()+2*NB*NB+NB,2*NB);

    // MY
    SetMat('N',NB,NB,dcomplex(1.),W.Y().pointer(),NB,coreH->Y().pointer()+2*NB*NB+NB,2*NB);

    // MX
    SetMat('N',NB,NB,dcomplex(1.),W.X().pointer(),NB,coreH->X().pointer()+2*NB*NB+NB,2*NB);

  };  // void FourComponent::compute4CCH(std::vector<MatsT*> &CH)


  template void FourComponent<dcomplex,double>::compute4CCH(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<dcomplex>>);

  template<> void FourComponent<dcomplex,dcomplex>::compute4CCH(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<dcomplex>>) {
    CErr("4C + Complex Ints NYI",std::cout);
  }

  template<> void FourComponent<double,double>::compute4CCH(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<double>>) {
    CErr("Scalar Only Real 4C NYI",std::cout);
  }


  /**
   *  \brief Compute the 4C Core Hamiltonian
   */
  template <typename MatsT, typename IntsT>
  void FourComponent<MatsT,IntsT>::computeCoreH(EMPerturbation& emPert,
      std::shared_ptr<PauliSpinorSquareMatrices<MatsT>> coreH) {

    compute4CCH(emPert, coreH);

  };  // void FourComponent::computeCoreH(std::vector<MatsT*> &CH)

  template void FourComponent<dcomplex,double>::computeCoreH(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<dcomplex>>);

  template<> void FourComponent<dcomplex,dcomplex>::computeCoreH(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<dcomplex>>) {
    CErr("4C + Complex Ints NYI",std::cout);
  }

  template<> void FourComponent<double,double>::computeCoreH(EMPerturbation&,
      std::shared_ptr<PauliSpinorSquareMatrices<double>>) {
    CErr("Scalar Only Real 4C NYI",std::cout);
  }


}; // namespace ChronusQ



