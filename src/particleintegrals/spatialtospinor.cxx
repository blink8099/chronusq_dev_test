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

#include <particleintegrals/twopints/incore4indextpi.hpp>
#include <particleintegrals/twopints/incoreritpi.hpp>
#include <particleintegrals/onepints/relativisticints.hpp>
#include <matrix.hpp>
#include <cqlinalg.hpp>

namespace ChronusQ {

  template <typename IntsT>
  template <typename IntsU>
  InCore4indexTPI<IntsU> InCore4indexTPI<IntsT>::spatialToSpinBlock() const {
    size_t NB = this->nBasis();
    InCore4indexTPI<IntsU> spinor(this->memManager(), 2*NB);
    spinor.clear();

    for (auto spls = 0ul; spls < 2; spls++)
    for (auto spmn = 0ul; spmn < 2; spmn++)
    for (auto sg = 0ul; sg < NB; sg++)
    for (auto lm = 0ul; lm < NB; lm++)
    for (auto nu = 0ul; nu < NB; nu++)
    for (auto mu = 0ul; mu < NB; mu++) {
      spinor(spmn*NB+mu, spmn*NB+nu, spls*NB+lm, spls*NB+sg) = (*this)(mu, nu, lm, sg);
    }

    return spinor;
  }

  template <typename IntsT>
  template <typename IntsU>
  InCoreRITPI<IntsU> InCoreRITPI<IntsT>::spatialToSpinBlock() const {
    size_t NB = this->nBasis();
    InCoreRITPI<IntsU> spinor(this->memManager(), 2*NB, NBRI);
/*
    for ( auto sp = 0ul; sp < 2; sp++)
    for ( auto nu = 0ul; nu < NB; nu++)
    for ( auto mu = 0ul; mu < NB; mu++)
    for ( auto L = 0ul; L < nRIBasis(); L++) {
      spinor(L, sp*NB + mu, sp*NB + nu) = (*this)(L, mu, nu);
    }
*/
    SetMatDiag(NB*NBRI, NB, pointer(), NB*NBRI, spinor.pointer(), 2*NB*NBRI);
    return spinor;
  }

  template <typename IntsT>
  template <typename MatsT>
  SquareMatrix<MatsT> OnePRelInts<IntsT>::formW() const {
    if (hasSpinOrbit()) {
      size_t NB = this->nBasis();
      SquareMatrix<MatsT> W(this->ParticleIntegrals::memManager(), 2*NB);
      // W = [ W1  W2 ]
      //     [ W3  W4 ]
      dcomplex *W1 = W.pointer();
      dcomplex *W2 = W1 + 2*NB*NB;
      dcomplex *W3 = W1 + NB;
      dcomplex *W4 = W2 + NB;
      // W1 = pV.p + i (pVxp)(Z)
      MatAdd('N','N',NB,NB,dcomplex(1.),scalar().pointer(),NB,
             dcomplex(0.,1.),SOZ().pointer(),NB,W1,2*NB);
      // W4 = pV.p - i (pVxp)(Z)
      MatAdd('N','N',NB,NB,dcomplex(1.),scalar().pointer(),NB,
             dcomplex(0.,-1.),SOZ().pointer(),NB,W4,2*NB);
      // W2 = (pVxp)(Y) + i (pVxp)(X)
      MatAdd('N','N',NB,NB,dcomplex(1.),SOY().pointer(),NB,
             dcomplex(0.,1.),SOX().pointer(),NB,W2,2*NB);
      // W3 = -(pVxp)(Y) + i (pVxp)(X)
      MatAdd('N','N',NB,NB,dcomplex(-1.),SOY().pointer(),NB,
             dcomplex(0.,1.),SOX().pointer(),NB,W3,2*NB);
      return W;
    } else
      return scalar().matrix().template spatialToSpinBlock<MatsT>();
  }

  template <>
  template <>
  SquareMatrix<double> OnePRelInts<double>::formW() const {
    if (hasSpinOrbit()) {
      SquareMatrix<double> dummy(this->ParticleIntegrals::memManager(),1);
      CErr("W matrix with spin-orbit cannot be real.");
      return dummy;
    } else {
      return scalar().matrix().template spatialToSpinBlock<double>();
    }
  }
 
  template InCore4indexTPI<double> InCore4indexTPI<double>::spatialToSpinBlock() const;
  template InCore4indexTPI<dcomplex> InCore4indexTPI<double>::spatialToSpinBlock() const;
  template InCore4indexTPI<dcomplex> InCore4indexTPI<dcomplex>::spatialToSpinBlock() const;

  template InCoreRITPI<double> InCoreRITPI<double>::spatialToSpinBlock() const;
  template InCoreRITPI<dcomplex> InCoreRITPI<double>::spatialToSpinBlock() const;
  template InCoreRITPI<dcomplex> InCoreRITPI<dcomplex>::spatialToSpinBlock() const;

  template SquareMatrix<double> OnePRelInts<double>::formW() const;
  template SquareMatrix<dcomplex> OnePRelInts<double>::formW() const;
  template SquareMatrix<dcomplex> OnePRelInts<dcomplex>::formW() const;

}; // namespace ChronusQ
