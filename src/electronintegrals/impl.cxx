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

#include <integrals.hpp>
#include <electronintegrals/contract/incore.hpp>
#include <electronintegrals/contract/direct.hpp>
#include <electronintegrals/contract/direct3index.hpp>
#include <electronintegrals/oneeints/relativisticints.hpp>


namespace ChronusQ {

  template <>
  void InCore4indexERIContraction<double,dcomplex>::JContract(
      MPI_Comm, TwoBodyContraction<double>&) const { CErr("NYI"); }
  template <>
  void InCore4indexERIContraction<double,dcomplex>::KContract(
      MPI_Comm, TwoBodyContraction<double>&) const { CErr("NYI"); }

  template <>
  void InCoreRIERIContraction<double,dcomplex>::JContract(
      MPI_Comm, TwoBodyContraction<double>&) const { CErr("NYI"); }
  template <>
  void InCoreRIERIContraction<double,dcomplex>::KContract(
      MPI_Comm, TwoBodyContraction<double>&) const { CErr("NYI"); }

  template class OneEInts<double>;
  template class OneEInts<dcomplex>;

  template class OneERelInts<double>;
  template class OneERelInts<dcomplex>;

  template OneEInts<dcomplex>::OneEInts(const OneEInts<double>&, int);
  template OneERelInts<dcomplex>::OneERelInts(const OneERelInts<double>&, int);
  template MultipoleInts<dcomplex>::MultipoleInts(const MultipoleInts<double>&, int);
  template DirectERI<dcomplex>::DirectERI(const DirectERI<double>&, int);
  template InCore4indexERI<dcomplex>::InCore4indexERI(const InCore4indexERI<double>&, int);
  template InCoreRIERI<dcomplex>::InCoreRIERI(const InCoreRIERI<double>&, int);
  template InCoreAuxBasisRIERI<dcomplex>::InCoreAuxBasisRIERI(const InCoreAuxBasisRIERI<double>&, int);

  template class InCore4indexERIContraction<double, double>;
  template class InCore4indexERIContraction<dcomplex, double>;
  template class InCore4indexERIContraction<dcomplex, dcomplex>;

  template class InCoreRIERIContraction<double, double>;
  template class InCoreRIERIContraction<dcomplex, double>;
  template class InCoreRIERIContraction<dcomplex, dcomplex>;

  template class GTODirectERIContraction<double, double>;
  template class GTODirectERIContraction<dcomplex, double>;
  template class GTODirectERIContraction<dcomplex, dcomplex>;

  template class InCore4indexRelERIContraction<double, double>;
  template class InCore4indexRelERIContraction<dcomplex, double>;
  template class InCore4indexRelERIContraction<dcomplex, dcomplex>;

  template class GTODirectRelERIContraction<double, double>;
  template class GTODirectRelERIContraction<dcomplex, double>;
  template class GTODirectRelERIContraction<dcomplex, dcomplex>;

  template class Integrals<double>;
  template class Integrals<dcomplex>;

}; // namespace ChronusQ
