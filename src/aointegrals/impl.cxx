/* 
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *  
 *  Copyright (C) 2014-2019 Li Research Group (University of Washington)
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
#include <aointegrals/impl.hpp>

namespace ChronusQ {

  template class AOIntegrals<double>;
  template class AOIntegrals<dcomplex>;

  // Explicit instantiations of 2-body contraction engines
  template void AOIntegrals<double>::twoBodyContractIncore(
    MPI_Comm c, std::vector<TwoBodyContraction<double>> &list);
  template void AOIntegrals<double>::twoBodyContractIncore(
    MPI_Comm c, std::vector<TwoBodyContraction<dcomplex>> &list);
  template void AOIntegrals<dcomplex>::twoBodyContractIncore(
    MPI_Comm c, std::vector<TwoBodyContraction<dcomplex>> &list);

  template void AOIntegrals<double>::twoBodyContractDirect(
    MPI_Comm c, const bool screen, std::vector<TwoBodyContraction<double>> &list, EMPerturbation &pert);
  template void AOIntegrals<double>::twoBodyContractDirect(
    MPI_Comm c, const bool screen, std::vector<TwoBodyContraction<dcomplex>> &list, EMPerturbation &pert);
  template void AOIntegrals<dcomplex>::twoBodyContractDirect(
    MPI_Comm c, const bool screen, std::vector<TwoBodyContraction<dcomplex>> &list, EMPerturbation &pert);


}; // namespace ChronusQ
