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
#include <singleslater/neoss.hpp>
#include <singleslater/impl.hpp>
#include <corehbuilder/impl.hpp>
#include <fockbuilder/impl.hpp>

namespace ChronusQ {

  template class SingleSlater<double,double>;
  template class SingleSlater<dcomplex,double>;
  template class SingleSlater<dcomplex,dcomplex>;

  // Instantiate copy constructors
  template SingleSlater<dcomplex,double>::SingleSlater(const SingleSlater<double,double> &, int);
  template SingleSlater<dcomplex,double>::SingleSlater(const SingleSlater<dcomplex,double> &, int);
  template SingleSlater<dcomplex,dcomplex>::SingleSlater(const SingleSlater<dcomplex,dcomplex> &, int);

  // Instantiate move ctors
  template SingleSlater<dcomplex,double>::SingleSlater( SingleSlater<double,double> &&, int);

  template class NEOSingleSlater<double,double>;
  template class NEOSingleSlater<dcomplex,double>;
  template class NEOSingleSlater<dcomplex,dcomplex>;

  // Instantiate copy constructors
  template NEOSingleSlater<dcomplex,double>::NEOSingleSlater(const NEOSingleSlater<double,double> &, int);
  template NEOSingleSlater<dcomplex,dcomplex>::NEOSingleSlater(const NEOSingleSlater<dcomplex,dcomplex> &, int);

  // Instantiate move ctors
  template NEOSingleSlater<dcomplex,double>::NEOSingleSlater( NEOSingleSlater<double,double> &&, int);

  template class NEOKohnSham<double,double>;
  template class NEOKohnSham<dcomplex,double>;
  template class NEOKohnSham<dcomplex,dcomplex>;
  // Instantiate copy constructors
  template NEOKohnSham<dcomplex,double>::NEOKohnSham(const NEOKohnSham<double,double> &, int);
  // Instantiate copy ructors
  template NEOKohnSham<dcomplex,double>::NEOKohnSham( NEOKohnSham<double,double> &&, int);

  template class HartreeFock<double,double>;
  template class HartreeFock<dcomplex,double>;
  template class HartreeFock<dcomplex,dcomplex>;
  // Instantiate copy constructors
  template HartreeFock<dcomplex,double>::HartreeFock(const HartreeFock<double,double> &, int);
  // Instantiate copy ructors
  template HartreeFock<dcomplex,double>::HartreeFock( HartreeFock<double,double> &&, int);

  template class KohnSham<double,double>;
  template class KohnSham<dcomplex,double>;
  template class KohnSham<dcomplex,dcomplex>;
  // Instantiate copy constructors
  template KohnSham<dcomplex,double>::KohnSham(const KohnSham<double,double> &, int);
  // Instantiate copy ructors
  template KohnSham<dcomplex,double>::KohnSham( KohnSham<double,double> &&, int);

  template void KohnSham<double,double>::formFXC(MPI_Comm,std::vector<TwoBodyContraction<double>> &);
  template void KohnSham<double,double>::formFXC(MPI_Comm,std::vector<TwoBodyContraction<dcomplex>> &);

  template void KohnSham<dcomplex,double>::formFXC(MPI_Comm,std::vector<TwoBodyContraction<dcomplex>> &);
  template void KohnSham<dcomplex,dcomplex>::formFXC(MPI_Comm,std::vector<TwoBodyContraction<dcomplex>> &);

  template class NEOSS<double,double>;

}; // namespace ChronusQ
