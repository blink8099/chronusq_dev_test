/* 
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *  
 *  Copyright (C) 2014-2017 Li Research Group (University of Washington)
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
#ifndef __INCLUDED_BOILERPLATE_HPP__
#define __INCLUDED_BOILERPLATE_HPP__


#include <chronusq_sys.hpp>
#include <libint2/cxxapi.h>

namespace ChronusQ {

  /**
   *  \brief Initialize the ChronusQ environment
   *
   *  Sets up to the environment for a ChronusQ calculation.
   *  Currently initialized the libint2 environment and sets
   *  the default thread pool to a single thread (serial 
   *  calculation)
   */ 
  inline void initialize() {

    // Bootstrap libint2 env
    libint2::initialize();
#ifdef _OPENMP
    // Thread pool (default serial)
    omp_set_num_threads(1);
#endif

  }; // initialize

  /**
   *  \brief Finalized the ChronusQ environment
   *
   *  Cleans up anything that was initialized by ChronusQ::initialize.
   *  Currently finalized the libint2 environment.
   */ 
  inline void finalize() {

    // Finalize libint2 env
    libint2::finalize();

  }; // finalize

}; // namespace ChronusQ

#endif