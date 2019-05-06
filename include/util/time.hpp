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
#ifndef __INCLUDED_UTIL_TIME_HPP__
#define __INCLUDED_UTIL_TIME_HPP__

#include <chronusq_sys.hpp>

namespace ChronusQ {

#ifndef CQ_ENABLE_MPI
  using time_point = std::chrono::high_resolution_clock::time_point;
#else
  using time_point = double;
#endif


  static inline time_point tick() {

#ifndef CQ_ENABLE_MPI
    return std::chrono::high_resolution_clock::now();
#else
    return MPI_Wtime();
#endif

  }

  static inline double tock(const time_point& pt) {

    time_point now = tick();

#ifndef CQ_ENABLE_MPI
    return std::chrono::duration<double>(now - pt).count();
#else
    return (now - pt);
#endif

  }





}; // namespace ChronusQ

#endif
