# 
#  This file is part of the Chronus Quantum (ChronusQ) software package
#  
#  Copyright (C) 2014-2020 Li Research Group (University of Washington)
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License along
#  with this program; if not, write to the Free Software Foundation, Inc.,
#  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#  
#  Contact the Developers:
#    E-Mail: xsli@uw.edu
#  
#
#only support openblas built by CQ now.
if(NOT CQ_NEED_OPENBLAS)
  message ( FATAL_ERROR "Only OPENBLAS is supported!" )
endif()

include(ExternalProject)
set( TA_PREFIX ${PROJECT_SOURCE_DIR}/external/tiledarray )
set( TA_INCLUDEDIR ${TA_PREFIX}/include )
set( TA_LIBDIR ${TA_PREFIX}/lib )

include_directories("${TA_INCLUDEDIR}")
link_directories("${TA_LIBDIR}")

if( NOT EXISTS "${TA_INCLUDEDIR}/tiledarray.h" )
  ExternalProject_Add(tiledarray
    PREFIX ${TA_PREFIX}
    GIT_REPOSITORY https://github.com/ValeevGroup/tiledarray.git
    GIT_TAG 2aef4e6552a9ae5f4b1a4dbdaf1346adf3610abd 
    CMAKE_ARGS
    	-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    	-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    	-DMPI_C_COMPILER=${MPI_C_COMPILER}
    	-DMPI_CXX_COMPILER=${MPI_CXX_COMPILER}
  	-DCMAKE_INSTALL_PREFIX=${TA_PREFIX}
          -DEIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR}
  	-DENABLE_MKL=OFF
  	-D LAPACK_LIBRARIES="${PROJECT_SOURCE_DIR}/external/openblas/lib/libopenblas.a -lm -lgfortran"
    BUILD_IN_SOURCE 1
    UPDATE_COMMAND echo 'NO UPDATE'
  )
  if( TARGET openblas )
  add_dependencies( tiledarray openblas ) 
  endif()
  list(APPEND CQEX_DEP tiledarray)
  
  message( STATUS "Opting to build a copy of TiledArray")
else()
  message( STATUS "Found TiledArray")
endif()
list(APPEND CQEX_LINK ${TA_LIBDIR}/libtiledarray.so)
list(APPEND CQEX_LINK ${TA_LIBDIR}/libMADworld.so)


