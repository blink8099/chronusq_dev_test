#
# This file is part of the Chronus Quantum (ChronusQ) software package
# 
# Copyright (C) 2014-2019 Li Research Group (University of Washington)
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
# 
# Contact the Developers:
#   E-Mail: xsli@uw.edu
#

cmake_minimum_required(VERSION 3.11...3.14 FATAL_ERROR)


project(ChronusQ
    VERSION 0.5.0
    DESCRIPTION
    "Chronus Quantum is an ab initio computational chemistry \
software package that focuses on explicitly time-dependent and unconventional \
post-SCF quantum mechanical methods."
    LANGUAGES CXX C Fortran)


# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------

option(CQ_ENABLE_MPI       "Enable MPI parallelism"          OFF)
option(ENABLE_COVERAGE     "Enable coverage and profiling"   OFF)
option(CQ_LINALG_USESYSTEM "Use System defaults for LA libs" OFF)
option(CQ_EXTERNAL_OPENMP  "Force linking to an external OpenMP library" OFF)
option(CQ_ENABLE_TA  "Enable TA"  OFF)


# Libint building options
set(BUILD_LIBINT_TYPES NONE ALLOW FORCE)
set(CQ_BUILD_LIBINT_TYPE ALLOW CACHE STRING "Disable, enable, or force CQ to build libint")
set_property(CACHE CQ_BUILD_LIBINT_TYPE PROPERTY STRINGS ${BUILD_LIBINT_TYPES})

if ( NOT CQ_BUILD_LIBINT_TYPE IN_LIST BUILD_LIBINT_TYPES )
  message ( FATAL_ERROR "CQ_BUILD_LIBINT_TYPE must be one of ${BUILD_LIBINT_TYPES}" )
endif()


# Whether to build in external or build directory
option(CQ_EXTERNAL_IN_BUILD_DIR "Build external dependencies in the build directory" OFF)

if ( NOT FETCHCONTENT_BASE_DIR AND NOT CQ_EXTERNAL_IN_BUILD_DIR )
  set ( FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/external"
        CACHE STRING "" FORCE )
endif()

# -----------------------------------------------------------------------------

# Add specified paths to CMAKE_PREFIX_PATH
if( MPI_PREFIX )
  set( CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${MPI_PREFIX} )
endif()

# Append local cmake directory to find CMAKE Modules
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")


# C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)

# Intel C++14
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14" )
endif()



# Parallelism

message( "\n\n" )
message( "ChronusQ Parallelism Settings" )

# OpenMP
message( "" )
if( CQ_EXTERNAL_OPENMP )
  message ("  Forcing linking to external OpenMP library!")
  message ("  Using user specified OpenMP flags: ${OpenMP_CXX_FLAGS}")
  set(CQ_EXT_LINK "omp;${CQ_EXT_LINK}")
else()
  find_package(OpenMP REQUIRED)
  message ("  Using automatic OpenMP flags: ${OpenMP_CXX_FLAGS}")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


if( MPI_PREFIX )
  # MPI_PREFIX paths to CMAKE_PREFIX_PATH
  message( "" )
  message( STATUS "Adding ${MPI_PREFIX} to CMAKE_PREFIX_PATH" )
  set( CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${MPI_PREFIX} )
endif()

# MPI
if(CQ_ENABLE_MPI)
  include(HandleMPI)
endif()


message( "\n\n" )









# FLAGS
include(HandleCXXFlags)


# Define include directories
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_BINARY_DIR}/include")

# Linear Algebra
include(HandleLA)



# Non-trivial dependencies
message( "ChronusQ Dependencies" )


include(HandleLibint)
include(HandleLibcint)
include(FindLibXC)

set(CQEX_LINK ${CQ_EXT_LINK} )

# HDF5
find_package(HDF5 REQUIRED CXX)
add_library( ChronusQ::HDF5 INTERFACE IMPORTED )
set_target_properties( ChronusQ::HDF5 PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIRS}"
  INTERFACE_LINK_LIBRARIES      "${HDF5_CXX_LIBRARIES}"
)
list(APPEND CQEX_LINK ChronusQ::HDF5 )


message( "\n\n" )


if( CQ_ENABLE_TA )
  if(NOT CQ_ENABLE_MPI)
    message(FATAL_ERROR "TiledArray must be compiled with mpi!")
  endif()
  message("Enable TiledArray")
  include(FindTiledArray)
  set( CQ_HAS_TA ON )
else()
  message("TiledArray will not be compiled, Coupled-Cluster will be disabled.")
endif()

# Compiling with coverage report
if(ENABLE_COVERAGE)

  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    message(STATUS "ENABLE_COVERAGE Only Compatible with GNU Compilers!")
  else()
    message(STATUS "Enabling Coverage and Profiling -- Warning! May impede performance")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  endif()
endif()



set(BASIS_PATH     "${PROJECT_SOURCE_DIR}/basis"     )
set(AUTHOR_PATH    "${PROJECT_SOURCE_DIR}/AUTHORS"   )

configure_file(
  "${PROJECT_SOURCE_DIR}/include/chronusq_config.hpp.in"
  "${PROJECT_BINARY_DIR}/include/chronusq_config.hpp"
)

# Recurse into subdirectories
add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
