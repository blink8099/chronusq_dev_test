#
# This file is part of the Chronus Quantum (ChronusQ) software package
# 
# Copyright (C) 2014-2020 Li Research Group (University of Washington)
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


include(ExternalProject)

message( "\n\n" )
message( "ChronusQ Linear Algebra Settings:\n" )

# Eigen3
find_package(Eigen3 REQUIRED)
include_directories("${EIGEN3_INCLUDE_DIR}")

######## BLAS + LAPACK LIBRARIES ########

set( BLAS_FOUND OFF )

# Find external BLAS installation
if( BLAS_EXTERNAL )

  find_package( BLAS QUIET )

  if( BLAS_FOUND )
    message( STATUS "Found BLAS library: ${BLAS_LIBRARIES}" )
    add_library( ChronusQ::BLAS INTERFACE IMPORTED )
    set_target_properties( ChronusQ::BLAS PROPERTIES
      INTERFACE_LINK_LIBRARIES      "${BLAS_LIBRARIES}"
      INTERFACE_LINK_FLAGS          "${BLAS_LINKER_FLAGS}"
    )
  endif()

endif()

if (NOT BLAS_FOUND)
  
  # Try to find OpenBLAS already compiled by CQ
  set( OPENBLAS_PREFIX ${PROJECT_SOURCE_DIR}/external/openblas )
  list( APPEND CMAKE_PREFIX_PATH ${OPENBLAS_PREFIX} )
  find_package( OpenBLAS PATHS ${OPENBLAS_PREFIX} NO_DEFAULT_PATH)
  
  # Check if we found it
  if( OpenBLAS_DIR )
    message( STATUS "Found BLAS library: ${OpenBLAS_LIBRARIES}" )
    set( BLAS_FOUND TRUE )
    add_library( ChronusQ::BLAS INTERFACE IMPORTED )
    set_target_properties( ChronusQ::BLAS PROPERTIES
      INTERFACE_LINK_LIBRARIES      "${OpenBLAS_LIBRARIES}"
      INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIRS}"
    )
  
    
  # Build OpenBLAS
  else() 
  
    message(STATUS "No BLAS/LAPACK Libraries Have Been Found: Defaulting to Build OpenBLAS")
    set( OPENBLAS_PREFIX      ${PROJECT_SOURCE_DIR}/external/openblas )
    set( OPENBLAS_INCLUDEDIR  ${OPENBLAS_PREFIX}/include )
    set( OPENBLAS_LIBDIR      ${OPENBLAS_PREFIX}/lib )
    set( OPENBLAS_LAPACK_SRC  ${OPENBLAS_PREFIX}/src/openblas/lapack-netlib/SRC/ )

    if( OPENBLAS_TARGET )
      message( STATUS "---> Forcing OpenBLAS TARGET = ${OPENBLAS_TARGET}" )
      set(OPENBLAS_BUILD_COMMAND $(MAKE) TARGET=${OPENBLAS_TARGET})
    else()
      message( STATUS "---> Allowing OpenBLAS to determine CPU TARGET" )
      set(OPENBLAS_BUILD_COMMAND $(MAKE))
    endif()

    if(OPENBLAS_DYNAMIC_ARCH)
      message(" Turn On Dynamic_ARCH for OpenBlas")
      set(OPENBLAS_BUILD_COMMAND ${OPENBLAS_BUILD_COMMAND} DYNAMIC_ARCH=1)
    endif()

    ExternalProject_Add(
      openblas
      PREFIX ${OPENBLAS_PREFIX}
      GIT_REPOSITORY "https://github.com/xianyi/OpenBLAS.git"
      GIT_TAG "v0.3.9"
      CONFIGURE_COMMAND cd ${OPENBLAS_LAPACK_SRC}
        && patch < ${OPENBLAS_PREFIX}/patch/chgeqz.patch
        && patch < ${OPENBLAS_PREFIX}/patch/zhgeqz.patch
  
      BUILD_COMMAND ${OPENBLAS_BUILD_COMMAND} CFLAGS='-Wno-error=implicit-function-declaration'
      BUILD_IN_SOURCE 1
      INSTALL_COMMAND make install PREFIX=${OPENBLAS_PREFIX}
        && cd ${OPENBLAS_INCLUDEDIR}
        && patch < ${OPENBLAS_PREFIX}/patch/lapack.patch
        && patch < ${OPENBLAS_PREFIX}/patch/f77blas.patch
    )
  
    file( MAKE_DIRECTORY ${OPENBLAS_PREFIX}/include )
    file( MAKE_DIRECTORY ${OPENBLAS_PREFIX}/lib )

    add_library( ChronusQ::BLAS INTERFACE IMPORTED )
    set_target_properties( ChronusQ::BLAS PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${OPENBLAS_INCLUDEDIR}
      INTERFACE_LINK_LIBRARIES      ${OPENBLAS_LIBDIR}/libopenblas.a
      INTERFACE_LINK_FLAGS          -lopenblas
    )
    add_dependencies( ChronusQ::BLAS openblas )
  
    # Mirror find_package( BLAS ) variables
    set( BLAS_FOUND TRUE )
    set( BLAS_LIBRARIES ${OPENBLAS_LIBDIR}/libopenblas.a )
    set( BLAS_INCLUDE_DIRS ${OPENBLAS_INCLUDEDIR} )
  
    set( CQ_LINALG_INCLUDEDIR ${OPENBLAS_INCLUDEDIR} )
  
    link_directories( ${OPENBLAS_LIBDIR} )
  
  endif()
endif()

set( CQ_LINALG_LIBRARIES ChronusQ::BLAS )

if( CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set( CQ_LINALG_LIBRARIES gfortran "${CQ_LINALG_LIBRARIES}" )
elseif( CMAKE_CXX_COMPILER_ID STREQUAL "Intel" )
    set( _CQ_MKL 1 )
    get_target_property( link_flags ChronusQ::BLAS INTERFACE_LINK_FLAGS )
    message( " Intel link flags: ${link_flags}" )
endif()

######## BLACS + ScaLAPACK LIBRARIES ########
if( CQ_ENABLE_MPI )


  message( STATUS "CQ_ENABLE_MPI Triggers Search for ScaLAPACK/BLACS" )

  # CXXBLACS
  set( CQ_ENABLE_CXXBLACS TRUE )
  message( STATUS "---> Creating CMake Target for CXXBLACS" )

  set( CXXBLACS_PREFIX     ${PROJECT_SOURCE_DIR}/external/cxxblacs )
  set( CXXBLACS_INCLUDEDIR ${CXXBLACS_PREFIX}/src/cxxblacs/include )
  
  ExternalProject_Add(cxxblacs
    PREFIX ${CXXBLACS_PREFIX}
    GIT_REPOSITORY https://github.com/wavefunction91/CXXBLACS.git
    CONFIGURE_COMMAND echo 'No CXXBLACS Configure'
    UPDATE_COMMAND echo 'No CXXBLACS Update Command'
    BUILD_COMMAND echo 'No CXXBLACS Build'
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND echo 'No CXXBLACS Install'
  )

  list(APPEND CQEX_DEP cxxblacs)
  
  # CXXBLACS Includes
  include_directories(${CXXBLACS_INCLUDEDIR})



  # Try to find ScaLAPACK
  if( NOT CQ_SCALAPACK_LIBRARIES )

    # Intel Compilers 
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    
      message( STATUS "---> Setting ScaLAPACK/BLACS Defaults for MKL" )
      set( CQ_SCALAPACK_LIBRARIES "-lmkl_scalapack_lp64"      )
      set( CQ_BLACS_LIBRARIES     "-lmkl_blacs_intelmpi_lp64" )
    
    else()

      # Attempt to find, but if not, trigger build
      #find_package(SCALAPACK)
      #if( SCALAPACK_FOUND )
      #  set( CQ_SCALAPACK_LIBRARIES ${SCALAPACK_LIBRARIES} )
      #endif()

    endif()


  endif()


  # If ScaLAPACK still not found, build it
  if( NOT CQ_SCALAPACK_LIBRARIES )

    message(STATUS "---> No BLACS/ScaLAPACK Libraries Have Been Found: Defaulting to Build A Local Copy")

    set( SCALAPACK_PREFIX      ${PROJECT_SOURCE_DIR}/external/scalapack )
    set( SCALAPACK_INCLUDEDIR  ${SCALAPACK_PREFIX}/include )
    set( SCALAPACK_LIBDIR      ${SCALAPACK_PREFIX}/lib )
    set( SCALAPACK_LIBRARIES   ${SCALAPACK_LIBDIR}/libscalapack.a )

    if( NOT EXISTS ${SCALAPACK_LIBRARIES} )

      ExternalProject_Add(libscalapack_build
        PREFIX ${SCALAPACK_PREFIX}
        URL "http://www.netlib.org/scalapack/scalapack-2.0.2.tgz"
        UPDATE_COMMAND echo 'No ScaLAPACK Update Command'
        PATCH_COMMAND  echo 'No ScaLAPACK Patch Command'
        CMAKE_ARGS
          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
          -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
          -DMPI_C_COMPILER=${MPI_C_COMPILER}
          -DMPI_Fortran_COMPILER=${MPI_Fortran_COMPILER}
          -DCMAKE_INSTALL_PREFIX=${SCALAPACK_PREFIX}
      )
      
      install(DIRECTORY "${SCALAPACK_PREFIX}/include" DESTINATION ".")
      install(DIRECTORY "${SCALAPACK_PREFIX}/lib"     DESTINATION ".")

      list(APPEND CQEX_DEP libscalapack_build)


    endif()

    set( CQ_SCALAPACK_LIBRARIES ${SCALAPACK_LIBRARIES} )



  endif()







  # Append ScaLAPACK / BLACS to linker
  set( CQ_LINALG_LIBRARIES ${CQ_LINALG_LIBRARIES} "${CQ_SCALAPACK_LIBRARIES}")
  if( CQ_BLACS_LIBRARIES )
    set( CQ_LINALG_LIBRARIES ${CQ_LINALG_LIBRARIES} "${CQ_BLACS_LIBRARIES}")
  endif()

endif( CQ_ENABLE_MPI )










# Add CQ_LINALG_LIBRARIES to linker
if( CQ_LINALG_LIBRARIES )

  list(APPEND CQ_EXT_LINK ${CQ_LINALG_LIBRARIES})

  message(STATUS "CQ_LINALG_LIBRARIES = ${CQ_LINALG_LIBRARIES}")

endif( CQ_LINALG_LIBRARIES )


# If we need headers for LA
if( CQ_LINALG_INCLUDEDIR )

  include_directories( ${CQ_LINALG_INCLUDEDIR} )
  message(STATUS "CQ_LINALG_INCLUDEDIR = ${CQ_LINALG_INCLUDEDIR}")

endif( CQ_LINALG_INCLUDEDIR)


# Find external BLAS++ installation
find_package( blaspp QUIET )

if( blaspp_FOUND )
  message( STATUS "Found external BLAS++ installation" )
  add_library( ChronusQ::blaspp ALIAS blaspp )
else() 

  set( blaspp_PREFIX ${PROJECT_SOURCE_DIR}/external/blaspp )
  list( APPEND CMAKE_PREFIX_PATH ${blaspp_PREFIX} )
  find_package( blaspp QUIET )

  if( blaspp_FOUND )
    message( STATUS "Found previously installed BLAS++" )
    add_library( ChronusQ::blaspp ALIAS blaspp )
  else()
    message( STATUS "Could not find BLAS++. Compiling BLAS++." )
    string( REPLACE ";" ":" prefix_path_arg "${CMAKE_PREFIX_PATH}" )
    string( REPLACE ";" ":" blas_libs_arg "${BLAS_LIBRARIES}" )
    ExternalProject_Add (
      blaspp
      PREFIX ${blaspp_PREFIX}
      GIT_REPOSITORY https://bitbucket.org/icl/blaspp.git
      GIT_TAG ed392fe
      LIST_SEPARATOR ":"
      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${blaspp_PREFIX}
                 -DBLAS_LIBRARIES=${blas_libs_arg}
                 -DCMAKE_PREFIX_PATH=${prefix_path_arg}
                 -DCMAKE_INSTALL_NAME_DIR=${blaspp_PREFIX}/lib
                 -DCMAKE_INSTALL_LIBDIR=lib
    )
    add_dependencies( blaspp ChronusQ::BLAS )

    # Add target with nice properties since ExternalProject can't figure it out
    add_library( ChronusQ::blaspp INTERFACE IMPORTED )
    file( MAKE_DIRECTORY ${blaspp_PREFIX}/include )
    set_target_properties( ChronusQ::blaspp PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${blaspp_PREFIX}/include
      INTERFACE_LINK_LIBRARIES      "${blaspp_PREFIX}/lib/libblaspp${CMAKE_SHARED_LIBRARY_SUFFIX};ChronusQ::BLAS"
    )
    add_dependencies( ChronusQ::blaspp blaspp )
  endif()
endif()



# Find external LAPACK++ installation
find_package( lapackpp QUIET )

if( lapackpp_FOUND )
  message( STATUS "Found external LAPACK++ installation" )
  add_library( ChronusQ::lapackpp ALIAS lapackpp )
else() 

  set( lapackpp_PREFIX ${PROJECT_SOURCE_DIR}/external/lapackpp )
  list( APPEND CMAKE_PREFIX_PATH ${lapackpp_PREFIX} )
  find_package( lapackpp QUIET )

  if( lapackpp_FOUND )
    message( STATUS "Found previously installed LAPACK++" )
    add_library( ChronusQ::lapackpp ALIAS lapackpp )
  else()
    message( STATUS "Could not find LAPACK++. Compiling LAPACK++." )
    string( REPLACE ";" ":" prefix_path_arg "${CMAKE_PREFIX_PATH}" )
    ExternalProject_Add (
      lapackpp
      PREFIX ${lapackpp_PREFIX}
      GIT_REPOSITORY https://bitbucket.org/icl/lapackpp.git
      GIT_TAG dbcf60f
      LIST_SEPARATOR ":"
      CMAKE_ARGS -DCMAKE_PREFIX_PATH=${prefix_path_arg}
                 -DCMAKE_INSTALL_PREFIX=${lapackpp_PREFIX}
                 -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                 -DCMAKE_INSTALL_NAME_DIR=${lapackpp_PREFIX}/lib
                 -DCMAKE_INSTALL_LIBDIR=lib
      PATCH_COMMAND cd ${lapackpp_PREFIX}/src/lapackpp
      && patch -p0 < ${lapackpp_PREFIX}/patch/fortran.patch 
      && patch -p0 < ${lapackpp_PREFIX}/patch/config.patch || exit 0
    )
    add_dependencies( lapackpp ChronusQ::blaspp )

    # Add target with nice properties since ExternalProject can't figure it out
    add_library( ChronusQ::lapackpp INTERFACE IMPORTED )
    file( MAKE_DIRECTORY ${lapackpp_PREFIX}/include )
    set_target_properties( ChronusQ::lapackpp PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${lapackpp_PREFIX}/include
      INTERFACE_LINK_LIBRARIES      "${lapackpp_PREFIX}/lib/liblapackpp${CMAKE_SHARED_LIBRARY_SUFFIX};ChronusQ::BLAS"
    )
    add_dependencies( ChronusQ::lapackpp lapackpp )
  endif()
endif()


include_directories($<TARGET_PROPERTY:ChronusQ::blaspp,INTERFACE_INCLUDE_DIRECTORIES>)
include_directories($<TARGET_PROPERTY:ChronusQ::lapackpp,INTERFACE_INCLUDE_DIRECTORIES>)
list(APPEND CQEX_DEP ChronusQ::blaspp ChronusQ::lapackpp )


message( "\n\n\n" )
