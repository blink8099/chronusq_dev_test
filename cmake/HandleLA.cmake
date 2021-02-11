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

# Whether or not we need to build OpenBLAS
set( CQ_NEED_OPENBLAS ON )





# Use Externally set Linear Algebra libraries
if( CQ_LINALG_LIBRARIES )
  set( CQ_NEED_OPENBLAS OFF )
  message( STATUS 
    "Using User Specified CQ_LINALG_LIBRARIES Linker: ${CQ_LINALG_LIBRARIES}" )

# Try to figure out  LA linkage
else( CQ_LINALG_LIBRARIES )


  # Use system LA (not recommended
  if( CQ_LINALG_USESYSTEM )

    message( STATUS 
      "CQ_LINALG_USESYSTEM Triggered Search for Sytem Installation of BLAS/LAPACK" )
    set( CQ_NEED_OPENBLAS OFF )
    find_package( BLAS REQUIRED   )
    find_package( LAPACK REQUIRED )

    set( CQ_LINALG_LIBRARIES ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} )

  endif( CQ_LINALG_USESYSTEM )




  # Intel Compilers 
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  
    set( CQ_NEED_OPENBLAS OFF )
    message(STATUS "Intel Compiler -- Using MKL: -mkl=parallel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=parallel")
    set(_CQ_MKL 1)
  
  endif()


endif( CQ_LINALG_LIBRARIES )




# Build OpenBLAS
if( CQ_NEED_OPENBLAS )

  message(STATUS "No BLAS/LAPACK Libraries Have Been Found: Defaulting to Build OpenBLAS")
  set( OPENBLAS_PREFIX      ${PROJECT_SOURCE_DIR}/external/openblas )
  set( OPENBLAS_INCLUDEDIR  ${OPENBLAS_PREFIX}/include )
  set( OPENBLAS_LIBDIR      ${OPENBLAS_PREFIX}/lib )
  set( OPENBLAS_LAPACK_SRC      ${OPENBLAS_PREFIX}/src/openblas/lapack-netlib/SRC/ )
  if( NOT EXISTS "${OPENBLAS_INCLUDEDIR}/f77blas.h" )


    if( OPENBLAS_TARGET )
      message( STATUS "---> Forcing OpenBLAS TARGET = ${OPENBLAS_TARGET}" )
      set(OPENBLAS_BUILD_COMMAND make TARGET=${OPENBLAS_TARGET} -j2)
    else()
      message( STATUS "---> Allowing OpenBLAS to determine CPU TARGET" )
      set(OPENBLAS_BUILD_COMMAND make -j2)
    endif()
    If(OPENBLAS_DYNAMIC_ARCH)
      message(" Turn On Dynamic_ARCH for OpenBlas")
      set(OPENBLAS_BUILD_COMMAND ${OPENBLAS_BUILD_COMMAND} DYNAMIC_ARCH=1)
    endif()
    ExternalProject_Add(openblas
      PREFIX ${OPENBLAS_PREFIX}
      GIT_REPOSITORY "https://github.com/xianyi/OpenBLAS.git"
      GIT_TAG "v0.3.9"
      CONFIGURE_COMMAND cd ${OPENBLAS_LAPACK_SRC}
        && patch < ${OPENBLAS_PREFIX}/patch/chgeqz.patch
        && patch < ${OPENBLAS_PREFIX}/patch/zhgeqz.patch

      BUILD_COMMAND ${OPENBLAS_BUILD_COMMAND}
      BUILD_IN_SOURCE 1
      INSTALL_COMMAND make install PREFIX=${OPENBLAS_PREFIX}
        && cd ${OPENBLAS_INCLUDEDIR}
        && patch < ${OPENBLAS_PREFIX}/patch/lapack.patch
        && patch < ${OPENBLAS_PREFIX}/patch/f77blas.patch

    )

    install(DIRECTORY "${OPENBLAS_PREFIX}/include" DESTINATION ".")
    install(DIRECTORY "${OPENBLAS_PREFIX}/lib"     DESTINATION ".")

    list(APPEND CQEX_DEP openblas)


  else()
    message( STATUS "---> OpenBLAS Has Already Been Built for ChronusQ!" )
  endif()



  set( CQ_LINALG_LIBRARIES  gfortran ${OPENBLAS_LIBDIR}/libopenblas.a )
  set( CQ_LINALG_INCLUDEDIR ${OPENBLAS_INCLUDEDIR} )

  link_directories( ${OPENBLAS_LIBDIR} )

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



message( "\n\n\n" )
