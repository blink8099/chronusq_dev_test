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
#

include(CheckCSourceCompiles)
include(FetchContent)

message ( "\n == Libcint ==\n" )

#
#  Find preinstalled Libcint unless turned off
#

if ( NOT CQ_BUILD_LIBCINT_TYPE STREQUAL "FORCE" )

  # Otherwise create dummy target
  if ( DEFINED Libcint_ROOT )

    set ( LIBCINT_LIBRARIES_FOUND FALSE )

    if ( EXISTS "${Libcint_ROOT}/lib64/libcint.a" )
      set ( LIBCINT_LIBRARIES "${Libcint_ROOT}/lib64/libcint.a" )
      set ( LIBCINT_LIBRARIES_FOUND TRUE )
    elseif ( EXISTS "${Libcint_ROOT}/lib64/libcint.so" )
      set ( LIBCINT_LIBRARIES "${Libcint_ROOT}/lib64/libcint.so" )
      set ( LIBCINT_LIBRARIES_FOUND TRUE )
    endif()

    if ( LIBCINT_LIBRARIES_FOUND AND
         EXISTS "${Libcint_ROOT}/include/cint.h" )
      
      set ( LIBCINT_INCLUDE_DIRS
          ${Libcint_ROOT}/include
      )

      # We don't know if quadmath was linked against this by this method
      # We'll just assume that if we can find it, it was.
      set(QUADMATH_TEST_SOURCE
      "
      #include <quadmath.h>
      int main() {
      fabsq(1);
      }
      ")
      
      set(CMAKE_REQUIRED_LIBRARIES "quadmath")
      check_c_source_compiles("${QUADMATH_TEST_SOURCE}" QUADMATH_FOUND)
      if ( QUADMATH_FOUND )
        list( APPEND LIBCINT_LIBRARIES quadmath )
      endif()

      add_library ( ChronusQ::Libcint INTERFACE IMPORTED )
      set_target_properties ( ChronusQ::Libcint PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LIBCINT_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES      "${LIBCINT_LIBRARIES}"
      )

      set ( LIBCINT_FOUND TRUE )

    endif()

  endif()

  if ( LIBCINT_FOUND )
    message ( STATUS "Found External Libcint Installation" )
  endif()

endif()


#
#  Build Libcint if a suitable libcint hasn't been found
#

if ( NOT TARGET ChronusQ::Libcint )

  if ( NOT CQ_BUILD_LIBCINT_TYPE STREQUAL "NONE" )

    message( STATUS "Opting to build a copy of Libcint" )

    # Set prefix
    if ( CQ_EXTERNAL_IN_BUILD_DIR )
      set ( CUSTOM_LIBCINT_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external/libcint )
      set ( CQ_EXT_ROOT ${CMAKE_CURRENT_BINARY_DIR}/external )
    else()
      set ( CUSTOM_LIBCINT_PREFIX ${PROJECT_SOURCE_DIR}/external/libcint )
      set ( CQ_EXT_ROOT ${PROJECT_SOURCE_DIR}/external )
    endif() 

    set ( ENABLE_STATIC_old ${ENABLE_STATIC} )
    set ( ENABLE_STATIC ON )

    FetchContent_Declare (
      Libcint
      PREFIX ${CUSTOM_LIBCINT_PREFIX}
      GIT_REPOSITORY "https://github.com/sunqm/libcint"
      GIT_TAG "v3.1.1"
      UPDATE_COMMAND cd ${CQ_EXT_ROOT}/libcint-src && patch -N < ${PROJECT_SOURCE_DIR}/external/libcint/patch/CMakeLists.txt.patch || patch -N < ${PROJECT_SOURCE_DIR}/external/libcint/patch/CMakeLists.txt.patch | grep "Skipping patch" -q
    )
  
    FetchContent_GetProperties ( Libcint )

    if ( NOT Libcint_POPULATED )

      message ( STATUS "Downloading Libcint..." )
      FetchContent_Populate ( Libcint )
      message ( STATUS "Downloading Libcint - Done" )

      add_subdirectory ( ${libcint_SOURCE_DIR} ${libcint_BINARY_DIR} )

    endif()

    set ( ENABLE_STATIC ${ENABLE_STATIC_old} )
    unset ( ENABLE_STATIC_old )

    if ( TARGET openblas )
      add_dependencies( cint openblas )
    endif()
  
    add_library ( ChronusQ::Libcint ALIAS cint )
  
  else()
  
    message ( FATAL_ERROR "Suitable Libcint installation could not be found! \
Set Libcint_ROOT to the prefix of the Libcint installation or turn \
CQ_ALLOW_BUILD_LIBCINT on."
    )
  
  endif()

endif()


list(APPEND CQ_EXT_LINK ChronusQ::Libcint)

message ( " == End Libcint ==\n" )

