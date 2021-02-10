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

include(FetchContent)

message ( "\n == Libint ==\n" )

#
#  Find preinstalled Libint unless turned off
#

if ( NOT CQ_BUILD_LIBINT_TYPE STREQUAL "FORCE" )

  find_package ( Libint2 QUIET )


  # Prefer CMake installed
  if ( TARGET Libint2::int2 )

    get_target_property ( libint_alias Libint2::int2 ALIASED_TARGET )

    if ( libint_alias )
      add_library ( ChronusQ::Libint2 ALIAS ${libint_alias} )
    else()
      add_library ( ChronusQ::Libint2 ALIAS Libint2::int2 )
    endif()

  # Otherwise create dummy target
  elseif ( DEFINED Libint2_ROOT )

    set ( LIBINT2_LIBRARIES_FOUND FALSE )

    if ( EXISTS "${Libint2_ROOT}/lib/libint2.a" )
      set ( LIBINT2_LIBRARIES "${Libint2_ROOT}/lib/libint2.a" )
      set ( LIBINT2_LIBRARIES_FOUND TRUE )
    elseif ( EXISTS "${Libint2_ROOT}/lib/liblibint2.a" )
      set ( LIBINT2_LIBRARIES "${Libint2_ROOT}/lib/liblibint2.a" )
      set ( LIBINT2_LIBRARIES_FOUND TRUE )
    endif()

    if ( LIBINT2_LIBRARIES_FOUND AND
         EXISTS "${Libint2_ROOT}/include/libint2.hpp" )
      
      set ( LIBINT2_INCLUDE_DIRS
          ${Libint2_ROOT}/include
          ${Libint2_ROOT}/include/libint2
      )

      add_library ( ChronusQ::Libint2 INTERFACE IMPORTED )
      set_target_properties ( ChronusQ::Libint2 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LIBINT2_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES      "${LIBINT2_LIBRARIES}"
      )

      set ( LIBINT2_FOUND TRUE )

    endif()

  endif()

  if ( LIBINT2_FOUND )
    message ( STATUS "Found External Libint Installation" )
  endif()

endif()


#
#  Build Libint if a suitable libint hasn't been found
#

if ( NOT TARGET ChronusQ::Libint2 )

  if ( NOT CQ_BUILD_LIBINT_TYPE STREQUAL "NONE" )

    message( STATUS "Opting to build a copy of Libint" )

    # Set prefix
    if ( CQ_EXTERNAL_IN_BUILD_DIR )
      set ( CUSTOM_LIBINT_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external/libint2 )
    else()
      set ( CUSTOM_LIBINT_PREFIX ${PROJECT_SOURCE_DIR}/external/libint2 )
    endif() 

    # Update policy for Libint
    set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)
  
    FetchContent_Declare (
      Libint2
      PREFIX ${CUSTOM_LIBINT_PREFIX}
      GIT_REPOSITORY "https://urania.chem.washington.edu/chronusq/libint-cq.git"
      GIT_TAG "2.7.0-beta.6"
    )
  
    FetchContent_GetProperties ( Libint2 )

    if ( NOT Libint2_POPULATED )

      message ( STATUS "Downloading Libint..." )
      FetchContent_Populate ( Libint2 )
      message ( STATUS "Downloading Libint - Done" )

      add_subdirectory ( ${libint2_SOURCE_DIR} ${libint2_BINARY_DIR} )

    endif()
  
    install(TARGETS libint2  
            ARCHIVE DESTINATION  "lib/libint2"
            LIBRARY DESTINATION  "lib/libint2"
            INCLUDES DESTINATION "include/libint2")
    
    add_library ( ChronusQ::Libint2 ALIAS libint2 )
  
  else()
  
    message ( FATAL_ERROR "Suitable Libint installation could not be found! \
Set Libint2_ROOT to the prefix of the Libint installation or turn \
CQ_ALLOW_BUILD_LIBINT on."
    )
  
  endif()

endif()

list(APPEND CQ_EXT_LINK ChronusQ::Libint2)

message ( " == End Libint ==\n" )
