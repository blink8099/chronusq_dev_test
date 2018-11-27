#
# This file is part of the Chronus Quantum (ChronusQ) software package
# 
# Copyright (C) 2014-2018 Li Research Group (University of Washington)
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

if( NOT FORCE_SHARED_BOOST )
  message(STATUS "ChronusQ will require static libraries to be installed: use FORCE_SHARED_BOOST=ON to toggle")
  set(Boost_USE_STATIC_LIBS    ON)
else()
  message(STATUS "ChronusQ will allow linkage to shared Boost libraries")
endif()

set(Boost_USE_MULTITHREADED  ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED system filesystem)

include_directories(${Boost_INCLUDE_DIRS})
list(APPEND CQ_EXT_LINK ${Boost_LIBRARIES})
