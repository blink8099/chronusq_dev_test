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

# Directories for SCF tests
set(SCF_TEST_SOURCE_ROOT "${TEST_ROOT}/scf" )
set(SCF_TEST_BINARY_ROOT "${TEST_BINARY_ROOT}/scf" )
  

# Set up compilation of SCF test exe
add_executable(scftest ../ut.cxx 
	rhf.cxx uhf.cxx x2chf.cxx rhfnr.cxx
        rhf_giao.cxx uhf_giao.cxx ghf_giao.cxx
	rks.cxx uks.cxx x2cks.cxx ks.cxx
        misc.cxx
)

target_include_directories(scftest PUBLIC ${SCF_TEST_SOURCE_ROOT} 
  ${TEST_BINARY_ROOT})
target_link_libraries(scftest PUBLIC ${CQEX_LINK})
target_link_libraries(scftest PUBLIC ChronusQ::gtest)

if(CQEX_DEP)
  add_dependencies(scftest ${CQEX_DEP})
endif()

# Generate directories
file(MAKE_DIRECTORY ${SCF_TEST_BINARY_ROOT}/serial/rhf)
file(MAKE_DIRECTORY ${SCF_TEST_BINARY_ROOT}/serial/rks)
file(MAKE_DIRECTORY ${SCF_TEST_BINARY_ROOT}/serial/uhf)
file(MAKE_DIRECTORY ${SCF_TEST_BINARY_ROOT}/serial/uks)
file(MAKE_DIRECTORY ${SCF_TEST_BINARY_ROOT}/serial/x2c)
file(MAKE_DIRECTORY ${SCF_TEST_BINARY_ROOT}/serial/rhf_giao)
file(MAKE_DIRECTORY ${SCF_TEST_BINARY_ROOT}/serial/uhf_giao)
file(MAKE_DIRECTORY ${SCF_TEST_BINARY_ROOT}/serial/ghf_giao)

file(MAKE_DIRECTORY ${SCF_TEST_BINARY_ROOT}/parallel/rhf)
file(MAKE_DIRECTORY ${SCF_TEST_BINARY_ROOT}/parallel/rks)
file(MAKE_DIRECTORY ${SCF_TEST_BINARY_ROOT}/parallel/uhf)
file(MAKE_DIRECTORY ${SCF_TEST_BINARY_ROOT}/parallel/uks)
file(MAKE_DIRECTORY ${SCF_TEST_BINARY_ROOT}/parallel/x2c)
file(MAKE_DIRECTORY ${SCF_TEST_BINARY_ROOT}/parallel/rhf_giao)
file(MAKE_DIRECTORY ${SCF_TEST_BINARY_ROOT}/parallel/uhf_giao)
file(MAKE_DIRECTORY ${SCF_TEST_BINARY_ROOT}/parallel/ghf_giao)


include( CQTestGeneration )

# Add tests
add_cq_test( RHF_SCF      scftest "RHF.*"      )
add_cq_test( UHF_SCF      scftest "UHF.*"      )
add_cq_test( X2CHF_SCF    scftest "X2CHF.*"    )
add_cq_test( RHF_NR_SCF   scftest "RHF_NR.*"   )
add_cq_test( RHF_GIAO_SCF scftest "RHF_GIAO.*" )
add_cq_test( UHF_GIAO_SCF scftest "UHF_GIAO.*" )
add_cq_test( GHF_GIAO_SCF scftest "GHF_GIAO.*" )

add_cq_test( RKS_SCF      scftest "RKS.*"        )
add_cq_test( UKS_SCF      scftest "UKS.*"        )
add_cq_test( X2CKS_SCF    scftest "X2CKS.*"      )
add_cq_test( KS_KEYWORD   scftest "KS_KEYWORD.*" )
add_cq_test( KS_FUNC      scftest "KS_FUNC.*"    )

add_cq_test( MISC_SCF     scftest "MISC_SCF.*" )


if( CQ_ENABLE_MPI )
  add_cq_mpi_test( RHF_SCF_MPI      2   scftest "RHF.*"   )
  add_cq_mpi_test( UHF_SCF_MPI      2   scftest "UHF.*"   )
  add_cq_mpi_test( X2CHF_SCF_MPI    2   scftest "X2CHF.*" )
  add_cq_mpi_test( RHF_GIAO_SCF_MPI 2   scftest "RHF_GIAO.*"   )
  #add_cq_mpi_test( UHF_GIAO_SCF_MPI 2   scftest "UHF_GIAO.*"   )

  add_cq_mpi_test( RKS_SCF_MPI 2   scftest "RKS.*"   )
  add_cq_mpi_test( UKS_SCF_MPI 2   scftest "UKS.*"   )
  #add_cq_mpi_test( X2CKS_SCF_MPI 2 scftest "X2CKS.*" )
endif()



