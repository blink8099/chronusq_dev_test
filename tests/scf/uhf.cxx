/* 
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *  
 *  Copyright (C) 2014-2020 Li Research Group (University of Washington)
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

#include "scf.hpp"



// Li 6-31G(d) test
TEST( UHF, Li_631Gd ) {

  CQSCFTEST( "scf/serial/uhf/li_6-31Gd", "li_6-31Gd.bin.ref" );

};

// O2 6-31G(d) test
TEST( UHF, O2_631Gd ) {

  CQSCFTEST( "scf/serial/uhf/oxygen_6-31Gd", "oxygen_6-31Gd.bin.ref" );

};

// MnHe sto-3g test
TEST( UHF, MnHe_sto3G ) {

  CQSCFTEST( "scf/serial/uhf/MnHe_sto-3G", "MnHe_sto-3G.bin.ref" );

};

// RbHe sto-3g test
TEST( UHF, RbHe_sto3G ) {

  CQSCFTEST( "scf/serial/uhf/RbHe_sto-3G", "RbHe_sto-3G.bin.ref" );

};

#ifdef _CQ_DO_PARTESTS

// SMP Li 6-31G(d) test
TEST( UHF, PAR_Li_631Gd ) {

  CQSCFTEST( "scf/parallel/uhf/li_6-31Gd", "li_6-31Gd.bin.ref" );

};

// SMP O2 6-31G(d) test
TEST( UHF, PAR_O2_631Gd ) {

  CQSCFTEST( "scf/parallel/uhf/oxygen_6-31Gd", "oxygen_6-31Gd.bin.ref" );

};

#endif




