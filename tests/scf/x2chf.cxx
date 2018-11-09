/* 
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *  
 *  Copyright (C) 2014-2018 Li Research Group (University of Washington)
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

// Water 6-311+G(d,p) (Spherical) test
TEST( X2CHF, Water_6311pGdp_sph ) {

  CQSCFTEST( "scf/serial/x2c/water_6-311+Gdp_sph", 
    "water_6-311+Gdp_sph_x2c.bin.ref",1e-6 );
 
};

// Water 6-311+G(d,p) (Cartesian) test
TEST( X2CHF, Water_6311pGdp_cart ) {

  CQSCFTEST( "scf/serial/x2c/water_6-311+Gdp_cart", 
    "water_6-311+Gdp_cart_x2c.bin.ref",1e-6 );
 
};

// Hg SAPPORO DZP DKH_2012 SP
TEST( X2CHF, Hg_SAP_DZP_DKH3_2012_SP  ) {

  CQSCFTEST( "scf/serial/x2c/hg_sap_dz_dkh3_2012_sp", 
    "hg_sap_dz_dkh3_2012_sp.bin.ref",1e-6 );
 
};

// Zn SAPPORO DZP DKH_2012 SP
TEST( X2CHF, Zn_SAP_DZP_DKH3_2012_SP  ) {

  CQSCFTEST( "scf/serial/x2c/zn_sap_dz_dkh3_2012_sp", 
    "zn_sap_dz_dkh3_2012_sp.bin.ref",1e-6 );
 
};

// Cd SAPPORO DZP DKH_2012 SP
TEST( X2CHF, Cd_SAP_DZP_DKH3_2012_SP  ) {

  CQSCFTEST( "scf/serial/x2c/cd_sap_dz_dkh3_2012_sp", 
    "cd_sap_dz_dkh3_2012_sp.bin.ref",1e-6 );
 
};

#ifdef _CQ_DO_PARTESTS

// SMP Water 6-311+G(d,p) (Spherical) test
TEST( X2CHF, PAR_Water_6311pGdp_sph ) {

  CQSCFTEST( "scf/parallel/x2c/water_6-311+Gdp_sph", 
    "water_6-311+Gdp_sph_x2c.bin.ref",1e-6 );
 
};

/*
// SMP Hg SAPPORO DZP DKH_2012 SP
TEST( X2CHF, PAR_Hg_SAP_DZP_DKH3_2012_SP  ) {

  CQSCFTEST( scf/parallel/x2c/hg_sap_dz_dkh3_2012_sp, 
    hg_sap_dz_dkh3_2012_sp.bin.ref );
 
};

// SMP Zn SAPPORO DZP DKH_2012 SP
TEST( X2CHF, PAR_Zn_SAP_DZP_DKH3_2012_SP  ) {

  CQSCFTEST( scf/parallel/x2c/zn_sap_dz_dkh3_2012_sp, 
    zn_sap_dz_dkh3_2012_sp.bin.ref );
 
};

// SMP Cd SAPPORO DZP DKH_2012 SP
TEST( X2CHF, PAR_Cd_SAP_DZP_DKH3_2012_SP  ) {

  CQSCFTEST( scf/parallel/x2c/cd_sap_dz_dkh3_2012_sp, 
    cd_sap_dz_dkh3_2012_sp.bin.ref );
 
};
*/


#endif



