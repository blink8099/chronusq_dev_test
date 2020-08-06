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



// Water RHF/6-31G(d)/cc-pVDZ-rifit test
TEST( RI_RHF, water_631Gd_ccpvdzrifit ) {

  CQSCFTEST( "scf/serial/ri_rhf/water_6-31Gd_cc-pvdz-rifit",
             "water_6-31Gd_cc-pvdz-rifit.bin.ref" );

};

// Water RB3LYP/6-31G(d)/cc-pVDZ-rifit test
TEST( RI_RKS, water_631Gd_ccpvdzrifit_B3LYP ) {

  CQSCFTEST( "scf/serial/ri_rks/water_6-31Gd_cc-pvdz-rifit_b3lyp",
             "water_6-31Gd_cc-pvdz-rifit_b3lyp.bin.ref" );

};

// O2 ROHF/cc-pVTZ/cc-pVTZ-jkfit test
TEST( RI_ROHF, Oxygen_ccpvtzjkfit ) {

  CQSCFTEST( "scf/serial/ri_rohf/oxygen_cc-pvtz_cc-pvtz-jkfit",
             "oxygen_cc-pvtz_cc-pvtz-jkfit.bin.ref" );

};

// O2 UHF/def2-tzvpd/def2-tzvpd-rifit test
TEST( RI_UHF, Oxygen_def2tzvp_rifit ) {

  CQSCFTEST( "scf/serial/ri_uhf/oxygen_def2-tzvp_def2-tzvp-rifit",
             "oxygen_def2-tzvp_def2-tzvp-rifit.bin.ref" );

};

// O2 UPBE0/6-31++g(d)/aug-cc-pvdz-rifit test
TEST( RI_UKS, Oxygen_631ppGd_augccpvdzrifit_PBE0 ) {

  CQSCFTEST( "scf/serial/ri_uks/oxygen_631++Gd_aug-cc-pvdz-rifit_pbe0",
             "oxygen_631++Gd_aug-cc-pvdz-rifit_pbe0.bin.ref" );

};

// Cd Scalar X2C UHF/sapporo_dz_dkh3_2012_sp/x2c-jfit test
TEST( RI_X2CHF, cd_sap_dz_dkh3_2012_sp_x2c_jfit ) {

  CQSCFTEST( "scf/serial/ri_x2c/cd_sap-dz-dkh3-2012-sp_x2c-jfit",
             "cd_sap-dz-dkh3-2012-sp_x2c-jfit.bin.ref",
             1e-7 );

};

// Hg X2C UHF/sapporo_dz_dkh3_2012_sp/x2c-jfit test
TEST( RI_X2CHF, hg_sap_dz_dkh3_2012_sp_x2c_jfit ) {

  CQSCFTEST( "scf/serial/ri_x2c/hg_sap-dz-dkh3-2012-sp_x2c-jfit",
             "hg_sap-dz-dkh3-2012-sp_x2c-jfit.bin.ref",
             1e-7 );

};

// Hg X2C UHF/sapporo_dz_dkh3_2012_sp/x2c-jfit test
TEST( RI_X2CKS, hg_sap_dz_dkh3_2012_sp_x2c_jfit_b3lyp ) {

  CQSCFTEST( "scf/serial/ri_x2c/hg_sap-dz-dkh3-2012-sp_x2c-jfit_b3lyp",
             "hg_sap-dz-dkh3-2012-sp_x2c-jfit_b3lyp.bin.ref",
             1e-7 );

};

#ifdef _CQ_DO_PARTESTS

// Water RHF/6-31G(d)/cc-pVDZ-rifit test
TEST( RI_RHF, PAR_water_631Gd_ccpvdzrifit ) {

  CQSCFTEST( "scf/parallel/ri_rhf/water_6-31Gd_cc-pvdz-rifit",
             "water_6-31Gd_cc-pvdz-rifit.bin.ref" );

};

// Water RB3LYP/6-31G(d)/cc-pVDZ-rifit test
TEST( RI_RKS, PAR_water_631Gd_ccpvdzrifit_B3LYP ) {

  CQSCFTEST( "scf/parallel/ri_rks/water_6-31Gd_cc-pvdz-rifit_b3lyp",
             "water_6-31Gd_cc-pvdz-rifit_b3lyp.bin.ref" );

};

// O2 ROHF/cc-pVTZ/cc-pVTZ-jkfit test
TEST( RI_ROHF, PAR_Oxygen_ccpvtzjkfit ) {

  CQSCFTEST( "scf/parallel/ri_rohf/oxygen_cc-pvtz_cc-pvtz-jkfit",
             "oxygen_cc-pvtz_cc-pvtz-jkfit.bin.ref" );

};

// O2 UHF/def2-tzvpd/def2-tzvpd-rifit test
TEST( RI_UHF, PAR_Oxygen_def2tzvp_rifit ) {

  CQSCFTEST( "scf/parallel/ri_uhf/oxygen_def2-tzvp_def2-tzvp-rifit",
             "oxygen_def2-tzvp_def2-tzvp-rifit.bin.ref" );

};

// O2 UPBE0/6-31++g(d)/aug-cc-pvdz-rifit test
TEST( RI_UKS, PAR_Oxygen_631ppGd_augccpvdzrifit_PBE0 ) {

  CQSCFTEST( "scf/parallel/ri_uks/oxygen_631++Gd_aug-cc-pvdz-rifit_pbe0",
             "oxygen_631++Gd_aug-cc-pvdz-rifit_pbe0.bin.ref" );

};

// Cd Scalar X2C UHF/sapporo_dz_dkh3_2012_sp/x2c-jfit test
TEST( RI_X2CHF, PAR_cd_sap_dz_dkh3_2012_sp_x2c_jfit ) {

  CQSCFTEST( "scf/parallel/ri_x2c/cd_sap-dz-dkh3-2012-sp_x2c-jfit",
             "cd_sap-dz-dkh3-2012-sp_x2c-jfit.bin.ref",
             1e-7 );

};

// Hg X2C UHF/sapporo_dz_dkh3_2012_sp/x2c-jfit test
TEST( RI_X2CHF, PAR_hg_sap_dz_dkh3_2012_sp_x2c_jfit ) {

  CQSCFTEST( "scf/parallel/ri_x2c/hg_sap-dz-dkh3-2012-sp_x2c-jfit",
             "hg_sap-dz-dkh3-2012-sp_x2c-jfit.bin.ref",
             1e-7 );

};

// Hg X2C UHF/sapporo_dz_dkh3_2012_sp/x2c-jfit test
TEST( RI_X2CKS, PAR_hg_sap_dz_dkh3_2012_sp_x2c_jfit_b3lyp ) {

  CQSCFTEST( "scf/parallel/ri_x2c/hg_sap-dz-dkh3-2012-sp_x2c-jfit_b3lyp",
             "hg_sap-dz-dkh3-2012-sp_x2c-jfit_b3lyp.bin.ref",
             1e-7 );

};

#endif




