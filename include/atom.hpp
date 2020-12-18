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
#pragma once

#include <chronusq_sys.hpp>

namespace ChronusQ {

  /**
   *  \brief The Atom struct. Contains pertinant information for the
   *  nuclear structure as it relates to the construction of a Molecule
   *  object.
   */ 
  struct Atom {
    size_t atomicNumber; ///< Atomic Number (# of protons)
    double nucCharge;    ///< Nuclear Charge
    size_t massNumber;   ///< Mass Number
    double atomicMass;   ///< Atomic Mass (in a.u.)
    double slaterRadius; ///< Slater radius (in Bohr)

    std::array<double,3> coord; ///< X,Y,Z coordinates


    /**
     *  Coordinate array constructor (Default)
     *
     *  \param [in] AN   Atomic Number
     *  \param [in] NC   Nuclear Charge
     *  \param [in] MN   Mass Number
     *  \param [in] MASS Atomic Mass (A.U.)
     *  \param [in] RAD  Slater radium (Bohr)
     *  \param [in] XYZ  Cartesian atomic coordinates (X,Y,Z)
     */
    Atom(const size_t AN = 0, const double NC = 0., const size_t MN = 0,
      const double MASS = 0., const double RAD = 0.,
      std::array<double,3> XYZ = {0.,0.,0.}) :
      atomicNumber(AN), nucCharge(NC), massNumber(MN), atomicMass(MASS),
      slaterRadius(RAD), coord(std::move(XYZ)){ };

    /**
     *  X,Y,Z constructor
     *
     *  \param [in] AN   Atomic Number
     *  \param [in] NC   Nuclear Charge 
     *  \param [in] MN   Mass Number
     *  \param [in] MASS Atomic Mass (A.U.)
     *  \param [in] RAD  Slater radium (Bohr)
     *  \param [in] X    X cartesian coordinate of atomic coordinates
     *  \param [in] Y    Y cartesian coordinate of atomic coordinates
     *  \param [in] Z    Z cartesian coordinate of atomic coordinates
     */
    Atom(const size_t AN, const double NC, const size_t MN,
      const double MASS, const double RAD, double X, double Y, double Z) :
      Atom(AN,NC,MN,MASS,RAD,{X,Y,Z}){ };

    /**
     *  Symbol + coordinate array constructor
     *
     *  \param [in] symb Atomic sybmol
     *  \param [in] XYZ  Cartesian atomic coordinates (X,Y,Z)
     */ 
    Atom(std::string symb, std::array<double,3> XYZ);

    /**
     *  Symbol + coordinate array constructor
     *
     *  \param [in] symb Atomic sybmol
     *  \param [in] X    X cartesian coordinate of atomic coordinates
     *  \param [in] Y    Y cartesian coordinate of atomic coordinates
     *  \param [in] Z    Z cartesian coordinate of atomic coordinates
     */ 
    Atom(std::string symb, double X = 0., double Y = 0., double Z = 0.) :
      Atom(symb,{X,Y,Z}){ };



    /**
     *  Copy constructor
     *
     *  Copies one Atom object to another
     */ 
    Atom(const Atom &) = default;

    /**
     *  Move constructor
     *
     *  Moves one Atom object to another
     */ 
    Atom(Atom &&)      = default;


    /**
     *  Copy assignment operator
     *
     *  Assigns one Atom object to another through a copy
     */ 
    Atom& operator=(const Atom &) = default;

    /**
     *  Move assignment operator
     *
     *  Assigns one Atom object to another through a move (rvalue reference)
     */ 
    Atom& operator=(Atom &&)      = default;

  }; // Atom struct


  // Map atomic symbol to predefined Atom objects. Support
  // for non-standard isotopes forthcoming
  //
  // Last values: Atomic Radius (in Ang). According Slater, J. Chem. Phys. 41, pg 3199 (1964).
  // Note for Hydrogen is already multiplied by 2 (to prevent the scaling later on) 
  // all noble gasses values are computed values from Reinhardt J. Chem. Phys. 47, pg 1300 (1967)
  // after Am all set to 2.01 (Ang)
  // FIXME: Need to specify isotope label for non-default isotopes
  static std::map<std::string,Atom> atomicReference(
    {
      { "H-1"      , {   1  ,   1.  ,   1  ,   1.0078250321  ,   1.058  } },
      { "H-2"      , {   1  ,   1.  ,   2  ,   2.0141017780  ,   1.058  } },
      { "H-3"      , {   1  ,   1.  ,   3  ,   3.0160492675  ,   1.058  } },
      { "HE-3"     , {   2  ,   2.  ,   3  ,   3.0160293097  ,   0.310  } },
      { "HE-4"     , {   2  ,   2.  ,   4  ,   4.0026032497  ,   0.310  } },
      { "LI-6"     , {   3  ,   3.  ,   6  ,   6.0151223     ,   1.450  } },
      { "LI-7"     , {   3  ,   3.  ,   7  ,   7.0160040     ,   1.450  } },
      { "BE-9"     , {   4  ,   4.  ,   9  ,   9.0121821     ,   1.050  } },
      { "B-10"     , {   5  ,   5.  ,  10  ,  10.012937      ,   0.85   } },
      { "B-11"     , {   5  ,   5.  ,  11  ,  11.0093055     ,   0.85   } },
      { "C-12"     , {   6  ,   6.  ,  12  ,  12.0000000     ,   0.70   } },
      { "C-13"     , {   6  ,   6.  ,  13  ,  13.0033548378  ,   0.70   } },
      { "C-14"     , {   6  ,   6.  ,  14  ,  14.003241988   ,   0.70   } },
      { "N-14"     , {   7  ,   7.  ,  14  ,  14.0030740052  ,   0.65   } },
      { "N-15"     , {   7  ,   7.  ,  15  ,  15.0001088984  ,   0.65   } },
      { "O-16"     , {   8  ,   8.  ,  16  ,  15.9949146221  ,   0.60   } },
      { "O-17"     , {   8  ,   8.  ,  17  ,  16.99913150    ,   0.60   } },
      { "O-18"     , {   8  ,   8.  ,  18  ,  17.9991604     ,   0.60   } },
      { "F-19"     , {   9  ,   9.  ,  19  ,  18.9984032     ,   0.50   } },
      { "NE-20"    , {  10  ,  10.  ,  20  ,  19.9924401759  ,   0.38   } },
      { "NE-21"    , {  10  ,  10.  ,  21  ,  20.99384674    ,   0.38   } },
      { "NE-22"    , {  10  ,  10.  ,  22  ,  21.99138551    ,   0.38   } },
      { "NA-23"    , {  11  ,  11.  ,  23  ,  22.98976967    ,   1.80   } },
      { "MG-24"    , {  12  ,  12.  ,  24  ,  23.98504190    ,   1.50   } },
      { "MG-25"    , {  12  ,  12.  ,  25  ,  24.98583702    ,   1.50   } },
      { "MG-26"    , {  12  ,  12.  ,  26  ,  25.98259304    ,   1.50   } },
      { "AL-27"    , {  13  ,  13.  ,  27  ,  26.98153844    ,   1.25   } },
      { "SI-28"    , {  14  ,  14.  ,  28  ,  27.9769265327  ,   1.10   } },
      { "SI-29"    , {  14  ,  14.  ,  29  ,  28.97649472    ,   1.10   } },
      { "SI-30"    , {  14  ,  14.  ,  30  ,  29.97377022    ,   1.10   } },
      { "P-31"     , {  15  ,  15.  ,  31  ,  30.97376151    ,   1.00   } },
      { "S-32"     , {  16  ,  16.  ,  32  ,  31.97207069    ,   1.00   } },
      { "S-33"     , {  16  ,  16.  ,  33  ,  32.97145850    ,   1.00   } },
      { "S-34"     , {  16  ,  16.  ,  34  ,  33.96786683    ,   1.00   } },
      { "S-36"     , {  16  ,  16.  ,  36  ,  35.96708088    ,   1.00   } },
      { "CL-35"    , {  17  ,  17.  ,  35  ,  34.96885271    ,   1.00   } },
      { "CL-37"    , {  17  ,  17.  ,  37  ,  36.96590260    ,   1.00   } },
      { "AR-36"    , {  18  ,  18.  ,  36  ,  35.96754628    ,   0.71   } },
      { "AR-38"    , {  18  ,  18.  ,  38  ,  37.9627322     ,   0.71   } },
      { "AR-40"    , {  18  ,  18.  ,  40  ,  39.962383123   ,   0.71   } },
      { "K-39"     , {  19  ,  19.  ,  39  ,  38.9637069     ,   2.20   } },
      { "K-40"     , {  19  ,  19.  ,  40  ,  39.96399867    ,   2.20   } },
      { "K-41"     , {  19  ,  19.  ,  41  ,  40.96182597    ,   2.20   } },
      { "CA-40"    , {  20  ,  20.  ,  40  ,  39.9625912     ,   1.80   } },
      { "CA-42"    , {  20  ,  20.  ,  42  ,  41.9586183     ,   1.80   } },
      { "CA-43"    , {  20  ,  20.  ,  43  ,  42.9587668     ,   1.80   } },
      { "CA-44"    , {  20  ,  20.  ,  44  ,  43.9554811     ,   1.80   } },
      { "CA-46"    , {  20  ,  20.  ,  46  ,  45.9536928     ,   1.80   } },
      { "CA-48"    , {  20  ,  20.  ,  48  ,  47.952534      ,   1.80   } },
      { "SC-45"    , {  21  ,  21.  ,  45  ,  44.9559102     ,   1.60   } },
      { "TI-46"    , {  22  ,  22.  ,  46  ,  45.9526295     ,   1.40   } },
      { "TI-47"    , {  22  ,  22.  ,  47  ,  46.9517638     ,   1.40   } },
      { "TI-48"    , {  22  ,  22.  ,  48  ,  47.9479471     ,   1.40   } },
      { "TI-49"    , {  22  ,  22.  ,  49  ,  48.9478708     ,   1.40   } },
      { "TI-50"    , {  22  ,  22.  ,  50  ,  49.9447921     ,   1.40   } },
      { "V-50"     , {  23  ,  23.  ,  50  ,  49.9471628     ,   1.35   } },
      { "V-51"     , {  23  ,  23.  ,  51  ,  50.9439637     ,   1.35   } },
      { "CR-50"    , {  24  ,  24.  ,  50  ,  49.9460496     ,   1.40   } },
      { "CR-52"    , {  24  ,  24.  ,  52  ,  51.9405119     ,   1.40   } },
      { "CR-53"    , {  24  ,  24.  ,  53  ,  52.9406538     ,   1.40   } },
      { "CR-54"    , {  24  ,  24.  ,  54  ,  53.9388849     ,   1.40   } },
      { "MN-55"    , {  25  ,  25.  ,  55  ,  54.9380496     ,   1.40   } },
      { "FE-54"    , {  26  ,  26.  ,  54  ,  53.9396148     ,   1.40   } },
      { "FE-56"    , {  26  ,  26.  ,  56  ,  55.9349421     ,   1.40   } },
      { "FE-57"    , {  26  ,  26.  ,  57  ,  56.9353987     ,   1.40   } },
      { "FE-58"    , {  26  ,  26.  ,  58  ,  57.9332805     ,   1.40   } },
      { "CO-59"    , {  27  ,  27.  ,  59  ,  58.9332002     ,   1.35   } },
      { "NI-58"    , {  28  ,  28.  ,  58  ,  57.9353479     ,   1.35   } },
      { "NI-60"    , {  28  ,  28.  ,  60  ,  59.9307906     ,   1.35   } },
      { "NI-61"    , {  28  ,  28.  ,  61  ,  60.9310604     ,   1.35   } },
      { "NI-62"    , {  28  ,  28.  ,  62  ,  61.9283488     ,   1.35   } },
      { "NI-64"    , {  28  ,  28.  ,  64  ,  63.9279696     ,   1.35   } },
      { "CU-63"    , {  29  ,  29.  ,  63  ,  62.9296011     ,   1.35   } },
      { "CU-65"    , {  29  ,  29.  ,  65  ,  64.9277937     ,   1.35   } },
      { "ZN-64"    , {  30  ,  30.  ,  64  ,  63.9291466     ,   1.35   } },
      { "ZN-66"    , {  30  ,  30.  ,  66  ,  65.9260368     ,   1.35   } },
      { "ZN-67"    , {  30  ,  30.  ,  67  ,  66.9271309     ,   1.35   } },
      { "ZN-68"    , {  30  ,  30.  ,  68  ,  67.9248476     ,   1.35   } },
      { "ZN-70"    , {  30  ,  30.  ,  70  ,  69.925325      ,   1.35   } },
      { "GA-69"    , {  31  ,  31.  ,  69  ,  68.925581      ,   1.30   } },
      { "GA-71"    , {  31  ,  31.  ,  71  ,  70.9247050     ,   1.30   } },
      { "GE-70"    , {  32  ,  32.  ,  70  ,  69.9242504     ,   1.25   } },
      { "GE-72"    , {  32  ,  32.  ,  72  ,  71.9220762     ,   1.25   } },
      { "GE-73"    , {  32  ,  32.  ,  73  ,  72.9234594     ,   1.25   } },
      { "GE-74"    , {  32  ,  32.  ,  74  ,  73.9211782     ,   1.25   } },
      { "GE-76"    , {  32  ,  32.  ,  76  ,  75.9214027     ,   1.25   } },
      { "AS-75"    , {  33  ,  33.  ,  75  ,  74.9215964     ,   1.15   } },
      { "SE-74"    , {  34  ,  34.  ,  74  ,  73.9224766     ,   1.15   } },
      { "SE-76"    , {  34  ,  34.  ,  76  ,  75.9192141     ,   1.15   } },
      { "SE-77"    , {  34  ,  34.  ,  77  ,  76.9199146     ,   1.15   } },
      { "SE-78"    , {  34  ,  34.  ,  78  ,  77.9173095     ,   1.15   } },
      { "SE-80"    , {  34  ,  34.  ,  80  ,  79.9165218     ,   1.15   } },
      { "SE-82"    , {  34  ,  34.  ,  82  ,  81.9167000     ,   1.15   } },
      { "BR-79"    , {  35  ,  35.  ,  79  ,  78.9183376     ,   1.15   } },
      { "BR-81"    , {  35  ,  35.  ,  81  ,  80.916291      ,   1.15   } },
      { "KR-78"    , {  36  ,  36.  ,  78  ,  77.920386      ,   0.88   } },
      { "KR-80"    , {  36  ,  36.  ,  80  ,  79.916378      ,   0.88   } },
      { "KR-82"    , {  36  ,  36.  ,  82  ,  81.9134846     ,   0.88   } },
      { "KR-83"    , {  36  ,  36.  ,  83  ,  82.914136      ,   0.88   } },
      { "KR-84"    , {  36  ,  36.  ,  84  ,  83.911507      ,   0.88   } },
      { "KR-86"    , {  36  ,  36.  ,  86  ,  85.9106103     ,   0.88   } },
      { "RB-85"    , {  37  ,  37.  ,  85  ,  84.9117893     ,   2.35   } },
      { "RB-87"    , {  37  ,  37.  ,  87  ,  86.9091835     ,   2.35   } },
      { "SR-84"    , {  38  ,  38.  ,  84  ,  83.913425      ,   2.00   } },
      { "SR-86"    , {  38  ,  38.  ,  86  ,  85.9092624     ,   2.00   } },
      { "SR-87"    , {  38  ,  38.  ,  87  ,  86.9088793     ,   2.00   } },
      { "SR-88"    , {  38  ,  38.  ,  88  ,  87.9056143     ,   2.00   } },
      { "Y-89"     , {  39  ,  39.  ,  89  ,  88.9058479     ,   1.80   } },
      { "ZR-90"    , {  40  ,  40.  ,  90  ,  89.9047037     ,   1.55   } },
      { "ZR-91"    , {  40  ,  40.  ,  91  ,  90.9056450     ,   1.55   } },
      { "ZR-92"    , {  40  ,  40.  ,  92  ,  91.9050401     ,   1.55   } },
      { "ZR-94"    , {  40  ,  40.  ,  94  ,  93.9063158     ,   1.55   } },
      { "ZR-96"    , {  40  ,  40.  ,  96  ,  95.908276      ,   1.55   } },
      { "NB-93"    , {  41  ,  41.  ,  93  ,  92.9063775     ,   1.45   } },
      { "MO-92"    , {  42  ,  42.  ,  92  ,  91.906810      ,   1.45   } },
      { "MO-94"    , {  42  ,  42.  ,  94  ,  93.9050876     ,   1.45   } },
      { "MO-95"    , {  42  ,  42.  ,  95  ,  94.9058415     ,   1.45   } },
      { "MO-96"    , {  42  ,  42.  ,  96  ,  95.9046789     ,   1.45   } },
      { "MO-97"    , {  42  ,  42.  ,  97  ,  96.9060210     ,   1.45   } },
      { "MO-98"    , {  42  ,  42.  ,  98  ,  97.9054078     ,   1.45   } },
      { "MO-100"   , {  42  ,  42.  , 100  ,  99.907477      ,   1.45   } },
      { "TC-97"    , {  43  ,  43.  ,  97  ,  96.906365      ,   1.35   } },
      { "TC-98"    , {  43  ,  43.  ,  98  ,  97.907216      ,   1.35   } },
      { "TC-99"    , {  43  ,  43.  ,  99  ,  98.9062546     ,   1.35   } },
      { "RU-96"    , {  44  ,  44.  ,  96  ,  95.907598      ,   1.30   } },
      { "RU-98"    , {  44  ,  44.  ,  98  ,  97.905287      ,   1.30   } },
      { "RU-99"    , {  44  ,  44.  ,  99  ,  98.9059393     ,   1.30   } },
      { "RU-100"   , {  44  ,  44.  , 100  ,  99.9042197     ,   1.30   } },
      { "RU-101"   , {  44  ,  44.  , 101  , 100.9055822     ,   1.30   } },
      { "RU-102"   , {  44  ,  44.  , 102  , 101.9043495     ,   1.30   } },
      { "RU-104"   , {  44  ,  44.  , 104  , 103.905430      ,   1.30   } },
      { "RH-103"   , {  45  ,  45.  , 103  , 102.905504      ,   1.35   } },
      { "PD-102"   , {  46  ,  46.  , 102  , 101.905608      ,   1.40   } },
      { "PD-104"   , {  46  ,  46.  , 104  , 103.904035      ,   1.40   } },
      { "PD-105"   , {  46  ,  46.  , 105  , 104.905084      ,   1.40   } },
      { "PD-106"   , {  46  ,  46.  , 106  , 105.903483      ,   1.40   } },
      { "PD-108"   , {  46  ,  46.  , 108  , 107.903894      ,   1.40   } },
      { "PD-110"   , {  46  ,  46.  , 110  , 109.905152      ,   1.40   } },
      { "AG-107"   , {  47  ,  47.  , 107  , 106.905093      ,   1.60   } },
      { "AG-109"   , {  47  ,  47.  , 109  , 108.904756      ,   1.60   } },
      { "CD-106"   , {  48  ,  48.  , 106  , 105.906458      ,   1.55   } },
      { "CD-108"   , {  48  ,  48.  , 108  , 107.904183      ,   1.55   } },
      { "CD-110"   , {  48  ,  48.  , 110  , 109.903006      ,   1.55   } },
      { "CD-111"   , {  48  ,  48.  , 111  , 110.904182      ,   1.55   } },
      { "CD-112"   , {  48  ,  48.  , 112  , 111.9027572     ,   1.55   } },
      { "CD-113"   , {  48  ,  48.  , 113  , 112.9044009     ,   1.55   } },
      { "CD-114"   , {  48  ,  48.  , 114  , 113.9033581     ,   1.55   } },
      { "CD-116"   , {  48  ,  48.  , 116  , 115.904755      ,   1.55   } },
      { "IN-113"   , {  49  ,  49.  , 113  , 112.904061      ,   1.55   } },
      { "IN-115"   , {  49  ,  49.  , 115  , 114.903878      ,   1.55   } },
      { "SN-112"   , {  50  ,  50.  , 112  , 111.904821      ,   1.45   } },
      { "SN-114"   , {  50  ,  50.  , 114  , 113.902782      ,   1.45   } },
      { "SN-115"   , {  50  ,  50.  , 115  , 114.903346      ,   1.45   } },
      { "SN-116"   , {  50  ,  50.  , 116  , 115.901744      ,   1.45   } },
      { "SN-117"   , {  50  ,  50.  , 117  , 116.902954      ,   1.45   } },
      { "SN-118"   , {  50  ,  50.  , 118  , 117.901606      ,   1.45   } },
      { "SN-119"   , {  50  ,  50.  , 119  , 118.903309      ,   1.45   } },
      { "SN-120"   , {  50  ,  50.  , 120  , 119.9021966     ,   1.45   } },
      { "SN-122"   , {  50  ,  50.  , 122  , 121.9034401     ,   1.45   } },
      { "SN-124"   , {  50  ,  50.  , 124  , 123.9052746     ,   1.45   } },
      { "SB-121"   , {  51  ,  51.  , 121  , 120.9038180     ,   1.45   } },
      { "SB-123"   , {  51  ,  51.  , 123  , 122.9042157     ,   1.45   } },
      { "TE-120"   , {  52  ,  52.  , 120  , 119.904020      ,   1.40   } },
      { "TE-122"   , {  52  ,  52.  , 122  , 121.9030471     ,   1.40   } },
      { "TE-123"   , {  52  ,  52.  , 123  , 122.9042730     ,   1.40   } },
      { "TE-124"   , {  52  ,  52.  , 124  , 123.9028195     ,   1.40   } },
      { "TE-125"   , {  52  ,  52.  , 125  , 124.9044247     ,   1.40   } },
      { "TE-126"   , {  52  ,  52.  , 126  , 125.9033055     ,   1.40   } },
      { "TE-128"   , {  52  ,  52.  , 128  , 127.9044614     ,   1.40   } },
      { "TE-130"   , {  52  ,  52.  , 130  , 129.9062228     ,   1.40   } },
      { "I-127"    , {  53  ,  53.  , 127  , 126.904468      ,   1.40   } },
      { "XE-124"   , {  54  ,  54.  , 124  , 123.9058958     ,   1.08   } },
      { "XE-126"   , {  54  ,  54.  , 126  , 125.904269      ,   1.08   } },
      { "XE-128"   , {  54  ,  54.  , 128  , 127.9035304     ,   1.08   } },
      { "XE-129"   , {  54  ,  54.  , 129  , 128.9047795     ,   1.08   } },
      { "XE-130"   , {  54  ,  54.  , 130  , 129.9035079     ,   1.08   } },
      { "XE-131"   , {  54  ,  54.  , 131  , 130.9050819     ,   1.08   } },
      { "XE-132"   , {  54  ,  54.  , 132  , 131.9041545     ,   1.08   } },
      { "XE-134"   , {  54  ,  54.  , 134  , 133.9053945     ,   1.08   } },
      { "XE-136"   , {  54  ,  54.  , 136  , 135.907220      ,   1.08   } },
      { "CS-133"   , {  55  ,  55.  , 133  , 132.905447      ,   2.60   } },
      { "BA-130"   , {  56  ,  56.  , 130  , 129.906310      ,   2.15   } },
      { "BA-132"   , {  56  ,  56.  , 132  , 131.905056      ,   2.15   } },
      { "BA-134"   , {  56  ,  56.  , 134  , 133.904503      ,   2.15   } },
      { "BA-135"   , {  56  ,  56.  , 135  , 134.905683      ,   2.15   } },
      { "BA-136"   , {  56  ,  56.  , 136  , 135.904570      ,   2.15   } },
      { "BA-137"   , {  56  ,  56.  , 137  , 136.905821      ,   2.15   } },
      { "BA-138"   , {  56  ,  56.  , 138  , 137.905241      ,   2.15   } },
      { "LA-138"   , {  57  ,  57.  , 138  , 137.907107      ,   1.95   } },
      { "LA-139"   , {  57  ,  57.  , 139  , 138.906348      ,   1.95   } },
      { "CE-136"   , {  58  ,  58.  , 136  , 135.907140      ,   1.85   } },
      { "CE-138"   , {  58  ,  58.  , 138  , 137.905986      ,   1.85   } },
      { "CE-140"   , {  58  ,  58.  , 140  , 139.905434      ,   1.85   } },
      { "CE-142"   , {  58  ,  58.  , 142  , 141.909240      ,   1.85   } },
      { "PR-141"   , {  59  ,  59.  , 141  , 140.907648      ,   1.85   } },
      { "ND-142"   , {  60  ,  60.  , 142  , 141.907719      ,   1.85   } },
      { "ND-143"   , {  60  ,  60.  , 143  , 142.909810      ,   1.85   } },
      { "ND-144"   , {  60  ,  60.  , 144  , 143.910083      ,   1.85   } },
      { "ND-145"   , {  60  ,  60.  , 145  , 144.912569      ,   1.85   } },
      { "ND-146"   , {  60  ,  60.  , 146  , 145.913112      ,   1.85   } },
      { "ND-148"   , {  60  ,  60.  , 148  , 147.916889      ,   1.85   } },
      { "ND-150"   , {  60  ,  60.  , 150  , 149.920887      ,   1.85   } },
      { "PM-145"   , {  61  ,  61.  , 145  , 144.912744      ,   1.85   } },
      { "PM-147"   , {  61  ,  61.  , 147  , 146.915134      ,   1.85   } },
      { "SM-144"   , {  62  ,  62.  , 144  , 143.911995      ,   1.85   } },
      { "SM-147"   , {  62  ,  62.  , 147  , 146.914893      ,   1.85   } },
      { "SM-148"   , {  62  ,  62.  , 148  , 147.914818      ,   1.85   } },
      { "SM-149"   , {  62  ,  62.  , 149  , 148.917180      ,   1.85   } },
      { "SM-150"   , {  62  ,  62.  , 150  , 149.917271      ,   1.85   } },
      { "SM-152"   , {  62  ,  62.  , 152  , 151.919728      ,   1.85   } },
      { "SM-154"   , {  62  ,  62.  , 154  , 153.922205      ,   1.85   } },
      { "EU-151"   , {  63  ,  63.  , 151  , 150.919846      ,   1.85   } },
      { "EU-153"   , {  63  ,  63.  , 153  , 152.921226      ,   1.85   } },
      { "GD-152"   , {  64  ,  64.  , 152  , 151.919788      ,   1.80   } },
      { "GD-154"   , {  64  ,  64.  , 154  , 153.920862      ,   1.80   } },
      { "GD-155"   , {  64  ,  64.  , 155  , 154.922619      ,   1.80   } },
      { "GD-156"   , {  64  ,  64.  , 156  , 155.922120      ,   1.80   } },
      { "GD-157"   , {  64  ,  64.  , 157  , 156.923957      ,   1.80   } },
      { "GD-158"   , {  64  ,  64.  , 158  , 157.924101      ,   1.80   } },
      { "GD-160"   , {  64  ,  64.  , 160  , 159.927051      ,   1.80   } },
      { "TB-159"   , {  65  ,  65.  , 159  , 158.925343      ,   1.75   } },
      { "DY-156"   , {  66  ,  66.  , 156  , 155.924278      ,   1.75   } },
      { "DY-158"   , {  66  ,  66.  , 158  , 157.924405      ,   1.75   } },
      { "DY-160"   , {  66  ,  66.  , 160  , 159.925194      ,   1.75   } },
      { "DY-161"   , {  66  ,  66.  , 161  , 160.926930      ,   1.75   } },
      { "DY-162"   , {  66  ,  66.  , 162  , 161.926795      ,   1.75   } },
      { "DY-163"   , {  66  ,  66.  , 163  , 162.928728      ,   1.75   } },
      { "DY-164"   , {  66  ,  66.  , 164  , 163.929171      ,   1.75   } },
      { "HO-165"   , {  67  ,  67.  , 165  , 164.930319      ,   1.75   } },
      { "ER-162"   , {  68  ,  68.  , 162  , 161.928775      ,   1.75   } },
      { "ER-164"   , {  68  ,  68.  , 164  , 163.929197      ,   1.75   } },
      { "ER-166"   , {  68  ,  68.  , 166  , 165.930290      ,   1.75   } },
      { "ER-167"   , {  68  ,  68.  , 167  , 166.932045      ,   1.75   } },
      { "ER-168"   , {  68  ,  68.  , 168  , 167.932368      ,   1.75   } },
      { "ER-170"   , {  68  ,  68.  , 170  , 169.935460      ,   1.75   } },
      { "TM-169"   , {  69  ,  69.  , 169  , 168.934211      ,   1.75   } },
      { "YB-168"   , {  70  ,  70.  , 168  , 167.933894      ,   1.75   } },
      { "YB-170"   , {  70  ,  70.  , 170  , 169.934759      ,   1.75   } },
      { "YB-171"   , {  70  ,  70.  , 171  , 170.936322      ,   1.75   } },
      { "YB-172"   , {  70  ,  70.  , 172  , 171.9363777     ,   1.75   } },
      { "YB-173"   , {  70  ,  70.  , 173  , 172.9382068     ,   1.75   } },
      { "YB-174"   , {  70  ,  70.  , 174  , 173.9388581     ,   1.75   } },
      { "YB-176"   , {  70  ,  70.  , 176  , 175.942568      ,   1.75   } },
      { "LU-175"   , {  71  ,  71.  , 175  , 174.9407679     ,   1.75   } },
      { "LU-176"   , {  71  ,  71.  , 176  , 175.9426824     ,   1.75   } },
      { "HF-174"   , {  72  ,  72.  , 174  , 173.940040      ,   1.55   } },
      { "HF-176"   , {  72  ,  72.  , 176  , 175.9414018     ,   1.55   } },
      { "HF-177"   , {  72  ,  72.  , 177  , 176.9432200     ,   1.55   } },
      { "HF-178"   , {  72  ,  72.  , 178  , 177.9436977     ,   1.55   } },
      { "HF-179"   , {  72  ,  72.  , 179  , 178.9458151     ,   1.55   } },
      { "HF-180"   , {  72  ,  72.  , 180  , 179.9465488     ,   1.55   } },
      { "TA-180"   , {  73  ,  73.  , 180  , 179.947466      ,   1.45   } },
      { "TA-181"   , {  73  ,  73.  , 181  , 180.947996      ,   1.45   } },
      { "W-180"    , {  74  ,  74.  , 180  , 179.946706      ,   1.35   } },
      { "W-182"    , {  74  ,  74.  , 182  , 181.948206      ,   1.35   } },
      { "W-183"    , {  74  ,  74.  , 183  , 182.9502245     ,   1.35   } },
      { "W-184"    , {  74  ,  74.  , 184  , 183.9509326     ,   1.35   } },
      { "W-186"    , {  74  ,  74.  , 186  , 185.954362      ,   1.35   } },
      { "RE-185"   , {  75  ,  75.  , 185  , 184.9529557     ,   1.35   } },
      { "RE-187"   , {  75  ,  75.  , 187  , 186.9557508     ,   1.35   } },
      { "OS-184"   , {  76  ,  76.  , 184  , 183.952491      ,   1.30   } },
      { "OS-186"   , {  76  ,  76.  , 186  , 185.953838      ,   1.30   } },
      { "OS-187"   , {  76  ,  76.  , 187  , 186.9557479     ,   1.30   } },
      { "OS-188"   , {  76  ,  76.  , 188  , 187.9558360     ,   1.30   } },
      { "OS-189"   , {  76  ,  76.  , 189  , 188.9581449     ,   1.30   } },
      { "OS-190"   , {  76  ,  76.  , 190  , 189.958445      ,   1.30   } },
      { "OS-192"   , {  76  ,  76.  , 192  , 191.961479      ,   1.30   } },
      { "IR-191"   , {  77  ,  77.  , 191  , 190.960591      ,   1.35   } },
      { "IR-193"   , {  77  ,  77.  , 193  , 192.962924      ,   1.35   } },
      { "PT-190"   , {  78  ,  78.  , 190  , 189.959930      ,   1.35   } },
      { "PT-192"   , {  78  ,  78.  , 192  , 191.961035      ,   1.35   } },
      { "PT-194"   , {  78  ,  78.  , 194  , 193.962664      ,   1.35   } },
      { "PT-195"   , {  78  ,  78.  , 195  , 194.964774      ,   1.35   } },
      { "PT-196"   , {  78  ,  78.  , 196  , 195.964935      ,   1.35   } },
      { "PT-198"   , {  78  ,  78.  , 198  , 197.967876      ,   1.35   } },
      { "AU-197"   , {  79  ,  79.  , 197  , 196.966552      ,   1.35   } },
      { "HG-196"   , {  80  ,  80.  , 196  , 195.965815      ,   1.50   } },
      { "HG-198"   , {  80  ,  80.  , 198  , 197.966752      ,   1.50   } },
      { "HG-199"   , {  80  ,  80.  , 199  , 198.968262      ,   1.50   } },
      { "HG-200"   , {  80  ,  80.  , 200  , 199.968309      ,   1.50   } },
      { "HG-201"   , {  80  ,  80.  , 201  , 200.970285      ,   1.50   } },
      { "HG-202"   , {  80  ,  80.  , 202  , 201.970626      ,   1.50   } },
      { "HG-204"   , {  80  ,  80.  , 204  , 203.973476      ,   1.50   } },
      { "TL-203"   , {  81  ,  81.  , 203  , 202.972329      ,   1.90   } },
      { "TL-205"   , {  81  ,  81.  , 205  , 204.974412      ,   1.90   } },
      { "PB-204"   , {  82  ,  82.  , 204  , 203.973029      ,   1.90   } },
      { "PB-206"   , {  82  ,  82.  , 206  , 205.974449      ,   1.90   } },
      { "PB-207"   , {  82  ,  82.  , 207  , 206.975881      ,   1.90   } },
      { "PB-208"   , {  82  ,  82.  , 208  , 207.976636      ,   1.90   } },
      { "BI-209"   , {  83  ,  83.  , 209  , 208.980383      ,   1.60   } },
      { "PO-209"   , {  84  ,  84.  , 209  , 208.982416      ,   1.90   } },
      { "PO-210"   , {  84  ,  84.  , 210  , 209.982857      ,   1.90   } },
      { "AT-210"   , {  85  ,  85.  , 210  , 209.987131      ,   1.90   } },
      { "AT-211"   , {  85  ,  85.  , 211  , 210.987481      ,   1.90   } },
      { "RN-211"   , {  86  ,  86.  , 211  , 210.990585      ,   2.01   } },
      { "RN-220"   , {  86  ,  86.  , 220  , 220.0113841     ,   2.01   } },
      { "RN-222"   , {  86  ,  86.  , 222  , 222.0175705     ,   2.01   } },
      { "FR-223"   , {  87  ,  87.  , 223  , 223.0197307     ,   1.90   } },
      { "RA-223"   , {  88  ,  88.  , 223  , 223.018497      ,   1.90   } },
      { "RA-224"   , {  88  ,  88.  , 224  , 224.0202020     ,   1.90   } },
      { "RA-226"   , {  88  ,  88.  , 226  , 226.0254026     ,   2.15   } },
      { "RA-228"   , {  88  ,  88.  , 228  , 228.0310641     ,   2.15   } },
      { "AC-227"   , {  89  ,  89.  , 227  , 227.0277470     ,   1.95   } },
      { "TH-230"   , {  90  ,  90.  , 230  , 230.0331266     ,   1.80   } },
      { "TH-232"   , {  90  ,  90.  , 232  , 232.0380504     ,   1.80   } },
      { "PA-231"   , {  91  ,  91.  , 231  , 231.0358789     ,   1.80   } },
      { "U-233"    , {  92  ,  92.  , 233  , 233.039628      ,   1.75   } },
      { "U-234"    , {  92  ,  92.  , 234  , 234.0409456     ,   1.75   } },
      { "U-235"    , {  92  ,  92.  , 235  , 235.0439231     ,   1.75   } },
      { "U-236"    , {  92  ,  92.  , 236  , 236.0455619     ,   1.75   } },
      { "U-238"    , {  92  ,  92.  , 238  , 238.0507826     ,   1.75   } },
      { "NP-237"   , {  93  ,  93.  , 237  , 237.0481673     ,   1.75   } },
      { "NP-239"   , {  93  ,  93.  , 239  , 239.0529314     ,   1.75   } },
      { "PU-238"   , {  94  ,  94.  , 238  , 238.0495534     ,   1.75   } },
      { "PU-239"   , {  94  ,  94.  , 239  , 239.0521565     ,   1.75   } },
      { "PU-240"   , {  94  ,  94.  , 240  , 240.0538075     ,   1.75   } },
      { "PU-241"   , {  94  ,  94.  , 241  , 241.0568453     ,   1.75   } },
      { "PU-242"   , {  94  ,  94.  , 242  , 242.0587368     ,   1.75   } },
      { "PU-244"   , {  94  ,  94.  , 244  , 244.064198      ,   1.75   } },
      { "AM-241"   , {  95  ,  95.  , 241  , 241.0568229     ,   1.75   } },
      { "AM-243"   , {  95  ,  95.  , 243  , 243.0613727     ,   1.75   } },
      { "CM-243"   , {  96  ,  96.  , 243  , 243.0613822     ,   2.01   } },
      { "CM-244"   , {  96  ,  96.  , 244  , 244.0627463     ,   2.01   } },
      { "CM-245"   , {  96  ,  96.  , 245  , 245.0654856     ,   2.01   } },
      { "CM-246"   , {  96  ,  96.  , 246  , 246.0672176     ,   2.01   } },
      { "CM-247"   , {  96  ,  96.  , 247  , 247.070347      ,   2.01   } },
      { "CM-248"   , {  96  ,  96.  , 248  , 248.072342      ,   2.01   } },
      { "BK-247"   , {  97  ,  97.  , 247  , 247.070299      ,   2.01   } },
      { "BK-249"   , {  97  ,  97.  , 249  , 249.074980      ,   2.01   } },
      { "CF-249"   , {  98  ,  98.  , 249  , 249.074847      ,   2.01   } },
      { "CF-250"   , {  98  ,  98.  , 250  , 250.0764000     ,   2.01   } },
      { "CF-251"   , {  98  ,  98.  , 251  , 251.079580      ,   2.01   } },
      { "CF-252"   , {  98  ,  98.  , 252  , 252.081620      ,   2.01   } },
      { "ES-252"   , {  99  ,  99.  , 252  , 252.082970      ,   2.01   } },
      { "FM-257"   , { 100  , 100.  , 257  , 257.095099      ,   2.01   } },
      { "MD-256"   , { 101  , 101.  , 256  , 256.094050      ,   2.01   } },
      { "MD-258"   , { 101  , 101.  , 258  , 258.098425      ,   2.01   } },
      { "NO-259"   , { 102  , 102.  , 259  , 259.10102       ,   2.01   } },
      { "LR-262"   , { 103  , 103.  , 262  , 262.10969       ,   2.01   } },
      { "RF-261"   , { 104  , 104.  , 261  , 261.10875       ,   2.01   } },
      { "DB-262"   , { 105  , 105.  , 262  , 262.11415       ,   2.01   } },
      { "SG-266"   , { 106  , 106.  , 266  , 266.12193       ,   2.01   } },
      { "BH-264"   , { 107  , 107.  , 264  , 264.12473       ,   2.01   } },
      { "MT-268"   , { 109  , 109.  , 268  , 268.13882       ,   2.01   } } 
    }
  ); ///< Reference map for atomic lookup 

  // Map atomic symbol to atomic number
  static std::unordered_map<std::string,int> atomicNumMap(
    {
      { "H"    ,  1   }, 
      { "He"   ,  2   },
      { "HE"   ,  2   },
      { "Li"   ,  3   },
      { "LI"   ,  3   },
      { "Be"   ,  4   },
      { "BE"   ,  4   },
      { "B"    ,  5   },
      { "C"    ,  6   },
      { "N"    ,  7   },
      { "O"    ,  8   },
      { "F"    ,  9   },
      { "Ne"   ,  10  },
      { "NE"   ,  10  },
      { "Na"   ,  11  },
      { "NA"   ,  11  },
      { "Mg"   ,  12  },
      { "MG"   ,  12  },
      { "Al"   ,  13  },
      { "AL"   ,  13  },
      { "Si"   ,  14  },
      { "SI"   ,  14  },
      { "P"    ,  15  },
      { "S"    ,  16  },
      { "Cl"   ,  17  },
      { "CL"   ,  17  },
      { "Ar"   ,  18  },
      { "AR"   ,  18  },
      { "K"    ,  19  },
      { "Ca"   ,  20  },
      { "CA"   ,  20  },
      { "Sc"   ,  21  },
      { "SC"   ,  21  },
      { "Ti"   ,  22  },
      { "TI"   ,  22  },
      { "V"    ,  23  },
      { "Cr"   ,  24  },
      { "CR"   ,  24  },
      { "Mn"   ,  25  },
      { "MN"   ,  25  },
      { "Fe"   ,  26  },
      { "FE"   ,  26  },
      { "Co"   ,  27  },
      { "CO"   ,  27  },
      { "Ni"   ,  28  },
      { "NI"   ,  28  },
      { "Cu"   ,  29  },
      { "CU"   ,  29  },
      { "Zn"   ,  30  },
      { "ZN"   ,  30  },
      { "Ga"   ,  31  },
      { "GA"   ,  31  },
      { "Ge"   ,  32  },
      { "GE"   ,  32  },
      { "As"   ,  33  },
      { "AS"   ,  33  },
      { "Se"   ,  34  },
      { "SE"   ,  34  },
      { "Br"   ,  35  },
      { "BR"   ,  35  },
      { "Kr"   ,  36  },
      { "KR"   ,  36  },
      { "Rb"   ,  37  },
      { "RB"   ,  37  },
      { "Sr"   ,  38  },
      { "SR"   ,  38  },
      { "Y"    ,  39  },
      { "Zr"   ,  40  },
      { "ZR"   ,  40  },
      { "Nb"   ,  41  },
      { "NB"   ,  41  },
      { "Mo"   ,  42  },
      { "MO"   ,  42  },
      { "Tc"   ,  43  },
      { "TC"   ,  43  },
      { "Ru"   ,  44  },
      { "RU"   ,  44  },
      { "Rh"   ,  45  },
      { "RH"   ,  45  },
      { "Pd"   ,  46  },
      { "PD"   ,  46  },
      { "Ag"   ,  47  },
      { "AG"   ,  47  },
      { "Cd"   ,  48  },
      { "CD"   ,  48  },
      { "In"   ,  49  },
      { "IN"   ,  49  },
      { "Sn"   ,  50  },
      { "SN"   ,  50  },
      { "Sb"   ,  51  },
      { "SB"   ,  51  },
      { "Te"   ,  52  },
      { "TE"   ,  52  },
      { "I"    ,  53  },
      { "Xe"   ,  54  },
      { "XE"   ,  54  },
      { "Cs"   ,  55  },
      { "CS"   ,  55  },
      { "Ba"   ,  56  },
      { "BA"   ,  56  },
      { "La"   ,  57  },
      { "LA"   ,  57  },
      { "Ce"   ,  58  },
      { "CE"   ,  58  },
      { "Pr"   ,  59  },
      { "PR"   ,  59  },
      { "Nd"   ,  60  },
      { "ND"   ,  60  },
      { "Pm"   ,  61  },
      { "PM"   ,  61  },
      { "Sm"   ,  62  },
      { "SM"   ,  62  },
      { "Eu"   ,  63  },
      { "EU"   ,  63  },
      { "Gd"   ,  64  },
      { "GD"   ,  64  },
      { "Tb"   ,  65  },
      { "TB"   ,  65  },
      { "Dy"   ,  66  },
      { "DY"   ,  66  },
      { "Ho"   ,  67  },
      { "HO"   ,  67  },
      { "Er"   ,  68  },
      { "ER"   ,  68  },
      { "Tm"   ,  69  },
      { "TM"   ,  69  },
      { "Yb"   ,  70  },
      { "YB"   ,  70  },
      { "Lu"   ,  71  },
      { "LU"   ,  71  },
      { "Hf"   ,  72  },
      { "HF"   ,  72  },
      { "Ta"   ,  73  },
      { "TA"   ,  73  },
      { "W"    ,  74  },
      { "Re"   ,  75  },
      { "RE"   ,  75  },
      { "Os"   ,  76  },
      { "OS"   ,  76  },
      { "Ir"   ,  77  },
      { "IR"   ,  77  },
      { "Pt"   ,  78  },
      { "PT"   ,  78  },
      { "Au"   ,  79  },
      { "AU"   ,  79  },
      { "Hg"   ,  80  },
      { "HG"   ,  80  },
      { "Tl"   ,  81  },
      { "TL"   ,  81  },
      { "Pb"   ,  82  },
      { "PB"   ,  82  },
      { "Bi"   ,  83  },
      { "BI"   ,  83  },
      { "Po"   ,  84  },
      { "PO"   ,  84  },
      { "At"   ,  85  },
      { "AT"   ,  85  },
      { "Rn"   ,  86  },
      { "RN"   ,  86  },
      { "Fr"   ,  87  },
      { "FR"   ,  87  },
      { "Ra"   ,  88  },
      { "RA"   ,  88  },
      { "Ac"   ,  89  },
      { "AC"   ,  89  },
      { "Th"   ,  90  },
      { "TH"   ,  90  },
      { "Pa"   ,  91  },
      { "PA"   ,  91  },
      { "U"    ,  92  },
      { "Np"   ,  93  },
      { "NP"   ,  93  },
      { "Pu"   ,  94  },
      { "PU"   ,  94  },
      { "Am"   ,  95  },
      { "AM"   ,  95  },
      { "Cm"   ,  96  },
      { "CM"   ,  96  },
      { "Bk"   ,  97  },
      { "BK"   ,  97  },
      { "Cf"   ,  98  },
      { "CF"   ,  98  },
      { "Es"   ,  99  },
      { "ES"   ,  99  },
      { "Fm"   ,  100 },
      { "FM"   ,  100 },
      { "Md"   ,  101 },
      { "MD"   ,  101 },
      { "No"   ,  102 },
      { "NO"   ,  102 },
      { "Lr"   ,  103 },
      { "LR"   ,  103 },
      { "Rf"   ,  104 },
      { "RF"   ,  104 },
      { "Db"   ,  105 },
      { "DB"   ,  105 },
      { "Sg"   ,  106 },
      { "SG"   ,  106 },
      { "Bh"   ,  107 },
      { "BH"   ,  107 },
      { "Mt"   ,  109 },
      { "MT"   ,  109 } 
    }
  ); ///< Reference map for atomic number 




  // Map atomic symbol to default isotope
  // https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&all=all&isotype=some
  static std::unordered_map<std::string,std::string> defaultIsotope(
    {
      { "H"    ,  "H-1"  }, 
      { "HE"   ,  "HE-4" },
      { "LI"   ,  "LI-7" },
      { "BE"   ,  "BE-9" },
      { "B"    ,  "B-11"  },
      { "C"    ,  "C-12"  },
      { "N"    ,  "N-14"  },
      { "O"    ,  "O-16"  },
      { "F"    ,  "F-19"  },
      { "NE"   ,  "NE-20" },
      { "NA"   ,  "NA-23" },
      { "MG"   ,  "MG-24" },
      { "AL"   ,  "AL-27" },
      { "SI"   ,  "SI-28" },
      { "P"    ,  "P-31"  },
      { "S"    ,  "S-32"  },
      { "CL"   ,  "CL-35" },
      { "AR"   ,  "AR-40" },
      { "K"    ,  "K-39"  },
      { "CA"   ,  "CA-40" },
      { "SC"   ,  "SC-45" },
      { "TI"   ,  "TI-48" },
      { "V"    ,  "V-51"  },
      { "CR"   ,  "CR-52" },
      { "MN"   ,  "MN-55" },
      { "FE"   ,  "FE-56" },
      { "CO"   ,  "CO-59" },
      { "NI"   ,  "NI-58" },
      { "CU"   ,  "CU-63" },
      { "ZN"   ,  "ZN-64" },
      { "GA"   ,  "GA-69" },
      { "GE"   ,  "GE-74" },
      { "AS"   ,  "AS-75" },
      { "SE"   ,  "SE-80" },
      { "BR"   ,  "BR-79" },
      { "KR"   ,  "KR-84" },
      { "RB"   ,  "RB-85" },
      { "SR"   ,  "SR-88" },
      { "Y"    ,  "Y-89"  },
      { "ZR"   ,  "ZR-90" },
      { "NB"   ,  "NB-93" },
      { "MO"   ,  "MO-98" },
      { "TC"   ,  "TC-98" },
      { "RU"   ,  "RU-102" },
      { "RH"   ,  "RH-103" },
      { "PD"   ,  "PD-106" },
      { "AG"   ,  "AG-107" },
      { "CD"   ,  "CD-114" },
      { "IN"   ,  "IN-115" },
      { "SN"   ,  "SN-120" },
      { "SB"   ,  "SB-121" },
      { "TE"   ,  "TE-130" },
      { "I"    ,  "I-127"  },
      { "XE"   ,  "XE-132" },
      { "CS"   ,  "CS-133" },
      { "BA"   ,  "BA-138" },
      { "LA"   ,  "LA-139" },
      { "CE"   ,  "CE-140" },
      { "PR"   ,  "PR-141" },
      { "ND"   ,  "ND-142" },
      { "PM"   ,  "PM-145" },
      { "SM"   ,  "SM-152" },
      { "EU"   ,  "EU-153" },
      { "GD"   ,  "GD-158" },
      { "TB"   ,  "TB-159" },
      { "DY"   ,  "DY-164" },
      { "HO"   ,  "HO-165" },
      { "ER"   ,  "ER-166" },
      { "TM"   ,  "TM-169" },
      { "YB"   ,  "YB-174" },
      { "LU"   ,  "LU-175" },
      { "HF"   ,  "HF-180" },
      { "TA"   ,  "TA-181" },
      { "W"    ,  "W-184"  },
      { "RE"   ,  "RE-187" },
      { "OS"   ,  "OS-192" },
      { "IR"   ,  "IR-193" },
      { "PT"   ,  "PT-195" },
      { "AU"   ,  "AU-197" },
      { "HG"   ,  "HG-202" },
      { "TL"   ,  "TL-205" },
      { "PB"   ,  "PB-208" },
      { "BI"   ,  "BI-209" },
      { "PO"   ,  "PO-209" },
      { "AT"   ,  "AT-210" },
      { "RN"   ,  "RN-222" },
      { "FR"   ,  "FR-223" },
      { "RA"   ,  "RA-226" },
      { "AC"   ,  "AC-227" },
      { "TH"   ,  "TH-232" },
      { "PA"   ,  "PA-231" },
      { "U"    ,  "U-238"  },
      { "NP"   ,  "NP-237" },
      { "PU"   ,  "PU-244" },
      { "AM"   ,  "XXX" },
      { "CM"   ,  "XXX" },
      { "BK"   ,  "XXX" },
      { "CF"   ,  "XXX" },
      { "ES"   ,  "XXX" },
      { "FM"   ,  "XXX" },
      { "MD"   ,  "XXX" },
      { "NO"   ,  "XXX" },
      { "LR"   ,  "XXX" },
      { "RF"   ,  "XXX" },
      { "DB"   ,  "XXX" },
      { "SG"   ,  "XXX" },
      { "BH"   ,  "XXX" },
      { "MT"   ,  "XXX" } 
    }
  ); ///< Reference map for atomic number 

  
  // Define Symbol + coordinate array constructor
  inline Atom::Atom(std::string symb, std::array<double,3> XYZ) :
    Atom(atomicReference[symb]){ coord = XYZ; };

}; // namespace ChronusQ

