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
#include <integrals.hpp>
#include <particleintegrals/twopints/incore4indextpi.hpp>
#include <particleintegrals/twopints/incoreritpi.hpp>
#include <particleintegrals/onepints/relativisticints.hpp>
#include <matrix.hpp>

namespace ChronusQ {

  /**
   *  \brief Allocate, compute  and store the 1-e integrals +
   *  orthonormalization matricies over the given CGTO basis.
   *
   *  Computes:
   *    Overlap + length gauge Electric Multipoles
   *    Kinetic energy matrix
   *    Nuclear potential energy matrix
   *    Core Hamiltonian (T + V)
   *    Orthonormalization matricies (Lowdin / Cholesky)
   *
   */
  template <typename IntsT>
  void Integrals<IntsT>::computeAOOneP(CQMemManager &mem,
      Molecule &mol, BasisSet &basis, EMPerturbation &emPert,
      const std::vector<std::pair<OPERATOR,size_t>> &ops,
      const HamiltonianOptions &options) {

    const std::array<std::string,3> dipoleList =
      { "X","Y","Z" };
    const std::array<std::string,6> quadrupoleList =
      { "XX","XY","XZ","YY","YZ","ZZ" };
    const std::array<std::string,10> octupoleList =
      { "XXX","XXY","XXZ","XYY","XYZ","XZZ","YYY",
        "YYZ","YZZ","ZZZ" };
    const std::array<std::string,9> quadrupoleListAsymm =
      { "XX","XY","XZ","YX","YY","YZ","ZX","ZY","ZZ" };

    size_t NB = basis.nBasis;

    bool electron = (options.particle.charge < 0.);

    std::string prefix = electron ? "INTS/" : "PINTS/";

    for (const std::pair<OPERATOR,size_t> &op : ops)
      switch (op.first) {
      case OVERLAP:
        overlap = std::make_shared<OnePInts<IntsT>>(mem, NB);
        overlap->computeAOInts(basis, mol, emPert, OVERLAP, options);
        if( savFile.exists() )
          savFile.safeWriteData(prefix + "OVERLAP", overlap->pointer(), {NB,NB});
        break;

      case KINETIC:
        kinetic = std::make_shared<OnePInts<IntsT>>(mem, NB);
        kinetic->computeAOInts(basis, mol, emPert, KINETIC, options);
        if( savFile.exists() )
          savFile.safeWriteData(prefix + "KINETIC", kinetic->pointer(), {NB,NB});
        break;

      case NUCLEAR_POTENTIAL:
        // Use Libint for point nuclei, in-house for Gaussian nuclei
        if (options.OneEScalarRelativity)
          potential = std::make_shared<OnePRelInts<IntsT>>(
              mem, NB, options.OneESpinOrbit);
        else
          potential = std::make_shared<OnePInts<IntsT>>(mem, NB);
        potential->computeAOInts(basis, mol, emPert, NUCLEAR_POTENTIAL, options);
        if( savFile.exists() ) {
          std::string potentialTag = options.finiteWidthNuc ? "_FINITE_WIDTH" : "";
          savFile.safeWriteData(prefix + "POTENTIAL" + potentialTag,potential->pointer(),{NB,NB});
        }
        break;

      case LEN_ELECTRIC_MULTIPOLE:
        lenElectric = std::make_shared<MultipoleInts<IntsT>>(mem, NB, op.second, true);
        lenElectric->computeAOInts(basis, mol, emPert, LEN_ELECTRIC_MULTIPOLE, options);
        if( savFile.exists() ) {
          // Length Gauge electric dipole
          for(auto i = 0; i < 3; i++)
            savFile.safeWriteData(prefix + "ELEC_DIPOLE_LEN_" +
              dipoleList[i], (*lenElectric)[dipoleList[i]].pointer(), {NB,NB} );

          // Length Gauge electric quadrupole
          if(op.second >= 2)
            for(auto i = 0; i < 6; i++)
              savFile.safeWriteData(prefix + "ELEC_QUADRUPOLE_LEN_" +
                quadrupoleList[i], (*lenElectric)[quadrupoleList[i]].pointer(), {NB,NB} );

          // Length Gauge electric octupole
          if(op.second >= 3)
            for(auto i = 0; i < 10; i++)
              savFile.safeWriteData(prefix + "/ELEC_OCTUPOLE_LEN_" +
                octupoleList[i], (*lenElectric)[octupoleList[i]].pointer(), {NB,NB} );
        }
        break;

      case VEL_ELECTRIC_MULTIPOLE:
        velElectric = std::make_shared<MultipoleInts<IntsT>>(mem, NB, op.second, true);
        velElectric->computeAOInts(basis, mol, emPert, VEL_ELECTRIC_MULTIPOLE, options);
        if( savFile.exists() ) {
          // Velocity Gauge electric dipole
          for(auto i = 0; i < 3; i++)
            savFile.safeWriteData(prefix + "ELEC_DIPOLE_VEL_" +
              dipoleList[i], (*velElectric)[dipoleList[i]].pointer(), {NB,NB} );

          // Velocity Gauge electric quadrupole
          if(op.second >= 2)
            for(auto i = 0; i < 6; i++)
              savFile.safeWriteData(prefix + "ELEC_QUADRUPOLE_VEL_" +
                quadrupoleList[i], (*velElectric)[quadrupoleList[i]].pointer(), {NB,NB} );

          // Velocity Gauge electric octupole
          if(op.second >= 3)
            for(auto i = 0; i < 10; i++)
              savFile.safeWriteData(prefix + "ELEC_OCTUPOLE_VEL_" +
                octupoleList[i], (*velElectric)[octupoleList[i]].pointer(), {NB,NB} );
        }
        break;

      case MAGNETIC_MULTIPOLE:
        magnetic = std::make_shared<MultipoleInts<IntsT>>(mem, NB, op.second, false);
        magnetic->computeAOInts(basis, mol, emPert, MAGNETIC_MULTIPOLE, options);
        if( savFile.exists() ) {
          // Magnetic Dipole
          for(auto i = 0; i < 3; i++)
            savFile.safeWriteData(prefix + "MAG_DIPOLE_" +
              dipoleList[i], (*magnetic)[dipoleList[i]].pointer(), {NB,NB} );

          // Magnetic Quadrupole
          if(op.second >= 2)
            for(auto i = 0; i < 6; i++)
              savFile.safeWriteData(prefix + "MAG_QUADRUPOLE_" +
                quadrupoleList[i], (*magnetic)[i+3].pointer(), {NB,NB} );
        }
        break;

      case ELECTRON_REPULSION:
        CErr("Electron repulsion integrals are not implemented in computeAOOneP,"
             " they are implemented in TwoPInts",std::cout);
        break;
      }

  }; // AOIntegrals<IntsT>::computeAOOneP

}; // namespace ChronusQ
