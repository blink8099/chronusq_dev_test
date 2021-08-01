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
#include <molecule.hpp>
#include <geometrymodifier.hpp>


namespace ChronusQ {

  /**
   * \brief The MolecularDynamics class
   */
  class MolecularDynamics : public GeometryModifier {

  public:

    std::vector<double> velocityHalfTime;  ///< nuclear velocity at half time, (t-1/2) upon entry and (t+1/2) upon exist
    std::vector<double> velocityCurrent;   ///< nuclear velocity at the current time (t)
    std::vector<double> acceleration;      ///< acceleration at the current time (t)

    double   nuclearKineticEnergy; ///< Nuclear kinetic energy

    // Constructors
    MolecularDynamics() = delete;
    MolecularDynamics(MolecularOptions& molecularOptions, Molecule& molecule) :
      GeometryModifier(molecularOptions),
      velocityHalfTime(3*molecule.nAtoms, 0.),
      velocityCurrent(3*molecule.nAtoms, 0.),
      acceleration(3*molecule.nAtoms, 0.) {};

    MolecularDynamics(MolecularOptions &molecularOptions):
        GeometryModifier(molecularOptions) {}

    // Different type
    MolecularDynamics(const MolecularDynamics &other):
        GeometryModifier(other){}
    MolecularDynamics(MolecularDynamics &&other):
        GeometryModifier(other){}

    // Virtual destructor
    virtual ~MolecularDynamics() {}

    void initializeMD(Molecule& molecule){

      std::vector<double> velocityHalfTime(3*molecule.nAtoms, 0.);
      std::vector<double> velocityCurrent(3*molecule.nAtoms, 0.);
      std::vector<double> acceleration(3*molecule.nAtoms, 0.);
      nuclearKineticEnergy = 0.0;

    }

    
    /**
     *  \brief Updates the positions of the classical nuclei. Recompute
     *  all member data to fit with the new positions
     *
     *  \param[in] pos  An array holding the new positions
     */
    void updateNuclearCoordinates(bool print, 
                                  Molecule &molecule, 
                                  std::vector<double> gradientCurrent,
                                  bool firstStep, 
                                  bool moveGeometry,
                                  bool moveVelocity) {

      size_t i = 0;

      if(print) {
        std::cout << std::scientific << std::setprecision(12);

        std::cout << std::endl<<"Molecular Geometry: (Bohr)"<<std::endl;
        for( Atom& atom : molecule.atoms ) {

          std::cout << std::right <<"AtomicNumber = " << std::setw(4) << atom.atomicNumber 
                    << std::right <<"  X= "<< std::setw(19) << atom.coord[0]
                    << std::right <<"  Y= "<< std::setw(19) << atom.coord[1]
                    << std::right <<"  Z= "<< std::setw(19) << atom.coord[2] <<std::endl;
          i += 3;
 
        }
      }

      // if velocity Verlet
      if(moveVelocity) {
        velocityVV(molecule, gradientCurrent, molecularOptions_.timeStepAU, firstStep);
        // compute kinetic energy
        computeKineticEnergy(molecule);
      }
      if(moveGeometry) geometryVV(molecule, gradientCurrent, molecularOptions_.timeStepAU);

      // update other quantities
      molecule.update();



      // output important dynamic information
      if(print) {
        std::cout << "Velocity:"<<std::endl;
        i = 0;
        for( Atom& atom : molecule.atoms ) {

          std::cout << std::right <<"AtomicNumber = " << std::setw(4) << atom.atomicNumber 
                    << std::right <<"  X= "<< std::setw(19) <<  velocityCurrent[i  ]
                    << std::right <<"  Y= "<< std::setw(19) <<  velocityCurrent[i+1]
                    << std::right <<"  Z= "<< std::setw(19) <<  velocityCurrent[i+2]<<std::endl;
          i += 3;
 
        }

        std::cout << "Forces: (Hartrees/Bohr)"<<std::endl;
        i = 0;
        for( Atom& atom : molecule.atoms ) {

          std::cout <<"AtomicNumber = " << std::setw(4) <<  atom.atomicNumber 
                    << std::right <<"  X= "<< std::setw(19) <<  -gradientCurrent[i  ]
                    << std::right <<"  Y= "<< std::setw(19) <<  -gradientCurrent[i+1]
                    << std::right <<"  Z= "<< std::setw(19) <<  -gradientCurrent[i+2]<<std::endl;

          i += 3;
 
        }

        std::cout << std::endl<<"Predicted Molecular Geometry: (Bohr)"<<std::endl;
        i = 0;
        for( Atom& atom : molecule.atoms ) {

          std::cout << std::right <<"AtomicNumber = " << std::setw(4) << atom.atomicNumber 
                    << std::right <<"  X= "<< std::setw(19) << atom.coord[0]
                    << std::right <<"  Y= "<< std::setw(19) << atom.coord[1]
                    << std::right <<"  Z= "<< std::setw(19) << atom.coord[2] <<std::endl;
          i += 3;
 
        }
      }


    }

    // Advance the velocity using the Verlet
    void velocityVV(Molecule &molecule, std::vector<double> gradientCurrent, double timeStep, bool firstStep){

      size_t i = 0;

      // loop over atoms
      for( Atom& atom : molecule.atoms ) {
        //compute acceleration = -g/m
	acceleration[i  ] = -gradientCurrent[i  ]/(AUPerAMU*atom.atomicMass);
	acceleration[i+1] = -gradientCurrent[i+1]/(AUPerAMU*atom.atomicMass);
	acceleration[i+2] = -gradientCurrent[i+2]/(AUPerAMU*atom.atomicMass);

        //advance the half-time velocity to the current step 
        //v(t+1) = v(t+1/2) + 1/2dT∙a(t+1)
	if(not firstStep) {
	  velocityCurrent[i  ] = velocityHalfTime[i  ] + 0.5*timeStep*acceleration[i  ];
	  velocityCurrent[i+1] = velocityHalfTime[i+1] + 0.5*timeStep*acceleration[i+1];
	  velocityCurrent[i+2] = velocityHalfTime[i+2] + 0.5*timeStep*acceleration[i+2];
	}

        //prepare the next velocity at half-time
	//v(t+1/2) = v(t) + 1/2dT∙a(t)
	velocityHalfTime[i  ] = velocityCurrent[i  ] + 0.5*timeStep*acceleration[i  ];
	velocityHalfTime[i+1] = velocityCurrent[i+1] + 0.5*timeStep*acceleration[i+1];
	velocityHalfTime[i+2] = velocityCurrent[i+2] + 0.5*timeStep*acceleration[i+2];

        i+=3;
      }

    }

    // Advance the geometry using the velocity
    void geometryVV(Molecule &molecule, std::vector<double> gradientCurrent, double timeStep){

      size_t i = 0;

      // loop over atoms
      for( Atom& atom : molecule.atoms ) {
        //advance the geometry to the next time 
        //r(t+1) = r(t) + dT∙v(t+1/2)
        atom.coord[0] += timeStep*velocityHalfTime[i  ]; // x
        atom.coord[1] += timeStep*velocityHalfTime[i+1]; // y
        atom.coord[2] += timeStep*velocityHalfTime[i+2]; // z

        i+=3;
      }
 
    }


    /**
    *  \brief Compute the nuclear-nuclear repulsion energy for classical
    *  point nuclei using the Atoms contained in the atoms array
    *
    *  \f[
    *    V_{NN} = \sum_{A < B} \frac{Z_A Z_B}{R_{AB}}
    *  \f]
    */ 
    void computeKineticEnergy(Molecule& molecule) {

      nuclearKineticEnergy = 0.;

      size_t i = 0;
      for( Atom& atom : molecule.atoms ) {
        if (not atom.quantum) {
          nuclearKineticEnergy += 0.5*velocityCurrent[i  ]*velocityCurrent[i  ]*atom.atomicMass*AUPerAMU;
          nuclearKineticEnergy += 0.5*velocityCurrent[i+1]*velocityCurrent[i+1]*atom.atomicMass*AUPerAMU;
          nuclearKineticEnergy += 0.5*velocityCurrent[i+2]*velocityCurrent[i+2]*atom.atomicMass*AUPerAMU;
	}
	i += 3;
      }
    }

  };

}
