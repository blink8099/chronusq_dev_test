                     CHRONUSQ RELEASE HISTORY


  This file simply provides a brief overview of the release history of the
  ChronusQ program as released by the Li Research Group at the University
  of Washington. For the most up-to-date record of functionality, etc,
  please refer to the [ChronusQ Wiki](https://urania.chem.washington.edu/chronusq/chronusq_public/wikis/home).


  FORMAT: YYYY-MM-DD

  - 2020-03-12 0.3.1 (BETA)
    - Add ability to restart interrupted RT jobs
    - Add ability to turn the SCF field off in RT jobs
    - Fix sign bug in X2C calculations
    - Fix CXXBLACS linking error for MPI builds
    - Add MPI build into continuous integration
    - Fix bug in reading basis sets using Fortran float notation

  - 2019-05-06 0.3.0 (BETA)
    - Implementation of Explicit Magnus 2nd order step in RT module
    - Default parameter changes in GPLHR and SCF
    - CI through GitLab
    - Clang 9+ compatible

  - 2018-11-28 0.2.1 (BETA)
    - Removed Boost depedency
      - Switched Boost::Test -> GTest for UT system
      - Created in house segregated storage engine to remove boost::simple_segregated_storage
      - Removed Boost::Filesystem for file search / copy
    - Fixed HDF5 link in the presense of static libraries
      - This allows CQ to be compiled statically
    - Fixed dependency tree to allow for parallel make in openblas builds    
    - Misc GCC 6 comptibility fixes
    - Parallel (SMP + MPI) GIAO Fock builds
    - Direct GIAO Fock builds in RT module
    - Bump Libint -> 2.5.0-beta
    - Bump CMake  -> 3.11


  - 2018-07-13 0.2.0 (BETA)
    - Full integration of GIAO basis set into SCF and RT modules
    - Added RESPONSE module
      - Supports the PolarizationPropagator (TDDFT/TDHF) and ParticleParticlePropagator (pp-RPA/pp-TDA/hh-TDA)
      - RESIDUE -> eigen decomposition
      - (D)FDR -> (damped) frequency depenent response
      - GPLHR for partial diagonalization
        - Supports arbitrary energy domain for diagonalization
    - Full integration of MPI functionality throughout (CQ_ENABLE_MPI)
      - Using MXX for C++11 MPI bindings
      - Using CXXBLACS for C++ interface to BLACS type functionality
      - Integration of ScaLAPACK into RESPONSE module
    - Bump Libint -> 2.4.2
    - Bump Libxc  -> 4.0.4
    - Added support for coverage checks (CQ_ENABLE_COVERAGE)
    - Various logic checks / bug fixes
  

  - 2017-09-01: 0.1.0 (BETA)
    - Complete overhaul of ChronusQ development stream (new repo)
    - Currently tested functionality:
      - Full support for Hartree-Fock and Kohn-Sham references
      - SCF general to Restricted (R), Unrestricted (U), and Generalized (G) references
      - Real-time propagation of R/U/G references
      - X2C Relativistic references (both HF and KS)
      - Full integration of OpenMP in performance critical code
      - Support for both INCORE (full ERI) and DIRECT integral contraction schemes
