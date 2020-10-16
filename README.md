<div align="center">
  <img src="cq_logo.png" height="220px"/>
</div>

Chronus Quantum 
===============

The Chronus Quantum (ChronusQ) Software Package [v. Beta] is a high-performance
computational chemistry software package with a strong emphasis on explicitly
time-dependent and post-SCF quantum mechanical methods.

* [Changelog](CHANGELOG.md)
* [Documentation](../../wikis/home)
* [Installation](#installation)



Installation
------------

### Prerequisites

- C++14 compiler 
- C compiler (for LibXC)
- Fortran compiler (for LibXC)
- [CMake](http://cmake.org) build system (Version 3.11+).
- [HDF5](https://support.hdfgroup.org/HDF5/)
- [Eigen3](http://eigen.tuxfamily.org)

### Quickstart

If you have all the prerequisites above, all you need to do is:

```
git clone https://urania.chem.washington.edu/chronusq/chronusq_public.git
mkdir chronusq_public/build && cd chronusq_public/build
cmake ..
cmake --build .
```
To install, you then run
```
cmake --build . --target install
```

For more details on installation requirements, and running ChronusQ, see the [getting ChronusQ](../../wikis/compilation)
and [running ChronusQ](../../wikis/Running-ChronusQ) wiki pages.


Citing ChronusQ
---------------
The following WIREs paper and software citation should be cited in publications using the ChronusQ package located in [CITE.txt](CITE.txt).


Found a bug or want a new feature?
----------------------------------
Please submit a bug report or feature request on the [issues](https://urania.chem.washington.edu/chronusq/chronusq_public/-/issues) page.


General Inquiries
-----------------
- Contact xsli at uw dot edu

