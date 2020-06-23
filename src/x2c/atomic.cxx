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
#include <corehbuilder/x2c/atomic.hpp>
#include <util/preprocessor.hpp>
#include <singleslater/guess.hpp>
#include <cqlinalg.hpp>
#include <physcon.hpp>

namespace ChronusQ {

  template <typename MatsT, typename IntsT>
  AtomicX2C<MatsT,IntsT>::AtomicX2C(const AtomicX2C<MatsT,IntsT> &other) :
    AtomicX2C(other,0) {}

  template <typename MatsT, typename IntsT>
  AtomicX2C<MatsT,IntsT>::AtomicX2C(AtomicX2C<MatsT,IntsT> &&other) :
    AtomicX2C(std::move(other),0) {}

  template <typename MatsT, typename IntsT>
  template <typename MatsU>
  AtomicX2C<MatsT,IntsT>::AtomicX2C(const AtomicX2C<MatsU,IntsT> &other, int dummy) :
    X2C<MatsT,IntsT>(other),
    type_(other.type_), atomIdx_(other.atomIdx_) {
    for (const auto& atom : other.atoms_)
      atoms_.emplace_back(atom);
  }

  template <typename MatsT, typename IntsT>
  template <typename MatsU>
  AtomicX2C<MatsT,IntsT>::AtomicX2C(AtomicX2C<MatsU,IntsT> &&other, int dummy) :
    X2C<MatsT,IntsT>(other),
    type_(other.type_), atomIdx_(other.atomIdx_) {
    for (const auto& atom : other.atoms_)
      atoms_.emplace_back(atom);
  }

  template <typename MatsT, typename IntsT>
  void AtomicX2C<MatsT,IntsT>::dealloc() {
    X2C<MatsT,IntsT>::dealloc();
  }

  /**
   *  \brief Compute the AtomicX2C Core Hamiltonian
   */
  template <typename MatsT, typename IntsT>
  void AtomicX2C<MatsT, IntsT>::computeX2C(EMPerturbation &emPert, std::vector<MatsT*> &CH) {
    size_t NP = this->uncontractedBasis_.nPrimitive;
    size_t NB = this->basisSet_.nBasis;

    size_t Natom = this->molecule_.atoms.size();

    std::map<size_t, std::vector<size_t>> uniqueElements;
    std::vector<size_t> cumeNBs;

    if (type_.diagonalOnly)
      NRCoreH<MatsT, IntsT>(this->aoints_).computeNRCH(emPert, CH);

    atoms_.clear();
    atoms_.reserve(Natom);

    size_t maxAtomNB = 0, maxAtomNP = 0;

    size_t cumeNB = 0;
    for (size_t i = 0; i < Natom; i++) {

      Atom atom(this->molecule_.atoms[i]);
      if (type_.isolateAtom) {
        size_t aN = atom.atomicNumber;
        uniqueElements[aN].push_back(i);
        if (uniqueElements[aN][0] == i) {
          atomIdx_.push_back(atoms_.size());
          atom.coord = {0., 0., 0.};
        } else {
          atomIdx_.push_back(atomIdx_[uniqueElements[aN][0]]);
          cumeNBs.push_back(cumeNB);
          cumeNB += atoms_[atomIdx_.back()].basisSet_.nBasis;
          continue;
        }
      } else {
        uniqueElements[i].push_back(i);
        atomIdx_.push_back(i);
      }

      Molecule atomMol(0, atom.atomicNumber % 2 + 1, { atom });
      BasisSet basis(this->aoints_.basisSet().basisName,
        this->aoints_.basisSet().basisDef, this->aoints_.basisSet().inputDef,
        atomMol, this->aoints_.basisSet().basisType,
        this->aoints_.basisSet().forceCart, false);

      cumeNBs.push_back(cumeNB);
      cumeNB += basis.nBasis;
      maxAtomNB = std::max(basis.nBasis, maxAtomNB);
      maxAtomNP = std::max(basis.nPrimitive, maxAtomNP);

      std::shared_ptr<AOIntegrals<IntsT>> aointsAtom;
      if (type_.isolateAtom) {
        aointsAtom = std::make_shared<AOIntegrals<IntsT>>(this->memManager_,atomMol,basis);
      } else
        aointsAtom = std::make_shared<AOIntegrals<IntsT>>(this->memManager_,this->molecule_,basis);

      atoms_.emplace_back(*aointsAtom, not this->oneETerms_.SORelativistic);
    }

    std::vector<MatsT*> atomCH(CH.size(), nullptr);
    for (auto &CHi : atomCH) {
      CHi = this->memManager_.template malloc<MatsT>(maxAtomNB*maxAtomNB);
    }

    for (size_t k = 0; k < atoms_.size(); k++) {
      if (type_.diagonalOnly) {
        atoms_[k].computeX2C_corr(emPert, atomCH);
        size_t atomNB = atoms_[k].basisSet_.nBasis;
        size_t aN = k;
        if (type_.isolateAtom)
          aN = atoms_[k].molecule_.atoms[0].atomicNumber;
        for (size_t i : uniqueElements[aN])
          for (size_t j=0; j<atomCH.size(); j++)
            MatAdd('N','N', atomNB, atomNB,
              MatsT(1.), atomCH[j], atomNB,
              MatsT(1.), CH[j] + NB * cumeNBs[i] + cumeNBs[i], NB,
              CH[j] + NB * cumeNBs[i] + cumeNBs[i], NB);
      } else {
        atoms_[k].computeX2C(emPert, atomCH);
        atoms_[k].computeU();
      }
    }

    for (auto &CHi : atomCH) {
      this->memManager_.free(CHi);
    }

    assert(cumeNB == NB);

    if (type_.diagonalOnly) return;

    /// DLU/ALU O(N^3) Algorithm (commented out):
    ///   1> Combile diagonal matrix U = diag(U_A, U_B, U_C, ...)
    ///   2> Compute Hx2c = U * D * U
    //computeU();
    //this->uncontractedInts_.computeAOOneE(emPert,this->oneETerms_);
    //this->computeX2C_UDU(emPert, CH);
    //return;


    /// DLU/ALU O(N^2) Algorithm:
    ///   Hx2c_AB = U_A * D_AB * U_B

    // Allocate memory
    IntsT *T2c = this->memManager_.template malloc<IntsT>(4*maxAtomNP*maxAtomNP);
    IntsT *V2c = this->memManager_.template malloc<IntsT>(4*maxAtomNP*maxAtomNP);
    MatsT *W2c = this->memManager_.template malloc<MatsT>(4*maxAtomNP*maxAtomNP);
    MatsT *SCR = this->memManager_.template malloc<MatsT>(4*maxAtomNP*maxAtomNB);
    MatsT *Hx2c = this->memManager_.template malloc<MatsT>(4*maxAtomNB*maxAtomNB);

    // Allocate memory for the uncontracted spin components
    // of the 2C CH
    MatsT *HUnS = this->memManager_.template malloc<MatsT>(maxAtomNB*maxAtomNB);
    MatsT *HUnZ = this->memManager_.template malloc<MatsT>(maxAtomNB*maxAtomNB);
    MatsT *HUnX = this->memManager_.template malloc<MatsT>(maxAtomNB*maxAtomNB);
    MatsT *HUnY = this->memManager_.template malloc<MatsT>(maxAtomNB*maxAtomNB);

    this->uncontractedInts_.computeAOOneE(emPert,this->oneETerms_);

    this->W = this->memManager_.template malloc<MatsT>(4*NP*NP);

    if (this->oneETerms_.SORelativistic)
      formW(NP,this->W,2*NP,this->uncontractedInts_.PVdotP,NP,
        this->uncontractedInts_.PVcrossP[2],NP,
        this->uncontractedInts_.PVcrossP[1],NP,
        this->uncontractedInts_.PVcrossP[0],NP,
        not this->oneETerms_.SORelativistic);
    else
      formW(NP,this->W,2*NP,this->uncontractedInts_.PVdotP,NP,
        reinterpret_cast<IntsT*>(NULL),0,
        reinterpret_cast<IntsT*>(NULL),0,
        reinterpret_cast<IntsT*>(NULL),0,
        not this->oneETerms_.SORelativistic);

    size_t cumeINP = 0;
    size_t cumeINB = 0;
    for (size_t i = 0; i < Natom; i++) {
      size_t I = atomIdx_[i];

      size_t atomINP = atoms_[I].uncontractedBasis_.nPrimitive;
      size_t atomINB = atoms_[I].basisSet_.nBasis;

      size_t cumeJNP = 0;
      size_t cumeJNB = 0;
      for (size_t j = 0; j <= i; j++) {
        size_t J = atomIdx_[j];

        size_t atomJNP = atoms_[J].uncontractedBasis_.nPrimitive;
        size_t atomJNB = atoms_[J].basisSet_.nBasis;

        size_t TVshift = cumeINP + NP*cumeJNP;
        // T2c_AB = [ T_AB   0   ]
        //          [  0    T_AB ]
        SetMatDiag(atomINP,atomJNP,
          this->uncontractedInts_.kinetic + TVshift,NP,T2c,2*atomINP);

        // V2c_AB = [ V_AB   0   ]
        //          [  0    V_AB ]
        SetMatDiag(atomINP,atomJNP,
          this->uncontractedInts_.potential + TVshift,NP,V2c,2*atomINP);

        // W2c_AB = [ W11_AB  W12_AB]
        //          [ W21_AB  W22_AB]
        size_t WsubShift = cumeINP + 2*NP*cumeJNP;
        memset(W2c,0,4*atomINP*atomJNP*sizeof(MatsT));
        SetMat('N',atomINP,atomJNP,MatsT(1.),
          reinterpret_cast<MatsT*>(this->W)
            + WsubShift,2*NP,W2c,2*atomINP);
        SetMat('N',atomINP,atomJNP,MatsT(1.),
          reinterpret_cast<MatsT*>(this->W)
            + 2*NP*NP + WsubShift,2*NP,
          W2c + 2*atomINP*atomJNP,2*atomINP);
        SetMat('N',atomINP,atomJNP,MatsT(1.),
          reinterpret_cast<MatsT*>(this->W)
            + NP + WsubShift,2*NP,
          W2c + atomINP,2*atomINP);
        SetMat('N',atomINP,atomJNP,MatsT(1.),
          reinterpret_cast<MatsT*>(this->W)
            + 2*NP*NP + NP + WsubShift,2*NP,
          W2c + 2*atomINP*atomJNP + atomINP,2*atomINP);

        // Hx2c_AB = U_AA D_AB U_BB
        // Hx2c = UL^H * T2c * US
        Gemm('N','N',2*atomINP,2*atomJNB,2*atomJNP,MatsT(1.),
          T2c,2*atomINP,atoms_[J].US,2*atomJNP,MatsT(0.),SCR,2*atomINP);
        Gemm('C','N',2*atomINB,2*atomJNB,2*atomINP,MatsT(1.),
          atoms_[I].UL,2*atomINP,SCR,2*atomINP,MatsT(0.),Hx2c,2*atomINB);
        // Hx2c += US^H * T2c * UL
        Gemm('N','N',2*atomINP,2*atomJNB,2*atomJNP,MatsT(1.),
          T2c,2*atomINP,atoms_[J].UL,2*atomJNP,MatsT(0.),SCR,2*atomINP);
        Gemm('C','N',2*atomINB,2*atomJNB,2*atomINP,MatsT(1.),
          atoms_[I].US,2*atomINP,SCR,2*atomINP,MatsT(1.),Hx2c,2*atomINB);
        // Hx2c -= US^H * T2c * US
        Gemm('N','N',2*atomINP,2*atomJNB,2*atomJNP,MatsT(1.),
          T2c,2*atomINP,atoms_[J].US,2*atomJNP,MatsT(0.),SCR,2*atomINP);
        Gemm('C','N',2*atomINB,2*atomJNB,2*atomINP,MatsT(-1.),
          atoms_[I].US,2*atomINP,SCR,2*atomINP,MatsT(1.),Hx2c,2*atomINB);
        // Hx2c += UL^H * V2c * UL
        Gemm('N','N',2*atomINP,2*atomJNB,2*atomJNP,MatsT(1.),
          V2c,2*atomINP,atoms_[J].UL,2*atomJNP,MatsT(0.),SCR,2*atomINP);
        Gemm('C','N',2*atomINB,2*atomJNB,2*atomINP,MatsT(1.),
          atoms_[I].UL,2*atomINP,SCR,2*atomINP,MatsT(1.),Hx2c,2*atomINB);
        // Hx2c += 1/(4*C**2) US^H * W * US
        Gemm('N','N',2*atomINP,2*atomJNB,2*atomJNP,
          MatsT(0.25/SpeedOfLight/SpeedOfLight),
          W2c,2*atomINP,atoms_[J].US,2*atomJNP,MatsT(0.),SCR,2*atomINP);
        Gemm('C','N',2*atomINB,2*atomJNB,2*atomINP,MatsT(1.),
          atoms_[I].US,2*atomINP,SCR,2*atomINP,MatsT(1.),Hx2c,2*atomINB);

        if (this->oneETerms_.SORelativistic)
          SpinScatter(atomINB,atomJNB,Hx2c,2*atomINB,HUnS,atomINB,
            HUnZ,atomINB,HUnY,atomINB,HUnX,atomINB);
        else {
          MatAdd('N','N',atomINB,atomJNB,MatsT(1.),Hx2c,2*atomINB,MatsT(1.),
            Hx2c+atomINB+2*atomINB*atomJNB,2*atomINB,HUnS,atomINB);
          if (CH.size() > 1) {
            memset(HUnZ,0,atomINB*atomJNB*sizeof(MatsT));
            memset(HUnY,0,atomINB*atomJNB*sizeof(MatsT));
            memset(HUnX,0,atomINB*atomJNB*sizeof(MatsT));
          }
        }

        size_t CHshift = cumeINB + NB*cumeJNB;
        SetMat('N',atomINB,atomJNB,MatsT(1.),HUnS,atomINB,CH[0]+CHshift,NB);
        if (CH.size() > 1) {
          SetMat('N',atomINB,atomJNB,MatsT(1.),HUnZ,atomINB,CH[1]+CHshift,NB);
          SetMat('N',atomINB,atomJNB,MatsT(1.),HUnY,atomINB,CH[2]+CHshift,NB);
          SetMat('N',atomINB,atomJNB,MatsT(1.),HUnX,atomINB,CH[3]+CHshift,NB);
        }

        if (i != j) {
          CHshift = cumeJNB + NB*cumeINB;
          SetMat('C',atomINB,atomJNB,MatsT(1.),HUnS,atomINB,CH[0]+CHshift,NB);
          if (CH.size() > 1) {
            SetMat('C',atomINB,atomJNB,MatsT(1.),HUnZ,atomINB,CH[1]+CHshift,NB);
            SetMat('C',atomINB,atomJNB,MatsT(1.),HUnY,atomINB,CH[2]+CHshift,NB);
            SetMat('C',atomINB,atomJNB,MatsT(1.),HUnX,atomINB,CH[3]+CHshift,NB);
          }
        }

        cumeJNP += atomJNP;
        cumeJNB += atomJNB;
      }

      cumeINP += atomINP;
      cumeINB += atomINB;
    }

    this->memManager_.free(T2c, V2c, W2c, SCR, Hx2c, HUnS, HUnZ, HUnY, HUnX);

  }

  template<> void AtomicX2C<dcomplex,dcomplex>::computeX2C(EMPerturbation&, std::vector<dcomplex*>&) {
    CErr("X2C + Complex Ints NYI",std::cout);
  }

  /**
   *  \brief Compute the picture change matrices UL, US
   */
  template <typename MatsT, typename IntsT>
  void AtomicX2C<MatsT, IntsT>::computeU() {

    size_t NP = this->uncontractedBasis_.nPrimitive;
    size_t NB = this->basisSet_.nBasis;

    this->UL = this->memManager_.template malloc<MatsT>(4*NP*NB);
    memset(this->UL,0,4*NP*NB*sizeof(MatsT));
    this->US = this->memManager_.template malloc<MatsT>(4*NP*NB);
    memset(this->US,0,4*NP*NB*sizeof(MatsT));

    size_t cumeNP = 0, cumeNB = 0;

    for (size_t k : atomIdx_) {
      const auto &at = atoms_[k];

      size_t atomNP = at.uncontractedBasis_.nPrimitive;
      size_t atomNB = at.basisSet_.nBasis;

      SetMat('N', atomNP, atomNB, MatsT(1.),
        at.UL, 2*atomNP,
        this->UL + 2*NP*cumeNB + cumeNP, 2*NP);
      SetMat('N', atomNP, atomNB, MatsT(1.),
        at.UL + atomNP, 2*atomNP,
        this->UL + 2*NP*cumeNB + NP + cumeNP, 2*NP);
      SetMat('N', atomNP, atomNB, MatsT(1.),
        at.UL + 2*atomNP*atomNB, 2*atomNP,
        this->UL + 2*NP*NB + 2*NP*cumeNB + cumeNP, 2*NP);
      SetMat('N', atomNP, atomNB, MatsT(1.),
        at.UL + 2*atomNP*atomNB + atomNP, 2*atomNP,
        this->UL + 2*NP*NB + 2*NP*cumeNB + NP + cumeNP, 2*NP);

      SetMat('N', atomNP, atomNB, MatsT(1.),
        at.US, 2*atomNP,
        this->US + 2*NP*cumeNB + cumeNP, 2*NP);
      SetMat('N', atomNP, atomNB, MatsT(1.),
        at.US + atomNP, 2*atomNP,
        this->US + 2*NP*cumeNB + NP + cumeNP, 2*NP);
      SetMat('N', atomNP, atomNB, MatsT(1.),
        at.US + 2*atomNP*atomNB, 2*atomNP,
        this->US + 2*NP*NB + 2*NP*cumeNB + cumeNP, 2*NP);
      SetMat('N', atomNP, atomNB, MatsT(1.),
        at.US + 2*atomNP*atomNB + atomNP, 2*atomNP,
        this->US + 2*NP*NB + 2*NP*cumeNB + NP + cumeNP, 2*NP);

      cumeNP += atomNP;
      cumeNB += atomNB;

    }

  }

  template<> void AtomicX2C<dcomplex,dcomplex>::computeU() {
    CErr("X2C + Complex Ints NYI",std::cout);
  }

  template class AtomicX2C<double,double>;
  template class AtomicX2C<dcomplex,double>;
  template class AtomicX2C<dcomplex,dcomplex>;

  // Instantiate copy constructors
  template AtomicX2C<dcomplex,double>::AtomicX2C(const AtomicX2C<double,double> &, int);
  template AtomicX2C<dcomplex,dcomplex>::AtomicX2C(const AtomicX2C<dcomplex,dcomplex> &, int);

}; // namespace ChronusQ
