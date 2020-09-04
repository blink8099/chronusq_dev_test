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

#include <electronintegrals/inhouseaointegral.hpp>

namespace ChronusQ {


  std::vector<double> RealGTOIntEngine::BottomupHGP( 
    libint2::ShellPair &pair1, libint2::ShellPair &pair2, 
    libint2::Shell &shell1, libint2::Shell &shell2,
    libint2::Shell &shell3, libint2::Shell &shell4) {

    int LA, LB, LC, LD;
    LA = shell1.contr[0].l;
    LB = shell2.contr[0].l;
    LC = shell3.contr[0].l;
    LD = shell4.contr[0].l;

    
    int L = shell1.contr[0].l + shell2.contr[0].l 
            + shell3.contr[0].l + shell4.contr[0].l;

    int Lbra = shell1.contr[0].l + shell2.contr[0].l;
    int Lket = shell3.contr[0].l + shell4.contr[0].l;
    
    double A[3],B[3],C[3],D[3];

    for (int iWork = 0; iWork < 3; iWork++) {

      A[iWork] = shell1.O[iWork];
      B[iWork] = shell2.O[iWork];
      C[iWork] = shell3.O[iWork];
      D[iWork] = shell4.O[iWork];

    }

    double AB[3], CD[3];

    for (int iWork = 0; iWork < 3; iWork++) {

      AB[iWork] = A[iWork] - B[iWork];
      CD[iWork] = C[iWork] - D[iWork];

    }

    // here allocate the boys function  

    double *FT = new double[L + 1];  
    double P[3],Q[3];
 
    // Allocate the vectors for Vbraket 
    std::vector<std::vector<std::vector<double>>> Vbraket((Lbra - LA + 1) 
        * (Lket - LC + 1));

    for (int lA = LA; lA <= Lbra; lA++) {

      for (int lC = LC; lC <= Lket; lC++) {

        Vbraket[(lA - LA) * (Lket - LC + 1) + (lC - LC)].resize((Lbra - lA + 1)
            * (Lket - lC + 1));

        for (int lB = 0; lB <= Lbra - lA; lB++) {

          for (int lD = 0; lD <= Lket - lC; lD++) {

            Vbraket[(lA - LA) * (Lket - LC + 1) + (lC - LC)][
                    lB * (Lket - lC + 1) + lD
                   ].assign(cart_ang_list[lA].size() 
                            * cart_ang_list[lB].size() 
                            * cart_ang_list[lC].size() 
                            * cart_ang_list[lD].size(), 0.0);

          } // for (int lD = 0; lD <= Lket - lC; lD++)

        } // for (int lB = 0; lB <= Lbra - lA; lB++)

      } // for (int lC = LC; lC <= Lket; lC++)

    } // for (int lA = LA; lA <= Lbra; lA++)

    // loop over primitive shellpairs in bra side 
    for (auto &pripair1 : pair1.primpairs) { 

      for (int iWork = 0; iWork < 3; iWork++) {

        P[iWork] = pripair1.P[iWork];

      } // for (int iWork = 0; iWork < 3; iWork++)

      for (auto &pripair2 : pair2.primpairs) {

        for (int iWork = 0; iWork < 3; iWork++) {

          Q[iWork] = pripair2.P[iWork];

        } //  for (int iWork = 0; iWork < 3; iWork++)

        double sqrPQ = 0.0;

        for (int mu = 0; mu < 3; mu++) {

          double PQ = (pripair1.P[mu] - pripair2.P[mu]); 
          sqrPQ += PQ * PQ;

        } // for (int mu=0; mu < 3; mu++)

        auto zeta = 1.0 / pripair1.one_over_gamma;
        auto eta  = 1.0 / pripair2.one_over_gamma;
        auto zetaG = zeta + eta;
        auto rho = zeta * eta / zetaG;
        auto T = rho * sqrPQ;

        // calculate Fm(T) list
        computeFmTTaylor(FT, T, L, 0);

        // zeta + eta is expoT
        double expoT = 1.0 / pripair1.one_over_gamma 
                       + 1.0 / pripair2.one_over_gamma;
      
        // T= rho * PQ^2
        // T = sqrPQ / (pripair1.one_over_gamma + pripair2.one_over_gamma);
      
        // calculate F0(T)
        // computeFmTTaylor(FT, T, 0, 0);
      
        double Kab = sqrt(2.0) * pow(M_PI, 1.25) * pripair1.K; 
        double Kcd = sqrt(2.0) * pow(M_PI, 1.25) * pripair2.K;
      
        double norm = shell1.contr[0].coeff[pripair1.p1]
                      * shell2.contr[0].coeff[pripair1.p2]
                      * shell3.contr[0].coeff[pripair2.p1]
                      * shell4.contr[0].coeff[pripair2.p2];  
      
        double pref = norm * Kab * Kcd / sqrt(expoT); 

        std::vector<std::vector<std::vector<double>>> Vtempbraket((Lbra + 1) 
            * (Lket + 1));

        for (int k = 0; k <= Lbra; k++) {
        
          for (int l = 0; l <= Lket; l++) {

            Vtempbraket[k * (Lket + 1) + l].resize(cart_ang_list[k].size()
                                                   * cart_ang_list[l].size());

            int mbraket = L - k - l;

            for (int cart_i = 0; cart_i < cart_ang_list[k].size(); cart_i++) {

              for (int cart_j = 0; cart_j < cart_ang_list[l].size(); cart_j++) 
              {

                Vtempbraket[k * (Lket + 1) + l][
                  cart_i * cart_ang_list[l].size() + cart_j
                  ].resize(mbraket + 1); 

              } // for (int cart_j = 0; cart_j < cart_ang_list[l].size(); 
                // cart_j++)

            } // for (int cart_i = 0; cart_i < cart_ang_list[k].size(); 
              // cart_i++)
          
          } // for (int l = 0; l <= Lket; l++)

        } // for (int k = 0; k <= Lbra; k++)

        for (int ii = 0; ii <= L; ii++) {

          Vtempbraket[0][0][ii] = FT[ii] * pref;

          // std::cout << "FT" << ii << "= " << FT[ii] << " pref " << pref << 
          //   std::endl;

        } // for (int ii = 0; ii <= L; ii++)

        double W[3];

        for (int ii = 0; ii < 3; ii++) {

          W[ii] = (zeta * P[ii] + eta * Q[ii]) / zetaG;

        } // for (int ii = 0; ii < 3; ii++)

        for (int k = 0; k <= Lbra; k++) { // Loop over the bra angular momentum
          
          for (int cart_i = 0; cart_i < cart_ang_list[k].size(); cart_i++) {

            int lA_xyz[3];     

            for (int ii = 0; ii < 3; ii++) { 

              lA_xyz[ii] = cart_ang_list[k][cart_i][ii];

            } // for (int ii = 0; ii < 3; ii++)

            int mbra = L - k; // mbra is the highest auxiliary number for a 
            // given k

            if (k > 0) {

              int iwork;

              if (lA_xyz[0] > 0) {

                iwork = 0;

              } // if (lA_xyz[0] > 0)

              else if (lA_xyz[1] > 0) {

                iwork = 1;

              } // else if (lA_xyz[1] > 0)

              else if (lA_xyz[2] > 0) { 

                iwork = 2;

              } // else if (lA_xyz[2] > 0)

              // Calculate the index of a lower angular momentum integral
              int lAtemp[3];

              for (int ii = 0; ii < 3; ii++) { 

                lAtemp[ii] = lA_xyz[ii];

              } // for (int ii = 0; ii < 3; ii++)

              lAtemp[iwork] = lA_xyz[iwork] - 1;

              int indexlm1xyz = indexmap(k - 1, lAtemp[0], lAtemp[1], 
                                         lAtemp[2]);

              // Calculate index of lAm1
              
              for (int m = 0; m <= mbra; m++) {
              
                double ERIscratch = 0.0;

                ERIscratch = (P[iwork] - A[iwork]) 
                             * Vtempbraket[
                             (k - 1) * (Lket + 1)][
                             indexlm1xyz][
                             m]
                             + (W[iwork] - P[iwork])
                             * Vtempbraket[
                             (k - 1) * (Lket + 1)][
                             indexlm1xyz][
                             m + 1];

                // if (std::abs(ERIscratch) > 1.0e-10) {
                //  std::cout << "Line 178. ERIscratch = " << ERIscratch << 
                //    std::endl;
                // }
                // Equation 6. First line HGP

                if (lA_xyz[iwork] > 1) {

                  for (int ii = 0; ii < 3; ii++) {

                    lAtemp[ii] = lA_xyz[ii];

                  } // for (int ii = 0; ii < 3; ii++)

                  lAtemp[iwork] = lA_xyz[iwork] - 2;

                  int indexlm2xyz = indexmap(k - 2, lAtemp[0], lAtemp[1], 
                                             lAtemp[2]);
                  ERIscratch = ERIscratch + 1 / (2 * zeta) * (lA_xyz[iwork] - 1) 
                               * (Vtempbraket[
                                   (k - 2) * (Lket + 1)][
                                   indexlm2xyz][
                                   m]
                               - rho / zeta 
                               * Vtempbraket[
                               (k - 2) * (Lket + 1)][
                               indexlm2xyz][
                               m + 1]);

                  // if (std::abs(ERIscratch) > 1.0e-10) { 
                  //   std::cout << "Line 197. ERIscratch = " << ERIscratch << 
                  //     std::endl;
                  // }
                  // Equation 6. Second line HGP

                } // if (lA_xyz[iwork] > 1)

                Vtempbraket[k * (Lket + 1)][cart_i][m] = ERIscratch;

              } // for (int m = 0; m <= mbra; m++) 

            } // if (k > 0)

            for (int l = 0; l <= Lket; l++) {
            
              for (int cart_j = 0; cart_j < cart_ang_list[l].size(); cart_j++) 
              {
              
                int lC_xyz[3];

                for (int ii = 0; ii < 3; ii++) {
                
                  lC_xyz[ii] = cart_ang_list[l][cart_j][ii];

                } // for (int ii = 0; ii < 3; ii++)

                int mbraket = L - k - l;

                if (l > 0) {

                  int iwork;

                  if (lC_xyz[0] > 0) {

                    iwork = 0;

                  } // if (lC_xyz[0] > 0)

                  else if (lC_xyz[1] > 0) {

                    iwork = 1;

                  } // else if (lC_xyz[1] > 0)

                  else if (lC_xyz[2] > 0) {

                    iwork = 2;

                  } // else if (lC_xyz[2] > 0)

                  // Calculate the index of a lower angular momentum integral
                  int lCtemp[3];

                  for (int ii = 0; ii < 3; ii++) {

                    lCtemp[ii] = lC_xyz[ii];

                  } // for (int ii = 0; ii < 3; ii++)

                  // For ket side. Equation 6. HGP
                  lCtemp[iwork] = lC_xyz[iwork] - 1;

                  int indexlm1xyz = indexmap(l - 1, lCtemp[0], lCtemp[1], 
                                             lCtemp[2]);

                  // Calculate indeix of lCm1
                  for (int m = 0; m <= mbraket; m++) {

                    double ERIscratch = 0.0;

                    ERIscratch = (Q[iwork] - C[iwork]) 
                                 * Vtempbraket[
                                 k * (Lket + 1) + (l - 1)][
                                 (cart_i) * cart_ang_list[l - 1].size() 
                                 + indexlm1xyz][
                                 m] 
                                 + (W[iwork] - Q[iwork]) 
                                 * Vtempbraket[
                                 k * (Lket + 1) + (l - 1)][
                                 cart_i * cart_ang_list[l - 1].size() 
                                 + indexlm1xyz][
                                 m + 1];

                    // if (std::abs(ERIscratch) > 1.0e-10) { 
                    //   std::cout << "Line 264. ERIscratch = " << ERIscratch << 
                    //     std::endl;
                    // }

                    if (lC_xyz[iwork] > 1) {

                      for (int ii = 0; ii < 3; ii++) {

                        lCtemp[ii] = lC_xyz[ii];

                      } // for (int ii = 0; ii < 3; ii++)

                      lCtemp[iwork] = lC_xyz[iwork] - 2;

                      int indexlm2xyz = indexmap(l - 2, lCtemp[0], lCtemp[1], 
                                                 lCtemp[2]);
                      ERIscratch = ERIscratch 
                                   + 1 / (2 * eta) * (lC_xyz[iwork] - 1) 
                                   * (Vtempbraket[
                                   k * (Lket + 1) + (l - 2)][
                                   cart_i * cart_ang_list[l - 2].size() 
                                   + indexlm2xyz][m] 
                                   - rho / eta 
                                   * Vtempbraket[
                                   k * (Lket + 1) + (l - 2)][
                                   cart_i * cart_ang_list[l - 2].size() 
                                   + indexlm2xyz][
                                   m + 1]);

                      // if (std::abs(ERIscratch) > 1.0e-10) { 
                      //   std::cout << "Line 277. ERIscratch = " << ERIscratch << 
                      //     std::endl;
                      // }

                    } // if (lC_xyz[iwork] > 1)

                    if (lA_xyz[iwork] > 0) {

                      int lAtemp[3];

                      for (int ii = 0; ii < 3; ii++) {

                        lAtemp[ii] = lA_xyz[ii];

                      } // for (int ii = 0; ii < 3; ii++)

                      lAtemp[iwork] -= 1; 

                      int indexlAm1 = indexmap(k - 1, lAtemp[0], lAtemp[1], 
                                               lAtemp[2]);
                      ERIscratch += 1.0 / (2.0 * zetaG) * lA_xyz[iwork]
                                    * Vtempbraket[
                                    (k - 1) * (Lket + 1) + (l - 1)][
                                    (indexlAm1) * cart_ang_list[l - 1].size() 
                                    + indexlm1xyz][
                                    m + 1];

                      // if (std::abs(ERIscratch) > 1.0e-10) {
                      //   std::cout << "Line 299. ERIscratch = " << ERIscratch << 
                      //     std::endl;
                      // }

                    } // if (lA_xyz[iwork] > 0)

                    Vtempbraket[k * (Lket + 1) + l][
                      cart_i * cart_ang_list[l].size() + cart_j][
                      m] 
                      = ERIscratch;

                  } // for (int m = 0; m <= mbraket; m++)

                } // if (l > 0)

              } // for (int cart_j = 0; cart_j < cart_ang_list[l].size(); 
                // cart_j++)

            } // for (int l = 0; l <= Lket; l++)

          } // for (int cart_i = 0; cart_i < cart_ang_list[k].size(); cart_i++)

        } // for (int k = 0; k <= Lbra; k++)
        
        // Start Contraction Here
        // Vbracket[]+= Vtempbraket[]
        for (int lA = LA; lA <= Lbra; lA++) {

          for (int lC = LC; lC <= Lket; lC++) {

            for (int i = 0; i < cart_ang_list[lA].size(); i++) {

              for (int k = 0; k < cart_ang_list[lC].size(); k++) {

                Vbraket[(lA - LA) * (Lket - LC + 1) + (lC - LC)][0][
                  i * cart_ang_list[lC].size() + k] 
                  += Vtempbraket[
                  lA * (Lket + 1) + lC][i * cart_ang_list[lC].size() + k][0];

                // if (std::abs(Vbraket[(lA - LA) * (Lket - LC + 1) + lC - LC][0][i * cart_ang_list[lC].size() + k]) > 1.0e-10) { 
                //   std::cout << "Line 322. Vbraket = " << Vbraket[(lA - LA) * (Lket - LC + 1) + lC - LC][0][i * cart_ang_list[lC].size() + k] << std::endl;
                // }

              } // for (int k = 0; k <= cart_ang_list[lC].size(); k++)

            } // for (int i = 0; i <= cart_ang_list[lA].size(); i++)

          } // for (int lC = LC; lC <= Lket; lC++)

        } // for (int lA = LA; lA <= Lbra; lA++)

      } // for ( auto &pripair2 : pair2.primpairs )   

    }  // for ( auto &pripair1 : pair1.primpairs )

    for (int lB = 1; lB <= LB; lB++) {

      // Implies LB > 0
      for (int lA = LA; lA <= Lbra - lB; lA++) {

        // Loop over (lA||lB)
        // Loop over elements in lA
        for (int Aidx = 0; Aidx < cart_ang_list[lA].size(); Aidx++) {

          int lA_xyz[3];

          for (int ii = 0; ii < 3; ii++) {

            lA_xyz[ii] = cart_ang_list[lA][Aidx][ii];

          } // for (int ii = 0; ii < 3; ii++)

          for (int Bidx = 0; Bidx < cart_ang_list[lB].size(); Bidx++) {

            int lB_xyz[3];

            for (int ii = 0; ii < 3; ii++) {

              lB_xyz[ii] = cart_ang_list[lB][Bidx][ii];

            } // for (int ii = 0; ii < 3; ii++)

            int iwork;

            if (lB_xyz[0] > 0) { // Means lA_x > 0

              iwork = 0;

            } // if (lB_xyz[0] > 0)

            else if (lB_xyz[1] > 0) { // Means lA_y > 0

              iwork = 1;

            } // else if (lB_xyz[1] > 0)

            else if (lB_xyz[2] > 0) { // Means lA_z > 0

              iwork = 2;

            } // else if (lB_xyz[2] > 0)

            int lBm1[3];

            for (int ii = 0; ii < 3; ii++) {

              lBm1[ii] = lB_xyz[ii];

            } // for (int ii = 0; ii < 3; ii++)

            lBm1[iwork] = lB_xyz[iwork] - 1;

            int lAp1[3];

            for (int ii = 0; ii < 3; ii++) {

              lAp1[ii] = lA_xyz[ii];

            } // for (int ii = 0; ii < 3; ii++)

            lAp1[iwork] = lA_xyz[iwork] + 1;

            int idxBtemp = indexmap(lB - 1, lBm1[0], lBm1[1], lBm1[2]);
            int lD = 0;

            for (int lC = LC; lC <= Lket; lC++) {

              for (int Cidx = 0; Cidx < cart_ang_list[lC].size(); Cidx++) {

                int lC_xyz[3];

                for (int ii = 0; ii < 3; ii++) {

                  lC_xyz[ii] = cart_ang_list[lC][Cidx][ii];

                } // for (int ii = 0; ii < 3; ii++)

                Vbraket[(lA - LA) * (Lket - LC + 1) + (lC - LC)][
                  lB * (Lket - lC + 1)][
                  Aidx * cart_ang_list[lB].size() * cart_ang_list[lC].size() 
                  + Bidx * cart_ang_list[lC].size()
                  + Cidx] 
                  = Vbraket[(lA - LA + 1) * (Lket - LC + 1) + (lC - LC)][
                  (lB - 1) * (Lket - lC + 1)][
                  indexmap(lA + 1, lAp1[0], lAp1[1], lAp1[2]) 
                  * cart_ang_list[lB - 1].size() * cart_ang_list[lC].size() 
                  + idxBtemp * cart_ang_list[lC].size() + Cidx] 
                  + (AB[iwork]) 
                  * Vbraket[(lA - LA) * (Lket - LC + 1) + lC - LC][
                  (lB - 1) * (Lket - lC + 1)][
                  Aidx * cart_ang_list[lB - 1].size() 
                       * cart_ang_list[lC].size() 
                  + idxBtemp * cart_ang_list[lC].size() 
                  + Cidx];

                // if (std::abs(Vbraket[(lA - LA) * (Lket - LC + 1) + lC - LC][lB * (Lket - lC + 1)][Aidx * cart_ang_list[lB].size() * cart_ang_list[lC].size() + Bidx * cart_ang_list[lC].size() + Cidx]) > 1.0e-10) { 
                //   std::cout << "Line 377. Vbraket = " << Vbraket[(lA - LA) * (Lket - LC + 1) + lC - LC][lB * (Lket - lC + 1)][Aidx * cart_ang_list[lB].size() * cart_ang_list[lC].size() + Bidx * cart_ang_list[lC].size() + Cidx] << std::endl;
                // }

              } // for (int Cidx = 0; Cidx < cart_ang_list[lC]; Cidx++)

            } // for (int lC = LC; lC <= Lket; lC++)

          } // for (int Bidx = 0; Bidx < cart_ang_list[lB].size(); Bidx++)

        } // for (int Aidx = 0; Aidx < cart_ang_list[lA].size(); Aidx++)

      } // for (int lA = LA; lA <= Lbra - lB; lB++)

    } // for (int lB = 1; lB <= LB; lB++)

    // Horizontal of bra side finished
    // Horizontal of ket side start
    for (int lD = 1; lD <= LD; lD++) {

      // Implies LD > 0
      for (int lC = LC; lC <= Lket - lD; lC++) {

        // Loop over (lC||lD)
        // Loop over elements in lC
        for (int Cidx = 0; Cidx < cart_ang_list[lC].size(); Cidx++) {

          // Calculate the lC_x, lC_y, lC_z
          int lC_xyz[3];

          for (int ii = 0; ii < 3; ii++) {

            lC_xyz[ii] = cart_ang_list[lC][Cidx][ii];

          } // for (int ii = 0; ii < 3; ii++)
          // Loop over elements in lD

          for (int Didx = 0; Didx < cart_ang_list[lD].size(); Didx++) {

            int lD_xyz[3];

            for (int ii = 0; ii < 3; ii++) {

              lD_xyz[ii] = cart_ang_list[lD][Didx][ii];

            } // for (int ii = 0; ii < 3; ii++)

            int iwork;

            if (lD_xyz[0] > 0) { // Means lC_x > 0

              iwork = 0;

            } // if (lD_xyz[0] > 0)

            else if (lD_xyz[1] > 0) { // Means lC_y > 0

              iwork = 1;

            } // else if (lD_xyz[1] > 0)

            else if (lD_xyz[2] > 0) { // Means lC_z > 0

              iwork = 2;

            } // else if (lD_xyz[2] > 0)

            int lDm1[3];

            for (int ii = 0; ii < 3; ii++) {

              lDm1[ii] = lD_xyz[ii];

            } // for (int ii = 0; ii < 3; ii++)

            lDm1[iwork] = lD_xyz[iwork] - 1;

            int lCp1[3];

            for (int ii = 0; ii < 3; ii++) {

              lCp1[ii] = lC_xyz[ii];

            } // for (int ii = 0; ii < 3; ii++)

            lCp1[iwork] = lC_xyz[iwork] + 1;

            int idxDtemp = indexmap(lD - 1, lDm1[0], lDm1[1], lDm1[2]);

            for (int Aidx = 0; Aidx < cart_ang_list[LA].size(); Aidx++) {

              // Calculate the lA_x, lA_y, lA_z
              int lA_xyz[3];

              for (int ii = 0; ii < 3; ii++) {

                lA_xyz[ii] = cart_ang_list[LA][Aidx][ii];

              } // for (int ii = 0; ii < 3; ii++)
              for (int Bidx = 0; Bidx < cart_ang_list[LB].size(); Bidx++) {

                int lB_xyz[3];

                for (int ii = 0; ii < 3; ii++) {

                  lB_xyz[ii] = cart_ang_list[LB][Bidx][ii];

                } // for (int ii = 0; ii < 3; ii++)

                Vbraket[lC - LC][LB * (Lket - lC + 1) + lD][
                  Aidx * cart_ang_list[LB].size() 
                  * cart_ang_list[lC].size() 
                  * cart_ang_list[lD].size() 
                  + Bidx * cart_ang_list[lC].size()* cart_ang_list[lD].size() 
                  + Cidx * cart_ang_list[lD].size()
                  + Didx] 
                  = Vbraket[lC - LC + 1][
                  LB * (Lket - (lC + 1) + 1) + (lD - 1)][
                  Aidx * cart_ang_list[LB].size() 
                    * cart_ang_list[lC + 1].size() 
                    * cart_ang_list[lD - 1].size() 
                  + Bidx * cart_ang_list[lC + 1].size() 
                    * cart_ang_list[lD - 1].size() 
                  + indexmap(lC + 1, lCp1[0], lCp1[1], lCp1[2]) 
                    * cart_ang_list[lD - 1].size() 
                  + idxDtemp] 
                  + (CD[iwork]) * Vbraket[lC - LC][
                  LB * (Lket - lC + 1) + (lD - 1)][
                  Aidx * cart_ang_list[LB].size() 
                  * cart_ang_list[lC].size() 
                  * cart_ang_list[lD - 1].size() 
                  + Bidx * cart_ang_list[lC].size() 
                  * cart_ang_list[lD - 1].size() 
                  + Cidx * cart_ang_list[lD - 1].size() 
                  + idxDtemp];

                // if (std::abs(Vbraket[lC - LC][LB * (Lket - lC + 1) + lD][Aidx * cart_ang_list[LB].size() * cart_ang_list[lC].size() * cart_ang_list[lD].size() + Bidx * cart_ang_list[lC].size() * cart_ang_list[lD].size() + Cidx * cart_ang_list[lD].size() + Didx]) > 1.0e-10) { 
                //   std::cout << "Line 449. Vbraket = " << Vbraket[lC - LC][LB * (Lket - lC + 1) + lD][Aidx * cart_ang_list[LB].size() * cart_ang_list[lC].size() * cart_ang_list[lD].size() + Bidx * cart_ang_list[lC].size() * cart_ang_list[lD].size() + Cidx * cart_ang_list[lD].size() + Didx] << std::endl;
                // }

              } // for (int Bidx = 0; Bidx < cart_ang_list[LB].size(); Bidx++)

            } // for (int Aidx = 0; Aidx < cart_ang_list[LA].size(); Aidx++)

          } // for (int Didx = 0; Didx < cart_ang_list[lD].size(); Didx++)

        } // for (int Cidx = 0; Cidx < cart_ang_list[lC].size(); Cidx++)

      } // for (int lC = LC; LC <= Lket - lD; lC++)

    } // for (int lD = 1; lD <= LD; lD++)

    if ( ( not shell1.contr[0].pure ) and ( not shell2.contr[0].pure ) and 
         ( not shell3.contr[0].pure ) and ( not shell4.contr[0].pure ) ) {  
      // if both sides are cartesian, return cartesian gaussian integrals
      return Vbraket[0][LB * (LD + 1) + LD];
    }

    std::vector<double> ERI_sph;

    ERI_sph.assign(((2*shell1.contr[0].l+1)*(2*shell2.contr[0].l+1)
                   *(2*shell3.contr[0].l+1)*(2*shell4.contr[0].l+1)),0.0); 

    cart2sph_2e_transform(shell1.contr[0].l,shell2.contr[0].l,
                          shell3.contr[0].l,shell4.contr[0].l,
                          ERI_sph,Vbraket[0][LB * (LD + 1) + LD]);

    return ERI_sph; 



  }  // BottomupHGP

  int indexmap(int L, int x, int y, int z) {
   int a = L - x;
   int b = a - y;
   int indexinshell = a*(a+1)/2+b;

   return indexinshell;

  }

  /**
   *  \brief Compute the ERI of two shell pairs
   *
   *
   *  \param [in] pair1  bra shell pair data for shell1,shell2
   *  \param [in] pair2  ket shell pair data for shell3,shell4
   *  \param [in] shell1
   *  \param [in] shell2
   *  \param [in] shell3
   *  \param [in] shell4
   *
   *  \return ERI of two shell pairs ( shell1 shell2 | shell3 shell4 )
   */
   
  std::vector<double> RealGTOIntEngine::computeERIabcd(libint2::ShellPair &pair1 ,
    libint2::ShellPair &pair2, libint2::Shell &shell1, libint2::Shell &shell2,
    libint2::Shell &shell3, libint2::Shell &shell4)  {
    
    double tmpVal=0.0,sqrPQ,PQ;
    std::vector<double> ERI_cart;
    
    int lA[3],lB[3],lC[3],lD[3];

    // compute total angular momentum
    auto lTotal = shell1.contr[0].l + shell2.contr[0].l
                + shell3.contr[0].l + shell4.contr[0].l;

/*
    std::cerr<<"LA "<<shell1.contr[0].l
    <<" LB "<<shell2.contr[0].l
    <<" LC "<<shell3.contr[0].l 
    <<" LD "<<shell4.contr[0].l<<std::endl;
*/

    // pre calculate all the Boys functions 
    // dimension is FmT_2e[shellpair1.prim][shellpair2.prim][lTotal+1]
    std::vector<std::vector<std::vector<double>>> FmT_2e;
    FmT_2e.resize(pair1.primpairs.size());

    double *FmT = new double[lTotal+1];
    int shellpair1_i=0, shellpair2_j ;
    for ( auto &pripair1 : pair1.primpairs ) {
      FmT_2e[shellpair1_i].resize( pair2.primpairs.size() );

      shellpair2_j = 0 ; 
      for ( auto &pripair2 : pair2.primpairs ) {
        sqrPQ = 0.0;
        for ( int mu=0 ; mu<3 ; mu++ ) {
          PQ = ( pripair1.P[mu]-pripair2.P[mu] ); 
          sqrPQ += PQ*PQ;
        }
        auto Zeta = 1.0/pripair1.one_over_gamma;
        auto Eta  = 1.0/pripair2.one_over_gamma;
        
        auto rho = Zeta*Eta/(Zeta+Eta);
        auto T = rho*sqrPQ;
        // calculate Fm(T) list
        computeFmTTaylor( FmT, T, lTotal, 0 );

        for ( int lcurr = 0 ; lcurr < lTotal+1 ; lcurr++ ) {
           if ( std::abs(FmT[lcurr]) < 1.0e-15 ) 
             FmT_2e[shellpair1_i][shellpair2_j].push_back(0.0);
           else
             FmT_2e[shellpair1_i][shellpair2_j].push_back(FmT[lcurr]);
        } // for lcurr
        shellpair2_j ++;
      } // for pripair2
    shellpair1_i++;
    } // for pripair1
    delete[] FmT;

    for(int i = 0; i < cart_ang_list[shell1.contr[0].l].size(); i++) 
    for(int j = 0; j < cart_ang_list[shell2.contr[0].l].size(); j++)
    for(int k = 0; k < cart_ang_list[shell3.contr[0].l].size(); k++)
    for(int l = 0; l < cart_ang_list[shell4.contr[0].l].size(); l++) {
      for (int mu=0 ; mu<3 ; mu++) {
        lA[mu] = cart_ang_list[shell1.contr[0].l][i][mu];
        lB[mu] = cart_ang_list[shell2.contr[0].l][j][mu];
        lC[mu] = cart_ang_list[shell3.contr[0].l][k][mu];
        lD[mu] = cart_ang_list[shell4.contr[0].l][l][mu];
      }  // for mu

      
      tmpVal = twoehRRabcd(pair1,pair2,shell1,shell2,shell3,shell4,
                 FmT_2e,shell1.contr[0].l, lA, shell2.contr[0].l, lB, 
                 shell3.contr[0].l, lC, shell4.contr[0].l, lD ); 
      
      ERI_cart.push_back(tmpVal);
    }   // for l


    if ( ( not shell1.contr[0].pure ) and ( not shell2.contr[0].pure ) and 
         ( not shell3.contr[0].pure ) and ( not shell4.contr[0].pure ) ) {  
      // if both sides are cartesian, return cartesian gaussian integrals
      return ERI_cart;
    }

    std::vector<double> ERI_sph;

    ERI_sph.assign(((2*shell1.contr[0].l+1)*(2*shell2.contr[0].l+1)
                   *(2*shell3.contr[0].l+1)*(2*shell4.contr[0].l+1)),0.0); 

    cart2sph_2e_transform( shell1.contr[0].l,shell2.contr[0].l,
      shell3.contr[0].l,shell4.contr[0].l,ERI_sph,ERI_cart );

    return ERI_sph; 

  }   // computeERIabcd


  /**
   *  \brief horizontal recursion of ERI when all angular momentum are nonzero
   *
   *
   *  \param [in] pair1  bra shell pair data for shell1,shell2
   *  \param [in] pair2  ket shell pair data for shell3,shell4
   *  \param [in] shell1
   *  \param [in] shell2
   *  \param [in] shell3
   *  \param [in] shell4
   *  \param [in] FmT_2e Boys function between two shell pairs
   *  \param [in] LA
   *  \param [in] lA
   *  \param [in] LB
   *  \param [in] lB
   *  \param [in] LC
   *  \param [in] lC
   *  \param [in] LD
   *  \param [in] lD
   *
   *  \return ERI of two shell pairs ( shell1 shell2 | shell3 shell4 )
   */
  //----------------------------------------------------//
  // two-e horizontal recursion from (ab|cd) to (a0|cd) //
  // (ab|cd)=(a+1,b-1|cd)+(A-B)*(a,b-1|cd)              //
  //----------------------------------------------------//
   
  double RealGTOIntEngine::twoehRRabcd( 
     libint2::ShellPair &pair1 ,libint2::ShellPair &pair2 ,
     libint2::Shell &shell1, libint2::Shell &shell2,
     libint2::Shell &shell3, libint2::Shell &shell4,
     std::vector<std::vector<std::vector<double>>> &FmT_2e,
     int LA,int *lA,int LB,int *lB,int LC,int *lC,int LD,int *lD) {

     double tmpVal = 0.0, tmpVal1=0.0;

  // iWork is used to indicate which Cartesian angular momentum we are reducing (x,y,z)

    int iWork;
    int totalL = LA + LB + LC + LD;


    if(totalL==0) {   // (SS||SS) type 

      return twoeSSSS0(pair1,pair2,shell1,shell2,shell3,shell4);

      } else { // when totalL! = 0

        if( LB>=1 ) {

          int lAp1[3],lBm1[3];
          for( iWork=0 ; iWork<3 ; iWork++ ){
            lAp1[iWork]=lA[iWork];     
            lBm1[iWork]=lB[iWork];     
          } // for iWork

          if (lB[0]>0)      iWork=0;   
          else if (lB[1]>0) iWork=1;
          else if (lB[2]>0) iWork=2;
          lAp1[iWork]+=1;
          lBm1[iWork]-=1;

          tmpVal += twoehRRabcd(pair1,pair2,shell1,shell2,shell3,shell4,
                      FmT_2e, LA+1,lAp1,LB-1,lBm1,LC,lC,LD,lD);

          if ( std::abs(pair1.AB[iWork]) > 1.0e-15 ) {
            tmpVal += pair1.AB[iWork]*twoehRRabcd(pair1,pair2,shell1,shell2,shell3,
                        shell4,FmT_2e, LA,lA,LB-1,lBm1,LC,lC,LD,lD);
          } 

        } else if ( LB == 0 ) {
          tmpVal = twoehRRa0cd(pair1,pair2,shell1,shell2,shell3,shell4,FmT_2e,
                                 LA,lA,LC,lC,LD,lD);

        } // LB == 0
      }  // else ( LTOTAL != 0 )

      return tmpVal;

  }  // twoehRRabcd

  /**
   *  \brief horizontal recursion of ERI when all LB=0
   *
   *
   *  \param [in] pair1  bra shell pair data for shell1,shell2
   *  \param [in] pair2  ket shell pair data for shell3,shell4
   *  \param [in] shell1
   *  \param [in] shell2
   *  \param [in] shell3
   *  \param [in] shell4
   *  \param [in] FmT_2e Boys function between two shell pairs
   *  \param [in] LA
   *  \param [in] lA
   *  \param [in] LC
   *  \param [in] lC
   *  \param [in] LD
   *  \param [in] lD
   *
   *  \return ERI of two shell pairs ( shell1 shell2 | shell3 shell4 )
   */
  //----------------------------------------------------//
  // two-e horizontal recursion from (a0|cd) to (a0|c0) //
  // (a0|cd)=(a,0|c+1,d-1)+(C-D)*(a,0|c,d-1)            //
  //----------------------------------------------------//
   
  double RealGTOIntEngine::twoehRRa0cd(
    libint2::ShellPair &pair1, libint2::ShellPair &pair2,
    libint2::Shell &shell1, libint2::Shell &shell2,
    libint2::Shell &shell3, libint2::Shell &shell4, 
    std::vector<std::vector<std::vector<double>>> &FmT_2e,
    int LA,int *lA,int LC,int *lC,int LD,int *lD)  {

    double tmpVal=0.0;
    if(LD==0) {
      int pair1index=0, pair2index=0;
      // go into the vertical recursion
      for ( auto &pripair1 : pair1.primpairs ) {
        pair2index = 0; 
        for ( auto &pripair2 : pair2.primpairs ) {

          auto norm = 
                 shell1.contr[0].coeff[pripair1.p1]* 
                 shell2.contr[0].coeff[pripair1.p2]* 
                 shell3.contr[0].coeff[pripair2.p1]* 
                 shell4.contr[0].coeff[pripair2.p2];  

          tmpVal +=  norm * twoevRRa0c0( pripair1, pripair2,  
             FmT_2e[pair1index][pair2index], shell1,shell3, 0, LA,lA,LC,lC);

          pair2index++;
        } // for pripair2
        pair1index++;
      } // for pripair1
    } else { // if LD>0, go into horizontal recursion 

      int iWork;
      int lCp1[3],lDm1[3];  
     
      for( int iWork=0 ; iWork<3 ; iWork++ ){
        lCp1[iWork]=lC[iWork];
        lDm1[iWork]=lD[iWork];
      }
     
      if (lD[0]>0) iWork=0;
      else if (lD[1]>0) iWork=1;
      else if (lD[2]>0) iWork=2;
     
      lCp1[iWork]+=1;


    // when LD > 0

      lDm1[iWork] -=1 ;
      tmpVal = twoehRRa0cd(pair1, pair2, shell1, shell2, shell3, shell4, 
                             FmT_2e,LA,lA, LC+1,lCp1, LD-1,lDm1 ); 
      if ( std::abs(pair2.AB[iWork]) > 1.0e-15 ){
        tmpVal += pair2.AB[iWork] * twoehRRa0cd( pair1, pair2, shell1, shell2, 
               shell3, shell4, FmT_2e, LA,lA, LC,lC, LD-1,lDm1 );
      }

    } // else ( that means LD > 0 )
    return tmpVal;
  }  // twoehRRa0cd


  /**
   *  \brief vertical recursion of ERI when all LA, LC > 0
   *
   *
   *  \param [in] pripair1  primitive bra shell pair data for shell1,shell2
   *  \param [in] pripair2  primitive ket shell pair data for shell3,shell4
   *  \param [in] pair1index index of primitive shell pair among the contracted pair
   *  \param [in] pair2index index of primitive shell pair among the contracted pair
   *  \param [in] FmT_2e Boys function between two primitive shell pairs
   *  \param [in] shell1    
   *  \param [in] shell3    
   *  \param [in] m         order of auxiliary function
   *  \param [in] LA
   *  \param [in] lA
   *  \param [in] LC
   *  \param [in] lC
   *
   *  \return ERI of two primitive shell pairs ( shell1 shell2 | shell3 shell4 )
   */

//---------------------------------------------------------------//
// two-e vertical recursion from [a0|c0] to [a0|00]              //
// [a0|c0]^m = (Q-C)*[a0|c-1,0]^m                                //
//           + (W-Q)*[a0|c-1,0]^(m+1)                            //
//           + N(a)/(2*(zeta+eta))*[a-1,0|c-1,0]^(m+1)           //
//           + (N(c)-1)/(2*eta)*[a0|c-2,0]^m                     //
//           - (N(c)-1)/(2*eta)*zeta/(zeta+eta)*[a0|c-2,0]^(m+1) //
//---------------------------------------------------------------//
   
  double RealGTOIntEngine::twoevRRa0c0(
    libint2::ShellPair::PrimPairData &pripair1,
    libint2::ShellPair::PrimPairData &pripair2, 
    std::vector<double> &FmT_2epri, 
    libint2::Shell &shell1, libint2::Shell &shell3,
    int m, int LA, int *lA, int LC, int *lC ) {
 
 
    if(LC==0) return twoevRRa000( pripair1, pripair2, FmT_2epri,
                                  shell1, m, LA, lA );
 
    int lAm1[3],lCm1[3];  
 
    for ( int iWork=0 ; iWork<3 ; iWork++ ){
      lAm1[iWork]=lA[iWork];     
      lCm1[iWork]=lC[iWork];
    }
 
    double tmpVal=0.0;
    double W_iWork,Zeta;
    int iWork;
 
    if (lC[0]>0) iWork=0;
    else if (lC[1]>0) iWork=1;
    else if (lC[2]>0) iWork=2;

    lCm1[iWork]-=1;

    if ( std::abs(pripair2.P[iWork] - shell3.O[iWork]) > 1.0e-15 ) {
      tmpVal += ( pripair2.P[iWork] - shell3.O[iWork] ) * 
                twoevRRa0c0( pripair1, pripair2, FmT_2epri, 
                  shell1, shell3, m, LA,lA, LC-1,lCm1 );
    }  // if (Q-C) > 1.0e-15 


    Zeta = 1.0/pripair1.one_over_gamma + 1.0/pripair2.one_over_gamma; 
/*
      for ( int mu = 0 ; mu<3 ; mu++ )
      W[mu] = (pripair1.P[mu]/pripair1.one_over_gamma 
              + pripair2.P[mu]/pripair2.one_over_gamma )/Zeta ; 
*/

    W_iWork = ( pripair1.P[iWork]/pripair1.one_over_gamma + 
               pripair2.P[iWork]/pripair2.one_over_gamma )/Zeta;

    if( std::abs( W_iWork- pripair2.P[iWork] ) > 1.0e-15 ) {
      tmpVal += ( W_iWork- pripair2.P[iWork] ) * twoevRRa0c0( pripair1, pripair2, 
                 FmT_2epri, shell1,shell3, m+1, LA,lA, LC-1,lCm1 );
    } // if( abs( W_iWork- Q[iWork] ) > 1.0e-15 )

    if (lA[iWork]>0) {

      lAm1[iWork] -= 1;
      tmpVal += (lAm1[iWork]+1) / (2.0*Zeta) * twoevRRa0c0( pripair1, pripair2, 
                 FmT_2epri, shell1, shell3, m+1, LA-1,lAm1, LC-1,lCm1 );
    } // if (lA[iWork]>0) 

    if ( lC[iWork]>=2 ){

      lCm1[iWork] -=1; // right now lCm1(iWork) = lC[iWork]-2 
      tmpVal += 0.5 * (lCm1[iWork]+1) * pripair2.one_over_gamma * 
        ( twoevRRa0c0( pripair1, pripair2, FmT_2epri, 
                       shell1, shell3, m, LA,lA, LC-2,lCm1 )

               - ( 1.0/pripair1.one_over_gamma )/Zeta 
          * twoevRRa0c0( pripair1, pripair2, FmT_2epri, 
                         shell1, shell3, m+1, LA,lA, LC-2,lCm1 ) );
    } // if ( lC[iWork]>=2 )

    return tmpVal;
  }  // twoevRRa0c0



  /**
   *  \brief vertical recursion of ERI when all LA > 0, all the others are 0
   *
   *
   *  \param [in] pripair1  primitive bra shell pair data for shell1,shell2
   *  \param [in] pripair2  primitive ket shell pair data for shell3,shell4
   *  \param [in] pair1index index of primitive shell pair among the contracted pair
   *  \param [in] pair2index index of primitive shell pair among the contracted pair
   *  \param [in] FmT_2e Boys function between two primitive shell pairs
   *  \param [in] shell1    
   *  \param [in] m         order of auxiliary function
   *  \param [in] LA
   *  \param [in] lA
   *
   *  \return ERI of two primitive shell pairs ( shell1 shell2 | shell3 shell4 )
   */
//---------------------------------------------------------------//
// two-e vertical recursion from [a0|00] to [00|00]              //
// [a0|00]^m = (P-A)*[a-1,0|00]^m                                //
//           + (W-P)*[a-1,0|00]^(m+1)                            //
//           + (N(a)-1)/(2*zeta)*[a-2,0|00]^m                    //
//           - (N(a)-1)/(2*zeta)*eta/(zeta+eta)*[a-2,0|00]^(m+1) //
//---------------------------------------------------------------//
   
  double RealGTOIntEngine::twoevRRa000(
    libint2::ShellPair::PrimPairData &pripair1,
    libint2::ShellPair::PrimPairData &pripair2, std::vector<double> &FmT_2epri,
    libint2::Shell &shell1, int m,int LA,int *lA ) {


    if(LA==0) {
      // calculate the (SS||SS) integral  	
      double expoT,Kab,Kcd,SSSS=0.0 ;
 
      // zeta+eta is expoT
      expoT = 1.0/pripair1.one_over_gamma + 1.0/pripair2.one_over_gamma;  
 
      Kab = sqrt(2.0)* pow(M_PI,1.25) * pripair1.K; 
      Kcd = sqrt(2.0)* pow(M_PI,1.25) * pripair2.K;
 
      SSSS += Kab * Kcd * FmT_2epri[m] / sqrt(expoT); 
 
      return SSSS;
 
    } // if LA==0
 
    // here LA != 0
    double tmpVal=0.0,W[3],Zeta;
    int iWork;
    int lAm1[3];
 
    for( iWork=0 ; iWork<3 ; iWork++ ) lAm1[iWork]=lA[iWork];
 
    if (lA[0]>0) iWork=0;
    else if (lA[1]>0) iWork=1;
    else if (lA[2]>0) iWork=2;
 
    if( LA>=1 ) {
 
      lAm1[iWork]-=1;

      Zeta = 1.0/pripair1.one_over_gamma + 1.0/pripair2.one_over_gamma;
      for ( int mu = 0 ; mu<3 ; mu++ )
        W[mu] = (pripair1.P[mu]/pripair1.one_over_gamma 
              + pripair2.P[mu]/pripair2.one_over_gamma )/Zeta ; 
 
      if ( std::abs( W[iWork]- pripair1.P[iWork] ) > 1.0e-15 ) {
 
        tmpVal += ( W[iWork]- pripair1.P[iWork] ) * twoevRRa000( pripair1, pripair2,
          FmT_2epri, shell1, m+1, LA-1,lAm1 );
 
      }  // if ( abs( W_iWork-P[iWork] )>1.0e-15 )

      if ( std::abs( pripair1.P[iWork] - shell1.O[iWork] )>1.0e-15 ) {  
        tmpVal+= ( pripair1.P[iWork] - shell1.O[iWork] ) * twoevRRa000( pripair1, 
          pripair2, FmT_2epri, shell1, m, LA-1,lAm1 );
      } // if ( abs( P[iWork]-A[iWork] )>1.0e-15 ) 

      if ( lA[iWork]>=2 ) {

        lAm1[iWork] -=1; // now lAm1[iWork] == lA[iWork]-2
        tmpVal += 0.5 * ( lAm1[iWork]+1 ) * pripair1.one_over_gamma  
                  *(twoevRRa000( pripair1, pripair2, FmT_2epri,
                    shell1, m, LA-2,lAm1 )
                - 1.0/(pripair2.one_over_gamma*Zeta) * twoevRRa000( pripair1, pripair2,
                  FmT_2epri, shell1, m+1, LA-2,lAm1 ) );
      } // if lA[iWork]>=2

    } // if( LA>=1 ) 

    return tmpVal;

  };  // twoevRRa000



  /**
   *  \brief vertical recursion of ERI when all the angular momentum are 0
   *
   *
   *  \param [in] pair1   bra shell pair data for shell1,shell2
   *  \param [in] pair2   ket shell pair data for shell3,shell4
   *  \param [in] shell1    
   *  \param [in] shell2    
   *  \param [in] shell3    
   *  \param [in] shell4    
   *
   *  \return ERI of two primitive shell pairs ( shell1 shell2 | shell3 shell4 )
   */
   
  double RealGTOIntEngine::twoeSSSS0(
    libint2::ShellPair &pair1, libint2::ShellPair &pair2,
    libint2::Shell &shell1, libint2::Shell &shell2,
    libint2::Shell &shell3, libint2::Shell &shell4 ) {

    // in this auxiliary integral, m=0 
    double norm,sqrPQ,PQ,expoT,T,FmT[1],Kab,Kcd,SSSS0=0.0 ;
    for ( auto &pripair1 : pair1.primpairs )
    for ( auto &pripair2 : pair2.primpairs ) {
       
      sqrPQ = 0.0;
      for(int m=0 ; m<3 ; m++ ) {
        PQ = ( pripair1.P[m]-pripair2.P[m] );
        sqrPQ += PQ*PQ;
      } 

      // zeta+eta is expoT
      expoT = 1.0/pripair1.one_over_gamma + 1.0/pripair2.one_over_gamma;  

      // T= \rho * PQ^2
      T = sqrPQ/( pripair1.one_over_gamma + pripair2.one_over_gamma );

      // calculate F0(T)
      computeFmTTaylor( FmT, T, 0, 0 );

      Kab = sqrt(2.0)* pow(M_PI,1.25) * pripair1.K; 
      Kcd = sqrt(2.0)* pow(M_PI,1.25) * pripair2.K;

      norm = shell1.contr[0].coeff[pripair1.p1]* 
             shell2.contr[0].coeff[pripair1.p2]* 
             shell3.contr[0].coeff[pripair2.p1]* 
             shell4.contr[0].coeff[pripair2.p2];  

      SSSS0 += norm * Kab * Kcd * FmT[0] / sqrt(expoT); 

    } // for pripair2

    return SSSS0;
    
  } // twoeSSSS0




}  // namespace ChronusQ 

