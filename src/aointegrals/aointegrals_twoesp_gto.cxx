#include <aointegrals.hpp>
#include <aointegrals/inhouseaointegral.hpp>

namespace ChronusQ {

  std::vector<std::vector<double>> RealGTOIntEngine::BottomupHGP_TwoESP(
    libint2::ShellPair &pair1, libint2::ShellPair &pair2,
    libint2::Shell &shell1, libint2::Shell &shell2,
    libint2::Shell &shell3, libint2::Shell &shell4) {

    // LA, LB, LC, LD are the total angular momenta of each shell

    int LA, LB, LC, LD;
    LA = shell1.contr[0].l;
    LB = shell2.contr[0].l;
    LC = shell3.contr[0].l;
    LD = shell4.contr[0].l;

    // L is the total angular momentum of the integral

    int L = shell1.contr[0].l + shell2.contr[0].l
          + shell3.contr[0].l + shell4.contr[0].l;

    // Lbra is the total angular momentum of the left side
    // Lket is the total angular momentum of the right side

    int Lbra = shell1.contr[0].l + shell2.contr[0].l;
    int Lket = shell3.contr[0].l + shell4.contr[0].l;

    double A[3],B[3],C[3],D[3];

    for (int iWork = 0; iWork < 3; iWork++) {

      A[iWork] = shell1.O[iWork];
      B[iWork] = shell2.O[iWork];
      C[iWork] = shell3.O[iWork];
      D[iWork] = shell4.O[iWork];

    } // for (int iWork = 0; iWork < 3; iWork++)

    double AB[3], CD[3];

    for (int iWork = 0; iWork < 3; iWork++) {

      AB[iWork] = A[iWork] - B[iWork];
      CD[iWork] = C[iWork] - D[iWork];

    } // for (int iWork = 0; iWork < 3; iWork++)

    int shift[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    // here allocate the boys function
    // FT for the nonrelativistic case
    // ssss for the relativistic case

    double *FT = new double[L + 2];
    double *ssss = new double[L + 2];

    double P[3],Q[3];

    // Allocate the vectors for Finalbraket
    // Finalbraket is Final Braket

    std::vector<std::vector<std::vector<std::vector<double>>>> Finalbraket(4);

    for (int ijk = 0; ijk < 4; ijk++) {

      Finalbraket[ijk].resize((Lbra - LA + 1) * (Lket - LC + 1));

      for (int lA = LA; lA <= Lbra; lA++) {

        for (int lC = LC; lC <= Lket; lC++) {

          Finalbraket[ijk][(lA - LA) * (Lket - LC + 1) + (lC - LC)].resize((Lbra - lA + 1)
            * (Lket - lC + 1));

          for (int lB = 0; lB <= Lbra - lA; lB++) {

            for (int lD = 0; lD <= Lket - lC; lD++) {

              Finalbraket[ijk][(lA - LA) * (Lket - LC + 1) + (lC - LC)][
                      lB * (Lket - lC + 1) + lD
                      ].assign(cart_ang_list[lA].size()
                               * cart_ang_list[lB].size()
                               * cart_ang_list[lC].size()
                               * cart_ang_list[lD].size(), 0.0);

            } // for (int lD = 0; lD <= Lket - lC; lD++)

          } // for (int lB = 0; lB <= Lbra - lA; lB++)

        } // for (int lC = LC; lC <= Lket; lC++)

      } // for (int lA = LA; lA <= Lbra; lA++)

    } // for (int ijk = 0; ijk < 3; ijk++)

    // loop over primitive shellpairs in bra side
    for (auto &pripair1 : pair1.primpairs) {

      for (int iWork = 0; iWork < 3; iWork++) {

        P[iWork] = pripair1.P[iWork];

      } // for (int iWork = 0; iWork < 3; iWork++)

      double alpha = shell1.alpha[pripair1.p1];
      double beta = shell2.alpha[pripair1.p2];

      for (auto &pripair2 : pair2.primpairs) {

        for (int iWork = 0; iWork < 3; iWork++) {

          Q[iWork] = pripair2.P[iWork];

        } //  for (int iWork = 0; iWork < 3; iWork++)

        double sqrPQ = 0.0;

        for (int mu = 0; mu < 3; mu++) {

          double PQ = (pripair1.P[mu] - pripair2.P[mu]);
          sqrPQ += PQ * PQ;

        } // for (int mu=0; mu < 3; mu++)

        double gamma = shell3.alpha[pripair2.p1];
        double delta = shell4.alpha[pripair2.p2];
        auto zeta = 1.0 / pripair1.one_over_gamma;
        auto eta  = 1.0 / pripair2.one_over_gamma;
        auto zetaG = zeta + eta;
        auto rho = zeta * eta / zetaG;
        auto T = rho * sqrPQ;

        // calculate Fm(T) list
        computeFmTTaylor(FT, T, L, 0);
        computeFmTTaylor(ssss, T, L + 1, 0);

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

        // Vnreri is the storage structure used for the nonrelativistic
        // electron repulsion integral for vertical recursion
        // Vtempbraket is the storage structure used for the relativistic
        // electron repulsion integral for vertical recursion

        std::vector<std::vector<std::vector<double>>> Vnreri((Lbra + 2)
            * (Lket + 1));

        std::vector<std::vector<std::vector<std::vector<double>>>>
          Vtempbraket((Lbra + 1) * (Lket + 1));

        for (int k = 0; k <= Lbra + 1; k++) {

          for (int l = 0; l <= Lket; l++) {

            Vnreri[k * (Lket + 1) + l].resize(cart_ang_list[k].size()
                                              * cart_ang_list[l].size());

            int mbraket = L + 1 - k - l;

            for (int cart_i = 0; cart_i < cart_ang_list[k].size(); cart_i++) {

              for (int cart_j = 0; cart_j < cart_ang_list[l].size(); cart_j++)
              {

                Vnreri[k * (Lket + 1) + l][
                  cart_i * cart_ang_list[l].size() + cart_j
                  ].resize(mbraket + 1);

              } // for (int cart_j = 0; cart_j < cart_ang_list[l].size();
                // cart_j++)

            } // for (int cart_i = 0; cart_i < cart_ang_list[k].size();
              // cart_i++)

          } // for (int l = 0; l <= Lket; l++)

        } // for (int k = 0; k <= Lbra; k++)

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

                for (int m = 0; m < mbraket + 1; m++) {

                  Vtempbraket[k * (Lket + 1) + l][
                    cart_i * cart_ang_list[l].size() + cart_j
                    ][m].resize(4);

                } // for (int m = 0; m < mbraket + 1; m++)

              } // for (int cart_j = 0; cart_j < cart_ang_list[l].size();
                // cart_j++)

            } // for (int cart_i = 0; cart_i < cart_ang_list[k].size();
              // cart_i++)

          } // for (int l = 0; l <= Lket; l++)

        } // for (int k = 0; k <= Lbra; k++)

    // Allocate the vectors for Hbraket
    // Hbraket is used for nonrelativistic horizontal recursion

    std::vector<std::vector<std::vector<double>>> Hbraket((Lbra - LA + 1)
      * (Lket - LC + 1));

    for (int lA = LA; lA <= Lbra; lA++) {

      for (int lC = LC; lC <= Lket; lC++) {

        Hbraket[(lA - LA) * (Lket - LC + 1) + (lC - LC)].resize((Lbra - lA + 1)
                * (Lket - lC + 1));

        for (int lB = 0; lB <= Lbra - lA; lB++) {

          for (int lD = 0; lD <= Lket - lC; lD++) {

            Hbraket[(lA - LA) * (Lket - LC + 1) + (lC - LC)][
                    lB * (Lket - lC + 1) + lD
                    ].assign(cart_ang_list[lA].size()
                             * cart_ang_list[lB].size()
                             * cart_ang_list[lC].size()
                             * cart_ang_list[lD].size(), 0.0);

          } // for (int lD = 0; lD <= Lket - lC; lD++)

        } // for (int lB = 0; lB <= Lbra - lA; lB++)

      } // for (int lC = LC; lC <= Lket; lC++)

    } // for (int lA = LA; lA <= Lbra; lA++)

    std::vector<std::vector<std::vector<double>>> Hbraketm1(Lket - LC + 1);

    for (int lC = LC; lC <= Lket; lC++) {

      Hbraketm1[lC - LC].resize((Lbra - LA + 1));

      for (int lB = 0; lB <= Lbra - LA; lB++) {

        if (LA > 0) {

          Hbraketm1[lC - LC][lB].assign(cart_ang_list[LA - 1].size() 
                                      * cart_ang_list[lB].size()
                                      * cart_ang_list[lC].size()
                                      * cart_ang_list[0].size(), 0.0);
        
        } else { // if (LA > 0) {
        
          Hbraketm1[lC - LC][lB].assign(1, 0.0);

        } // else{

      } // for (int(lB = 0; lB <= Lbra; lB++) {

    } // for (int lC = LC; lC <= Lket; lC++) {

    // Allocate the vectors for Vbraket
    // Vbraket is used for relativistic horizontal recursion

    std::vector<std::vector<std::vector<std::vector<double>>>> Vbraket(4);

    for (int ijk = 0; ijk < 4; ijk++) {

      Vbraket[ijk].resize((Lbra - LA + 1) * (Lket - LC + 1));

      for (int lA = LA; lA <= Lbra; lA++) {

        for (int lC = LC; lC <= Lket; lC++) {

          Vbraket[ijk][(lA - LA) * (Lket - LC + 1) + (lC - LC)].resize((Lbra - lA + 1)
            * (Lket - lC + 1));

          for (int lB = 0; lB <= Lbra - lA; lB++) {

            for (int lD = 0; lD <= Lket - lC; lD++) {

              Vbraket[ijk][(lA - LA) * (Lket - LC + 1) + (lC - LC)][
                      lB * (Lket - lC + 1) + lD
                      ].assign(cart_ang_list[lA].size()
                               * cart_ang_list[lB].size()
                               * cart_ang_list[lC].size()
                               * cart_ang_list[lD].size(), 0.0);

            } // for (int lD = 0; lD <= Lket - lC; lD++)

          } // for (int lB = 0; lB <= Lbra - lA; lB++)

        } // for (int lC = LC; lC <= Lket; lC++)

      } // for (int lA = LA; lA <= Lbra; lA++)

    } // for (int ijk = 0; ijk < 3; ijk++) {

        double W[3];

        for (int ii = 0; ii < 3; ii++) {

          W[ii] = (zeta * P[ii] + eta * Q[ii]) / zetaG;

        } // for (int ii = 0; ii < 3; ii++)

        double PA[3], PB[3], WP[3];

        for (int iWork = 0; iWork < 3; iWork++) {

          PA[iWork] = A[iWork] - B[iWork];
          PB[iWork] = C[iWork] - D[iWork];
          WP[iWork] = C[iWork] - D[iWork];

        } // for (int iWork = 0; iWork < 3; iWork++)

        for (int ii = 0; ii <= L + 1; ii++) {

          Vnreri[0][0][ii] = FT[ii] * pref;

        } // for (int ii = 0; ii <= L; ii++)

        for (int ii = 0; ii <= L + 1; ii++) {

          ssss[ii] = ssss[ii] * pref;

        } // for (int ii = 0; ii <= L + 1; ii++) {

        for (int ii = 0; ii <= L; ii++) {

          for (int cart_i = 0; cart_i < 3; cart_i++) {
            //std::cout << "Made it before cross at 364 and 365" << std::endl; 
            Vtempbraket[0][0][ii][cart_i] =
              4 * alpha * beta
              * (cross(PA, PB)[cart_i] * ssss[ii]
              + cross(WP, AB)[cart_i] * ssss[ii + 1]);
            //std::cout << "Made it past cross call at 364 and 365" << std::endl;
          } // for (int cart_i = 0; cart_i < 3; cart_i++) {

            Vtempbraket[0][0][ii][3] =
              4 * alpha * beta
              * (dot(PA, PB) * ssss[ii]
              + dot(WP, AB) * ssss[ii + 1]);

        } // for (int ii = 0; ii <= L; ii++) {

        for (int k = 0; k <= Lbra + 1; k++) { // Loop over the bra angular momentum

          for (int cart_i = 0; cart_i < cart_ang_list[k].size(); cart_i++) {

            int lA_xyz[3];

            for (int ii = 0; ii < 3; ii++) {

              lA_xyz[ii] = cart_ang_list[k][cart_i][ii];

              // lA_xyz is part of the shell we want to compute ([a]0|00)(m)

            } // for (int ii = 0; ii < 3; ii++)

            int mbra = L + 1 - k; // mbra is the highest auxiliary number for a
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
              // lAtemp is ((a-1i)0|00)(m)

              int indexlm1xyz = indexmap(k - 1, lAtemp[0], lAtemp[1],
                                         lAtemp[2]);
              // indexlm1xyz is the index of ([(a - 1i)]0|00)(m)

              // Calculate index of lAm1

              for (int m = 0; m <= mbra; m++) {

                double ERIscratch = 0.0;

                ERIscratch = (P[iwork] - A[iwork])
                             * Vnreri[(k - 1) * (Lket + 1)][indexlm1xyz][m]
                             + (W[iwork] - P[iwork])
                             * Vnreri[(k - 1) * (Lket + 1)][indexlm1xyz][m + 1];

                if (lA_xyz[iwork] > 1) {

                  // resets lAtemp to ([a]0|00)(m)

                  for (int ii = 0; ii < 3; ii++) {

                    lAtemp[ii] = lA_xyz[ii];

                  } // for (int ii = 0; ii < 3; ii++)

                  lAtemp[iwork] = lA_xyz[iwork] - 2;
                  // lAtemp is now ([(a - 2i)]0|00)(m)

                  int indexlm2xyz = indexmap(k - 2, lAtemp[0], lAtemp[1],
                                             lAtemp[2]);
                  // indexlm1xyz is the index of ([(a - 2i)]0|00)(m)

                  ERIscratch = ERIscratch
                             + 1 / (2 * zeta) * (lA_xyz[iwork] - 1)
                             * (Vnreri[(k - 2) * (Lket + 1)][indexlm2xyz][m]
                             - rho / zeta
                             * Vnreri[(k - 2) * (Lket + 1)][indexlm2xyz][m + 1]);

                } // if (lA_xyz[iwork] > 1)

                Vnreri[k * (Lket + 1)][cart_i][m] = ERIscratch;

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

                  // lCtemp = (a0|[c]0)(m)

                  // For ket side. Equation 6. HGP
                  lCtemp[iwork] = lC_xyz[iwork] - 1;

                  // lCtemp = (a0|[(c - 1i)]0)(m)

                  int indexlm1xyz = indexmap(l - 1, lCtemp[0], lCtemp[1],
                                             lCtemp[2]);
                  // indexlm1xyz = index for (a0|[(c - 1i)]0)(m)

                  // Calculate indeix of lCm1
                  for (int m = 0; m <= mbraket; m++) {

                    double ERIscratch = 0.0;

                    // cart_i is a index

                    ERIscratch = (Q[iwork] - C[iwork])
                                 * Vnreri[k * (Lket + 1) + (l - 1)][
                                 (cart_i) * cart_ang_list[l - 1].size()
                                 + indexlm1xyz][m]
                                 + (W[iwork] - Q[iwork])
                                 * Vnreri[k * (Lket + 1) + (l - 1)][
                                 cart_i * cart_ang_list[l - 1].size()
                                 + indexlm1xyz][m + 1];

                    if (lC_xyz[iwork] > 1) {

                      for (int ii = 0; ii < 3; ii++) {

                        lCtemp[ii] = lC_xyz[ii];

                      } // for (int ii = 0; ii < 3; ii++)

                      // Resets lCtemp to (a0|[c]0)(m)

                      lCtemp[iwork] = lC_xyz[iwork] - 2;

                      // lAtemp = (a0|[(c - 2i)]0)(m)

                      int indexlm2xyz = indexmap(l - 2, lCtemp[0], lCtemp[1],
                                                 lCtemp[2]);
                      ERIscratch = ERIscratch
                                   + 1 / (2 * eta) * (lC_xyz[iwork] - 1)
                                   * (Vnreri[k * (Lket + 1) + (l - 2)][
                                   cart_i * cart_ang_list[l - 2].size()
                                   + indexlm2xyz][m]
                                   - rho / eta
                                   * Vnreri[k * (Lket + 1) + (l - 2)][
                                   cart_i * cart_ang_list[l - 2].size()
                                   + indexlm2xyz][m + 1]);

                    } // if (lC_xyz[iwork] > 1)

                    if (lA_xyz[iwork] > 0) {

                      // lCtemp = (a0|[(c - 1i)]0)(m)

                      int lAtemp[3];

                      for (int ii = 0; ii < 3; ii++) {

                        lAtemp[ii] = lA_xyz[ii];

                      } // for (int ii = 0; ii < 3; ii++)

                      // lAtemp = ([a]0|(c - 1i)0)(m) a value

                      lAtemp[iwork] -= 1;

                      // lAtemp = ([(a - 1i)]0|(c - 1i)0)(m)

                      int indexlAm1 = indexmap(k - 1, lAtemp[0], lAtemp[1],
                                               lAtemp[2]);

                      // indexlAm1 = index of ([(a - 1i)]0|(c - 1i)0)(m)

                      ERIscratch += 1.0 / (2.0 * zetaG) * lA_xyz[iwork]
                                    * Vnreri[(k - 1) * (Lket + 1) + (l - 1)][
                                    (indexlAm1) * cart_ang_list[l - 1].size()
                                    + indexlm1xyz][m + 1];

                    } // if (lA_xyz[iwork] > 0)

                    Vnreri[k * (Lket + 1) + l][
                      cart_i * cart_ang_list[l].size() + cart_j][m]
                      = ERIscratch;

                  } // for (int m = 0; m <= mbraket; m++)

                } // if (l > 0)

              } // for (int cart_j = 0; cart_j < cart_ang_list[l].size();
                // cart_j++)

            } // for (int l = 0; l <= Lket; l++)

          } // for (int cart_i = 0; cart_i < cart_ang_list[k].size(); cart_i++)

        } // for (int k = 0; k <= Lbra; k++)

        // ================================================================== //

        // Relativistic Vertical Recursion

        for (int k = 0; k <= Lbra; k++) { // Loop over the bra angular momentum

          for (int cart_i = 0; cart_i < cart_ang_list[k].size(); cart_i++) {

            int lA_xyz[3];

            for (int ii = 0; ii < 3; ii++) {

              lA_xyz[ii] = cart_ang_list[k][cart_i][ii];

            } // for (int ii = 0; ii < 3; ii++)

            //

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

              // lAtemp = ([a]0|00)(m)

              lAtemp[iwork] = lA_xyz[iwork] - 1;

              // lAtemp = ([(a - 1i)]0|00)(m)

              int indexlm1xyz = indexmap(k - 1, lAtemp[0], lAtemp[1],
                                         lAtemp[2]);

              // indexlm1xyz = index of ([(a - 1i)]0|00)(m)

              // Calculate index of lAm1

              for (int m = 0; m <= mbra; m++) {

                for (int ijk = 0; ijk < 3; ijk++) {

                  double ERIscratch = 0.0;

                  ERIscratch = (P[iwork] - A[iwork])
                               * Vtempbraket[(k - 1) * (Lket + 1)][
                                             indexlm1xyz][m][ijk]
                               + (W[iwork] - P[iwork])
                               * Vtempbraket[(k - 1) * (Lket + 1)][
                                             indexlm1xyz][m + 1][ijk];

                  if (lA_xyz[iwork] > 1) {

                    for (int ii = 0; ii < 3; ii++) {

                      lAtemp[ii] = lA_xyz[ii];

                    } // for (int ii = 0; ii < 3; ii++)

                    // lAtemp = ([a]0|00)(m)

                    lAtemp[iwork] = lA_xyz[iwork] - 2;

                    // lAtemp = ([(a - 2i)]0|00)(m)

                    int indexlm2xyz = indexmap(k - 2, lAtemp[0], lAtemp[1],
                                               lAtemp[2]);

                    // indexlm2xyz = index of ([a - 2i]0|00)(m)

                    ERIscratch = ERIscratch
                                 + 1 / (2 * zeta) * (lA_xyz[iwork] - 1)
                                 * (Vtempbraket[(k - 2) * (Lket + 1)][
                                                indexlm2xyz][m][ijk]
                                 - rho / zeta
                                 * Vtempbraket[(k - 2) * (Lket + 1)][
                                               indexlm2xyz][m + 1][ijk]);

                  } // if (lA_xyz[iwork] > 1)

                  for (int u = 0; u < 3; u++) {

                    for (int ii = 0; ii < 3; ii++) {

                      lAtemp[ii] = lA_xyz[ii];

                    } // for (int ii = 0; ii < 3; ii++)

                    // reset lAtemp = ([a]0|00)(m)

                    lAtemp[iwork]--;

                    int indexlm1xyz = indexmap(k - 1, lAtemp[0], lAtemp[1], lAtemp[2]);

                    // lAtemp = ([(a - 1i)]0|00)(m)

                    lAtemp[u]++;

                    // lAtemp = ([(a - 1i + 1u)]0|00)(m)

                    int indexluxyz = indexmap(k, lAtemp[0], lAtemp[1], lAtemp[2]);

                    // indexluxyz = index of lAtemp = ([(a - 1i + 1u)]0|00)(m)

                    //std::cout << "Made it before cross call at 754" << std::endl;
                    ERIscratch = ERIscratch
                     + cross(shift[iwork], shift[u])[ijk]
                     * (-beta / zeta * 2 * (alpha
                     * Vnreri[k * (Lket + 1)][indexluxyz][m]
                     + beta
                     * Vnreri[k * (Lket + 1)][indexluxyz][m]
                     + AB[u]
                     * Vnreri[(k - 1) * (Lket + 1)][indexlm1xyz][m])
                     - (beta/zetaG - beta/zeta) * 2 * alpha
                     * Vnreri[k * (Lket + 1)][indexluxyz][m + 1]
                     + (alpha/zetaG - alpha/zeta) * 2 * beta
                     * Vnreri[k * (Lket + 1)][indexluxyz][m + 1]
                     + AB[u]
                     * Vnreri[(k - 1) * (Lket + 1)][indexlm1xyz][m + 1]);
                    //std::cout << "Made it past cross call at 754" << std::endl;
                    for (int ii = 0; ii < 3; ii++) {

                     lAtemp[ii] = lA_xyz[ii];

                    } // for (int ii = 0; ii < 3; ii++)

                    // reset lAtemp = ([a]0|00)(m)

                    lAtemp[iwork] = lA_xyz[iwork] - 1;

                    // lAtemp = ([(a - 1i)]0|00)(m)

                    if (lAtemp[u] > 0) {

                      lAtemp[u]--;

                      // lAtemp = ([(a - 1i - 1u)]0|00)(m)
                      //std::cout << "Made it before cross call at 787" << std::endl;
                      int indexlmuxyz = indexmap(k - 2, lAtemp[0], lAtemp[1], lAtemp[2]);
                      ERIscratch = ERIscratch + cross(shift[iwork], shift[u])[ijk]
                        * ((beta / zeta * (lAtemp[u] + 1)
                        * Vnreri[(k - 2) * (Lket + 1)][indexlmuxyz][m])
                        + (beta / zetaG - beta / zeta) * (lAtemp[u] + 1)
                        * Vnreri[(k - 2) * (Lket + 1)][indexlmuxyz][m + 1]);
                      //std::cout << "Made it past cross call at 787" << std::endl;
                      } // if (lAtemp[u] > 0)

                  } // for (int u = 0; u < 3; u++) 

                  Vtempbraket[k * (Lket + 1)][cart_i][m][ijk] = ERIscratch;

                } // for (int ijk = 0; ijk < 3; ijk++) 

                  double ERIscratch = 0.0;

                  ERIscratch = (P[iwork] - A[iwork])
                               * Vtempbraket[(k - 1) * (Lket + 1)][
                                             indexlm1xyz][m][3]
                               + (W[iwork] - P[iwork])
                               * Vtempbraket[(k - 1) * (Lket + 1)][
                                             indexlm1xyz][m + 1][3];

                  if (lA_xyz[iwork] > 1) {

                    for (int ii = 0; ii < 3; ii++) {

                      lAtemp[ii] = lA_xyz[ii];

                    } // for (int ii = 0; ii < 3; ii++)

                    // lAtemp = ([a]0|00)(m)

                    lAtemp[iwork] = lA_xyz[iwork] - 2;

                    // lAtemp = ([(a - 2i)]0|00)(m)

                    int indexlm2xyz = indexmap(k - 2, lAtemp[0], lAtemp[1],
                                               lAtemp[2]);

                    // indexlm2xyz = index of ([a - 2i]0|00)(m)

                    ERIscratch = ERIscratch
                                 + 1 / (2 * zeta) * (lA_xyz[iwork] - 1)
                                 * (Vtempbraket[(k - 2) * (Lket + 1)][
                                                indexlm2xyz][m][3]
                                 - rho / zeta
                                 * Vtempbraket[(k - 2) * (Lket + 1)][
                                               indexlm2xyz][m + 1][3]);

                  } // if (lA_xyz[iwork] > 1)

                  for (int u = 0; u < 3; u++) {

                    for (int ii = 0; ii < 3; ii++) {

                      lAtemp[ii] = lA_xyz[ii];

                    } // for (int ii = 0; ii < 3; ii++)

                    // reset lAtemp = ([a]0|00)(m)

                    lAtemp[iwork]--;

                    int indexlm1xyz = indexmap(k - 1, lAtemp[0], lAtemp[1], lAtemp[2]);

                    // lAtemp = ([(a - 1i)]0|00)(m)

                    lAtemp[u]++;

                    // lAtemp = ([(a - 1i + 1u)]0|00)(m)

                    int indexluxyz = indexmap(k, lAtemp[0], lAtemp[1], lAtemp[2]);

                    // indexluxyz = index of lAtemp = ([(a - 1i + 1u)]0|00)(m)

                    ERIscratch = ERIscratch
                     + dot(shift[iwork], shift[u])
                     * (-beta / zeta * 2 * (alpha
                     * Vnreri[k * (Lket + 1)][indexluxyz][m]
                     + beta
                     * Vnreri[k * (Lket + 1)][indexluxyz][m]
                     + AB[u]
                     * Vnreri[(k - 1) * (Lket + 1)][indexlm1xyz][m])
                     - (beta/zetaG - beta/zeta) * 2 * alpha
                     * Vnreri[k * (Lket + 1)][indexluxyz][m + 1]
                     + (alpha/zetaG - alpha/zeta) * 2 * beta
                     * Vnreri[k * (Lket + 1)][indexluxyz][m + 1]
                     + AB[u]
                     * Vnreri[(k - 1) * (Lket + 1)][indexlm1xyz][m + 1]);

                    for (int ii = 0; ii < 3; ii++) {

                     lAtemp[ii] = lA_xyz[ii];

                    } // for (int ii = 0; ii < 3; ii++)

                    // reset lAtemp = ([a]0|00)(m)

                    lAtemp[iwork] = lA_xyz[iwork] - 1;

                    // lAtemp = ([(a - 1i)]0|00)(m)

                    if (lAtemp[u] > 0) {

                      lAtemp[u]--;

                      // lAtemp = ([(a - 1i - 1u)]0|00)(m)

                      int indexlmuxyz = indexmap(k - 2, lAtemp[0], lAtemp[1], lAtemp[2]);
                      ERIscratch = ERIscratch + dot(shift[iwork], shift[u])
                        * ((beta / zeta * (lAtemp[u] + 1)
                        * Vnreri[(k - 2) * (Lket + 1)][indexlmuxyz][m])
                        + (beta / zetaG - beta / zeta) * (lAtemp[u] + 1)
                        * Vnreri[(k - 2) * (Lket + 1)][indexlmuxyz][m + 1]);

                    } // if (lAtemp[u] > 0)

                  } // for (int u = 0; u < 3; u++)

                  Vtempbraket[k * (Lket + 1)][cart_i][m][3] = ERIscratch;

              } // for (int m = 0; m <= mbra; m++)

            } // if (k > 0)

            // Right Hand Side Relativistic Vertical Recursion

            for (int l = 0; l <= Lket; l++) {

              for (int cart_j = 0; cart_j < cart_ang_list[l].size(); cart_j++)
              {

                int lC_xyz[3];

                for (int ii = 0; ii < 3; ii++) {

                  lC_xyz[ii] = cart_ang_list[l][cart_j][ii];

                } // for (int ii = 0; ii < 3; ii++)

                // lC_xyz = (a0|[c]0)(m)

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

                  // lC_temp = (a0|[c]0)(m)

                  // For ket side. Equation 6. HGP
                  lCtemp[iwork] = lC_xyz[iwork] - 1;

                  // lC_temp = (a0|[c - 1i]0)(m)

                  // Calculate indeix of lCm1
                  int indexlm1xyz = indexmap(l - 1, lCtemp[0], lCtemp[1],
                                             lCtemp[2]);

                  // indexlm1xyz is the index of (a0|[c - 1i]0)(m)

                  for (int m = 0; m <= mbraket; m++) {

                    for (int ijk = 0; ijk < 3; ijk++) {

                      double ERIscratch = 0.0;

                      ERIscratch = (Q[iwork] - C[iwork])
                                   * Vtempbraket[k * (Lket + 1) + (l - 1)][
                                     (cart_i) * cart_ang_list[l - 1].size()
                                     + indexlm1xyz][m][ijk]
                                   + (W[iwork] - Q[iwork])
                                   * Vtempbraket[k * (Lket + 1) + (l - 1)][
                                     cart_i * cart_ang_list[l - 1].size()
                                     + indexlm1xyz][m + 1][ijk];

                      if (lC_xyz[iwork] > 1) {

                        for (int ii = 0; ii < 3; ii++) {

                          lCtemp[ii] = lC_xyz[ii];

                        } // for (int ii = 0; ii < 3; ii++)

                        // reset lC_temp = (a0|[c]0)(m)

                        lCtemp[iwork] = lC_xyz[iwork] - 2;

                        // lC_temp = (a0|[c - 2i]0)(m)

                        int indexlm2xyz = indexmap(l - 2, lCtemp[0], lCtemp[1],
                                                   lCtemp[2]);

                        // indexlm2xyz is the index of (a0|[c - 2i]0)(m)

                        ERIscratch = ERIscratch
                                     + 1 / (2 * eta) * (lC_xyz[iwork] - 1)
                                     * (Vtempbraket[k * (Lket + 1) + (l - 2)][
                                       cart_i * cart_ang_list[l - 2].size()
                                       + indexlm2xyz][m][ijk]
                                     - rho / eta
                                     * Vtempbraket[k * (Lket + 1) + (l - 2)][
                                       cart_i * cart_ang_list[l - 2].size()
                                       + indexlm2xyz][m + 1][ijk]);

                      } // if (lC_xyz[iwork] > 1)

                      if (lA_xyz[iwork] > 0) {

                        int lAtemp[3];

                        for (int ii = 0; ii < 3; ii++) {

                          lAtemp[ii] = lA_xyz[ii];

                        } // for (int ii = 0; ii < 3; ii++)

                        // lA_temp = ([a]0|c0)(m)

                        lAtemp[iwork]--;

                        // lA_temp = ([a - 1i]0|c0)(m)

                        int indexlAm1 = indexmap(k - 1, lAtemp[0], lAtemp[1],
                                                 lAtemp[2]);

                        // indexlAm1 is the index of ([a - 1i]0|c0)(m)

                        ERIscratch += 1.0 / (2.0 * zetaG) * lA_xyz[iwork]
                                      * Vtempbraket[
                                        (k - 1) * (Lket + 1) + (l - 1)][
                                        (indexlAm1) * cart_ang_list[l - 1].size()
                                        + indexlm1xyz][m + 1][ijk];

                      } // if (lA_xyz[iwork] > 0)

                      int lAtemp[3];

                      for (int u = 0; u < 3; u++) {

                        for (int ii = 0; ii < 3; ii++) {

                          lAtemp[ii] = lA_xyz[ii];

                        } // for (int ii = 0; ii < 3; ii++)

                        int indexluxyz = indexmap(k, lAtemp[0], lAtemp[1], lAtemp[2]);

                        // reset lA_temp = ([a]0|(c - 1i)0)(m)

                          lAtemp[u]++;

                          // lA_temp = ([a + 1u]0|(c - 1i)0)(m)

                          int indexlpuxyz = indexmap(k + 1, lAtemp[0], lAtemp[1], lAtemp[2]);

                          // indexluxyz is the index of // lA_temp = ([a + 1u]0|(c - 1i)0)(m)

                          //std::cout << "Made it before cross call at 1075" << std::endl;
                          ERIscratch = ERIscratch
                            + cross(shift[iwork], shift[u])[ijk]
                            * (-beta/zetaG) * 2 * alpha
                            * Vnreri[(k + 1) * (Lket + 1) + (l - 1)][indexlpuxyz * cart_ang_list[l - 1].size() + indexlm1xyz][m + 1]
                            + (alpha/zetaG) * 2 * beta
                            * (Vnreri[(k + 1) * (Lket + 1) + (l - 1)][indexlpuxyz * cart_ang_list[l - 1].size() + indexlm1xyz][m + 1]
                            + AB[u]
                            * Vnreri[k * (Lket + 1) + (l - 1)][indexluxyz * cart_ang_list[l - 1].size() + indexlm1xyz][m + 1]);
                          //std::cout << "Made it past cross call at 1075" << std::endl;
                            for (int ii = 0; ii < 3; ii++) {

                              lAtemp[ii] = lA_xyz[ii];

                            } // for (int ii = 0; ii < 3; ii++)

                          if (lA_xyz[u] > 0) {

                              lAtemp[u]--;
                              int indexlmuxyz = indexmap(k - 1, lAtemp[0], lAtemp[1], lAtemp[2]);
                              //std::cout << "Made it before cross call at 1094" << std::endl;
                              ERIscratch = ERIscratch + cross(shift[iwork], shift[u])[ijk]
                                * ((beta / zetaG * (lA_xyz[u])
                                * Vnreri[(k - 1) * (Lket + 1) + (l - 1)][indexlmuxyz * cart_ang_list[l - 1].size() + indexlm1xyz][m + 1]));
                              //std::cout << "Made it past cross call at 1094" << std::endl;

                          } // if (lAtemp[u] > 0)

                      } // for (int u = 0; u < 3; u++)

                      Vtempbraket[k * (Lket + 1) + l][
                        cart_i * cart_ang_list[l].size() + cart_j][
                        m][ijk]
                        = ERIscratch;

                    } // for (int ijk = 0; ijk < 3; ijk++)

                    double ERIscratch = 0.0;

                      ERIscratch = (Q[iwork] - C[iwork])
                                   * Vtempbraket[k * (Lket + 1) + (l - 1)][
                                     (cart_i) * cart_ang_list[l - 1].size()
                                     + indexlm1xyz][m][3]
                                   + (W[iwork] - Q[iwork])
                                   * Vtempbraket[k * (Lket + 1) + (l - 1)][
                                     cart_i * cart_ang_list[l - 1].size()
                                     + indexlm1xyz][m + 1][3];

                      if (lC_xyz[iwork] > 1) {

                        for (int ii = 0; ii < 3; ii++) {

                          lCtemp[ii] = lC_xyz[ii];

                        } // for (int ii = 0; ii < 3; ii++)

                        // reset lC_temp = (a0|[c]0)(m)

                        lCtemp[iwork] = lC_xyz[iwork] - 2;

                        // lC_temp = (a0|[c - 2i]0)(m)

                        int indexlm2xyz = indexmap(l - 2, lCtemp[0], lCtemp[1],
                                                   lCtemp[2]);

                        // indexlm2xyz is the index of (a0|[c - 2i]0)(m)

                        ERIscratch = ERIscratch
                                     + 1 / (2 * eta) * (lC_xyz[iwork] - 1)
                                     * (Vtempbraket[k * (Lket + 1) + (l - 2)][
                                       cart_i * cart_ang_list[l - 2].size()
                                       + indexlm2xyz][m][3]
                                     - rho / eta
                                     * Vtempbraket[k * (Lket + 1) + (l - 2)][
                                       cart_i * cart_ang_list[l - 2].size()
                                       + indexlm2xyz][m + 1][3]);

                      } // if (lC_xyz[iwork] > 1)

                      if (lA_xyz[iwork] > 0) {

                        int lAtemp[3];

                        for (int ii = 0; ii < 3; ii++) {

                          lAtemp[ii] = lA_xyz[ii];

                        } // for (int ii = 0; ii < 3; ii++)

                        // lA_temp = ([a]0|c0)(m)

                        lAtemp[iwork]--;

                        // lA_temp = ([a - 1i]0|c0)(m)

                        int indexlAm1 = indexmap(k - 1, lAtemp[0], lAtemp[1],
                                                 lAtemp[2]);

                        // indexlAm1 is the index of ([a - 1i]0|c0)(m)

                        ERIscratch += 1.0 / (2.0 * zetaG) * lA_xyz[iwork]
                                      * Vtempbraket[
                                        (k - 1) * (Lket + 1) + (l - 1)][
                                        (indexlAm1) * cart_ang_list[l - 1].size()
                                        + indexlm1xyz][m + 1][3];

                      } // if (lA_xyz[iwork] > 0)

                      int lAtemp[3];

                      for (int u = 0; u < 3; u++) {

                        for (int ii = 0; ii < 3; ii++) {

                          lAtemp[ii] = lA_xyz[ii];

                        } // for (int ii = 0; ii < 3; ii++)

                        int indexluxyz = indexmap(k, lAtemp[0], lAtemp[1], lAtemp[2]);

                        // reset lA_temp = ([a]0|(c - 1i)0)(m)

                          lAtemp[u]++;

                          // lA_temp = ([a + 1u]0|(c - 1i)0)(m)

                          int indexlpuxyz = indexmap(k + 1, lAtemp[0], lAtemp[1], lAtemp[2]);

                          // indexluxyz is the index of // lA_temp = ([a + 1u]0|(c - 1i)0)(m)

                          ERIscratch = ERIscratch
                            + dot(shift[iwork], shift[u])
                            * (-beta/zetaG) * 2 * alpha
                            * Vnreri[(k + 1) * (Lket + 1) + (l - 1)][indexlpuxyz * cart_ang_list[l - 1].size() + indexlm1xyz][m + 1]
                            + (alpha/zetaG) * 2 * beta
                            * (Vnreri[(k + 1) * (Lket + 1) + (l - 1)][indexlpuxyz * cart_ang_list[l - 1].size() + indexlm1xyz][m + 1]
                            + AB[u]
                            * Vnreri[k * (Lket + 1) + (l - 1)][indexluxyz * cart_ang_list[l - 1].size() + indexlm1xyz][m + 1]);

                            for (int ii = 0; ii < 3; ii++) {

                              lAtemp[ii] = lA_xyz[ii];

                            } // for (int ii = 0; ii < 3; ii++)

                          if (lA_xyz[u] > 0) {

                              lAtemp[u]--;
                              int indexlmuxyz = indexmap(k - 1, lAtemp[0], lAtemp[1], lAtemp[2]);
                              ERIscratch = ERIscratch + dot(shift[iwork], shift[u])
                                * ((beta / zetaG * (lA_xyz[u])
                                * Vnreri[(k - 1) * (Lket + 1) + (l - 1)][indexlmuxyz * cart_ang_list[l - 1].size() + indexlm1xyz][m + 1]));

                          } // if (lAtemp[u] > 0)

                      } // for (int u = 0; u < 3; u++)
 
                      Vtempbraket[k * (Lket + 1) + l][
                        cart_i * cart_ang_list[l].size() + cart_j][
                        m][3]
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

                Hbraket[(lA - LA) * (Lket - LC + 1) + (lC - LC)][0][
                  i * cart_ang_list[lC].size() + k]
                  = Vnreri[
                  lA * (Lket + 1) + lC][i * cart_ang_list[lC].size() + k][0];

                for (int ijk = 0; ijk < 4; ijk++) {

                  Vbraket[ijk][(lA - LA) * (Lket - LC + 1) + (lC - LC)][0][
                    i * cart_ang_list[lC].size() + k]
                    = Vtempbraket[
                    lA * (Lket + 1) + lC][i * cart_ang_list[lC].size() + k][0][ijk];

                } // for (int ijk = 0; ijk < 3; ijk++)

              } // for (int k = 0; k <= cart_ang_list[lC].size(); k++)

            } // for (int i = 0; i <= cart_ang_list[lA].size(); i++)

          } // for (int lC = LC; lC <= Lket; lC++)

        } // for (int lA = LA; lA <= Lbra; lA++)

        if (LA > 0) {

          for (int lC = LC; lC <= Lket; lC++) {

            for (int i = 0; i < cart_ang_list[LA - 1].size(); i++) {

              for (int k = 0; k < cart_ang_list[lC].size(); k++) {

                Hbraketm1[lC - LC][0][
                  i * cart_ang_list[lC].size() + k]
                  = Vnreri[(LA - 1) * (Lket + 1) + lC][i * cart_ang_list[lC].size() + k][0];

              } // for (int k = 0; k <= cart_ang_list[lC].size(); k++)

            } // for (int i = 0; i <= cart_ang_list[lA].size(); i++)

          } // for (int lC = LC; lC <= Lket; lC++)

        } // if (LA > 0) {

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

                Hbraket[(lA - LA) * (Lket - LC + 1) + (lC - LC)][
                  lB * (Lket - lC + 1)][
                  Aidx * cart_ang_list[lB].size() * cart_ang_list[lC].size()
                  + Bidx * cart_ang_list[lC].size()
                  + Cidx]
                  = Hbraket[(lA - LA + 1) * (Lket - LC + 1) + (lC - LC)][
                  (lB - 1) * (Lket - lC + 1)][
                  indexmap(lA + 1, lAp1[0], lAp1[1], lAp1[2])
                  * cart_ang_list[lB - 1].size() * cart_ang_list[lC].size()
                  + idxBtemp * cart_ang_list[lC].size() + Cidx]
                  + (AB[iwork])
                  * Hbraket[(lA - LA) * (Lket - LC + 1) + lC - LC][
                  (lB - 1) * (Lket - lC + 1)][
                  Aidx * cart_ang_list[lB - 1].size()
                       * cart_ang_list[lC].size()
                  + idxBtemp * cart_ang_list[lC].size()
                  + Cidx];

              } // for (int Cidx = 0; Cidx < cart_ang_list[lC]; Cidx++)

            } // for (int lC = LC; lC <= Lket; lC++)

          } // for (int Bidx = 0; Bidx < cart_ang_list[lB].size(); Bidx++)

        } // for (int Aidx = 0; Aidx < cart_ang_list[lA].size(); Aidx++)

      } // for (int lA = LA; lA <= Lbra - lB; lB++)

    } // for (int lB = 1; lB <= LB; lB++)

    if (LA > 0) {

      int lA = LA - 1;

      for (int lB = 1; lB <= LB; lB++) {

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

                  Hbraketm1[lC - LC][
                    lB][
                    Aidx * cart_ang_list[lB].size() * cart_ang_list[lC].size()
                    + Bidx * cart_ang_list[lC].size()
                    + Cidx]
                    = Hbraket[(lA - LA + 1) * (Lket - LC + 1) + (lC - LC)][
                    (lB - 1) * (Lket - lC + 1)][
                    indexmap(lA + 1, lAp1[0], lAp1[1], lAp1[2])
                    * cart_ang_list[lB - 1].size() * cart_ang_list[lC].size()
                    + idxBtemp * cart_ang_list[lC].size() + Cidx]
                    + (AB[iwork])
                    * Hbraketm1[lC - LC][
                    lB - 1][
                    Aidx * cart_ang_list[lB - 1].size()
                         * cart_ang_list[lC].size()
                    + idxBtemp * cart_ang_list[lC].size()
                    + Cidx];

                } // for (int Cidx = 0; Cidx < cart_ang_list[lC]; Cidx++)

              } // for (int lC = LC; lC <= Lket; lC++)

            } // for (int Bidx = 0; Bidx < cart_ang_list[lB].size(); Bidx++)

          } // for (int Aidx = 0; Aidx < cart_ang_list[lA].size(); Aidx++)

      } // for (int lB = 1; lB <= LB; lB++)

    } // if (LA > 0);

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

                Hbraket[lC - LC][LB * (Lket - lC + 1) + lD][
                  Aidx * cart_ang_list[LB].size()
                  * cart_ang_list[lC].size()
                  * cart_ang_list[lD].size()
                  + Bidx * cart_ang_list[lC].size()* cart_ang_list[lD].size()
                  + Cidx * cart_ang_list[lD].size()
                  + Didx]
                  = Hbraket[lC - LC + 1][
                  LB * (Lket - (lC + 1) + 1) + (lD - 1)][
                  Aidx * cart_ang_list[LB].size()
                    * cart_ang_list[lC + 1].size()
                    * cart_ang_list[lD - 1].size()
                  + Bidx * cart_ang_list[lC + 1].size()
                    * cart_ang_list[lD - 1].size()
                  + indexmap(lC + 1, lCp1[0], lCp1[1], lCp1[2])
                    * cart_ang_list[lD - 1].size()
                  + idxDtemp]
                  + (CD[iwork]) * Hbraket[lC - LC][
                  LB * (Lket - lC + 1) + (lD - 1)][
                  Aidx * cart_ang_list[LB].size()
                  * cart_ang_list[lC].size()
                  * cart_ang_list[lD - 1].size()
                  + Bidx * cart_ang_list[lC].size()
                  * cart_ang_list[lD - 1].size()
                  + Cidx * cart_ang_list[lD - 1].size()
                  + idxDtemp];

              } // for (int Bidx = 0; Bidx < cart_ang_list[LB].size(); Bidx++)

            } // for (int Aidx = 0; Aidx < cart_ang_list[LA].size(); Aidx++)

          } // for (int Didx = 0; Didx < cart_ang_list[lD].size(); Didx++)

        } // for (int Cidx = 0; Cidx < cart_ang_list[lC].size(); Cidx++)

      } // for (int lC = LC; LC <= Lket - lD; lC++)

    } // for (int lD = 1; lD <= LD; lD++)

    // =============== Relativistic Horizontal Recursion =======================

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

                for (int ijk = 0; ijk < 3; ijk++) {

                  double ERIscratch = 0.0;
                  
                  int lAp1[3];

                  for (int ii = 0; ii < 3; ii++) {

                    lAp1[ii] = lA_xyz[ii];

                  } // for (int ii = 0; ii < 3; ii++)

                  lAp1[iwork] = lA_xyz[iwork] + 1;

                  ERIscratch
                    = Vbraket[ijk][(lA - LA + 1) * (Lket - LC + 1) + (lC - LC)][
                    (lB - 1) * (Lket - lC + 1)][
                    indexmap(lA + 1, lAp1[0], lAp1[1], lAp1[2])
                    * cart_ang_list[lB - 1].size() * cart_ang_list[lC].size()
                    + idxBtemp * cart_ang_list[lC].size() + Cidx]
                    + (AB[iwork])
                    * Vbraket[ijk][(lA - LA) * (Lket - LC + 1) + lC - LC][
                    (lB - 1) * (Lket - lC + 1)][
                    Aidx * cart_ang_list[lB - 1].size()
                         * cart_ang_list[lC].size()
                    + idxBtemp * cart_ang_list[lC].size()
                    + Cidx];

                    for (int u = 0; u < 3; u++) {

                      int lApu[3];
                      int lBpu[3];
            
                      for (int ii = 0; ii < 3; ii++) {

                        lApu[ii] = lA_xyz[ii];
                        lBpu[ii] = lB_xyz[ii];

                      } // for (int ii = 0; ii < 3; ii++) 

                        lAp1[u]++;
                        lBpu[iwork]--;
                        lBpu[u]++;
                        int idxButemp = indexmap(lB, lBpu[0], lBpu[1], lBpu[2]);
                        //std::cout << "Made it before cross call at 1785" << std::endl;
                        ERIscratch = ERIscratch
                          + cross(shift[iwork], shift[u])[ijk]
                          * (2 * alpha
                          * Hbraket[(lA - LA + 1) * (Lket - LC + 1) + (lC - LC)][
                          (lB - 1) * (Lket - lC + 1)][
                          indexmap(lA + 1, lAp1[0], lAp1[1], lAp1[2])
                          * cart_ang_list[lB - 1].size() * cart_ang_list[lC].size()
                          + idxBtemp * cart_ang_list[lC].size() + Cidx]
                          + 2 * beta
                          * Hbraket[(lA - LA) * (Lket - LC + 1) + (lC - LC)][
                            lB * (Lket - lC + 1)][
                            Aidx * cart_ang_list[lB].size() * cart_ang_list[lC].size()
                            + idxButemp * cart_ang_list[lC].size()
                            + Cidx]);
                        //std::cout << "Made it past cross call at 1785" << std::endl;
                        if (lA_xyz[u] > 0) {

                          int lAtemp[3];

                          for (int ii = 0; ii < 3; ii++) {

                            lAtemp[ii] = lA_xyz[ii];

                          } // for (int ii = 0; ii < 3; ii++)
                          
                          if (lA - LA - 1 < 0) {

                            lAtemp[u]--;
                            int indexlmuxyz = indexmap(lA - 1, lAtemp[0], lAtemp[1], lAtemp[2]);
                            //std::cout << "Made it before cross call at 1814" << std::endl;
                            ERIscratch = ERIscratch + cross(shift[iwork], shift[u])[ijk] 
                              * (-lA_xyz[u])
                              * Hbraketm1[lC - LC][lB - 1][
                              indexlmuxyz * cart_ang_list[lB - 1].size() 
                              * cart_ang_list[lC].size()
                              + idxBtemp * cart_ang_list[lC].size()
                              + Cidx];
                            //std::cout << "Made it past cross call at 1814" << std::endl;
                          } else { // if (lA - LA - 1 < 0) {

                            lAtemp[u]--;
                            int indexlmuxyz = indexmap(lA - 1, lAtemp[0], lAtemp[1], lAtemp[2]);
                            //std::cout << "Made it before cross call at 1827" << std::endl;
                            ERIscratch = ERIscratch + cross(shift[iwork], shift[u])[ijk] 
                              * (-lA_xyz[u])
                              * Hbraket[(lA - LA - 1) * (Lket - LC + 1) + (lC - LC)][
                              (lB - 1) * (Lket - lC + 1)][
                              indexlmuxyz * cart_ang_list[lB - 1].size() * cart_ang_list[lC].size()
                              + idxBtemp * cart_ang_list[lC].size()
                              + Cidx];
                            //std::cout << "Made it past cross call at 1827" << std::endl;
                          } // else {

                        } // if (lA_xyz[u] > 0)

                          int lBtemp[3];

                          for (int ii = 0; ii < 3; ii++) {

                            lBtemp[ii] = lB_xyz[ii];

                          } // for (int ii = 0; ii < 3; ii++)

                          lBtemp[iwork]--;

                        if (lBtemp[u] > 0) {

                            lBtemp[u]--;
                            int indexlbmuxyz = indexmap(lB - 2, lBtemp[0], lBtemp[1], lBtemp[2]);
                            //std::cout << "Made it before cross call at 1854" << std::endl;
                            ERIscratch = ERIscratch + cross(shift[iwork], shift[u])[ijk]
                              * (-lB_xyz[u])
                              * Hbraket[(lA - LA) * (Lket - LC + 1) + (lC - LC)][
                              (lB - 2) * (Lket - lC + 1)][
                              Aidx * cart_ang_list[lB - 2].size() * cart_ang_list[lC].size()
                              + indexlbmuxyz * cart_ang_list[lC].size()
                              + Cidx];
                            //std::cout << "Made it past cross call at 1854" << std::endl;
                        } // if (lAtemp[u] > 0)

                    } // for (int u = 0; u < 3; u++) 

                    Vbraket[ijk][(lA - LA) * (Lket - LC + 1) + (lC - LC)][
                      lB * (Lket - lC + 1)][
                      Aidx * cart_ang_list[lB].size() * cart_ang_list[lC].size()
                      + Bidx * cart_ang_list[lC].size()
                      + Cidx] = ERIscratch;

                  } // for (int ijk = 0; ijk < 3; ijk++) 

                  int lAp1[3];

                  for (int ii = 0; ii < 3; ii++) {

                    lAp1[ii] = lA_xyz[ii];

                  } // for (int ii = 0; ii < 3; ii++)

                  lAp1[iwork] = lA_xyz[iwork] + 1;

                  double ERIscratch = 0.0;
                  ERIscratch
                    = Vbraket[3][(lA - LA + 1) * (Lket - LC + 1) + (lC - LC)][
                    (lB - 1) * (Lket - lC + 1)][
                    indexmap(lA + 1, lAp1[0], lAp1[1], lAp1[2])
                    * cart_ang_list[lB - 1].size() * cart_ang_list[lC].size()
                    + idxBtemp * cart_ang_list[lC].size() + Cidx]
                    + (AB[iwork])
                    * Vbraket[3][(lA - LA) * (Lket - LC + 1) + lC - LC][
                    (lB - 1) * (Lket - lC + 1)][
                    Aidx * cart_ang_list[lB - 1].size()
                         * cart_ang_list[lC].size()
                    + idxBtemp * cart_ang_list[lC].size()
                    + Cidx];

                    for (int u = 0; u < 3; u++) {

                      int lApu[3];
                      int lBpu[3];

                      for (int ii = 0; ii < 3; ii++) {

                        lApu[ii] = lA_xyz[ii];
                        lBpu[ii] = lB_xyz[ii];

                      } // for (int ii = 0; ii < 3; ii++) 

                        lAp1[u]++;
                        lBpu[iwork]--;
                        lBpu[u]++;
                        int idxButemp = indexmap(lB, lBpu[0], lBpu[1], lBpu[2]);

                        ERIscratch = ERIscratch
                          + dot(shift[iwork], shift[u])
                          * (2 * alpha
                          * Hbraket[(lA - LA + 1) * (Lket - LC + 1) + (lC - LC)][
                          (lB - 1) * (Lket - lC + 1)][
                          indexmap(lA + 1, lAp1[0], lAp1[1], lAp1[2])
                          * cart_ang_list[lB - 1].size() * cart_ang_list[lC].size()
                          + idxBtemp * cart_ang_list[lC].size() + Cidx]
                          + 2 * beta
                          * Hbraket[(lA - LA) * (Lket - LC + 1) + (lC - LC)][
                            lB * (Lket - lC + 1)][
                            Aidx * cart_ang_list[lB].size() * cart_ang_list[lC].size()
                            + idxButemp * cart_ang_list[lC].size()
                            + Cidx]);

                        if (lA_xyz[u] > 0) {
                  
                          int lAtemp[3];

                          for (int ii = 0; ii < 3; ii++) {

                            lAtemp[ii] = lA_xyz[ii];

                          } // for (int ii = 0; ii < 3; ii++)

                          if (lA - LA - 1 < 0) {
                            
                            lAtemp[u]--;
                            int indexlmuxyz = indexmap(lA - 1, lAtemp[0], lAtemp[1], lAtemp[2]);
                            ERIscratch = ERIscratch + dot(shift[iwork], shift[u]) 
                              * (-lA_xyz[u])
                              * Hbraketm1[lC - LC][lB - 1][
                              indexlmuxyz * cart_ang_list[lB - 1].size() 
                              * cart_ang_list[lC].size()
                              + idxBtemp * cart_ang_list[lC].size()
                              + Cidx];

                          } else { // if (lA - LA - 1 < 0) {
                            lAtemp[u]--;
                            int indexlmuxyz = indexmap(lA - 1, lAtemp[0], lAtemp[1], lAtemp[2]);
                            ERIscratch = ERIscratch + dot(shift[iwork], shift[u]) 
                              * (-lA_xyz[u])
                              * Hbraket[(lA - LA - 1) * (Lket - LC + 1) + (lC - LC)][
                              (lB - 1) * (Lket - lC + 1)][
                              indexlmuxyz * cart_ang_list[lB - 1].size() * cart_ang_list[lC].size()
                              + idxBtemp * cart_ang_list[lC].size()
                              + Cidx];

                          } // else {

                        } // if (lAtemp[u] > 0) 

                          int lBtemp[3];

                          for (int ii = 0; ii < 3; ii++) {

                            lBtemp[ii] = lB_xyz[ii];

                          } // for (int ii = 0; ii < 3; ii++) 

                          lBtemp[iwork]--;

                        if (lBtemp[u] > 0) {

                            lBtemp[u]--;
                            int indexlbmuxyz = indexmap(lB - 2, lBtemp[0], lBtemp[1], lBtemp[2]);
                            ERIscratch = ERIscratch + dot(shift[iwork], shift[u])
                              * (-lB_xyz[u])
                              * Hbraket[(lA - LA) * (Lket - LC + 1) + (lC - LC)][
                              (lB - 2) * (Lket - lC + 1)][
                              Aidx * cart_ang_list[lB - 2].size() * cart_ang_list[lC].size()
                              + indexlbmuxyz * cart_ang_list[lC].size()
                              + Cidx];

                        } // if (lAtemp[u] > 0) 
                    } // for (int u = 0; u < 3; u++) 

                    Vbraket[3][(lA - LA) * (Lket - LC + 1) + (lC - LC)][
                      lB * (Lket - lC + 1)][
                      Aidx * cart_ang_list[lB].size() * cart_ang_list[lC].size()
                      + Bidx * cart_ang_list[lC].size()
                      + Cidx] = ERIscratch;

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

                for (int ijk = 0; ijk < 4; ijk++) {

                  Vbraket[ijk][lC - LC][LB * (Lket - lC + 1) + lD][
                    Aidx * cart_ang_list[LB].size()
                    * cart_ang_list[lC].size()
                    * cart_ang_list[lD].size()
                    + Bidx * cart_ang_list[lC].size()* cart_ang_list[lD].size()
                    + Cidx * cart_ang_list[lD].size()
                    + Didx]
                    = Vbraket[ijk][lC - LC + 1][
                    LB * (Lket - (lC + 1) + 1) + (lD - 1)][
                    Aidx * cart_ang_list[LB].size()
                      * cart_ang_list[lC + 1].size()
                      * cart_ang_list[lD - 1].size()
                    + Bidx * cart_ang_list[lC + 1].size()
                      * cart_ang_list[lD - 1].size()
                    + indexmap(lC + 1, lCp1[0], lCp1[1], lCp1[2])
                      * cart_ang_list[lD - 1].size()
                    + idxDtemp]
                    + (CD[iwork]) * Vbraket[ijk][lC - LC][
                    LB * (Lket - lC + 1) + (lD - 1)][
                    Aidx * cart_ang_list[LB].size()
                    * cart_ang_list[lC].size()
                    * cart_ang_list[lD - 1].size()
                    + Bidx * cart_ang_list[lC].size()
                    * cart_ang_list[lD - 1].size()
                    + Cidx * cart_ang_list[lD - 1].size()
                    + idxDtemp];
                  //std::cout << "Made it to line 2111" << std::endl;
                } // for (int ijk = 0; ijk < 3; ijk++) 

              } // for (int Bidx = 0; Bidx < cart_ang_list[LB].size(); Bidx++)

            } // for (int Aidx = 0; Aidx < cart_ang_list[LA].size(); Aidx++)

          } // for (int Didx = 0; Didx < cart_ang_list[lD].size(); Didx++)

        } // for (int Cidx = 0; Cidx < cart_ang_list[lC].size(); Cidx++)

      } // for (int lC = LC; LC <= Lket - lD; lC++)

    } // for (int lD = 1; lD <= LD; lD++)

    for (int ii = 0; ii < (2*shell1.contr[0].l+1)*(2*shell2.contr[0].l+1)
                          *(2*shell3.contr[0].l+1)*(2*shell4.contr[0].l+1); ii++) {

      Finalbraket[3][0][LB * (LD + 1) + LD][ii] += Vbraket[3][0][LB * (LD + 1) + LD][ii];
      Finalbraket[0][0][LB * (LD + 1) + LD][ii] += Vbraket[0][0][LB * (LD + 1) + LD][ii];
      Finalbraket[1][0][LB * (LD + 1) + LD][ii] += Vbraket[1][0][LB * (LD + 1) + LD][ii];
      Finalbraket[2][0][LB * (LD + 1) + LD][ii] += Vbraket[2][0][LB * (LD + 1) + LD][ii];

    }

          } // for ( auto &pripair2 : pair2.primpairs )

    }  // for ( auto &pripair1 : pair1.primpairs )


    std::vector<std::vector<double>> ERI_cart((2*shell1.contr[0].l+1)*(2*shell2.contr[0].l+1)
                                              *(2*shell3.contr[0].l+1)*(2*shell4.contr[0].l+1));

    for (int ii = 0; ii < (2*shell1.contr[0].l+1)*(2*shell2.contr[0].l+1)
                          *(2*shell3.contr[0].l+1)*(2*shell4.contr[0].l+1); ii++) {

      ERI_cart[ii].resize(4);

    }

    for (int ii = 0; ii < (2*shell1.contr[0].l+1)*(2*shell2.contr[0].l+1)
                          *(2*shell3.contr[0].l+1)*(2*shell4.contr[0].l+1); ii++) {

      ERI_cart[ii][0] = Finalbraket[3][0][LB * (LD + 1) + LD][ii];
      ERI_cart[ii][1] = Finalbraket[0][0][LB * (LD + 1) + LD][ii];
      ERI_cart[ii][2] = Finalbraket[1][0][LB * (LD + 1) + LD][ii];
      ERI_cart[ii][3] = Finalbraket[2][0][LB * (LD + 1) + LD][ii];

    }

    if ( ( not shell1.contr[0].pure ) and ( not shell2.contr[0].pure ) and
         ( not shell3.contr[0].pure ) and ( not shell4.contr[0].pure ) ) {
      // if both sides are cartesian, return cartesian gaussian integrals
      return ERI_cart;
    }

    std::vector<std::vector<double>> ERI_sph((2*shell1.contr[0].l+1)*(2*shell2.contr[0].l+1)
                                             *(2*shell3.contr[0].l+1)*(2*shell4.contr[0].l+1));

    for (int ii = 0; ii < (2*shell1.contr[0].l+1)*(2*shell2.contr[0].l+1)
                          *(2*shell3.contr[0].l+1)*(2*shell4.contr[0].l+1); ii++) {

      ERI_sph[ii].resize(4);

    }

    std::vector<double> ERI_sph0;
    std::vector<double> ERI_sph1;
    std::vector<double> ERI_sph2;
    std::vector<double> ERI_sph3;
    ERI_sph0.assign(((2*shell1.contr[0].l+1)*(2*shell2.contr[0].l+1)
                   *(2*shell3.contr[0].l+1)*(2*shell4.contr[0].l+1)),0.0);
    ERI_sph1.assign(((2*shell1.contr[0].l+1)*(2*shell2.contr[0].l+1)
                   *(2*shell3.contr[0].l+1)*(2*shell4.contr[0].l+1)),0.0);
    ERI_sph2.assign(((2*shell1.contr[0].l+1)*(2*shell2.contr[0].l+1)
                   *(2*shell3.contr[0].l+1)*(2*shell4.contr[0].l+1)),0.0);
    ERI_sph3.assign(((2*shell1.contr[0].l+1)*(2*shell2.contr[0].l+1)
                   *(2*shell3.contr[0].l+1)*(2*shell4.contr[0].l+1)),0.0);
    cart2sph_2e_transform(shell1.contr[0].l,shell2.contr[0].l,
                          shell3.contr[0].l,shell4.contr[0].l,
                          ERI_sph0,Finalbraket[0][0][LB * (LD + 1) + LD]);
    cart2sph_2e_transform(shell1.contr[0].l,shell2.contr[0].l,
                          shell3.contr[0].l,shell4.contr[0].l,
                          ERI_sph1,Finalbraket[1][0][LB * (LD + 1) + LD]);
    cart2sph_2e_transform(shell1.contr[0].l,shell2.contr[0].l,
                          shell3.contr[0].l,shell4.contr[0].l,
                          ERI_sph2,Finalbraket[2][0][LB * (LD + 1) + LD]);
    cart2sph_2e_transform(shell1.contr[0].l,shell2.contr[0].l,
                          shell3.contr[0].l,shell4.contr[0].l,
                          ERI_sph3,Finalbraket[3][0][LB * (LD + 1) + LD]);


    for (int ii = 0; ii < (2*shell1.contr[0].l+1)*(2*shell2.contr[0].l+1)
                          *(2*shell3.contr[0].l+1)*(2*shell4.contr[0].l+1); ii++) {

      ERI_sph[ii][1] = ERI_sph0[ii];
      ERI_sph[ii][2] = ERI_sph1[ii];
      ERI_sph[ii][3] = ERI_sph2[ii];
      ERI_sph[ii][0] = ERI_sph3[ii];

    }

    return ERI_sph;

  }  // BottomupHGP

  double * cross(double *A, double *B) {

    static double result[3];
    result[0] = A[1] * B[2] - A[2] * B[1];
    result[1] = A[2] * B[0] - A[0] * B[2];
    result[2] = A[0] * B[1] - A[1] * B[0];

    return result;

  }

  int * cross(int *A, int *B) {

    static int result[3];
    result[0] = A[1] * B[2] - A[2] * B[1];
    result[1] = A[2] * B[0] - A[0] * B[2];
    result[2] = A[0] * B[1] - A[1] * B[0];

    return result;


  }

  double dot(double *A, double *B) {
  
    static double result;
    result = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
    
    return result;

  }

  int dot(int *A, int *B) {
  
    static int result;
    result = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
    
    return result;

  }

}  // namespace ChronusQ
