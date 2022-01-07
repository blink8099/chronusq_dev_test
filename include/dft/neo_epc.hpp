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

#include <dft.hpp>

namespace ChronusQ {
  /**
   *  \brief EPC-17 functional
   */
  class EPC17 : public DFTFunctional {

  public:
    EPC17(std::string epcName) : DFTFunctional(epcName) 
    { 
      this->isGGA_ = false; 
      this->isEPC_ = true; 
    };

    void evalEXC_VXC(size_t N, double *rho, double *aux_rho, double *eps, double *vxc, bool electron) {

      // loop over grid points
      for(auto iPt = 0; iPt < N; iPt++) {

        // for electron, augment the current exc and vxc
        if (electron) {

          // total electron density 
          double total_erho = std::abs(rho[2*iPt]+rho[2*iPt+1]) > 1e-15 ? rho[2*iPt]+rho[2*iPt+1] : 0.0; 
          // total proton density 
          double total_prho = std::abs(aux_rho[2*iPt]+aux_rho[2*iPt+1]) > 1e-15 ? aux_rho[2*iPt]+aux_rho[2*iPt+1] : 0.0; 

          // skip this point if the density is too small
          if(total_erho < 1e-15 or total_prho < 1e-15) continue; 

          // epc17-2 denominator
          double dn = 2.35 - 2.4 * std::sqrt(total_erho * total_prho) + 6.6 * (total_erho * total_prho);

          //std::cout << eps[iPt] << std::endl;
          eps[iPt]     += -1.0 * total_prho / dn;

          vxc[2*iPt]   += ( -1.0 * total_prho / dn + (-1.2 * std::sqrt(total_erho) * std::sqrt(total_prho) * total_prho 
                       +     6.6 * total_erho * total_prho * total_prho ) / (dn * dn) );
          vxc[2*iPt+1] += ( -1.0 * total_prho / dn + (-1.2 * std::sqrt(total_erho) * std::sqrt(total_prho) * total_prho 
                       + 6.6 * total_erho * total_prho * total_prho ) / (dn * dn) );

          if (std::isnan(vxc[2*iPt]))
            std::cout << total_prho << " " << std::sqrt(total_erho) << "  " << std::sqrt(total_prho) << std::endl;

        }
        // for proton wave function, the Vxc is zero, so we directly assign EPC to Vxc
        else {

          double total_erho = std::abs(aux_rho[2*iPt]+aux_rho[2*iPt+1]) > 1e-15 ? aux_rho[2*iPt]+aux_rho[2*iPt+1] : 0.0;
          // proton total density 
          double total_prho = std::abs(rho[2*iPt]+rho[2*iPt+1]) > 1e-15 ? rho[2*iPt]+rho[2*iPt+1] : 0.0;

          // skip this ppint if density is too small
          if(total_erho < 1e-15 or total_prho < 1e-15) {
            eps[iPt] = 0.;
            vxc[2*iPt] = 0.;
            vxc[2*iPt+1] = 0.;

            continue;
          }

          // epc17-2 denominator
          double dn = 2.35 - 2.4 * std::sqrt(total_erho * total_prho) + 6.6 * (total_erho * total_prho);
          eps[iPt] = -1.0 * total_erho / dn;

          vxc[2*iPt] = ( -1.0 * total_erho / dn + (-1.2 * std::sqrt(total_prho) * std::sqrt(total_erho) * total_erho 
                     +    6.6 * total_erho * total_erho * total_prho ) / (dn * dn) );

          vxc[2*iPt+1] = 0.0;
        }
      }
    } // epc17-2 functional

  }; 

  /**
   *  \brief EPC-19 functional
   *
   *  Ref: J. Chem. Phys. 151, 124102 (2019)
   */
  class EPC19 : public DFTFunctional {

  public:
    EPC19(std::string epcName) : DFTFunctional(epcName) 
    { 
      this->isGGA_ = true; 
      this->isEPC_ = true;
    };

    void evalEXC_VXC(size_t N, double *rho, double *aux_rho, double *sigma, double *aux_sigma,
                     double *cross_sigma, double *eps, double *vrho, double *vsigma, 
                     double *vcsigma, bool electron)
    {

      // parameters
      double a = 1.9, b = 1.3, c = 8.1, d = 1600.0, q = 8.2;

      // proton mass 
      double pMass = 1836.152676;

      // lambda function that computes X and its derivatives
      auto computeX = [&](double rho_e, double rho_p, std::vector<double> & outvec) {
        // size output vector correctly
        if(outvec.size() != 6)
          outvec.resize(6);

        // denominator
        double de = a - b * std::sqrt(rho_e * rho_p) + c * rho_e * rho_p;
        outvec[0] = 1.0 / de;

        // first-derivatives
        outvec[1] = ( 0.5 * b * pow(rho_e,-0.5) * std::sqrt(rho_p) - c * rho_p ) / (de * de);
        outvec[2] = ( 0.5 * b * pow(rho_p,-0.5) * std::sqrt(rho_e) - c * rho_e ) / (de * de);

        // second-derivatives
        double nu1 = -a * b / 4 * pow(rho_e, -1.5) * std::sqrt(rho_p);
        double nu2 =  3 * b * b / 4 / rho_e * rho_p;
        double nu3 = -9 * b * c / 4.0 * pow(rho_e, -0.5) * pow(rho_p, 1.5);
        double nu4 =  2 * c * c * rho_p * rho_p;
        double nu = nu1 + nu2 + nu3 + nu4;
        outvec[3] = nu / (de * de * de);  // ee

        // compute ep
        nu1 =  a * b / 4 * pow(rho_e * rho_p, -0.5);
        nu2 = -a * c + b * b / 4;
        nu3 = -3.0 * b * c  / 4 * std::sqrt(rho_e * rho_p);
        nu4 =  c * c * rho_e * rho_p;
        nu = nu1 + nu2 + nu3 + nu4;
        outvec[4] = nu / (de * de * de); // ep

        // compute pp
        nu1 = -a * b / 4 * pow(rho_p, -1.5) * std::sqrt(rho_e);
        nu2 =  3 * b * b / 4 / rho_p * rho_e;
        nu3 = -9 * b * c / 4.0 * pow(rho_p, -0.5) * pow(rho_e, 1.5);
        nu4 =  2 * c * c * rho_e * rho_e;
        nu = nu1 + nu2 + nu3 + nu4;
        outvec[5] = nu / (de * de * de);  // pp

      };

      // lambda function that computes Y0 and its derivatives
      auto computeY0 = [&](double rho_e, double rho_p, std::vector<double> & outvec) {
        // size output vector correctly
        if(outvec.size() != 6)
          outvec.resize(6);

        outvec[0] = rho_e * rho_p;
        outvec[1] = rho_p;
        outvec[2] = rho_e;

        outvec[3] = 0.0;
        outvec[4] = 1.0;
        outvec[5] = 0.0;

      };

      // lambda function that computes Y1
      auto computeY1 = [&](double rho_e, double rho_p, std::vector<double> & outvec) {
        // size output vector correctly
        if(outvec.size() != 6)
          outvec.resize(6);       
        
        outvec[0] = pMass * pMass * pow(rho_e, -1.0 / 3) * pow(rho_p, 2.0 / 3) / ((1 + pMass) * (1 + pMass));
        outvec[1] = -1.0 / 3 * outvec[0] / rho_e;
        outvec[2] =  2.0 / 3 * outvec[0] / rho_p;

        // second-order derivatives
        outvec[3] =  4.0 / 9 * outvec[0] / (rho_e * rho_e); 
        outvec[4] = -2.0 / 9 * outvec[0] / (rho_e * rho_p);
        outvec[5] =  0.0;
      };

      // lambda function that computes Y2
      auto computeY2 = [&](double rho_e, double rho_p, std::vector<double> & outvec) {
        // size output vector correctly
        if(outvec.size() != 3)
          outvec.resize(3);       

        outvec[0] =  2.0 * pMass * pow(rho_e * rho_p, -1.0 / 3) / ((1 + pMass) * (1 + pMass));
        outvec[1] = -1.0 / 3 * outvec[0] / rho_e;
        outvec[2] = -1.0 / 3 * outvec[0] / rho_p;
      };

      // lambda function that computes Y3
      auto computeY3 = [&](double rho_e, double rho_p, std::vector<double> & outvec) {
        // size output vector correctly
        if(outvec.size() != 6)
          outvec.resize(6);       

        outvec[0] = pow(rho_e, 2.0 / 3) * pow(rho_p, -1.0 / 3) / ((1 + pMass) * (1 + pMass));
        outvec[1] =  2.0 / 3 * outvec[0] / rho_e;
        outvec[2] = -1.0 / 3 * outvec[0] / rho_p;

        // second-order derivatives
        outvec[3] =  0.0;
        outvec[4] = -2.0 / 9 * outvec[0] / (rho_e * rho_p);
        outvec[5] =  4.0 / 9 * outvec[0] / (rho_p * rho_p);
      };

      // lambda function that computes Z
      auto computeZ = [&](double rho_e, double rho_p, std::vector<double> & outvec) {
        // size output vector correctly
        if(outvec.size() != 6)
          outvec.resize(6);       

        double factor = -1.0 * q / pow(rho_e * rho_p, 1.0 / 6);
        outvec[0] = exp(factor);
        outvec[1] = outvec[0] * q / (6.0 * rho_e * pow(rho_e * rho_p, 1.0 / 6));
        outvec[2] = outvec[0] * q / (6.0 * rho_p * pow(rho_e * rho_p, 1.0 / 6));

        // second-order derivatives
        outvec[3] = outvec[0] * q * q / (36 * rho_e * rho_e * pow(rho_e * rho_p, 1.0 / 3)) - 7 * outvec[0] * q / (36 * rho_e * rho_e * pow(rho_e * rho_p, 1.0 / 6));
        outvec[4] = outvec[0] * q * q / (36 * rho_e * rho_p * pow(rho_e * rho_p, 1.0 / 3)) - outvec[0] * q / (36 * rho_e * rho_p * pow(rho_e * rho_p, 1.0 / 6));
        outvec[5] = outvec[0] * q * q / (36 * rho_p * rho_p * pow(rho_e * rho_p, 1.0 / 3)) - 7 * outvec[0] * q / (36 * rho_p * rho_p * pow(rho_e * rho_p, 1.0 / 6));
      };

      // lambda function that compites Ge
      auto computeGe = [&](std::vector<double> & X, std::vector<double> & Y1, std::vector<double> & Z, std::vector<double> & outvec) {
        // size output vector correctly
        if(outvec.size() != 6)
          outvec.resize(6);       

        outvec[0] = -d * X[0] * Y1[0] * Z[0];
        outvec[1] = -d * (X[1] * Y1[0] * Z[0] + X[0] * Y1[1] * Z[0] + X[0] * Y1[0] * Z[1]);
        outvec[2] = -d * (X[2] * Y1[0] * Z[0] + X[0] * Y1[2] * Z[0] + X[0] * Y1[0] * Z[2]);

        // second-order derivatives
        outvec[3] = -d * (Y1[0] * X[3] * Z[0] + X[0] * Y1[3] * Z[0] + X[0] * Y1[0] * Z[3]
                    + 2 * X[1] * Y1[1] * Z[0] + 2 * X[1] * Y1[0] * Z[1] + 2 * X[0] * Y1[1] * Z[1]);

        outvec[4] = -d * (Y1[0] * X[4] * Z[0] + X[0] * Y1[4] * Z[0] + X[0] * Y1[0] * Z[4]
                    + X[1] * Y1[2] * Z[0] + X[2] * Y1[1] * Z[0] 
                    + X[1] * Y1[0] * Z[2] + X[2] * Y1[0] * Z[1]
                    + X[0] * Y1[2] * Z[1] + X[0] * Y1[1] * Z[2]);

        outvec[5] = 0.0;
        
      };

      // lambda function that computes F0
      auto computeF0 = [&](std::vector<double> & X, std::vector<double> & Y0, std::vector<double> & Y2, std::vector<double> & Z, double csigma, std::vector<double> & outvec) {
        // size output vector correctly
        if(outvec.size() != 3)
          outvec.resize(3);       

        outvec[0] = X[0] * Y0[0] + d * X[0] * Y2[0] * csigma * Z[0];
        outvec[1] = Y0[0] * X[1] + X[0] * Y0[1] + d * csigma * (Y2[0] * X[1] * Z[0] 
                  + X[0] * Y2[1] * Z[0] + X[0] * Y2[0] * Z[1]);
        outvec[2] = Y0[0] * X[2] + X[0] * Y0[2] + d * csigma * (Y2[0] * X[2] * Z[0] 
                  + X[0] * Y2[2] * Z[0] + X[0] * Y2[0] * Z[2]);
      };

      // lambda function that compites Gp
      auto computeGp = [&](std::vector<double> & X, std::vector<double> & Y3, std::vector<double> & Z, std::vector<double> & outvec) {
        // size output vector correctly
        if(outvec.size() != 6)
          outvec.resize(6);       

        outvec[0] = -d * X[0] * Y3[0] * Z[0];
        outvec[1] = -d * (X[1] * Y3[0] * Z[0] + X[0] * Y3[1] * Z[0] + X[0] * Y3[0] * Z[1]);
        outvec[2] = -d * (X[2] * Y3[0] * Z[0] + X[0] * Y3[2] * Z[0] + X[0] * Y3[0] * Z[2]);

        // second-order derivatives
        outvec[5] = -d * (Y3[0] * X[5] * Z[0] + X[0] * Y3[5] * Z[0] + X[0] * Y3[0] * Z[5]
                    + 2 * X[2] * Y3[2] * Z[0] + 2 * X[2] * Y3[0] * Z[2] + 2 * X[0] * Y3[2] * Z[2]);

        outvec[4] = -d * (Y3[0] * X[4] * Z[0] + X[0] * Y3[4] * Z[0] + X[0] * Y3[0] * Z[4]
                    + X[1] * Y3[2] * Z[0] + X[2] * Y3[1] * Z[0] 
                    + X[1] * Y3[0] * Z[2] + X[2] * Y3[0] * Z[1]
                    + X[0] * Y3[2] * Z[1] + X[0] * Y3[1] * Z[2]);

        outvec[3] = 0.0;
        
      };


      // loop over grid points 
      for(auto iPt = 0; iPt < N; iPt++)  { 
        // for electron wave function, we need to assume that Vxc have already been computed, so we add the EPC contribution to it
        if (electron) {

          // skip this point if density is too small
          if (std::abs(rho[2*iPt]+rho[2*iPt+1]) < 1e-15 or std::abs(aux_rho[2*iPt]+aux_rho[2*iPt+1]) < 1e-15) {
            eps[iPt] += 0.;
            vrho[2*iPt] += 0.;
            vrho[2*iPt+1] += 0.;

            // compute vsigma
            vsigma[3*iPt]   += 0.0; 
            vsigma[3*iPt+1] += 0.0;
            vsigma[3*iPt+2] += 0.0;

            // compute vcsigma
            vcsigma[4*iPt]   = 0.0;
            vcsigma[4*iPt+1] = 0.0;
            vcsigma[4*iPt+2] = 0.0;
            vcsigma[4*iPt+3] = 0.0;

            continue;
          }
            

          // electron total density 
          double total_erho = std::abs(rho[2*iPt]+rho[2*iPt+1]) > 1e-15 ? rho[2*iPt]+rho[2*iPt+1] : 0.0;
          // proton total density 
          double total_prho = std::abs(aux_rho[2*iPt]+aux_rho[2*iPt+1]) > 1e-15 ? aux_rho[2*iPt]+aux_rho[2*iPt+1] : 0.0;

          // X and derivatives
          std::vector<double> X_vec, Y0_vec, Y1_vec, Y2_vec, Y3_vec, Z_vec;

          // non-gradient pieces
          computeX(total_erho,  total_prho, X_vec);
          computeY0(total_erho, total_prho, Y0_vec);
          computeY1(total_erho, total_prho, Y1_vec);
          computeY2(total_erho, total_prho, Y2_vec);
          computeY3(total_erho, total_prho, Y3_vec);
          computeZ(total_erho,  total_prho, Z_vec);

          // compute Ge and Gp
          std::vector<double> F0, Ge, Gp;
          computeGe(X_vec, Y1_vec, Z_vec, Ge);
          computeGp(X_vec, Y3_vec, Z_vec, Gp);

          // electron total density gradient
          double total_egrad = sigma[3*iPt] + sigma[3*iPt+2] + 2*sigma[3*iPt+1];

          // proton total density gradient
          double total_pgrad = aux_sigma[3*iPt] + aux_sigma[3*iPt+2] + 2*aux_sigma[3*iPt+1];

          // total cross density gradient
          double total_cgrad = cross_sigma[4*iPt] + cross_sigma[4*iPt+1] + cross_sigma[4*iPt+2] + cross_sigma[4*iPt+3]; 

          // compute F0
          computeF0(X_vec, Y0_vec, Y2_vec, Z_vec, total_cgrad, F0);

          // eps (energy per unit particle) 
          eps[iPt]  += -1.0 * (F0[0] - Ge[1] * total_egrad - Gp[2] * total_pgrad) / total_erho;

          // compute vrho
          double local_vrho = -1.0 * (F0[1] - Ge[3] * total_egrad - Gp[4] * total_pgrad);
          vrho[2*iPt]   += local_vrho;
          vrho[2*iPt+1] += local_vrho;

          // compute vsigma
          vsigma[3*iPt]   += Ge[1]; // aa
          vsigma[3*iPt+1] += 2.0 * Ge[1]; // ab
          vsigma[3*iPt+2] += Ge[1]; // bb

          // compute vcsigma
          vcsigma[4*iPt]   = -1.0 * X_vec[0] * ( d * Y2_vec[0] * Z_vec[0]); // a(e) * a(p)
          vcsigma[4*iPt+1] =  0.0; // a(e) * b(p)
          vcsigma[4*iPt+2] = -1.0 * X_vec[0] * ( d * Y2_vec[0] * Z_vec[0]); // b(e) * a(p)
          vcsigma[4*iPt+3] =  0.0; // b(e) * b(p)

        }
        // for proton wave function, the Vxc is zero, so we directly assign EPC to Vxc
        else {
          double total_erho = std::abs(aux_rho[2*iPt]+aux_rho[2*iPt+1]) > 1e-15 ? aux_rho[2*iPt]+aux_rho[2*iPt+1] : 0.0;
          // proton total density 
          double total_prho = std::abs(rho[2*iPt]+rho[2*iPt+1]) > 1e-15 ? rho[2*iPt]+rho[2*iPt+1] : 0.0;

          // skip this point if density is too small
          if (std::abs(rho[2*iPt]+rho[2*iPt+1]) < 1e-15 or std::abs(aux_rho[2*iPt]+aux_rho[2*iPt+1]) < 1e-15) {
            eps[iPt] = 0.;
            vrho[2*iPt] = 0.;
            vrho[2*iPt+1] = 0.;

            // compute vsigma
            vsigma[3*iPt]   = 0.0; 
            vsigma[3*iPt+1] = 0.0;
            vsigma[3*iPt+2] = 0.0;

            // compute vcsigma
            vcsigma[4*iPt]   = 0.0;
            vcsigma[4*iPt+1] = 0.0;
            vcsigma[4*iPt+2] = 0.0;
            vcsigma[4*iPt+3] = 0.0;

            continue;
          }

          // X and derivatives
          std::vector<double> X_vec, Y0_vec, Y1_vec, Y2_vec, Y3_vec, Z_vec;

          // non-gradient pieces
          computeX(total_erho, total_prho, X_vec);
          computeY0(total_erho, total_prho, Y0_vec);
          computeY1(total_erho, total_prho, Y1_vec);
          computeY2(total_erho, total_prho, Y2_vec);
          computeY3(total_erho, total_prho, Y3_vec);
          computeZ(total_erho, total_prho, Z_vec);

          // compute Ge and Gp
          std::vector<double> F0, Ge, Gp;
          computeGe(X_vec, Y1_vec, Z_vec, Ge);
          computeGp(X_vec, Y3_vec, Z_vec, Gp);

          // electron total density gradient
          double total_egrad = aux_sigma[3*iPt] + aux_sigma[3*iPt+2] + 2*aux_sigma[3*iPt+1];

          // proton total density gradient
          double total_pgrad = sigma[3*iPt] + sigma[3*iPt+2] + 2*sigma[3*iPt+1];

          // total cross density gradient
          double total_cgrad = cross_sigma[4*iPt] + cross_sigma[4*iPt+1] + cross_sigma[4*iPt+2] + cross_sigma[4*iPt+3]; 

          // compute F0
          computeF0(X_vec, Y0_vec, Y2_vec, Z_vec, total_cgrad, F0);

          // eps (energy per unit particle)
          eps[iPt] = -1.0 * (F0[0] - Ge[1] * total_egrad - Gp[2] * total_pgrad) / total_prho;

          // compute vrho
          double local_vrho = -1.0 * (F0[2] - Ge[4] * total_egrad - Gp[5] * total_pgrad);
          vrho[2*iPt]   = local_vrho;
          vrho[2*iPt+1] = 0.;

          // compute vsigma
          vsigma[3*iPt]    = Gp[2]; // aa
          vsigma[3*iPt+1]  = 0.0; // ab
          vsigma[3*iPt+2]  = 0.0; // bb

          // compute vcsigma
          vcsigma[4*iPt]   = -1.0 * X_vec[0] * ( d * Y2_vec[0] * Z_vec[0]); // a(p) * a(e)
          vcsigma[4*iPt+1] = -1.0 * X_vec[0] * ( d * Y2_vec[0] * Z_vec[0]); // a(p) * b(e)
          if (std::isnan(vcsigma[4*iPt]))
            std::cout << "X " << X_vec[0] << " Y2 " << Y2_vec[0] << " Z " << Z_vec[0]  << std::endl;
          vcsigma[4*iPt+2] =  0.0; // b(p) * a(e)
          vcsigma[4*iPt+3] =  0.0; // b(p) * b(e)
        }
      }
    } // exc_vxc evaluation for EPC19

  }; // class EPC19

} // namespace ChronusQ
