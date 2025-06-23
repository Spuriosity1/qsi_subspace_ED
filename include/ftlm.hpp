#pragma once
#include "Spectra/MatOp/SparseGenMatProd.h"
#include "expectation_eval.hpp"
#include "krylov_matmul.hpp"
#include <cstdio>
#include <iostream>
#include <ostream>
#include <random>

inline void init_random(Eigen::VectorXd& v0, std::mt19937& rng){
        // choose arandom normal vector
        std::normal_distribution<double> n;
        for (Eigen::Index i=0; i<v0.size(); i++){
            v0[i] = n(rng);
        }
        v0.normalize();
}

template <typename OpType>
class ftlm_computer {
public:
    ftlm_computer(
        const std::vector<OpType>& _ops_to_evaluate,   
        const OpType& _H,
        double T_min, double T_max, int _n_Ts
        ):
        H(_H), n_Ts(_n_Ts) {
            // load all the operators as Spectra-inspired thin wrappers
            // Note that _ops_to_evaluate must stay in scope
            for (const auto& o : _ops_to_evaluate){
                operators.emplace_back(o);
            }
            beta_grid.reserve(_n_Ts);
            if ( T_min >= T_max ){
              throw std::runtime_error(
                  "Invalid spec: T_max must be greater htan T_min");
            }
            double beta_min = 1./T_max;
            double beta_max = 1./T_min;

            double beta=beta_min;
            double dBeta = (beta_max-beta_min) / (n_Ts - 1);
            for (int j=0; j<n_Ts; j++){
                beta_grid.push_back(beta);
                beta += dBeta;
            }

            tmp.resize(H.cols());
            op_expect_sum.resize(beta_grid.size()*operators.size());
            op_expect_2_sum.resize(beta_grid.size()*operators.size());
            memset(op_expect_sum.data(), 0, n_Ts * operators.size()*sizeof(double));
            memset(op_expect_2_sum.data(), 0, n_Ts * operators.size()*sizeof(double));
            psi.resize(_H.rows());
        }

    void evolve(std::mt19937& rng){
        init_random(psi, rng);

        double beta = beta_grid[0]/2;
        for (int n=0; n<beta_grid.size(); n++){
            // Current state: psi is e^-beta[n-1]/2 H * |u>
            double dbeta = beta_grid[n] - beta;
            beta = beta_grid[n];
            psi = krylov_expv(H, psi, -0.5 * dbeta, krylov_dim, tol);
            // Current state: |psi> is e^-beta[n]/2 H * |u>
            // evaluate <psi | psi>
            double denom=psi.squaredNorm();

            if (denom < 1e-10) {
                throw std::runtime_error("Numerical error: psi has disappeared");
            }

            // evaluate <psi | O | psi>
            for (int op_i=0; op_i < operators.size(); op_i++) {
                auto& o = operators[op_i];
                o.perform_op(psi.data(), tmp.data());
                double& exp_acc = op_expect_sum[n_Ts * op_i + n];
                double& exp2_acc = op_expect_2_sum[n_Ts * op_i + n];

                for (int i=0; i<psi.size(); i++){
                    // add on <psi | O | psi > / < psi | psi >
                    auto x = psi[i]*tmp[i];
                    exp_acc += x;
                    exp2_acc += x*x;
                }
                exp_acc /= denom;
                exp2_acc /= denom;

            }
        }
    }


    void set_numerical_params(
        int _krylov_dim = 30,
        double _tol = 1e-10){
        krylov_dim = _krylov_dim;
        tol = _tol;
    }
    
    const std::vector<double>& get_beta_grid() const { return beta_grid; }

    std::vector<double> get_temperature_grid() const {
        std::vector<double> temps;
        for (double beta : beta_grid) {
            temps.push_back(1.0 / beta);
        }
        return temps;
    }

    void write_to_h5(hid_t file_id, int op_idx, const std::string& dset_name){
        hsize_t dims[1] = {beta_grid.size()};

        write_dataset(file_id, (dset_name+"_sum").c_str(), 
                op_expect_sum.data() + op_idx* n_Ts,
                dims, 1);

        write_dataset(file_id, (dset_name+"_sum_2").c_str(), 
                op_expect_2_sum.data() + op_idx* n_Ts,
                dims, 1);

    }

    
private:
    size_t n_Ts;
    // storage for the psi val
    Eigen::VectorXd psi;
    std::vector<double> tmp;
    // numerical params for the time evolution
        int krylov_dim;
        double tol;
    // storage for the expectation vals and their squares
        std::vector<double> op_expect_sum;
        std::vector<double> op_expect_2_sum;
    //grid of betas, sorted ascending
        std::vector<double> beta_grid;
    // the Hamiltonian
        const Spectra::SparseGenMatProd<double> H;
    // the operators
        std::vector<Spectra::SparseGenMatProd<double>> operators;
};
