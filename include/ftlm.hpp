#pragma once
#include "expectation_eval.hpp"
#include "krylov_matmul.hpp"
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
        std::vector<double>& _T_grid
        ):
        H(_H), operators(_ops_to_evaluate), n_Ts(_T_grid.size()) {
            beta_grid.reserve(_T_grid.size());
            for (auto T : _T_grid){
                assert(T>0);
                beta_grid.push_back(1.0/T);
            }
            std::sort(beta_grid.begin(), beta_grid.end());

            tmp = new double[H.cols()];
            exp_vals = new double[beta_grid.size()*operators.size()];
            exp_vals_2 = new double[beta_grid.size()*operators.size()];
            memset(exp_vals, 0, n_Ts * operators.size()*sizeof(double));
            memset(exp_vals_2, 0, n_Ts * operators.size()*sizeof(double));
            psi.resize(_H.rows());
        }

    ~ftlm_computer(){
        delete[] exp_vals;
        delete[] exp_vals_2;

        delete[] tmp;
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
            double denom=0;
            for (int i=0; i<psi.size(); i++){denom += psi[i]*psi[i];}

            // evaluate <psi | O | psi>
            for (int op_i=0; op_i < operators.size(); op_i++) {
                auto& o = operators[op_i];
                o.perform_op(psi.data(), tmp);
                double& exp_acc = exp_vals[n_Ts * op_i + n];
                double& exp2_acc = exp_vals_2[n_Ts * op_i + n];
                for (int i=0; i<psi.size(); i++){
                    auto _t = psi[i]*tmp[i];
                    exp_acc += _t;
                    exp2_acc += _t*_t;
                }
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
                &exp_vals[op_idx* n_Ts],
                dims, 1);

        write_dataset(file_id, (dset_name+"_sum_2").c_str(), 
                &exp_vals_2[op_idx* n_Ts],
                dims, 1);

    }

    
private:
    size_t n_Ts;
    // storage for the psi val
    Eigen::VectorXd psi;
    double* tmp;
    // numerical params for the time evolution
        int krylov_dim;
        double tol;
    // storage for the expectation vals and their squares
        double* exp_vals;
        double* exp_vals_2;
    //grid of betas, sorted ascending
        std::vector<double> beta_grid;
    // the Hamiltonian
        const OpType& H;
    // the operators
        const std::vector<OpType>& operators;
};
