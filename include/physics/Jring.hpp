#pragma once 

#include <Eigen/Core>
#include <cmath>

inline Eigen::Matrix<double, 4, 3> get_loc_z(){
    Eigen::Matrix<double, 4, 3> local_z;
	local_z <<  1.,  1,  1,
				1., -1, -1,
			   -1.,  1, -1,
			   -1., -1,  1;
	local_z /= std::sqrt(3.0);
    return local_z;
}


inline std::vector<double> g_vals(double Jpm, const Eigen::Vector3d& B ) {

    double Bx = B[0];
    double By = B[1];
    double Bz = B[2];

    std::vector<double> J = {1. / 36 *
                (-21. / 4 * pow((Bx + (By + -1 * Bz)), 2) *
                     pow((Bx + (-1 * By + Bz)), 2) *
                     pow((-1 * Bx + (By + Bz)), 2) +
                 (-35. / 2 * (Bx + (-1 * By + -1 * Bz)) * (Bx + (By + -1 * Bz)) *
                      (Bx + (-1 * By + Bz)) * (Bx + (By + Bz)) * Jpm +
                  (-15 * pow((Bx + (By + Bz)), 2) * Jpm * Jpm + 54 * Jpm * Jpm * Jpm))),
            1. / 36 *
                (-21. / 4 * pow((Bx + (By + -1 * Bz)), 2) *
                     pow((Bx + (-1 * By + Bz)), 2) * pow((Bx + (By + Bz)), 2) +
                 (-35. / 2 * (Bx + (-1 * By + -1 * Bz)) * (Bx + (By + -1 * Bz)) *
                      (Bx + (-1 * By + Bz)) * (Bx + (By + Bz)) * Jpm +
                  (-15 * pow((-1 * Bx + (By + Bz)), 2) * Jpm * Jpm +
                   54 * Jpm * Jpm * Jpm))),
            1. / 36 *
                (-21. / 4 * pow((Bx + (By + -1 * Bz)), 2) *
                     pow((-1 * Bx + (By + Bz)), 2) * pow((Bx + (By + Bz)), 2) +
                 (-35. / 2 * (Bx + (-1 * By + -1 * Bz)) * (Bx + (By + -1 * Bz)) *
                      (Bx + (-1 * By + Bz)) * (Bx + (By + Bz)) * Jpm +
                  (-15 * pow((Bx + (-1 * By + Bz)), 2) * Jpm * Jpm +
                   54 * Jpm * Jpm * Jpm))),
            1. / 36 *
                (-21. / 4 * pow((Bx + (-1 * By + Bz)), 2) *
                     pow((-1 * Bx + (By + Bz)), 2) * pow((Bx + (By + Bz)), 2) +
                 (-35. / 2 * (Bx + (-1 * By + -1 * Bz)) * (Bx + (By + -1 * Bz)) *
                      (Bx + (-1 * By + Bz)) * (Bx + (By + Bz)) * Jpm +
                  (-15 * pow((Bx + (By + -1 * Bz)), 2) * Jpm * Jpm +
                   54 * Jpm * Jpm * Jpm)))
    };

    return J;
}
