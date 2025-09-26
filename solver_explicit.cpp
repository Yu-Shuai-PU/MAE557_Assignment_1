#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>

int main() {

    const double pi = M_PI;
    const double Lx = 2.0 * pi;
    const static Eigen::IOFormat TXTFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n");

    double Lt = 1;
    double dt = 0.00001;
    double dt_sample = dt;
    int    Nx = 120;
    int    Nt = 1 + Lt / dt;
    int    Ns = 1 + Lt / dt_sample;
    double dx = Lx / Nx; // periodic boundary condition
    double nu = 0.01;

    // generate spatial grid
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(Nx, 0, Lx - dx);
    // specify the initial condition
    Eigen::VectorXd u_init = Eigen::VectorXd::Zero(Nx);
    // create a vector to store the old & new current solution
    Eigen::VectorXd u_old = Eigen::VectorXd::Zero(Nx);
    Eigen::VectorXd u_new = Eigen::VectorXd::Zero(Nx);
    // create a vector to store the advection term
    Eigen::VectorXd uu_dx = Eigen::VectorXd::Zero(Nx);
    // create a vector to store the diffusion term
    Eigen::VectorXd u_dxx = Eigen::VectorXd::Zero(Nx);
    // create a matrix to store the sampled solution
    Eigen::MatrixXd u_sample = Eigen::MatrixXd::Zero(Nx, Ns);

    int j, k, kl, kr;

    // set the initial condition
    u_init = x.array().sin() * (-(x.array() - pi) * (x.array() - pi)).exp();
    u_old = u_init; // set the initial condition as the old current solution

    // The first approach is explicit: forward Euler + 1st-order upwind scheme + 2nd-order centered scheme

    double dt_max = dx * dx / (2 * nu + u_init.cwiseAbs().maxCoeff() * dx); // stability condition
    std::cout << "Maximum allowable time step size for stability: dt_max = " << dt_max << std::endl;

    for (j = 0; j < Nt; ++j) {

        std::cout << "current maximal abs value of u = " << u_old.cwiseAbs().maxCoeff() << std::endl;
        if (!u_old.allFinite()) {
            std::cerr << "Simulation diverged at time step j = " << j << " (t = " << j * dt << ")" << std::endl;
            break;
        }

        if (j % int(dt_sample / dt) == 0) {
            u_sample.col(j / int(dt_sample / dt)) = u_old; // sample the current solution
        }

        for (k = 0; k < Nx; ++k) {
            kl = (k - 1 + Nx) % Nx; // periodic boundary condition
            kr = (k + 1) % Nx;
            // 1st-order upwind scheme for the advection term
            // recall that if u>=0 (u<0), we have to use backward (forward) scheme to ensure numerical stability
            uu_dx(k) = ((u_old(k) + std::abs(u_old(k))) / 2.0) * (u_old(k) - u_old(kl)) / dx
                     + ((u_old(k) - std::abs(u_old(k))) / 2.0) * (u_old(kr) - u_old(k)) / dx;
            u_dxx(k) = (u_old(kl) - 2 * u_old(k) + u_old(kr)) / (dx * dx); // 2nd-order centered scheme
        }

        u_new = u_old.array() + dt * (-1 * uu_dx.array() + nu * u_dxx.array()); // update the solution
        u_old = u_new; // update the old current solution

    }
    
    std::stringstream filename_str;
    filename_str << "burgers_explicit_nu_" << nu << "_nx_" << Nx << "_dt_" << dt << ".txt";
    std::string filename = filename_str.str();
    std::ofstream file_explicit(filename);
    file_explicit << u_sample.format(TXTFormat);
    file_explicit.close();

    return 0;
}