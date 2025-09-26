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
    double dt = 0.0001;
    double dt_sample = dt;
    int    Nx = 10;
    int    Nt = 1 + Lt / dt;
    int    Ns = 1 + Lt / dt_sample;
    double dx = Lx / Nx; // periodic boundary condition
    double nu = 0.01;

    double implicit_solver_threshold = 1e-12; // stopping criterion for the iterative solver
    int    implicit_solver_max_iter  = 10; // maximum number of iterations for the iterative solver

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

    // The second approach is implicit: backward Euler + 2nd-order centered scheme

    int counter = 0;

    Eigen::SparseMatrix<double> Jacobian(Nx, Nx); // Jacobian for the Newton method
    std::vector<Eigen::Triplet<double>> Jacobian_idx; // list of non-zero entries
    Jacobian_idx.reserve(3 * Nx); // each row has at most 3 non-zero entries

    Eigen::VectorXd residual = Eigen::VectorXd::Zero(Nx); // residual vector
    Eigen::VectorXd u_diff   = Eigen::VectorXd::Zero(Nx); // diff between u_new and u_old
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver; // linear solver of Ax = b where A is a sparse matrix
    
    u_old = u_init; // reset the initial condition as the old current solution

    for (j = 0; j < Nt; ++j) {

        std::cout << "current maximal abs value of u = " << u_old.cwiseAbs().maxCoeff() << std::endl;
        if (!u_old.allFinite()) {
        std::cerr << "Simulation diverged at time step j = " << j << " (t = " << j * dt << ")" << std::endl;
        break;
        }

        if (j % int(dt_sample / dt) == 0) {
            u_sample.col(j / int(dt_sample / dt)) = u_old; // sample the current solution
        }

        u_new = u_old; // initial guess for the iterative solver
        residual.setOnes();
        counter = 0;
        
        while (counter <= implicit_solver_max_iter && residual.cwiseAbs().maxCoeff() > implicit_solver_threshold) {
            Jacobian_idx.clear(); // initialize the non-zero entries list
            std::cout << "  iteration " << counter << ": max residual = " << residual.cwiseAbs().maxCoeff() << std::endl;
            // compute the residual and the non-zero entries of the Jacobian matrix
            for (k = 0; k < Nx; ++k) {
                kl = (k - 1 + Nx) % Nx; // periodic boundary condition
                kr = (k + 1) % Nx;
                residual(k) = u_new(k) - u_old(k) + (dt / (2.0 * dx)) * (u_new(k) * (u_new(kr) - u_new(kl)))
                        - (nu * dt / (dx * dx)) * (u_new(kl) - 2 * u_new(k) + u_new(kr)); // compute the residual F（u）
                Jacobian_idx.emplace_back(k, kl, - (dt / (2.0 * dx)) * u_new(k) - dt * (nu / (dx * dx)));
                Jacobian_idx.emplace_back(k, k, 1 + (dt / (2.0 * dx)) * (u_new(kr) - u_new(kl)) + dt * (2.0 * nu / (dx * dx)));
                Jacobian_idx.emplace_back(k, kr, (dt / (2.0 * dx)) * u_new(k) - dt * (nu / (dx * dx))); // compute the non-zero entries of the Jacobian matrix J（u）
            }
            Jacobian.setFromTriplets(Jacobian_idx.begin(), Jacobian_idx.end()); // assemble the Jacobian matrix
            solver.compute(Jacobian); // Let Eigen perform the sparse LU decomposition of the Jacobian matrix
            if(solver.info() != Eigen::Success) {
                std::cerr << "The sparse LU decomposition of the Jacobian failed at time step j = " << j << " (t = " << j * dt << ")" << std::endl;
                break;
            }
            // solve the linear system to get the update
            u_diff = solver.solve(-residual); // solve u_diff = u_new - u from: J(u)(u_new - u) = -F(u)
            u_new  += u_diff;
            counter++;
        }

        if(solver.info() != Eigen::Success) {
                std::cerr << "Simulation aborted due to the failure of the sparse LU decomposition of the Jacobian at time step j = " << j << " (t = " << j * dt << ")" << std::endl;
                break;
            }

        u_old = u_new; // update the old current solution
    }

    std::stringstream filename_str;
    filename_str << "burgers_implicit_nu_" << nu << "_nx_" << Nx << "_dt_" << dt << ".txt";
    std::string filename = filename_str.str();
    std::ofstream file_implicit(filename);

    file_implicit << u_sample.format(TXTFormat);
    file_implicit.close();

    return 0;
}