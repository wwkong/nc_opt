% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Solve a multivariate minmax nonconvex quadratic programming problem constrained to the unit simplex

% The function of interest is
%
%   f(x) :=  max_i 
%     {
%       -xi / 2 * ||D_i * B_i * x|| ^ 2 + tau / 2 * ||A_i * x - b_i|| ^ 2
%     }
%
% with curvature pairs {(m_i, M_i)}. 

% Use a problem instance generator to create the oracle factory and hyperparameters.
N = 1000;
M = 100;
m = 1;
seed = 777;
dimM = 20;
dimN = 100;
density = 0.05;
k = 5;
[oracle_factory, hparams] = ...
  test_fn_mm_unconstr_01(k, N, M, m, seed, dimM, dimN, density);

% Set up the oracle and special tolerances.
hparams.rho_x = 1e-2;
hparams.rho_y = 1e-1;
[xi, M_smoothed] = compute_smoothed_parameters(hparams, 'AIPP-S');
oracle = oracle_factory(xi);

% Create the Model object and specify the solver and tolerance.
ncvx_mm_qp = CompModel(oracle);
ncvx_mm_qp.solver = @AIPP;
ncvx_mm_qp.opt_tol = hparams.rho_x;

% Set the curvatures and the starting point x0.
ncvx_mm_qp.M = M_smoothed;
ncvx_mm_qp.m = hparams.m;
ncvx_mm_qp.x0 = hparams.x0;

% Use a relative termination criterion.
ncvx_mm_qp.opt_type = 'relative';

% Solve the problem.
ncvx_mm_qp.optimize;
