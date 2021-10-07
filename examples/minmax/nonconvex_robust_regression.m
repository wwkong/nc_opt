% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Solve a nonconvex robust regression problem on a specified dataset.

% Use a problem instance generator to create the oracle factory and hyperparameters.
data_name = '../../data/heart_scale.txt';
alpha = 10;
[oracle_factory, hparams] = ...
  test_fn_robust_regression_01(data_name, alpha);

% Set up the oracle and special tolerances.
hparams.rho_x = 1e-3;
hparams.rho_y = 1e-3;
[xi, M_smoothed] = compute_smoothed_parameters(hparams, 'AIPP-S');
oracle = oracle_factory(xi);

% Create the Model object and specify the solver and tolerance.
ncvx_rr = CompModel(oracle);
ncvx_rr.solver = @AIPP;
ncvx_rr.opt_tol = hparams.rho_x;

% Set the curvatures and the starting point x0.
ncvx_rr.M = M_smoothed;
ncvx_rr.m = hparams.m;
ncvx_rr.x0 = hparams.x0;

% Use a relative termination criterion.
ncvx_rr.opt_type = 'relative';

% Solve the problem.
ncvx_rr.optimize;
