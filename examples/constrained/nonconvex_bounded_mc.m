% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Solve a bounded matrix completion problem instance.

% Test generator
beta = 0.5;
theta = sqrt(2);
mu = 2;
seed = 777;
data_name = 'movielens_100k_610u_9724m';
[oracle, hparams] = ...
  test_fn_bmc_01(data_name, beta, theta, mu, seed);

% Create the Model object and specify the solver.
ncvx_bmc = ConstrCompModel(oracle);
ncvx_bmc.solver = @AIPP;
ncvx_bmc.feas_type = 'relative';
ncvx_bmc.time_limit = 5;

% Add linear constraints
ncvx_bmc.constr_fn = hparams.constr_fn;
ncvx_bmc.grad_constr_fn = hparams.grad_constr_fn;
ncvx_bmc.set_projector = hparams.set_projector;

% Add penalty framework
ncvx_bmc.K_constr = hparams.K_constr;
ncvx_bmc.opt_tol = 1e1;
ncvx_bmc.feas_tol = 1e1;
ncvx_bmc.framework = @penalty;

% Set the curvatures and the starting point x0.
ncvx_bmc.M = hparams.M;
ncvx_bmc.m = hparams.m;
ncvx_bmc.x0 = hparams.x0;

% Solve the problem.
ncvx_bmc.optimize;
