% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Solve a multivariate nonconvex quadratic programming problem constrained to the unit simplex

% The function of interest is
%
%  f(Z) :=  -xi / 2 * ||D * B * Z|| ^ 2 + tau / 2 * ||C * Z - d|| ^ 2 
%
% with curvature pair (m, M) and the domain of f being dimN-by-dimN sized matrices. 

% Use a problem instance generator to create the oracle and hyperparameters.
N = 100;
M = 1000;
m = 1;
seed = 777;
dimM = 25;
dimN = 50;
density = 0.025;
[oracle, hparams] = ...
  test_fn_unconstr_02(N, M, m, seed, dimM, dimN, density);

% Create the Model object and specify the solver.
ncvx_qsdp = CompModel(oracle);
ncvx_qsdp.solver = @AG;

% Set the curvatures and the starting point x0.
ncvx_qsdp.M = hparams.M;
ncvx_qsdp.m = hparams.m;
ncvx_qsdp.x0 = hparams.x0;

% Use a relative termination criterion.
ncvx_qsdp.opt_type = 'relative';

% Solve the problem.
ncvx_qsdp.optimize;
