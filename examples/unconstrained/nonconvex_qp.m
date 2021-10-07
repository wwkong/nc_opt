% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Solve a multivariate nonconvex quadratic programming problem constrained to the unit simplex

% The function of interest is
%
%  f(x) :=  -xi / 2 * ||D * B * x|| ^ 2 + tau / 2 * ||A * x - b|| ^ 2
%
% with curvature pair (m, M). 

% Use a problem instance generator to create the oracle and hyperparameters.
N = 1000;
M = 1000;
m = 10;
seed = 777;
dimM = 15;
dimN = 30;
[oracle, hparams] = test_fn_unconstr_01(N, M, m, seed, dimM, dimN);

% Create the Model object and specify the solver.
ncvx_qp = CompModel(oracle);
ncvx_qp.solver = @ECG;

% Set the curvatures and the starting point x0.
ncvx_qp.M = hparams.M;
ncvx_qp.m = hparams.m;
ncvx_qp.x0 = hparams.x0;

% Use a relative termination criterion.
ncvx_qp.opt_type = 'relative';

% Solve the problem.
ncvx_qp.optimize;
