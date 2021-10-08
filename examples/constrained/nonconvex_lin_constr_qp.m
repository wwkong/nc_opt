% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Solve a multivariate nonconvex quadratic programming problem constrained to the unit simplex and a linear constraint A * x = b.

% The function of interest is
%
%  f(x) :=  -xi / 2 * ||D * B * x|| ^ 2 + tau / 2 * ||A * x - b|| ^ 2
%
% with curvature pair (m, M). 

% Test generator
N = 1000;
M = 1000;
m = 1;
seed = 777;
dimM = 5;
dimN = 10;
[oracle, hparams] = test_fn_lin_constr_01(N, M, m, seed, dimM, dimN);

% Create the Model object and specify the solver.
ncvx_lc_qp = ConstrCompModel(oracle);
ncvx_lc_qp.solver = @AIPP;
ncvx_lc_qp.feas_type = 'relative';

% Add linear constraints
ncvx_lc_qp.constr_fn = hparams.constr_fn;
ncvx_lc_qp.grad_constr_fn = hparams.grad_constr_fn;
ncvx_lc_qp.set_projector = hparams.set_projector;

% Add penalty framework
ncvx_lc_qp.K_constr = hparams.K_constr;
ncvx_lc_qp.opt_tol = 1e-4;
ncvx_lc_qp.feas_tol = 1e-2;
ncvx_lc_qp.framework = @penalty;

% Set the curvatures and the starting point x0.
ncvx_lc_qp.M = hparams.M;
ncvx_lc_qp.m = hparams.m;
ncvx_lc_qp.x0 = hparams.x0;

% Solve the problem.
ncvx_lc_qp.optimize;
