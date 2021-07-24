% Solve a multivariate nonconvex quadratic programming problem 
% constrained to the unit simplex and a linear constraint A * x = b.

% The function of interest is
%
%  f(x) :=  -xi / 2 * ||D * B * x|| ^ 2 + tau / 2 * ||A * x - b|| ^ 2
%
% with curvature pair (m, M). 

% Test generator
N = 1000;
M = 100;
m = 1;
seed = 777;
dimM = 30;
dimN = 100;
density = 0.05;
[oracle, hparams] = ...
  test_fn_lin_constr_02(N, M, m, seed, dimM, dimN, density);

% Create the Model object and specify the solver.
ncvx_lc_qsdp = ConstrCompModel(oracle);
ncvx_lc_qsdp.solver = @AIPP;
ncvx_lc_qsdp.opt_type = 'relative';
ncvx_lc_qsdp.feas_type = 'relative';

% Add linear constraints
ncvx_lc_qsdp.constr_fn = hparams.constr_fn;
ncvx_lc_qsdp.grad_constr_fn = hparams.grad_constr_fn;
ncvx_lc_qsdp.set_projector = hparams.set_projector;

% Add penalty framework
ncvx_lc_qsdp.K_constr = hparams.K_constr;
ncvx_lc_qsdp.opt_tol = 1e-3;
ncvx_lc_qsdp.feas_tol = 1e-3;
ncvx_lc_qsdp.framework = @penalty;

% Set the curvatures and the starting point x0.
ncvx_lc_qsdp.M = hparams.M;
ncvx_lc_qsdp.m = hparams.m;
ncvx_lc_qsdp.x0 = hparams.x0;

% Solve the problem.
ncvx_lc_qsdp.optimize;