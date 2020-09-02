% Solve a multivariate nonconvex quadratic programming problem 
% constrained to the unit simplex using MULTIPLE SOLVERS.

% The function of interest is
%
%  f(x) :=  -xi / 2 * ||D * B * X|| ^ 2 + tau / 2 * ||A * X - b|| ^ 2
%
% with curvature pair (m, M). 

% Use a problem instance generator to create the oracle and
% hyperparameters.
N = 1000;
M = 1000;
m = 1;
seed = 777;
dimM = 10;
dimN = 20;
density = 0.01;
[oracle, hparams] = ...
  test_fn_lin_cone_constr_02(N, M, m, seed, dimM, dimN, density);

% Create the Model object and specify the solver.
ncvx_lc_qp = ConstrCompModel(oracle);

% Set the curvatures and the starting point x0.
ncvx_lc_qp.x0 = hparams.x0;
ncvx_lc_qp.M = hparams.M;
ncvx_lc_qp.m = hparams.m;
ncvx_lc_qp.K_constr = hparams.K_constr;

% Set the tolerances
ncvx_lc_qp.opt_tol = 1e-1;
ncvx_lc_qp.feas_tol = 1e-1;

% Add linear constraints
ncvx_lc_qp.constr_fn = @(x) hparams.constr_fn(x);
ncvx_lc_qp.grad_constr_fn = hparams.grad_constr_fn;

% Use a relative termination criterion.
ncvx_lc_qp.feas_type = 'relative';
ncvx_lc_qp.opt_type = 'relative';

% Create some basic hparams.
base_hparam = struct();
aipp_hparam = base_hparam;
aipp_hparam.aipp_type = 'aipp';

% % Run a benchmark test and print the summary.
% hparam_arr = {base_hparam, aipp_hparam, base_hparam};
% name_arr = {'QP_AIPP', 'R_QP_AIPP', 'IAPIAL'};
% framework_arr = {@penalty, @penalty, @iapial};
% solver_arr = {@AIPP, @AIPP, @AIPP};

% hparam_arr = {aipp_hparam, base_hparam, base_hparam};
% name_arr = {'R_QP_AIPP', 'AIDAL', 'IAPIAL'};
% framework_arr = {@penalty, @aidal, @iapial};
% solver_arr = {@AIPP, @AIPP, @AIPP};

hparam_arr = {base_hparam};
name_arr = {'AIDAL'};
framework_arr = {@aidal};
solver_arr = {@AIPP};


[summary_tables, comp_models] = ...
  run_CCM_benchmark(...
    ncvx_lc_qp, framework_arr, solver_arr, hparam_arr, name_arr);
disp(summary_tables.all);