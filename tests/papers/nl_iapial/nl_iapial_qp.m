% Solve a multivariate nonconvex quadratic programming problem 
% constrained to the unit simplex using MULTIPLE SOLVERS.

% The function of interest is
%
%  f(x) :=  (z - d)' * Q * (z - d) / 2
%
% with curvature pair (m, M). 

% Set up paths.
run('../../../init.m');

% Use a problem instance generator to create the oracle and
% hyperparameters.
M = 1000;
m = 10;
seed = 777;
dimM = 2;
dimN = 10;
[oracle, hparams] = test_fn_lin_box_constr_01(M, m, seed, dimM, dimN);

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
ncvx_lc_qp.time_limit = 100;

% Add linear constraints
ncvx_lc_qp.constr_fn = hparams.constr_fn;
ncvx_lc_qp.grad_constr_fn = hparams.grad_constr_fn;

% Use a relative termination criterion.
ncvx_lc_qp.feas_type = 'relative';
ncvx_lc_qp.opt_type = 'relative';

% Create some basic hparams.
base_hparam = struct();
hiapem_hparam = base_hparam;
hiapem_hparam.full_L_min = true;
hiapem_hparam.is_box = hparams.is_box;
hiapem_hparam.box_lower = hparams.box_lower;
hiapem_hparam.box_upper = hparams.box_upper;
hiapem_hparam.lin_constr_fn = hparams.lin_constr_fn;
hiapem_hparam.lin_grad_constr_fn = hparams.lin_grad_constr_fn;
hiapem_hparam.nonlin_constr_fn = hparams.nonlin_constr_fn;
hiapem_hparam.nonlin_grad_constr_fn = hparams.nonlin_grad_constr_fn;
aipp_hparam = base_hparam;
aipp_hparam.aipp_type = 'aipp';

% % Run a benchmark test and print the summary.
hparam_arr = {hiapem_hparam};
name_arr = {'HiAPeM'};
framework_arr = {@HiAPeM};
solver_arr = {@ECG};
[summary_tables, comp_models] = ...
  run_CCM_benchmark(...
    ncvx_lc_qp, framework_arr, solver_arr, hparam_arr, name_arr);
disp(summary_tables.all);