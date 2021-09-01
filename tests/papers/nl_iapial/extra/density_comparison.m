% Solve a multivariate nonconvex quadratically constrained quadratic programming  
% problem constrained to the unit simplex using MULTIPLE SOLVERS.

% The function of interest is
%
%  f(x) :=  (z - d)' * Q * (z - d) / 2
%
% with curvature pair (m, M). 
%
% Constraint is of the form


% Set up paths.
run('../../../init.m');

% Use a problem instance generator to create the oracle and
% hyperparameters.
seed = 777;
dimM = 10;
N = 1000;
global_tol = 1e-4;

% ==============================================================================
%% Tables for density 1%
% ==============================================================================
first_tbl = true;

m = 1e0;
r = 1;
dimN = 100;
density = 0.01;
for M=[1e2, 1e3, 1e4, 1e5]
  tbl_row = run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol);
  if first_tbl
    o_tbl = tbl_row;
    first_tbl = false;
  else
    o_tbl = [o_tbl; tbl_row];
  end
end

m = 1e0;
M = 1e4;
for r=[1, 2.5, 5, 10]
  tbl_row = run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol);
  if first_tbl
    o_tbl = tbl_row;
    first_tbl = false;
  else
    o_tbl = [o_tbl; tbl_row];
  end
end
disp('Tables for density = 1%')
disp(o_tbl);

% ==============================================================================
%% Tables density = 2.5%
% ==============================================================================
first_tbl = true;

m = 1e0;
r = 1;
dimN = 100;
density = 0.025;
for M=[1e2, 1e3, 1e4, 1e5]
  tbl_row = run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol);
  if first_tbl
    o_tbl = tbl_row;
    first_tbl = false;
  else
    o_tbl = [o_tbl; tbl_row];
  end
end

m = 1e0;
M = 1e4;
for r=[1, 2.5, 5, 10]
  tbl_row = run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol);
  if first_tbl
    o_tbl = tbl_row;
    first_tbl = false;
  else
    o_tbl = [o_tbl; tbl_row];
  end
end
disp('Tables for density = 2.5%')
disp(o_tbl);

% ==============================================================================
%% Tables density = 5%
% ==============================================================================
first_tbl = true;

m = 1e0;
r = 1;
dimN = 100;
density = 0.05;
for M=[1e2, 1e3, 1e4, 1e5]
  tbl_row = run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol);
  if first_tbl
    o_tbl = tbl_row;
    first_tbl = false;
  else
    o_tbl = [o_tbl; tbl_row];
  end
end

m = 1e0;
M = 1e4;
for r=[1, 2.5, 5, 10]
  tbl_row = run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol);
  if first_tbl
    o_tbl = tbl_row;
    first_tbl = false;
  else
    o_tbl = [o_tbl; tbl_row];
  end
end
disp('Tables for density = 5%')
disp(o_tbl);

% ==============================================================================
%% Tables density = 10%
% ==============================================================================
first_tbl = true;

m = 1e0;
r = 1;
dimN = 100;
density = 0.1;
for M=[1e2, 1e3, 1e4, 1e5]
  tbl_row = run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol);
  if first_tbl
    o_tbl = tbl_row;
    first_tbl = false;
  else
    o_tbl = [o_tbl; tbl_row];
  end
end

m = 1e0;
M = 1e4;
for r=[1, 2.5, 5, 10]
  tbl_row = run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol);
  if first_tbl
    o_tbl = tbl_row;
    first_tbl = false;
  else
    o_tbl = [o_tbl; tbl_row];
  end
end
disp('Tables for density = 10%')
disp(o_tbl);

%% Utility functions
function o_tbl = ...
  run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol)

  [oracle, hparams] = ...
    test_fn_lin_cone_constr_03(N, r, M, m, seed, dimM, dimN, density);

  % Create the Model object and specify the solver.
  ncvx_qsdp = ConstrCompModel(oracle);
  
  % Set the curvatures and the starting point x0.
  ncvx_qsdp.x0 = hparams.x0;
  ncvx_qsdp.M = hparams.M;
  ncvx_qsdp.m = hparams.m;
  ncvx_qsdp.K_constr = hparams.K_constr;
  
  % Set the tolerances
  ncvx_qsdp.opt_tol = global_tol;
  ncvx_qsdp.feas_tol = global_tol;
  ncvx_qsdp.time_limit = 2000;
  
  % Add linear constraints
  ncvx_qsdp.constr_fn = hparams.constr_fn;
  ncvx_qsdp.grad_constr_fn = hparams.grad_constr_fn;
  ncvx_qsdp.set_projector = hparams.set_projector;
  
  % Use a relative termination criterion.
  ncvx_qsdp.feas_type = 'relative';
  ncvx_qsdp.opt_type = 'relative';
  
  % Create some basic hparams.
  base_hparam = struct();
  
  % Create the IAPIAL hparams.
  ipl_hparam = base_hparam;
  ipl_hparam.acg_steptype = 'constant';
  ipla_hparam = base_hparam;
  ipla_hparam.acg_steptype = 'variable';
  rqp_hparam = base_hparam;
  rqp_hparam.acg_steptype = 'variable';
  qp_hparam = base_hparam;
  qp_hparam.acg_steptype = 'constant';
  qp_hparam.aipp_type = 'aipp';
  qpa_hparam = base_hparam;
  qpa_hparam.acg_steptype = 'variable';
  qpa_hparam.aipp_type = 'aipp';
  
  % Run a benchmark test and print the summary.
  hparam_arr = ...
    {qp_hparam, qpa_hparam, rqp_hparam, ipl_hparam, ipla_hparam};
  name_arr = {'QP', 'QP_A', 'RQP', 'IPL', 'IPL_A'};
  framework_arr = {@penalty, @penalty, @penalty, @IAIPAL, @IAIPAL};
  solver_arr = {@AIPP, @AIPP, @AIPP, @ECG, @ECG};
  
  % Run the test.
  % profile on;
  [summary_tables, ~] = ...
    run_CCM_benchmark(...
    ncvx_qsdp, framework_arr, solver_arr, hparam_arr, name_arr);
  o_tbl = [table(r), summary_tables.all];
  disp(o_tbl);
  
end