% Solve a multivariate nonconvex quadratically constrained quadratic programming  
% problem constrained to the unit simplex using MULTIPLE SOLVERS.

% Set up paths.
run('../../../init.m');

% Use a problem instance generator to create the oracle and
% hyperparameters.
seed = 777;
dimM = 20;
N = 1000;
density = 0.05;
global_tol = 1e-4;

% ==============================================================================
%% Tables for dimN = 50
% ==============================================================================
first_tbl = true;

m = 1e0;
r = 1;
dimN = 50;
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
disp(['Tables for dimN = ', num2str(dimN)]);
disp(o_tbl);

% ==============================================================================
%% Tables for dimN = 100
% ==============================================================================
first_tbl = true;

m = 1e0;
r = 1;
dimN = 100;
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
disp(['Tables for dimN = ', num2str(dimN)]);
disp(o_tbl);

% ==============================================================================
%% Tables for dimN = 200
% ==============================================================================
first_tbl = true;

m = 1e0;
r = 1;
dimN = 200;
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
disp(['Tables for dimN = ', num2str(dimN)]);
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
  ipl0_hparam = base_hparam;
  ipl0_hparam.theta = 0;
  ipl0_hparam.acg_steptype = 'constant';
  qp_hparam = base_hparam;
  qp_hparam.acg_steptype = 'constant';
  qp_hparam.aipp_type = 'aipp';
  
  % Run a benchmark test and print the summary.
  hparam_arr = {qp_hparam, ipl_hparam, ipl0_hparam};
  name_arr = {'QP', 'IPL', 'IPL0'};
  framework_arr = {@penalty, @IAIPAL, @IAIPAL};
  solver_arr = {@AIPP, @ECG, @ECG};
  
  % Run the test.
  % profile on;
  [summary_tables, ~] = ...
    run_CCM_benchmark(...
    ncvx_qsdp, framework_arr, solver_arr, hparam_arr, name_arr);
  o_tbl = [table(r), summary_tables.all];
  disp(o_tbl);
  
end