% Solve a multivariate nonconvex quadratically constrained quadratic programming  
% problem constrained to an ellitope.

% Set up paths.
run('../../../../init.m');

% Use a problem instance generator to create the oracle and
% hyperparameters.
seed = 777;
dimM = 10;
N = 1000;
density = 1.00;
global_tol = 1e-3;
M_vec = [1e3, 1e4, 1e5, 1e6];
r_vec = [2.5, 5, 10, 20];

% ==============================================================================
%% Table for dimN = 50
% ==============================================================================
first_tbl = true;
dimN = 50;

m = 1e0;
r = 1;
for M=M_vec
  tbl_row = run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol);
  if first_tbl
    o_tbl = tbl_row;
    first_tbl = false;
  else
    o_tbl = [o_tbl; tbl_row];
  end
end

m = 1e0;
M = 1e6;
for r=r_vec
  tbl_row = run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol);
  if first_tbl
    o_tbl = tbl_row;
    first_tbl = false;
  else
    o_tbl = [o_tbl; tbl_row];
  end
end
disp(o_tbl);
disp(['Tables for dimN = ', num2str(dimN)]);

% ==============================================================================
%% Table for dimN = 100
% ==============================================================================
first_tbl = true;
dimN = 100;

m = 1e0;
r = 1;
for M=M_vec
  tbl_row = run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol);
  if first_tbl
    o_tbl = tbl_row;
    first_tbl = false;
  else
    o_tbl = [o_tbl; tbl_row];
  end
end

m = 1e0;
M = 1e6;
for r=r_vec
  tbl_row = run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol);
  if first_tbl
    o_tbl = tbl_row;
    first_tbl = false;
  else
    o_tbl = [o_tbl; tbl_row];
  end
end
disp(o_tbl);
disp(['Tables for dimN = ', num2str(dimN)]);

% ==============================================================================
%% Table for dimN = 200
% ==============================================================================
first_tbl = true;
dimN = 200;

m = 1e0;
r = 1;
for M=M_vec
  tbl_row = run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol);
  if first_tbl
    o_tbl = tbl_row;
    first_tbl = false;
  else
    o_tbl = [o_tbl; tbl_row];
  end
end

m = 1e0;
M = 1e6;
for r=r_vec
  tbl_row = run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol);
  if first_tbl
    o_tbl = tbl_row;
    first_tbl = false;
  else
    o_tbl = [o_tbl; tbl_row];
  end
end
disp(o_tbl);
disp(['Tables for dimN = ', num2str(dimN)]);

%% Utility functions
function o_tbl = ...
  run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol)

  [oracle, hparams] = ...
    test_fn_quad_cone_constr_02(N, r, M, m, seed, dimM, dimN, density);

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
  ncvx_qsdp.time_limit = 4000;
  
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
  ipla_hparam = base_hparam;
  ipla_hparam.acg_steptype = 'variable';
  
  % Run a benchmark test and print the summary.
  hparam_arr = {ipla_hparam};
  name_arr = {'IPL_A'};
  framework_arr = {@IAIPAL};
  solver_arr = {@ECG};
  
  % Run the test.
  % profile on;
  [summary_tables, ~] = ...
    run_CCM_benchmark(...
    ncvx_qsdp, framework_arr, solver_arr, hparam_arr, name_arr);
  o_tbl = [table(r), summary_tables.all];
  disp(o_tbl);
  
end