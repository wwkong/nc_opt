% Solve a multivariate nonconvex quadratically constrained quadratic programming  
% problem constrained to a box using MULTIPLE SOLVERS.

% Set up paths.
run('../../../../init.m');

% Use a problem instance generator to create the oracle and
% hyperparameters.
seed = 777;
dimM = 10;
global_tol = 1e-5;

% ==============================================================================
%% Table for dimN = 200
% ==============================================================================
first_tbl = true;

dimN = 200;
m = 1e0;
r = 5;
M_vec = [1e3, 1e4, 1e5, 1e6];
for M=M_vec
  tbl_row = run_experiment(M, m, dimM, dimN, -r, r, seed, global_tol);
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
function o_tbl = run_experiment(M, m, dimM, dimN, x_l, x_u, seed, global_tol)

  [oracle, hparams] = ...
    test_fn_quad_box_constr_02(M, m, seed, dimM, dimN, x_l, x_u);

  % Create the Model object and specify the solver.
  ncvx_qc_qp = ConstrCompModel(oracle);
  
  % Set the curvatures and the starting point x0.
  ncvx_qc_qp.x0 = hparams.x0;
  ncvx_qc_qp.M = hparams.M;
  ncvx_qc_qp.m = hparams.m;
  ncvx_qc_qp.K_constr = hparams.K_constr;
  ncvx_qc_qp.L_constr = hparams.L_constr;
  
  % Set the tolerances
  ncvx_qc_qp.opt_tol = global_tol;
  ncvx_qc_qp.feas_tol = global_tol;
  ncvx_qc_qp.time_limit = 4000;
  
  % Add linear constraints
  ncvx_qc_qp.constr_fn = hparams.constr_fn;
  ncvx_qc_qp.grad_constr_fn = hparams.grad_constr_fn;
  ncvx_qc_qp.set_projector = hparams.set_projector;
  ncvx_qc_qp.dual_cone_projector = hparams.dual_cone_projector;
  
  % Use a relative termination criterion.
  ncvx_qc_qp.feas_type = 'relative';
  ncvx_qc_qp.opt_type = 'relative';
  
  % Create some basic hparams.
  base_hparam = struct();
  
  % Create the IAPIAL hparams.
  ipl_hparam = base_hparam;
  ipl_hparam.acg_steptype = 'constant';
  ipla_hparam = base_hparam;
  ipla_hparam.acg_steptype = 'variable';
  
  % Run a benchmark test and print the summary.
  hparam_arr = {ipl_hparam, ipla_hparam};
  name_arr = {'IPL', 'IPL_A'};
  framework_arr = {@IAIPAL, @IAIPAL};
  solver_arr = {@ECG, @ECG};

  % Run the test.
  [summary_tables, history_tables] = ...
    run_CCM_benchmark(...
      ncvx_qc_qp, framework_arr, solver_arr, hparam_arr, name_arr);  
  o_tbl = [table(x_l, x_u), summary_tables.all];
  disp(o_tbl);
  
  % Print ACG ratio
  ratio_tbl = table();
  for i=1:length(name_arr)
    alg_ratio_tbl = table(history_tables.(name_arr{i}).history.acg_ratio);
    alg_ratio_tbl.Properties.VariableNames = {[name_arr{i}, '_ACG_ratio']};
    ratio_tbl = [ratio_tbl, alg_ratio_tbl];
  end
  disp(ratio_tbl);

end

