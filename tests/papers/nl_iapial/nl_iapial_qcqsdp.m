% Solve a multivariate nonconvex quadratically constrained quadratic programming  
% problem constrained to an ellitope.
run('../../../init.m');

% Run an instance via the command line.
print_tbls(n);

%% Utility functions
function print_tbls(dimN) 

  % Initialize
  seed = 777;
  dimM = 25;
  N = 1000;
  density = 1.00;
  global_tol = 1e-2;
  m_vec = [1e2, 1e3, 1e4];
  M_vec = [1e4, 1e5, 1e6];
  r_vec = [5, 10, 20];
  first_tbl = true;

  % Variable M.
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
  
  % Variable m.
  M = 1e6;
  r = 1;
  for m=m_vec
    tbl_row = run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol);
    if first_tbl
      o_tbl = tbl_row;
      first_tbl = false;
    else
      o_tbl = [o_tbl; tbl_row];
    end
  end
  
  % Variable r.
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
  
  disp(['Tables for dimN = ', num2str(dimN)]);
  disp(o_tbl);
  
end
function o_tbl =   run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol)

  [oracle, hparams] = ...
    test_fn_quad_cone_constr_02(N, r, M, m, seed, dimM, dimN, density);

  % Create the Model object and specify the solver.
  ncvx_qc_qsdp = ConstrCompModel(oracle);
  
  % Set the curvatures and the starting point x0.
  ncvx_qc_qsdp.x0 = hparams.x0;
  ncvx_qc_qsdp.M = hparams.M;
  ncvx_qc_qsdp.m = hparams.m;
  ncvx_qc_qsdp.K_constr = hparams.K_constr;
  ncvx_qc_qsdp.L_constr = hparams.L_constr;
  
  % Set the tolerances
  ncvx_qc_qsdp.opt_tol = global_tol;
  ncvx_qc_qsdp.feas_tol = global_tol;
  ncvx_qc_qsdp.time_limit = 6000;
  
  % Add linear constraints
  ncvx_qc_qsdp.constr_fn = hparams.constr_fn;
  ncvx_qc_qsdp.grad_constr_fn = hparams.grad_constr_fn;
  ncvx_qc_qsdp.set_projector = hparams.set_projector;
  
  % Use a relative termination criterion.
  ncvx_qc_qsdp.feas_type = 'relative';
  ncvx_qc_qsdp.opt_type = 'relative';
  
  % Create some basic hparams.
  base_hparam = struct();
  
  % Create the IAPIAL hparams.
  ipl_hparam = base_hparam;
  ipl_hparam.acg_steptype = 'constant';
  ipla_hparam = base_hparam;
  ipla_hparam.acg_steptype = 'variable';
  
  % Create the complicated iALM hparams.
  ialm_hparam = base_hparam;
  ialm_hparam.i_ineq_constr = true;
  ialm_hparam.rho0 = hparams.m;
  ialm_hparam.L0 = max([hparams.m, hparams.M]);
  ialm_hparam.rho_vec = hparams.m_constr_vec;
  ialm_hparam.L_vec = hparams.L_constr_vec;
  % Note that we are using the fact that |X|_F <= 1 over the eigenbox.
  ialm_hparam.B_vec = hparams.K_constr_vec;
  
  % Run a benchmark test and print the summary.
  hparam_arr = {ialm_hparam, ipl_hparam, ipla_hparam};
  name_arr = {'iALM', 'IPL', 'IPL_A'};
  framework_arr = {@iALM, @IAIPAL, @IAIPAL};
  solver_arr = {@ECG, @ECG, @ECG};
  
  % Run the test.
  % profile on;
  [summary_tables, ~] = ...
    run_CCM_benchmark(...
    ncvx_qc_qsdp, framework_arr, solver_arr, hparam_arr, name_arr);
  o_tbl = [table(dimN, r), summary_tables.all];
  disp(o_tbl);
  
end