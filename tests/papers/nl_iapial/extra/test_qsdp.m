% Solve a multivariate nonconvex quadratically constrained quadratic programming  
% problem constrained to the unit simplex using MULTIPLE SOLVERS.

run('../../../../init.m');
profile on;
print_tbls(50);
profile viewer
profile off;

%% Utility functions
function print_tbls(dimN) 

  % Initialize
  seed = 777;
  dimM = 25;
  N = 1000;
  density = 0.05;
  global_tol = 1e-2;
  
  % Variable m.
  m = 1e0;
  M = 1e4;
  r = 1;
  tbl_row = run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol);
  disp(tbl_row);
  
end
function o_tbl = run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol)

  [oracle, hparams] = ...
    test_fn_lin_cone_constr_02r(N, r, M, m, seed, dimM, dimN, density);

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
  ipl_hparam = base_hparam;
  ipl_hparam.acg_steptype = 'constant';
  ipla_hparam = base_hparam;
  ipla_hparam.acg_steptype = 'variable';
  qp_hparam = base_hparam;
  qp_hparam.acg_steptype = 'constant';
  qp_hparam.aipp_type = 'aipp';
  qpa_hparam = base_hparam;
  qpa_hparam.acg_steptype = 'variable';
  qpa_hparam.aipp_type = 'aipp';
  
  % Create the complicated iALM hparams.
  ialm_hparam = base_hparam;
  ialm_hparam.i_ineq_constr = false;
  ialm_hparam.rho0 = hparams.m;
  ialm_hparam.L0 = max([hparams.m, hparams.M]);
  ialm_hparam.rho_vec = hparams.m_constr_vec;
  ialm_hparam.L_vec = hparams.L_constr_vec;
  % Note that we are using the fact that |X|_F <= 1 over the eigenbox.
  ialm_hparam.B_vec = hparams.K_constr_vec;
  
%   % Run a benchmark test and print the summary.
%   hparam_arr = ...
%     {ialm_hparam, qp_hparam, qpa_hparam, ipl_hparam, ipla_hparam};
%   name_arr = {'iALM', 'QP', 'QP_A', 'IPL', 'IPL_A'};
%   framework_arr = {@iALM, @penalty, @penalty, @IAIPAL, @IAIPAL};
%   solver_arr = {@ECG, @AIPP, @AIPP, @ECG, @ECG};
  
%   hparam_arr = {qp_hparam, qpa_hparam, ipl_hparam, ipla_hparam};
%   name_arr = {'QP', 'QP_A', 'IPL', 'IPL_A'};
%   framework_arr = { @penalty, @penalty, @IAIPAL, @IAIPAL};
%   solver_arr = {@AIPP, @AIPP, @ECG, @ECG};
  
%   hparam_arr = {ipl_hparam, ipla_hparam};
%   name_arr = {'IPL', 'IPL_A'};
%   framework_arr = {@IAIPAL, @IAIPAL};
%   solver_arr = {@ECG, @ECG};

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