%% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Solve a multivariate nonconvex quadratically constrained quadratic programming problem constrained to an ellitope.
run('../../../init.m');

% Run an instance via the command line.
print_tbls(n);

%% Utility functions
function print_tbls(dimN) 

  % Initialize
  seed = 777;
  dimM = 10;
  N = 1000;
  density = 1.00;
  global_tol = 1e-3;
  m_vec = [1e1, 1e2, 1e3];
  M_vec = [1e3, 1e4, 1e5];
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
  M = 1e5;
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
  M = 1e5;
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
function o_tbl = run_experiment(N, r, M, m, dimM, dimN, density, seed, global_tol)

  [oracle, hparams] = test_fn_quad_cone_constr_02r(N, r, M, m, seed, dimM, dimN, density);

  % Set up the termination function. The domain is 0 <= lam_i(X) <= r.
  function proj = proj_dh(A, B)
    % Projection of `B` onto the subdifferential of `h` at `A`.
    proj = normal_eigenbox_proj(A, B, r); 
  end
  function proj = proj_NKt(A, B)
    % Projection of `B` onto the normal cone of the dual cone of `K`=S_+^n at `A`.
    proj = normal_eigenbox_proj(A, B, Inf);
  end
  o_at_x0 = copy(oracle);
  o_at_x0.eval(hparams.x0);
  g0 = hparams.constr_fn(hparams.x0);
  rho = global_tol * (1 + hparams.norm_fn(o_at_x0.grad_f_s()));
  eta = global_tol * (1 + hparams.norm_fn(g0 - hparams.set_projector(g0)));
  alt_grad_constr_fn = @(x, p) hparams.grad_constr_fn(x, p);
  term_wrap = @(x,p) ...
    termination_check(x, p, o_at_x0, hparams.constr_fn, alt_grad_constr_fn, @proj_dh, @proj_NKt, hparams.norm_fn, rho, eta);
  
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
  base_hparam.termination_fn = term_wrap;
  base_hparam.check_all_terminations = true;
  
  % Create the IAPIAL hparams.
  ipl_hparam = base_hparam;
  ipl_hparam.sigma_min = sqrt(0.3);
  ipl_hparam.acg_steptype = 'constant';
  ipla_hparam = ipl_hparam;
  ipla_hparam.L_start = (hparams.M / (2 * hparams.m) + 1) / 1000;
  ipla_hparam.acg_steptype = 'variable';
  
  % Create the complicated iALM hparams.
  ialm_hparam = base_hparam;
  ialm_hparam.proj_dh = @proj_dh;
  ialm_hparam.i_ineq_constr = true;
  ialm_hparam.rho0 = hparams.m;
  ialm_hparam.L0 = max([hparams.m, hparams.M]);
  ialm_hparam.rho_vec = hparams.m_constr_vec;
  ialm_hparam.L_vec = hparams.L_constr_vec;
  ialm_hparam.B_vec = hparams.K_constr_vec;
  
  % Run a benchmark test and print the summary.
  hparam_arr = {ialm_hparam, ipl_hparam, ipla_hparam};
  name_arr = {'iALM', 'IPL', 'IPL_A'};
  framework_arr = {@iALM, @IAIPAL, @IAIPAL};
  solver_arr = {@ECG, @ECG, @ECG};
  
  % Run the test.
  % profile on;
  [summary_tables, ~] = run_CCM_benchmark(ncvx_qc_qsdp, framework_arr, solver_arr, hparam_arr, name_arr);
  o_tbl = [table(dimN, r), summary_tables.all];
  disp(o_tbl);
  
end
