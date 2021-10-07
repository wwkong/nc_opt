% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Solve a multivariate nonconvex linearly constrained quadratic programming problem constrained to a box.
run('../../../init.m');

% Run an instance via the command line.
print_tbls(n);

%% Utility functions
function print_tbls(dimN) 

  % Initialize
  seed = 777;
  dimM = 25;
  N = 1000;
  global_tol = 1e-5;
  m_vec = [1e2, 1e3, 1e4];
  M_vec = [1e4, 1e5, 1e6];
  r_vec = [5, 10, 20];
  first_tbl = true;

  % Variable M.
  m = 1e0;
  r = 1;
  for M=M_vec
    [~, out_models] = run_experiment(N, r, M, m, dimM, dimN, seed, global_tol);
    tbl_row = parse_models(out_models);
    tbl_row = [table(dimN, r), tbl_row];
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
    [~, out_models] = run_experiment(N, r, M, m, dimM, dimN, seed, global_tol);
    tbl_row = parse_models(out_models);
    tbl_row = [table(dimN, r), tbl_row];
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
    [~, out_models] = run_experiment(N, r, M, m, dimM, dimN, seed, global_tol);
    tbl_row = parse_models(out_models);
    tbl_row = [table(dimN, r), tbl_row];
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
function out_tbl = parse_models(models)
% Parse the output models and log the output.

  % Initialize.
  alg_names = fieldnames(models);
  
  % Loop over the different algorithms.
  
  % First for |w| + |Az-b|
  i_first_alg = true;
  for i=1:length(alg_names)
    cur_mdl = models.(alg_names{i});
    if i_first_alg
      m = cur_mdl.m;
      M = cur_mdl.M;
      time_limit = cur_mdl.time_limit;
      out_tbl = table(m, M, time_limit);
      i_first_alg = false;
    end
    cur_mdl.oracle.eval(cur_mdl.x0);
    opt_mult = 1  / (1 + cur_mdl.norm_fn(cur_mdl.oracle.grad_f_s()));
    g0 = cur_mdl.constr_fn(cur_mdl.x0);
    feas_mult = 1 / (1 + cur_mdl.norm_fn(-g0-cur_mdl.set_projector(-g0)));
    eval([alg_names{i}, '_resid =', num2str(max([cur_mdl.norm_of_v * opt_mult, cur_mdl.norm_of_w * feas_mult])), ';'])
    eval(['out_tbl = [out_tbl, table(', alg_names{i}, '_resid)];'])
  end
  
  % Next for iter
  for i=1:length(alg_names)
    cur_mdl = models.(alg_names{i});
    eval([alg_names{i}, '_iter =', num2str(cur_mdl.iter), ';'])
    eval(['out_tbl = [out_tbl, table(', alg_names{i}, '_iter)];'])
  end
  
  % Next for time
  for i=1:length(alg_names)
    cur_mdl = models.(alg_names{i});
    eval([alg_names{i}, '_time =', num2str(cur_mdl.runtime), ';'])
    eval(['out_tbl = [out_tbl, table(', alg_names{i}, '_time)];'])
  end
  

end
function [o_tbl, o_mdl] = run_experiment(N, r, M, m, dimM, dimN, seed, global_tol)

  [oracle, hparams] = test_fn_lin_cone_constr_03r(N, r, M, m, seed, dimM, dimN);

  % Set up the termination function. The domain is -r <= x <= r.
  % Projection of `b` onto the subdifferential of `h` at `a`.
  function proj = proj_dh(a, b)
    I1 = (abs(a + r) <= 1e-12); 
    I2 = (abs(a - r) <= 1e-12);
    I3 = (abs(a + r) > 1e-12 & abs(a - r) > 1e-12);
    proj = b;
    proj(I1) = min(0, b(I1));
    proj(I2) = max(0, b(I2));
    proj(I3) = 0;
  end
% Projection of `b` onto the normal cone of the dual cone of `K` at `a`.
  function proj = proj_NKt(~, b)
    proj = zeros(size(b));
  end
  o_at_x0 = copy(oracle);
  o_at_x0.eval(hparams.x0);
  g0 = hparams.constr_fn(hparams.x0);
  rho = global_tol * (1 + hparams.norm_fn(o_at_x0.grad_f_s()));
  eta = global_tol * (1 + hparams.norm_fn(-g0-hparams.set_projector(-g0)));
  alt_grad_constr_fn = @(x, p) tsr_mult(hparams.grad_constr_fn(x), p, 'dual');
  term_wrap = @(x,p) termination_check(x, p, o_at_x0, hparams.constr_fn, alt_grad_constr_fn, @proj_dh, @proj_NKt, hparams.norm_fn, rho, eta);  
  
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
  ncvx_qsdp.time_limit = 600;
  
  % Add linear constraints
  ncvx_qsdp.constr_fn = hparams.constr_fn;
  ncvx_qsdp.grad_constr_fn = hparams.grad_constr_fn;
  ncvx_qsdp.set_projector = hparams.set_projector;
  
  % Use a relative termination criterion.
  ncvx_qsdp.feas_type = 'relative';
  ncvx_qsdp.opt_type = 'relative';
  
  % Create some basic hparams.
  base_hparam = struct();
  base_hparam.termination_fn = term_wrap;
  base_hparam.check_all_terminations = true;
  
  % Create the IAPIAL hparams.
  ipla_hparam = base_hparam;
  ipla_hparam.acg_steptype = 'variable';
  qpa_hparam = base_hparam;
  qpa_hparam.acg_steptype = 'variable';
  qpa_hparam.aipp_type = 'aipp';
  
  % Luo's method.
  spa1_hparam = base_hparam;
  spa1_hparam.Gamma = 1;
  spa2_hparam = base_hparam;
  spa2_hparam.Gamma = 10;
  
  % Create the complicated iALM hparams.
  ialm_hparam = base_hparam;
  ialm_hparam.proj_dh = @proj_dh;
  ialm_hparam.i_ineq_constr = false;
  ialm_hparam.rho0 = hparams.m;
  ialm_hparam.L0 = max([hparams.m, hparams.M]);
  ialm_hparam.rho_vec = hparams.m_constr_vec;
  ialm_hparam.L_vec = hparams.L_constr_vec;
  % Note that we are using the fact that |X|_F <= 1 over the simplex.
  ialm_hparam.B_vec = hparams.K_constr_vec;
  
  % Run a benchmark test and print the summary.
  hparam_arr = {ialm_hparam, qpa_hparam, ipla_hparam, spa1_hparam, spa2_hparam};
  name_arr = {'iALM', 'QP_A', 'IPL_A', 'SPA1', 'SPA2'};
  framework_arr = {@iALM, @penalty, @IAIPAL, @sProxALM, @sProxALM};
  solver_arr = {@ECG, @AIPP, @ECG, @ECG, @ECG};
  
  % Run the test.
  [summary_tables, o_mdl] = run_CCM_benchmark(ncvx_qsdp, framework_arr, solver_arr, hparam_arr, name_arr);
  o_tbl = [table(dimN, r), summary_tables.all];
  disp(o_tbl);
  
end
