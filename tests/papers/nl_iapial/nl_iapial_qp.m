% Solve a multivariate nonconvex linearly constrained quadratic programming  
% problem constrained to the unit simplex using MULTIPLE SOLVERS.
run('../../../init.m');

% Run an instance via the command line.
print_tbls(n);

%% Utility functions
function print_tbls(dimN) 

  % Initialize
  seed = 777;
  dimM = 25;
  N = 1000;
  global_tol = 1e-6;
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
    grad_f_at_x0 = cur_mdl.oracle.grad_f_s();
    opt_mult = 1  / (1 + cur_mdl.norm_fn(grad_f_at_x0));
    feas_mult = 1 / (1 + cur_mdl.norm_fn(cur_mdl.constr_fn(cur_mdl.x0)));
    eval([alg_names{i}, '_resid =', ...
         num2str(cur_mdl.norm_of_v * opt_mult + ...
                 cur_mdl.norm_of_w * feas_mult), ';'])
    eval(['out_tbl = [out_tbl, table(', alg_names{i}, '_resid)];'])
  end
  
%   % Next for |Az-b|
%   for i=1:length(alg_names)
%     cur_mdl = models.(alg_names{i});
%     feas_mult = 1 / (1 + cur_mdl.norm_fn(cur_mdl.constr_fn(cur_mdl.x0)));
%     eval([alg_names{i}, '_feas =', num2str(cur_mdl.norm_of_w * feas_mult), ';'])
%     eval(['out_tbl = [out_tbl, table(', alg_names{i}, '_feas)];'])
%   end
  
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

  [oracle, hparams] = ...
    test_fn_lin_cone_constr_01r(N, r, M, m, seed, dimM, dimN);

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
  
  % Run a benchmark test and print the summary.
  hparam_arr = ...
    {qpa_hparam, ipla_hparam, spa1_hparam, spa2_hparam};
  name_arr = {'QP_A', 'IPL_A', 'SPA1', 'SPA2'};
  framework_arr = {@penalty, @IAIPAL, @sProxALM, @sProxALM};
  solver_arr = {@AIPP, @ECG, @ECG, @ECG};
  
  % Run the test.
  % profile on;
  [summary_tables, o_mdl] = ...
    run_CCM_benchmark(...
    ncvx_qsdp, framework_arr, solver_arr, hparam_arr, name_arr);
  o_tbl = [table(dimN, r), summary_tables.all];
  disp(o_tbl);
  
end