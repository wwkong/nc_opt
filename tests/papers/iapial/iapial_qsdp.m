% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Solve a multivariate nonconvex quadratic programming problem constrained to the unit simplex.

% The function of interest is
%
%  f(x) :=  -xi / 2 * ||D * B * X|| ^ 2 + tau / 2 * ||A * X - b|| ^ 2
%
% with curvature pair (m, M). 

%% Initialization

% Set up paths.
run('../../../init.m');

% Global parameters for each experiment.
globals.N = 1000;
globals.seed = 777;
globals.dimM = 30;
globals.dimN = 100;
globals.density = 0.05;
globals.opt_tol = 1e-4;
globals.feas_tol = 1e-4;

%% Run an experiment

% The main parameters (mM_mat) should be spec'd by Condor.

% E.g. 

% mM_mat = [1e1, 1e3];

mM_mat = ...
  [1e0, 1e2; ...
   1e0, 1e3; ...
   1e0, 1e4; ...
   1e1, 1e5; ...
   1e2, 1e5; ...
   1e3, 1e5];

i_first_row = true;

for i=1:size(mM_mat, 1)
  [out_summaries, out_models] = run_experiment(mM_mat(i, 1), mM_mat(i, 2), globals);
  tbl_row = parse_outputs(out_summaries, out_models);
  disp(tbl_row);
  if i_first_row
    tbl = tbl_row;
    i_first_row = false;
  else
    tbl = [tbl; tbl_row]; 
  end
end
disp(tbl);
writetable(tbl, "iapial_qsdp.xlsx");

%% Helper Functions

% Parse the output models and log the output.
function out_tbl = parse_outputs(summaries, models)

  % Initialize.
  alg_names = fieldnames(models);
  
  % Loop over the different algorithms.
  
  % First for |w|
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
    eval([alg_names{i}, '_opt =', num2str(cur_mdl.norm_of_v * opt_mult), ';'])
    eval(['out_tbl = [out_tbl, table(', alg_names{i}, '_opt)];'])
  end
  
  % Next for |Az-b|
  for i=1:length(alg_names)
    cur_mdl = models.(alg_names{i});
    feas_mult = 1 / (1 + cur_mdl.norm_fn(cur_mdl.constr_fn(cur_mdl.x0)));
    eval([alg_names{i}, '_feas =', num2str(cur_mdl.norm_of_w * feas_mult), ';'])
    eval(['out_tbl = [out_tbl, table(', alg_names{i}, '_feas)];'])
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
  
  % Next for function value
  for i=1:length(alg_names)
    cur_mdl = models.(alg_names{i});
    o_at_x = cur_mdl.oracle.eval(cur_mdl.x);
    f_at_x = o_at_x.f_s() + o_at_x.f_n();
    eval([alg_names{i}, '_fval =', num2str(f_at_x), ';'])
    eval(['out_tbl = [out_tbl, table(', alg_names{i}, '_fval)];'])
  end

  % Next for multiplier ratio
  for i=1:length(alg_names)
    if (ismember(['k0_avg_', alg_names{i}], summaries.Properties.VariableNames))
      eval(['out_tbl = [out_tbl, summaries(:,["', 'k0_avg_', alg_names{i}, '"])];'])
    end
  end
  
end

% Run a single experiment and output the summary row.
function [out_tbl, out_models] = run_experiment(m, M, params) 

  % Gather the oracle and hparam instance.
  [oracle, hparams] = test_fn_lin_cone_constr_02(params.N, M, m, params.seed, params.dimM, params.dimN, params.density);

  % Create the Model object and specify the limits (if any).
  ncvx_lc_qp = ConstrCompModel(oracle);
  ncvx_lc_qp.time_limit = 4000;
  
  % Set the curvatures and the starting point x0.
  ncvx_lc_qp.x0 = hparams.x0;
  ncvx_lc_qp.M = hparams.M;
  ncvx_lc_qp.m = hparams.m;
  ncvx_lc_qp.K_constr = hparams.K_constr;
  
  % Set the tolerances
  ncvx_lc_qp.opt_tol = params.opt_tol;
  ncvx_lc_qp.feas_tol = params.feas_tol;
  
  % Add linear constraints.
  ncvx_lc_qp.constr_fn = @(x) hparams.constr_fn(x);
  ncvx_lc_qp.grad_constr_fn = hparams.grad_constr_fn;
  
  % Use a relative termination criterion.
  ncvx_lc_qp.feas_type = 'relative';
  ncvx_lc_qp.opt_type = 'relative';
  
  % Create some basic hparams.
  base_hparam = struct();
  ipla0_hparam = base_hparam;
  ipla0_hparam.acg_steptype = 'variable';
  ipla0_hparam.k0_type = 1;
  ipla1_hparam = base_hparam;
  ipla1_hparam.acg_steptype = 'variable';
  ipla1_hparam.k0_type = 2;
  ipla2_hparam = base_hparam;
  ipla2_hparam.acg_steptype = 'variable';
  ipla2_hparam.k0_type = 3;
  rqp_hparam = base_hparam;
  rqp_hparam.acg_steptype = 'variable';
  qpa_hparam = base_hparam;
  qpa_hparam.acg_steptype = 'variable';
  qpa_hparam.aipp_type = 'aipp';
  
  % Create the complicated iALM hparams.
  ialm_hparam = base_hparam;
  ialm_hparam.rho0 = hparams.m;
  ialm_hparam.L0 = max([hparams.m, hparams.M]);
  ialm_hparam.rho_vec = hparams.m_constr_vec;
  ialm_hparam.L_vec = hparams.L_constr_vec;
  ialm_hparam.B_vec = hparams.K_constr_vec;
  ialm_hparam.sigma = 2;
  
%   % Old version
%   hparam_arr = {ialm_hparam, qp_hparam, qpa_hparam, rqp_hparam, ipl_hparam, ipla_hparam};
%   name_arr = {'iALM', 'QP', 'QP_A', 'RQP', 'IPL', 'IPL_A'};
%   framework_arr = {@iALM, @penalty, @penalty, @penalty, @IAIPAL, @IAIPAL};
%   solver_arr = {@ECG, @AIPP, @AIPP, @AIPP, @ECG, @ECG};
  
  % Run a benchmark test and print the summary.
  hparam_arr = {ialm_hparam, qpa_hparam, rqp_hparam, ipla0_hparam, ipla1_hparam, ipla2_hparam};
  name_arr = {'iALM', 'QP_A', 'RQP', 'IPL_A0', 'IPL_A1', 'IPL_A2'};
  framework_arr = {@iALM, @penalty, @penalty, @IAIPAL, @IAIPAL, @IAIPAL};
  solver_arr = {@ECG, @AIPP, @AIPP, @ECG, @ECG, @ECG};
  
  % Run the test.
  [summary_tables, out_models] = run_CCM_benchmark(ncvx_lc_qp, framework_arr, solver_arr, hparam_arr, name_arr);
  out_tbl = summary_tables.all;
end


