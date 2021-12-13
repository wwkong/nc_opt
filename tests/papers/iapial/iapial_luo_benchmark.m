%% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Solve a multivariate nonconvex quadratic programming problem constrained to the unit simplex.

% The function of interest is
%
%  f(x) :=  -xi / 2 * ||D * B * X|| ^ 2 + tau / 2 * ||A * X - b|| ^ 2
%
% with curvature pair (m, M). 

% NOTE: the purpose of this experiment is to benchmark IAPIAL with Tom Luo's proximal method.

%% Initialization

% Set up paths.
run('../../../init.m');

% Global parameters for each experiment.
globals.N = 1000;
globals.seed = 777;
globals.dimM = 20;
globals.dimN = 1000;
globals.opt_tol = 1e-8;
globals.feas_tol = 1e-8;

%% Run a single experiment.

% The main parameters (m, M) should be spec'd by Condor.

% E.g.
% mM_mat = ...
%   [1e1, 1e2; ...
%    1e1, 1e3; ...
%    1e1, 1e4; ...
%    1e1, 1e5; ...
%    1e1, 1e6; ];

% Run an experiment.
time_limit = 600;
i_first_row = true;
for i=1:size(mM_mat, 1)
  [~, out_models] = run_experiment(mM_mat(i, 1), mM_mat(i, 2), time_limit, globals);
  tbl_row = parse_models(out_models);
  disp(tbl_row);
  if i_first_row
    tbl = tbl_row;
    i_first_row = false;
  else
    tbl = [tbl; tbl_row]; 
  end
end
disp(tbl);


%% Create the graphs for the paper. (DEPRECATED)
% save('luo_models.mat', 'out_models')
% load('luo_models.mat')
% generate_graphs(out_models)

%% Helper Functions

% Generate the graphs for the paper. (DEPRECATED)
function generate_graphs(models)

  % Initialize.
  alg_names = fieldnames(models);
  plot_opts = {'LineWidth', 1.2};
  
  % Loop over the different algorithms.
  for i=1:length(alg_names)
    cur_hist = models.(alg_names{i}).history;
    
    % log(|w_hat|) vs. time
    figure(1); 
    hold on;
    plot(cur_hist.time_values, log(cur_hist.norm_w_hat_values), plot_opts{:});
    hold off;
    xlabel('Time (seconds)', 'interpreter', 'latex');
    ylabel('$\log\|\hat{w}\|$', 'interpreter', 'latex');
    title('Stationarity over time', 'interpreter', 'latex');
    
    % log(|q_hat|) vs. time
    figure(2); 
    hold on;
    plot(cur_hist.time_values, log(cur_hist.norm_q_hat_values), plot_opts{:});
    hold off;
    xlabel('Time (seconds)', 'interpreter', 'latex');
    ylabel('$\log\|A\hat{z}-b\|$', 'interpreter', 'latex');
    title('Feasibility over time', 'interpreter', 'latex');
    
    % log(|w_hat|) vs. iter
    figure(3); 
    hold on;
    plot(cur_hist.iteration_values, log(cur_hist.norm_w_hat_values), plot_opts{:});
    hold off;
    xlabel('Iteration count', 'interpreter', 'latex');
    ylabel('$\log\|\hat{w}\|$', 'interpreter', 'latex');
    title('Stationarity over iteration count', 'interpreter', 'latex');
    
    % log(|q_hat|) vs. iter
    figure(4); 
    hold on;
    plot(cur_hist.iteration_values, log(cur_hist.norm_q_hat_values), plot_opts{:});
    hold off;
    xlabel('Iteration count', 'interpreter', 'latex');
    ylabel('$\log\|A\hat{z}-b\|$', 'interpreter', 'latex');
    title('Feasibility over iteration count', 'interpreter', 'latex');
  end
  
  % Output to SVG.
  fheight = 6;
  fwidth = 16;
  
  figure(1);
  legend(alg_names);
  set(gcf, 'PaperUnits', 'centimeters');
  set(gcf, 'PaperPosition', [0 0 fwidth fheight]);
  saveas(gcf, 'luo_wt.svg', 'svg')
  
  figure(2);
  legend(alg_names);
  set(gcf, 'PaperUnits', 'centimeters');
  set(gcf, 'PaperPosition', [0 0 fwidth fheight]);
  saveas(gcf, 'luo_qt.svg', 'svg')
  
  figure(3);
  legend(alg_names);
  set(gcf, 'PaperUnits', 'centimeters');
  set(gcf, 'PaperPosition', [0 0 fwidth fheight]);
  saveas(gcf, 'luo_wi.svg', 'svg')
  
  figure(4);
  legend(alg_names);
  set(gcf, 'PaperUnits', 'centimeters');
  set(gcf, 'PaperPosition', [0 0 fwidth fheight]);
  saveas(gcf, 'luo_qi.svg', 'svg')
  
end

% Parse the output models and log the output.
function out_tbl = parse_models(models)

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
  
end

% Run a single experiment and output the summary models.
function [out_tbl, out_models] = run_experiment(m, M, time_limit, params) 

  % Gather the oracle and hparam instance.
  [oracle, hparams] = test_fn_lin_cone_constr_01(params.N, M, m, params.seed, params.dimM, params.dimN);

  % Create the Model object and specify the limits (if any).
  ncvx_lc_qp = ConstrCompModel(oracle);
  ncvx_lc_qp.time_limit = time_limit;
  
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
  base_hparam.i_logging = false; % Enable or diable logging.
  spa1_hparam = base_hparam;
  spa1_hparam.Gamma =0.1;
  spa2_hparam = base_hparam;
  spa2_hparam.Gamma = 1;
  spa3_hparam = base_hparam;
  spa3_hparam.Gamma = 10;  
  rqp_hparam = base_hparam;
  rqp_hparam.acg_steptype = 'variable';
  ipla_hparam = base_hparam;
  ipla_hparam.acg_steptype = 'variable';
  qpa_hparam = base_hparam;
  qpa_hparam.acg_steptype = 'variable';
  qpa_hparam.aipp_type = 'aipp';
  
  % Run a benchmark test and print the summary.
  hparam_arr = {qpa_hparam, rqp_hparam, ipla_hparam, spa1_hparam, spa2_hparam, spa3_hparam};
  name_arr = {'QP_A', 'RQP', 'IPL_A', 'SPA1', 'SPA2', 'SPA3'};
  framework_arr = {@penalty, @penalty, @IAIPAL, @sProxALM, @sProxALM, @sProxALM};
  solver_arr = {@AIPP, @AIPP, @ECG, @ECG, @ECG, @ECG};
  
  % Run the test.
  [out_tbl, out_models] = run_CCM_benchmark(ncvx_lc_qp, framework_arr, solver_arr, hparam_arr, name_arr);
  
end
