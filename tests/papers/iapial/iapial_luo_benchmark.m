% Solve a multivariate nonconvex quadratic programming problem 
% constrained to the unit simplex using MULTIPLE SOLVERS.

% The function of interest is
%
%  f(x) :=  -xi / 2 * ||D * B * X|| ^ 2 + tau / 2 * ||A * X - b|| ^ 2
%
% with curvature pair (m, M). 

% NOTE: the purpose of this experiment is to benchmark IAPIAL with Tom
% Luo's proximal method.

%% Initialization

% Set up paths.
run('../../../init.m');

% Global parameters for each experiment.
globals.N = 1000;
globals.seed = 777;
globals.dimM = 20;
globals.dimN = 1000;
globals.opt_tol = 1e-20;
globals.feas_tol = 1e-20;

% Run a single experiment.
m = 1e1;
M = 1e3;
time_limit = 200;
out_models = run_experiment(m, M, time_limit, globals);
save('luo_models.mat', 'out_models')

%% Create the graphs for the paper.
load('luo_models.mat')
generate_graphs(out_models)

%% Helper Functions

% Generate the graphs for the paper.
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

% Run a single experiment and output the summary models.
function out_models = run_experiment(m, M, time_limit, params) 

  % Gather the oracle and hparam instance.
  [oracle, hparams] = ...
    test_fn_lin_cone_constr_01(...
      params.N, M, m, params.seed, params.dimM, params.dimN);

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
  base_hparam.i_logging = true;
  spa1_hparam = base_hparam;
  spa1_hparam.Gamma =0.1;
  spa2_hparam = base_hparam;
  spa2_hparam.Gamma = 1;
  spa3_hparam = base_hparam;
  spa3_hparam.Gamma = 10;  
  iapial_hparam = base_hparam;
  iapial_hparam.acg_steptype = 'constant';
  
  % Run a benchmark test and print the summary.
  hparam_arr = {iapial_hparam, spa1_hparam, spa2_hparam, spa3_hparam};
  name_arr = {'IAPIAL', 'SPA1', 'SPA2', 'SPA3'};
  framework_arr = {@IAPIAL, @sProxALM, @sProxALM, @sProxALM};
  solver_arr = {@ECG, @ECG, @ECG, @ECG};
  
  % Run the test.
  [~, out_models] = run_CCM_benchmark(...
    ncvx_lc_qp, framework_arr, solver_arr, hparam_arr, name_arr);
  
end