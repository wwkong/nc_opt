% SPDX-License-Identifier: MIT
% Copyright © 2022 Weiwei "William" Kong

% The function of interest is
%
%  ????

%% Initialization

% Set up paths.
run('../../../init.m');

% Set up variables
seed = 777;
alpha = 1E-2;
beta = 10;
theta = 1E-1;
mu = 1.0;
global_tol = 1e-8;
time_limit = 2400;
iter_limit = 500000;
data_names = {"music_1429u_900m", ...
              "patio_1686u_962m", ...
              "movielens_100k_610u_9724m", ...
              "jester_24938u_100j", ...
              "filmtrust_1508u_2071m"
              };

% Set hyperparameters.
base_hparam = struct();
base_hparam.i_logging = true;

ncf_hparam = base_hparam;
ncf_hparam.m0 = 1;
ncf_hparam.M0 = 1;

upf_hparam = base_hparam;
upf_hparam.line_search_type = 'monotonic';

apd_hparam = base_hparam;
apd_hparam.m0 = 1;
apd_hparam.M0 = 1;
apd_hparam.theta = 4;
apd_hparam.beta = 2;
apd_hparam.alpha = 2;
apd1_hparam = apd_hparam;
apd1_hparam.line_search_type = 'monotonic';
apd2_hparam = apd_hparam;
apd2_hparam.line_search_type = 'optimistic';

aipp_hparam = base_hparam;
aipp_hparam.aipp_type = 'aipp';
aipp_hparam.acg_steptype = 'constant';
aipp_hparam.sigma = 1/4;

%% Generate Tables

% Loop over the curvature pair (m, M).
offset = 0;
for i = 1:length(data_names) - offset
  % Use a problem instance generator to create the oracle and
  % hyperparameters.
  id = data_names{i + offset};
  disp(id);
  load(id);
  rng(seed);
  [oracle, hparams] = test_fn_penalty_spca_01(data, alpha, beta, theta, mu, seed);

  % Create the Model object and specify the solver.
  spca = CompModel(oracle);

  % Set the curvatures and the starting point x0.
  spca.M = hparams.M;
  spca.m = hparams.m;
  spca.x0 = hparams.x0;
  
  % Adjustments for AIPP.
  aipp_hparam.m = hparams.m;
  aipp_hparam.M = hparams.M;

  % Set up the termination criterion.
  spca.opt_type = 'relative';
  spca.opt_tol = global_tol;
  spca.time_limit = time_limit;
  spca.iter_limit = iter_limit;

  % Run a benchmark test and print the summary.
  solver_arr = {@UPFAG, @ADAP_FISTA, @AIPP, @APD};
  hparam_arr = {upf_hparam, ncf_hparam, aipp_hparam, apd2_hparam};
  name_arr = {'UPF', 'ANCF', 'AIPP', 'APD'};
  
%   % Run a benchmark test and print the summary.
%   aipp_hparam.aipp_type = 'aipp';
%   solver_arr = {@AIPP};
%   hparam_arr = {aipp_hparam};
%   name_arr = {'AIPP'};

%   solver_arr = {@ADAP_FISTA, @APD};
%   hparam_arr = {ncf_hparam, apd2_hparam};
%   name_arr = {'ANCF', 'APD2'};
  
  [summary_tables, comp_models] = run_CM_benchmark(spca, solver_arr, hparam_arr, name_arr);
  tbl_row = [table(id), summary_tables.all];
  disp(tbl_row);
  writetable(tbl_row, "adp_sparse_pca_" + id + ".xlsx");
  
  % Set up the final table.
  if (i == 1)
    final_table = tbl_row;
  else
    final_table = [final_table; tbl_row];
  end
end

% Display final table for logging.
disp(final_table);
writetable(final_table, "adp_sparse_pca.xlsx");

%% Generate Plots.
exp_indices = [3, 4, 5];
iter_limits = [10000, 30000, 45000];
plot_x_vals = {};
plot_y_vals = {};
figure;
for l=1:length(exp_indices)

subplot(1, length(exp_indices), l);

id = data_names{exp_indices(l)};
disp(id);
load(id);

upf_hparam.i_logging = true;
ncf_hparam.i_logging = true;
aipp_hparam.i_logging = true;
apd2_hparam.i_logging = true;

[oracle, hparams] =  test_fn_penalty_spca_01(data, alpha, beta, theta, mu, seed);
M = hparams.M;
m = hparams.m;
spca = CompModel(oracle);
spca.M = M;
spca.m = m;
spca.x0 = hparams.x0;

aipp_hparam.m = m;
aipp_hparam.M = M;

spca.opt_type = 'relative';
spca.opt_tol = 1E-8;
spca.time_limit = time_limit;
spca.iter_limit = iter_limits(l);

solver_arr = {@UPFAG, @ADAP_FISTA, @AIPP, @APD};
hparam_arr = {upf_hparam, ncf_hparam, aipp_hparam, apd2_hparam};
name_arr = {'UPF', 'ANCF', 'AIPP', 'APD'};

[summary_tables, comp_models] = run_CM_benchmark(spca, solver_arr, hparam_arr, name_arr);

oracle.eval(hparams.x0);
tol_factor = 1 + hparams.norm_fn(oracle.grad_f_s());
line_styles = {'-.', '--', ':', '-'};
for i=1:max(size(name_arr))
  x = comp_models.(name_arr{i}).history.stationarity_iters;
  y = cummin(comp_models.(name_arr{i}).history.stationarity_values) / tol_factor;
  plot_x_vals{l, i} = x;
  plot_y_vals{l, i} = y;
  semilogy(x, y, line_styles{i}, 'LineWidth', 1.5);
  hold on;
end
title("QSDP Residuals vs. Iterations", 'Interpreter', 'latex');
xlim([1, spca.iter_limit]);
xlabel("Iteration Count", 'Interpreter', 'latex');
ylim([spca.opt_tol, 1E-4]);
ylabel("$$\min_{1\leq i \leq k} \|\bar{v}_i\| / (1 + \|\nabla f(z_0)\|)$$", 'Interpreter', 'latex');
legend(name_arr);
ax = gca;
ax.FontSize = 12;
saveas(gcf, "spca_" + num2str(l) + ".svg");

end
saveas(gcf, "spca.svg");