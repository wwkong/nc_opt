% SPDX-License-Identifier: MIT
% Copyright © 2022 Weiwei "William" Kong

% The function of interest is
%
%  f(z) :=  (1 / k) * sum_{i=1,..,k} (1 - tanh(v_i * <u_i, z>)) + (1 / 2*k) ||z|| ^ 2.

%% Initialization

% Set up paths.
run('../../../init.m');

% Set up variables
N = 10;
seed = 777;
r = 10;
density = 1.0;
global_tol = 1e-4;
time_limit = 1200;
iter_limit = 1E6;

%% Generate Tables

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

pgd_hparam = base_hparam;
pgd_hparam.m0 = global_tol;
pgd_hparam.M0 = 1;
pgd_hparam.steptype = 'adaptive';

aipp_hparam = base_hparam;
aipp_hparam.steptype = 'aipp';
aipp_hparam.acg_steptype = 'constant';
aipp_hparam.sigma = 1/4;

% Loop over the curvature pair (m, M).
nk_vec = ...
  [200, 100;
%    400, 200;
%    600, 300;
%    800, 400;
%    1000, 500;
   ];
[nrows, ncols] = size(nk_vec);
offset = 0;

for i = 1:(nrows-offset)
  % Use a problem instance generator to create the oracle and
  % hyperparameters.
  n = nk_vec(i, 1);
  k = nk_vec(i, 2);
  [oracle, hparams] = test_fn_svm_01(n, k, seed, density, r);

  % Create the Model object and specify the solver.
  ncvx_svm = CompModel(oracle);

  % Set the curvatures and the starting point x0.
  ncvx_svm.M = hparams.M;
  ncvx_svm.m = hparams.m;
  ncvx_svm.x0 = hparams.x0 + 1;
  
  % Adjustments for AIPP.
  aipp_hparam.m = hparams.m;
  aipp_hparam.M = hparams.M;

  % Set up the termination criterion.
  ncvx_svm.opt_type = 'relative';
  ncvx_svm.opt_tol = global_tol;
  ncvx_svm.time_limit = time_limit;
  ncvx_svm.iter_limit = iter_limit;

  % Run a benchmark test and print the summary.
  solver_arr = {@UPFAG, @ADAP_FISTA, @AIPP, @APD};
  hparam_arr = {upf_hparam, ncf_hparam, aipp_hparam, apd2_hparam};
  name_arr = {'UPF', 'ANCF', 'AIPP', 'APD'};
  
%   solver_arr = {@UPFAG, @APD};
%   hparam_arr = {upf_hparam, apd1_hparam};
%   name_arr = {'UPF', 'APD1'};

%   solver_arr = {@APD};
%   hparam_arr = {apd2_hparam};
%   name_arr = {'APD2'};

%   solver_arr = {@ADAP_FISTA, @APD};
%   hparam_arr = {ncf_hparam, apd1_hparam};
%   name_arr = {'NCF', 'APD1'};

  solver_arr = {@ADAP_FISTA, @APD};
  hparam_arr = {ncf_hparam, apd2_hparam};
  name_arr = {'ANCF', 'APD2'};
  
%   solver_arr = {@AIPP, @APD};
%   hparam_arr = {aipp_hparam, apd2_hparam};
%   name_arr = {'AIPP', 'APD2'};
  
  [summary_tables, comp_models] = run_CM_benchmark(ncvx_svm, solver_arr, hparam_arr, name_arr);
  disp(summary_tables.all);
  writetable(summary_tables.all, "adp_svm" + num2str(i + offset) + ".xlsx");
  
  % Set up the final table.
  if (i == 1)
    final_table = summary_tables.all;
  else
    final_table = [final_table; summary_tables.all];
  end
end

% Display final table for logging.
disp(final_table);
writetable(final_table, "adp_svm.xlsx");

%% Generate Plot data.
% 
% m = 1E2;
% M = 1E6;
% 
% upf_hparam.i_logging = true;
% ncf_hparam.i_logging = true;
% aipp_hparam.i_logging = true;
% apd2_hparam.i_logging = true;
% 
% [oracle, hparams] =  test_fn_unconstr_02(N, M, m, seed, dimM, dimN, density);
% ncvx_svm = CompModel(oracle);
% ncvx_svm.M = hparams.M;
% ncvx_svm.m = hparams.m;
% ncvx_svm.x0 = hparams.x0;
% 
% aipp_hparam.m = hparams.m;
% aipp_hparam.M = hparams.M;
% 
% ncvx_svm.opt_type = 'relative';
% ncvx_svm.opt_tol = 1E-6;
% ncvx_svm.time_limit = time_limit;
% ncvx_svm.iter_limit = 20000;
% 
% solver_arr = {@UPFAG, @ADAP_FISTA, @AIPP, @APD};
% hparam_arr = {upf_hparam, ncf_hparam, aipp_hparam, apd2_hparam};
% name_arr = {'UPF', 'ANCF', 'AIPP', 'APD'};
% 
% [summary_tables, comp_models] = run_CM_benchmark(ncvx_svm, solver_arr, hparam_arr, name_arr);
% 
% %% Generate Plots
% oracle.eval(hparams.x0);
% tol_factor = 1 + hparams.norm_fn(oracle.grad_f_s());
% line_styles = {'-.', '--', ':', '-'};
% figure;
% for i=1:max(size(name_arr))
%   semilogy(cummin(comp_models.(name_arr{i}).history.stationarity_values) / tol_factor, ...
%            line_styles{i}, 'LineWidth', 1.5);
%   hold on;
% end
% title("QSDP Residuals vs. Iterations", 'Interpreter', 'latex');
% xlim([1, 1E4]);
% xlabel("Iteration Count", 'Interpreter', 'latex');
% ylim([1E-6, 1E-2]);
% ylabel("$$\min_{1\leq i \leq k} \|\bar{v}_i\| / (1 + \|\nabla f(z_0)\|)$$", 'Interpreter', 'latex');
% legend(name_arr);
% ax = gca;
% ax.FontSize = 16;
% hold off;
% saveas(gcf,'qsdp.svg')