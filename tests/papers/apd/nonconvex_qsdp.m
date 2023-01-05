% SPDX-License-Identifier: MIT
% Copyright © 2022 Weiwei "William" Kong

% Solve a multivariate nonconvex quadratic programming problem constrained to the unit spectraplex.

% The function of interest is
%
%  f(x) :=  -xi / 2 * ||D * B * x|| ^ 2 + tau / 2 * ||A * x - b|| ^ 2
%
% with curvature pair (m, M). 

%% Initialization

% Set up paths.
run('../../../init.m');

% Set up variables
N = 1000;
seed = 7777777;
dimM = 10;
dimN = 20;
density = 1.0;
global_tol = 1e1;
time_limit = 2400;
iter_limit = 1E6;

%% Generate Tables

% Set hyperparameters.
base_hparam = struct();
base_hparam.i_logging = false;

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
aipp_hparam.aipp_type = 'aipp';
aipp_hparam.acg_steptype = 'constant';
aipp_hparam.sigma = 1/4;

% Loop over the curvature pair (m, M).
base = 10;
mM_vec = [base^2, base^4;  base^2, base^5;  base^2, base^6; ...
          base^3, base^7;  base^2, base^7;  base^1, base^7;];

[nrows, ncols] = size(mM_vec);
offset = 2;

for i = 1:(nrows-offset)
  % Use a problem instance generator to create the oracle and
  % hyperparameters.
  m = mM_vec(i + offset, 1);
  M = mM_vec(i + offset, 2);
  [oracle, hparams] =  test_fn_unconstr_02(N, M, m, seed, dimM, dimN, density);

  % Create the Model object and specify the solver.
  ncvx_qp = CompModel(oracle);

  % Set the curvatures and the starting point x0.
  ncvx_qp.M = hparams.M;
  ncvx_qp.m = hparams.m;
  ncvx_qp.x0 = hparams.x0;
  
  % Adjustments for AIPP.
  aipp_hparam.m = hparams.m;
  aipp_hparam.M = hparams.M;

  % Set up the termination criterion.
  ncvx_qp.opt_type = 'relative';
  ncvx_qp.opt_tol = global_tol;
  ncvx_qp.time_limit = time_limit;
  ncvx_qp.iter_limit = iter_limit;

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

%   solver_arr = {@ECG, @APD};
%   hparam_arr = {pgd_hparam, apd1_hparam};
%   name_arr = {'APGD', 'APD1'};

%   solver_arr = {@ADAP_FISTA, @APD};
%   hparam_arr = {ncf_hparam, apd2_hparam};
%   name_arr = {'ANCF', 'APD2'};
  
%   solver_arr = {@AIPP, @APD};
%   hparam_arr = {aipp_hparam, apd2_hparam};
%   name_arr = {'AIPP', 'APD2'};
  
  [summary_tables, ~] = run_CM_benchmark(ncvx_qp, solver_arr, hparam_arr, name_arr);
  disp(summary_tables.all);
  writetable(summary_tables.all, "adp_qsdp" + num2str(i + offset) + ".xlsx");
  
  % Set up the final table.
  if (i == 1)
    final_table = summary_tables.all;
  else
    final_table = [final_table; summary_tables.all];
  end
end

% Display final table for logging.
disp(final_table);
writetable(final_table, "adp_qsdp.xlsx");

%% Generate Plot data.

m = 1E2;
M = 1E6;

upf_hparam.i_logging = true;
ncf_hparam.i_logging = true;
aipp_hparam.i_logging = true;
apd2_hparam.i_logging = true;

[oracle, hparams] =  test_fn_unconstr_02(N, M, m, seed, dimM, dimN, density);
ncvx_qp = CompModel(oracle);
ncvx_qp.M = hparams.M;
ncvx_qp.m = hparams.m;
ncvx_qp.x0 = hparams.x0;

aipp_hparam.m = hparams.m;
aipp_hparam.M = hparams.M;

ncvx_qp.opt_type = 'relative';
ncvx_qp.opt_tol = 1E-6;
ncvx_qp.time_limit = time_limit;
ncvx_qp.iter_limit = 20000;

solver_arr = {@UPFAG, @ADAP_FISTA, @AIPP, @APD};
hparam_arr = {upf_hparam, ncf_hparam, aipp_hparam, apd2_hparam};
name_arr = {'UPF', 'ANCF', 'AIPP', 'APD'};

[summary_tables, comp_models] = run_CM_benchmark(ncvx_qp, solver_arr, hparam_arr, name_arr);

%% Generate Plots
oracle.eval(hparams.x0);
tol_factor = 1 + hparams.norm_fn(oracle.grad_f_s());
line_styles = {'-.', '--', ':', '-'};
figure;
for i=1:max(size(name_arr))
  semilogy(cummin(comp_models.(name_arr{i}).history.stationarity_values) / tol_factor, ...
           line_styles{i}, 'LineWidth', 1.5);
  hold on;
end
title("QSDP Residuals vs. Iterations", 'Interpreter', 'latex');
xlim([1, 1E4]);
xlabel("Iteration Count", 'Interpreter', 'latex');
ylim([1E-6, 1E-2]);
ylabel("$$\min_{1\leq i \leq k} \|\bar{v}_i\| / (1 + \|\nabla f(z_0)\|)$$", 'Interpreter', 'latex');
legend(name_arr);
ax = gca;
ax.FontSize = 16;
hold off;
saveas(gcf,'qsdp.svg')