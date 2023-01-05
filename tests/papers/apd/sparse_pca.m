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
time_limit = 1200;
iter_limit = 200000;
data_names = {"jester_24938u_100j", ...
              "anime_500K_506u_9437a", ...
              "movielens_100k_610u_9724m", ...
              "filmtrust_1508u_2071m", ...
              "movielens_1m_6040u_3952m"};

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
  id = data_names{i};
  load(id);
  rng(seed);
  [oracle, hparams] = test_fn_penalty_spca_01(data, alpha, beta, theta, mu, seed);

  % Create the Model object and specify the solver.
  sparse_pca_exp = CompModel(oracle);

  % Set the curvatures and the starting point x0.
  sparse_pca_exp.M = hparams.M;
  sparse_pca_exp.m = hparams.m;
  sparse_pca_exp.x0 = hparams.x0;
  
  % Adjustments for AIPP.
  aipp_hparam.m = hparams.m;
  aipp_hparam.M = hparams.M;

  % Set up the termination criterion.
  sparse_pca_exp.opt_type = 'relative';
  sparse_pca_exp.opt_tol = global_tol;
  sparse_pca_exp.time_limit = time_limit;
  sparse_pca_exp.iter_limit = iter_limit;

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
  
  [summary_tables, comp_models] = run_CM_benchmark(sparse_pca_exp, solver_arr, hparam_arr, name_arr);
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