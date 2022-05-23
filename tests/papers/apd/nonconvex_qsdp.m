% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

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
seed = 7;
dimM = 10;
dimN = 35;
density = 1.0;
global_tol = 1e-5;
time_limit = 2000;

% Set hyperparameters.
base_hparam = struct();

ncf_hparam = base_hparam;
ncf_hparam.m0 = 1;
ncf_hparam.M0 = 1;

upf_hparam = base_hparam;
upf_hparam.line_search_type = 'monotonic';

apd_hparam = base_hparam;
apd_hparam.m0 = 1;
apd_hparam.M0 = 1;
apd1_hparam = apd_hparam;
apd1_hparam.line_search_type = 'monotonic';
apd2_hparam = base_hparam;
apd2_hparam.line_search_type = 'optimistic';

pgd_hparam = base_hparam;
pgd_hparam.m0 = global_tol;
pgd_hparam.M0 = 1;
pgd_hparam.steptype = 'adaptive';

% Loop over the curvature pair (m, M).
base = 5;
mM_vec = [base^1, base^3;  base^1, base^4;  base^1, base^5; ...
          base^2, base^5;  base^3, base^5;  base^4, base^5; ];

% mM_vec = [1E0, 1E5];

[nrows, ncols] = size(mM_vec);

for i = 1:nrows
  % Use a problem instance generator to create the oracle and
  % hyperparameters.
  m = mM_vec(i, 1);
  M = mM_vec(i, 2);
  [oracle, hparams] =  test_fn_unconstr_02(N, M, m, seed, dimM, dimN, density);

  % Create the Model object and specify the solver.
  ncvx_qp = CompModel(oracle);

  % Set the curvatures and the starting point x0.
  ncvx_qp.M = hparams.M;
  ncvx_qp.m = hparams.m;
  ncvx_qp.x0 = hparams.x0;

  % Set up the termination criterion.
  ncvx_qp.opt_type = 'relative';
  ncvx_qp.opt_tol = global_tol;
  ncvx_qp.time_limit = time_limit;

  % Run a benchmark test and print the summary.
  solver_arr = {@ECG, @UPFAG, @ADAP_FISTA, @APD};
  hparam_arr = {pgd_hparam, upf_hparam, ncf_hparam, apd2_hparam};
  name_arr = {'APGD', 'UPF', 'ANCF', 'APD'};
  
%   solver_arr = {@UPFAG, @APD};
%   hparam_arr = {upf_hparam, apd1_hparam};
%   name_arr = {'UPF', 'APD1'};

%   solver_arr = {@APD};
%   hparam_arr = {apd1_hparam};
%   name_arr = {'APD1'};

%   solver_arr = {@ECG, @APD};
%   hparam_arr = {pgd_hparam, apd1_hparam};
%   name_arr = {'APGD', 'APD1'};

%   solver_arr = {@ADAP_FISTA, @APD};
%   hparam_arr = {ncf_hparam, apd2_hparam};
%   name_arr = {'ANCF', 'APD2'};
  
  [summary_tables, comp_models] = run_CM_benchmark(ncvx_qp, solver_arr, hparam_arr, name_arr);
  disp(summary_tables.all);
  writetable(summary_tables.all, "adp_qsdp" + num2str(i) + ".xlsx");
  
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
