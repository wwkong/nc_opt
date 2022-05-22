% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Solve a multivariate nonconvex quadratic programming problem constrained to the unit spectraplex.

% The function of interest is
%
%  f(x) :=  -xi / 2 * ||D * B * x|| ^ 2 + tau / 2 * ||A * x - b|| ^ 2
%
% with curvature pair (m, M). 

% Initialize
N = 100;
seed = 777;
dimM = 10;
dimN = 35;
density = 0.70;
global_tol = 1e-4;
time_limit = 1200;

% Set hyperparameters
base_hparam = struct();
base_hparam.m0 = global_tol;
base_hparam.M0 = 1;
apd_hparam = base_hparam;
apd_hparam.line_search_type = 'optimistic';
pgd_hparam = base_hparam;
pgd_hparam.steptype = 'adaptive';

% Loop over the curvature pair (m, M).
mM_vec = [1E1, 1E4;     1E1, 1E5;     1E1, 1E6; ...
          1E2, 1E6;     1E3, 1E6;     1E4, 1E6;];
        
mM_vec = [1E1, 1E5;];

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
  solver_arr = {@ECG, @UPFAG, @NC_FISTA, @APD};
  hparam_arr = {pgd_hparam, base_hparam, base_hparam, apd_hparam};
  name_arr = {'PGD', 'UPF', 'NCF', 'APD'};
 
%   solver_arr = {@UPFAG, @APD};
%   hparam_arr = {base_hparam, apd_hparam};
%   name_arr = {'UPF', 'APD'};
% 
%   solver_arr = {@APD};
%   hparam_arr = {apd_hparam};
%   name_arr = {'APD'};
% 
%   solver_arr = {@ECG, @APD};
%   hparam_arr = {apgd_hparam, apd_hparam};
%   name_arr = {'APGD', 'APD'};
  
  [summary_tables, comp_models] = run_CM_benchmark(ncvx_qp, solver_arr, hparam_arr, name_arr);
  disp(summary_tables.all);
  
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
