% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Solve a sigmoidal SVM problem.

% The function of interest is
%
%  f(z) :=  (1 / k) * sum_{i=1,..,k} (1 - tanh(v_i * <u_i, z>)) + (1 / 2*k) ||z|| ^ 2.

% Set up paths.
run('../../../init.m');

% -------------------------------------------------------------------------
%% Global Variables
% -------------------------------------------------------------------------

% Create basic hparams.
base_hparam = struct();
aipp_hparam = base_hparam;
aipp_hparam.tau = 5000; % SET A SPECIAL VALUE OF TAU
aipp_c_hparam = aipp_hparam;
aipp_c_hparam.aipp_type = 'aipp_c';
aipp_v1_hparam = aipp_hparam;
aipp_v1_hparam.aipp_type = 'aipp_v1';
aipp_v2_hparam = aipp_hparam;
aipp_v2_hparam.aipp_type = 'aipp_v2';

% Create global hyperparams
N = 1000;
seed = 777;
r = 50;
density = 0.05;
opt_tol = 1e-3;
time_limit = 4000;

% -------------------------------------------------------------------------
%% Table 
% -------------------------------------------------------------------------
% Loop over the upper curvature M.
nk_vec = ...
  [1000, 500;
   2000, 1000;
   4000, 2000;
   8000, 4000];
[nrows, ncols] = size(nk_vec);
for i = 1:nrows
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
  ncvx_svm.x0 = hparams.x0;

  % Set up the termination criterion.
  ncvx_svm.opt_type = 'relative';
  ncvx_svm.opt_tol = opt_tol;
  ncvx_svm.time_limit = time_limit;

  % Run a benchmark test and print the summary.
  solver_arr = {@UPFAG, @NC_FISTA, @AG, @AIPP, @AIPP, @AIPP};
  hparam_arr = {base_hparam, base_hparam, base_hparam, aipp_c_hparam, aipp_v1_hparam, aipp_v2_hparam};
  name_arr = {'UPFAG', 'NC_FISTA', 'AG', 'AIPP_c', 'AIPP_v1', 'AIPP_v2'};
  [summary_tables, comp_models] = run_CM_benchmark(ncvx_svm, solver_arr, hparam_arr, name_arr);
  
  % Set up the final table.
  sub_table = summary_tables.all;
  sub_table.n = n;
  sub_table.k = k;
  disp(sub_table);
  if (i == 1)
    final_table = sub_table;
  else
    final_table = [final_table; sub_table];
  end
end

% Display final table for logging.
disp(final_table);
