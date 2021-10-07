% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Solve a sparse PCA problem.

% Set up paths.
run('../../init.m');

% -------------------------------------------------------------------------
%% Global Variables
% -------------------------------------------------------------------------

% Create basic hparams.
base_hparam = struct();
aipp_hparam = base_hparam;
aipp_hparam.aipp_type = 'aipp_v2';

% Create global hyperparams
b = 0.1;
nu = 100;
p = 100;
n = 100;
k = 1;
seed = 777;
global_opt_tol = 1e-6;
global_feas_tol = 1e-3;
time_limit = 4000;

% -------------------------------------------------------------------------
%% Table 1
% -------------------------------------------------------------------------
disp('========')
disp('TABLE 1');
disp('========')
% Loop over the upper curvature M.
s_vec = [5, 10, 15];
for i = 1:length(s_vec)
  % Use a problem instance generator to create the oracle and hyperparameters.
  s = s_vec(i);
  [oracle, hparams] = test_fn_spca_01(b, nu, p, n, s, k, seed);

  % Create the Model object and specify the solver.
  spca = ConstrCompModel(oracle);

  % Set the curvatures, the starting point x0, and special functions.
  spca.M = hparams.M;
  spca.m = hparams.m;
  spca.x0 = hparams.x0;
  spca.K_constr = hparams.K_constr;
  
  % Add linear constraints.
  spca.constr_fn = @(x) hparams.constr_fn(x);
  spca.grad_constr_fn = hparams.grad_constr_fn;

  % Set up the termination criterion.
  spca.opt_type = 'relative';
  spca.feas_type = 'relative';
  spca.opt_tol = global_opt_tol;
  spca.feas_tol = global_feas_tol;
  spca.time_limit = time_limit;

  % Run a benchmark test and print the summary.
  hparam_arr = {aipp_hparam, base_hparam};
  name_arr = {'AIP_QP', 'AG_QP'};
  framework_arr = {@penalty, @penalty};
  solver_arr = {@AIPP, @AG};
  [summary_tables, comp_models] = run_CCM_benchmark(spca, framework_arr, solver_arr, hparam_arr, name_arr);
  
   % Set up the final table.
  sub_table = summary_tables.all;
  sub_table.s = s;
  disp(sub_table);
  if (i == 1)
    final_table = sub_table;
  else
    final_table = [final_table; sub_table];
  end
end

% Display final table for logging.
disp(final_table);
