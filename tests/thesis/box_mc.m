%% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Solve a box-constrained matrix completion problem using MULTIPLE SOLVERS.

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
theta = sqrt(2);
mu = 2;
seed = 777;
data_name = 'jester_24938u_100j';
global_opt_tol = 1e-2;
global_feas_tol = 1e-2;
time_limit = 4000;

% -------------------------------------------------------------------------
%% Table 1
% -------------------------------------------------------------------------
disp('========')
disp('TABLE 1');
disp('========')
% Loop over the upper curvature M.
beta_vec = [1/2, 1, 2];
for i = 1:length(beta_vec)
  % Use a problem instance generator to create the oracle and hyperparameters.
  beta = beta_vec(i);
  [oracle, hparams] = test_fn_bmc_01(data_name, beta, theta, mu, seed);

  % Create the Model object and specify the solver.
  ncvx_box_mc = ConstrCompModel(oracle);

  % Set the curvatures, the starting point x0, and special functions.
  ncvx_box_mc.M = hparams.M;
  ncvx_box_mc.m = hparams.m;
  ncvx_box_mc.x0 = hparams.x0;
  ncvx_box_mc.K_constr = hparams.K_constr;
  
  % Add linear constraints.
  ncvx_box_mc.constr_fn = @(x) hparams.constr_fn(x);
  ncvx_box_mc.grad_constr_fn = hparams.grad_constr_fn;

  % Set up the termination criterion.
  ncvx_box_mc.opt_type = 'relative';
  ncvx_box_mc.feas_type = 'relative';
  ncvx_box_mc.opt_tol = global_opt_tol;
  ncvx_box_mc.feas_tol = global_feas_tol;
  ncvx_box_mc.time_limit = time_limit;

  % Run a benchmark test and print the summary.
  hparam_arr = {aipp_hparam, base_hparam};
  name_arr = {'AIP_QP', 'AG_QP'};
  framework_arr = {@penalty, @penalty};
  solver_arr = {@AIPP, @AG};
  [summary_tables, comp_models] = run_CCM_benchmark(ncvx_box_mc, framework_arr, solver_arr, hparam_arr, name_arr);
  
  % Set up the final table.
  sub_table = summary_tables.all;
  sub_table.beta = beta;
  disp(sub_table);
  if (i == 1)
    final_table = sub_table;
  else
    final_table = [final_table; sub_table];
  end
end

% Display final table for logging.
disp(final_table);
