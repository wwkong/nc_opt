%% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Solve a multivariate nonconvex quadratic programming problem constrained to the unit simplex intersected with an affine manifold. 

% The function of interest is
%
%  f(x) :=  -xi / 2 * ||D * B * x|| ^ 2 + tau / 2 * ||A * x - b|| ^ 2
%
% with curvature pair (m, M). 

% Set up paths.
run('../../../init.m');

% -------------------------------------------------------------------------
%% Global Variables
% -------------------------------------------------------------------------

% .........................................................................
% Create basic hparams.
base_hparam = struct();
spa_hparam = base_hparam;
spa_hparam.Gamma = 10;

% End basic hparams.
% .........................................................................

% Create global hyperparams
N = 1000;
seed = 777;
dimM = 10;
dimN = 50;
global_tol = 1e-3;
time_limit = 3600;

% -------------------------------------------------------------------------
%% Table 1
% -------------------------------------------------------------------------
disp('========')
disp('TABLE 1');
disp('========')
% Loop over the upper curvature M.
M_vec = [1e2, 1e3, 1e4, 1e5, 1e6];

% % DEBUG
% M_vec = 1e3;

for i = 1:length(M_vec)
  % Use a problem instance generator to create the oracle and
  % hyperparameters.
  M = M_vec(i);
  m = M / 3;
  [oracle, hparams] = test_fn_lin_cone_constr_01(N, M, m, seed, dimM, dimN);

  % Create the Model object and specify the solver.
  ncvx_lc_qp = ConstrCompModel(oracle);

  % Set the curvatures, the starting point x0, and special functions.
  ncvx_lc_qp.M = hparams.M;
  ncvx_lc_qp.m = hparams.m;
  ncvx_lc_qp.x0 = hparams.x0;
  ncvx_lc_qp.K_constr = hparams.K_constr;
  
  % Add linear constraints.
  ncvx_lc_qp.constr_fn = @(x) hparams.constr_fn(x);
  ncvx_lc_qp.grad_constr_fn = hparams.grad_constr_fn;

  % Set up the termination criterion.
  ncvx_lc_qp.opt_type = 'relative';
  ncvx_lc_qp.feas_type = 'relative';
  ncvx_lc_qp.opt_tol = global_tol;
  ncvx_lc_qp.feas_tol = global_tol;
  ncvx_lc_qp.time_limit = time_limit;

  % Run a benchmark test and print the summary.
  hparam_arr = {spa_hparam};
  name_arr = {'SPA'};
  framework_arr = { @sProxALM};
  solver_arr = {@ECG};
  
  [summary_tables, comp_models] = run_CCM_benchmark(ncvx_lc_qp, framework_arr, solver_arr, hparam_arr, name_arr);
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
writetable(final_table, 'aidal_qvp_sProxALM.xlsx')
