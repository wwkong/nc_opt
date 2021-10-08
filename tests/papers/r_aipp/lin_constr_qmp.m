% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Solve a multivariate nonconvex quadratic programming problem constrained to the unit spectraplex intersected with an affine manifold.
% using MULTIPLE SOLVERS.

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
dimM = 50;
dimN = 200;
density = 0.01;
global_tol = 1e-3;
time_limit = 4000;

% -------------------------------------------------------------------------
%% Table
% -------------------------------------------------------------------------
% Loop over the upper curvature M.
mM_vec = ...
  [1e0, 1e1;
   1e0, 1e2;
   1e0, 1e3];
[nrows, ncols] = size(mM_vec);
for i = 1:nrows
  % Use a problem instance generator to create the oracle and
  % hyperparameters.
  m = mM_vec(i, 1);
  M = mM_vec(i, 2);
  [oracle, hparams] = test_fn_lin_cone_constr_02(N, M, m, seed, dimM, dimN, density);

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
  ncvx_lc_qp.feas_type = 'absolute';
  ncvx_lc_qp.opt_tol = global_tol;
  ncvx_lc_qp.feas_tol = global_tol;
  ncvx_lc_qp.time_limit = time_limit;
  
  % Create the complicated iALM hparams.
  ialm_hparam = base_hparam;
  ialm_hparam.rho0 = hparams.m;
  ialm_hparam.L0 = max([hparams.m, hparams.M]);
  ialm_hparam.rho_vec = hparams.m_constr_vec;
  ialm_hparam.L_vec = hparams.L_constr_vec;
  % Note that we are using the fact that |X|_F <= 1 over the spectraplex.
  ialm_hparam.B_vec = hparams.K_constr_vec;

  % Run a benchmark test and print the summary.
  solver_arr = {@UPFAG, @NC_FISTA, @AG, @AIPP, @AIPP, @AIPP};
  hparam_arr = {base_hparam, base_hparam, base_hparam, aipp_c_hparam, aipp_v1_hparam, aipp_v2_hparam};
  name_arr = {'UPFAG', 'NC_FISTA', 'AG', 'AIPP_c', 'AIPP_v1', 'AIPP_v2'};
  framework_arr = {@penalty, @penalty, @penalty, @penalty, @penalty, @penalty}; 
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
