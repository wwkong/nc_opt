% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Solve a multivariate nonconvex quadratic programming problem constrained to the unit spectraplex.

% The function of interest is
%
%  f(x) :=  -xi / 2 * ||D * B * X|| ^ 2 + tau / 2 * ||A * X - b|| ^ 2
%
% with curvature pair (m, M). 

% Set up paths.
run('../../init.m');

% % Set up curvatures (this can alternatively be set in Condor).
% M = 1000; % Same as L_f
% m = 10;

% Use a problem instance generator to create the oracle and
% hyperparameters.
N = 1000;
seed = 777;
dimM = 10;
dimN = 50;
density = 0.01;
[oracle, hparams] = test_fn_lin_cone_constr_02(N, M, m, seed, dimM, dimN, density);

% Create the Model object and specify the limits (if any).
ncvx_lc_qp = ConstrCompModel(oracle);
ncvx_lc_qp.time_limit = 10000;

% Set the curvatures and the starting point x0.
ncvx_lc_qp.x0 = hparams.x0;
ncvx_lc_qp.M = hparams.M;
ncvx_lc_qp.m = hparams.m;
ncvx_lc_qp.K_constr = hparams.K_constr;

% Set the tolerances
ncvx_lc_qp.opt_tol = 1e-4;
ncvx_lc_qp.feas_tol = 1e-4;

% Add linear constraints.
ncvx_lc_qp.constr_fn = @(x) hparams.constr_fn(x);
ncvx_lc_qp.grad_constr_fn = hparams.grad_constr_fn;

% Use a relative termination criterion.
ncvx_lc_qp.feas_type = 'relative';
ncvx_lc_qp.opt_type = 'relative';

% Create some basic hparams.
base_hparam = struct();
aipp_hparam = base_hparam;
aipp_hparam.aipp_type = 'aipp';

% Create the complicated iALM hparams.
ialm_hparam = base_hparam;
ialm_hparam.rho0 = hparams.m;
ialm_hparam.L0 = max([hparams.m, hparams.M]);
ialm_hparam.rho_vec = hparams.m_constr_vec;
ialm_hparam.L_vec = hparams.L_constr_vec;
ialm_hparam.B_vec = hparams.K_constr_vec;

% Run a benchmark test and print the summary.
hparam_arr = {ialm_hparam, aipp_hparam, base_hparam};
name_arr = {'iALM', 'QP_AIPP', 'IAPIAL'};
framework_arr = {@iALM, @penalty, @IAIPAL};
solver_arr = {@ECG, @AIPP, @ECG};

% Run the test.
[summary_tables, comp_models] = run_CCM_benchmark(ncvx_lc_qp, framework_arr, solver_arr, hparam_arr, name_arr);
disp(summary_tables.all);

%% Print the history for additional diagnostics.
fprintf('IAPIAL History: \n\n')
disp(comp_models.IAPIAL.history);
fprintf('iALM History: \n\n')
disp(comp_models.iALM.history);
