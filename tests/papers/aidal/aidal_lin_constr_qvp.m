% Solve a multivariate nonconvex quadratic programming problem 
% constrained to the unit simplex intersected with an affine manifold 
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

% .........................................................................
% Create basic hparams.
base_hparam = struct();

ialm_hparam = base_hparam;
ialm_hparam.i_ineq_constr = false;

qp_aipp_hparam = base_hparam;
qp_aipp_hparam.aipp_type = 'aipp';
qp_aipp_hparam.acg_steptype = 'variable';
qp_aipp_hparam.i_reset_prox_center = true;

rqp_aipp_hparam = base_hparam;
rqp_aipp_hparam.aipp_type = 'aipp_v2';
qp_aipp_hparam.acg_steptype = 'variable';
rqp_aipp_hparam.i_reset_prox_center = false;

iapial_hparam = base_hparam;
iapial_hparam.acg_steptype = 'variable';
iapial_hparam.sigma_min = 0.3;
iapial_hparam.penalty_multiplier = 2;
iapial_hparam.i_reset_multiplier = false;
iapial_hparam.i_reset_prox_center = false;

aidal_hparam = base_hparam;
aidal_hparam.acg_steptype = 'variable';
aidal_hparam.sigma_type = 'constant';
aidal_hparam.sigma_min = 0.3;
aidal0_hparam = aidal_hparam;
aidal0_hparam.theta = 0;
aidal0_hparam.chi = 1;
aidal1_hparam = aidal_hparam;
aidal1_hparam.theta = 0.5;
aidal2_hparam = aidal_hparam;
aidal2_hparam.theta = 0.7640;

% End basic hparams.
% .........................................................................

% Create global hyperparams
N = 1000;
seed = 777;
dimM = 10;
dimN = 50;
global_tol = 1e-3;
time_limit = 4000;

% -------------------------------------------------------------------------
%% Table 1
% -------------------------------------------------------------------------
disp('========')
disp('TABLE 1');
disp('========')
% Loop over the upper curvature M.
M_vec = [1e2, 1e3, 1e4, 1e5, 1e6];
for i = 1:length(M_vec)
  % Use a problem instance generator to create the oracle and
  % hyperparameters.
  M = M_vec(i);
  m = 1e1;
  [oracle, hparams] = ...
    test_fn_lin_cone_constr_01(N, M, m, seed, dimM, dimN);

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
  
  % Create the complicated iALM hparams.
  ialm_hparam = base_hparam;
  ialm_hparam.rho0 = hparams.m;
  ialm_hparam.L0 = max([hparams.m, hparams.M]);
  ialm_hparam.rho_vec = hparams.m_constr_vec;
  ialm_hparam.L_vec = hparams.L_constr_vec;
  % Note that we are using the fact that |X|_F <= 1 over the spectraplex.
  ialm_hparam.B_vec = hparams.K_constr_vec;

  % Run a benchmark test and print the summary.
  hparam_arr = ...
    {aidal0_hparam, aidal1_hparam, aidal2_hparam, ...
     ialm_hparam, iapial_hparam, qp_aipp_hparam, rqp_aipp_hparam};
  name_arr = {...
    'ADL0', 'ADL1', 'ADL2', 'iALM', 'IPL', 'QP', 'RQP'};
  framework_arr = {...
    @AIDAL, @AIDAL, @AIDAL, @iALM, @IAPIAL, @penalty, @penalty};
  solver_arr = {...
    @ECG, @ECG, @ECG, @ECG, @ECG, @AIPP, @AIPP};
  [summary_tables, comp_models] = ...
    run_CCM_benchmark(...
      ncvx_lc_qp, framework_arr, solver_arr, hparam_arr, name_arr);
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
