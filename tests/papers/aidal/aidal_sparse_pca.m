% Solve a sparse PCA problem using MULTIPLE SOLVERS.

% Set up paths.
run('../../../init.m');

% -------------------------------------------------------------------------
%% Global Variables
% -------------------------------------------------------------------------

% Create basic hparams.
base_hparam = struct();

ialm_hparam = base_hparam;
ialm_hparam.i_ineq_constr = false;

aipp_hparam = base_hparam;
aipp_hparam.aipp_type = 'aipp';
aipp_hparam.acg_steptype = 'constant';

iapial_hparam = base_hparam;
iapial_hparam.acg_steptype = 'constant';
iapial_hparam.sigma_type = 'constant';
iapial_hparam.sigma_min = 0.3;
iapial_hparam.penalty_multiplier = 2;
iapial_hparam.i_reset_multiplier = false;
iapial_hparam.i_reset_prox_center = false;

aidal_hparam = base_hparam;
aidal_hparam.acg_steptype = 'constant';
aidal_hparam.sigma_type = 'constant';
aidal_hparam.sigma_min = 0.3;
aidal0_hparam = aidal_hparam;
aidal0_hparam.theta = 0;
aidal0_hparam.chi = 1;
aidal1_hparam = aidal_hparam;
aidal1_hparam.theta = 0.5;
aidal2_hparam = aidal_hparam;
aidal2_hparam.theta = 0.7640;

% Create global hyperparams
b = 0.1;
nu = 100;
p = 100;
n = 100;
k = 1;
seed = 777;
global_tol = 1e-3;
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
  % Use a problem instance generator to create the oracle and
  % hyperparameters.
  s = s_vec(i);
  [oracle, hparams] = ...
    test_fn_spca_01(b, nu, p, n, s, k, seed);
  
  % Problem dependent hparams
  ialm_hparam.rho0 = hparams.m;
  ialm_hparam.L0 = max([hparams.m, hparams.M]);
  ialm_hparam.rho_vec = hparams.m_constr_vec;
  ialm_hparam.L_vec = hparams.L_constr_vec;
  % Note that we are using the fact that |X|_F <= 1 over the spectraplex.
  ialm_hparam.B_vec = hparams.K_constr_vec;

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
  spca.opt_tol = global_tol;
  spca.feas_tol = global_tol;
  spca.time_limit = time_limit;

  % Run a benchmark test and print the summary.
  hparam_arr = ...
    {ialm_hparam, aipp_hparam, iapial_hparam, ...
     aidal0_hparam, aidal1_hparam, aidal2_hparam};
  name_arr = {'iALM', 'QP_AIPP', 'IAPIAL', 'AIDAL0', 'AIDAL1', 'AIDAL2'};
  framework_arr = {@iALM, @penalty, @IAPIAL, @AIDAL, @AIDAL, @AIDAL};
  solver_arr = {@ECG, @AIPP, @ECG, @ECG, @ECG, @ECG};
  [summary_tables, comp_models] = ...
    run_CCM_benchmark(...
      spca, framework_arr, solver_arr, hparam_arr, name_arr);
  
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
