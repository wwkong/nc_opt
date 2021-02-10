% Solve a multivariate nonconvex quadratic programming problem 
% constrained to the unit simplex using MULTIPLE SOLVERS.

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
[oracle, hparams] = ...
  test_fn_quad_cone_constr_02(N, M, m, seed, dimM, dimN, density);

% Create the Model object and specify the limits (if any).
ncvx_qc_qp = ConstrCompModel(oracle);
ncvx_qc_qp.time_limit = 10000;

% Set the curvatures and the starting point x0.
ncvx_qc_qp.x0 = hparams.x0;
ncvx_qc_qp.M = hparams.M;
ncvx_qc_qp.m = hparams.m;
ncvx_qc_qp.K_constr = hparams.K_constr;
ncvx_qc_qp.L_constr = hparams.L_constr;

% Set the tolerances
ncvx_qc_qp.opt_tol = 1e-3;
ncvx_qc_qp.feas_tol = 1e-3;

% Add quadratic constraints.
ncvx_qc_qp.constr_fn = hparams.constr_fn;
ncvx_qc_qp.grad_constr_fn = hparams.grad_constr_fn;
ncvx_qc_qp.set_projector = hparams.set_projector;
ncvx_qc_qp.dual_cone_projector = hparams.dual_cone_projector;

% Use a relative termination criterion.
ncvx_qc_qp.feas_type = 'relative';
ncvx_qc_qp.opt_type = 'relative';

% Create some basic hparams.
base_hparam = struct();

% Create the IAPIAL hparams.
iapial_hparam = base_hparam;

% Create the complicated iALM hparams.
ialm_hparam = base_hparam;
ialm_hparam.i_ineq_constr = true;
ialm_hparam.rho0 = hparams.m;
ialm_hparam.L0 = max([hparams.m, hparams.M]);
ialm_hparam.rho_vec = hparams.m_constr_vec;
ialm_hparam.L_vec = hparams.L_constr_vec;
% Note that we are using the fact that |X|_F <= 1 over the eigenbox.
ialm_hparam.B_vec = hparams.K_constr_vec;

% Run a benchmark test and print the summary.
hparam_arr = {ialm_hparam, iapial_hparam};
name_arr = {'iALM', 'IAPIAL'};
framework_arr = {@iALM, @IAPIAL};
solver_arr = {@ECG, @ECG};

% Run the test.
% profile on;
[summary_tables, comp_models] = ...
  run_CCM_benchmark(...
    ncvx_qc_qp, framework_arr, solver_arr, hparam_arr, name_arr);
disp(summary_tables.all);
% profile viewer;
% profile off;

%% Print the history for additional diagnostics.
fprintf('IAPIAL History: \n\n')
disp(comp_models.IAPIAL.history);
fprintf('iALM History: \n\n')
disp(comp_models.iALM.history);