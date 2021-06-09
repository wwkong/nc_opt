% Solve a multivariate nonconvex quadratic programming problem 
% constrained to the unit simplex using MULTIPLE SOLVERS.

% The function of interest is
%
%  f(x) :=  -xi / 2 * ||D * B * X|| ^ 2 + tau / 2 * ||A * X - b|| ^ 2
%
% with curvature pair (m, M). 

%% Initialization

% Set up paths.
run('../../../init.m');

% Global parameters for each experiment.
globals.N = 1000;
globals.seed = 777;
globals.dimM = 20;
globals.dimN = 100;
globals.density = 0.01;
globals.opt_tol = 1e-3;
globals.feas_tol = 1e-3;

%% Run an experiment

% The main parameters (mM_mat) should be spec'd by Condor.

% E.g. Table 1.
% mM_mat = ...
%   [1e1, 1e2; ...
%    1e1, 1e3; ...
%    1e1, 1e4; ...
%    1e1, 1e5; ...
%    1e1, 1e6; ];
 
% E.g. Table 2.
% mM_mat = ...
%   [1e2, 1e2; ...
%    1e3, 1e3; ...
%    1e4, 1e4; ...
%    1e5, 1e5; ...
%    1e6, 1e6; ];
 
i_first_row = true;
for i=1:size(mM_mat, 1)
  tbl_row = run_experiment(mM_mat(i, 1), mM_mat(i, 2), globals);
  disp(tbl_row);
  if i_first_row
    tbl = tbl_row;
    i_first_row = false;
  else
    tbl = [tbl; tbl_row]; 
  end
end
disp(tbl);

%% Helper Functions

% Run a single experiment and output the summary row.
function out_tbl = run_experiment(m, M, params) 

  % Gather the oracle and hparam instance.
  [oracle, hparams] = ...
    test_fn_lin_cone_constr_02(...
      params.N, M, m, params.seed, params.dimM, params.dimN, params.density);

  % Create the Model object and specify the limits (if any).
  ncvx_lc_qp = ConstrCompModel(oracle);
  ncvx_lc_qp.time_limit = 4000;
  
  % Set the curvatures and the starting point x0.
  ncvx_lc_qp.x0 = hparams.x0;
  ncvx_lc_qp.M = hparams.M;
  ncvx_lc_qp.m = hparams.m;
  ncvx_lc_qp.K_constr = hparams.K_constr;
  
  % Set the tolerances
  ncvx_lc_qp.opt_tol = params.opt_tol;
  ncvx_lc_qp.feas_tol = params.feas_tol;
  
  % Add linear constraints.
  ncvx_lc_qp.constr_fn = @(x) hparams.constr_fn(x);
  ncvx_lc_qp.grad_constr_fn = hparams.grad_constr_fn;
  
  % Use a relative termination criterion.
  ncvx_lc_qp.feas_type = 'relative';
  ncvx_lc_qp.opt_type = 'relative';
  
  % Create some basic hparams.
  base_hparam = struct();
  iapial_hparam = base_hparam;
  iapial_hparam.acg_steptype = 'constant';
  raipp_hparam = base_hparam;
  raipp_hparam.acg_steptype = 'variable';
  aipp_hparam = base_hparam;
  aipp_hparam.acg_steptype = 'constant';
  aipp_hparam.aipp_type = 'aipp';
  
  % Create the complicated iALM hparams.
  ialm_hparam = base_hparam;
  ialm_hparam.rho0 = hparams.m;
  ialm_hparam.L0 = max([hparams.m, hparams.M]);
  ialm_hparam.rho_vec = hparams.m_constr_vec;
  ialm_hparam.L_vec = hparams.L_constr_vec;
  % Note that we are using the fact that |X|_F <= 1 over the spectraplex.
  ialm_hparam.B_vec = hparams.K_constr_vec;
  
  % Run a benchmark test and print the summary.
  hparam_arr = {ialm_hparam, aipp_hparam, raipp_hparam, iapial_hparam};
  name_arr = {'iALM', 'QP_AIPP', 'R_QP_AIPP', 'IAPIAL'};
  framework_arr = {@iALM, @penalty, @penalty, @IAPIAL};
  solver_arr = {@ECG, @AIPP, @AIPP, @ECG};
  
  % Run the test.
  [summary_tables, ~] = ...
    run_CCM_benchmark(...
    ncvx_lc_qp, framework_arr, solver_arr, hparam_arr, name_arr);
  out_tbl = summary_tables.all;
end