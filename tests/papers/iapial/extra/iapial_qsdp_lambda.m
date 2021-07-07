% Solve a multivariate nonconvex quadratic programming problem 
% constrained to the unit simplex using MULTIPLE SOLVERS.

% The function of interest is
%
%  f(x) :=  -xi / 2 * ||D * B * X|| ^ 2 + tau / 2 * ||A * X - b|| ^ 2
%
% with curvature pair (m, M). 

%% Initialization

% Set up paths.
run('../../../../init.m');

% Global parameters for each experiment.
globals.N = 1000;
globals.seed = 777;
globals.dimM = 30;
globals.dimN = 100;
globals.density = 0.05;
globals.opt_tol = 1e-4;
globals.feas_tol = 1e-4;

%% Run an experiment

m = 1e1;
M = 1e4;
mEst_vec = M ./ [1; 2; 4; 8; 16; 32; 64; 128; 256; 512; 1000];

i_first_row = true;
for i=1:length(mEst_vec)
  mEst = mEst_vec(i);
  tbl_row = run_experiment(m, M, mEst, globals);
  disp([table(mEst), tbl_row]);
  if i_first_row
    tbl = tbl_row;
    i_first_row = false;
  else
    tbl = [tbl; tbl_row]; 
  end
end
disp([table(mEst_vec), tbl]);

%% Helper Functions

% Run a single experiment and output the summary row.
function out_tbl = run_experiment(m, M, m_est, params) 

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
  ipl_hparam = base_hparam;
  ipl_hparam.acg_steptype = 'constant';
  ipl_hparam.lambda = 1 / (2 * m_est);
  
  % Run a benchmark test and print the summary.
  hparam_arr = {ipl_hparam};
  name_arr = {'IPL'};
  framework_arr = {@IAIPAL};
  solver_arr = {@ECG};
  
  % Run the test.
  [summary_tables, ~] = ...
    run_CCM_benchmark(...
    ncvx_lc_qp, framework_arr, solver_arr, hparam_arr, name_arr);
  out_tbl = summary_tables.all;
end