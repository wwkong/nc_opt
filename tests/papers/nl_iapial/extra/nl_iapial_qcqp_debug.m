% Solve a multivariate nonconvex quadratically constrained quadratic programming  
% problem constrained to a box using MULTIPLE SOLVERS.
run('../../../../init.m');
format long

% Comment out later.
dimN = 250;
r = 1; 
m = 1; 
M = 1e4;

% Run an instance via the command line.
print_tbls(dimN, r, m, M);

%% Utility functions
function print_tbls(dimN, r, m, M) 

  % Initialize
  seed = 77777;
  dimM = 10;
  global_tol = 1e-5;
  disp(table(dimN, r, m, M));
  o_tbl = run_experiment(M, m, dimM, dimN, -r, r, seed, global_tol);
  disp(o_tbl);
  
end
function o_tbl = run_experiment(M, m, dimM, dimN, x_l, x_u, seed, global_tol)

  [oracle, hparams] = test_fn_quad_box_constr_02(M, m, seed, dimM, dimN, x_l, x_u);
  
  % Set up the termination function.
  function proj = proj_dh(a, b)
    I1 = (abs(a - x_l) < 1e-12); 
    I2 = (abs(a - x_u) <= 1e-12);
    I3 = (abs(a - x_l) > 1e-12 & abs(a - x_u) > 1e-12);
    proj = b;
    proj(I1) = min(0, b(I1));
    proj(I2) = max(0, b(I2));
    proj(I3) = 0;
  end
  function proj = proj_NKt(~, b)
    proj = min(0, b);
  end
  o_at_x0 = copy(oracle);
  o_at_x0.eval(hparams.x0);
  g0 = hparams.constr_fn(hparams.x0);
  rho = global_tol * (1 + hparams.norm_fn(o_at_x0.grad_f_s()));
  eta = global_tol * (1 + hparams.norm_fn(g0 - hparams.set_projector(g0)));
  term_wrap = @(x,p) ...
    termination_check(x, p, o_at_x0, hparams.constr_fn, hparams.grad_constr_fn, @proj_dh, @proj_NKt, hparams.norm_fn, rho, eta);

  % Create the Model object and specify the solver.
  ncvx_qc_qp = ConstrCompModel(oracle);
  
  % Set the curvatures and the starting point x0.
  ncvx_qc_qp.x0 = hparams.x0;
  ncvx_qc_qp.M = hparams.M;
  ncvx_qc_qp.m = hparams.m;
  ncvx_qc_qp.K_constr = hparams.K_constr;
  ncvx_qc_qp.L_constr = hparams.L_constr;
  
  % Set the tolerances
  ncvx_qc_qp.opt_tol = global_tol;
  ncvx_qc_qp.feas_tol = global_tol;
  ncvx_qc_qp.time_limit = 18000;
  ncvx_qc_qp.iter_limit = 1000000;
  
  % Add linear constraints
  ncvx_qc_qp.constr_fn = hparams.constr_fn;
  ncvx_qc_qp.grad_constr_fn = hparams.grad_constr_fn;
  ncvx_qc_qp.set_projector = hparams.set_projector;
  ncvx_qc_qp.dual_cone_projector = hparams.dual_cone_projector;
  
  % Use a relative termination criterion.
  ncvx_qc_qp.feas_type = 'relative';
  ncvx_qc_qp.opt_type = 'relative';
  
  % Create some basic hparams.
  base_hparam = struct();
  base_hparam.i_debug = true;
  base_hparam.termination_fn = term_wrap;
  base_hparam.check_all_terminations = true;
  
  % Create the IAPIAL hparams.
  ipl_hparam = base_hparam;
  ipl_hparam.acg_steptype = 'constant';
  ipl_hparam.sigma_min = 1/sqrt(2);
  ipla_hparam = base_hparam;
  ipla_hparam.acg_steptype = 'variable';
  ipla_hparam.init_mult_L = 0.5;
  ipla_hparam.sigma_min = 1/sqrt(2);
  
  % Run a benchmark test and print the summary.
  hparam_arr = {ipla_hparam, ipl_hparam};
  name_arr = {'IPL_A', 'IPL'};
  framework_arr = {@IAIPAL, @IAIPAL};
  solver_arr = {@ECG, @ECG};
  
  % Run the test.
  [summary_tables, ~] = run_CCM_benchmark(ncvx_qc_qp, framework_arr, solver_arr, hparam_arr, name_arr);
  o_tbl = summary_tables.all;

end

