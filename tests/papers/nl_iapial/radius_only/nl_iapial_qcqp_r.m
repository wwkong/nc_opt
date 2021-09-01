% Solve a multivariate nonconvex quadratically constrained quadratic programming  
% problem constrained to a box using MULTIPLE SOLVERS.

% Set up paths.
run('../../../../init.m');

% Use a problem instance generator to create the oracle and
% hyperparameters.
seed = 777;
dimM = 10;
global_tol = 1e-5;
r_vec = [2.5, 5, 10, 20];

% ==============================================================================
%% Table for dimN = 200
% ==============================================================================
first_tbl = true;

dimN = 200;
m = 1e0;
M = 1e6;
for r=r_vec
  tbl_row = run_experiment(M, m, dimM, dimN, -r, r, seed, global_tol);
  if first_tbl
    o_tbl = tbl_row;
    first_tbl = false;
  else
    o_tbl = [o_tbl; tbl_row];
  end
end
disp(['Tables for dimN = ', num2str(dimN)]);
disp(o_tbl);

% ==============================================================================
%% Table for dimN = 500
% ==============================================================================
first_tbl = true;

dimN = 500;
m = 1e0;
M = 1e6;
for r=r_vec
  tbl_row = run_experiment(M, m, dimM, dimN, -r, r, seed, global_tol);
  if first_tbl
    o_tbl = tbl_row;
    first_tbl = false;
  else
    o_tbl = [o_tbl; tbl_row];
  end
end
disp(['Tables for dimN = ', num2str(dimN)]);
disp(o_tbl);

% ==============================================================================
%% Table for dimN = 1000
% ==============================================================================
first_tbl = true;

dimN = 1000;
m = 1e0;
M = 1e6;
for r=r_vec
  tbl_row = run_experiment(M, m, dimM, dimN, -r, r, seed, global_tol);
  if first_tbl
    o_tbl = tbl_row;
    first_tbl = false;
  else
    o_tbl = [o_tbl; tbl_row];
  end
end
disp(['Tables for dimN = ', num2str(dimN)]);
disp(o_tbl);

%% Utility functions
function o_tbl = run_experiment(M, m, dimM, dimN, x_l, x_u, seed, global_tol)

  [oracle, hparams] = ...
    test_fn_quad_box_constr_02(M, m, seed, dimM, dimN, x_l, x_u);

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
  ncvx_qc_qp.time_limit = 4000;
  
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
  
  % Create the IAPIAL hparams.
  ipl_hparam = base_hparam;
  ipl_hparam.acg_steptype = 'constant';
  ipla_hparam = base_hparam;
  ipla_hparam.acg_steptype = 'variable';
  
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
  hparam_arr = {ialm_hparam, ipl_hparam, ipla_hparam};
  name_arr = {'iALM', 'IPL', 'IPL_A'};
  framework_arr = {@iALM, @IAIPAL, @IAIPAL};
  solver_arr = {@ECG, @ECG, @ECG};

  % Run the test.
  [summary_tables, ~] = ...
    run_CCM_benchmark(...
      ncvx_qc_qp, framework_arr, solver_arr, hparam_arr, name_arr);
  
  % Set up HiAPeM options.
  o_at_x0 = copy(oracle);
  o_at_x0.eval(hparams.x0);
  feas_at_x0 = feasibility(hparams.x0);
  rel_tol = min([...
    global_tol * (1 + hparams.norm_fn(o_at_x0.grad_f_s())), ...
    global_tol * (1 + feas_at_x0)]);
  opts = struct();
  opts.x0 = hparams.x0;
  opts.Lip0 = max([hparams.m, hparams.M]);
  opts.tol = rel_tol;
  opts.maxit = 1000000;
  opts.inc = 2;
  opts.dec= 2.5;
  opts.sig = 3;
  opts.maxsubit = 1000000;
  opts.rho = hparams.m; % Lower curvature.
  x_l_vec = ones(dimN, 1) * hparams.x_l;
  x_u_vec = ones(dimN, 1) * hparams.x_u;
  
  % Run the HiAPeM code.
  tic;
  [x_hpm, ~, out_hpm] = HiAPeM_qcqp(...
    hparams.Q, hparams.c, hparams.d, dimM, x_l_vec, x_u_vec, opts);
  t_hpm = toc;
  o_at_x_hpm = copy(oracle);
  o_at_x_hpm.eval(x_hpm);
  fval_hpm = o_at_x_hpm.f_s();
  iter_hpm = out_hpm.niter;
  disp(['Total number of iterations is ', num2str(iter_hpm)])
  disp(['Function value is ', num2str(fval_hpm)])
  disp(['Feasibility is ', num2str(feasibility(x_hpm))]);
  
  %Aggregate
  o_tbl = agg_tbl(summary_tables, fval_hpm, iter_hpm, t_hpm);
  disp(o_tbl);
  
  % Compute feasibility
  function feas = feasibility(x)
    m_cx = -hparams.constr_fn(x);
    m_cxp = hparams.set_projector(m_cx);
    feas = norm(m_cxp - m_cx);
  end

  function o_tbl = agg_tbl(summary_tbls, f_HiAPeM, iter_HiAPeM, t_HiAPeM)
    o_tbl = [...
      table(x_l, x_u), summary_tbls.pdata, summary_tbls.fval, ...
      table(f_HiAPeM), summary_tbls.iter, table(iter_HiAPeM), ...
      summary_tbls.runtime, table(t_HiAPeM), summary_tbls.mdata];
  end

end

