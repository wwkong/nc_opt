%% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

% Solve a multivariate nonconvex quadratically constrained quadratic programming problem constrained to a box.
run('../../../init.m');
format long

% Run an instance via the command line.
print_tbls(n);

%% Utility functions
function print_tbls(dimN) 

  % Initialize
  seed = 77777;
  dimM = 10;
  global_tol = 1e-5;
  m_vec = [1e2, 1e3, 1e4];
  M_vec = [1e4, 1e5, 1e6];
  r_vec = [5, 10, 15];
  first_tbl = true;

  % Variable M.
  m = 1e0;
  r = 1;
  for M=M_vec
    tbl_row = run_experiment(M, m, dimM, dimN, -r, r, seed, global_tol);
    if first_tbl
      o_tbl = tbl_row;
      first_tbl = false;
    else
      o_tbl = [o_tbl; tbl_row];
    end
  end
  
  % Variable m.
  M = 1e6;
  r = 1;
  for m=m_vec
    tbl_row = run_experiment(M, m, dimM, dimN, -r, r, seed, global_tol);
    if first_tbl
      o_tbl = tbl_row;
      first_tbl = false;
    else
      o_tbl = [o_tbl; tbl_row];
    end
  end
  
  % Variable r.
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
  term_wrap = @(x,p) termination_check(x, p, o_at_x0, hparams.constr_fn, hparams.grad_constr_fn, @proj_dh, @proj_NKt, ...
                                       hparams.norm_fn, rho, eta);

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
  ncvx_qc_qp.time_limit = 3000;
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
  base_hparam.termination_fn = term_wrap;
  base_hparam.check_all_terminations = true;
  
  % Create the IAPIAL hparams.
  ipl_hparam = base_hparam;
  ipl_hparam.acg_steptype = 'constant';
  ipl_hparam.sigma_min = sqrt(0.3);
  ipla_hparam = base_hparam;
  ipla_hparam.acg_steptype = 'variable';
  ipla_hparam.sigma_min = sqrt(0.3);
  
  % Run a benchmark test and print the summary.
  hparam_arr = {ipl_hparam, ipla_hparam};
  name_arr = {'IPL', 'IPL_A'};
  framework_arr = {@IAIPAL, @IAIPAL};
  solver_arr = {@ECG, @ECG};
  
  % Run the test.
  [summary_tables, ~] = run_CCM_benchmark(ncvx_qc_qp, framework_arr, solver_arr, hparam_arr, name_arr);
  
  % Set up HiAPeM options.
  o_at_x0 = copy(oracle);
  o_at_x0.eval(hparams.x0);
  feas_at_x0 = feasibility(hparams.x0);
  rel_tol = min([global_tol * (1 + hparams.norm_fn(o_at_x0.grad_f_s())), global_tol * (1 + feas_at_x0)]);
  opts = struct();
  opts.x0 = hparams.x0;
  opts.Lip0 = max([hparams.m, hparams.M]);
  opts.tol = rel_tol;
  opts.maxit = ncvx_qc_qp.iter_limit;
  opts.inc = 2;
  opts.dec= 2.5;
  opts.sig = 3;
  opts.maxsubit = ncvx_qc_qp.iter_limit;
  opts.maxniter = ncvx_qc_qp.iter_limit;
  opts.rho = hparams.m; % Lower curvature.
  x_l_vec = ones(dimN, 1) * hparams.x_l;
  x_u_vec = ones(dimN, 1) * hparams.x_u;
  
  % EXTRA opts (for using the relaxed termination).
  opts.termination_fn = term_wrap;
  
  % Run the HiAPeM code.
  tic;
  [x_hpm, ~, out_hpm] = HiAPeM_qcqp(hparams.Q, hparams.c, hparams.d, dimM, x_l_vec, x_u_vec, opts);
  t_hpm = toc;
  o_at_x_hpm = copy(oracle);
  o_at_x_hpm.eval(x_hpm);
  fval_hpm = o_at_x_hpm.f_s();
  iter_hpm = out_hpm.niter;
  disp(['Total number of iterations is ', num2str(iter_hpm)])
  disp(['Function value is ', num2str(fval_hpm)])
  disp(['Feasibility is ', num2str(feasibility(x_hpm))]);
  
  % Aggregate
  o_tbl = agg_tbl(summary_tables, fval_hpm, iter_hpm, t_hpm);
  disp(o_tbl);
  
  % Compute feasibility
  function feas = feasibility(x)
    m_cx = -hparams.constr_fn(x);
    m_cxp = hparams.set_projector(m_cx);
    feas = norm(m_cxp - m_cx);
  end

  function o_tbl = agg_tbl(summary_tbls, f_HiAPeM, iter_HiAPeM, t_HiAPeM)
    o_tbl = [table(dimN, dimM, x_l, x_u), summary_tbls.pdata, summary_tbls.fval, table(f_HiAPeM), summary_tbls.iter, ...
             table(iter_HiAPeM), summary_tbls.runtime, table(t_HiAPeM), summary_tbls.mdata];
  end

end

