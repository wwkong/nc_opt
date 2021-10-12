% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [model, history] = iALM(~, oracle, params)
% An inexact augmented Lagrangian method (iALM) for solving a nonconvex composite optimization model with nonlinear equality
% constraints, i.e. $g(x) = 0$.
% 
% Note:
% 
%   Based on the paper:
%
%   Li, Z., Chen, P. Y., Liu, S., Lu, S., & Xu, Y. (2020). Rate-improved inexact augmented Lagrangian method for constrained
%   nonconvex optimization. *arXiv preprint arXiv:2007.01284*\.
%
% Arguments:
% 
%   oracle (Oracle): The oracle underlying the optimization problem.
% 
%   params (struct): Contains instructions on how to call the framework.
% 
% Returns:
%   
%   A pair of structs containing model and history related outputs of the solved problem associated with the oracle and input
%   parameters.

  % Timer start.
  t_start = tic;
  
  % Initialize constant params.
  o_params = params;
  o_oracle = copy(oracle);
  opt_tol = params.opt_tol;
  feas_tol = params.feas_tol;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  
  % Set the topology.
  norm_fn = params.norm_fn;
  prod_fn = params.prod_fn;
  
  % Fill in OPTIONAL input params (custom implementation).
  params = set_default_params(params);
  
  % Check if we need to introduce slack and modify inputs as necessary.
  if (params.i_ineq_constr)
    oracle = copy(o_oracle);
    [oracle, params] = add_slack(oracle, params);
  end
  
  % Initialize the augmented Lagrangian penalty (ALP) functions.
  alp_n = @(x) 0;
  alp_prox = @(x, lam) x;
  % Value of the AL function:
  %   Q_beta(x, y) := <y, c(x)> + (beta / 2) * |c(x)| ^ 2,
  function alp_val = alp_fn(x, y, beta)
    alp_val = prod_fn(y, params.constr_fn(x)) + (beta / 2) * norm_fn(params.constr_fn(x)) ^ 2;
  end
  % Gradient of the function Q_beta(x, y) with respect to x.
  function grad_alp_val = grad_alp_fn(x, y, beta)
    prox_ctr = y + beta * params.constr_fn(x);
    grad_constr_fn = params.grad_constr_fn;
    %  If the gradient function has a single argument, assume that the gradient at a point is a constant tensor.
    if nargin(grad_constr_fn) == 1
      grad_alp_val = tsr_mult(grad_constr_fn(x), prox_ctr, 'dual');
    % Else, assume that the gradient is a bifunction; the first argument is the point of evaluation, and the second one is what
    % the gradient operator acts on.
    elseif nargin(grad_constr_fn) == 2
      grad_alp_val = grad_constr_fn(x, prox_ctr);
    else
      error('Unknown function prototype for the gradient of the constraint function');
    end
  end
  
  % Initialize basic algorithm params.
  sigma = params.sigma;
  beta0 = params.beta0;
  x0 = params.x0;
  w0 = params.w0;
  y0 = zeros(size(params.constr_fn(x0)));
  L0 = params.L0;
  rho0 = params.rho0;
  first_beta0 = beta0;
  
  % Initialize iPPM params.
  ippm_params = params;
  ippm_params.tol = opt_tol;
  
  % Set up rho_c, L_bar, and L_c.
  L_bar = sqrt(sum(params.L_vec .^ 2));
  rho_c = sum(params.B_vec .* params.rho_vec);
  L_c = sum(params.B_vec .* params.L_vec + params.B_vec .^ 2);
  
  % Iterate.
  iter = 0; % Set to 0 because it calls an inner subroutine.
  outer_iter = 0;  
  stage = 1;
  while true
    
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit && outer_iter > 0)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Create the penalty and ALM oracle objects.
    alp0_s = @(z) alp_fn(z, y0, beta0);
    grad_alp0_s = @(z) grad_alp_fn(z, y0, beta0);
    alp0_oracle = Oracle(alp0_s, alp_n, grad_alp0_s, alp_prox);
    ippm_oracle = copy(oracle);
    ippm_oracle.add_smooth_oracle(alp0_oracle);
    
    % Create the curvatures.
    rho_hat = rho0 + L_bar * norm_fn(y0) + beta0 * rho_c;
    L_hat = L0 + L_bar * norm_fn(y0) + beta0 * L_c;
    if (iter == 0)
      first_L_hat = L_hat;
    end
    
    % Set up the other iPPM params.
    ippm_params.rho = rho_hat;
    ippm_params.L_phi = L_hat;
    ippm_params.x0 = x0;
    ippm_params.t_start = t_start;
    
    % Call the iPPM and parse the outputs.
    [ippm_model, ippm_history] = iPPM(ippm_oracle, ippm_params);
    x = ippm_model.x;
    v = ippm_model.v;
    iter = iter + ippm_history.iter;
    outer_iter = outer_iter + ippm_history.outer_iter;
    
    % Update the multiplier.
    q = params.constr_fn(x);
    y = y0 + w0 * q;
    
    % Update w.
    if (stage == 1)
      c_at_x1 = q;
    end
    w = w0 * min([1, params.gamma_fn(stage - 1, c_at_x1) / norm_fn(q)]);
    
    % Check for termination.
    if (isempty(params.termination_fn) || params.check_all_terminations)
      if (norm_fn(q) <= feas_tol)
        break;
      end
    end
    if (~isempty(params.termination_fn) || params.check_all_terminations)
      dims_x = size(x);
      if dims_x(2) == 1 && params.i_ineq_constr
        orig_dim = length(x) - length(y);
        x_hat = x(1:orig_dim);
      elseif dims_x(2) > 1
        orig_dim = min(size(x));
        x_hat = x(1:orig_dim, 1:orig_dim);
      else
        x_hat = x;
      end
      [tpred, v, q] = params.termination_fn(x_hat, y);
      if tpred
        break;
      end
    end
    
    % Update iterates
    beta0 = beta0 * sigma;
    x0 = x;
    y0 = y;
    w0 = w;
    stage = stage + 1;
    
  end
  
  % Get ready to output
  model.x = x;
  model.v = v;
  model.w = q;
  history.acg_ratio = iter / outer_iter;
  history.iter = iter;
  history.outer_iter = outer_iter;
  history.stage = stage;
  history.first_L_hat = first_L_hat;
  history.last_L_hat = L_hat;
  history.first_c0 = first_beta0;
  history.last_c0 = beta0;
  history.runtime = toc(t_start);

  % Add special logic for inequality constraints
  if (params.i_ineq_constr)
    % Create a helper string for indexing the eventual tensor representation.
    x_sz = size(o_params.x0);
    x_ind = cell(2, 1);
    x_ind{1} = 1:x_sz(1);
    x_ind{2} = 1:x_sz(2);
    % Remove the slack
    model.x = x(x_ind{:});
    model.v = v(x_ind{:});
  end
  
end

function [model, history] = iPPM(oracle, params)
% Inexact Proximal Point Method (iPPM) used in iALM.

  % Parse algorithm inputs.
  rho = params.rho;
  L_phi = params.L_phi;
  tol = params.tol;
  
  % Solver params.
  norm_fn = params.norm_fn;
  t_start = params.t_start;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  
  % Initialize the invariant APG params.
  apg_params = params;
  apg_params.mu = rho;
  apg_params.L_g = L_phi + 2 * rho;
  apg_params.tol = tol / 4;
  quad_f_n = @(x) 0;
  quad_prox_f_n = @(x, lam) x;
  
  % Initialize other iPPM parameters.
  x0 = params.x0;
  
  % Iterate.
  iter = 0; % Set to 0 because it calls an inner subroutine.
  outer_iter = 1;
  while true
    
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Create the APG oracle.
    apg_oracle = copy(oracle);
    quad_f_s = @(x) rho * norm_fn(x - x0) ^ 2;
    grad_quad_f_s = @(x) 2 * rho * (x - x0);
    quad_oracle = Oracle(quad_f_s, quad_f_n, grad_quad_f_s, quad_prox_f_n);
    apg_oracle.add_smooth_oracle(quad_oracle);
    
    % Set the other APG params (custom implementation here!)
    apg_params.t_start = t_start;
    apg_params.xm1_bar = x0;
    
    % Call the APG and parse the outputs.
    [apg_model, apg_history] = APG(apg_oracle, apg_params);
    x = apg_model.x;
    v = apg_model.v + 2 * rho * (x0 - x);
    iter = iter + apg_history.iter;
    
    % Check for termination.
    if (2 * rho * norm_fn(x - x0) <= tol / 2)
      break;
    end
    
    % Update iterates
    x0 = x;
    outer_iter = outer_iter + 1;
    
  end
  
  % Get ready to output
  model.v = v;
  model.x = x;
  history.iter = iter;
  history.outer_iter = outer_iter;
  
end

function [model, history] = APG(oracle, params)
% Accelerated Proximal Gradient (APG) Method used in iPPM.

  % Parse algorithm inputs.
  mu = params.mu;
  L_g = params.L_g;
  tol = params.tol;
  
  % Solver params.
  norm_fn = params.norm_fn;
  t_start = params.t_start;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;

  % Set up some oracles.
  o_at_x0_bar = copy(oracle);
  o_at_x0 = copy(oracle);
  o_at_x = copy(oracle);

  % Compute alpha, x0, x_bar_0.
  alpha = sqrt(mu / L_g);
  xm1_bar = params.xm1_bar;
  o_at_x0_bar.eval(xm1_bar);
  o_at_x0.eval(xm1_bar - o_at_x0_bar.grad_f_s() / L_g); 
  x0 = o_at_x0.prox_f_n(1 / L_g);
  x0_bar = x0;
  x = x0;
  
  % Iterate.
  iter = 2; % Each call does at least two prox evaluations.
  while true
    
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Compute x and x_bar.
    o_at_x0_bar.eval(x0_bar);
    o_at_x0.eval(x0_bar - o_at_x0_bar.grad_f_s() / L_g);
    x = o_at_x0.prox_f_n(1 / L_g);
    o_at_x.eval(x);
    x_bar = x + (1 - alpha) / (1 + alpha) * (x - x0);

    % Check for termination.
    if isempty(params.proj_dh)
      % Sufficient APG termination.
      if (2 * L_g * norm_fn(x - x0_bar) <= tol)
        break;
      end
    else
      % Canonical APG termination.
      grad_G_at_x = o_at_x.grad_f_s();
      dims_x = size(x);
      if dims_x(2) == 1  
        proj_mG = params.proj_dh(x, -grad_G_at_x);
      else
        orig_length = min(size(x));
        x_orig = x(1:orig_length, 1:orig_length);
        grad_G_at_x_orig = grad_G_at_x(1:orig_length, 1:orig_length);
        proj_mG_orig = params.proj_dh(x_orig, -grad_G_at_x_orig);
        proj_mG = [proj_mG_orig; -grad_G_at_x(orig_length+1:end, 1:end)];
      end
      if (norm(-grad_G_at_x - proj_mG) <= tol)
        break;
      end
    end
    
    % Update iterates
    iter = iter + 1;
    x0 = x;
    x0_bar = x_bar;
    
  end
  
  % Get ready to output
  model.v = L_g * (x0_bar - x) + o_at_x.grad_f_s() - o_at_x0_bar.grad_f_s();
  model.x = x;
  history.iter = iter;
  
end

%% Helper functions

% Fills in parameters that were not set as input.
function params = set_default_params(params)

  % Overwrite if necessary.
  if (~isfield(params, 'sigma')) 
    params.sigma = 5;
  end
  if (~isfield(params, 'beta0')) 
%     params.beta0 = 0.01;
    params.beta0 = max([1, params.L / params.K_constr ^ 2]);
  end
  if (~isfield(params, 'w0')) 
    params.w0 = 1;
  end
  if (~isfield(params, 'gamma_fn')) 
    params.gamma_fn = @(k, c_at_x1) (log(2)) ^ 2 * params.norm_fn(c_at_x1) / (k + 1) * (log(k + 2)) ^ 2;
  end
  % Indicator for if the constraint is a conic inequality constraint.
  if (~isfield(params, 'i_ineq_constr')) 
    params.i_ineq_constr = false; 
  end
  if (~isfield(params, 'termination_fn'))
    params.termination_fn = [];
  end
  if (~isfield(params, 'proj_dh'))
    params.proj_dh = [];
  end
  if (~isfield(params, 'check_all_terminations'))
    params.check_all_terminations = false;
  end
end

function [oracle, params] = add_slack(oracle, params)
% Modify the problem with the constraint c(x) <= 0 to an equivalent one with the expanded constraints c(x) + s = 0, s >= 0.

  % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  % NOTE: Only 1D or 2D inputs are currently supported!
  % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  % Create a helper string for indexing the eventual tensor
  % representation.
  x_sz = size(params.x0);
  c_sz = size(params.constr_fn(params.x0));
  sz = [x_sz(1) + c_sz(1), max([x_sz(2), c_sz(2)])];
  x_ind = cell(2, 1);
  x_ind{1} = 1:x_sz(1);
  x_ind{2} = 1:x_sz(2);
  c_ind = cell(2, 1);
  c_ind{1} = (x_sz(1)+1):(x_sz(1)+c_sz(1));
  c_ind{2} = 1:c_sz(2);
  
  % Sanity checks.
  if (length(x_sz) > 2)
    error('Only scalar, 1D, and 2D functions are currently supported!');
  end
  
  % Modify the input point to be [x0; s].
  slack_x0 = zeros(sz);
  slack_x0(x_ind{:}) = params.x0;
  params.x0 = slack_x0;
  
  % Modify the default oracle so that the input uses an additional 
  % dimension, i.e., we use a (+1) dimensional tensor.
  [o_f_s, o_f_n, o_grad_f_s, o_prox_f_n] = oracle.decompose();
  f_s = @(xs) o_f_s(xs(x_ind{:}));
  f_n = @(xs) o_f_n(xs(x_ind{:}));
  function out_tsr = grad_f_s(xs)
    out_tsr = zeros(sz);
    out_tsr(x_ind{:}) = o_grad_f_s(xs(x_ind{:}));
  end
  % The prox will use the primal cone projector.
  function out_tsr = prox_f_n(xs, lam)
    out_tsr = zeros(sz);
    out_tsr(x_ind{:}) = o_prox_f_n(xs(x_ind{:}), lam);
    out_tsr(c_ind{:}) = params.set_projector(xs(c_ind{:}));
  end
  oracle = Oracle(f_s, f_n, @grad_f_s, @prox_f_n);
  
  % Modify the constraint function.
  o_constr_fn = params.constr_fn;
  o_grad_constr_fn = params.grad_constr_fn;
  slack_constr_fn = @(xs) o_constr_fn(xs(x_ind{:})) + xs(c_ind{:});
  function out_tsr = slack_grad_constr_fn1(xs, grad_fn)
    if (length(x_sz) > 1)
      error('Only vector functions are supported for gradients of this type!');
    end
    out_tsr = zeros(sz);
    out_tsr(x_ind{:}) = grad_fn(xs(x_ind{:}));
    out_tsr(c_ind{:}) = eye(x_sz, x_sz);
  end
  function out_tsr = slack_grad_constr_fn2(xs, Delta, grad_fn)
    out_tsr = zeros(sz);
    out_tsr(x_ind{:}) = grad_fn(xs(x_ind{:}), Delta);
    out_tsr(c_ind{:}) = Delta;
  end  
  %  If the gradient function has a single argument, assume that the
  %  gradient at a point is a constant tensor.
  if nargin(o_grad_constr_fn) == 1
    slack_grad_constr_fn = @(xs) slack_grad_constr_fn1(xs, o_grad_constr_fn);
  % Else, assume that the gradient is a bifunction; the first argument is
  % the point of evaluation, and the second one is what the gradient
  % operator acts on.
  elseif nargin(o_grad_constr_fn) == 2
    slack_grad_constr_fn = @(xs, DeltaS) slack_grad_constr_fn2(xs, DeltaS, o_grad_constr_fn);
  else
    error('Unknown function prototype for the gradient of the constraint function');
  end
  params.constr_fn = slack_constr_fn;
  params.grad_constr_fn = slack_grad_constr_fn;
  
  % Change the topology (assume the squared norm and inner product are
  % decomposable).
  o_norm_fn = params.norm_fn;
  o_prod_fn = params.prod_fn;
  params.norm_fn = @(xs) sqrt(o_norm_fn(xs(x_ind{:})) ^ 2 + norm(xs(c_ind{:}), 'fro') ^ 2);
  params.prod_fn = @(as, bs) o_prod_fn(as(x_ind{:}), bs(x_ind{:})) + sum(dot(as(c_ind{:}), bs(c_ind{:})));
  
end
