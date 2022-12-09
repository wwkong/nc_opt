% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [model, history] = AIDAL(~, oracle, params)
% An accelerated inexact dampened augmeneted Lagrangian (AIDAL) framework for solving a nonconvex composite optimization problem
% with linear constraints
%
% Note:
%
%   Based on the paper:
%
%   **TBD**
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
%

  % Timer start.
  t_start = tic;
  
  % Initialize constant params.
  m = params.m;
  L = params.L;
  % We will assume this is B_g = B_g^{(0)}
  B_constr = params.B_constr;
  % We will assume this is L_g
  L_constr = params.L_constr;
  % We will assume this is K_g = B_g^{(1)}
  K_constr = params.K_constr;
  opt_tol = params.opt_tol;
  feas_tol = params.feas_tol;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  
  % Initialize special constraint functions
  prod_fn = params.prod_fn;
  norm_fn = params.norm_fn;
  pc_fn = params.set_projector;
  dc_fn = params.dual_cone_projector;
  
  % Initialize the augmented Lagrangian penalty (ALP) functions.
  alp_n = @(x) 0;
  alp_prox = @(x, lam) x;
  function alp_val = alp_fn(x, p, c, theta)
    p_step = (1 - theta) * p + c * params.constr_fn(x);
    dist_val = norm_fn((-p_step) - pc_fn(-p_step));
    alp_val = 1 / (2 * c) * (dist_val ^ 2 - (1 - theta) ^ 2 * norm_fn(p) ^ 2);
  end
  % Computes the point
  %   Proj_{K^*}((1 - theta) * p + c * const_fn(x)).
  function proj_point = dual_update(x, p, c, theta, chi)
    proj_point = dc_fn((1 - theta) * p + chi * c * (params.constr_fn(x)));
  end
  % Gradient of the function Q(x; p) with respect to x.
  function grad_alp_val = grad_alp_fn(x, p, c, theta)
    p_step = dc_fn((1 - theta) * p + c * (params.constr_fn(x)));
    grad_constr_fn = params.grad_constr_fn;
    %  If the gradient function has a single argument, assume that the
    %  gradient at a point is a constant tensor.
    if nargin(grad_constr_fn) == 1
      grad_alp_val = tsr_mult(grad_constr_fn(x), p_step, 'dual');
    % Else, assume that the gradient is a bifunction; the first argument is
    % the point of evaluation, and the second one is what the gradient
    % operator acts on.
    elseif nargin(grad_constr_fn) == 2
      grad_alp_val = grad_constr_fn(x, p_step);
    else
      error('Unknown function prototype for the gradient of the constraint function');
    end
  end

  % Fill in OPTIONAL input params.
  params = set_default_params(params);
  
  % Initialize other params.
  lambda = params.lambda;
  sigma = params.sigma;
  theta = params.theta;
  chi = params.chi;

  % Initialize history parameters.
  if params.i_logging
    history.function_values = [];
    history.iteration_values = [];
    history.time_values = [];
  end

  % Initialize framework parameters.
  Psi0 = Inf;
  iter = 0;
  outer_iter = 1;
  cycle_outer_iter = 1;
  sum_wc = 0;
  stage = 1;
  z0 = params.x0;
  p0 = zeros(size(params.constr_fn(z0)));
  c = params.c0;
  c_prev = c;
  params_acg = params;
  params_acg.mu = 1/2;
  params_acg.theta = Inf;
  params_acg.sigma = sigma;
  params_acg.termination_type = 'apd';
  params_acg.eta_type = 'accumulative';
  
  % Initial ACG Lipshchitz estimate.
  params_acg.L = lambda * (L + L_constr * norm_fn(p0) + c * (B_constr * L_constr + K_constr ^ 2)) + 1;
  if (strcmp(params.steptype, 'constant'))
    params_acg.L_est = lambda * (L + L_constr * norm_fn(p0) + c * (B_constr * L_constr + K_constr ^ 2)) + 1;
  elseif (strcmp(params.steptype, 'variable'))
    params_acg.L_est = lambda * (L + c * (B_constr * L_constr + K_constr ^ 2)) + 1;
  end
  % -----------------------------------------------------------------------
  %% MAIN ALGORITHM
  % -----------------------------------------------------------------------
  while true
    
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Create the penalty and ALM oracle objects.
    alp0_s = @(z) alp_fn(z, p0, c, theta);
    grad_alp0_s = @(z) grad_alp_fn(z, p0, c, theta);
    alp0_oracle = Oracle(alp0_s, alp_n, grad_alp0_s, alp_prox);
    oracle_AL0 = copy(oracle);
    oracle_AL0.add_smooth_oracle(alp0_oracle);

    % Create the ACG oracle.
    oracle_acg = copy(oracle_AL0);
    oracle_acg.proxify(lambda, z0);
    
    % Create the ACG params.
    params_acg.x0 = z0;
    params_acg.z0 = z0;
    params_acg.sigma = sigma;
    params_acg.t_start = t_start;
    params_acg.L = lambda * (L + L_constr * norm_fn(p0) + c * (B_constr * L_constr + K_constr ^ 2)) + 1;
    if (strcmp(params.steptype, 'constant'))
      params_acg.L_est = lambda * (L + L_constr * norm_fn(p0) + c * (B_constr * L_constr + K_constr ^ 2)) + 1;
    end
    
    % Call the ACG algorithm and update parameters.
    [model_acg, history_acg] = ACG(oracle_acg, params_acg);
    iter = iter + history_acg.iter;
    params_acg.L_est = model_acg.L_est;
    
    % Check for early failure.
    if (strcmp(params.steptype, 'variable') && model_acg.status < 0)
      lambda = lambda / params.adap_gamma;
      continue;
    end
    
    % Apply the refinement.
    z = model_acg.y;
    v_hat = (model_acg.u + z0 - z) / lambda;
    
    % Check for termination.
    p = dual_update(z, p0, c, theta, chi); % dual update
    q_hat = (1 / (chi * c)) * (p - (1 - theta) * p0);
    norm_v_hat = norm_fn(v_hat);
    norm_q_hat = norm_fn(q_hat);
    if (norm_v_hat <= opt_tol && norm_q_hat <= feas_tol)
      sum_wc = sum_wc + (outer_iter - cycle_outer_iter + 1) * c;
      break;
    end
    
    % Check for the variable telescoping bound and increase lambda if it does not hold.
    if (strcmp(params.steptype, 'variable'))
      if (theta == 0)
        alpha = 0;
        a_theta = 1;
      else
        a_theta = theta * (1-theta);
        b_theta = (2-theta) * (1-theta);
        alpha = ((1-2*chi*b_theta) - (1-theta)^2) / (2*chi); 
      end
      oracle.eval(z);
      Psi = oracle.f_s() + oracle.f_n() + alp_fn(z, p, c, theta) ...
             - a_theta * norm_fn(p)^2 / (2 * chi *c) + alpha * norm_fn(p-p0)^2 / (4 * chi *c);
      if (outer_iter == 1)
        Psi0 = Psi;
      else
        if (c == c_prev && norm_fn(model_acg.u + z0 - z)^2 > 32 * lambda * (Psi0 - Psi) + 1E-1)
          lambda = lambda / params.adap_gamma;
          continue;
        end
        Psi0 = Psi;
      end
    end    
    
    % Check if we need to double c. 
    c_prev = c;
    if ((norm_q_hat > feas_tol) && (norm_v_hat <= opt_tol))
      sum_wc = sum_wc + (outer_iter - cycle_outer_iter + 1) * c;
      cycle_outer_iter = outer_iter + 1;
      c = 2 * c;
      stage = stage + 1;      
    end
    
    % Update the other iterates.
    p0 = p;
    z0 = z;
    outer_iter = outer_iter + 1;
    
  end
  
  % Prepare to output
  model.x = z;
  model.y = p;
  model.v = v_hat;
  model.w = q_hat;
  history.iter = iter;
  history.stage = stage;
  history.c0 = params.c0;
  history.c = c;
  history.wavg_c = sum_wc / outer_iter;
  history.runtime = toc(t_start);
  
end

% Fills in parameters that were not set as input.
function params = set_default_params(params)

  % Global constants.
  MIN_PENALTY_CONST = 1;

  % Overwrite if necessary.
  if (~isfield(params, 'i_logging')) 
    params.i_logging = false;
  end
  if (~isfield(params, 'lambda'))
    params.lambda = 1 / (2 * params.m);
  end
  if (~isfield(params, 'theta'))
    params.theta = 0.01;
  end
  if (~isfield(params, 'c0'))
    params.c0 = max([MIN_PENALTY_CONST, params.L / params.K_constr ^ 2]);
  end
  if (~isfield(params, 'chi'))
    theta = params.theta;
    params.chi = theta ^ 2 / (2 * (2 - theta) * (1 - theta));
  end
  if (~isfield(params, 'chi_type'))
    params.chi_type = 'constant';
  end
  if (~isfield(params, 'acg_steptype'))
    params.acg_steptype = "variable";
  end  
  if (~isfield(params, 'incr_cond'))
    params.incr_cond = "default";
  end
  if (~isfield(params, 'steptype'))
    params.steptype = "constant";
  end
  if (~isfield(params, 'adap_gamma'))
    params.adap_gamma = 2.0;
  end

end
