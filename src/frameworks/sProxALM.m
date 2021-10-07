% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [model, history] = sProxALM(~, oracle, params)
% A smoothed proximal augmented Lagrangian method (s-Prox-ALM) for solving a nonconvex composite optimization model with linear
% constraints, where the composite term is the indicator of a polyhedron.
% 
% Note:
% 
%   Based on the paper:
%
%   Zhang, J., & Luo, Z. (2020). A global dual error bound and its application to the analysis of linearly constrained nonconvex 
%   optimization. *arXiv preprint arXiv:2006.16440*\.
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
  opt_tol = params.opt_tol;
  feas_tol = params.feas_tol;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  
  % Fill in OPTIONAL input params (custom implementation).
  params = set_default_params(params);
  
  % Initialize history parameters.
  if params.i_logging
    logging_oracle = copy(oracle);
    history.function_values = [];
    history.norm_w_hat_values = [];
    history.norm_q_hat_values = [];
    history.iteration_values = [];
    history.time_values = [];
  end
  
  % Set the topology.
  norm_fn = params.norm_fn;
  prod_fn = params.prod_fn;
  
  % Gradient of the constraint function
  function grad_c_at_x_op_y = grad_constr_fn(x, y)
    grad_constr_fn = params.grad_constr_fn;
    %  If the gradient function has a single argument, assume that the gradient at a point is a constant tensor.
    if nargin(grad_constr_fn) == 1
      grad_c_at_x_op_y = tsr_mult(grad_constr_fn(x), y, 'dual');
    % Else, assume that the gradient is a bifunction; the first argument is the point of evaluation, and the second one is what
    % the gradient operator acts on.
    elseif nargin(grad_constr_fn) == 2
      grad_c_at_x_op_y = grad_constr_fn(x, y);
    else
      error('Unknown function prototype for the gradient of the constraint function');
    end
  end
  
  
  % Initialize the augmented Lagrangian penalty (ALP) functions.
  alp_n = @(x) 0;
  alp_prox = @(x, lam) x;
  % Value of the AL function: 
  %   Q_beta(x, y) = <y, c(x)> + (beta / 2) * |c(x)| ^ 2,
  % where K is the primal cone.
  function alp_val = alp_fn(x, y, beta)
    alp_val = prod_fn(y, params.constr_fn(x)) + (beta / 2) * norm_fn(params.constr_fn(x)) ^ 2;
  end
  % Gradient of the function Q_beta(x, y) with respect to x.
  function grad_alp_val = grad_alp_fn(x, y, beta)
    prox_ctr = y + beta * params.constr_fn(x);
    grad_alp_val = grad_constr_fn(x, prox_ctr);
  end
  
  % Initialize the algorithm's hyperparameters
  x0 = params.x0; 
  y0 = params.y0;
  z0 = params.z0;
  Gamma = params.Gamma;
  alpha = params.alpha;
  p = params.p;
  c = params.c;
  beta = params.beta;
  
  % Iterate.
  iter = 1;
  while true
    
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Take a dual step.
    y = y0 + alpha * params.constr_fn(x0);
    
    % Create the penalty and ALM oracle objects.
    alp0_s = @(x) alp_fn(x, y, Gamma);
    grad_alp0_s = @(x) grad_alp_fn(x, y, Gamma);
    alp0_oracle = Oracle(alp0_s, alp_n, grad_alp0_s, alp_prox);
    ippm_oracle = copy(oracle);
    ippm_oracle.add_smooth_oracle(alp0_oracle);
        
    % Compute the gradient of K at (x0,z0;y).
    ippm_oracle.eval(x0);
    grad_K0 = ippm_oracle.grad_f_s() + p * (x0 - z0);
    
    % Take a primal + auxilary step.
    o_at_x_next = copy(oracle);
    o_at_x_next.eval(x0 - c * grad_K0);
    x = o_at_x_next.prox_f_n(1 / c);
    z = z0 + beta * (x - z0);
    
    % Compute the gradient of K at (x,z;y).
    ippm_oracle.eval(x);
    grad_K = ippm_oracle.grad_f_s() + p * (x - z);
    
    % Compute v.
    v = grad_K - grad_K0 - (1 / c) * (x - x0) - Gamma * grad_constr_fn(x, params.constr_fn(x)) - p * (x - z0);
    
    % Update the model and history.
    iter = iter + 1;
    
    % Compute some residuals.
    w_hat = v;
    q_hat = params.constr_fn(x);
    
    % Log some numbers if necessary.
    if params.i_logging
      logging_oracle.eval(x);
      history.function_values = [history.function_values; logging_oracle.f_s() + logging_oracle.f_n()];
      history.norm_w_hat_values = [history.norm_w_hat_values; norm_fn(w_hat)];
      history.norm_q_hat_values = [history.norm_q_hat_values; norm_fn(q_hat)];
      history.iteration_values = [history.iteration_values; iter];
      history.time_values = [history.time_values; toc(t_start)];
    end
    
    % Check for the termination of the method.
    if isempty(params.termination_fn)
      if ((params.norm_fn(w_hat) <= opt_tol) && (params.norm_fn(q_hat) <= feas_tol))
        break;
      end
    else
      [tpred, w_hat, q_hat] = params.termination_fn(x, y);
      if tpred
        break;
      end
    end
    
    % Update other iterates.
    y0 = y;
    x0 = x;
    z0 = z;    
  end
  
  % Get ready to output
  model.x = x;
  model.v = w_hat;
  model.w = q_hat;
  history.iter = iter;
  history.runtime = toc(t_start);

end

%% Helper functions

% Fills in parameters that were not set as input.
function params = set_default_params(params)

  % Overwrite if necessary.
  if (~isfield(params, 'i_logging')) 
    params.i_logging = false;
  end
  if (~isfield(params, 'Gamma')) 
    params.Gamma = 10;
  end
  if (~isfield(params, 'alpha')) 
    params.alpha = params.Gamma / 4;
  end
  if (~isfield(params, 'p')) 
    params.p = params.L;
  end
  if (~isfield(params, 'c')) 
    params.c = 1 / (2 * (params.L + params.p + params.Gamma * params.K_constr ^ 2));
  end
  if (~isfield(params, 'beta')) 
    params.beta = 0.5; 
  end
  if (~isfield(params, 'y0')) 
    params.y0 = zeros(size(params.constr_fn(params.x0))); 
  end
  if (~isfield(params, 'z0')) 
    params.z0 = params.x0; 
  end 
  if (~isfield(params, 'termination_fn'))
    params.termination_fn = [];
  end
  
end
