%{

FILE DATA
---------
Last Modified: 
  August 5, 2020
Coders: 
  Weiwei Kong

%}

function [model, history] = AIDAL(~, oracle, params)
% An accelerated inexact dampened augmeneted Lagrangian (AIDAL) framework 
% for solving a nonconvex composite optimization problem with linear
% constraints
% 
% Note:
% 
%   Based on the paper:
%
%     ?????
%
% Arguments:
% 
%   solver (function handle): A solver for unconstrained composite
%     optimization.
% 
%   oracle (Oracle): The oracle underlying the optimization problem.
% 
%   params (struct): Contains instructions on how to call the framework.
%   
%   ??? add defaults here ???
% 
% Returns:
%   
%   A pair of structs containing model and history related outputs of the 
%   solved problem associated with the oracle and input parameters.
%

  % Global constants.
  MIN_PENALTY_CONST = 1;
  BISECTION_TOL = 1e-3;

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
  % Value of the AL function: 
  %   Q(x; p) = 1 / (2 * c) * [
  %     dist((1-theta) * p + c * constr_fn(x), -K) ^ 2 - 
  %     (1 - theta) ^ 2 |p| ^ 2 ],
  % where K is the primal cone.
  function alp_val = alp_fn(x, p, c, theta)
    p_step = (1 - theta) * p + c * params.constr_fn(x);
    dist_val = norm_fn((-p_step) - pc_fn(-p_step));
    alp_val = ...
      1 / (2 * c) * (dist_val ^ 2 - (1 - theta) ^ 2 * norm_fn(p) ^ 2);
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
      grad_alp_val = ...
        tsr_mult(grad_constr_fn(x), p_step, 'dual');
    % Else, assume that the gradient is a bifunction; the first argument is
    % the point of evaluation, and the second one is what the gradient
    % operator acts on.
    elseif nargin(grad_constr_fn) == 2
      grad_alp_val = grad_constr_fn(x, p_step);
    else
      error(...
        ['Unknown function prototype for the gradient of the ', ...
         'constraint function']);
    end
  end

  % -----------------------------------------------------------------------
  % ADAPTIVE SUBROUTINES 
  % -----------------------------------------------------------------------
  % Finds the best choice of chi based on a bisection search
  function chi = find_chi(oracle, v, z0, z, p00, p0, c0, c, theta, sigma)
    % Basic evaluations
    o_at_z0 = copy(oracle);
    o_at_z0.eval(z0);
    phi_at_z0 = o_at_z0.f_s() + o_at_z0.f_n();
    o_at_z = copy(oracle);
    o_at_z.eval(z);
    phi_at_z = o_at_z.f_s() + o_at_z.f_n();
    % Value of (1 - sigma ^ 2) / (2 * lambda) * |r| ^2.
    lhs_val = ...
      (1 - sigma ^ 2) / (2 * lambda) * norm_fn(v + z0 - z) ^ 2;
    % Value of Psi_{j-1} - Psi_j + pi_{j}
    function val = rhs_val(chi)
      p = dual_update(z, p0, c, theta, chi);
      f0 = (1 / (chi * c0)) * (p0 - (1 - theta) * p00);
      a_theta = theta * (1 - theta);
      b_theta = (2 - theta) * (1 - theta);
      alpha_fn = @(chi) ...
        ((1 - 2 * chi * b_theta) - (1 - theta) ^ 2) / ...
        (2 * chi);
      cur_Psi = ...
        phi_at_z + alp_fn(z, p, c, theta) - ...
        a_theta / (2 * chi * c) * norm_fn(p) ^ 2 + ...
        alpha_fn(chi) / (4 * chi * c) * norm_fn(p - p0) ^ 2;
      prev_Psi = ...
        phi_at_z0 + alp_fn(z0, p0, c0, theta) - ...
        a_theta / (2 * chi * c0) * norm_fn(p0) ^ 2 + ...
        alpha_fn(chi) / (4 * chi * c0) * norm_fn(p0 - p00) ^ 2;
      cur_pi = ...
        (c - c0) / 2 * (...
          norm_fn(f0) ^ 2 + ...
          1 / (chi * c0) * ...
            prod_fn((p - p0) - (1 - theta) * (p0 - p00), f0)) + ...
        a_theta / (2 * chi) * (1 / c0 - 1 / c) * norm_fn(p0) ^ 2;
      val = prev_Psi - cur_Psi + cur_pi;

    end
    % Run a bisection method to find the tightest chi.
    low = theta ^ 2 / (2 * (2 - theta) * (1 - theta));
    high = 1;
    mid = (high + low) / 2;
    while (abs(low - high) > BISECTION_TOL)
      if (lhs_val <= rhs_val(mid))
        low = mid;
      else
        high = mid;
      end
      mid = (low + high) / 2;
    end
    chi = low;
  end
  % -----------------------------------------------------------------------

  % Fill in OPTIONAL input params.
  params = set_default_params(params);
  
  % Initialize other params.
  lambda = params.lambda;
  nu = params.nu;
  sigma_min = params.sigma_min;
  sigma_type = params.sigma_type;
  theta = params.theta;
  chi = params.chi;

  % Initialize history parameters.
  if params.i_logging
    history.function_values = [];
    history.iteration_values = [];
    history.time_values = [];
  end

  % Initialize framework parameters.
  iter = 0;
  outer_iter = 1;
  stage = 1;
  z0 = params.x0;
  p0 = zeros(size(params.constr_fn(z0)));
  p00 = p0;
  c = max([MIN_PENALTY_CONST, L / K_constr ^ 2]);
  c0 = c;
  params_acg = params;
  params_acg.mu = 1 - lambda * m;
  params_acg.termination_type = 'aipp_sqr';

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
    L_psi = lambda * (L + L_constr * norm_fn(p0) + ...
      c * (B_constr * L_constr + K_constr ^ 2)) + 1;
    if (strcmp(sigma_type, 'constant'))
      sigma = sigma_min;
    elseif (strcmp(sigma_type, 'variable'))
      sigma = min([nu / sqrt(L_psi), sigma_min]);
    else 
      error('Unknown sigma type!');
    end
    params_acg.x0 = z0;
    params_acg.z0 = z0;
    params_acg.sigma = sigma;
    params_acg.L = L_psi;
    params_acg.t_start = t_start;
    
    % Call the ACG algorithm and update parameters.
    [model_acg, history_acg] = ACG(oracle_acg, params_acg);
    z = model_acg.y;
    v = model_acg.u;
    iter = iter + history_acg.iter;
    
    % Apply the refinement.
    model_refine = ...
      refine_IPP(oracle_AL0, params, L_psi, lambda, z0, z, v);
    
    % Check for termination.
    p_hat = dual_update(model_refine.z_hat, p0, c, theta, chi);
    q_hat = (1 / (chi * c)) * (p_hat - (1 - theta) * p0);
    w_hat = model_refine.v_hat;
    norm_w_hat = norm_fn(w_hat);
    norm_q_hat = norm_fn(q_hat);
    if (norm_w_hat <= opt_tol && norm_q_hat <= feas_tol)
      break;
    end
    
    % Check if we need to double c (and do some useful precomputations).    
    if ((nu * K_constr * norm_fn(v + z0 - z) <= feas_tol / 2) && ...
        (norm_q_hat > feas_tol))
      c = 2 * c;
      stage = stage + 1;
    end
    
%     % Find a relaxed value for chi. (WIP)
%     if ((outer_iter >= 3) && (strcmp(params.chi_type, 'adaptive')))
%       chi = find_chi(oracle, v, z0, z, p00, p0, c0, c, theta, sigma);
%     end
    chi = 1;
    
    % Apply the dual update and create a new ALP oracle.
    p = dual_update(z, p0, c, theta, chi);
    
    % Update the other iterates.
    p00 = p0;
    c0 = c;
    p0 = p;
    z0 = z;
    outer_iter = outer_iter + 1;
    
  end
  
  % Prepare to output
  model.x = model_refine.z_hat;
  model.y = p_hat;
  model.v = w_hat;
  model.w = q_hat;
  history.iter = iter;
  history.stage = stage;
  history.runtime = toc(t_start);
  
end

% Fills in parameters that were not set as input.
function params = set_default_params(params)

  % Overwrite if necessary.
  if (~isfield(params, 'i_logging')) 
    params.i_logging = false;
  end
  if (~isfield(params, 'lambda'))
    params.lambda = 1 / (2 * params.m);
  end
  if (~isfield(params, 'nu'))
    params.nu = sqrt(0.3 * (params.lambda * params.M + 1));
  end
  if (~isfield(params, 'sigma_min'))
    params.sigma_min = 1 / sqrt(2);
  end
  if (~isfield(params, 'sigma_type'))
    params.sigma_type = 'constant';
  end
  if (~isfield(params, 'theta'))
    params.theta = 0.01;
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

end