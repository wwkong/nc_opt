% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [model, history] = IAIPAL(~, oracle, params)
% An inexact accelerated proximal augmeneted Lagrangian (IAPIAL) framework for solving a nonconvex composite optimization problem
% with convex cone constraints.
% 
% Note:
% 
%   Based on the paper:
%
%   Kong, W., Melo, J. G., & Monteiro, R. D. (2020). Iteration-complexity of a proximal augmented Lagrangian method for solving
%   nonconvex composite optimization problems with nonlinear convex constraints. *arXiv preprint arXiv:2008.07080*\.
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
  m = params.m;
  M = params.M;
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
  norm_fn = params.norm_fn;
  pc_fn = params.set_projector; % projection onto the cone K.
  
  % Cone projector function on the point q = p + c * g(x) on the cones:
  %   -K (primal) and K^* (dual).
  function [dual_point, primal_point]= cone_proj(x, p, c)
    p_step = p + c * params.constr_fn(x);
    primal_point = -pc_fn(-p_step);
    dual_point = p_step - primal_point;
  end
  
  % Function that creates the augmented Lagrangian oracle for the function
  %
  %   L_c(x; p) := phi(x) + 1 / (2 * c) * [dist(p + c * constr_fn(x), -K) - |p| ^ 2],
  %
  % where K is the primal cone.
  function al_oracle = create_al_oracle(p, c)
    
    % Create the penalty oracle.
    function oracle_struct = alp_eval_fn(~)
      % Wrapper functions.
      function val = wrap_f_s(x)
        [~, primal_proj_point]= cone_proj(x, p, c);
        p_step = p + c * params.constr_fn(x);
        dist_val = norm_fn((-p_step) - primal_proj_point);
        val = 1 / (2 * c) * (dist_val ^ 2 - norm_fn(p) ^ 2);
      end
      function val = wrap_grad_f_s(x)
        [dual_proj_proint, ~]= cone_proj(x, p, c);
        grad_constr_fn = params.grad_constr_fn;
        %  If the gradient function has a single argument, assume that the gradient at a point is a constant tensor.
        if nargin(grad_constr_fn) == 1
          val = tsr_mult(grad_constr_fn(x), dual_proj_proint, 'dual');
          % Else, assume that the gradient is a bifunction; the first argument is the point of evaluation, and the second one is
          % what the gradient operator acts on.
        elseif nargin(grad_constr_fn) == 2
          val = grad_constr_fn(x, dual_proj_proint);
        else
          error('Unknown function prototype for the gradient of the constraint function');
        end
      end
      % Create the function struct.
      oracle_struct.f_s = @wrap_f_s;
      oracle_struct.f_n = @(x) 0;
      oracle_struct.grad_f_s = @wrap_grad_f_s;
      oracle_struct.prox_f_n = @(x, lam) x;
    end
    oracle_AL1 = Oracle(@alp_eval_fn);
    
    % Create the combined oracle.
    al_oracle = copy(oracle);
    al_oracle.add_smooth_oracle(oracle_AL1);
    
  end

  % Fill in OPTIONAL input params.
  params = set_default_params(params);
  
  % Initialize other params.
  lambda = params.lambda;
  nu = params.nu;
  sigma_min = params.sigma_min;
  sigma_type = params.sigma_type;
  penalty_multiplier = params.penalty_multiplier;
  i_reset_multiplier = params.i_reset_multiplier;
  i_reset_prox_center = params.i_reset_prox_center;

  % Initialize history parameters.
  if params.i_logging
    logging_oracle = copy(oracle);
    history.function_values = [];
    history.norm_w_hat_values = [];
    history.norm_q_hat_values = [];
    history.iteration_values = [];
    history.time_values = [];
  end
  
  % DEBUG ONLY.
  if params.i_debug
    history.theta1 = [];
    history.theta2 = [];
    history.inner_iter = [];
    history.c = [];
    history.sigma = [];
    history.L_est = [];
    history.norm_p_hat = [];
  end

  % Initialize framework parameters.
  iter = 0;
  outer_iter = 1;
  stage = 1;
  z0 = params.x0;
  z_hat = z0;
  p0 = zeros(size(params.constr_fn(z0)));
  p_hat = p0;
  o_p0 = p0;
  o_z0 = z0;
  c0 = params.c0;
  first_c0 = c0;
  params_acg = params;
  params_acg.mu = 1 - lambda * m;
  params_acg.termination_type = 'aipp_sqr';
  i_first_acg_run = true;
  w_hat = Inf;
  q_hat = Inf;
  
  % Set up some parameters used to define Delta_k.
  stage_outer_iter = 1;
  
  %% MAIN ALGORITHM
  while true
    
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit && outer_iter > 1)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Create the AL oracle.
    oracle_al0 = create_al_oracle(p0, c0);
       
    % Create the ACG oracle.
    oracle_acg = copy(oracle_al0);
    oracle_acg.proxify(lambda, z0);
    
    % Create the ACG params.
    L_psi = lambda * (M + L_constr * norm_fn(p0) + c0 * (B_constr * L_constr + K_constr ^ 2)) + 1;
    if (outer_iter == 1)
      first_L_psi = L_psi;
    end
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
    params_acg.t_start = t_start;
    
    % Stepsize selection
    if strcmp(params.acg_steptype, 'constant') 
      params_acg.L_est = L_psi;
    elseif strcmp(params.acg_steptype, 'variable') 
      if (i_first_acg_run)
        o_z0 = copy(oracle);
        o_z0.eval(z0);
        iter = iter + 1;
        o_z1 = copy(oracle);
        o_z1.eval(z0 - o_z0.grad_f_s() / M);
        iter = iter + 1;
        z1 = o_z1.prox_f_n(1 / M);
        o_z1.eval(z1);
        iter = iter + 1;
        if (norm_fn(z0 - z1) <= 1e-6)
          params_acg.L_est = lambda * M + 1;
        else
          M_est = norm_fn(o_z0.grad_f_s() - o_z1.grad_f_s()) / norm_fn(z0 - z1);
          params_acg.L_est = lambda * M_est + 1;
        end
      else
        % Use the previous estimate.
        params_acg.L_est = model_acg.L_est;
      end
    else
      error('Unknown ACG steptype!');
    end
    
    % Call the ACG algorithm and update parameters.
    [model_acg, history_acg] = ACG(oracle_acg, params_acg);
    iter = iter + history_acg.iter;
    i_first_acg_run = false;
    
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
        
    % Apply the refinement.
    model_refine = refine_IPP(oracle_al0, params, L_psi, lambda, z0, model_acg.y, model_acg.u);
    z_hat = model_refine.z_hat;
    
    % Compute refined quantities.
    p_hat = cone_proj(z_hat, p0, c0);
    q_hat = (1 / c0) * (p0 - p_hat);
    w_hat = model_refine.v_hat;
    
    % Log some numbers if necessary.
    if params.i_logging
      logging_oracle.eval(z_hat);
      history.function_values = [history.function_values; logging_oracle.f_s() + logging_oracle.f_n()];
      history.norm_w_hat_values = [history.norm_w_hat_values; norm_fn(w_hat)];
      history.norm_q_hat_values = [history.norm_q_hat_values; norm_fn(q_hat)];
      history.iteration_values = [history.iteration_values; iter];
      history.time_values = [history.time_values; toc(t_start)];
    end
    
    % Check for termination.
    if (isempty(params.termination_fn) || params.check_all_terminations)
      if (norm_fn(w_hat) <= opt_tol && norm_fn(q_hat) <= feas_tol)
        break;
      end
    end
    if (~isempty(params.termination_fn) || params.check_all_terminations)
      [tpred, w_hat, q_hat] = params.termination_fn(model_refine.z_hat, p_hat);
      if tpred
        break;
      end
    end
    
    % DEBUG ONLY.
    if params.i_debug
      history.theta1 = [history.theta1; norm_fn(w_hat) / norm_fn(model_refine.v_hat)];
      history.theta2 = [history.theta2; norm_fn(q_hat) / norm_fn((1 / c0) * (p0 - p_hat))];
      history.inner_iter = [history.inner_iter; history_acg.iter];
      history.c = [history.c; c0];
      history.sigma = [history.sigma; sigma];
      history.L_est = [history.L_est; params_acg.L_est];
      history.norm_p_hat = [history.norm_p_hat; norm_fn(p_hat)];
    end
    
    % Apply the dual update and create a new ALP oracle.
    x = model_acg.y;
    p = cone_proj(x, p0, c0);
    oracle_Delta = create_al_oracle(p, c0);
    
    % Check if we need to double c0 (and do some useful precomputations).
    if (outer_iter == stage_outer_iter)
      oracle_Delta.eval(x);
      stage_al_base = oracle_Delta.f_s() + oracle_Delta.f_n();
    elseif (outer_iter > stage_outer_iter)
      % Compute Delta_k.
      oracle_Delta.eval(x);
      stage_al_val = oracle_Delta.f_s() + oracle_Delta.f_n();
      Delta = 1 / (outer_iter - stage_outer_iter) * (stage_al_base - stage_al_val - norm_fn(p) ^ 2 / (2 * c0));
      % Check the update condition and update the relevant constants.
      Delta_mult = lambda * (1 - sigma ^ 2) / (2 * (1 + 2 * nu) ^ 2);
      if (Delta <= ((opt_tol ^ 2) * Delta_mult))
        c0 = penalty_multiplier * c0;
        stage = stage + 1;
        stage_outer_iter = outer_iter + 1;
        if (i_reset_multiplier)
          p = o_p0;
        end
        if (i_reset_prox_center)
          x = o_z0;
        end
      end
    else
      error('Something went wrong with the variable `outer_iter`');
    end
    
    % Update the other iterates.
    p0 = p;
    z0 = x;
    outer_iter = outer_iter + 1;
    
  end
  
  % Prepare to output
  model.x = z_hat;
  model.y = p_hat;
  model.v = w_hat;
  model.w = q_hat;
  history.acg_ratio = iter / outer_iter;
  history.iter = iter;
  history.outer_iter = outer_iter;
  history.stage = stage;
  history.first_L_psi = first_L_psi;
  history.last_L_psi = L_psi;
  history.last_sigma = sigma;
  history.first_c0 = first_c0;
  history.last_c0 = c0;
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
  if (~isfield(params, 'i_debug')) 
    params.i_debug = false;
  end
  if (~isfield(params, 'i_reset_multiplier')) 
    params.i_reset_multiplier = false;
  end
  if (~isfield(params, 'i_reset_prox_center')) 
    params.i_reset_prox_center = false;
  end
  if (~isfield(params, 'lambda'))
    params.lambda = 1 / (2 * params.m);
  end
  if (~isfield(params, 'sigma_min'))
    params.sigma_min = 1 / sqrt(2);
  end
  if (~isfield(params, 'c0'))
    params.c0 = max([MIN_PENALTY_CONST, params.M / params.K_constr ^ 2]);
  end
  if (~isfield(params, 'penalty_multiplier'))
    params.penalty_multiplier = 2;
  end
  if (~isfield(params, 'nu'))
    params.nu = sqrt(params.sigma_min * (params.lambda * params.M + 1));
  end
  if (~isfield(params, 'sigma_type'))
    params.sigma_type = 'constant';
  end
  if (~isfield(params, 'acg_steptype'))
    params.acg_steptype = "variable";
  end  
  if (~isfield(params, 'termination_fn'))
    params.termination_fn = [];
  end
  if (~isfield(params, 'check_all_terminations'))
    params.check_all_terminations = false;
  end
end
