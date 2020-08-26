%{

FILE DATA
---------
Last Modified: 
  August 5, 2020
Coders: 
  Weiwei Kong

%}

function [model, history] = iapial(solver, oracle, params)
% An inexact accelerated proximal augmeneted Lagrangian (IAPIAL) framework 
% for solving a nonconvex composite optimization problem with 
% convex cone constraints
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
  norm_fn = params.norm_fn;
  pc_fn = params.set_projector;
  dc_fn = params.dual_cone_projector;
  
  % Initialize the augmented Lagrangian penalty (ALP) functions.
  alp_n = @(x) 0;
  alp_prox = @(x, lam) x;
  % Value of the AL function: 
  %  Q(x; p) = 1 / (2 * c) * [dist(p + c * constr_fn(x), -K) - |p| ^ 2],
  %  where K is the primal cone.
  function alp_val = alp_fn(x, p, c)
    p_step = p + c * params.constr_fn(x);
    dist_val = norm_fn((-p_step) - pc_fn(-p_step));
    alp_val = 1 / (2 * c) * (dist_val ^ 2 - norm_fn(p) ^ 2);
  end
  % Computes the point Proj_{K^*}(p + c * const_fn(x)).
  function proj_point = dual_alp_proj(x, p, c)
    p_step = p + c * params.constr_fn(x);
    proj_point = dc_fn(p_step);
  end
  % Gradient of the function Q(x; p) with respect to x.
  function grad_alp_val = grad_alp_fn(x, p, c)
    proj_point = dual_alp_proj(x, p, c);
    grad_constr_fn = params.grad_constr_fn;
    %  If the gradient function has a single argument, assume that the
    %  gradient at a point is a constant tensor.
    if nargin(grad_constr_fn) == 1
      grad_alp_val = ...
        tsr_mult(grad_constr_fn(x), proj_point, 'dual');
    % Else, assume that the gradient is a bifunction; the first argument is
    % the point of evaluation, and the second one is what the gradient
    % operator acts on.
    elseif nargin(grad_constr_fn) == 2
      grad_alp_val = grad_constr_fn(x, proj_point);
    else
      error(...
        ['Unknown function prototype for the gradient of the ', ...
         'constraint function']);
    end
  end

  % Fill in OPTIONAL input params.
  params = set_default_params(params);
  
  % Initialize other params.
  lambda = params.lambda;
  sigma = params.sigma;

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
  c0 = max([MIN_PENALTY_CONST, L / K_constr ^ 2]);
  params_acg = params;
  params_acg.mu = 1 - lambda * m;
  params_acg.termination_type = 'aipp';
  
  % Set up some parameters used to define Delta_k.
  stage_outer_iter = 1;

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
    alp0_s = @(z) alp_fn(z, p0, c0);
    grad_alp0_s = @(z) grad_alp_fn(z, p0, c0);
    alp0_oracle = Oracle(alp0_s, alp_n, grad_alp0_s, alp_prox);
    oracle_AL0 = copy(oracle);
    oracle_AL0.add_smooth_oracle(alp0_oracle);
       
    % Create the ACG oracle.
    oracle_acg = copy(oracle_AL0);
    oracle_acg.proxify(lambda, z0);
    
    % Create the ACG params.
    L_psi = lambda * (L + L_constr * norm_fn(p0) + ...
      c0 * (B_constr * L_constr + K_constr ^ 2)) + 1;
    params_acg.x0 = z0;
    params_acg.z0 = z0;
    params_acg.sigma = sigma / sqrt(L_psi);
    params_acg.L = L_psi;
    params_acg.t_start = t_start;
    
    % Call the ACG algorithm and update parameters.
    [model_acg, history_acg] = ACG(oracle_acg, params_acg);
    iter = iter + history_acg.iter;
    
    % Apply the refinement.
    model_refine = refine_IPP(...
      oracle_AL0, params, L_psi, lambda, z0, model_acg.y, model_acg.u);
    
    % Check for termination.
    p_hat = dual_alp_proj(model_refine.z_hat, p0, c0);
    q_hat = (1 / c0) * (p0 - p_hat);
    w_hat = model_refine.v_hat;
    if (norm_fn(w_hat) <= opt_tol && norm_fn(q_hat) <= feas_tol)
      break;
    end
    
    % Apply the dual update and create a new ALP oracle.
    x = model_acg.y;
    p = dual_alp_proj(x, p0, c0);
    alp_s = @(z) alp_fn(z, p, c0);
    grad_alp_s = @(z) grad_alp_fn(z, p, c0);
    alp_oracle = Oracle(alp_s, alp_n, grad_alp_s, alp_prox);
    
    % Check if we need to double c0 (and do some useful precomputations).
    if (outer_iter == stage_outer_iter)
      oracle_Delta_base = copy(oracle);
      oracle_Delta_base.add_smooth_oracle(alp_oracle);
      oracle_Delta_base.eval(x);
      stage_al_base = oracle_Delta_base.f_s() + oracle_Delta_base.f_n();
    elseif (outer_iter > stage_outer_iter)
      % Compute Delta_k.
      oracle_Delta = copy(oracle);
      oracle_Delta.add_smooth_oracle(alp_oracle);
      oracle_Delta.eval(x);
      stage_al_val = oracle_Delta.f_s() + oracle_Delta.f_n();
      Delta = 1 / (outer_iter - stage_outer_iter) * ...
        (stage_al_base - stage_al_val);
      % Check the update condition and update the relevant constants.
      % !!! The constant Delta_mult will need to be changed in the
      % nonlinear case !!!
      Delta_mult = lambda * (1 - sigma ^ 2) / (4 * (1 + 2 * sigma)) ^ 2;
      if (Delta <= opt_tol ^ 2 / Delta_mult)
        c0 = 2 * c0;
        stage = stage + 1;
        stage_outer_iter = outer_iter + 1;
      end
      
%       % For logging
%       disp(table(...
%         norm_fn(w_hat), norm_fn(q_hat), iter, outer_iter, ...
%         stage_outer_iter, stage, c0, Delta, norm_fn(model_acg.u)));
    else
      error('Something went wrong with the variable `outer_iter`');
    end
    
    % Update the other iterates.
    p0 = p;
    z0 = x;
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
  if (~isfield(params, 'sigma'))
    params.sigma = 0.3 * (params.lambda * params.M + 1);
  end
  if (~isfield(params, 'acg_steptype'))
    params.acg_steptype = "variable";
  end  

end