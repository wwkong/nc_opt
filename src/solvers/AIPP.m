%{

DESCRIPTION
-----------
Variants of the accelerated inexact proximal point (AIPP) method from the
papers:

"Complexity of a quadratic penalty accelerated inexact proximal point 
method for solving linearly constrained nonconvex composite programs", 
SIAM Journal on Optimization.

"An efficient adaptive accelerated inexact proximal point method for 
solving linearly constrained nonconvex composite problems",  Computational 
Optimization and Applications.

FILE DATA
---------
Last Modified: 
  August 2, 2020
Coders: 
  Weiwei Kong

INPUT
-----
oracle:
  An Oracle object.
params:
  A struct containing input parameters for this function.

OUTPUT
------
model:
  A struct containing model related outputs (e.g. solutions).
history:
  A struct containing history related outputs (e.g. runtimes).

%}

function [model, history] = AIPP(oracle, params)
  
  % Start the timer.
  t_start = tic;
  
  % -----------------------------------------------------------------------
  %% PRE-PROCESSING
  % -----------------------------------------------------------------------
  
  % Set REQUIRED input params.
  z0 = params.x0; 
  M = params.M; 
  prod_fn = params.prod_fn;
  norm_fn = params.norm_fn;
  opt_tol = params.opt_tol; 
  
  % Fill in OPTIONAL input params.
  params = set_default_params(params);
  
  % Set other input params.
  lambda = params.lambda;
  mu_fn = params.mu_fn;
  m = params.m;
  
  % Solver params
  iter_limit = params.iter_limit;
  time_limit = params.time_limit;
  aipp_type = params.aipp_type;
  i_variable_tau = params.i_variable_tau;
  i_increase_lambda = params.i_increase_lambda;
  
  % Data checks
  if (strcmp(aipp_type, 'aipp_c') && params.lambda > 1 / m)
    error('Constant stepsize parameter lambda is too large!')
  end
  
  % Initialize the model struct in case the ACG fails
  model.x = z0;
  model.v = Inf;
  
  % Initialize some auxillary functions and constants.
  iter = 0; % Set to zero to offset ACG iterations
  outer_iter = 1;
  M_est = M;
  L = max(m, M);
  L_prox_fn = @(lam, L) lam * L + 1;
  L_grad_f_s_est = L;
  params_acg = params;
  params_acg.mult_L = params.acg_ls_multiplier;

  % Initialize other input parameters split by aipp_type.
  if (any(strcmp(aipp_type, {'aipp_c', 'aipp_v1', 'aipp_v2'})))
    tau = params.tau;
    params_acg.termination_type = 'gd';
  elseif (strcmp(aipp_type, 'aipp'))
    params_acg.termination_type = 'aipp';
  else
    error(...
      ['Incorrect AIPP type specified! Valid types are: ', ...
       '{aipp, aipp_c, aipp_v1, aipp_v2}'])
  end
  
  % -----------------------------------------------------------------------
  %% REFINEMENT 
  % -----------------------------------------------------------------------
  function model = refine_IPP(L, lambda, z0, z, v)

    % Helper variables
    M_lam = lambda * L + 1;
    o_z = oracle.eval(z);
    f_at_z = o_z.f_s() + o_z.f_n();
    grad_f_s_at_z = o_z.grad_f_s();
    z_next = z - lambda / M_lam * ...
      (grad_f_s_at_z + (z - z0) / lambda - v / lambda);
    o_z_next = oracle.eval(z_next);

    % Compute z_hat variables
    z_hat = o_z_next.prox_f_n(lambda / M_lam);
    o_z_hat = oracle.eval(z_hat);
    f_at_z_hat = o_z_hat.f_s() + o_z_hat.f_n();
    grad_f_s_at_z_hat = o_z_hat.grad_f_s();

    % Compute v_hat;
    q_hat =  1 / lambda * ((v + z0 - z) + M_lam * (z - z_hat));
    v_hat = q_hat + grad_f_s_at_z_hat - grad_f_s_at_z;

    % Compute Delta
    refine_fn_at_z = ...
      (lambda * f_at_z + 1 / 2 * norm_fn(z - z0) ^ 2 - prod_fn(v, z));
    refine_fn_at_z_hat = ...
      (lambda * f_at_z_hat + ...
       1 / 2 * norm_fn(z_hat - z0) ^ 2 - prod_fn(v, z_hat));
    Delta = refine_fn_at_z - refine_fn_at_z_hat;
    
    % Compute some auxillary quantities (for updating tau)
    model.residual_v_hat = norm_fn(v_hat);
    model.residual_1 = norm_fn(v + z0 - z) / lambda;
    model.residual_2 = norm_fn(...
      grad_f_s_at_z_hat - grad_f_s_at_z + M_lam * (z - z_hat) / lambda);

    % Output refinement
    model.o_z_hat = o_z_hat;
    model.z_hat = z_hat;
    model.v_hat = v_hat;
    model.Delta = Delta;

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
    
    % ---------------------------------------------------------------------
    %% ACG CALL AND POST-PROCESSING
    % ---------------------------------------------------------------------
    
    % Set up the ACG oracle. 
    % NOTE: We need to explicitly call copy() because the 'Oracle' class
    % inherits from the 'handle' class
    oracle_acg = copy(oracle);
    oracle_acg.proxify(lambda, z0);
    
    % Set ACG parameters.
    params_acg.lambda = lambda;
    params_acg.x0 = z0;
    params_acg.z0 = z0;
    params_acg.mu = mu_fn(lambda);
    params_acg.L = L_prox_fn(lambda, M); % upper curvature
    params_acg.L_est = L_prox_fn(lambda, M_est); % est. upper curvature
    params_acg.L_grad_f_s_est = L_prox_fn(lambda, L_grad_f_s_est);
    params_acg.t_start = t_start;
        
    % Call the ACG.
    [model_acg, history_acg] = ACG(oracle_acg, params_acg);
    
    % Parse ACG outputs that are invariant of output status.
    iter = iter + history_acg.iter;
    M_est = (model_acg.L_est - 1) / lambda;
    
    % Check for failure of the ACG method.
    if (model_acg.status == -1)
      if (any(strcmp(aipp_type, {'aipp_v1', 'aipp_v2'})))
        i_increase_lambda = false;
        lambda = lambda / 2;
        continue;
      else
        error('Convexity assumption violated for R-AIPPc method!')
      end     
    end
    
    % ---------------------------------------------------------------------
    %% OTHER UPDATES
    % ---------------------------------------------------------------------
    
    % Check for termination using the refinement
    model_refine = refine_IPP(L, lambda, z0, model_acg.y, model_acg.u);
    x = model_refine.z_hat;
    v = model_refine.v_hat;
    if(norm_fn(v) <= opt_tol)
      break;
    end
    
    % Update tau if necessary.
    if (i_variable_tau)
      residual_ratio = ...
        model_refine.residual_v_hat / model_refine.residual_1;
      high_ratio = 1.5;
      low_ratio = 1.2;
      if (residual_ratio > high_ratio)
        tau = tau * high_ratio / residual_ratio;
      elseif (residual_ratio < low_ratio)
        tau = tau * low_ratio / residual_ratio;
      end
    end
       
    % Update lambda if we are using an adaptive stepsize variant.
    if (any(strcmp(aipp_type, {'aipp_v1', 'aipp_v2'})))
            
      % Restart if we have Delta > sigma / 2 * |v + z0 - z| ^ 2.
      if (params_acg.termination_type == "gd")
        if (2 * model_refine.Delta > tau / (lambda * M + 1) * ...
            norm_fn(model_acg.u + z0 - model_acg.y) ^ 2)
          i_increase_lambda = false;
          lambda = lambda / 2;
          continue;
        end
      end
      
      % If there is no failure and lambda is sufficiently small, then 
      % double lambda up to some point.
      lambda_upper = max([100 * 1 / m, 10 * L]); 
      if (i_increase_lambda)
        lambda = min(lambda_upper, lambda * 2);
      end 
    end
    
    % ---------------------------------------------------------------------
    %% ORIGINAL AIPP'S PHASE II CHECK
    % ---------------------------------------------------------------------
    if strcmp(aipp_type, 'aipp')
      if (norm_fn(model_acg.u + z0 - model_acg.y) <= lambda * opt_tol / 5)
        % Final ACG call.
        params_acg.termination_type = 'aipp_phase2';
        params_acg.epsilon_bar = opt_tol ^ 2 / (32 * (M + 2 * m));        
        [model_acg, history_acg] = ACG(oracle_acg, params_acg);
        iter = iter + history_acg.iter;
        % Final refinement.
        model_refine = refine_IPP(L, lambda, z0, model_acg.y, model_acg.u);
        x = model_refine.z_hat;
        v = model_refine.v_hat;
        break;
      end
    end
    
    % Update iterates
    z0 = model_acg.y;
    outer_iter = outer_iter + 1;
                   
  end % main loop end
  
  % -----------------------------------------------------------------------
  %% FINAL POST-PROCESSING
  % -----------------------------------------------------------------------
  
  % Prepare the model and history
  model.x = x;
  model.v = v;
  model.last_lambda = lambda;
  history.iter = iter;
  history.outer_iter = outer_iter;
  history.runtime = toc(t_start);
  history.lambda = lambda;

end % function end
 
% ================================================
% --------------- Other Helper Functions ---------------  
% ================================================

% Fills in parameters that were not set as input
function params = set_default_params(params)

  % aipp_type = 'aipp_v2' ---> necessary to set other default params
  if (~isfield(params, 'aipp_type'))
    params.aipp_type = 'aipp_v2';
    aipp_type = params.aipp_type;
  else
    aipp_type = params.aipp_type;
  end
  
  % m = 1
  if(~isfield(params, 'm'))
    if (strcmp(aipp_type, 'aipp') || strcmp(aipp_type, 'aipp_c'))
      warning('No lower curvature m. Using m = M instead.');
    end
    params.m = params.M;
  end
  
  % sigma = 0.3
  if (~isfield(params, 'sigma'))
    params.sigma = 0.3;
  end
  
  % theta = 4
  if(~isfield(params, 'theta') && strcmp(aipp_type, 'aipp'))
    params.theta = 2 / (1 - params.sigma);
  elseif(~isfield(params, 'theta'))
    params.theta = 4;
  end
  
  % acg_ls_multiplier = 1.25
  if (~isfield(params, 'acg_ls_multiplier'))
    params.acg_ls_multiplier = 1.25;
  end
  
  % lambda ----------------------------------------> dependent on aipp_type
  if strcmp(aipp_type, 'aipp')
    params.lambda = 1 / (2 * params.m);
    params.i_increase_lambda = false;
  elseif strcmp(aipp_type, 'aipp_c')
    params.lambda = 0.9 / params.m;
    params.i_increase_lambda = false;
  elseif strcmp(aipp_type, 'aipp_v1') % decreasing ONLY
    params.lambda = 1;
    params.i_increase_lambda  = false;
  elseif strcmp(aipp_type, 'aipp_v2')
    params.lambda =  1 / params.m;
    params.i_increase_lambda = true;
  end
  
  % mu_fn -----------------------------------------> dependent on aipp_type
  if strcmp(aipp_type, 'aipp')
    params.mu_fn = @(lambda) max(0, 1 - lambda * params.m);
  elseif strcmp(aipp_type, 'aipp_c')
    params.mu_fn = @(lambda) max(0, 1 - lambda * params.m);
  elseif strcmp(aipp_type, 'aipp_v1') % decreasing ONLY
    params.mu_fn = @(lambda) 1;
  elseif strcmp(aipp_type, 'aipp_v2')
    params.mu_fn = @(lambda) 1;
  end
  
  % tau_mult = 10
  if(~isfield(params, 'tau_mult') && strcmp(aipp_type, 'aipp'))
    params.tau_mult = params.sigma;
  elseif(~isfield(params, 'tau_mult'))
    params.tau_mult = 10;
  end
 
  % tau = tau_mult * (lambda * M + 1) in the general case
  if(~isfield(params, 'tau') && ~strcmp(aipp_type, 'aipp'))
    params.tau = params.tau_mult * (params.lambda * params.M + 1);
    params.i_variable_tau = true;
  else
    params.i_variable_tau = false;
  end
  
  % acg_steptype = "variable" (can be set to be "constant")
  if (~isfield(params, 'acg_steptype'))
    params.acg_steptype = "variable";
  end

end