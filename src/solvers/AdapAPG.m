%{

FILE DATA
---------
Last Modified: 
  June 19, 2021
Coders: 
  Weiwei Kong

%}


function [model, history] = AdapAPG(oracle, params)
% Adaptive Accelerated Proximal Gradient (APG) Method used in the iALM.
%
% Note:
% 
%   Its iterates are generated according the paper:
%
%   Lin, Q., & Xiao, L. (2014, January). An adaptive accelerated proximal 
%   gradient method and its homotopy continuation for sparse optimization. In
%   *International Conference on Machine Learning* (pp. 73-81). PMLR.
%
% Arguments:
%
%   oracle (Oracle): The oracle underlying the optimization problem.
%
%   params (struct): Contains instructions on how to call the algorithm. This 
%     should ideally  be customized from by caller of this algoirthm, rather 
%     than the user.
%
% Returns: 
%   
%   A pair of structs containing model and history related outputs of the 
%   solved problem associated with the oracle and input parameters.
%

  % Global constants.
  INEQ_TOL = 1e-10;
  
  % Set REQUIRED input params.
  x_m2 = params.x0;
  mu = params.mu;
  L = params.L;
  norm_fn = params.norm_fn;
  prod_fn = params.prod_fn;
  
  % Fill in OPTIONAL input params.
  params = set_default_params(params);
  
  % Parse other inputs.
  termination_type = params.termination_type;
  gamma1 = params.gamma1;
  gamma2 = params.gamma2;
  L_min = params.L_min;
  L_max = params.L_max;
  
  % Check for logging requirements.
  if params.i_logging
    history.function_values = [];
    history.iteration_values = [];
    history.time_values = [];
  end
  
  % Pull in some constants to save compute time
  if strcmp(termination_type, 'iALM')
    eps = params.eps;
  end
  
  % Solver params.
  t_start = tic;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  
  % Special oracles.
  o_at_x_m2 = copy(oracle);
  o_at_x_m1_next = copy(oracle);
  o_at_x_m1 = copy(oracle);
  
  % First loop to estimate L0.
  L0 = L_min;
  o_at_x_m2.eval(x_m2);
  iter = 1;
  i_early_stop = false;
  while true
       
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit && i_early_stop)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Compute L0 and x_m1.
    L0 = gamma1 * L0;
    o_at_x_m1_next.eval(x_m2 - (1 / L0) * o_at_x_m2.grad_f_s());
    x_m1 = o_at_x_m1_next.prox_f_n(1 / L0);
    
    if strcmp(params.acg_steptype, 'variable')
      % Check termination (in a relative sense).
      o_at_x_m1.eval(x_m1);
      if (o_at_x_m1.f_s() <= o_at_x_m2.f_s() + ...
          prod_fn(o_at_x_m2.grad_f_s(), x_m1 - x_m2) + ...
          (L0 / 2) * norm_fn(x_m1 - x_m2) ^ 2 + INEQ_TOL)
        break;
      end
    elseif strcmp(params.acg_steptype, 'constant')
      break
    end

    % Update iterates
    i_early_stop = true;
    iter = iter + 1;
    
  end
  
  % Main APG loop.
  L = L0;
  x = x_m1;
  x0 = x_m1;
  alpha0 = 1;
  o_at_tx = copy(oracle);
  o_at_x_p1 = copy(oracle);
  ALS_oracle = copy(oracle);
  ALS_params = params;
  i_early_stop = false;
  while true
        
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit && i_early_stop)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Line search subroutine.
    ALS_params.x = x;
    ALS_params.x0 = x0;
    ALS_params.L = L;
    ALS_params.alpha0 = alpha0;
    [ALS_model, ALS_history] = AccelLineSearch(ALS_oracle, ALS_params);
    
    % Parse output.
    x_p1 = ALS_model.x_p1;
    tx = ALS_model.tx;
    alpha = ALS_model.alpha;
    L_prev = L;
    L = max([L_min, L_max / gamma2]);
    
    if strcmp(termination_type, 'iALM')
      % Check for termination.
      o_at_tx.eval(tx);
      o_at_x_p1.eval(x_p1);
      if params.is_box
        % Here, |v| = dist(0, ∂F(x)).
        grad_at_x_p1 = o_at_x_p1.grad_f_s();
        v = ...
          (x_p1 == params.box_upper) .* max(0, grad_at_x_p1) + ...
          (x_p1 == params.box_lower) .* min(0, grad_at_x_p1) + ...
          (x_p1 < params.box_upper & x_p1 > params.box_lower) .* grad_at_x_p1;
      else
        % Here, |v| is an upper bound of dist(0, ∂F(x)).
        v = L_prev * (tx - x_p1) + o_at_x_p1.grad_f_s() - o_at_tx.grad_f_s();
      end
      if (norm_fn(v) <= eps)
        break;
      end
    end
    
    % Update iterates
    x0 = x;
    x = x_p1;
    alpha0 = alpha;
    i_early_stop = true;
    iter = iter + ALS_history.iter;
    
    % Add metrics for the current outer iteration if needed.
    if params.i_logging
      oracle.eval(x);
      history.function_values(end + 1) = oracle.f_s() + oracle.f_n();
      history.iteration_values(end + 1) = iter;
      history.time_values(end + 1) = toc(t_start);
    end
       
  end
  
  % Get ready to output.
  model.x = x_p1;
  model.L = L;
  history.iter = iter;
  
end

function [model, history] = AccelLineSearch(oracle, params)
% Line search subroutine used in AdapAPG

  % Global constants
  INEQ_TOL = 1e-10; % buffer room for comparing quantities

  % Parse inputs.
  x = params.x;
  x0 = params.x0;
  L = params.L;
  L_max = params.L_max;
  mu = params.mu;
  alpha0 = params.alpha0;
  gamma1 = params.gamma1;
  prod_fn = params.prod_fn;
  norm_fn = params.norm_fn;
  
  % Initialization.
  o_at_tx = copy(oracle);
  o_at_tx_next = copy(oracle);
  o_at_x_p1 = copy(oracle);
  
  % Main loop.
  M = L / gamma1; % M is equivalent to L (= Lk / gamma1) in the paper.
  iter = 1;
  while true
    M = M * gamma1;
    alpha = sqrt(mu / M);
    % Compute and evaluate tx_k (= y_k in the paper).
    tx = x + (alpha * (1 - alpha0)) / (alpha0 * (1 + alpha)) * (x - x0);
    o_at_tx.eval(tx);
    % Compute and evaluate x_{k+1}.
    o_at_tx_next.eval(tx - (1 / M) * o_at_tx.grad_f_s());
    x_p1 = o_at_tx_next.prox_f_n(1 / M);    
    o_at_x_p1.eval(x_p1);
    % Check the descent condition in a relative sense. 
    if (o_at_x_p1.f_s() < o_at_tx.f_s() + ...
        prod_fn(o_at_tx.grad_f_s(), x_p1 - tx) + ...
        M / 2 * norm_fn(x_p1 - tx) ^ 2 + INEQ_TOL)
      break;
    end
    % Safety check.
    if (M >= L_max)
      break
    end
    
    % Update iterates
    iter = iter + 1;
  end  

  % Prepare output.
  model.x_p1 = x_p1;
  model.M = M;
  model.alpha = alpha;
  model.tx = tx;
  history.iter = iter;
end

% -------------------------------------------------------------------------
%% HELPER FUNCTIONS
% -------------------------------------------------------------------------

% Fills in parameters that were not set as input.
function params = set_default_params(params)

  % Overwrite if necessary.
  if (~isfield(params, 'is_box'))
    params.is_box = false;
  end
  if (~isfield(params, 'termination_type'))
    params.termination_type = 'none';
  end
  if (~isfield(params, 'acg_steptype'))
    params.acg_steptype = 'constant';
  end
  if (~isfield(params, 'gamma1'))
    params.gamma1 = 1;
  end
  if (~isfield(params, 'gamma2'))
    params.gamma2 = 1;
  end
  if (~isfield(params, 'L_min'))
    params.L_min = params.L;
  end
  if (~isfield(params, 'L_max'))
    params.L_max = params.L;
  end
  if (~isfield(params, 'i_logging'))
    params.i_logging = false;
  end

end

%     % DEBUG
%     v_upper = L * (tx - x_p1) + o_at_x_p1.grad_f_s() - o_at_tx.grad_f_s();
%     nrm_v = norm_fn(v);
%     nrm_v_upper = norm_fn(v_upper);
%     mu = params.mu;
%     del1 = norm_fn(tx - x_p1);
%     del2 = norm_fn(x0 - x_p1);
%     disp(table(nrm_v, nrm_v_upper, del1, del2, M, L, mu, iter, alpha, eps));
%     disp(table(x0, tx, x_p1));