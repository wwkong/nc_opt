% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [model, history] = ECG(oracle, params)
% The well-known exact composite gradient (ECG) method with constant stepsize.
% 
% Note:
% 
%   For reference, see the paper:
%
%   Nesterov, Y. (2013). Gradient methods for minimizing composite functions. *Mathematical Programming, 140*\(1), 125-161.
%
% Arguments:
%
% 	oracle (Oracle): The oracle underlying the optimization problem.
%
% 	params (struct): Contains instructions on how to call the algorithm.
%
% Returns:
%   A pair of structs containing model and history related outputs of the solved problem associated with the oracle and input
%   parameters.
%

  % Timer start.
  t_start = tic;
  
  % Initialize params.
  x0 = params.x0;
  x_prev = x0;
  norm_fn = params.norm_fn;
  prod_fn = params.prod_fn;
  iter = 1;
  
  % Fill in OPTIONAL input params.
  params = set_default_params(params);
  
  % History params.
  if params.i_logging
    oracle.eval(x0);
    function_values = oracle.f_s() + oracle.f_n();
    iteration_values = 0;
    time_values = 0;
    vnorm_values = Inf;
  end
  history.min_norm_of_v = Inf;
  
  % Solver params.
  opt_tol = params.opt_tol;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  
  % Choose the intial estimate of L.
  if (strcmp(params.steptype, 'constant'))
    m = params.m;
    M = params.M;
    L = max(m, M);
  elseif strcmp(params.steptype, 'adaptive')
    m = params.m0;
    M = params.M0;
    L = max(m, M);
    Lf = L;
    L0 = L;
    gamma_u = 2;
    gamma_d = 2;
  else
    error('Unknown steptype!')
  end
  
  %% MAIN ALGORITHM

  while true
    
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Oracle at xPrev.
    o_xPrev = oracle.eval(x_prev);
    grad_f_s_at_x_prev = o_xPrev.grad_f_s();
    y_bar = x_prev - 1 / L * grad_f_s_at_x_prev;
    
    % Oracle at yBar.
    o_yBar = oracle.eval(y_bar);
    x_bar = o_yBar.prox_f_n(1 / L);
    
    % Oracle at xBar.
    o_xBar = oracle.eval(x_bar);
    grad_f_s_at_x_bar = o_xBar.grad_f_s();
    
    % Update L based on Nesterov's suggested scheme.
    if strcmp(params.steptype, 'adaptive')
      while (o_xBar.f_s() - (o_xPrev.f_s() + prod_fn(grad_f_s_at_x_prev, x_bar - x_prev)) > L * norm_fn(x_bar - x_prev) ^ 2 / 2 && L < Lf)
        L = L * gamma_u;
        iter = iter + 1;
        % Update oracles
        o_yBar = oracle.eval(x_prev - 1 / L * grad_f_s_at_x_prev);
        x_bar = o_yBar.prox_f_n(1 / L);
        o_xBar = oracle.eval(x_bar);
      end
    end
    
    % Check for termination.
    v_bar = L * (x_prev - x_bar) + grad_f_s_at_x_bar - grad_f_s_at_x_prev;
    norm_v = norm_fn(v_bar);
    history.min_norm_of_v = min([history.min_norm_of_v, norm_v]);
    if (norm_v <= opt_tol)
      break;
    end
    
    % Update history.
    if params.i_logging
      oracle.eval(x_bar);
      function_values(end + 1) = oracle.f_s() + oracle.f_n();
      iteration_values(end + 1) = iter;
      time_values(end + 1) = toc(t_start);
      vnorm_values(end + 1) = norm_v;
    end
    
    % Update iterates.
    x_prev = x_bar;
    iter = iter + 1;
    if strcmp(params.steptype, 'adaptive')
      L = max([L0, L / gamma_d]);
    end
    
  end
  
  % Parse outputs.
  model.x = x_bar;
  model.v = v_bar;
  history.iter = iter;
  history.runtime = toc(t_start);
  if params.i_logging
    history.function_values = function_values;
    history.iteration_values = iteration_values;
    history.time_values = time_values;
    history.vnorm_values = vnorm_values;
  end
  
end

% Fills in parameters that were not set as input.
function params = set_default_params(params)

  % Overwrite if necessary.
  if (~isfield(params, 'i_logging')) 
    params.i_logging = false;
  end
  
  % Default steptype is 'adaptive'. It is based on Nesterov's
  % adaptive scheme (see his paper titled "Gradient methods for minimizing 
  % composite functions").
  if (~isfield(params, 'steptype')) 
    params.steptype = 'adaptive';
  end

end
