% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [model, history] = AC_ACG(oracle, params)
% The average curvature accelerated composite gradient (AC-ACG) method. 
%
% Note:
%
%   Based on the paper: 
%
%   Liang, J., & Monteiro, R. D. (2019). An average curvature accelerated composite gradient method for nonconvex smooth composite
%   optimization problems. *arXiv preprint arXiv:1909.04248*.
%
% Arguments:
%
%   oracle (Oracle): The oracle underlying the optimization problem.
%
%   params.alpha (double): Controls the rate at which the upper curvature is updated (see $\alpha$ from the original paper).
%     Defaults to ``0.5``.
%
%   params.gamma (double): Controls the rate at which the upper curvature is updated (see $\gamma$ from the original paper).
%     Defaults to ``0.01``.
%
% Returns: 
%
%   A pair of structs containing model and history related outputs of the solved problem associated with the oracle and input
%   parameters.
%
   
  % Timer start.
  t_start = tic;

  % Initialize params.
  z0 = params.x0;
  M_bar = params.L;
  prod_fn = params.prod_fn;
  norm_fn = params.norm_fn;
  
  % Fill in OPTIONAL input params.
  params = set_default_params(params);
  alpha = params.alpha;
  gamma = params.gamma;
  
  % Initialize other params.
  M = gamma * M_bar;
  good = 0;
  CSUM = 0;
  A = 0;
  x = z0;      
  y = z0;   
  iter = 1;
  
  % History params.
  if params.i_logging
    oracle.eval(z0);
    function_values = oracle.f_s() + oracle.f_n();
    iteration_values = 0;
    time_values = 0;
  end
  
  % Solver params.
  opt_tol = params.opt_tol;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  
  % Main loop.
  while true
    
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Auxiliary updates.
    a = (1+sqrt(1+4*M*A))/2/M;
    ANext = A + a;
    tx = A/ANext*y + a/ANext*x;
    
    % Oracle at x_tilde.
    o_at_tx = oracle.eval(tx);
    grad_tx = o_at_tx.grad_f_s();
    f_tx = o_at_tx.f_s();   
    
    % Oracle at x_next.
    o_at_x_inter = oracle.eval(x - a * grad_tx);
    xNext = o_at_x_inter.prox_f_n(a);
    
    % Oracle at y_next.
    o_at_y_inter = oracle.eval(tx - 1 / M * grad_tx);
    yNext = o_at_y_inter.prox_f_n(1 / M);
    o_at_yNext = oracle.eval(yNext);
    grad_yNext = o_at_yNext.grad_f_s();
    f_s_yNext = o_at_yNext.f_s();
    
    % Main updates.
    v = M*(tx-yNext)+grad_yNext-grad_tx;    
    vnorm = norm_fn(v);
    C1 = 2*(f_s_yNext - f_tx - prod_fn(grad_tx,yNext-tx)) / norm_fn(yNext-tx)^2;
    C = max(real(C1),0);
    if C>0.9*M
        yNext = (A*y+a*xNext)/ANext;
    else
        good = good + 1;
    end
    
    % Check for termination of the method.
    if (vnorm < opt_tol)
        break;
    end
    
    % Update history
    if params.i_logging
      oracle.eval(yNext);
      function_values(end + 1) = oracle.f_s() + oracle.f_n();
      iteration_values(end + 1) = iter;
      time_values(end + 1) = toc(t_start);
    end
    
    % Update iterates.
    CSUM = CSUM + C;
    C_avg = CSUM / (iter + 1);
    MNext = 1 / alpha * C_avg;
    A = ANext;
    x = xNext;      
    y = yNext;      
    M = MNext;
    iter = iter + 1;    
  end
  
  % Post-processing.
  model.x = y;     
  model.v = v;
  history.good = good;
  history.iter = iter;
  history.runtime = toc(t_start);
  if params.i_logging
    history.function_values = function_values;
    history.iteration_values = iteration_values;
    history.time_values = time_values;
  end
  
end 

% Fills in parameters that were not set as input.
function params = set_default_params(params)

  % Overwrite if necessary.
  if (~isfield(params, 'alpha')) 
    % Based on the ACT in QP experiments in the AC paper.
    params.alpha = 0.5;
  end
  if (~isfield(params, 'gammma')) 
    % Based on the ACT in the AC paper.
    params.gamma = 0.01;
  end
  if (~isfield(params, 'i_logging')) 
    params.i_logging = false;
  end

end
