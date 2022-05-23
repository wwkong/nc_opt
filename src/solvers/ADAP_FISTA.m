% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [model, history] = ADAP_FISTA(oracle, params)
% The adaptive nonconvex fast iterative soft theresholding (ADAP-NC-FISTA) method. 
% 
% See Also:
% 
%   **src/solvers/NC_FISTA.m**
%
% Note:
% 
%   Based on the paper (see the ADAP-NC-FISTA method):
%
%   Liang, J., Monteiro, R. D., & Sim, C. K. (2019). A FISTA-type accelerated gradient algorithm for solving smooth nonconvex
%   composite optimization problems. *arXiv preprint arXiv:1905.07010*.
%
% Arguments:
%
%   oracle (Oracle): The oracle underlying the optimization problem.
%
%   params.theta (double): Controls how the stepsize is updated (see $\theta$ from the original paper). Defaults to ``1.25``.
%
% Returns: 
%
%   A pair of structs containing model and history related outputs of the solved problem associated with the oracle and input
%   parameters.
%

  % Timer start.
  t_start = tic;
  
  % Stay consistent with the framework.
  [f_s, ~, grad_f_s, prox_f_n] = oracle.decompose();
  f = @(x) f_s(x);
      
  % Main params.
  z0 = params.x0;
  
  % Fill in OPTIONAL input params.
  params = set_default_params(params);
  theta = params.theta;
  
  % History params.
  if params.i_logging
    oracle.eval(z0);
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
  
  % Initialize other params.
  prod_fn = params.prod_fn;
  norm_fn = params.norm_fn;
  ell_f = @(a,b) f(b) + prod_fn(grad_f_s(b), a - b);
  iter = 0;
  count = 0;
  A0 = 0;
  A = A0;
  x = z0;
  y = z0;
  m = params.m0;
  M = params.M0;
    
  %% LOCAL FUNCTIONS
  function [yNext, MNext, lam, xi, count] = SUB(tx, lam, xi, theta, mNext, a, count)

    stepsize = lam*a/(a+2*lam*xi);
    yNext = prox_f_n(tx - stepsize * grad_f_s(tx), stepsize);
    count = count + 1;
    MNext =  2*(f_s(yNext)-f_s(tx)-prod_fn(grad_f_s(tx),yNext-tx)) / norm_fn(yNext-tx)^2;
    lamk = lam;

    while (lam*MNext > 0.9) || (xi*(lamk - lam/a) < mNext*lam/2)
        if (lam*MNext > 0.9)
            lam = min([lam/theta,0.9/MNext]);
        end
        if (xi*(lamk - lam/a) < mNext*lam/2)
            xi = 2*xi;
        end
        stepsize = lam*a/(a+2*lam*xi);
        yNext = prox_f_n(tx- stepsize * grad_f_s(tx), stepsize);
        MNext = 2*(f_s(yNext)-f_s(tx)-prod_fn(grad_f_s(tx),yNext-tx)) / norm_fn(yNext-tx)^2;
        count = count + 1;
    end
  end % End SUB.

  function [lam, xi, count] = CURV(z0)
    
    curv_f = @(a,b) 2*(ell_f(a,b) - f_s(a)) / norm_fn(a-b)^2;
    x0 = z0;
    C = 0;
    scale = 100; % could play around with this parameter
    count = 0;
    
    % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    % NOTE: x0 CANNOT BE ZERO SINCE x0 = 2 * x0.
    % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if (norm_fn(x0) == 0)
      error('x0 cannot be the zero vector!')
    end
    
    theta = abs(curv_f(2 * x0, x0));
    while abs(C) <= theta
        theta = theta/2;
        x = prox_f_n(x0 - grad_f_s(x0)/theta, theta);
        C = curv_f(x,x0); 
        count = count + 1;
    end
    M = theta / scale;
    m = M;
    lam = 1 / M;
    xi = 2 * m;
    
  end % End CURV.

  % Use Jiaming's subroutine for choosing (lambda, xi).
  [lam, xi, count] = CURV(z0);
  iter = iter + count;

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
    
    % Main updates.
    a = (1+sqrt(1+4*A))/2;
    ANext = A + a;
    tx = A/ANext*y + a/ANext*x;
    ty = A/ANext*y + a/ANext*z0;    
    if norm_fn(ty - tx) < 1e-20
        mNext = 0;
    else
        mNext = max(2*(ell_f(ty,tx)-f(ty))/norm_fn(ty-tx)^2,0);
    end
        
    [yNext, MNext, lam, xi, count] = SUB(tx, lam, xi, theta, mNext, a, count);
    xNext = (a+xi*lam)/(xi*lam+1)*yNext - (a-1)/(xi*lam+1)*y;
    v = (1/lam + xi/a)*(tx-yNext)+grad_f_s(yNext)-grad_f_s(tx);
    iter = iter + count;

    % Check for early termination.
    norm_v = norm_fn(v);
    history.min_norm_of_v = min([history.min_norm_of_v, norm_v]);
    if (norm_v <= opt_tol)
      break
    end
    
    % Update history
    if params.i_logging
      oracle.eval(yNext);
      function_values(end + 1) = oracle.f_s() + oracle.f_n();
      iteration_values(end + 1) = iter;
      time_values(end + 1) = toc(t_start);
      vnorm_values(end + 1) = norm_v;
    end
    
    % Update.
    y = yNext;
    A = ANext;
    x = xNext;
   
  end
  
  %% POST-PROCESSING
  model.A = A;
  model.x = x;
  model.y = y;
  model.v = v;
  history.m = max(m, abs(mNext));
  history.M = max(M, abs(MNext));
  history.runtime = toc(t_start);
  if params.i_logging
    history.function_values = function_values;
    history.iteration_values = iteration_values;
    history.time_values = time_values;
    history.vnorm_values = vnorm_values;
  end

  % Count backtracking iterations.
  history.iter = count;
  
end

% Fills in parameters that were not set as input.
function params = set_default_params(params)

  % Overwrite if necessary.
  if (~isfield(params, 'theta')) 
    params.theta = 1.25;
  end
  if (~isfield(params, 'i_logging')) 
    params.i_logging = false;
  end

end
