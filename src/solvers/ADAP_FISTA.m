%{

DESCRIPTION
-----------
The adaptive nonconvex FISTA (ADAP-NC-FISTA) method from the paper:

"A FISTA-type accelerated gradient algorithm for solving smooth nonconvex 
composite optimization problems", arXiv:1905.07010 [math.OC].

FILE DATA
---------
Last Modified: 
  August 2, 2020
Coders: 
  Weiwei Kong, Jiaming Liang

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

function [model, history] = ADAP_FISTA(oracle, params)
%  check curvature + adaptive xi               

  % Timer start.
  t_start = tic;
  
  % Stay consistent with the framework.
  [f_s, ~, grad_f_s, prox_f_n] = oracle.decompose();
  f = @(x) f_s(x);
      
  % Main params.
  z0 = params.x0;
  theta = 1.25;
  
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
  m = params.m;
  M = params.M;
    
  % -----------------------------------------------------------------------
  %% LOCAL FUNCTIONS
  % -----------------------------------------------------------------------
  function [yNext, MNext, lam, xi, count] = ...
    SUB(tx, lam, xi, theta, mNext, a, count)

    stepsize = lam*a/(a+2*lam*xi);
    yNext = prox_f_n(tx - stepsize * grad_f_s(tx), stepsize);
    count = count + 1;
    MNext = ...
      2*(f_s(yNext)-f_s(tx)-prod_fn(grad_f_s(tx),yNext-tx)) / ...
      norm_fn(yNext-tx)^2;
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
        MNext = ...
          2*(f_s(yNext)-f_s(tx)-prod_fn(grad_f_s(tx),yNext-tx)) / ...
          norm_fn(yNext-tx)^2;
        count = count + 1;
    end
  end % End SUB.

  function [lam, xi] = CURV(z0)
    
    curv_f = @(a,b) 2*(ell_f(a,b) - f_s(a)) / norm_fn(a-b)^2;
    x0 = z0;
    C = 0;
    scale = 100; % could play around with this parameter
    
    % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    % NOTE: x0 CANNOT BE ZERO SINCE x0 = 2 * x0.
    % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if (norm_fn(x0) == 0)
      error('x0 cannot be the zero vector!')
    end
    
    theta = abs(curv_f(2 * x0, x0));
    while abs(C) <= theta
        theta = theta/2;
        x = prox_f_n(x0 - grad_f_s(x0)/theta, theta);
        C = curv_f(x,x0); 
    end
    M = theta / scale;
    m = M;
    lam = 1 / M;
    xi = 2 * m;
    
  end % End CURV.

  % Use Jiaming's subroutine for choosing (lambda, xi).
  [lam, xi] = CURV(z0);

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
        
    [yNext, MNext, lam, xi, count] = ...
      SUB(tx, lam, xi, theta, mNext, a, count);
    xNext = (a+xi*lam)/(xi*lam+1)*yNext - (a-1)/(xi*lam+1)*y;
    v = (1/lam + xi/a)*(tx-yNext)+grad_f_s(yNext)-grad_f_s(tx);

    % Check for early termination.
    if (norm_fn(v) <= opt_tol)
      break
    end
    
    % Update.
    y = yNext;
    A = ANext;
    x = xNext;
    iter = iter + 1;
   
  end
  
  % -----------------------------------------------------------------------
  %% POST-PROCESSING
  % -----------------------------------------------------------------------  
  model.A = A;
  model.x = x;
  model.y = y;
  model.v = v;
  history.m = max(m,abs(mNext));
  history.M = max(M,abs(MNext));
  history.runtime = toc(t_start);

  % Count backtracking iterations.
  history.iter = count;
  
end