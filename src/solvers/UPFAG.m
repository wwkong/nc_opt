%{

DESCRIPTION
-----------
The unified problem parameter free accelerated gradient (UPFAG) method 
with full Barzilai-Borwein (BB) type stepsizes from the paper:

"Generalized uniformly optimal methods for nonlinear programming", Journal 
of Scientific Computing.

NOTES
-----
* This is a version of the file provided by Ghadimi, Lan, and Zhang, which 
  was heavily edited to conform with the CompModel framework.

FILE DATA
---------
Last Modified: 
  August 2, 2020
Coders: 
  Saeed Ghadimi, Guanghui Lan, Hongchao Zhang, Weiwei Kong

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

function [model, history] = UPFAG(oracle, params)
 
  % Timer start
  t_start = tic;

  % Main params
  z0 = params.x0;
  M_bar = params.L; % Lipschitz constant
  M = 1 * M_bar;
  lambda0 = 1 / M;
  beta0 = 1 / M;
  gamma_1 = 0.4;
  gamma_2 = 0.4;
  % gamma_3 = 1 see (2.9)
  delta = 10e-3;
  sigma = 1.e-10;
  norm_fn = params.norm_fn;
  prod_fn = params.prod_fn;

  % Initialize
  k = 0;
  iter = 0;
  cvx_iter = 0;
  nocvx_iter = 0;
  x_k = z0;
  x_ag = z0;
  x_agold = x_ag;
  sum_lambda_k = 0;

  % Ease of reading
  [f_s, f_n, grad_f_s, prox_f_n] = oracle.decompose();
  f = @(x) f_s(x) + f_n(x);
  
  % Solver params.
  opt_tol = params.opt_tol;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;

  while true
    
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Main algorithm
    line_search_cvx = 0;
    itr_cvx = 0; % tau_1_k
    if k == 0
      lambda_bb = max(lambda0, sigma) ;        
    else
      sk = x_ag_tilde - x_md;
      yk = grad_f_s(x_ag_tilde) - grad_f_s(x_md);
      sty = prod_fn(sk, yk);
      if sty > 0.
        lambda_bb = max(norm_fn(sk) ^ 2 / sty, sigma);
      else
        lambda_bb = max(lambda0, sigma);
      end
    end  
    
    % Line search for convex
    while line_search_cvx == 0
      % partial line search
      eta_k= lambda_bb * gamma_1 ^ itr_cvx;
      lambda_k = ...
        eta_k * (1 + sqrt(1 + 4 * sum_lambda_k / eta_k)) / 2; % (2.2)
      alpha_k = lambda_k / (lambda_k + sum_lambda_k);                
      x_md = (1 - alpha_k) * x_ag + alpha_k * x_k; % (2.3)
      x_k = prox_f_n(x_k - grad_f_s(x_md) * lambda_k, lambda_k); % (2.4)
      x_ag_tilde = (1 - alpha_k) * x_ag + alpha_k * x_k; % (2.5)
      if (f(x_ag_tilde) <= ...
          f(x_md) + prod_fn(grad_f_s(x_md), x_ag_tilde - x_md) + ...
          norm_fn(x_ag_tilde - x_md) ^ 2 / (2 * alpha_k * lambda_k) + ...
          delta * alpha_k) % (2.6)
        line_search_cvx = 1;
      end
      itr_cvx = itr_cvx + 1;
    end
    sum_lambda_k = sum_lambda_k + lambda_k; % (2.7)
    line_search_nocvx = 0;
    itr_nocvx = 0; % tau_2_k
    if k == 0
      beta_bb = max(beta0, sigma);
    else
      sk = x_ag - x_agold;
      yk = grad_f_s(x_ag) - grad_f_s(x_agold);
      sty = prod_fn(sk,yk);
      if sty > 0.
        beta_bb = max(norm_fn(sk) ^ 2 / sty, sigma);
      else
        beta_bb = max(beta0, sigma);
      end
    end
    x_agold = x_ag ;
    
    % Line search for nonconvex
    while line_search_nocvx == 0
      % partial line search
      beta_k = beta_bb * gamma_2 ^ itr_nocvx; % (2.8)
      x_ag_bar = ...
        prox_f_n(x_ag - grad_f_s(x_ag) * beta_k, beta_k); % (2.10)
      if (f(x_ag_bar) <= ...
          f(x_ag) - norm_fn(x_ag_bar - x_ag) ^ 2 / (2 * beta_k) + ...
            1 / (k + 1)) % gamma_3 = 1 % (2.9)
        line_search_nocvx = 1;
      end
      itr_nocvx = itr_nocvx + 1;
    end
    v = (x_ag - x_ag_bar) / beta_k + grad_f_s(x_ag) - grad_f_s(x_ag_bar);
    [~, min_idx] = min([f(x_ag), f(x_ag_bar), f(x_ag_tilde)]);
    if (min_idx == 2)
      x_ag = x_ag_bar;
    elseif (min_idx == 3) % (2.11)
      x_ag = x_ag_tilde;
    else
      % do nothing
    end
    
    % Check for termination.
    if (norm_fn(v) <= opt_tol)
      break;
    end
    
    % Update iterates.
    k = k + 1;
    iter = iter + itr_cvx + itr_nocvx;
    cvx_iter = cvx_iter + itr_cvx;
    nocvx_iter = nocvx_iter + itr_nocvx;
   
  end
  
  % Post-processing
  model.v = v;
  model.x = x_ag;
  history.k = k;
  history.iter = iter;
  history.cvx_iter = cvx_iter;
  history.nocvx_iter = nocvx_iter;
  history.runtime = toc(t_start);
  
end