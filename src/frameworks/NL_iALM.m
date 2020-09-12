%{

FILE DATA
---------
Last Modified: 
  September 12, 2020
Coders: 
  Weiwei Kong

%}


%% Testing APG and iPPM
f_s = @(x) (1 / 2) * (x - 1) ^ 2;
f_n = @(x) 0;
grad_f_s = @(x) x - 1;
prox_f_n = @(x, lam) x;
t_oracle = Oracle(f_s, f_n, grad_f_s, prox_f_n);

t_params = struct();
% t_params.L_g = 100;
t_params.L_phi = 100;
% t_params.mu = 1 / 2;
t_params.rho = 1 / 2;
t_params.iter_limit = Inf;
t_params.time_limit = Inf;
t_params.norm_fn = @(x) norm(x);
t_params.t_start = tic;
t_params.tol = 1e-2;
t_params.xm1_bar = 0;
t_params.x0 = 0;
[model, history] = iPPM(t_params, t_oracle);

function [model, history] = iPPM(params, oracle)
% Inexact Proximal Point Method (iPPM)

  % Parse algorithm inputs.
  rho = params.rho;
  L_phi = params.L_phi;
  tol = params.tol;
  
  % Solver params.
  norm_fn = params.norm_fn;
  t_start = params.t_start;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  
  % Initialize the invariant APG params.
  apg_params = params;
  apg_params.mu = rho;
  apg_params.L_g = L_phi + 2 * rho;
  apg_params.tol = tol / 4;
  quad_f_n = @(x) 0;
  quad_prox_f_n = @(x, lam) x;
  
  % Initialize other iPPM parameters.
  x0 = params.x0;
  
  % Iterate.
  iter = 0; % Set to 0 because it calls an inner subroutine.
  while true
    
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Create the APG oracle.
    apg_oracle = copy(oracle);
    quad_f_s = @(x) rho * norm_fn(x - x0) ^ 2;
    grad_quad_f_s = @(x) 2 * rho * (x - x0);
    quad_oracle = Oracle(quad_f_s, quad_f_n, grad_quad_f_s, quad_prox_f_n);
    apg_oracle.add_smooth_oracle(quad_oracle);
    
    % Set the other APG params (custom implementation here!)
    apg_params.t_start = t_start;
    apg_params.xm1_bar = x0;
    
    % Call the APG and parse the outputs.
    [apg_model, apg_history] = APG(apg_params, apg_oracle);
    x = apg_model.x;
    iter = iter + apg_history.iter;
    
    % Check for termination.
    disp(norm_fn(x - x0));
    if (2 * rho * norm_fn(x - x0) <= tol / 2)
      break;
    end
    
    % Update iterates
    x0 = x;
    
  end
  
  % Get ready to output
  model.x = x;
  history.iter = iter;

end

function [model, history] = APG(params, oracle)
% Accelerated Proximal Gradient (APG) Method

  % Parse algorithm inputs.
  mu = params.mu;
  L_g = params.L_g;
  tol = params.tol;
  
  % Solver params.
  norm_fn = params.norm_fn;
  t_start = params.t_start;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;

  % Set up some oracles.
  o_at_x0_bar = copy(oracle);
  o_at_x0 = copy(oracle);

  % Compute alpha, x0, x_bar_0.
  alpha = sqrt(mu / L_g);
  xm1_bar = params.xm1_bar;
  o_at_x0_bar.eval(xm1_bar);
  o_at_x0.eval(xm1_bar - o_at_x0_bar.grad_f_s() / L_g); 
  x0 = o_at_x0.prox_f_n(1 / L_g);
  x0_bar = x0;
  
  % Iterate.
  iter = 2; % Each call does at least two prox evaluations.
  while true
    
    % If time is up, pre-maturely exit.
    if (toc(t_start) > time_limit)
      break;
    end
    
    % If there are too many iterations performed, pre-maturely exit.
    if (iter >= iter_limit)
      break;
    end
    
    % Compute x and x_bar.
    o_at_x0_bar.eval(x0_bar);
    o_at_x0.eval(x0_bar - o_at_x0_bar.grad_f_s() / L_g);
    x = o_at_x0.prox_f_n(1 / L_g);
    x_bar = x + (1 - alpha) / (1 + alpha) * (x - x0);

    % Check for termination.
    if (norm_fn(x - x_bar) <= tol / (2 * L_g))
      break;
    end
    
    % Update iterates
    iter = iter + 1;
    x0 = x;
    x0_bar = x_bar;
    
  end
  
  % Get ready to output
  model.x = x;
  history.iter = iter;
  
end