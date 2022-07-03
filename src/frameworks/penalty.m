% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [model, history] = penalty(solver, oracle, params)
% A quadratic penalty-based framework for solving a nonconvex composite optimization model with linear set constraints, i.e.
% $g(x)=Ax$ where $A$ is a linear operator.
% 
% Note:
% 
%   Based on the paper:
%
%   Kong, W., Melo, J. G., & Monteiro, R. D. (2020). An efficient adaptive accelerated inexact proximal point method for solving
%   linearly constrained nonconvex composite problems. *Computational Optimization and Applications, 76*\(2), 305-346. 
%
% Arguments:
% 
%   solver (function handle): A solver for unconstrained composite optimization.
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
  feas_tol = params.feas_tol;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  o_oracle = oracle;
  
  % Initialize helper functions.
  penalty_n = @(x) 0;
  penalty_prox = @(x, lam) x;
  % Value of the feasibility function: 
  %   feas(x) = |constr_fn(x) - set_projector(constr_fn(x))|. 
  function [feas, feas_vec] = feas_fn(x)
    constr_fn_vec = params.constr_fn(x);
    feas_vec = constr_fn_vec - params.set_projector(constr_fn_vec);
    feas = params.norm_fn(feas_vec);
  end
  % Value of the function (1 / 2) * feas(x) ^ 2.
  function sqr_feas = sqr_feas_fn(x)
    sqr_feas = (1 / 2) * feas_fn(x) ^ 2;
  end
  % Gradient of the function (1 / 2) * feas(x) ^ 2.
  function grad_sqr_feas = grad_sqr_feas_fn(x)
    [~, feas_vec] = feas_fn(x);
    grad_constr_fn = params.grad_constr_fn;
    % If the gradient function has a single argument, assume that the gradient at a point is a constant tensor.
    if nargin(grad_constr_fn) == 1
      grad_sqr_feas = tsr_mult(grad_constr_fn(x), feas_vec, 'dual');
    % Else, assume that the gradient is a bifunction; the first argument is the point of evaluation, and the second one is what
    % the gradient operator acts on.
    elseif nargin(grad_constr_fn) == 2
      grad_sqr_feas = grad_constr_fn(x, feas_vec);
    else
      error('Unknown function prototype for the gradient of the constraint function');
    end
  end

  % Fill in OPTIONAL input params.
  params = set_default_params(params);

  % Initialize history parameters.
  history = struct();
  if params.i_logging
    history.function_values = [];
    history.iteration_values = [];
    history.time_values = [];
  end
  
  % DEBUG ONLY
  if params.i_debug
    history.inner_iter = [];
    history.L_est = [];
    history.norm_v = [];
    history.norm_w = [];
    history.c = [];
  end

  % Initialize solver parameters.
  iter = 0; % Set to 0 because it calls an inner subroutine.
  stage = 1;
  solver_params = params;
  i_reset_prox_center = params.i_reset_prox_center;
  c_prev = params.c0;
  c = c_prev;
  outer_iter = 0;
  cycle_outer_iter = 0;
  sum_wc = 0;

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
    
    % Create the penalty oracle object.
    penalty_s = @(x) c * sqr_feas_fn(x);
    grad_penalty = @(x) c * grad_sqr_feas_fn(x);
    penalty_oracle = Oracle(penalty_s, penalty_n, grad_penalty, penalty_prox);
    solver_oracle = copy(o_oracle);
    solver_oracle.add_smooth_oracle(penalty_oracle)
    
    % Modify the termination of the termination funciton of the subproblem solver if needed.
    if (~isempty(params.termination_fn))
      solver_params.termination_fn = @(x) wrap_termination(params, x, c);
    end
    
    % Update curvatures and time limit, call the solver, and update iteration count.
    solver_params.time_limit = max([0, time_limit - toc(t_start)]);
    if (stage == 1)
      solver_params.M = params.M_start;
    else
      solver_params.M = solver_params.M + (c - c_prev) * params.K_constr ^ 2;
    end
    [solver_model, solver_history] = solver(solver_oracle, solver_params);
    iter = iter + solver_history.iter;
    if (isfield(solver_history, 'outer_iter'))
      outer_iter = outer_iter + solver_history.outer_iter;
    end    
    
    % Update history.
    if params.i_logging
      if isfield(solver_history, 'function_values')
        history.function_values = [history.function_values, solver_history.function_values];
      end
      if isfield(solver_history, 'iteration_values')
        if isempty(history.iteration_values)
          history.iteration_values = solver_history.iteration_values;
        else
          history.iteration_values = [history.iteration_values, history.iteration_values(end) + 1 + solver_history.iteration_values];
        end
      end
      if isfield(solver_history, 'time_values')
        history.time_values = [history.time_values, solver_history.time_values];
      end
    end
    if (isfield(solver_history, 'outer_iter'))
      if (isfield(history, 'outer_iter'))
        history.outer_iter = history.outer_iter + solver_history.outer_iter;
      else
        history.outer_iter = solver_history.outer_iter;
      end
    end
    
    % DEBUG ONLY
    if params.i_debug      
      history.inner_iter = [history.inner_iter; solver_history.inner_iter];
      history.L_est = [history.L_est; solver_history.L_est];
      history.norm_v = [history.norm_v; solver_history.norm_v];
      w_vec = NaN(length(solver_history.inner_iter), 1);
      w_vec(end) = feas_fn(solver_model.x);
      history.norm_w = [history.norm_w; w_vec];
      history.c = [history.c; c * ones(length(solver_history.inner_iter), 1)];
    end
    
    % Check for termination.
    feas = feas_fn(solver_model.x);
    if (isempty(params.termination_fn) && ~params.check_all_terminations)
      if (feas <= feas_tol)
        if (isfield(solver_history, 'outer_iter'))
          sum_wc = sum_wc + (outer_iter - cycle_outer_iter + 1) * c;
        end
        break;
      end
    else
      if wrap_termination(params, solver_model.x, c)
        if (isfield(solver_history, 'outer_iter'))
          sum_wc = sum_wc + (outer_iter - cycle_outer_iter + 1) * c;
        end
        break;
      end
    end
    
    % Apply warm-start onto the initial point fed into the next iteration.
    if (~i_reset_prox_center)
      solver_params.x0 = solver_model.x;
    end

    % Update iterates.
    if (isfield(solver_history, 'outer_iter'))
      sum_wc = sum_wc + (outer_iter - cycle_outer_iter + 1) * c;
      cycle_outer_iter = outer_iter + 1;
    end
    c_prev = c;
    c = params.penalty_multiplier * c;
    stage = stage + 1;
    
  end
  
  % Prepare to output
  model = solver_model;
  if (isempty(params.termination_fn))
    [~, feas_vec] = feas_fn(solver_model.x);
    model.w = feas_vec;
  else
    [~, model.v, model.w] = wrap_termination(params, solver_model.x, c);
  end
  history.c0 = params.c0;
  history.c = c;
  history.iter = iter;
  history.stage = stage;
  history.runtime = toc(t_start);
  if (isfield(solver_history, 'outer_iter'))
    history.wavg_c = sum_wc / outer_iter;
  end
  
end

% Utility functions for custom termination conditions.
function [stop, w, q] = wrap_termination(params, x, c)
  % Based on the R-QP-AIPP paper.
  Ax = params.constr_fn(x);
  p = c * (Ax - params.set_projector(Ax));
  [stop, w, q] = params.termination_fn(x, p);
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
  if (~isfield(params, 'i_reset_prox_center')) 
    params.i_reset_prox_center = false;
  end
  if (~isfield(params, 'penalty_multiplier'))
    params.penalty_multiplier = 2;
  end
  if (~isfield(params, 'termination_fn'))
    params.termination_fn = [];
  end
  if (~isfield(params, 'check_all_terminations'))
    params.check_all_terminations = false;
  end
  if (~isfield(params, 'c0'))
    params.c0 = max([MIN_PENALTY_CONST, params.M / params.K_constr ^ 2]);
  end
  if (~isfield(params, 'M_start'))
    params.M_start = params.M + params.c0 * params.K_constr ^ 2;
  end
  
end
