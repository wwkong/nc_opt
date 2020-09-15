%{

FILE DATA
---------
Last Modified: 
  August 5, 2020
Coders: 
  Weiwei Kong

%}

function [model, history] = penalty(solver, oracle, params)
% A quadratic penalty-based framework for solving a nonconvex composite 
% optimization model with linear set constraints, i.e., $g(x)=Ax$ where $A$ is
% a linear operator.
% 
% Note:
% 
%   Based on the paper:
%
%   Kong, W., Melo, J. G., & Monteiro, R. D. (2020). An efficient adaptive 
%   accelerated inexact proximal point method for solving linearly constrained 
%   nonconvex composite problems. *Computational Optimization and 
%   Applications, 76*\(2), 305-346. 
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
  feas_tol = params.feas_tol;
  time_limit = params.time_limit;
  iter_limit = params.iter_limit;
  o_oracle = oracle;
  
  % Initialize helper functions.
  penalty_n = @(x) 0;
  penalty_prox = @(x, lam) x;
  % Value of the feasibility function: 
  % feas(x) = |constr_fn(x) - set_projector(constr_fn(x))|. 
  function [feas, feas_vec] = feas_fn(x)
    constr_fn_vec = params.constr_fn(x);
    feas_vec = ...
      constr_fn_vec - params.set_projector(constr_fn_vec);
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
    %  If the gradient function has a single argument, assume that the
    %  gradient at a point is a constant tensor.
    if nargin(grad_constr_fn) == 1
      grad_sqr_feas = ...
        tsr_mult(grad_constr_fn(x), feas_vec, 'dual');
    % Else, assume that the gradient is a bifunction; the first argument is
    % the point of evaluation, and the second one is what the gradient
    % operator acts on.
    elseif nargin(grad_constr_fn) == 2
      grad_sqr_feas = grad_constr_fn(x, feas_vec);
    else
      error(...
        ['Unknown function prototype for the gradient of the ', ...
         'constraint function']);
    end
  end

  % Fill in OPTIONAL input params.
  params = set_default_params(params);

  % Initialize history parameters.
  if params.i_logging
    history.function_values = [];
    history.iteration_values = [];
    history.time_values = [];
  end

  % Initialize solver parameters.
  iter = 0; % Set to 0 because it calls an inner subroutine.
  stage = 1;
  solver_params = params;
  c = max([MIN_PENALTY_CONST, params.M / params.K_constr ^ 2]);

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
    
    % Create the penalty oracle object.
    penalty_s = @(x) c * sqr_feas_fn(x);
    grad_penalty = @(x) c * grad_sqr_feas_fn(x);
    penalty_oracle = ...
      Oracle(penalty_s, penalty_n, grad_penalty, penalty_prox);
    
    % Create the main oracle and update the model.
    % NOTE: We need to explicitly call copy() because the 'Oracle' class
    % inherits from the 'handle' class.
    solver_oracle = copy(o_oracle);
    solver_oracle.add_smooth_oracle(penalty_oracle)
    
    % Update curvatures, call the solver, and update iteration count.
    solver_params.M = params.M + c * params.K_constr ^ 2;
    [solver_model, solver_history] = solver(solver_oracle, solver_params);
    iter = iter + solver_history.iter;
    
    % Update history.
    if params.i_logging
      if isfield(solver_history, 'function_values')
        history.function_values = ...
          [history.function_values, solver_history.function_values];
      end
      if isfield(solver_history, 'iteration_values')
        if isempty(history.iteration_values)
          history.iteration_values = solver_history.iteration_values;
        else
          history.iteration_values = ...
            [history.iteration_values, ...
             history.iteration_values(end) + ...
                1 + solver_history.iteration_values];
        end
      end
      if isfield(solver_history, 'time_values')
        history.time_values = ...
          [history.time_values, solver_history.time_values];
      end
    end
    
    % Check for termination.
    if (feas_fn(solver_model.x) <= feas_tol)
      break;
    end
    
    % Apply warm-start onto the initial point fed into the next iteration.
    solver_params.x0 = solver_model.x;

    % Update iterates.
    c = 2 * c;
    stage = stage + 1;
    
  end
  
  % Prepare to output
  [~, feas_vec] = feas_fn(solver_model.x);
  model = solver_model;
  model.w = feas_vec;
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

end