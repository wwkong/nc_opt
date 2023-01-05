% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [oracle, params] = test_fn_penalty_mc_01(data_matrix, alpha, beta, theta, mu, seed)
% Generates the needed functions for the penalized matrix completion
% problem. Uses the penalty alpha * ||z||^2 / 2.
% 
% Arguments:
% 
%   data_matrix (double): 2D data matrix for the problem instance. 
% 
%   beta (double): One of the objective function's hyperparameters.
% 
%   theta (double): One of the objective function's hyperparameters.
% 
%   mu (double): One of the objective function's hyperparameters. 
% 
%   seed (int): The number used to seed MATLAB's random number generator. 
% 
% Returns:
%
%   A pair consisting of an Oracle and a struct. The oracle is first-order oracle underyling the optimization problem and the 
%   struct contains the relevant hyperparameters of the problem. 
% 

  % Set the generator.
  rng(seed);

  % Initialize.
  [rId, cId] = find(data_matrix);
  [dim_m, dim_n] = size(data_matrix);
  k = nnz(data_matrix);
  P = zeros(dim_m, dim_n);
  for i=1:k
      P(rId(i), cId(i)) = 1;
  end
  
  % Set the topology (Euclidean).
  prod_fn = @(a,b) sum(dot(a, b));
  norm_fn = @(a) norm(a, 'fro');

  % Other params
  rho = 1 / theta;
  
  % Output params
  params.prod_fn = prod_fn;
  params.norm_fn = norm_fn;
  params.m = 2 * rho * mu;
  params.M = 1 + alpha;
  params.x0 = ones(dim_m, dim_n)  * mean(data_matrix, "all");
  
  % Create the oracle using a wrapper function.
  function oracle_struct = oracle_eval_fn(x)

    % Parse params
    size_data = size(data_matrix);
    min_rank = min(size_data);

    % Computed params
    kappa0 = beta;
    mu_bar = mu * kappa0;

    % Subroutines (MCP)
    kappa = @(x) (x <= beta * theta) .* (beta * x - x .* x / (2 * theta)) + ...
                 (x > beta * theta) .* (theta * beta * beta / 2);
    kappa_prime = @(x) (x <= beta * theta) .* (beta - x ./ theta) + ...
                       (x > beta * theta) .* 0.0;

    % Shared computations here
    [U, S, V] = svd(x, 'econ');
    s = diag(S);
    s_prox = @(lam) prox_l1(s, lam * mu_bar);

    % Create the standard oracle constructs
    oracle_struct.f_s = @() 1/2 * norm_fn(P .* (x - data_matrix)) ^ 2 + alpha * norm_fn(x) ^ 2 / 2 + mu * sum(kappa(s) - kappa0 * s);
    oracle_struct.f_n = @() mu_bar * sum(s);
    oracle_struct.grad_f_s = @() P .* (x - data_matrix) + alpha * x + U * bsxfun(@times,  mu * (kappa_prime(s) - kappa0), V(:, 1:min_rank)'); 
    oracle_struct.prox_f_n = @(lam) U * bsxfun(@times, s_prox(lam), V(:, 1:min_rank)'); 
  end
  oracle = Oracle(@oracle_eval_fn);

end