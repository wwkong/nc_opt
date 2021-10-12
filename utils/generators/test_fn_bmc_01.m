% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [oracle, params] = test_fn_bmc_01(data_name, beta, theta, mu, seed)
% Generates the needed functions for the regularized matrix completion problem under box constraints. Requires a data matrix with 
% the variable name 'data' that is read from a .mat file with name given by data_name.
% 
% Arguments:
% 
%   data_name (character vector): Name of the .mat dataset used for the problem. 
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
  load(data_name, 'data');
  [rId, cId] = find(data);
  [dim_m, dim_n] = size(data);
  k = nnz(data);
  P = zeros(dim_m, dim_n);
  for i=1:k
      P(rId(i), cId(i)) = 1;
  end
  
  % Set the topology (Euclidean).
  prod_fn = @(a,b) sum(dot(a, b));
  norm_fn = @(a) norm(a, 'fro');

  % Other params
  rho = beta / theta ^ 2;
  data_upper = 5.0;
  data_lower = 0.0;
  
  % Constraint
  params.constr_fn = @(Z) Z;
  params.grad_constr_fn = @(Z, Delta) Delta;
  params.K_constr = 1;
  params.set_projector = @(Z) max(min(Z, data_upper), data_lower);
  
  % Output params
  params.prod_fn = prod_fn;
  params.norm_fn = norm_fn;
  params.m = 2 * rho * mu;
  params.M = 1;
  params.x0 = zeros(dim_m, dim_n);
  
  % Create the oracle using a wrapper function.
  function oracle_struct = oracle_eval_fn(x)

    % Parse params
    size_data = size(data);
    min_rank = min(size_data);

    % Computed params
    kappa0 = beta / theta;
    mu_bar = mu * kappa0;

    % Subroutines
    kappa = @(x) beta * log(1 + x / theta);
    kappa_prime = @(x) beta ./ (theta + x);

    % Shared computations here
    [U, S, V] = svd(x, 'econ');
    s = diag(S);
    s_prox = @(lam) prox_l1(s, lam * mu_bar);

    % Create the standard oracle constructs
    oracle_struct.f_s = @() 1/2 * norm_fn(P .* (x - data)) ^ 2 + mu * sum(kappa(s) - kappa0 * s);
    oracle_struct.f_n = @() mu_bar * sum(s);
    oracle_struct.grad_f_s = @() P .* (x - data) + U * bsxfun(@times,  mu * (kappa_prime(s) - kappa0), V(:, 1:min_rank)'); 
    oracle_struct.prox_f_n = @(lam) U * bsxfun(@times, s_prox(lam), V(:, 1:min_rank)'); 

    % Create the Tier II special oracle constructs
    oracle_struct.f_s_at_prox_f_n = @(lam) 1/2 * norm_fn(P .* (x - data)) ^ 2 + mu * sum(kappa(s_prox(lam)) - kappa0 * s_prox(lam)); 
    oracle_struct.f_n_at_prox_f_n = @(lam) mu_bar * sum(s_prox(lam));
    oracle_struct.grad_f_s_at_prox_f_n = @(lam) ...
      P .* (x - data) + U * bsxfun(@times,  mu * (kappa_prime(s_prox(lam)) - kappa0), V(:, 1:min_rank)');
  end
  oracle = Oracle(@oracle_eval_fn);

end