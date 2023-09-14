% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [oracle, params] = test_fn_penalty_spca_01(data_matrix, alpha, beta, theta, mu, seed)
% Generates the needed functions for the data-driven sparse PCA problem under Laplace 
% sparsity regularization.
%
%
% Arguments:
%  
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
%   seed (int): The number used to seed MATLAB's random number generator. 
% 
% Returns:
%
%   A pair consisting of an Oracle and a struct. The oracle is first-order oracle underyling the optimization problem and the 
%   struct contains the relevant hyperparameters of the problem. 
% 

  % Initialize
  rng(seed);

  % Compute the data covariance matrix.
  [dim_m, dim_n] = size(data_matrix);
  if (dim_m < dim_n)
    X = data_matrix;
  else
    X = data_matrix';
  end
  XXt = X * X';

  % Set the topology (Euclidean)
  prod_fn = @(a,b) sum(dot(a, b));
  norm_fn = @(a) norm(a, 'fro');
  
  % Computed params
  kappa0 = beta / theta;
  mu_bar = mu * kappa0;
  rho = beta / theta ^ 2;
  
  % Subroutines (Laplace)
  kappa = @(t) beta * (1 - exp(-t / theta));
  kappa_prime = @(t) beta / theta .* exp(-t / theta);
  function grad = f_2_kappa_prime(z)
    nz = norm(z);
    if (nz == 0)
      grad = zeros(size(z));
    else
      grad = z .* (kappa_prime(nz) - kappa0) / nz;
    end
  end
  
  % Basic output params.
  r = min([dim_m, dim_n]);
  params.prod_fn = prod_fn;
  params.norm_fn = norm_fn;
  params.M = eigs(XXt, 1) + alpha;
  params.m = 2 * rho * mu;
  params.x0 = r * ones(r, 1);
  
  % Create the oracle using a wrapper function.
  f_s = @(z) (z' * XXt) * z / 2 + alpha * norm(z, 'fro')^2 / 2 + mu * (kappa(norm(z)) - kappa0 * norm(z));
  f_n = @(z) mu_bar * sum(abs(z));
  grad_f_s = @(z) XXt * z + alpha * z + mu * f_2_kappa_prime(z);
  prox_f_n = @(z, lam) prox_l1(z, lam * mu_bar); 
  oracle = Oracle(f_s, f_n, grad_f_s, prox_f_n);
 
end