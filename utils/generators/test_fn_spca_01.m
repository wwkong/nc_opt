% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [oracle, params] = test_fn_spca_01(b, nu, p, n, s, k, seed)
% Generates the needed functions for the sparse PCA problem under MCP sparsity regularization.
%
% Note:
%   
%   Implements the first series of tests (synthetic dataset I) in:
%   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4326026/
%
% Arguments:
%  
%   b (int): One of the objective function's hyperparameters.
%
%   nu (int): One of the objective function's hyperparameters.
%
%   p (int): One of the objective function's hyperparameters.
%
%   n (int): One of the objective function's hyperparameters.
%
%   s (int): One of the objective function's hyperparameters.
%
%   k (int): One of the objective function's hyperparameters.
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

  % Compute the sample covariance matrix
  eig_cov = sparse(1:p, 1:p, [100 * ones(k, 1); ones(p - k, 1)]);
  principle_eigvec = sparse(1:s, ones(s, 1), 1 / sqrt(s) * ones(s, 1), p , 1);
  low_eigvec = randn(p, p - k);
  eigvec_cov = [repmat(principle_eigvec, 1, k), low_eigvec];
  mat_cov = eigvec_cov * eig_cov * eigvec_cov';
  mu = zeros(p, 1);
  obs = mvnrnd(mu, mat_cov, n);
  Sigma = cov(obs);

  % Huber loss
  huber_val_sse = @(x) huber_val(x, nu, b);
  huber_grad_sse = @(x) huber_grad(x, nu, b);

  % Set the topology (Euclidean)
  prod_fn = @(a,b) sum(dot(a, b));
  norm_fn = @(a) norm(a, 'fro');
  
  % Constraint map methods.
  params.constr_fn = @(Z) Z(1:p, :) - Z(p+1:2*p, :);
  params.grad_constr_fn = @(Z, Delta) [Delta; -Delta];
  params.K_constr = 2;
  params.set_projector = @(Z) sparse(p, p);
  
  % Special params for individual constraints.
  params.K_constr_vec = ones(2 * p, 1);
  params.L_constr_vec = zeros(2 * p, 1);
  params.m_constr_vec = zeros(2 * p, 1);
  
  % Basic output params.
  params.prod_fn = prod_fn;
  params.norm_fn = norm_fn;
  params.M = 0;
  params.m = 1 / b;
  params.x0 = [diag([ones(k, 1); zeros(p - k, 1)]); zeros(p, p)];
  
  % Create the oracle using a wrapper function.
  f_s = @(x) prod_fn(x(1:p, :), Sigma) + sum(sum(huber_val_sse(x((p + 1):(2 * p), :))));
  f_n = @(x) nu * sum(sum(abs(x((p + 1):(2 * p), :))));
  grad_f_s = @(x) [Sigma; huber_grad_sse(x((p + 1):(2 * p), :))];
  prox_f_n = @(x, lam) [kf_proj(x(1:p, :), k); prox_l1(x((p + 1):(2 * p), :), lam * nu)]; 
  oracle = Oracle(f_s, f_n, grad_f_s, prox_f_n);
 
end

%% Utility functions

% Huber loss function (value)
function x_val = huber_val(x, nu, b) 
  mask = abs(x) > b * nu;
  x_val = (b * nu ^ 2 / 2 - nu * abs(x)) .* mask + (- x ^ 2 / (2 * b)) .* (1 - mask);
end

% Huber loss function (gradient)
function grad_x_val = huber_grad(x, nu, b) 
  mask = abs(x) > b * nu;
  grad_x_val = (- nu * sign(x)) .* mask + (- x / b) .* (1 - mask);
end