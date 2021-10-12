% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [oracle, params] = test_fn_unconstr_01(N, M, m, seed, dimM, dimN)
% Generator of a test suite of unconstrained nonconvex QP functions.
%
% Arguments:
%  
%   N (int): One of the objective function's hyperparameters.
%
%   dimM (int): One of the objective function's hyperparameters.
%
%   dimN (int): One of the objective function's hyperparameters.
% 
%   M (double): The target upper curvature of the objective function.
% 
%   m (double): The target lower curvature of the objective function.
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
  D = sparse(1:dimN, 1:dimN, randi([1, N], 1, dimN), dimN, dimN);
  A = rand([dimM, dimN]);
  B = rand([dimN, dimN]);
  b = rand([dimM, 1]);
  Hn = B' * (D' * D) * B;
  Hp = A' * A;
  
  % Smooth rounding errors
  Hn = (Hn + Hn') / 2;
  Hp = (Hp + Hp') / 2;  
  [tau, xi] = eigen_bisection(M, m, A, D * B);
  
  % Set the topology (Euclidean)
  prod_fn = @(a,b) sum(dot(a, b));
  norm_fn = @(a) norm(a, 'fro');
  
  % Output params
  params.prod_fn = prod_fn;
  params.norm_fn = norm_fn;
  params.M = eigs(-xi * Hn + tau * Hp, 1, 'la');
  params.m = -eigs(-xi * Hn + tau * Hp, 1, 'sa');
  params.x0 = ones(dimN, 1) / dimN;
  
  % Create the Oracle object
  f_s = @(x) -xi / 2 * norm_fn(D * B * x) ^ 2 + tau / 2 * norm_fn(A * x - b) ^ 2;
  f_n = @(x) 0;
  grad_f_s = @(x) -xi * B' * (D' * D) * B * x + tau * (A' * A * x - A' * b); 
  prox_f_n = @(x, lam) sm_proj(x, 1); 
  oracle = Oracle(f_s, f_n, grad_f_s, prox_f_n);
  
end