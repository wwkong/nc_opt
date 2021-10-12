% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [oracle, params] = test_fn_lin_constr_01(N, M, m, seed, dimM, dimN)
% Generator of a test suite of linearly constrained nonconvex QP functions.
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

  % Initialize.
  rng(seed);
  D = sparse(1:dimN, 1:dimN, randi([1, N], 1, dimN), dimN, dimN);
  C = rand(dimM, dimN);
  B = rand(dimN, dimN);
  A = rand(dimM, dimN);
  d = rand([dimM, 1]);
  
   
  % Choose (xi, tau).
  [tau, xi, ~, ~] = eigen_bisection(M, m, C, D * B);
  
  % Compute the norm of A and other factors.
  Hn = B' * (D' * D) * B;
  Hp = A' * A;
  Hn = (Hn + Hn') / 2;
  Hp = (Hp + Hp') / 2;
  norm_A = sqrt(eigs(Hp, 1, 'la')); % same as lamMax(A'*A)
   
  % Set the topology (Euclidean)
  prod_fn = @(a,b) sum(dot(a, b));
  norm_fn = @(a) norm(a, 'fro');
  
  % Compute the b vector;
  b = A * ones(dimN, 1) / dimN;
  
  % Constraint map methods.
  params.constr_fn = @(z) A * z;
  params.grad_constr_fn = @(z) A';
  params.set_projector = @(z) b;
  params.K_constr = norm_A;
  
  % Basic output params.
  params.prod_fn = prod_fn;
  params.norm_fn = norm_fn;
  params.M = eigs(-xi * Hn + tau * Hp, 1, 'la');
  params.m = -eigs(-xi * Hn + tau * Hp, 1, 'sa');
  x0_base = rand(dimN, 1);
  params.x0 = x0_base / sum(x0_base);
  
  % Oracle construction
  f_s = @(x) -xi / 2 * norm_fn(D * B * x) ^ 2 + tau / 2 * norm_fn(C * x - d) ^ 2;
  f_n = @(x) 0;
  grad_f_s = @(x) -xi * B' * (D' * D) * B * x + tau * C' * (C * x -  d);
  prox_f_n = @(x, lam) sm_proj(x, 1);
  oracle = Oracle(f_s, f_n, grad_f_s, prox_f_n);
 
end