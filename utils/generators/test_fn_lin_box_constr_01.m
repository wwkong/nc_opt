% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [oracle, params] = test_fn_lin_box_constr_01(M, m, seed, dimM, dimN)
% Generator of a test suite of linearly constrained nonconvex QP functions.
%
% Arguments:
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
  A = rand(dimM, dimN);
  
  [U, ~] = qr(rand(dimN));
  v = -m + (M + m) * rand(dimN, 1);
  v(1) = -m;
  v(end) = M;
  Q = U * diag(v) * U';
  Q = (Q + Q') / 2;
  d = rand([dimN, 1]);
   
  % Set the topology (Euclidean)
  prod_fn = @(a,b) sum(dot(a, b));
  norm_fn = @(a) norm(a, 'fro');
  
  % Compute the b vector;
  b = A * rand(dimN, 1) * 5;
  
  % Compute the norm of A and other factors.
  Hp = A' * A;
  Hp = (Hp + Hp') / 2;
  norm_A = sqrt(eigs(Hp, 1, 'la')); % same as lamMax(A'*A)
  
  % Constraint map methods.
  params.constr_fn = @(z) A * z - b;
  params.grad_constr_fn = @(z) A';
  params.lin_constr_fn = params.constr_fn;
  params.lin_grad_constr_fn = params.grad_constr_fn;
  params.nonlin_constr_fn = @(z) 0;
  params.nonlin_grad_constr_fn = @(z) zeros(size(z));
  params.set_projector = @(z) zeros(size(b));
  params.K_constr = norm_A;
  
  % Basic output params.
  params.prod_fn = prod_fn;
  params.norm_fn = norm_fn;
  params.M = max(eigs(Q, dimN));
  params.m = -min(eigs(Q, dimN));
  x0_base = rand(dimN, 1) * 5;
  params.x0 = x0_base;
  
  % Box params
  params.box_upper = 5;
  params.box_lower = 0;
  params.is_box = true;
  
  % Extra params
  params.Q = Q;
  params.d = d;
  params.A = A;
  params.b = b;  
  
  % Special params for individual constraints.
  params.K_constr_vec = full(sqrt(sum(A .^ 2, 2)));
  params.L_constr_vec = zeros(dimM, 1);
  params.m_constr_vec = zeros(dimM, 1);
  
  % Oracle construction
  f_s = @(x) (x - d)' * Q * (x - d) / 2;
  f_n = @(x) 0;
  grad_f_s = @(x) Q * (x - d);
  prox_f_n = @(x, lam) min(max(x, 0), 5);
  oracle = Oracle(f_s, f_n, grad_f_s, prox_f_n);
 
end