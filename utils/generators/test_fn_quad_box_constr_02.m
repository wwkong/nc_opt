% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [oracle, params] = test_fn_quad_box_constr_02(M, m, seed, dimM, dimN, x_l, x_u)
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

  % Initialize globals.
  rng(seed);

  % Initialize Q{1:dimM} and d{1:dimM}.
  for i=1:dimM
    [U, ~] = qr(rand(dimN));
    v = log(M / m) / 3 * rand(dimN, 1);
    Q{i} = U * diag(v) * U';
    Q{i} = (Q{i} + Q{i}') / 2;
    c{i} = rand([dimN, 1]);
    d{i} = -20 - 10 * rand();
  end
  
  % Initialize Q{1:dimM} and d{1:dimM}.
  [U, ~] = qr(rand(dimN));
  v = -m + (M + m) * rand(dimN, 1);
  v(1) = -m;
  v(end) = M;
  Q{dimM + 1} = U * diag(v) * U';
  Q{dimM + 1} = (Q{dimM + 1} + Q{dimM + 1}') / 2;
  c{dimM + 1} = rand([dimN, 1]);
  d{dimM + 1} = rand();
   
  % Set the topology (Euclidean)
  prod_fn = @(a,b) sum(dot(a, b));
  norm_fn = @(a) norm(a, 'fro');
  
  % Constraint utility functions.
  function cx = constr_sub_fn(z)
    cx = zeros(dimM, 1);
    for j=1:dimM
      cx(j) = z' * Q{j} * z / 2 + c{j}' * z + d{j};
    end
  end
  function gcx = grad_constr_sub_fn(z, delta)
    gcx = zeros(dimN, 1);
    for j=1:dimM
      gcx = gcx + delta(j) * (Q{j} * z + c{j});
    end
  end
  
  % Note that the gradient of the constraint function is a matrix, whose i-th row is (Q{i} z + c{i})'. The Frobenius norm of this
  % row, for x in [x_l, x_u], is bounded by
  %
  %   (|Q{i}| * |ones(dimM, 1) * (x_u - x_l)| + |c{i}|)
  %
  % and this is just the Lipschitz constant of the i-th constraint.
  K_constr_vec = zeros(dimM, 1);
  norm_Xbd = norm(ones(dimN, 1) * (x_u - x_l));
  for i=1:dimM
    K_constr_vec(i) = norm(Q{i}) * norm_Xbd + norm(c{i});
  end
  
  % The Lipschitz constant for the i-th constraint is just |Q{i}|.
  L_constr_vec = zeros(dimM, 1);
  for i=1:dimM
    L_constr_vec(i) = norm(Q{i});
  end

  % Constraint map methods.
  params.constr_fn = @(z) constr_sub_fn(z);
  params.grad_constr_fn = @(z, delta) grad_constr_sub_fn(z, delta);
  params.set_projector = @(y) max(y, 0.0);
  params.dual_cone_projector = @(y) max(y, 0.0);
  params.K_constr = norm(K_constr_vec); % Aggregated by the Euclidean norm.
  params.L_constr = norm(L_constr_vec); % Aggregated by the Euclidean norm.
  
  % Basic output params.
  params.prod_fn = prod_fn;
  params.norm_fn = norm_fn;
  params.M = M;
  params.m = m;
  x0_base = rand(dimN, 1) * (x_u - x_l) + x_l;
  params.x0 = x0_base;
  
  % Extra params
  params.x_u = x_u;
  params.x_l = x_l;
  params.Q = Q;
  params.c = c;
  params.d = d;
  
  % Special params for individual constraints.
  params.K_constr_vec = K_constr_vec;
  params.L_constr_vec = L_constr_vec;
  params.m_constr_vec = zeros(dimM, 1);
  
  % Oracle construction
  f_s = @(x) x' * Q{dimM + 1} * x / 2 + c{dimM + 1}' * x + d{dimM + 1};
  f_n = @(x) 0;
  grad_f_s = @(x) Q{dimM + 1} * x + c{dimM + 1};
  prox_f_n = @(x, lam) box_proj(x, x_l, x_u);
  oracle = Oracle(f_s, f_n, grad_f_s, prox_f_n);
 
end