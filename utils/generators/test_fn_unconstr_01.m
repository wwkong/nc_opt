%{

DESCRIPTION
-----------
Generator of a test suite of unconstrained nonconvex QP functions.

NOTES
-----
** xi and tau are chosen so that upper and lower curvatures are M and -m 
   respectively
** Entries of A, B, and b are drawn randomly from a U(0, 1) distribution
** D is a diagonal matrix with integer elements chosen at random 
  from [1, N]
** Function is -xi / 2 * ||D * B * x|| ^ 2 + tau / 2 * ||A * x - b|| ^ 2 
** Gradient is -xi * B' * (D' * D) * B * x + tau * (A' * A * x - A' * b)

FILE DATA
---------
Last Modified: 
  August 2, 2020
Coders: 
  Weiwei Kong

INPUT
-----
(N, M, m, seed, dimM, dimN):
  Input parameters as described in the function description.

OUTPUT
------
oracle:
  An Oracle object.
params:
  A struct containing input parameters for this function.

%}

function [oracle, params] = test_fn_unconstr_01(N, M, m, seed, dimM, dimN)

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
  f_s = @(x) ...
    -xi / 2 * norm_fn(D * B * x) ^ 2 + tau / 2 * norm_fn(A * x - b) ^ 2;
  f_n = @(x) 0;
  grad_f_s = @(x) ...
    -xi * B' * (D' * D) * B * x + tau * (A' * A * x - A' * b); 
  prox_f_n = @(x, lam) sm_proj(x, 1); 
  oracle = Oracle(f_s, f_n, grad_f_s, prox_f_n);
  
end