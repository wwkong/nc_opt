%{

DESCRIPTION
-----------

Generator of a test suite of linearly constrained nonconvex QP functions.

NOTES
-------
** ENSURES feasibility with a particular b vector
** Makes the vector b dense via b = A(E) where E is the diagonal
   matrix whose diagonal is a vector of all ones
** xi and tau are chosen so that upper and lower curvatures are M and -m 
   respectively
** Entries of Ai and Bj are drawn randomly from a U(0,1) distribution
** A is a dimM-by-(dimN * dimN) sized matrix
** b is defined as A(E) where E is a matrix of all ones 
** D is a diagonal matrix with integer elements chosen at random 
  from [1, N]
** Function is -xi / 2 * ||D * B * Z|| ^ 2 + tau / 2 * ||C * Z - d|| ^ 2 
** Gradient is -xi * B' * (D' * D) * B * Z + tau *  C' * (C * Z - d)
** Constraint is A_map(Z) = b;


FILE DATA
---------
Last Modified: 
  August 17, 2020
Coders: 
  Weiwei Kong

INPUT
-----
(N, Lf, seed, dimM, dimN, density):
  Input parameters as described in the function description.

OUTPUT
------
oracle:
  An oracle function (see README.md for details).
params:
  Input parameters of this function (see README.md for details).

%} 

function [oracle, params] = ...
  test_fn_lin_constr_01(N, M, m, seed, dimM, dimN, density)

  % Initialize
  rng(seed);
  E = diag(ones(dimN, 1) / dimN);
  D = sparse(1:dimN, 1:dimN, randi([1, N], 1, dimN), dimN, dimN);
  C = sprand(dimM, dimN * dimN, density);
  B = sprand(dimN, dimN * dimN, density);
  A = sprand(dimM, dimN * dimN, density);
  d = rand([dimM, 1]);
  b = lin_op(A, E);
   
  % Choosing (xi, tau)
  [tau, xi, Dfn, Z] = eigenBisection(M, m, C, D * B);
  
  % Computing norm of A
  Hp = A * A';
  Hp = (Hp + Hp') / 2;
  norm_A = sqrt(eigs(Hp, 1, 'la')); % same as lamMax(A'*A)
  
  % Constraint map methods
  params.constr_fn = @(Z) lin_op(A, Z);
  params.grad_constr_fn = @(Z) adj_op(A, Z);
  params.set_projector = @(Z) b;
  params.norm_A = norm_A;
   
  % Set the topology (Euclidean)
  prod_fn = @(a,b) sum(dot(a, b));
  norm_fn = @(a) norm(a, 'fro');
  
  % Output params
  A_map = @(Z) lin_op(A, Z);
  params.prod_fn = prod_fn;
  params.norm_fn = norm_fn;
  params.M = eigs(Dfn(xi, tau) * Z, 1, 'lr');
  params.m = -eigs(Dfn(xi, tau) * Z, 1, 'sr');
  params.z0 = init_point(dimN, A_map, b, seed);
  
  % Oracle construction
  f_s = @(x) ...
    -xi / 2 * normFn(D * lin_op(B, x)) ^ 2 + ...
    tau / 2 * normFn(lin_op(C, x) - d) ^ 2;
  f_n = @(x) 0;
  grad_f_s = @(x) ...
    -xi * adj_op(B, (D' * D) * lin_op(B, x)) + ...
    tau * (adj_op(C, lin_op(C, x)) - adj_op(C, d));
  prox_f_n = @(x, lam) sm_mat_proj(x, 1);
  oracle = Oracle(f_s, f_n, grad_f_s, prox_f_n);
 
end

%% Efficient operators used for A(X)

% Function that applies the operator A(:,:,:) to Z(:,:) and returns a
% vector that is the size of the third index of A
function AZ = lin_op(A, Z)
  [n, ~] = size(Z);
  tildeZ = reshape(Z, [n * n, 1]);
  AZ = A * tildeZ;
end

% Function that applies the adjoint of the operator A(:,:,:) to y(:) 
% and returns a matrix
function adj_Ay = adj_op(A, y)
  [~, sqrN] = size(A);
  n = sqrt(sqrN);
  ytildeA = diag(sparse(y)) * A;
  rsum_ytildeA = sum(ytildeA, 1);
  adj_Ay = reshape(rsum_ytildeA, [n, n]);
end

%% Generator of an initial point for some of the penalty problems
function x0 = init_point(dimN, AMap, b, seed)

  rng(seed);
  feasible = true;
  while (feasible)
    nVec = 3;
    vMat = rand(dimN, nVec);
    wMat = zeros(dimN, nVec);
    for j=1:nVec
      wMat(:, j) = vMat(:, j) / norm(vMat(:, j));
    end
    lam_unnormed = rand(nVec, 1);
    lam = lam_unnormed / sum(lam_unnormed);
    x0 = zeros(dimN, dimN);
    for j=1:nVec
      x0 = x0 + lam(j) * wMat(:, j) * wMat(:, j)';
    end
    feasible = (norm(AMap(x0) - b, 'fro') <= 1e-6);
  end

end