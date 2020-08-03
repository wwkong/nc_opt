%{

DESCRIPTION
-----------
Generator of a test suite of nonconvex support vector machine (SVM) 
problems.

NOTES
-----
** Function is
    (1 / k) * sum_{i=1,..,k} (1 - tanh(v_i * <u_i, z>)) + 
    (1 / 2*k) ||z|| ^ 2.

FILE DATA
---------
Last Modified: 
  August 2, 2020
Coders: 
  Weiwei Kong, Jiaming Liang

INPUT
-----
(n, k, seed, density, r):
  Input parameters as described in the function description.

INPUT
-----
oracle:
  An Oracle object.
params:
  A struct containing input parameters for this function.

%}

function [oracle, params] = test_fn_svm_01(n, k, seed, density, r)
  % problem generating function
  rng(seed);
  lam = 1 / k;

  % generate the data (matrices)
  U = sprand(n, k, density);
  x_bar = RandPtBall(n, 1, r);
  v = sign(U' * x_bar);

  % Set the topology (Euclidean)
  prod_fn = @(a,b) sum(dot(a, b));
  norm_fn = @(a) norm(a, 'fro');
  
  % estimates of upper and lower curvatures M,m and Lipschitz
  % continuity constant L
  M = norm(U, 'fro') ^ 2;
  params.M = 4 * sqrt(3) / 9 * M / k + lam;
  params.m = params.M;
  params.x0 = zeros(n, 1);
  params.prod_fn = prod_fn;
  params.norm_fn = norm_fn;

  % Sigmoid loss (William's version)
  LOSS = ...
    @(x) 1 / k * sum(1 - tanh(v .* U' * x)) + lam / 2 * norm_fn(x) ^ 2;
  GRAD = @(x) ...
    1 / k * sum(((tanh(v .* U' * x) .^ 2 - 1) .* v) .* U', 1)' + lam * x;

  % Create the Oracle object
  f_s = @(x) LOSS(x) + lam / 2 * norm(x) ^ 2;
  f_n = @(x) 0;
  grad_f_s = @(x) GRAD(x) + lam * x;
  prox_f_n = @(x, lam) x; 
  oracle = Oracle(f_s, f_n, grad_f_s, prox_f_n);

end

% This code generates k samples uniformly from an n-dimensional ball 
% centered at O with radius r.
function Z = RandPtBall(n,k,r)

% n = 3; % dimiension
% k = 100000; % samples
% r = 4; % radius

X = randn(n,k);
for i = 1:k
    X(:,i) = X(:,i)/norm(X(:,i));
end
U = rand(k,1);
F = r*U.^(1/n);
Z = zeros(n,k);
for i = 1:k
    Z(:,i) = X(:,i)*F(i);
end

end