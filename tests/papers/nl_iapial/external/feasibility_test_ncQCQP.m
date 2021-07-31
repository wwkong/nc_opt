clear; close all % Test code for nc_QCQP
%% generate data
rng(20190927); 
n = 100; % Dimension of the function input
m = 10; % Number of constraints

% Curvature constants, i.e., (rho, L) satisfying
%   -rho|x-u|^2/2 <= f0(u) - f0(x) - gradient_f0(x)'*(u-x) <= L|x-u|^2/2
% for every u and x in the domain of the constraints.
L = 1000; % Upper curvature
rho = 1; % Lower curvature

Q = cell(1,m+1);
c = cell(1,m+1);
d = cell(1,m+1);

[U,~] = qr(randn(n));

v = -rho + (L + rho) * rand(n, 1);
v(1) = -rho;
v(end) = L;
Q{m+1} = U * diag(v) * U';
Q{m+1} = (Q{m+1} + Q{m+1}') / 2; % symmetrization is important
c{m+1} = rand([n, 1]);
d{m+1} = rand();

fprintf('weak convexity constant is %f\n', min(eig(Q{m+1})));

%%
for j = 1:m
  [U, ~] = qr(rand(n));
  v = rand(n, 1);
  Q{j} = U * diag(v) * U';
  Q{j} = (Q{j} + Q{j}') / 2;
  c{j} = rand([n, 1]);
  d{j} = -1 - rand();
end

lb = -5 * ones(n,1); % can change.
ub = 5 * ones(n,1); % can change.

maxsubit = 1000000;
tol = 1e-3;

%%
opts = [];
opts.x0 = zeros(n,1);
opts.maxsubit = maxsubit;
opts.maxit = 500;
opts.sig = 3;
opts.beta0 = 0.01;
opts.tol = tol;
opts.ver = 3; % opts.ver = 3; %% 1 for const beta
opts.inc = 2;
opts.dec= 2.5;
opts.K0 = 100;

tic;
[x1,z,out] = HiAPeM_qcqp(Q,c,d,m,lb,ub,opts);
toc;
        
%% Compute the feasibility
[res_x1, feas_x1] = feasibility(Q, c, d, x1);
disp(['Feasibility is ', num2str(feas_x1)]);

% Utility function
function [residual_vec, feas] = feasibility(Q, c, d, x)
  m = length(Q) - 1;
  residual_vec = -Inf * ones(m, 1);
  for j=1:m
    residual_vec(j) = max([.5*x'*Q{j}*x + c{j}'*x + d{j}, 0]);
  end
  feas = norm(residual_vec);
end