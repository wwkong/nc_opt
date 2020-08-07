% Solve a basic univariate convex quadratic programming problem.

% The function of interest is
%
%   f(x) := (1 / 2) * (z - 1) ^ 2.

% Create the Model object and specify the solver.
cvx_qp = CompModel;
cvx_qp.solver = @AG;

% Define the function and its gradient.
cvx_qp.f_s = @(x) (1 / 2) * (x - 1) ^ 2;
cvx_qp.grad_f_s = @(x) x - 1;

% Set the Lipschitz constant of the gradient and the starting point x0.
cvx_qp.L = 10;
cvx_qp.x0 = 10;

% Use a relative termination criterion.
cvx_qp.opt_type = 'relative';

% Solve the problem.
cvx_qp.optimize;