%% A Basic Univariate Convex QP Problem

% Create the Model object and specify the solver.
cvx_qp = CompModel;
cvx_qp.solver = @AIPP;

% Define the function and its gradient.
cvx_qp.f_s = @(z) (1/2) * (z - 1) ^ 2;
cvx_qp.grad_f_s = @(z) z - 1;

% Set the Lipschitz constant of the gradient and the starting point x0.
cvx_qp.L = 5;
cvx_qp.x0 = 10;

% Solve the problem.
cvx_qp.optimize;