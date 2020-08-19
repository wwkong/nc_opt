% Solve a nonconvex support vector machine problem

% The function of interest is
%
%  f(z) :=  
%     (1 / k) * sum_{i=1,..,k} (1 - tanh(v_i * <u_i, z>)) + 
%     (1 / 2*k) ||z|| ^ 2.

% Use a problem instance generator to create the oracle and 
% hyperparameters.
n = 500;
k = 250;
r = 50;
density = 0.05;
seed = 777;
[oracle, hparams] = test_fn_svm_01(n, k, seed, density, r);

% Create the Model object and specify the solver.
ncvx_svm = CompModel(oracle);
ncvx_svm.solver = @UPFAG;

% Set the curvatures and the starting point x0.
ncvx_svm.M = hparams.M;
ncvx_svm.m = hparams.m;
ncvx_svm.x0 = hparams.x0;

% Use a relative termination criterion.
ncvx_svm.opt_type = 'relative';

% Solve the problem.
ncvx_svm.optimize;