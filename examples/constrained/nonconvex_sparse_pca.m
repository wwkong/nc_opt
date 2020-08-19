% Solve a Sparse PCA problem instance.

% Test generator
b = 0.1;
nu = 100;
p = 100;
n = 100;
s = 5;
k = 1;
seed = 777;
[oracle, hparams] = test_fn_spca_01(b, nu, p, n, s, k, seed);

% Create the Model object and specify the solver.
ncvx_spca = ConstrCompModel(oracle);
ncvx_spca.solver = @AIPP;
ncvx_spca.feas_type = 'relative';

% Add linear constraints
ncvx_spca.constr_fn = hparams.constr_fn;
ncvx_spca.grad_constr_fn = hparams.grad_constr_fn;
ncvx_spca.set_projector = hparams.set_projector;

% Add penalty framework
ncvx_spca.K_constr = hparams.K_constr;
ncvx_spca.opt_tol = 1e-3;
ncvx_spca.feas_tol = 1e-1;
ncvx_spca.framework = @penalty;

% Set the curvatures and the starting point x0.
ncvx_spca.M = hparams.M;
ncvx_spca.m = hparams.m;
ncvx_spca.x0 = hparams.x0;

% Solve the problem.
ncvx_spca.optimize;