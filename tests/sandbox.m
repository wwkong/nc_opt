data_name = 'jester_24938u_100j';
alpha = 10;
beta = 20;
mu = 2;
theta = 1;
seed = 777;

[spectral_oracle, hparams] = ...
  test_fn_spectral_mc_01(data_name, alpha, beta, theta, mu, seed);

%%
foo = CompModel(spectral_oracle);
foo.solver = @AG;
foo.time_limit = 5;

foo.solver_hparams = hparams;
foo.m = hparams.m;
foo.M = hparams.M;
foo.x0 = hparams.x0;

% profile on
foo.optimize;
% profile report
% profile off