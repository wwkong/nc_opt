data_name = 'filmtrust_1508u_2071m';
alpha = 10;
beta = 20;
mu = 2;
theta = 1;
seed = 777;

[spectral_oracle, params] = ...
  test_fn_spectral_mc_01(data_name, alpha, beta, theta, mu, seed);
spectral_oracle.eval(params.x0)