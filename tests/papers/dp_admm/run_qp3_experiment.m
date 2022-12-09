%% SPDX-License-Identifier: MIT
% Copyright © 2022 Weiwei "William" Kong

% Experiment runner.
function tbl = run_qp3_experiment(n, gamma, iter_limit, tol)

% Initialization.
seed = 777;
[grad_f_arr, prox_h_arr, prox_fh_arr] = qp3_factory(n, gamma, seed);
x0_arr = {};
x0_arr{1} = zeros(n, 1);
x0_arr{2} = zeros(n, 1);
x0_arr{3} = zeros(n, 1);
params = [];
params.iter_limit = iter_limit;
params.tol = tol;

% DP.ADMM runs.
dp_params = params;
dp_params.c0 = 1;
dp_params.lambda = 0.5;
dp_params.x0_arr = x0_arr;
dp_params.n = n;
% (theta, chi) = (0, 1).
dp_params1 = dp_params;
dp_params1.theta = 0;
dp_params1.chi = 1;
[~, history] = DP_ADMM_qp3(prox_fh_arr, dp_params1);
dp_iter1 = history.iter;
dp_time1 = history.runtime;
% (theta, chi) = (1/2, 1/18).
dp_params2 = dp_params;
dp_params2.theta = 1/2;
dp_params2.chi = 1/18;
[~, history] = DP_ADMM_qp3(prox_fh_arr, dp_params2);
dp_iter2 = history.iter;
dp_time2 = history.runtime;

% SDD-ADMM runs.
sdd_params = params;
sdd_params.rho = 0.01;
sdd_params.omega = 4;
sdd_params.theta = 2;
sdd_params.tau = 1;
sdd_params.x0_arr = x0_arr;
sdd_params.n = n;
% Constraint function constants.
sdd_params.Mh = 4 * gamma;
sdd_params.Kh = 1.0;
sdd_params.Jh = 1.0;
sdd_params.Lh = 0.0;
% Rho = 0.1.
sdd_params1 = sdd_params;
sdd_params1.rho = 0.1;
[~, history] = SDD_ADMM_qp3(grad_f_arr, prox_h_arr, sdd_params1);
sdd_iter1 = history.iter;
sdd_time1 = history.runtime;
% Rho = 1.0.
sdd_params2 = sdd_params;
sdd_params2.rho = 1.0;
[~, history] = SDD_ADMM_qp3(grad_f_arr, prox_h_arr, sdd_params2);
sdd_iter2 = history.iter;
sdd_time2 = history.runtime;
% Rho = 10.0.
sdd_params3 = sdd_params;
sdd_params3.rho = 10.0;
[~, history] = SDD_ADMM_qp3(grad_f_arr, prox_h_arr, sdd_params3);
sdd_iter3 = history.iter;
sdd_time3 = history.runtime;
% Output tables.
tbl = table(dp_iter1, dp_iter2, sdd_iter1, sdd_iter2, sdd_iter3, ...
            dp_time1, dp_time2, sdd_time1, sdd_time2, sdd_time3);
end