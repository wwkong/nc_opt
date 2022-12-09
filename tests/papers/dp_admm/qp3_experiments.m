%% SPDX-License-Identifier: MIT
% Copyright © 2022 Weiwei "William" Kong

% Set up paths.
run('../../../init.m');

% A collection of QP3 experiments for the DP.ADMM paper.
n = 3;
c0 = 1;
gamma = 5;
seed = 777;
prox_fh_arr = qp3_factory(n, gamma, seed);
x0_arr = {};
x0_arr{1} = zeros(n, 1);
x0_arr{2} = zeros(n, 1);
x0_arr{3} = zeros(n, 1);

params = [];
params.c0 = 1;
params.lambda = 0.5;
params.theta = 0.0;
params.chi = 1.0;
params.x0_arr = x0_arr;
params.n = n;

[model, history] = DP_ADMM_qp3(prox_fh_arr, params);
disp(model.x_arr{1});