%% SPDX-License-Identifier: MIT
% Copyright © 2022 Weiwei "William" Kong

% Generates a QP3 problem
function [grad_f_arr, prox_h_arr, prox_fh_arr] = qp3_factory(n, gamma, seed) 

rng(seed);
prox_fh_arr = {};
alpha1 = rand();
beta1 = rand(n, 1);
alpha2 = rand();
beta2 = rand(n, 1);
% Gradient functions of f_i.
grad_f_arr{1} = @(x) -alpha1 * x - beta1;
grad_f_arr{2} = @(x) -alpha2 * x - beta2;
grad_f_arr{3} = @(x) zeros(n, 1);
% Prox functions of lam * h_i.
prox_h_arr{1} = @(lam, y) min(gamma, max(-gamma, y));
prox_h_arr{2} = @(lam, y) min(gamma, max(-gamma, y));
prox_h_arr{3} = @(lam, y) min(gamma, max(-gamma, y));
% Prox functions of lam * (f_i + h_i).
prox_fh_arr{1} = @(lam, y) min(gamma, max(-gamma, (lam * beta1 + y) / (1 - lam * alpha1)));
prox_fh_arr{2} = @(lam, y) min(gamma, max(-gamma, (lam * beta2 + y) / (1 - lam * alpha2)));
prox_fh_arr{3} = @(lam, y) min(gamma, max(-gamma, y));

end