%% SPDX-License-Identifier: MIT
% Copyright © 2022 Weiwei "William" Kong

% Generates a QP3 problem
function prox_fh_arr = qp3_factory(n, gamma, seed) 

rng(seed);
% Prox functions of lam * (f_i + h_i).
prox_fh_arr = {};
alpha1 = rand();
beta1 = rand(n, 1);
prox_fh_arr{1} = @(lam, y) min(gamma, max(-gamma, (lam * beta1 + y) / (1 - lam * alpha1)));
alpha2 = rand();
beta2 = rand(n, 1);
prox_fh_arr{2} = @(lam, y) min(gamma, max(-gamma, (lam * beta2 + y) / (1 - lam * alpha2)));
prox_fh_arr{3} = @(lam, y) min(gamma, max(-gamma, y));

end