% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function [Delta_mu, prox_at_S, prox_at_Y] = Delta_mu_fn(params, mu, spo_at_S, spo_at_Y, spo_at_Y0, S, Y, Y0, V)
% The $\Delta_\mu$ function for the ICG methods

  % Induced norm
  norm_fn = params.norm_fn;
  prod_fn = params.prod_fn;

  % Set up the variables
  prox_at_S = ...
    spo_at_Y0.f1_s() + prod_fn(spo_at_Y0.grad_f1_s(), S - Y0) + spo_at_S.f2_s() + spo_at_S.f_n() - prod_fn(V, S) - ...
    mu * norm_fn(S - Y) ^ 2 / 2;
  prox_at_Y = spo_at_Y0.f1_s() + prod_fn(spo_at_Y0.grad_f1_s(), Y - Y0) + spo_at_Y.f2_s() + spo_at_Y.f_n() - prod_fn(V, Y);
  Delta_mu = prox_at_Y - prox_at_S;
  
end