% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function model = refine_ICG(spectral_oracle, params, M2_plus, lambda, Z0, x0, z, v)
% Refinement subroutine for the ICG methods.
% NOTE: M2_plus is the nonnegative upper cruvature constant for f2_s.

  % Auxilllary variables
  zR = length(z);
  P = params.P;
  Q = params.Q;
  spo_Z0 = params.spo_Z0;
  Z = P * spdiags(z, 0, zR, zR) * Q';
  V = P * spdiags(v, 0, zR, zR) * Q';
  
  % Helper variables
  nu = lambda / (lambda * M2_plus + 1);
  spo_at_Z = copy(spectral_oracle.spectral_eval(Z, z));
  w_r = lambda * spo_at_Z.spectral_grad_f2_s() - (v + x0 - z);
  z_hat_prox_ctr =  z - nu / lambda * w_r;
  Z_hat_prox_ctr = P * spdiags(z_hat_prox_ctr, 0, zR, zR) * Q';
  spo_u0 = copy(spectral_oracle.spectral_eval(Z_hat_prox_ctr, z_hat_prox_ctr));
    
  % Compute z_hat variables
  z_hat = spo_u0.spectral_prox_f_n(nu);
  Z_hat = P * spdiags(z_hat, 0, zR, zR) * Q';
  spo_at_Z_hat = copy(spectral_oracle.spectral_eval(Z_hat, z_hat));
  
  % Compute q_hat and v_hat;
  Q_hat = 1 / lambda * (V + Z0 - Z) + 1 / nu * (Z - Z_hat);
  delta_f2 = Q_hat + P * spdiags(spo_at_Z_hat.spectral_grad_f2_s() - spo_at_Z.spectral_grad_f2_s(), 0, 0) * Q';
  delta_f1 = spo_at_Z_hat.grad_f1_s() - spo_Z0.grad_f1_s();
  
  % Compute Q_hat and V_hat
  V_hat = Q_hat + delta_f2 + delta_f1;
    
  % Output refinement
  model.spo_at_z_hat = spo_at_Z_hat;
  model.z_hat_vec = z_hat;
  model.z_hat = Z_hat;
  model.q_hat = Q_hat;
  model.v_hat = V_hat;
  
end