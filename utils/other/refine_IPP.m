% SPDX-License-Identifier: MIT
% Copyright © 2021 Weiwei "William" Kong

function model = refine_IPP(oracle, params, L, lambda, z0, z, v)
% Refinement subroutine for the IPP methods.

  % Parse.
  norm_fn = params.norm_fn;
  prod_fn = params.prod_fn;

  % Helper variables.
  M_lam = lambda * L + 1;
  o_z = oracle.eval(z);
  f_at_z = o_z.f_s() + o_z.f_n();
  grad_f_s_at_z = o_z.grad_f_s();
  z_next = z - lambda / M_lam * (grad_f_s_at_z + (z - z0) / lambda - v / lambda);
  o_z_next = oracle.eval(z_next);

  % Compute z_hat variables.
  z_hat = o_z_next.prox_f_n(lambda / M_lam);
  o_z_hat = oracle.eval(z_hat);
  f_at_z_hat = o_z_hat.f_s() + o_z_hat.f_n();
  grad_f_s_at_z_hat = o_z_hat.grad_f_s();

  % Compute v_hat.
  q_hat =  1 / lambda * ((v + z0 - z) + M_lam * (z - z_hat));
  v_hat = q_hat + grad_f_s_at_z_hat - grad_f_s_at_z;

  % Compute Delta.
  refine_fn_at_z = (lambda * f_at_z + 1 / 2 * norm_fn(z - z0) ^ 2 - prod_fn(v, z));
  refine_fn_at_z_hat = (lambda * f_at_z_hat + 1 / 2 * norm_fn(z_hat - z0) ^ 2 - prod_fn(v, z_hat));
  Delta = refine_fn_at_z - refine_fn_at_z_hat;

  % Compute some auxillary quantities (for updating tau).
  model.residual_v_hat = norm_fn(v_hat);
  model.residual_1 = norm_fn(v + z0 - z) / lambda;
  model.residual_2 = norm_fn(grad_f_s_at_z_hat - grad_f_s_at_z + M_lam * (z - z_hat) / lambda);

  % Output refinement.
  model.o_z_hat = o_z_hat;
  model.z_hat = z_hat;
  model.v_hat = v_hat;
  model.Delta = Delta;

end