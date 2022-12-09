%% SPDX-License-Identifier: MIT
% Copyright © 2022 Weiwei "William" Kong

% Implementation of the SDD-ADMM algorithm for the QP3 problem.
function [model, history] = SDD_ADMM_qp3(grad_f_arr, prox_h_arr, params)

  % Initialization.
  x0_arr = params.x0_arr;
  rho = params.rho;
  omega = params.omega;
  theta = params.theta;
  tau = params.tau;
  Mh = params.Mh;
  Lh = params.Lh;
  Jh = params.Jh;
  Kh = params.Kh;
  n = params.n;
  mu0_arr = {};
  mu0_arr{1} = zeros(n, 1);
  mu0_arr{2} = zeros(n, 1);
  
  % Helper functions.
  function L = get_lipschitz(muPrev_arr)
    mu_norm = sqrt(norm(muPrev_arr{1}, 'fro') ^ 2 + norm(muPrev_arr{2}, 'fro') ^ 2);
    L = 1.0 + mu_norm * Lh + rho * (Jh * Kh + Mh * Lh);
  end
  function x_arr = primal_update(xPrev_arr, muPrev_arr)
    x_arr = {};
    L = get_lipschitz(muPrev_arr);
    prox1 = prox_h_arr{1};
    grad1 = grad_f_arr{1};
    g1 = grad1(xPrev_arr{1}) + muPrev_arr{1} + rho* (xPrev_arr{1} - xPrev_arr{3});
    x_arr{1} = prox1(1.0, xPrev_arr{1} - g1 / (theta * L));
    prox2 = prox_h_arr{2};
    grad2 = grad_f_arr{2};
    g2 = grad2(xPrev_arr{2}) + muPrev_arr{2} + rho * (xPrev_arr{2} - xPrev_arr{3});
    x_arr{2} = prox2(1.0, xPrev_arr{2} - g2 / (theta * L));
    prox3 = prox_h_arr{3};
    g3 = - muPrev_arr{1} - muPrev_arr{2} + rho * (xPrev_arr{3} - xPrev_arr{1}) + rho * (xPrev_arr{3} - xPrev_arr{1});
    x_arr{3} = prox3(1.0, xPrev_arr{3} - g3 / (theta * L));
  end
  function mu_arr = dual_update(xCurr_arr, muPrev_arr)
    mu_arr = {};
    mu_arr{1} = (1.0 / (1.0 + tau)) * (tau * muPrev_arr{1} - rho * (xCurr_arr{1} - xCurr_arr{3}) / omega);
    mu_arr{2} = (1.0 / (1.0 + tau)) * (tau * muPrev_arr{2} - rho * (xCurr_arr{2} - xCurr_arr{3}) / omega); 
  end
  function v_resid = get_primal_residual(xCurr_arr, xPrev_arr, muPrev_arr)
    L = get_lipschitz(muPrev_arr);
    delta_arr = {};
    for i=1:3
      delta_arr{i} = theta * L * (xPrev_arr{i} - xCurr_arr{i});
    end
    xi1 = delta_arr{1} + rho * (xCurr_arr{1} - xCurr_arr{3}) - rho * (xPrev_arr{1} - xPrev_arr{3});
    xi2 = delta_arr{2} + rho * (xCurr_arr{2} - xCurr_arr{3}) - rho * (xPrev_arr{2} - xPrev_arr{3});
    xi3 = delta_arr{3} - rho * (xCurr_arr{1} - xCurr_arr{3}) - rho * (xCurr_arr{2} - xCurr_arr{3}) ...
                       + rho * (xPrev_arr{1} + xPrev_arr{2} - 2.0 * xPrev_arr{3});
    v_resid = sqrt(norm(xi1, 'fro') ^ 2 + norm(xi2, 'fro') ^ 2 + norm(xi3, 'fro') ^ 2);
  end
  function f_resid = get_dual_residual(xCurr_arr)
    f1 = xCurr_arr{1} - xCurr_arr{3};
    f2 = xCurr_arr{2} - xCurr_arr{3};
    f_resid = sqrt(norm(f1, 'fro') ^ 2 + norm(f2, 'fro') ^ 2);
  end

  % Main loop.
  t_start = tic;
  iter = 0;
  while (iter < params.iter_limit)
    % Main steps.
    x_arr = primal_update(x0_arr, mu0_arr);
    v_resid = get_primal_residual(x_arr, x0_arr, mu0_arr);
    f_resid = get_dual_residual(x_arr);
    resid = max([f_resid, v_resid]); 
    if (resid <= params.tol)
      break;
    end
    mu_arr = dual_update(x_arr, mu0_arr);
    % Iteration update.
    iter = iter + 1;
    x0_arr = x_arr;
    mu0_arr = mu_arr;
  end
  
  model.x_arr = x_arr;
  model.c = rho;
  history.iter = iter;
  history.runtime = toc(t_start);
  history.resid = resid;

end