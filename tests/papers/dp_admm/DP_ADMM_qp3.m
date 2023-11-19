%% SPDX-License-Identifier: MIT
% Copyright © 2022 Weiwei "William" Kong

% Implementation of the DP.ADMM algorithm for the QP3 problem.
function [model, history] = DP_ADMM_qp3(prox_fh_arr, params)

  % Initialization.
  x0_arr = params.x0_arr;
  theta = params.theta;
  chi = params.chi;
  n = params.n;
  lam = params.lambda;
  c0 = params.c0;
  p0_arr = {};
  p0_arr{1} = zeros(n, 1);
  p0_arr{2} = zeros(n, 1);
  
  % Helper functions.
  function x_arr = primal_update(xPrev_arr, pPrev_arr, cPrev)
    x_arr = {};
    prox1 = prox_fh_arr{1};
    x_arr{1} = prox1(lam/(1 + cPrev * lam), ...
      (lam * (cPrev * xPrev_arr{3} - (1 - theta) * pPrev_arr{1}) + xPrev_arr{1}) / (1 + lam * cPrev));
    prox2 = prox_fh_arr{2};
    x_arr{2} = prox2(lam/(1 + cPrev * lam), ...
      (lam * (cPrev * xPrev_arr{3} - (1 - theta) * pPrev_arr{2}) + xPrev_arr{2}) / (1 + lam * cPrev));
    prox3 = prox_fh_arr{3};
    x_arr{3} = prox3(1 / (1 + 2 * lam * cPrev), ...
                     (lam * ( (1 - theta) * (pPrev_arr{1} + pPrev_arr{2}) + cPrev * x_arr{1} + cPrev * x_arr{2}) + xPrev_arr{3}) / ...
                     (1 + 2 * lam * cPrev));
  end
  function p_arr = dual_update(xPrev_arr, pPrev_arr, cPrev)
    p_arr = {};
    p_arr{1} = (1 - theta) * pPrev_arr{1} + chi * cPrev * (xPrev_arr{1} - xPrev_arr{3});
    p_arr{2} = (1 - theta) * pPrev_arr{2} + chi * cPrev * (xPrev_arr{2} - xPrev_arr{3});
  end
  function v_resid = get_primal_residual(xCurr_arr, xPrev_arr, cPrev)
    v1 = (xPrev_arr{1} - xCurr_arr{1}) / lam + cPrev * (xPrev_arr{3} - xCurr_arr{3});
    v2 = (xPrev_arr{2} - xCurr_arr{2}) / lam + cPrev * (xPrev_arr{3} - xCurr_arr{3});
    v3 = (xPrev_arr{3} - xCurr_arr{3}) / lam;
    v_resid = sqrt(norm(v1, 'fro') ^ 2 + norm(v2, 'fro') ^ 2 + norm(v3, 'fro') ^ 2);
  end
  function f_resid = get_dual_residual(xCurr_arr)
    f1 = xCurr_arr{1} - xCurr_arr{3};
    f2 = xCurr_arr{2} - xCurr_arr{3};
    f_resid = sqrt(norm(f1, 'fro') ^ 2 + norm(f2, 'fro') ^ 2);
  end

  % Main loop.
  t_start = tic;
  Sv = 0.0;
  Sf = 0.0;
  iter = 0;
  cycle_iter = 0;
  while (iter < params.iter_limit)
    % Step 1.
    x_arr = primal_update(x0_arr, p0_arr, c0);
    % Step 2a.  
    v_resid = get_primal_residual(x_arr, x0_arr, c0);
    f_resid = get_dual_residual(x_arr);
    resid = max([f_resid, v_resid]); 
    if (resid <= params.tol)
      break;
    end
    % Step 2b. (modified)
    Sv = (Sv * (cycle_iter + 1) / 2 + v_resid) * 2 / (cycle_iter + 2);
    Sf = (Sf * (cycle_iter + 1) / 2 + f_resid) * 2 / (cycle_iter + 2);
    if (Sv / params.tol + 1 / params.tol * sqrt(c0 ^ 3 / cycle_iter) * Sf <= 1.0)
      cycle_iter = 0.0;
      c0 = c0 * 2.0;
    else
      cycle_iter = cycle_iter + 1;
    end
    % Step 3.
    p_arr = dual_update(x_arr, p0_arr, c0);
    % Iteration update.
    iter = iter + 1;
    x0_arr = x_arr;
    p0_arr = p_arr;
  end

  model.x_arr = x_arr;
  model.c = c0;
  history.iter = iter;
  history.runtime = toc(t_start);
  history.resid = resid;
  
end