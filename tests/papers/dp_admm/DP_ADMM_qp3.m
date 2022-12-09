%% SPDX-License-Identifier: MIT
% Copyright © 2022 Weiwei "William" Kong

% Definition of the DP.ADMM algorithm for the QP3 problem.
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
  function x_arr = primal_update(xm_arr, pm_arr, cm)
    x_arr = {};
    prox1 = prox_fh_arr{1};
    x_arr{1} = prox1(lam/(1 + lam), ...
      (lam * (cm * xm_arr{3} - (1 - theta) * pm_arr{1}) + xm_arr{1}) / (1 + lam * cm));
    prox2 = prox_fh_arr{2};
    x_arr{2} = prox2(lam/(1 + lam), ...
      (lam * (cm * xm_arr{3} - (1 - theta) * pm_arr{2}) + xm_arr{2}) / (1 + lam * cm));
    prox3 = prox_fh_arr{3};
    x_arr{3} = prox3(1 / (1 + 2 * lam * cm), ...
                     (lam * ( (1 - theta) * (pm_arr{1} + pm_arr{2}) + cm * x_arr{1} + cm * x_arr{2}) + xm_arr{3}) / ...
                     (1 + 2 * lam * cm));
  end
  function p_arr = dual_update(xm_arr, pm_arr, cm)
    p_arr = {};
    p_arr{1} = (1 - theta) * pm_arr{1} + chi * cm * (xm_arr{1} - xm_arr{3});
    p_arr{2} = (1 - theta) * pm_arr{2} + chi * cm * (xm_arr{2} - xm_arr{3});
  end

  
  x_arr = primal_update(x0_arr, p0_arr, c0);
  p_arr = dual_update(x_arr, p0_arr, c0);
  
  model.x_arr = x_arr;
  history.iter = 1;
  
end