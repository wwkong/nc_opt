% SPDX-License-Identifier: MIT
% Copyright © 2022 Weiwei "William" Kong

% The function of interest is
%
%  ????

%% Initialization

% Set up paths.
run('../../../init.m');

% Set up variables
seed = 7777;
alpha = 1E-7;
beta = 450.0;
theta = 1E-4;
mu = 1.0;
global_tol = 1e-10;
time_limit = 1200;
iter_limit = 5000;
img_names = {'35008.jpg',  '41004.jpg', '68077.jpg', '271031.jpg', '310007.jpg'};

%% Generate Images

h = montage(img_names, 'size', [1, NaN]);
imwrite(h.CData, "mat_compl_images.jpg");

%% Generate Tables

% Set hyperparameters.
base_hparam = struct();
base_hparam.i_logging = true;

ncf_hparam = base_hparam;
ncf_hparam.m0 = 1;
ncf_hparam.M0 = 1;

upf_hparam = base_hparam;
upf_hparam.line_search_type = 'monotonic';

apd_hparam = base_hparam;
apd_hparam.m0 = 1;
apd_hparam.M0 = 1;
apd_hparam.theta = 4;
apd_hparam.beta = 2;
apd_hparam.alpha = 2;
apd1_hparam = apd_hparam;
apd1_hparam.line_search_type = 'monotonic';
apd2_hparam = apd_hparam;
apd2_hparam.line_search_type = 'optimistic';

aipp_hparam = base_hparam;
aipp_hparam.steptype = 'aipp';
aipp_hparam.acg_steptype = 'constant';
aipp_hparam.sigma = 1/4;

% Loop over the curvature pair (m, M).
offset = 0;
for i = 1:length(img_names) - offset
  % Use a problem instance generator to create the oracle and
  % hyperparameters.
  disp(img_names{i + offset});
  id = strrep(img_names{i}, ".jpg", "");
  rng(seed);
  snr = 200;
  density = 0.30;
  raw_img = imread(img_names{i + offset});
  img_matrix = corrupt_im(raw_img, snr, density);
  [oracle, hparams] = test_fn_penalty_mc_01(img_matrix, alpha, beta, theta, mu, seed);

  % Create the Model object and specify the solver.
  mat_compl = CompModel(oracle);

  % Set the curvatures and the starting point x0.
  mat_compl.M = hparams.M;
  mat_compl.m = hparams.m;
  mat_compl.x0 = hparams.x0;
  
  % Adjustments for AIPP.
  aipp_hparam.m = hparams.m;
  aipp_hparam.M = hparams.M;

  % Set up the termination criterion.
  mat_compl.opt_type = 'relative';
  mat_compl.opt_tol = global_tol;
  mat_compl.time_limit = time_limit;
  mat_compl.iter_limit = iter_limit;

  % Run a benchmark test and print the summary.
  solver_arr = {@UPFAG, @ADAP_FISTA, @AIPP, @APD};
  hparam_arr = {upf_hparam, ncf_hparam, aipp_hparam, apd2_hparam};
  name_arr = {'UPF', 'ANCF', 'AIPP', 'APD'};
  
%   % Run a benchmark test and print the summary.
%   solver_arr = {@AIPP};
%   hparam_arr = {aipp_hparam};
%   name_arr = {'AIPP'};
  
  [summary_tables, comp_models] = run_CM_benchmark(mat_compl, solver_arr, hparam_arr, name_arr);
  err_tbl = get_rel_err_tbl(comp_models, name_arr, raw_img);
  tbl_row = [table(id), ...
             summary_tables.pdata, ...
             summary_tables.fval, ...
             err_tbl, ...
             summary_tables.runtime, ...
             summary_tables.mdata];
  disp(tbl_row);
  writetable(tbl_row, "adp_mat_compl_" + strrep(img_names{i + offset}, ".jpg", "") + ".xlsx");
  
  % Make a montage and save.
  nplots = length(name_arr) + 2;
  tiledlayout(1, nplots + 1);
  img_array = {uint8(img_matrix)};
  for j=1:length(name_arr)
    img_array{end + 1} = uint8(comp_models.(name_arr{j}).model.x);
  end
  h = montage(img_array, 'size', [1, NaN]);
  imwrite(h.CData, "mat_compl_" + strrep(img_names{i + offset}, ".jpg", "") + ".jpg")
  
  % Set up the final table.
  if (i == 1)
    final_table = tbl_row;
  else
    final_table = [final_table; tbl_row];
  end
end

% Display final table for logging.
disp(final_table);
writetable(final_table, "adp_mat_compl.xlsx");

%% Helper functions.

function I2 = corrupt_im(I1, snr, density)
  I2 = add_random_zeros(double(add_noise(double(I1), snr)), density);
end

function err = compute_l2_err(I1, I2)
  im_diff = uint8(I1) - uint8(I2);
  err = norm(double(im_diff), 'fro');
end

function tbl = get_rel_err_tbl(comp_models, name_arr, ref_img)
  tbl = table();
  for j=1:length(name_arr)
    name = name_arr{j} + "_l2_err";
    abs_err = compute_l2_err(ref_img, comp_models.(name_arr{j}).model.x);
    flipped_img = 256 - ref_img;
    max_err = norm(double(max(ref_img, flipped_img)), 'fro');
    tbl.(name) = abs_err / max_err;
  end
end