%{

DESCRIPTION
-----------
A wrapper that runs several solvers on a user-provided CompModel object.

FILE DATA
---------
Last Modified: 
  August 4, 2020
Coders: 
  Weiwei Kong

INPUT
-----
comp_model:
  A CompModel object for running the solvers on.
solver_arr:
  An array of solvers to benchmark.
solver_arr (optional):
  An array of solver hyperparameters (can be left empty).

OUTPUT
------
summary_tables:
  A struct containing tables that summarize the results.
comp_models:
  A struct containing one CompModel per solver.

%}

function [summary_tables, comp_models] = ...
  run_benchmark(comp_model, solver_arr, solver_hparams_arr)

  % Initialize
  if (~isempty(solver_hparams_arr) && ...
      length(solver_arr) ~= length(solver_hparams_arr))
    error(...
      ['The number of solvers must equal the number of ', ...
       'hyperparameters arrays when the latter is nonempty!']);
  end
  n_solvers = length(solver_arr);
  summary_tables.runtime = table();
  summary_tables.iter = table();
  summary_tables.fval = table();
  summary_tables.pdata = table(); % Problem data.
  summary_tables.mdata = table(); % Model data.
  
  % Run an optimization routine for each algorithm.
  for i=1:n_solvers
    % Prepare.
    solver = solver_arr{i};
    solver_name = func2str(solver);
    comp_model.solver = solver;
    if (~isempty(solver_hparams_arr))
      comp_model.solver_hparams = solver_hparams_arr{i};
    end
    % Optimize.
    comp_model.optimize;
    % Record the results
    comp_models.(solver_name) = comp_model;
    summary_tables.runtime = add_column(...
      ['t_', solver_name], comp_model.runtime, summary_tables.runtime);
    summary_tables.iter = add_column(...
      ['iter_', solver_name], comp_model.iter, summary_tables.iter);
    summary_tables.fval = add_column(...
      ['fval_', solver_name], comp_model.f_at_x, summary_tables.fval);
  end
  
  % Add solver independent data.
  summary_tables.pdata = ...
    add_column('m', comp_model.m, summary_tables.pdata);
  summary_tables.pdata = ...
    add_column('M', comp_model.M, summary_tables.pdata);
  summary_tables.mdata = ...
    add_column('opt_tol', comp_model.opt_tol, summary_tables.mdata);
  summary_tables.mdata = ...
    add_column(...
      'opt_type', convertCharsToStrings(comp_model.opt_type), ...
      summary_tables.mdata);

  % Merge summary tables for easy reading
  summary_tables.all = ...
    [summary_tables.pdata, summary_tables.fval, summary_tables.iter, ...
     summary_tables.runtime, summary_tables.mdata];

end

% Utility function to add a column to a table if the value is nonempty
function out_tbl = add_column(col_name, col_val, in_tbl)
  if ~isempty(col_val)
    tmp_tbl = table(col_val);
    tmp_tbl.Properties.VariableNames = {col_name};
    out_tbl = [in_tbl, tmp_tbl];
  end
end