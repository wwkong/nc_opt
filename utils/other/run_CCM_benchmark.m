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

constr_comp_model:
  A CompModel object for running the solvers on.

framework_arr:
  An array of frameworks to benchmark.

solver_arr:
  An array of solvers that correspond to the frameworks.

framework_hparams_arr (optional):
  An array of solver hyperparameters (can be left empty).

name_arr (optional):
  An array of framework names (can be left empty).

OUTPUT
------

summary_tables:
  A struct containing tables that summarize the results.

comp_models:
  A struct containing one CompModel per solver.

%}

function [summary_tables, comp_models] = ...
  run_CCM_benchmark(base_constr_comp_model, framework_arr, solver_arr, framework_hparams_arr, name_arr)

  % Check lengths.
  if (length(framework_arr) ~= length(solver_arr))
    error('The number of frameworks must equal the number of solvers!');
  end
  if (~isempty(framework_hparams_arr) && length(framework_arr) ~= length(framework_hparams_arr))
    error('The number of frameworks must equal the number of hyperparameters when the latter is nonempty!');
  end
  if (~isempty(name_arr) && length(solver_arr) ~= length(name_arr))
    error('The number of solvers must equal the number of names when the latter is nonempty!');
  end

  % Initialize.
  n_frameworks = length(framework_arr);
  summary_tables.runtime = table();
  summary_tables.iter = table();
  summary_tables.fval = table();
  summary_tables.pdata = table(); % Problem data.
  summary_tables.mdata = table(); % Model data.
  
  % Run an optimization routine for each algorithm.
  for i=1:n_frameworks
    % Prepare.
    constr_comp_model = copy(base_constr_comp_model);
    framework = framework_arr{i};
    solver = solver_arr{i};
    if ~isempty(name_arr)
      framework_name = name_arr{i};
    else
      framework_name = [func2str(framework), '_', func2str(solver)];
    end
    constr_comp_model.solver = solver;
    constr_comp_model.framework = framework;
    if (~isempty(framework_hparams_arr))
      constr_comp_model.solver_hparams = framework_hparams_arr{i};
    end
    % Optimize.
    constr_comp_model.optimize;
    % Record the results.
    comp_models.(framework_name) = constr_comp_model;
    summary_tables.runtime = add_column(['t_', framework_name], constr_comp_model.runtime, summary_tables.runtime);
    summary_tables.iter = add_column(['iter_', framework_name], constr_comp_model.iter, summary_tables.iter);
    summary_tables.fval = add_column(['fval_', framework_name], constr_comp_model.f_at_x, summary_tables.fval);
    % Auxiliary results.
    if (isfield(constr_comp_model.history, 'outer_iter'))
      summary_tables.mdata = add_column(['oiter_', framework_name], constr_comp_model.history.outer_iter, summary_tables.mdata);
    end
  end
  
  % Add solver independent data.
  summary_tables.pdata = add_column('m', constr_comp_model.m, summary_tables.pdata);
  summary_tables.pdata = add_column('M', constr_comp_model.M, summary_tables.pdata);
  summary_tables.pdata = add_column('K_constr', constr_comp_model.K_constr, summary_tables.pdata);
  summary_tables.pdata = add_column('L_constr', constr_comp_model.L_constr, summary_tables.pdata);
  summary_tables.mdata = add_column('feas_tol', constr_comp_model.feas_tol, summary_tables.mdata);
  summary_tables.mdata = add_column('feas_type', convertCharsToStrings(constr_comp_model.feas_type), summary_tables.mdata);
  summary_tables.mdata = add_column('opt_tol', constr_comp_model.opt_tol, summary_tables.mdata);
  summary_tables.mdata = add_column('opt_type', convertCharsToStrings(constr_comp_model.opt_type), summary_tables.mdata);

  % Merge summary tables for easy reading
  summary_tables.all = ...
    [summary_tables.pdata, summary_tables.fval, summary_tables.iter, summary_tables.runtime, summary_tables.mdata];

end

% Utility function to add a column to a table if the value is nonempty
function out_tbl = add_column(col_name, col_val, in_tbl)
  if ~isempty(col_val)
    tmp_tbl = table(col_val);
    tmp_tbl.Properties.VariableNames = {col_name};
    out_tbl = [in_tbl, tmp_tbl];
  end
end