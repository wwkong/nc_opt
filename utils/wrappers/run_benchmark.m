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
  if (length(solver_arr) ~= length(solver_hparams_arr))
    error(...
      ['The number of solvers must equal the number of ', ...
       'hyperparameters arrays when the latter is nonempty!']);
  end
  n_solvers = length(solver_arr);
  summary_tables.runtime = [];
  
  % Run a routine for each algorithm
  for i=1:n_solvers
    solver = solver_arr{i};
    solver_name = func2str(solver);
    comp_model.solver = solver;
    if (~isempty(solver_hparams_arr))
      comp_model.solver_hparams = solver_hparams_arr{i};
    end
    comp_model.optimize;
    comp_models.(solver_name) = comp_model;
  end

end