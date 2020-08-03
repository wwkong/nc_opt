% A class defintion for composite optimization models
classdef CompModel < handle

  % -----------------------------------------------------------------------
  %% GLOBAL PROPERTIES
  % -----------------------------------------------------------------------  
  
  % Visible global function properties.
  properties (SetAccess = public)
    f_s
    grad_f_s
    f_n = @(x) zeros(size(x))
    prox_f_n = @(x, lam) x
    solver
    solver_hparams
    iter
    runtime
  end
  
  % Invisible global function properties.
  properties (Access = public, Hidden = true)
    prod_fn = @(a,b) sum(dot(a, b))
    norm_fn = @(a) norm(a, 'fro')
  end
  
  % -----------------------------------------------------------------------
  %% SOLVER-RELATED PROPERTIES
  % -----------------------------------------------------------------------
  
  % Visible solver properties.
  properties (SetAccess = public)
    L (1,1) double {mustBeReal, mustBeNonnegative}
    x0 double {mustBeReal, mustBeFinite}
  end
  
  % Invisible solver properties.
  properties (SetAccess = public, Hidden = true)
    m double {mustBeReal, mustBeFinite} = []
    M double {mustBeReal, mustBeFinite} = []
    oracle
  end
  
  % Invisible and protected solver properties 
  properties (SetAccess = protected, Hidden = true)
    solver_input_params
  end
  
  % -----------------------------------------------------------------------
  %% MODEL-RELATED PROPERTIES
  % -----------------------------------------------------------------------
  
  % Visible model properties.
  properties (SetAccess = protected)
    x double {mustBeReal, mustBeFinite}
    v double {mustBeReal, mustBeFinite}
  end
  
  % Invisible model properties.
  properties (SetAccess = protected, Hidden = true)
    f_at_x double {mustBeReal, mustBeFinite}
    norm_of_v double {mustBeReal, mustBeFinite}
  end
  
  % Invisible model descriptors.
  properties (SetAccess = protected, Hidden = true)
    model
    status (1,1) int32 {mustBeInteger} = 0
  end
  
  % Invisible model flags.
  properties (SetAccess = protected, Hidden = true)
    i_verbose (1,1) {mustBeNumericOrLogical} = true
    i_update_oracle (1,1) {mustBeNumericOrLogical} = true
    i_update_curvatures (1,1) {mustBeNumericOrLogical} = true
    i_update_solver_inputs (1,1) {mustBeNumericOrLogical} = true
  end
  
  % Invisible toleranace and limit properties.
  properties (SetAccess = public, Hidden = true)
    iter_limit (1,1) double {mustBeReal} = Inf
    time_limit (1,1) double {mustBeReal} = Inf
    opt_tol (1,1) double {mustBeReal, mustBePositive} = 1e-6
  end
  
  % -----------------------------------------------------------------------
  %% HISTORY-RELATED PROPERTIES
  % -----------------------------------------------------------------------
  
  % Invisible history properties.
  properties (SetAccess = protected, Hidden = true)
    history
  end
    
  % -----------------------------------------------------------------------
  %% PRIMARY METHODS
  % -----------------------------------------------------------------------
  
  % Static methods.
  methods (Access = private, Static = true)
    
    % Converts status codes into strings
    function status_string = parse_status(status_num)
      status_map = containers.Map('KeyType', 'int32', 'ValueType', 'char');
      status_map(-1) = 'STATIONARITY_CONDITION_FAILED';
      status_map(0)  = 'EMPTY_MODEL';
      status_map(1)  = 'MODEL_LOADED';
      status_map(2)  = 'STATIONARITY_ACHIEVED';
      status_map(3)  = 'TIME_LIMIT_EXCEEDED';
      status_map(4)  = 'ITER_LIMIT_EXCEEDED';
      status_string  = status_map(status_num);
    end
  end % End static methods.
  
  % Ordinary methods.
  methods (Access = public)
    
    % Initialization functions.
    function reset(obj)
      obj.iter = [];
      obj.runtime = [];
      obj.x = [];
      obj.v = [];
      obj.f_at_x = [];
      obj.norm_of_v = [];
    end
    
    % Optimization functions.
    function optimize(obj)
      
        % Pre-processing.
        obj.reset;
        obj.update;
        
        % Solve the model.
        if (obj.i_verbose)
          obj.log_input;
        end
        [obj.model, obj.history] = ...
          obj.solver(obj.oracle, obj.solver_input_params);
        
        % Post-processing.
        obj.x = obj.model.x;
        obj.v = obj.model.v;
        o_at_x = obj.oracle.eval(obj.x);
        obj.f_at_x = o_at_x.f_s() + o_at_x.f_n();
        obj.norm_of_v = obj.norm_fn(obj.v);
        if (obj.norm_of_v < obj.opt_tol)
          obj.status = 2; % STATIONARITY_ACHIEVED
        else
          obj.status = -1; % STATIONARITY_CONDITION_FAILED
        end
        
        % Status update.
        if isfield(obj.history, 'runtime')
          obj.runtime = obj.history.runtime;
          if (obj.runtime >= obj.time_limit)
            obj.status = 3; % TIME_LIMIT_EXCEEDED
          end
        end
        if isfield(obj.history, 'iter')
          obj.iter = obj.history.iter;
          if (obj.iter >= obj.iter_limit)
            obj.status = 4; % ITER_LIMIT_EXCEEDED
          end
        end
        
        % Log model state if necessary.
        if (obj.i_verbose)
          obj.log_output;
          fprintf('\n');
        end
    end
    
    % Logging functions.
    function log_input(obj)
      word_len = 8;
      prefix = ['%-' num2str(word_len) 's = '];
      fprintf('\n');
      fprintf('» Solving model with: \n');
      fprintf([prefix, '%s\n'], 'SOLVER', func2str(obj.solver));
      fprintf([prefix, '%.2e\n'], 'L', obj.L);
      fprintf([prefix, '(%.2e, %.2e)\n'], '(M, m)', obj.M, obj.m);
      fprintf([prefix, '%.2e\n'], 'OPT_TOL', obj.opt_tol);
    end
    function log_output(obj)
      word_len = 16;
      prefix = ['%-' num2str(word_len) 's = '];
      fprintf('\n');
      fprintf('» Model terminated successfully with: \n');
      fprintf([prefix, '%s\n'], 'STATUS', obj.parse_status(obj.status));
      fprintf([prefix, '%.2e\n'], 'FUNCTION VALUE', obj.f_at_x);
      fprintf([prefix, '%.2e\n'], 'NORM OF V', obj.norm_of_v);
      if isempty(obj.runtime)
        fprintf([prefix, 'UNKNOWN\n'], 'RUNTIME');
      else
        fprintf([prefix, '%.2e seconds\n'], 'RUNTIME', obj.runtime);
      end      
      if isempty(obj.iter)
        fprintf([prefix, 'UNKNOWN\n'], 'ITERATION COUNT');
      else
        fprintf([prefix, '%.0f\n'], 'ITERATION COUNT', obj.iter);
      end      
    end
    
    % Sub-update functions.
    function update_oracle(obj)
      obj.oracle = Oracle(obj.f_s, obj.grad_f_s, obj.f_n, obj.prox_f_n);
    end
    function update_curvatures(obj)
      if isempty(obj.M)
        obj.M = obj.L;
      end
      if isempty(obj.m)
        obj.m = obj.M;
      end
      if (~isempty(obj.M) && ~isempty(obj.m))
        obj.L = max([abs(obj.m), abs(obj.M)]);
      end
    end
    function update_solver_inputs(obj)
      MAIN_INPUT_NAMES = {...
        'L', 'M', 'm', 'opt_tol', 'x0', 'iter_limit', 'time_limit', ...
        'prod_fn', 'norm_fn'
      };
      % Create the input params (order is important here!)
      obj.solver_input_params = obj.solver_hparams;
      for i=1:length(MAIN_INPUT_NAMES)
        NAME = MAIN_INPUT_NAMES{i};
        obj.solver_input_params.(NAME) = obj.(NAME);
      end
    end
    
    % Update function.
    function update(obj)
      if obj.i_update_oracle
        obj.update_oracle;
      end
      if obj.i_update_curvatures
        obj.update_curvatures;
      end
      if obj.i_update_solver_inputs
        obj.update_solver_inputs;
      end
      obj.status = 1; % MODEL LOADED
    end
    
    % Viewing functions.
    function view_flags(obj)
      flags.i_verbose = obj.i_verbose;
      flags.i_update_oracle = obj.i_update_oracle;
      flags.i_update_curvatures = obj.i_update_curvatures;
      flags.i_update_solver_inputs = obj.i_update_solver_inputs;
      disp(flags);
    end
    function view_model_limits(obj)
      limits.iter_limit = obj.iter_limit;
      limits.time_limit = obj.time_limit;
      limits.opt_tol = obj.opt_tol;
      disp(limits);
    end
    function view_topology(obj)
      topo.norm_fn = obj.norm_fn;
      topo.prod_fn = obj.prod_fn;
      disp(topo);
    end
    function view_solution(obj)
      solns.x = obj.x;
      solns.v = obj.v;
      solns.f_at_x = obj.f_at_x;
      solns.norm_of_v = obj.norm_of_v;
      disp(solns);
    end
    function view_curvatures(obj)
      curvatures.L = obj.L;
      curvatures.M = obj.M;
      curvatures.m = obj.m;
      disp(curvatures);
    end
    function view_history(obj)
      disp(obj.history);
    end
    
  end % End ordinary methods.

end % End of classdef