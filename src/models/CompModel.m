classdef CompModel < matlab.mixin.Copyable
  % An abstract model class for composite optimization.
  %
  % Note:
  % 
  %   The following properties are necessary before the ``optimize()`` method
  %   can be called to solve the model: either {``f_s``, ``grad_f_s``} or 
  %   ``oracle``, ``L``, ``x0``, and ``solver``. 
  %
  % Attributes:
  %
  %   f_s (function handle): A one argument function that, when evaluated at 
  %     a point $x$, outputs $f_s(x)$. Defaults to ``None``.
  %
  %   grad_f_s (function handle): A one argument function that, when evaluated 
  %     at a point $x$, outputs $\nabla f_s(x)$. Defaults to ``None``.
  %
  %   f_n (function handle): A one argument function that, when evaluated at 
  %     a point $x$, outputs $f_n(x)$. Defaults to ``@(x) zeros(size(x))``.
  %
  %   prox_f_n (function handle): A two argument function that, when evaluated
  %     at $\{x,\lambda\}$, outputs $${\rm prox}_{\lambda f_n}(x) := 
  %     {\rm argmin}_u \left\{\lambda f_n(u) + \frac{1}{2}\|u-x\|^2\right\}.$$
  %     Defaults to ``@(x, lam) x``.
  %
  %   L (double): A **required** Lipschitz constant of $\nabla f_s$.
  %     Defaults to ``None``.
  %
  %   M (double): An **optional** upper curvature constant of $\nabla f_s$,
  %     i.e., a constant satisfying $$f_s(z) - f_s(u) - \langle \nabla f_s(u), 
  %     z - u \rangle \leq \frac{M}{2}\|u-z\|^2 \quad \forall u,z \in 
  %     {\rm dom}\, f_n.$$
  %     Defaults to ``None``.
  %
  %   m (double): An **optional** lower curvature constant of $\nabla f_s$,
  %     i.e., a constant satisfying $$f_s(z) - f_s(u) - \langle \nabla f_s(u), 
  %     z - u \rangle \geq -\frac{m}{2}\|u-z\|^2 \quad \forall u,z \in 
  %     {\rm dom}\, f_n.$$
  %     Defaults to ``None``.
  %
  %   x0 (double vector): A **required** starting point of the solver.
  %     Defaults to ``None``.
  %
  %   oracle (Oracle): An **optional** oracle that may be given to the model, 
  %     which will be passed to the solver if the flag i_update_oracle is 
  %     False. Defaults to ``None``.
  %
  %   solver (function handle): A **required** function handle to a solver 
  %     that solves unconstrained composite optimization problems (see 
  %     src.solvers). Defaults to ``None``.
  %
  %   model (struct): The model struct output by the solver. Defaults to 
  %     ``None``.
  %
  %   history (struct): The history struct output by the solver. Defaults
  %     to ``None``.
  %
  %   solver_hparams (struct): Contains additional hyperparameters that will
  %     be given to the solver when it is called. Defaults to ``None``.
  %
  %   iter_limit (int): An upper bound on the total number of gradient/
  %     function/proximal evaluations done by the solver. Defaults to
  %     ``Inf``.
  %
  %   time_limit (double): An upper bound on the total time used by the solver.
  %     Defaults is ``Inf``.
  %
  %   opt_tol (double): The tolerance for optimality, i.e., 
  %     $\rho=\text{opt_tol}$. Defaults to ``1e-6``.
  %
  %   opt_type (character vector): Is either 'relative' or 'absolute'. If
  %     it is 'absolute', then the optimality condition is $\|v\|\leq 
  %     \text{opt_tol}$. If it is 'relative', then the optimality condition
  %     is $\|v\|/[1+\|\nabla f_s(x_0)\|] \leq\text{opt_tol}$. Defaults to 
  %     ``'absolute'``.
  %
  %   prod_fn (function handle): A two argument function that, when evaluated
  %     at $\{a, b\}$, outputs the inner product $\langle a,b \rangle$. 
  %     Defaults to the Euclidean inner product, i.e., 
  %     ``@(a,b) sum(dot(a, b))``.
  %
  %   norm_fn (function handle): A one function that, when evaluated at a 
  %     point $a$, outputs $\|a\|$. Defaults to the Frobenius norm, i.e., 
  %     ``norm(a, 'fro')``.
  %
  %   iter (int): The number of gradient/function/proximal evaluations done
  %     by the solver. Defaults to ``0``. This property cannot be set by the
  %     user.
  %
  %   runtime (double): The total runtime used by the solver. Defaults to 
  %     ``0.0``. This property cannot be set by the user.
  %
  %   x (double vector): The stationary point returned by the solver. Defaults
  %     to ``None``. This property cannot be set by the user.
  %
  %   v (double vector): The stationarity residual returned by the solver.
  %     Defaults to ``None``. This property cannot be set by the user.

  % -----------------------------------------------------------------------
  %% CONSTRUCTORS
  % -----------------------------------------------------------------------
  methods
    function obj = CompModel(varargin)
      % The constructor for the CompModel class. It has three ways to 
      % initialize: **(i)** invoking ``CompModel`` creates a CompModel
      % object with the default properties; **(ii)** invoking 
      % ``CompModel(in_oracle)`` creates a CompModel object with the oracle 
      % property set to the input oracle; and **(iii)** invoking
      % ``CompModel(f_s, f_n, grad_f_s, prox_f_n)`` creates an 
      % Oracle object with the properties ``f_s``, ``f_n``, ``grad_f_s``, and 
      % ``prox_f_n`` filled by the corresponding input.
      %
      if (nargin == 0) % Default constructor
        % Do nothing here
      elseif (nargin == 1) % Oracle-based constructor
        obj.oracle = varargin{1};
        % Make sure we do not overwrite with empty functions during the
        % optimization of the model.
        obj.i_update_oracle = false;
      elseif (nargin == 4) % Function-based constructor
        obj.f_s = varargin{1};
        obj.f_n = varargin{2};
        obj.grad_f_s = varargin{3};
        obj.prox_f_n = varargin{4};        
      else
        error('Incorrect number of arguments');
      end    
    end
  end
  
  % -----------------------------------------------------------------------
  %% GLOBAL PROPERTIES
  % -----------------------------------------------------------------------  
  
  % Visible global function properties.
  properties (SetAccess = public)
    f_s
    grad_f_s
    f_n = @(x) zeros(size(x))
    prox_f_n = @(x, lam) x
    oracle
    solver
    solver_hparams
    opt_type {mustBeMember(opt_type, {'relative', 'absolute'})} = ...
      'absolute'
  end

  properties (SetAccess = protected)
    iter  int64 {mustBeReal, mustBeFinite} = 0
    runtime double {mustBeReal, mustBeFinite} = 0.0
  end
  
  % Invisible global function properties.
  properties (Access = public)
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
  properties (SetAccess = public)
    m double {mustBeReal, mustBeFinite} = []
    M double {mustBeReal, mustBeFinite} = []
  end
  
  % Invisible and protected solver properties 
  properties (SetAccess = protected, Hidden = true)
    solver_input_params
    internal_opt_tol
  end
  
  % -----------------------------------------------------------------------
  %% MODEL-RELATED PROPERTIES
  % -----------------------------------------------------------------------
  
  % Visible model properties.
  properties (SetAccess = protected)
    x double {mustBeReal}
    v double {mustBeReal}
  end
  
  % Invisible model properties.
  properties (SetAccess = protected, Hidden = true)
    f_at_x double {mustBeReal, mustBeFinite}
    norm_of_v double {mustBeReal}
    status (1,1) int32 {mustBeInteger} = 100
  end
  
  % Solver outputs.
  properties (SetAccess = protected)
    model
    history
  end
  
  % Invisible model flags.
  properties (SetAccess = public, Hidden = true)
    i_verbose (1,1) {mustBeNumericOrLogical} = true
    i_reset (1,1) {mustBeNumericOrLogical} = true
    i_update_oracle (1,1) {mustBeNumericOrLogical} = true
    i_update_curvatures (1,1) {mustBeNumericOrLogical} = true
    i_update_solver_inputs (1,1) {mustBeNumericOrLogical} = true
    i_update_tolerances (1,1) {mustBeNumericOrLogical} = true
  end
  
  % Toleranace and limit properties.
  properties (SetAccess = public)
    iter_limit (1,1) double {mustBeReal} = Inf
    time_limit (1,1) double {mustBeReal} = Inf
    opt_tol (1,1) double {mustBeReal, mustBePositive} = 1e-6
  end
    
  % -----------------------------------------------------------------------
  %% PRIMARY METHODS
  % -----------------------------------------------------------------------
  
  % Static methods.
  methods (Access = protected, Static = true, Hidden = true)
    
    % Converts status codes into strings
    function status_string = parse_status(status_num)
      status_map = containers.Map('KeyType', 'int32', 'ValueType', 'char');
      status_map(-101) = 'STATIONARITY_CONDITION_FAILED';
      status_map(100)  = 'EMPTY_MODEL';
      status_map(101)  = 'MODEL_LOADED';
      status_map(102)  = 'STATIONARITY_ACHIEVED';
      status_map(103)  = 'TIME_LIMIT_EXCEEDED';
      status_map(104)  = 'ITER_LIMIT_EXCEEDED';
      if isKey(status_map, status_num)
        status_string = status_map(status_num);
      else
        status_string = [];
      end
    end
  end % End static methods.
  
  % Ordinary methods.
  methods (Access = public, Hidden = true)
        
    % ---------------------------------------------------------------------
    %% MAIN OPTIMIZATION FUNCTIONS.
    % ---------------------------------------------------------------------
    
    % Key subroutines.
    function reset(obj)
      obj.iter = 0;
      obj.runtime = 0.0;
      obj.x = [];
      obj.v = [];
      obj.f_at_x = [];
      obj.norm_of_v = [];
    end
    function check_field(obj, fname)
      if isempty(obj.(fname))
        error(['Missing the required property: ', fname]);
      end
    end
    function check_inputs(obj)
      obj.check_field('solver');
      obj.check_field('L');
      obj.check_field('x0');
    end
    function pre_process(obj)
      % Start a new job
      if (obj.i_reset)
        obj.reset;
      end
      obj.update;
      obj.check_inputs;
    end
    function call_solver(obj)
      [obj.model, obj.history] = ...
          obj.solver(obj.oracle, obj.solver_input_params);
    end
    function get_status(obj)
      % Main conditions
      if (obj.norm_of_v <= obj.internal_opt_tol)
        obj.status = 102; % STATIONARITY_ACHIEVED
      else
        obj.status = -101; % STATIONARITY_CONDITION_FAILED
      end
      % Limiting statuses
      if isfield(obj.history, 'runtime')
        obj.runtime = obj.runtime + obj.history.runtime;
        if (obj.runtime >= obj.time_limit)
          obj.status = 103; % TIME_LIMIT_EXCEEDED
        end
      end
      if isfield(obj.history, 'iter')
        obj.iter = obj.iter + obj.history.iter;
        if (obj.iter >= obj.iter_limit)
          obj.status = 104; % ITER_LIMIT_EXCEEDED
        end
      end
    end
    function post_process(obj)
      obj.x = obj.model.x;
      obj.v = obj.model.v;
      o_at_x = obj.oracle.eval(obj.x);
      obj.f_at_x = o_at_x.f_s() + o_at_x.f_n();
      obj.norm_of_v = obj.norm_fn(obj.v);
    end
    
    % Logging functions.
    function log_input(obj)
      word_len = 8;
      prefix = ['%-' num2str(word_len) 's = '];
      fprintf('» Solving model with: \n');
      fprintf([prefix, '%s\n'], 'SOLVER', func2str(obj.solver));
      fprintf([prefix, '%.2e\n'], 'L', obj.L);
      fprintf([prefix, '(%.2e, %.2e)\n'], '(M, m)', obj.M, obj.m);
      fprintf([prefix, '%.2e\n'], 'OPT_TOL', obj.opt_tol);
    end
    function log_output(obj)
      word_len = 16;
      prefix = ['%-' num2str(word_len) 's = '];
      fprintf('» Model terminated with: \n');
      fprintf([prefix, '%s\n'], 'STATUS', obj.parse_status(obj.status));
      fprintf([prefix, '%.2e\n'], 'FUNCTION VALUE', obj.f_at_x);
      fprintf([prefix, '%.2e\n'], 'NORM OF V', obj.norm_of_v);
      fprintf([prefix, '%.2e\n'], 'RUNTIME', obj.runtime);     
      fprintf([prefix, '%.0f\n'], 'ITERATION COUNT', obj.iter);    
    end
    
    % Sub-update functions.
    function update_oracle(obj)
      obj.oracle = Oracle(obj.f_s, obj.f_n, obj.grad_f_s, obj.prox_f_n);
    end
    function update_curvatures(obj)
      if (~isempty(obj.M) && ~isempty(obj.m))
        obj.L = max([abs(obj.m), abs(obj.M)]);
      else
        if isempty(obj.M)
          obj.M = obj.L;
        end
      	if isempty(obj.m)
          obj.m = obj.L;
        end
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
    function update_tolerances(obj)
      obj.internal_opt_tol = obj.opt_tol;
      if strcmp(obj.opt_type, 'relative')
        o_at_x0 = obj.oracle.eval(obj.x0);
        grad_f_s_at_x0 = o_at_x0.grad_f_s();
        obj.internal_opt_tol = ...
          obj.internal_opt_tol * (1 + obj.norm_fn(grad_f_s_at_x0));
      end
      obj.solver_input_params.opt_tol = obj.internal_opt_tol;
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
      if obj.i_update_tolerances
        obj.update_tolerances;
      end
      obj.status = 101; % MODEL LOADED
    end
    
  end
  
  % Public methods.
  methods (Access = public)
    
    function optimize(obj)
      % A zero argument function that, when evaluated, calls the model's 
      % solver to solve the underlying optimization problem.
      obj.pre_process;
      if (obj.i_verbose)
        fprintf('\n');
        obj.log_input;
        fprintf('\n');
      end
      obj.call_solver;
      obj.post_process;
      obj.get_status;
      if (obj.i_verbose)
        fprintf('\n');
        obj.log_output;
        fprintf('\n');
      end
    end
    
    % Viewing functions.
    function view_flags(obj)
      % Displays flags that control the behavior of ``optimize()``.
      flags.i_verbose = obj.i_verbose;
      flags.i_reset = obj.i_reset;
      flags.i_update_oracle = obj.i_update_oracle;
      flags.i_update_curvatures = obj.i_update_curvatures;
      flags.i_update_solver_inputs = obj.i_update_solver_inputs;
      flags.i_update_tolerances = obj.i_update_tolerances;
      disp(flags);
    end
    function view_topology(obj)
      % Displays functions related to the underlying inner product space.
      topo.norm_fn = obj.norm_fn;
      topo.prod_fn = obj.prod_fn;
      disp(topo);
    end
    function view_solution(obj)
      % Displays metrics related to the obtained solution by the solver.
      solns.x = obj.x;
      solns.v = obj.v;
      solns.f_at_x = obj.f_at_x;
      solns.norm_of_v = obj.norm_of_v;
      disp(solns);
    end
    function view_curvatures(obj)
      % Displays the underlying curvatures.
      curvatures.L = obj.L;
      curvatures.M = obj.M;
      curvatures.m = obj.m;
      disp(curvatures);
    end
    function view_history(obj)
      % Displays the history structure output by the solver.
      disp(obj.history);
    end
    
  end % End ordinary methods.

end % End of classdef