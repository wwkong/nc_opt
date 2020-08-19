classdef Oracle < matlab.mixin.Copyable
  % An oracle abstact class for unconstrained composite optimization.
  
  % -----------------------------------------------------------------------
  %% CONSTRUCTORS
  % -----------------------------------------------------------------------
  methods
    function obj = Oracle(varargin)
      % The constructor for the Oracle class. Has two ways to initialize: 
      % **(i)** ``Oracle(f_s, f_n, grad_f_s, prox_f_n)`` creates an Oracle
      % object with the properties ``f_s``, ``f_n``, ``grad_f_s``, and 
      % ``prox_f_n`` filled by the relevant input; and 
      % **(ii)** ``Oracle(eval_fn)`` creates an Oracle object which, 
      % when evaluated at a point $x$, updates the properties ``f_s``, 
      % ``f_n``, ``grad_f_s``, and ``prox_f_n`` as follows:
      %
      % .. code-block:: matlab
      %
      %   f_s = eval_fn(x).f_s 
      %   f_n = eval_fn(x).f_n 
      %   grad_f_s = eval_fn(x).grad_f_s
      %   prox_f_n = eval_fn(x).prox_f_n
      %
      if (nargin == 1) % eval constructor
        obj.eval_proxy_fn = varargin{1};        
      elseif (nargin == 4) % basic oracle constructor
        obj.eval_proxy_fn = @(x) struct(...
          'f_s', @() feval(varargin{1}, x), ...
          'f_n', @() feval(varargin{2}, x), ...
          'grad_f_s', @() feval(varargin{3}, x), ...
          'prox_f_n', @(lam) feval(varargin{4}, x, lam));        
      else
        error('Incorrect number of arguments');
      end
    end
  end
  
  % -----------------------------------------------------------------------
  %% ORACLES
  % -----------------------------------------------------------------------
  
  properties (SetAccess = public)
    f_s % A zero argument function that, when evaluated, outputs $f_s(x)$.
    grad_f_s % A zero argument function that, when evaluated, outputs $\nabla f_s(x)$.
    f_n % A zero argument function that, when evaluated, outputs $f_n(x)$.
    prox_f_n % A single argument function that takes in an argument $\lambda$ and outputs $${\rm prox}_{\lambda f_n}(x) := {\rm argmin}_u \left\{\lambda f_n(u) + \frac{1}{2}\|u-x\|^2\right\}.$$
  end
  
  properties (Access = public)
    prod_fn = @(a,b) sum(dot(a, b)) % A function that takes in arguments ``{a, b}`` and outputs the inner product $\langle a,b \rangle$. Defaults to the Euclidean inner product.
    norm_fn = @(a) norm(a, 'fro') % A function that takes in one argument ``a`` and outputs the norm $\|a\|$. Defaults to the Frobenius norm.
  end
  
  % Evaluator oracles.
  properties (SetAccess = protected, Hidden = true)
    eval_proxy_fn
  end
  
  % -----------------------------------------------------------------------
  %% METHODS
  % -----------------------------------------------------------------------
  
  % Helper methods that depend on the object.
  methods (Hidden = true)
    
    function prox_struct = add_prox_to_struct(obj, base_struct, x, x_hat)
        % Adds a prox term of the form (1 / 2) * ||x - x_hat|| ^2 
        % to an oracle struct.
        prox_struct = base_struct;
        prox_struct.f_s = @() ...
          base_struct.f_s() + (1 / 2) * obj.norm_fn(x - x_hat) ^ 2;
        prox_struct.grad_f_s = @() ...
          base_struct.grad_f_s() + (x - x_hat);
        % Account for hidden oracles.
        if isfield(base_struct, 'f_s_at_prox_f_n')
          prox_struct.f_s_at_prox_f_n = @(lam) ...
            base_struct.f_s_at_prox_f_n(lam) + ...
            (1 / 2) * obj.norm_fn(x - x_hat) ^ 2;
        end
        if isfield(base_struct, 'grad_f_s_at_prox_f_n')
          prox_struct.grad_f_s_at_prox_f_n = @(lam) ...
            base_struct.grad_f_s_at_prox_f_n(lam) + (x - x_hat);
        end
    end  
  end
  
  % Helper methods that are independent of the object.
  methods (Static = true, Hidden = true)
      
    function scaled_struct = scale_struct(base_struct, alpha)
      % Scales an oracle by alpha.
      scaled_struct = base_struct;
      scaled_struct.f_s = @() alpha * base_struct.f_s();
      scaled_struct.grad_f_s = @() alpha * base_struct.grad_f_s();
      scaled_struct.f_n = @() alpha * base_struct.f_n();
      scaled_struct.prox_f_n = @(lam) base_struct.prox_f_n(lam * alpha);
      % Account for hidden oracles.
      if isfield(base_struct, 'f_s_at_prox_f_n')
        scaled_struct.f_s_at_prox_f_n = @(lam) ...
          alpha * base_struct.f_s_at_prox_f_n(lam * alpha);
      end
      if isfield(base_struct, 'f_n_at_prox_f_n')
        scaled_struct.f_n_at_prox_f_n = @(lam) ...
          alpha * base_struct.f_n_at_prox_f_n(lam * alpha);
      end
      if isfield(base_struct, 'grad_f_s_at_prox_f_n')
        scaled_struct.grad_f_s_at_prox_f_n = @(lam) ...
          alpha * base_struct.grad_f_s_at_prox_f_n(lam * alpha);
      end
    end
    
  end
  
  % Primary methods.
  methods
    
    function obj = eval(obj, x)
      % Evaluates the oracle at a point ``x`` and updates the properties 
      % ``f_s``, ``f_n``, ``grad_f_s``, and ``prox_f_n`` to be at this point.
      oracle_outputs = obj.eval_proxy_fn(x);
      obj.f_s = @() oracle_outputs.f_s();
      obj.f_n = @() oracle_outputs.f_n();
      obj.grad_f_s = @() oracle_outputs.grad_f_s();
      obj.prox_f_n = @(lam) oracle_outputs.prox_f_n(lam);
      % Add hidden oracles if they exist.
      if isfield(oracle_outputs, 'f_s_at_prox_f_n')
        obj.f_s_at_prox_f_n = ...
          @(lam) oracle_outputs.f_s_at_prox_f_n(lam);
      end
      if isfield(oracle_outputs, 'f_n_at_prox_f_n')
        obj.f_n_at_prox_f_n = ...
          @(lam) oracle_outputs.f_n_at_prox_f_n(lam);
      end
      if isfield(oracle_outputs, 'grad_f_s_at_prox_f_n')
        obj.grad_f_s_at_prox_f_n = ...
          @(lam) oracle_outputs.grad_f_s_at_prox_f_n(lam);
      end
    end
    
    function scale(obj, alpha)
      % Modified the suboracles by multiplying by a positive constant 
      % ``alpha``. That is, the properties `f_s``, ``f_n``, ``grad_f_s``, and 
      % ``prox_f_n`` are updated as follows:
      %
      % .. code-block:: matlab
      %
      %   f_s() = alpha * f_s();
      %   f_n() = alpha * f_n(); 
      %   grad_f_s() = alpha * grad_f_s();
      %   prox_f_n(lambda) = alpha * prox_f_n(alpha * lambda);
      %
      eval_proxy_fn_cpy = obj.eval_proxy_fn;
      function out_struct = scaled_eval(x)
        out_struct = eval_proxy_fn_cpy(x);
        out_struct = obj.scale_struct(out_struct, alpha);
      end
      obj.eval_proxy_fn = @(x) scaled_eval(x);
    end
      
    function add_prox(obj, x_hat)
      % Modified the suboracles by adding a prox term at a point ``x_hat``. 
      % That is, the properties `f_s``, ``f_n``, ``grad_f_s``, and 
      % ``prox_f_n`` are updated as follows:
      %
      % .. code-block:: matlab
      %
      %   f_s() = f_s() + (1 / 2) * norm_fn(x - x_hat) ^ 2;
      %   grad_f_s() = grad_f_s() + (x - x_hat);
      %
      eval_proxy_fn_cpy = obj.eval_proxy_fn;
      function out_struct = prox_eval(x)
        out_struct = eval_proxy_fn_cpy;
        out_struct = obj.add_prox_to_struct(out_struct, x, x_hat);
      end
      obj.eval_proxy_fn = @(x) prox_eval(x);
    end
    
    function proxify(obj, alpha, x_hat)
      % Modified the suboracles by multiplying by a positive constant 
      % ``alpha`` and then adding a prox term at a point ``x_hat``. 
      % That is, the properties `f_s``, ``f_n``, ``grad_f_s``, and 
      % ``prox_f_n`` are updated as follows:
      %
      % .. code-block:: matlab
      %
      %   f_s() = alpha * f_s() + (1 / 2) * norm_fn(x - x_hat) ^ 2;
      %   f_n() = alpha * f_n();
      %   grad_f_s() = alpha * grad_f_s() + (x - x_hat);
      %   prox_f_n(lambda) = prox_f_n(alpha * lambda);
      %
      eval_proxy_fn_cpy = obj.eval_proxy_fn;
      function combined_struct = proxify_eval(x)
        out_struct = eval_proxy_fn_cpy(x);
        scaled_struct = obj.scale_struct(out_struct, alpha);
        combined_struct = obj.add_prox_to_struct(scaled_struct, x, x_hat);
      end
      obj.eval_proxy_fn = @(x) proxify_eval(x);
    end
    
    function add_smooth_oracle(obj, input_oracle)
      % Modified the suboracles by adding the smooth suboracles of an input
      % oracle to the smooth suboracles of the current oracle. 
      % That is, the properties `f_s``, ``f_n``, ``grad_f_s``, and 
      % ``prox_f_n`` are updated as follows:
      %
      % .. code-block:: matlab
      %
      %   f_s() = f_s() + input_oracle.f_s();
      %   grad_f_s() = grad_f_s() + input_oracle.grad_f_s();
      %
      eval_proxy_fn_cpy = obj.eval_proxy_fn;
      function combined_struct = combined_eval(x)
        o_struct = eval_proxy_fn_cpy(x);
        input_struct = input_oracle.eval(x);
        combined_struct = o_struct;
        combined_struct.f_s = @() o_struct.f_s() + input_struct.f_s();
        combined_struct.grad_f_s = @() ...
          o_struct.grad_f_s() + input_struct.grad_f_s();
        % Account for hidden oracles.
        if isfield(o_struct, 'f_s_at_prox_f_n')
          combined_struct.f_s_at_prox_f_n = @(lam) ...
            o_struct.f_s_at_prox_f_n(lam) + ...
            input_struct.f_s_at_prox_f_n(lam);
        end
        if isfield(o_struct, 'grad_f_s_at_prox_f_n')
          combined_struct.grad_f_s_at_prox_f_n = @(lam) ...
            o_struct.grad_f_s_at_prox_f_n(lam) + ...
            input_struct.grad_f_s_at_prox_f_n(lam);
        end
      end
      obj.eval_proxy_fn = @(x) combined_eval(x);
    end
    
    function [f_s, f_n, grad_f_s, prox_f_n] = decompose(obj)
      % A zero argument method that, when evaluated, returns variants of the 
      % properties `f_s``, ``f_n``, ``grad_f_s``, and ``prox_f_n``, where 
      % an additional argument ``x`` is added to the beginning of the 
      % of the list of function inputs. For example, the value ``f = f_s(x)``,
      % computed from the output function, is equivalent to:
      %
      % .. code-block:: matlab
      %
      %   my_oracle.eval(x);
      %   f = my_oracle.f_s();
      %
      % where my_oracle refers to the oracle object that is calling 
      % ``decompose()``.
      f_s = @(x) feval(subsref(obj.eval(x), ...
        struct('type','.','subs','f_s')));
      f_n = @(x) feval(subsref(obj.eval(x), ...
        struct('type','.','subs','f_n')));
      grad_f_s = @(x) feval(subsref(obj.eval(x), ...
        struct('type','.','subs','grad_f_s')));

      % Multi-input functions
      prox_f_n = @(x, lam) feval(subsref(obj.eval(x), ...
        struct('type','.','subs','prox_f_n')), lam);

    end
    
  end
  
end