classdef SpectralOracle < Oracle
  % An abstact oracle class for unconstrained composite optimization.
  % 
  % Note:
  %
  %   ??? discuss the two argument nature of the oracle ???
  %
  %   with input {X, X_vec}
  %   where X is the matrix point and X_vec is the spectral vector
  %   (if available). If the latter is empty, this vector will be
  %   computed using the property decomp_fn applied to X.
  %
  % Attributes:
  %
  %   spectral_grad_f2_s (function handle): ???.
  %
  %   spectral_prox_f_n (function handle): ???.
  %
  %   f1_s (function handle): ???.
  %
  %   f2_s (function handle): ???.
  %
  %   grad_f1_s (function handle): ???.
  %
  %   grad_f2_s (function handle): ???.
  %
  %   decomp_fn (function handle): ???.
  %
  
  % -----------------------------------------------------------------------
  %% CONSTRUCTORS
  % -----------------------------------------------------------------------
  
  methods
    
    function obj = SpectralOracle(varargin)
      % TODO: ??? complete the documentation ???
      % Overloaded constructor. The main difference is that the one 
      % argument constructor should take a two argument function as input. 
      if (nargin == 0) % default constructor 
        obj.eval_proxy_fn = @(X, X_vec) struct(...
          'f_s', @() 0, ...
          'f_n', @() 0, ...
          'grad_f_s', @() zeros(size(X)), ...
          'prox_f_n', @(lam) zeros(size(X)), ...
          'spectral_grad_f2_s', @() zeros(size(X_vec)), ...
          'spectral_prox_f_n', @(lam) zeros(size(X_vec)), ...
          'f1_s', @() 0, ...
          'f2_s', @() 0, ...
          'grad_f1_s', @() zeros(size(X_vec)), ...
          'grad_f2_s', @() zeros(size(X_vec)), ...
          'decomp_fn', @(Z) svd(Z, 'econ'));
      elseif (nargin == 1) % eval constructor
        obj.eval_proxy_fn = varargin{1};
      elseif (nargin == 4) % basic oracle constructor
        obj.eval_proxy_fn = @(X, X_vec) struct(...
          'f_s', @() feval(varargin{1}, X), ...
          'f_n', @() feval(varargin{2}, X), ...
          'grad_f_s', @() feval(varargin{3}, X), ...
          'prox_f_n', @(lam) feval(varargin{4}, X, lam), ...
          'spectral_grad_f2_s', @() zeros(size(X_vec)), ...
          'spectral_prox_f_n', @(lam) zeros(size(X_vec)), ...
          'f1_s', @() 0, ...
          'f2_s', @() 0, ...
          'grad_f1_s', @() zeros(size(X_vec)), ...
          'grad_f2_s', @() zeros(size(X_vec)), ...
          'decomp_fn', @(Z) svd(Z, 'econ'));
      else
        error(['Incorrect number of arguments: ', num2str(nargin)]);
      end
    end
    
  end
  
  % -----------------------------------------------------------------------
  %% ORACLES
  % -----------------------------------------------------------------------
  
  properties (SetAccess = public)
    spectral_grad_f2_s
    spectral_prox_f_n
    f1_s
    f2_s
    grad_f1_s
    grad_f2_s
    decomp_fn
  end
  
  properties (SetAccess = private, Hidden = true)
    orig_f2_s
    orig_f_n
  end
  
  % -----------------------------------------------------------------------
  %% METHODS
  % -----------------------------------------------------------------------
    
  % Primary methods.
  methods
    
    function obj = eval(obj, X)
      % TODO: ??? overloaded eval method. Calls the spectral evaluator. ???
      obj.spectral_eval(X, []);
    end
    
    function obj = spectral_eval(obj, X, X_vec)
      % Evaluates the oracle at a point ``X`` with singular vectors 
      % ``X_vec`` and updates all of the relevant properties at this point. 
      oracle_outputs = obj.eval_proxy_fn(X, X_vec);
      obj.f_s = @() oracle_outputs.f_s();
      obj.f_n = @() oracle_outputs.f_n();
      obj.grad_f_s = @() oracle_outputs.grad_f_s();
      obj.prox_f_n = @(lam) oracle_outputs.prox_f_n(lam);
      obj.spectral_grad_f2_s = @() oracle_outputs.spectral_grad_f2_s();
      obj.spectral_prox_f_n = @(lam) oracle_outputs.spectral_prox_f_n(lam);
      obj.f1_s = @() oracle_outputs.f1_s();
      obj.f2_s = @() oracle_outputs.f2_s();
      obj.grad_f1_s = @() oracle_outputs.grad_f1_s();
      obj.grad_f2_s = @() oracle_outputs.grad_f2_s();
    end
    
    function vector_linear_proxify(obj, alpha, x_hat)
      % TODO: ??? document this ???
      eval_proxy_fn_cpy = obj.eval_proxy_fn;      
      function combined_struct = vlp_eval(x)
        dim_n = length(x);
        dummy_dg_x = spdiags(x, 0, dim_n, dim_n);
        base_struct = eval_proxy_fn_cpy(dummy_dg_x, x);
        combined_struct.orig_f2_s = @() base_struct.f2_s();
        combined_struct.orig_f_n = @() base_struct.f_n();
        combined_struct.f_s = @() ...
          alpha * base_struct.f2_s() - ...
          obj.prod_fn(x, x_hat) + obj.norm_fn(x) ^ 2 / 2;
        combined_struct.f_n = ...
          @() alpha * base_struct.f_n();
        combined_struct.grad_f_s = @() ...
          alpha * base_struct.spectral_grad_f2_s() + (x - x_hat);
        combined_struct.prox_f_n = ...
          @(lam) base_struct.spectral_prox_f_n(lam * alpha);
      end
      obj.eval_proxy_fn = @(x) vlp_eval(x);
    end
    
    function redistribute_curvature(obj, alpha)
      % TODO: ??? document this ???
      eval_proxy_fn_cpy = obj.eval_proxy_fn; 
      function combined_struct = redist_eval(X, X_vec)
        base_struct = eval_proxy_fn_cpy(X, X_vec);
        combined_struct = base_struct;
        combined_struct.f1_s = ...
          @() spectral_oracle_eval.f1_s() - alpha * norm_fn(x) ^ 2 / 2;
        combined_struct.f2_s = ...
          @() spectral_oracle_eval.f2_s() + alpha * norm_fn(x) ^ 2 / 2;
        combined_struct.grad_f1_s = ...
          @() spectral_oracle_eval.grad_f1_s() - alpha * x;
        combined_struct.grad_f2_s = ...
          @() spectral_oracle_eval.grad_f2_s() + alpha * x;
      end
      obj.eval_proxy_fn = @(X, X_vec) redist_eval(X, X_vec);
    end
    
  end
  
end