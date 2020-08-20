Solvers
=======

This section documents the solvers that available for solving unconstrained composite optimization problems. 
For simplicity, we only list the parameter inputs that affect the behavior of the method for a fixed optimization problem. All other inputs should be passed from the model object that is calling the solver.

.. automodule:: src.solvers

.. autofunction:: AC_ACG(oracle, params)

.. autofunction:: ACG(oracle, params)

.. autofunction:: ADAP_FISTA(oracle, params)

.. autofunction:: AG(oracle, params)

.. autofunction:: AIPP(oracle, params)

.. autofunction:: ECG(oracle, params)

.. autofunction:: NC_FISTA(oracle, params)

.. autofunction:: UPFAG(oracle, params)