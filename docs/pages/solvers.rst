Solvers
=======

This section documents the solvers that available for solving unconstrained composite optimization problems. 
For simplicity, we only list the parameter inputs that affect the behavior of the method for a fixed optimization problem. All other inputs should be passed from the model object that is calling the solver.

.. note::

    All solvers have a parameter ``i_logging`` that, when set to ``true``, will log the function value, (inner) iteration number, and time at each (outer) iteration. These values are then added to the ``history`` struct that is output by each solver.

.. automodule:: src.solvers

Auxiliary Solvers
-----------------

These solvers are used as (possible) subroutines for the other solvers of this section.

.. autofunction:: ACG(oracle, params)

NCO Solvers
-----------

These solvers are used for nonconvex composite optimization problems.

.. autofunction:: AC_ACG(oracle, params)

.. autofunction:: ADAP_FISTA(oracle, params)

.. autofunction:: AG(oracle, params)

.. autofunction:: AIPP(oracle, params)

.. autofunction:: ECG(oracle, params)

.. autofunction:: NC_FISTA(oracle, params)

.. autofunction:: UPFAG(oracle, params)

Spectral NCO Solvers
--------------------

These solvers are used for spectral nonconvex composite optimization problems.

.. autofunction:: DA_ICG(spectral_oracle, params)

.. autofunction:: IA_ICG(spectral_oracle, params)
