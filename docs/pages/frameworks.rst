Frameworks
==========

This section documents the frameworks that available for solving unconstrained composite optimization problems. 
For simplicity, we only list the parameter inputs that affect the behavior of the framework for a fixed optimization problem. All other inputs should be passed from the model object that is calling the framework.

.. note::

    All frameworks have a parameter ``i_logging`` that, when set to ``true``, will 
    log the function value, (inner) iteration number, and time at each (outer) iteration. These values are then added to the ``history`` struct that is output by each framework.

.. automodule:: src.frameworks

.. autofunction:: penalty(solver, oracle, params)

.. autofunction:: IAPIAL(~, oracle, params)

.. autofunction:: iALM(~, oracle, params)
