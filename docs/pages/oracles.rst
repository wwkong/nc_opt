Oracles
=======

This section documents the constructors, properties, and methods of the ``Oracle`` and ``SpectralOracle`` classes. Hidden and inherited properties are not shown here for simplicity.

.. warning::
    
    The oracle classes in this section inherit from the copyable ``handle`` class meaning that all assignments made on these objects will only hold references to the underlying data. **In other words, more than one variable can refer to the same underlying object!** For example, consider the following code:

    .. code-block:: matlab

        a = Oracle;
        a.f_s = @() 1;
        b = a;
        b.f_s = @() 10;
        disp(a.f_s()) 

    The above code will return a value of ``10`` to the terminal, instead of ``1`` because the underlying data of ``a`` was modified through ``b``.

    To make an explicit copy of an oracle object, **without creating a reference to the original object**, one must invoke the ``copy()`` function before the assignment. For example, consider the following code, which is a modification of the previous example:

    .. code-block:: matlab

        a = Oracle;
        a.f_s = @() 1;
        b = copy(a);
        b.f_s = @() 10;
        disp(a.f_s()) 

    The above code, this time, will return a value ``1`` to the terminal as ``a`` and ``b`` are now refering to different objects.


.. automodule:: src.oracles

.. autoclass:: Oracle
    :members:
    :show-inheritance:

.. autoclass:: SpectralOracle
    :members:
    :show-inheritance: