Installation
============

.. note::

    You will need a valid MATLAB installation and license in order to use NC-OPT's functions, classes, solvers, and frameworks.

To install NC-OPT, simply clone the latest version from GitHub by running the following command in your terminal:

``git clone https://github.com/wwkong/nc_opt.git``

After downloading the files, start up MATLAB and run the ``init.m`` in the root directory, which will add the paths to all of the functions in the library. You can check if everything is properly configured by running one of the examples in ``example`` such as:

``./nc_opt/examples/unconstrained/basic_convex_qp.m``