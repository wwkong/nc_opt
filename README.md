**NC-OPT** is an open source software package for **N**onconvex **C**omposite **Opt**imization in MATLAB. Its main purpose is to: (a) provide a general API for creating and models based on a first-order oracle framework; and (b) leverage the fast matrix subroutines in MATLAB to solve these models. The interface of NC-OPT is loosely based on the well-known [Gurobi](https://www.gurobi.com/documentation) API. 

Currently, the following problems classes are supported:

- **Unconstrained Composite Optimization**
- **Linearly Set Constrained Composite Optimization**
- **Convex Cone Constrained Composite Optimization** (WIP)

Instances of the above classes include *semidefinite programming*, *convex programming*, *cone programming*, *linear and quadratic programmin*, and *nonconvex programming*.

The user guide for NC-OPT can be found [here](https://nc-opt.readthedocs.io/).