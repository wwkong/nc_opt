Introduction
============

`NC-OPT
<https://github.com/wwkong/nc_opt>`_ is an open source software package for **N**\ onconvex **C**\ omposite **Opt**\ imization in MATLAB. Its main purpose is to: (a) provide a general API for creating and models based on a first-order oracle framework; and (b) leverage the fast matrix subroutines in MATLAB to solve these models. The interface of NC-OPT is loosely based on the well-known `Gurobi
<https://www.gurobi.com/documentation>`_ API. 

Currently, the following problems classes are supported:

	- **Unconstrained Composite Optimization**
	- **Linearly Set Constrained Composite Optimization**
	- **Convex Cone Constrained Composite Optimization** (WIP)

Instances of the above classes include *semidefinite programming*, *convex programming*, *cone programming*, *linear and quadratic programmin*, and *nonconvex programming*.

Features
--------

The features of NC-OPT can be split into the following categories.

**Oracles**:
A class that abstracts the idea of a first-order oracle at a point. It contains any one of the following oracles: function value, function gradient, and proximal oracle.

**Solvers**:
A collection of optimzation solvers that solve the problem associated with a particular first-order oracle. Some examples include: composite gradient method, accelerated composite gradient method, accelerated inexact proximal point method.

**Frameworks**:
A collection of optimization frameworks that use a solver to solve a constrained composite optimization model. Some examples include: quadratic penalty framework, augmented Lagrangian framework, and dampened augmented Lagrangian framework.

**Composite Models**:
A class that abstracts the idea of a composite optimization model. It contains properties that describe various aspects of the model, including: objective function, constraint(s), solver(s), framework(s), tolerances, and logging.
