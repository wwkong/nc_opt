Examples
========

This section documents the examples solved by NC-OPT. For brevity, only the optimization model is given. Moreover, to avoid repetition, we will also define 

.. math::

    \delta_{C}(x) = 
    \begin{cases}
        0, & x \in C, \\
        \infty, & x\notin C,
    \end{cases}

for any closed convex set $C$ and any point $x$. 

.. note::

    Additional details about these problems can be found in the papers:

    **[1]** Kong, W., Melo, J. G., & Monteiro, R. D. (2019). Complexity of a
    quadratic penalty accelerated inexact proximal point method for solving 
    linearly constrained nonconvex composite programs. *SIAM Journal on 
    Optimization, 29*\(4), 2566-2593.

    **[2]** Kong, W., Melo, J. G., & Monteiro, R. D. (2020). An efficient 
    adaptive accelerated inexact proximal point method for solving linearly 
    constrained nonconvex composite problems. *Computational Optimization and 
    Applications, 76*\(2), 305-346. 

    **[3]** Kong, W., & Monteiro, R. D. (2019). An accelerated inexact 
    proximal point method for solving nonconvex-concave min-max problems. 
    *arXiv preprint arXiv:1905.13433*.

Unconstrained Problems
----------------------

This subsection considers unconstrained composite optimization problems.

:scpt:`src.examples.unconstrained.basic_convex_qp`

This example solves the convex univariate optimization problem

.. math::

    \underset{x}{\text{minimize}}\quad  & \frac{1}{2}x^{2}-x+\frac{1}{2} \\
    \text{subject to}\quad & x\in\mathbb{R}.

:scpt:`src.examples.unconstrained.nonconvex_qp`

This example solves the nonconvex quadratic programming problem

.. math::

    \underset{x}{\text{minimize}}\quad  & -\frac{\xi}{2}\|DBx\|^{2}+\frac{\tau}{2}\|Ax-b\|^{2}+\delta_{\Delta^{n}}(x) \\
    \text{subject to}\quad  & x\in\mathbb{R}^{n},

where $\Delta^n$ is the unit simplex given by

.. math::
    
    \Delta^{n}:=\left\{ x\in\mathbb{R}^{n}:\sum_{i=1}^{n}x_{i}=1,0\leq x\leq1\right\}.

:scpt:`src.examples.unconstrained.nonconvex_qsdp`

This example solves the nonconvex quadratic semidefinite programming problem

.. math::

    \underset{X}{\text{minimize}}\quad  & -\frac{\xi}{2}\|DB(X)\|_{F}^{2}+\frac{\tau}{2}\|A(X)-b\|^{2}_{F}+\delta_{P^{n}}(X) \\
    \text{subject to}\quad  & X\in\mathbb{S}^{n}_{+},

where $\mathbb{S}^{n}_{+}$ is the collection of positive semidefinite matrices and $P^n$ is the unit spectrahedron given by

.. math::

    P^{n}:=\left\{ X\in\mathbb{S}^{n}_{+}: {\rm tr}\, X = 1\right\}.

:scpt:`src.examples.unconstrained.nonconvex_svm`

This example solves the nonconvex support vector machine problem

.. math::

    \underset{x}{\text{minimize}}\quad  & \frac{1}{n}\sum_{i=1}^{n}\left[1-\tanh\left(v_{i}\langle u_{i},x\rangle\right)\right]+\frac{1}{2n}\|x\|^{2} \\
    \text{subject to}\quad  & x\in\mathbb{R}^{n}.

Nonconvex-Concave Min-Max Problems
----------------------------------

This subsection considers nonconvex-concave min-max composite optimization problems of the form 

.. math::
    
    \underset{x}{\text{minimize}}  \quad & \left[\underset{y}{\text{max}} \,\, \Phi(x,y) + h(x) \right] \\
    \text{subject to}\quad  & x\in\mathbb{R}^{n}, \quad y \in \mathbb{R}^{m}

under a saddle-point termination criterion based on the one given in **[3]**. More specifically, given a tolerance pair $(\rho_x, \rho_y) \in \mathbb{R}_{++}^2$, the goal is to find a quadruple $(x,y,v,w)$ satisfying the conditions

.. math::

    \left(\begin{array}{c}
    v\\
    w
    \end{array}\right)\in\left(\begin{array}{c}
    \nabla_{x}\Phi(x,y)\\
    0
    \end{array}\right)+\left(\begin{array}{c}
    \partial h(x)\\
    \partial\left[-\Phi(x,\cdot)\right](y)
    \end{array}\right),\quad\|v\|\leq\rho_{x},\quad\|w\|\leq\rho_{y}.

:scpt:`src.examples.minmax.nonconvex_minmax_qp`

This example solves the nonconvex minmax quadratic programming problem

.. math::

    \underset{x}{\text{minimize}}  \quad &
    \left[
    \underset{i\in\{1,...,k\}}{\text{max}} \,\, 
    -\frac{ \xi_i}{2}\|D_i B_i x\|^{2}+\frac{\tau_i}{2}\|A_i x-b\|^{2}
    +\delta_{\Delta^{n}}(x)
    \right]  \\
    \text{subject to}\quad  & x\in\mathbb{R}^{n},

where $\Delta^n$ is the unit simplex given by

.. math::

    \Delta^{n}:=\left\{ x\in\mathbb{R}^{n}:\sum_{i=1}^{n}x_{i}=1,0\leq x\leq1\right\}.

:scpt:`src.examples.minmax.nonconvex_power_control`

This example solves the nonconvex power control problem

.. math::

    \underset{x}{\text{minimize}} \quad & 
    \left[\max_{y} \,\, \sum_{k=1}^{K}\sum_{n=1}^{N} f_{k,n}(X,y) + \delta_{B_x}(x) + \delta_{B_y}(y) \right] \\
    \text{subject to}\quad  & x\in\mathbb{R}^{K\times N}, \quad y\in\mathbb{R}^{N},

where $f_{k,n}$, $B_x$, and $B_y$ are given by

.. math::

    f_{k,n}(X,y) & := -\log\left(1+\frac{{\cal A}_{k,k,n}X_{k,n}}{\sigma^{2}+B_{k,n}y_{n}+\sum_{j=1,j\neq k}^{K}{\cal A}_{j,k,n}X_{j,n}}\right), \\
    B_x & := \left\{X\in \mathbb{R}^{K\times N} : 0 \leq X \leq R \right\}, \\
    B_y & := \left\{y\in \mathbb{R}^{N} : 0 \leq y \leq \frac{N}{2} \right\}.

:scpt:`src.examples.minmax.nonconvex_robust_regression`

This example solves the robust regression problem 

.. math::
    
    \underset{x}{\text{minimize}}  \quad &
    \left[
    \underset{i\in\{1,...,n\}}{\text{max}} \,\, 
    \phi_\alpha \circ \ell_i(x)
    \right]  \\
    \text{subject to}\quad  & x\in\mathbb{R}^{k},

where $\phi_\alpha$ and $\ell_j$ are given by

.. math::
    
    \phi_\alpha(t) := \alpha \log \left(1 + \frac{t}{\alpha} \right), \quad
    \ell_j := \log \left(1 + e^{-b_j \langle a_j, x \rangle}\right).

Linearly Set Constrained Problems
---------------------------------

This subsection considers linearly set constrained composite optimization problems where $g(x)=Ax$ for a linear operator $A$ and $S$ is a closed convex set.

:scpt:`src.examples.constrained.lin_constr_nonconvex_qp`

This example solves the linearly-constrained nonconvex quadratic programming problem

.. math::

    \underset{x}{\text{minimize}}\quad  & -\frac{ \xi}{2}\|DBx\|^{2}+\frac{\tau}{2}\|Ax-b\|^{2}+\delta_{\Delta^{n}}(x) \\
    \text{subject to}\quad  & C x = d,\quad x\in\mathbb{R}^{n},

where $\Delta^n$ is the unit simplex given by

.. math::

    \Delta^{n}:=\left\{ x\in\mathbb{R}^{n}:\sum_{i=1}^{n}x_{i}=1,0\leq x\leq1\right\}.

:scpt:`src.examples.constrained.nonconvex_lin_constr_qsdp`

This example solves the linearly-constrained nonconvex quadratic semidefinite programming problem

.. math::

    \underset{X}{\text{minimize}}\quad  & -\frac{\xi}{2}\|DB(X)\|_{F}^{2}+\frac{\tau}{2}\|A(X)-b\|^{2}_{F}+\delta_{P^{n}}(X) \\
    \text{subject to}\quad  & C(X)=d, \quad X\in\mathbb{S}^{n}_{+},

where $\mathbb{S}^{n}_{+}$ is the collection of positive semidefinite matrices and $P^n$ is the unit spectrahedron given by

.. math::

    P^{n}:=\left\{ X\in\mathbb{S}^{n}_{+}: {\rm tr}\, X = 1\right\}.

:scpt:`src.examples.constrained.nonconvex_sparse_pca`

This example solves the nonconvex sparse principal component analysis problem

.. math::

    
    \underset{\Pi,\Phi}{\text{minimize}}\quad  & \langle\Sigma,\Pi\rangle+\sum_{i,j=1}^{n}q_{\nu}(\Phi_{ij})+\nu\sum_{i,j=1}^{n}|\Phi_{ij}|+\delta_{{\cal F}^{k}}(\Pi) \\
    \text{subject to}\quad  & \Pi-\Phi=0, \quad(\Pi, \Phi)\in \mathbb{R}^{n\times n}\times\mathbb{R}^{n\times n},

where ${\cal F}^k$ is the $k$-Fantope given by

.. math::

    {\cal F}^k:=\left\{ X\in\mathbb{S}^{n}_{+}: 0 \preceq X \preceq I, {\rm tr}\, X = k\right\}.

:scpt:`src.examples.constrained.nonconvex_bounded_mc`

This example solves the nonconvex bounded matrix completion problem

.. math::

    \underset{X}{\text{minimize}}\quad  & \frac{1}{2}\|{\rm Proj}_{\Omega}(X-A)\|^{2}_{F}+{\cal R}_{\mu}(X) \\
    \text{subject to}\quad  & X \in B_{[l,u]},

where ${\cal R}_\mu$ is a nonconvex regularization function and $B_{[l,u]}$ is the box given by

.. math::

    B_{[l,u]}:=\left\{ X\in\mathbb{R}^{p\times q}:l\leq X_{ij}\leq u,(i,j)\in\{1,...,p\}\times\{1,...,q\}\right\}.