Examples
========

This section documents the examples solved by NC-OPT. For brevity, only the optimization model is given. Moreover, to avoid repetition, we will also make use of the notation $$\delta_{C}(x)=
\begin{cases}
0, & x\in C, \\\\
\infty, & x\notin C,
\end{cases}
$$
to refer to the indicator function of a closed convex set $C$.

.. note::

    Additional details about these problems can be found in the papers:

    Kong, W., Melo, J. G., & Monteiro, R. D. (2019). Complexity of a
    quadratic penalty accelerated inexact proximal point method for solving 
    linearly constrained nonconvex composite programs. *SIAM Journal on 
    Optimization, 29*\(4), 2566-2593.

    Kong, W., Melo, J. G., & Monteiro, R. D. (2020). An efficient 
    adaptive accelerated inexact proximal point method for solving linearly 
    constrained nonconvex composite problems. *Computational Optimization and 
    Applications, 76*\(2), 305-346. 

    Kong, W., & Monteiro, R. D. (2019). An accelerated inexact 
    proximal point method for solving nonconvex-concave min-max problems. 
    *arXiv preprint arXiv:1905.13433*.

:scpt:`src.examples.unconstrained.basic_convex_qp`

This example solves the convex univariate optimization problem
$$
\\begin{align}
\\underset{x}{\\text{minimize}}\\quad  & \\frac{1}{2}x^{2}-x+\\frac{1}{2} \\\\\\
\\text{subject to}\\quad  & x\\in\\mathbb{R}.
\\end{align}
$$

:scpt:`src.examples.unconstrained.nonconvex_qp`

This example solves the nonconvex quadratic programming problem
$$
\\begin{align}
\underset{x}{\\text{minimize}}\\quad  & -\\frac{\\xi}{2}\\|DBx\\|^{2}+\\frac{\\tau}{2}\\|Ax-b\\|^{2}+\\delta_{\\Delta^{n}}(x) \\\\\\
\\text{subject to}\\quad  & x\\in\\mathbb{R}^{n},
\\end{align}
$$
where $\Delta^n$ is the unit simplex given by
$$\\Delta^{n}:=\\left\\{ x\\in\\mathbb{R}^{n}:\\sum_{i=1}^{n}x_{i}=1,0\\leq x\\leq1\\right\\}.$$

:scpt:`src.examples.unconstrained.nonconvex_qsdp`

This example solves the nonconvex quadratic semidefinite programming problem
$$
\\begin{align}
\\underset{X}{\\text{minimize}}\\quad  & -\\frac{\\xi}{2}\\|DB(X)\\|_{F}^{2}+\\frac{\\tau}{2}\\|A(X)-b\\|^{2}_{F}+\\delta_{P^{n}}(X) \\\\\\
\\text{subject to}\\quad  & X\\in\\mathbb{S}^{n}_{+},
\\end{align}
$$
where $\mathbb{S}^{n}_{+}$ is the collection of positive semidefinite matrices and $P^n$ is the unit spectrahedron given by
$$P^{n}:=\\left\\{ X\\in\\mathbb{S}^{n}_{+}:\\sum_{i=1}^{n} \\lambda_i(x)
=1,0\\leq \\lambda(x) \\leq1\\right\\}.$$

:scpt:`src.examples.unconstrained.nonconvex_svm`

This example solves the nonconvex support vector machine problem
$$
\\begin{align}
\\underset{x}{\\text{minimize}}\\quad  & \\frac{1}{n}\\sum_{i=1}^{n}\\left[1-\\tanh\\left(v_{i}\\langle u_{i},x\\rangle\\right)\\right]+\\frac{1}{2n}\\|x\\|^{2} \\\\\\
\\text{subject to}\\quad  & x\\in\\mathbb{R}^{n}.
\\end{align}
$$

:scpt:`src.examples.constrained.lin_constr_nonconvex_qp`

This example solves the linearly-constrained nonconvex quadratic programming problem
$$
\\begin{align}
\underset{x}{\\text{minimize}}\\quad  & -\\frac{\\xi}{2}\\|DBx\\|^{2}+\\frac{\\tau}{2}\\|Ax-b\\|^{2}+\\delta_{\\Delta^{n}}(x) \\\\\\
\\text{subject to}\\quad  & C x = d,\\quad x\\in\\mathbb{R}^{n},
\\end{align}
$$
where $\Delta^n$ is the unit simplex given by
$$\\Delta^{n}:=\\left\\{ x\\in\\mathbb{R}^{n}:\\sum_{i=1}^{n}x_{i}=1,0\\leq x\\leq1\\right\\}.$$

:scpt:`src.examples.constrained.nonconvex_lin_constr_qsdp`

This example solves the linearly-constrained nonconvex quadratic semidefinite programming problem
$$
\\begin{align}
\\underset{X}{\\text{minimize}}\\quad  & -\\frac{\\xi}{2}\\|DB(X)\\|_{F}^{2}+\\frac{\\tau}{2}\\|A(X)-b\\|^{2}_{F}+\\delta_{P^{n}}(X) \\\\\\
\\text{subject to}\\quad  & C(X)=d,\\quad X\\in\\mathbb{S}^{n}_{+},
\\end{align}
$$
where $\mathbb{S}^{n}_{+}$ is the collection of positive semidefinite matrices and $P^n$ is the unit spectrahedron given by
$$P^{n}:=\\left\\{ X\\in\\mathbb{S}^{n}_{+}: {\\rm tr}\\, X = 1\\right\\}.$$

:scpt:`src.examples.constrained.nonconvex_sparse_pca`

This example solves the nonconvex sparse principal component analysis problem
$$
\\begin{align}
\\underset{\\Pi,\\Phi}{\\text{minimize}}\\quad  & \\langle\\Sigma,\\Pi\\rangle+\\sum_{i,j=1}^{n}q_{\\nu}(\\Phi_{ij})+\\nu\\sum_{i,j=1}^{n}|\\Phi_{ij}|+\\delta_{{\\cal F}^{k}}(\\Pi) \\\\\\
\\text{subject to}\\quad  & \\Pi-\\Phi=0, \\quad(\\Pi, \\Phi)\\in \\mathbb{R}^{n\\times n}\\times\\mathbb{R}^{n\\times n},
\\end{align}
$$
where ${\cal F}^k$ is the $k$-Fantope given by
$$
{\cal F}^k:=\\left\\{ X\\in\\mathbb{S}^{n}_{+}: 0 \\preceq X \\preceq I, {\\rm tr}\\, X = k\\right\\}.
$$

:scpt:`src.examples.constrained.nonconvex_bounded_mc`

This example solves the nonconvex bounded matrix completion problem
$$
\\begin{align}
\\underset{X}{\\text{minimize}}\\quad  & \\frac{1}{2}\\|{\\rm Proj}_{\\Omega}(X-A)\|^{2}_{F}+{\\cal R}_{\mu}(X) \\\\\\
\\text{subject to}\\quad  & X \\in B[l,u],
\\end{align}
$$
where ${\cal R}_\mu$ is a nonconvex regularization function and $B[l,u]$ is a box given by
$$
B[l,u]:=\\left\\{ X\\in\\mathbb{R}^{p\\times q}:l\\leq X_{ij}\\leq u,(i,j)\\in\\{1,...,p\\}\times\\{1,...,q\\}\\right\\}.
$$