"""
MOL-JAX: A JAX-based Method-of-Lines PDE Solver Library

This library provides a unified framework for solving PDEs using the
Method of Lines (MOL) approach with JAX for automatic differentiation
and JIT compilation.

Key Features:
- Multi-field PyTree state as first-class citizen
- Ghost-cell boundary conditions (periodic, Dirichlet, Neumann)
- Composable spatial operators (D1, D2, Laplacian, coupled)
- Explicit integrators: SSPRK3, RK4, Euler
- Implicit integrators: BE, CN, BDF2 with Newton-Krylov solver
- JIT-friendly adaptive time stepping via lax.while_loop
- Automatic dt policy: CFL limiter + PID controller
"""

from moljax.core.grid import Grid1D, Grid2D
from moljax.core.state import (
    StateDict,
    tree_add,
    tree_sub,
    tree_scale,
    tree_axpy,
    tree_norm2,
    tree_norm_inf,
    tree_vdot,
    flatten_to_vec,
)
from moljax.core.bc import BCType, FieldBCSpec, apply_bc
from moljax.core.operators import (
    d1_central_1d,
    d1_upwind_1d,
    d2_central_1d,
    laplacian_1d,
    d1_central_2d,
    d1_upwind_2d,
    d2_central_2d,
    laplacian_2d,
    LinearOp,
    NonlinearOp,
)
from moljax.core.model import MOLModel
from moljax.core.stepping import (
    IntegratorType,
    euler_step,
    ssprk3_step,
    rk4_step,
    be_step,
    cn_step,
    bdf2_step,
    adaptive_integrate,
)
from moljax.core.newton_krylov import newton_krylov_solve
from moljax.core.preconditioners import (
    IdentityPreconditioner,
    BlockJacobiPreconditioner,
    DiffusionPreconditioner,
)
from moljax.core.dt_policy import propose_dt, CFLParams, PIDParams

__version__ = "0.1.0"
__all__ = [
    # Grid
    "Grid1D",
    "Grid2D",
    # State
    "StateDict",
    "tree_add",
    "tree_sub",
    "tree_scale",
    "tree_axpy",
    "tree_norm2",
    "tree_norm_inf",
    "tree_vdot",
    "flatten_to_vec",
    # BC
    "BCType",
    "FieldBCSpec",
    "apply_bc",
    # Operators
    "d1_central_1d",
    "d1_upwind_1d",
    "d2_central_1d",
    "laplacian_1d",
    "d1_central_2d",
    "d1_upwind_2d",
    "d2_central_2d",
    "laplacian_2d",
    "LinearOp",
    "NonlinearOp",
    # Model
    "MOLModel",
    # Stepping
    "IntegratorType",
    "euler_step",
    "ssprk3_step",
    "rk4_step",
    "be_step",
    "cn_step",
    "bdf2_step",
    "adaptive_integrate",
    # Newton-Krylov
    "newton_krylov_solve",
    # Preconditioners
    "IdentityPreconditioner",
    "BlockJacobiPreconditioner",
    "DiffusionPreconditioner",
    # dt Policy
    "propose_dt",
    "CFLParams",
    "PIDParams",
]
