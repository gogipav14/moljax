"""
PDE Model assembly for MOL-JAX.

This module provides the MOLModel dataclass that assembles a complete
PDE system from grid, boundary conditions, operators, and parameters.
It also provides factory functions for common PDE systems.

Design decisions:
- MOLModel is a frozen dataclass for immutability
- RHS computation applies BCs first, then sums linear and nonlinear parts
- Factory functions create pre-configured models for common PDEs
- dtype is configurable via params['dtype']
- FFT cache can be stored in metadata for IMEX/FFT-based solvers
- Periodic models can optionally include precomputed FFT symbols
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from moljax.core.bc import BCSpec, BCType, FieldBCSpec, apply_bc
from moljax.core.grid import Grid1D, Grid2D, GridType
from moljax.core.operators import (
    LinearOp,
    NonlinearOp,
    acoustics_1d_linear_op,
    brusselator_reaction_op,
    d1_upwind_2d,
    gray_scott_reaction_op,
    laplacian_2d,
    schnakenberg_reaction_op,
)
from moljax.core.state import StateDict, tree_add, tree_zeros_like

# =============================================================================
# Identity for the integrators' compiled-driver caches
# =============================================================================

# How deep static_cache_key walks a container before it gives up and keys
# the rest by identity. Nothing moljax puts in a model, a parameter set or
# an FFT cache nests further than a handful of levels; the limit only
# exists so that a caller's own metadata cannot send the walk into a cycle.
_CACHE_KEY_MAX_DEPTH = 8


def static_cache_key(obj: Any, _depth: int = 0) -> Any:
    """
    A hashable stand-in for an object a compiled integrator closes over.

    The integrators cache their compiled time-stepping loops (see
    CompiledDriverCache and moljax.core.stepping) keyed on everything that
    determines the trace, and most of what they close over compares by
    value already: NKParams is a NamedTuple, CFLParams, PIDParams and the
    preconditioners are frozen dataclasses of scalars. Two separately
    built defaults must share one cache entry, or the default call path
    would miss the cache on every call, so those are walked field by field
    and keyed by their contents.

    What cannot be keyed by value is keyed by identity: arrays (walking
    every element of an FFT cache would cost more than the compilation it
    saves), callables, and anything else that is neither a container nor
    hashable. Scalars carry their Python type alongside their value
    because the type is part of the trace: a np.float32 dt is strongly
    typed and promotes differently from a Python float of the same value.

    Keying by identity is sound here only because every cache entry holds
    a reference to the objects its key was built from, so an id in a live
    key always belongs to a live object and cannot have been recycled for
    something else.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, bytes)):
        return (type(obj).__name__, obj)
    if isinstance(obj, (bool, int, float, complex)):
        return (type(obj).__name__, obj)
    if isinstance(obj, np.generic):
        return (type(obj).__name__, obj.item())
    if isinstance(obj, (jax.Array, np.ndarray)):
        return ('array', type(obj).__name__, id(obj), obj.shape, str(obj.dtype))
    if _depth < _CACHE_KEY_MAX_DEPTH:
        if isinstance(obj, dict):
            items = sorted(obj.items(), key=lambda kv: repr(kv[0]))
            return ('dict', tuple(
                (repr(k), static_cache_key(v, _depth + 1)) for k, v in items
            ))
        if isinstance(obj, (tuple, list)):
            return (type(obj).__name__, tuple(
                static_cache_key(v, _depth + 1) for v in obj
            ))
        if is_dataclass(obj) and not isinstance(obj, type):
            return (type(obj).__name__, tuple(
                (f.name, static_cache_key(getattr(obj, f.name), _depth + 1))
                for f in fields(obj)
            ))
    try:
        hash(obj)
    except TypeError:
        return ('id', type(obj).__name__, id(obj))
    return (type(obj).__name__, obj)


def state_cache_key(state: StateDict) -> tuple:
    """
    The part of a state that determines the trace: structure and avals.

    A compiled loop is reusable for any state with the same pytree
    structure and the same leaf shapes and dtypes; the values themselves
    are arguments. Works on tracers as well as on concrete arrays, so a
    driver traced inside a caller's own jit is keyed the same way.
    """
    leaves, treedef = jax.tree_util.tree_flatten(state)
    return (treedef, tuple(
        (jnp.shape(leaf), str(jnp.result_type(leaf))) for leaf in leaves
    ))


_COMPILED_DRIVER_CACHES: list["CompiledDriverCache"] = []


class CompiledDriverCache:
    """
    A bounded store of compiled integrator drivers, least-recently-used.

    Each integrator builds its time-stepping loop as a closure over its
    static parameters and hands that closure to jax.jit. Before this cache
    the closure was rebuilt on every call, so it was a different Python
    object each time and JAX's own compilation cache could never hit: three
    identical integrate_fixed_dt calls compiled three identical scans, and
    three adaptive_integrate calls three identical while loops, 0.7 to 3.2
    seconds each. Keeping the closure alive across calls is what lets
    JAX's cache do its job.

    The store is bounded because each entry pins a compiled executable and
    everything the driver closes over, the model included. Sweeps that
    build a fresh model per run would otherwise grow it without limit.
    Thirty-two entries is enough for the parameter sweeps in this tree
    (a handful of methods times a handful of grids) and small enough that
    the pinned models are not a memory problem; call
    clear_compiled_drivers to drop them all.
    """

    def __init__(self, name: str, max_entries: int = 32):
        self.name = name
        self.max_entries = max_entries
        self._entries: dict[Any, Any] = {}
        _COMPILED_DRIVER_CACHES.append(self)

    def get_or_build(self, key: Any, build: Callable[[], Any]) -> Any:
        """Return the driver stored under key, building it if it is absent."""
        if key in self._entries:
            # Re-insert so that insertion order is use order and the entry
            # dropped below is the least recently used one.
            self._entries[key] = self._entries.pop(key)
            return self._entries[key]
        driver = build()
        if len(self._entries) >= self.max_entries:
            self._entries.pop(next(iter(self._entries)))
        self._entries[key] = driver
        return driver

    def clear(self) -> None:
        """Forget every driver in this cache."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def clear_compiled_drivers() -> None:
    """
    Drop every compiled integrator loop moljax is holding on to.

    For tests that count compilations, and for callers that have run
    through many models or grids and want the executables and the pinned
    models released.
    """
    for cache in _COMPILED_DRIVER_CACHES:
        cache.clear()


@dataclass(frozen=True)
class MOLModel:
    """
    Method of Lines PDE model.

    Represents a PDE system of the form:
        du/dt = L(u, t) + N(u, t)

    where L is the linear part (diffusion, advection, coupling) and
    N is the nonlinear part (reactions, nonlinear fluxes).

    Attributes:
        grid: Grid1D or Grid2D defining the spatial domain
        bc_spec: Boundary conditions for each field
        params: Dict of parameters (diffusivities, velocities, etc.)
        linear_ops: Tuple of LinearOp objects
        nonlinear_ops: Tuple of NonlinearOp objects
        metadata: Additional metadata (field_names, wave_speed, etc.)

    Example:
        >>> grid = Grid2D.uniform(64, 64, 0, 1, 0, 1)
        >>> model = create_gray_scott_model(grid)
        >>> state = model.create_initial_state()
        >>> rhs = model.rhs(state, t=0.0)
    """
    grid: GridType
    bc_spec: BCSpec
    params: dict[str, Any]
    linear_ops: tuple[LinearOp, ...] = field(default_factory=tuple)
    nonlinear_ops: tuple[NonlinearOp, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dtype(self) -> jnp.dtype:
        """Data type for arrays (from params or default float64)."""
        return self.params.get('dtype', jnp.float64)

    @property
    def field_names(self) -> list[str]:
        """List of field names in the model."""
        return list(self.bc_spec.keys())

    @property
    def compile_key(self) -> tuple:
        """
        A hashable identity for this model, for the driver caches.

        A model holds callables (its operators, its boundary data), so
        there is no way to tell two models apart by value alone; identity
        is the only sound answer and this key leads with it. It then adds
        the contents of the model's mutable parts, the params and metadata
        dicts and everything reachable from them, because a frozen
        dataclass does not stop a caller from writing
        ``model.params['D_u'] = 0.5``: the params are traced into the
        compiled loop as constants, so a changed value has to miss the
        cache and retrace. The grid, the BC spec and the operator tuples
        cannot be reassigned on a frozen model, but they are cheap to walk
        and keying them keeps the key self-describing.

        See moljax.core.model.static_cache_key for what walking means
        here, and why keying by id is safe for a live cache entry.
        """
        return (
            'MOLModel',
            id(self),
            static_cache_key(self.grid),
            static_cache_key(self.bc_spec),
            static_cache_key(self.params),
            static_cache_key(self.linear_ops),
            static_cache_key(self.nonlinear_ops),
            static_cache_key(self.metadata),
        )

    def apply_bcs(self, state: StateDict, t: float = 0.0) -> StateDict:
        """Apply boundary conditions to state."""
        return apply_bc(state, self.grid, self.bc_spec, t, self.params)

    def linear_rhs(self, state: StateDict, t: float) -> StateDict:
        """Compute linear part of RHS: sum of all linear operators."""
        if len(self.linear_ops) == 0:
            return tree_zeros_like(state)

        result = self.linear_ops[0].apply(state, self.grid, t, self.params)
        for op in self.linear_ops[1:]:
            result = tree_add(result, op.apply(state, self.grid, t, self.params))
        return result

    def nonlinear_rhs(self, state: StateDict, t: float) -> StateDict:
        """Compute nonlinear part of RHS: sum of all nonlinear operators."""
        if len(self.nonlinear_ops) == 0:
            return tree_zeros_like(state)

        result = self.nonlinear_ops[0].apply(state, self.grid, t, self.params)
        for op in self.nonlinear_ops[1:]:
            result = tree_add(result, op.apply(state, self.grid, t, self.params))
        return result

    def rhs(self, state: StateDict, t: float) -> StateDict:
        """
        Compute full RHS: apply BCs, then sum linear and nonlinear parts.

        Args:
            state: Current state (with ghost cells)
            t: Current time

        Returns:
            RHS du/dt for each field
        """
        # Apply boundary conditions first
        state = self.apply_bcs(state, t)

        # Compute linear and nonlinear parts
        L = self.linear_rhs(state, t)
        N = self.nonlinear_rhs(state, t)

        return tree_add(L, N)

    def create_padded_field(
        self,
        init_fn: Callable | None = None,
        fill_value: float = 0.0
    ) -> jnp.ndarray:
        """
        Create a single padded field array.

        Args:
            init_fn: Optional function (X, Y) -> array for initialization
            fill_value: Value if init_fn is None

        Returns:
            Padded array
        """
        if isinstance(self.grid, Grid1D):
            shape = (self.grid.nx_total,)
            if init_fn is not None:
                x = self.grid.x_coords(include_ghost=True)
                return init_fn(x).astype(self.dtype)
            else:
                return jnp.full(shape, fill_value, dtype=self.dtype)
        else:
            shape = (self.grid.ny_total, self.grid.nx_total)
            if init_fn is not None:
                X, Y = self.grid.meshgrid(include_ghost=True)
                return init_fn(X, Y).astype(self.dtype)
            else:
                return jnp.full(shape, fill_value, dtype=self.dtype)

    def create_initial_state(
        self,
        init_fns: dict[str, Callable] | None = None,
        fill_values: dict[str, float] | None = None
    ) -> StateDict:
        """
        Create initial state for all fields.

        Args:
            init_fns: Dict of field_name -> init function (X, Y) -> array
            fill_values: Dict of field_name -> constant fill value

        Returns:
            Initial state with all fields initialized
        """
        if init_fns is None:
            init_fns = {}
        if fill_values is None:
            fill_values = {}

        state = {}
        for name in self.field_names:
            if name in init_fns:
                state[name] = self.create_padded_field(init_fn=init_fns[name])
            elif name in fill_values:
                state[name] = self.create_padded_field(fill_value=fill_values[name])
            else:
                state[name] = self.create_padded_field(fill_value=0.0)

        return state

    def cfl_dt(self, state: StateDict | None = None) -> float:
        """
        Compute CFL-limited time step from operator bounds.

        Args:
            state: Current state (needed for some nonlinear bounds)

        Returns:
            Maximum stable dt based on CFL conditions
        """
        dt_min = jnp.inf

        # Linear operator bounds
        for op in self.linear_ops:
            if op.dt_bound is not None:
                dt_op = op.dt_bound(self.grid, self.params)
                dt_min = jnp.minimum(dt_min, dt_op)

        # Nonlinear operator bounds (if state provided)
        if state is not None:
            for op in self.nonlinear_ops:
                if op.dt_bound is not None:
                    dt_op = op.dt_bound(self.grid, state, self.params)
                    dt_min = jnp.minimum(dt_min, dt_op)

        return dt_min


# =============================================================================
# Model Factory Functions
# =============================================================================

def create_gray_scott_model(
    grid: Grid2D,
    Du: float = 0.16,
    Dv: float = 0.08,
    F: float = 0.04,
    k: float = 0.06,
    bc_type: BCType = BCType.PERIODIC,
    dtype: jnp.dtype = jnp.float64
) -> MOLModel:
    """
    Create Gray-Scott reaction-diffusion model.

    System:
        du/dt = Du * Laplacian(u) - u*v^2 + F*(1-u)
        dv/dt = Dv * Laplacian(v) + u*v^2 - (F+k)*v

    Args:
        grid: 2D grid
        Du: Diffusion coefficient for u (default 0.16)
        Dv: Diffusion coefficient for v (default 0.08)
        F: Feed rate (default 0.04)
        k: Kill rate (default 0.06)
        bc_type: Boundary condition type (default PERIODIC)
        dtype: Data type (default float64)

    Returns:
        MOLModel configured for Gray-Scott
    """
    # Boundary conditions
    bc_spec = {
        'u': FieldBCSpec(kind=bc_type),
        'v': FieldBCSpec(kind=bc_type)
    }

    # Parameters
    params = {
        'Du': Du,
        'Dv': Dv,
        'F': F,
        'k': k,
        'dtype': dtype
    }

    # Linear operators (diffusion)
    def diffusion_apply(state: StateDict, grid: Grid2D, t: float, params: dict) -> StateDict:
        Du = params['Du']
        Dv = params['Dv']
        return {
            'u': Du * laplacian_2d(state['u'], grid),
            'v': Dv * laplacian_2d(state['v'], grid)
        }

    def diffusion_dt_bound(grid: Grid2D, params: dict) -> float:
        D_max = max(params['Du'], params['Dv'])
        return 0.25 * grid.min_dx2 / (D_max + 1e-14)

    diffusion_op = LinearOp(
        name="gray_scott_diffusion",
        apply=diffusion_apply,
        dt_bound=diffusion_dt_bound
    )

    # Nonlinear operators (reaction)
    reaction_op = gray_scott_reaction_op()

    # Metadata
    metadata = {
        'field_names': ['u', 'v'],
        'pde_type': 'reaction_diffusion',
        'stiff': True
    }

    return MOLModel(
        grid=grid,
        bc_spec=bc_spec,
        params=params,
        linear_ops=(diffusion_op,),
        nonlinear_ops=(reaction_op,),
        metadata=metadata
    )


def create_schnakenberg_model(
    grid: Grid2D,
    Du: float = 1.0,
    Dv: float = 40.0,
    a: float = 0.1,
    b: float = 0.9,
    gamma: float = 1000.0,
    bc_type: BCType = BCType.PERIODIC,
    dtype: jnp.dtype = jnp.float64
) -> MOLModel:
    """
    Create Schnakenberg reaction-diffusion model (Turing morphogenesis).

    System:
        du/dt = Du * Laplacian(u) + gamma * (a - u + u^2*v)
        dv/dt = Dv * Laplacian(v) + gamma * (b - u^2*v)

    Args:
        grid: 2D grid
        Du: Diffusion coefficient for u (default 1.0)
        Dv: Diffusion coefficient for v (default 40.0)
        a: Kinetic parameter (default 0.1)
        b: Kinetic parameter (default 0.9)
        gamma: Gierer-Meinhardt scaling (default 1000)
        bc_type: Boundary condition type (default PERIODIC)
        dtype: Data type (default float64)

    Returns:
        MOLModel configured for Schnakenberg
    """
    bc_spec = {
        'u': FieldBCSpec(kind=bc_type),
        'v': FieldBCSpec(kind=bc_type)
    }

    params = {
        'Du': Du, 'Dv': Dv,
        'a': a, 'b': b, 'gamma': gamma,
        'dtype': dtype
    }

    def diffusion_apply(state: StateDict, grid: Grid2D, t: float, params: dict) -> StateDict:
        Du = params['Du']
        Dv = params['Dv']
        return {
            'u': Du * laplacian_2d(state['u'], grid),
            'v': Dv * laplacian_2d(state['v'], grid)
        }

    def diffusion_dt_bound(grid: Grid2D, params: dict) -> float:
        D_max = max(params['Du'], params['Dv'])
        return 0.25 * grid.min_dx2 / (D_max + 1e-14)

    diffusion_op = LinearOp(
        name="schnakenberg_diffusion",
        apply=diffusion_apply,
        dt_bound=diffusion_dt_bound
    )

    reaction_op = schnakenberg_reaction_op()

    metadata = {
        'field_names': ['u', 'v'],
        'pde_type': 'reaction_diffusion',
        'stiff': True
    }

    return MOLModel(
        grid=grid,
        bc_spec=bc_spec,
        params=params,
        linear_ops=(diffusion_op,),
        nonlinear_ops=(reaction_op,),
        metadata=metadata
    )


def create_brusselator_model(
    grid: Grid2D,
    Du: float = 0.01,
    Dv: float = 0.02,
    a: float = 1.0,
    b: float = 3.4,
    bc_type: BCType = BCType.PERIODIC,
    dtype: jnp.dtype = jnp.float64
) -> MOLModel:
    """
    Create Brusselator reaction-diffusion model (oscillatory instability).

    System:
        du/dt = Du * Laplacian(u) + a - (b+1)*u + u^2*v
        dv/dt = Dv * Laplacian(v) + b*u - u^2*v

    Args:
        grid: 2D grid
        Du: Diffusion coefficient for u (default 0.01)
        Dv: Diffusion coefficient for v (default 0.02)
        a: Kinetic parameter (default 1.0)
        b: Kinetic parameter (default 3.4, above Hopf threshold)
        bc_type: Boundary condition type (default PERIODIC)
        dtype: Data type (default float64)

    Returns:
        MOLModel configured for Brusselator
    """
    bc_spec = {
        'u': FieldBCSpec(kind=bc_type),
        'v': FieldBCSpec(kind=bc_type)
    }

    params = {
        'Du': Du, 'Dv': Dv,
        'a': a, 'b': b,
        'dtype': dtype
    }

    def diffusion_apply(state: StateDict, grid: Grid2D, t: float, params: dict) -> StateDict:
        Du = params['Du']
        Dv = params['Dv']
        return {
            'u': Du * laplacian_2d(state['u'], grid),
            'v': Dv * laplacian_2d(state['v'], grid)
        }

    def diffusion_dt_bound(grid: Grid2D, params: dict) -> float:
        D_max = max(params['Du'], params['Dv'])
        return 0.25 * grid.min_dx2 / (D_max + 1e-14)

    diffusion_op = LinearOp(
        name="brusselator_diffusion",
        apply=diffusion_apply,
        dt_bound=diffusion_dt_bound
    )

    reaction_op = brusselator_reaction_op()

    metadata = {
        'field_names': ['u', 'v'],
        'pde_type': 'reaction_diffusion',
        'stiff': True
    }

    return MOLModel(
        grid=grid,
        bc_spec=bc_spec,
        params=params,
        linear_ops=(diffusion_op,),
        nonlinear_ops=(reaction_op,),
        metadata=metadata
    )


def create_advection_diffusion_model(
    grid: Grid2D,
    field_names: Sequence[str] = ('c1', 'c2'),
    D: float = 0.01,
    vx: float = 1.0,
    vy: float = 0.0,
    bc_type: BCType = BCType.PERIODIC,
    use_upwind: bool = True,
    dtype: jnp.dtype = jnp.float64
) -> MOLModel:
    """
    Create multi-species advection-diffusion model.

    System for each species c_i:
        dc_i/dt = D * Laplacian(c_i) - vx * dc_i/dx - vy * dc_i/dy

    Args:
        grid: 2D grid
        field_names: List of species field names
        D: Diffusion coefficient (same for all species)
        vx: x-velocity
        vy: y-velocity
        bc_type: Boundary condition type
        use_upwind: Use upwind scheme for advection
        dtype: Data type

    Returns:
        MOLModel configured for advection-diffusion
    """
    # Boundary conditions (same for all fields)
    bc_spec = {name: FieldBCSpec(kind=bc_type) for name in field_names}

    # Parameters
    params = {
        'D': D,
        'vx': vx,
        'vy': vy,
        'dtype': dtype
    }

    # Linear operators
    def advdiff_apply(state: StateDict, grid: Grid2D, t: float, params: dict) -> StateDict:
        D = params['D']
        vx = params['vx']
        vy = params['vy']

        result = {}
        for name in state.keys():
            f = state[name]
            # Diffusion
            diff = D * laplacian_2d(f, grid)
            # Advection (upwind or central)
            if use_upwind:
                adv_x = vx * d1_upwind_2d(f, vx, grid, axis=1)
                adv_y = vy * d1_upwind_2d(f, vy, grid, axis=0)
            else:
                from moljax.core.operators import d1_central_2d
                adv_x = vx * d1_central_2d(f, grid, axis=1)
                adv_y = vy * d1_central_2d(f, grid, axis=0)
            result[name] = diff - adv_x - adv_y

        return result

    def advdiff_dt_bound(grid: Grid2D, params: dict) -> float:
        D = params['D']
        vx = params['vx']
        vy = params['vy']
        v_max = jnp.abs(vx) + jnp.abs(vy)

        # Advection CFL
        dt_adv = grid.min_dx / (v_max + 1e-14)
        # Diffusion CFL
        dt_diff = 0.25 * grid.min_dx2 / (D + 1e-14)

        return jnp.minimum(dt_adv, dt_diff)

    advdiff_op = LinearOp(
        name="advection_diffusion",
        apply=advdiff_apply,
        dt_bound=advdiff_dt_bound
    )

    # Metadata (field_names may arrive as a tuple; every factory stores a list)
    metadata = {
        'field_names': list(field_names),
        'pde_type': 'advection_diffusion',
        'stiff': False
    }

    return MOLModel(
        grid=grid,
        bc_spec=bc_spec,
        params=params,
        linear_ops=(advdiff_op,),
        nonlinear_ops=(),
        metadata=metadata
    )


def create_acoustics_1d_model(
    grid: Grid1D,
    K: float = 1.0,
    rho: float = 1.0,
    bc_type: BCType = BCType.PERIODIC,
    dtype: jnp.dtype = jnp.float64
) -> MOLModel:
    """
    Create 1D linear acoustics model.

    System:
        dp/dt = -K * dv/dx
        dv/dt = -(1/rho) * dp/dx

    Wave speed c = sqrt(K/rho).

    Args:
        grid: 1D grid
        K: Bulk modulus
        rho: Density
        bc_type: Boundary condition type
        dtype: Data type

    Returns:
        MOLModel configured for 1D acoustics
    """
    # Boundary conditions
    bc_spec = {
        'p': FieldBCSpec(kind=bc_type),
        'v': FieldBCSpec(kind=bc_type)
    }

    # Parameters
    params = {
        'K': K,
        'rho': rho,
        'dtype': dtype
    }

    # Linear operator
    acoustics_op = acoustics_1d_linear_op()

    # Metadata
    c = jnp.sqrt(K / rho)
    metadata = {
        'field_names': ['p', 'v'],
        'pde_type': 'hyperbolic',
        'wave_speed': float(c),
        'stiff': False
    }

    return MOLModel(
        grid=grid,
        bc_spec=bc_spec,
        params=params,
        linear_ops=(acoustics_op,),
        nonlinear_ops=(),
        metadata=metadata
    )


def create_fisher_kpp_model(
    grid: Grid2D,
    D: float = 0.01,
    r: float = 1.0,
    bc_type: BCType = BCType.DIRICHLET,
    dtype: jnp.dtype = jnp.float64
) -> MOLModel:
    """
    Create Fisher-KPP reaction-diffusion model.

    System:
        du/dt = D * Laplacian(u) + r * u * (1 - u)

    Args:
        grid: 2D grid
        D: Diffusion coefficient
        r: Growth rate
        bc_type: Boundary condition type
        dtype: Data type

    Returns:
        MOLModel configured for Fisher-KPP
    """
    from moljax.core.operators import fisher_kpp_reaction_op

    # Boundary conditions
    bc_spec = {'u': FieldBCSpec(kind=bc_type)}

    # Parameters
    params = {
        'D': D,
        'r': r,
        'dtype': dtype
    }

    # Diffusion operator
    def diffusion_apply(state: StateDict, grid: Grid2D, t: float, params: dict) -> StateDict:
        D = params['D']
        return {'u': D * laplacian_2d(state['u'], grid)}

    def diffusion_dt_bound(grid: Grid2D, params: dict) -> float:
        D = params['D']
        return 0.25 * grid.min_dx2 / (D + 1e-14)

    diffusion_op = LinearOp(
        name="fisher_diffusion",
        apply=diffusion_apply,
        dt_bound=diffusion_dt_bound
    )

    # Reaction operator
    reaction_op = fisher_kpp_reaction_op('u', 'r')

    # Metadata
    metadata = {
        'field_names': ['u'],
        'pde_type': 'reaction_diffusion',
        'stiff': True
    }

    return MOLModel(
        grid=grid,
        bc_spec=bc_spec,
        params=params,
        linear_ops=(diffusion_op,),
        nonlinear_ops=(reaction_op,),
        metadata=metadata
    )


# =============================================================================
# FFT-Enabled Model Factories
# =============================================================================

def create_gray_scott_periodic_fft(
    grid: Grid2D,
    Du: float = 0.16,
    Dv: float = 0.08,
    F: float = 0.04,
    k: float = 0.06,
    dtype: jnp.dtype = jnp.float64
) -> tuple[MOLModel, Any, dict[str, float]]:
    """
    Create Gray-Scott model with periodic BCs and precomputed FFT cache.

    Returns the model, FFT cache, and diffusivities dict for use with
    IMEX integrators.

    Args:
        grid: 2D grid
        Du: Diffusion coefficient for u
        Dv: Diffusion coefficient for v
        F: Feed rate
        k: Kill rate
        dtype: Data type

    Returns:
        Tuple of (model, fft_cache, diffusivities)
    """
    from moljax.core.fft_solvers import create_fft_cache

    # Create base model with periodic BCs
    model = create_gray_scott_model(
        grid, Du=Du, Dv=Dv, F=F, k=k,
        bc_type=BCType.PERIODIC, dtype=dtype
    )

    # Create FFT cache
    fft_cache = create_fft_cache(grid, dtype)

    # Diffusivities for IMEX
    diffusivities = {'u': Du, 'v': Dv}

    # Add FFT cache to metadata
    metadata = dict(model.metadata)
    metadata['fft_cache'] = fft_cache
    metadata['diffusivities'] = diffusivities
    metadata['supports_fft'] = True

    # Create new model with updated metadata
    model_fft = MOLModel(
        grid=model.grid,
        bc_spec=model.bc_spec,
        params=model.params,
        linear_ops=model.linear_ops,
        nonlinear_ops=model.nonlinear_ops,
        metadata=metadata
    )

    return model_fft, fft_cache, diffusivities


def create_schnakenberg_periodic_fft(
    grid: Grid2D,
    Du: float = 1.0,
    Dv: float = 40.0,
    a: float = 0.1,
    b: float = 0.9,
    gamma: float = 1000.0,
    dtype: jnp.dtype = jnp.float64
) -> tuple[MOLModel, Any, dict[str, float]]:
    """
    Create Schnakenberg model with periodic BCs and precomputed FFT cache.

    Returns:
        Tuple of (model, fft_cache, diffusivities)
    """
    from moljax.core.fft_solvers import create_fft_cache

    model = create_schnakenberg_model(
        grid, Du=Du, Dv=Dv, a=a, b=b, gamma=gamma,
        bc_type=BCType.PERIODIC, dtype=dtype
    )

    fft_cache = create_fft_cache(grid, dtype)
    diffusivities = {'u': Du, 'v': Dv}

    metadata = dict(model.metadata)
    metadata['fft_cache'] = fft_cache
    metadata['diffusivities'] = diffusivities
    metadata['supports_fft'] = True

    model_fft = MOLModel(
        grid=model.grid,
        bc_spec=model.bc_spec,
        params=model.params,
        linear_ops=model.linear_ops,
        nonlinear_ops=model.nonlinear_ops,
        metadata=metadata
    )

    return model_fft, fft_cache, diffusivities


def create_brusselator_periodic_fft(
    grid: Grid2D,
    Du: float = 0.01,
    Dv: float = 0.02,
    a: float = 1.0,
    b: float = 3.4,
    dtype: jnp.dtype = jnp.float64
) -> tuple[MOLModel, Any, dict[str, float]]:
    """
    Create Brusselator model with periodic BCs and precomputed FFT cache.

    Returns:
        Tuple of (model, fft_cache, diffusivities)
    """
    from moljax.core.fft_solvers import create_fft_cache

    model = create_brusselator_model(
        grid, Du=Du, Dv=Dv, a=a, b=b,
        bc_type=BCType.PERIODIC, dtype=dtype
    )

    fft_cache = create_fft_cache(grid, dtype)
    diffusivities = {'u': Du, 'v': Dv}

    metadata = dict(model.metadata)
    metadata['fft_cache'] = fft_cache
    metadata['diffusivities'] = diffusivities
    metadata['supports_fft'] = True

    model_fft = MOLModel(
        grid=model.grid,
        bc_spec=model.bc_spec,
        params=model.params,
        linear_ops=model.linear_ops,
        nonlinear_ops=model.nonlinear_ops,
        metadata=metadata
    )

    return model_fft, fft_cache, diffusivities


def create_advdiff_periodic_fft(
    grid: Grid2D,
    field_names: Sequence[str] = ('c1', 'c2'),
    D: float = 0.01,
    vx: float = 1.0,
    vy: float = 0.0,
    use_upwind: bool = True,
    dtype: jnp.dtype = jnp.float64
) -> tuple[MOLModel, Any, dict[str, float]]:
    """
    Create advection-diffusion model with periodic BCs and FFT cache.

    For IMEX: diffusion is implicit (FFT), advection is explicit (upwind).

    Args:
        grid: 2D grid
        field_names: List of species field names
        D: Diffusion coefficient
        vx: x-velocity
        vy: y-velocity
        use_upwind: Use upwind scheme for advection
        dtype: Data type

    Returns:
        Tuple of (model, fft_cache, diffusivities)
    """
    from moljax.core.fft_solvers import create_fft_cache

    # Create base model
    model = create_advection_diffusion_model(
        grid, field_names=field_names, D=D, vx=vx, vy=vy,
        bc_type=BCType.PERIODIC, use_upwind=use_upwind, dtype=dtype
    )

    # Create FFT cache
    fft_cache = create_fft_cache(grid, dtype)

    # Diffusivities for all fields
    diffusivities = {name: D for name in field_names}

    # Add FFT cache to metadata
    metadata = dict(model.metadata)
    metadata['fft_cache'] = fft_cache
    metadata['diffusivities'] = diffusivities
    metadata['supports_fft'] = True

    model_fft = MOLModel(
        grid=model.grid,
        bc_spec=model.bc_spec,
        params=model.params,
        linear_ops=model.linear_ops,
        nonlinear_ops=model.nonlinear_ops,
        metadata=metadata
    )

    return model_fft, fft_cache, diffusivities


def get_fft_cache(model: MOLModel):
    """Extract FFT cache from model metadata if available."""
    return model.metadata.get('fft_cache', None)


def get_diffusivities(model: MOLModel) -> dict[str, float]:
    """Extract diffusivities from model metadata or params."""
    if 'diffusivities' in model.metadata:
        return model.metadata['diffusivities']

    # Try to construct from params
    diffusivities = {}
    for name in model.field_names:
        # Check for field-specific diffusivity
        D_key = f'D{name}'
        if D_key in model.params:
            diffusivities[name] = model.params[D_key]
        elif 'D' in model.params:
            diffusivities[name] = model.params['D']
        else:
            diffusivities[name] = 0.0

    return diffusivities
