"""
Tests for the model factories in moljax.core.model.
"""

import jax

jax.config.update("jax_enable_x64", True)

from moljax.core.grid import Grid2D
from moljax.core.model import create_advection_diffusion_model


def test_advection_diffusion_metadata_field_names_is_a_list():
    """metadata['field_names'] is a list whatever sequence the caller passed.

    The default field_names is a tuple (a list default would be a shared
    mutable), and it used to be stored as is; every other factory records
    a list, and callers that serialize or extend the metadata rely on that.
    """
    grid = Grid2D.uniform(4, 4, 0.0, 1.0, 0.0, 1.0)

    default = create_advection_diffusion_model(grid)
    assert default.metadata['field_names'] == ['c1', 'c2']
    assert isinstance(default.metadata['field_names'], list)

    named = create_advection_diffusion_model(grid, field_names=('a', 'b', 'c'))
    assert named.metadata['field_names'] == ['a', 'b', 'c']
    assert isinstance(named.metadata['field_names'], list)
    assert named.field_names == ['a', 'b', 'c']
