"""Suite-wide fixtures.

The suite compiles thousands of small XLA programs, and JAX keeps every
jitted executable alive in its in-process caches for the life of the
interpreter. On a 16 GB CI runner that growth eventually starved an XLA
compilation in the middle of the suite and the process died with a
segmentation fault, at a point that moved earlier as tests were added and
never reproduced on a machine with more memory. Dropping the caches at
every module boundary keeps the footprint bounded by the largest module,
and no test relies on a compilation surviving from a previous module.
"""

from __future__ import annotations

import gc

import jax
import pytest

from moljax.core.model import clear_compiled_drivers


@pytest.fixture(autouse=True, scope="module")
def _drop_compilation_caches_between_modules():
    yield
    clear_compiled_drivers()
    jax.clear_caches()
    gc.collect()
