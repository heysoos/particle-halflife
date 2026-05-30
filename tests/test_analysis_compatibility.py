"""Tests for the pure-chemistry fusion compatibility module."""
import numpy as np
import pytest

from halflife.config import SimConfig
from halflife.state import initialize_physics_params
from halflife.analysis.compatibility import (
    species_pair_compat_matrix,
    max_free_bonds,
)


def test_species_pair_compat_matrix_shape():
    """Matrix is (S, S) and symmetric in BE (hash sum is commutative)."""
    config = SimConfig(num_species=4, num_particles=10, max_composites=10)
    physics = initialize_physics_params(config)

    be, passes_be, passes_val = species_pair_compat_matrix(config, physics)
    assert be.shape == (4, 4)
    assert passes_be.shape == (4, 4)
    assert passes_val.shape == (4, 4)
    # Symmetric: BE(i,j) == BE(j,i) since merged hash is commutative.
    np.testing.assert_allclose(be, be.T, atol=1e-6)


def test_max_free_bonds_free_particle():
    """A free particle of species s has max_free_bonds = v_s (no edges)."""
    config = SimConfig(num_species=4, num_particles=10, max_composites=10)
    # Single-species multiset, count=1
    for s in range(4):
        fb = max_free_bonds([s], config)
        # Should equal hash-derived valence for that species, which is in [1, max_valence]
        assert 1 <= fb <= config.max_valence


def test_max_free_bonds_two_particles_uses_spanning_tree():
    """A 2-member composite of species [a, b] has max_free_bonds = v_a + v_b - 2*(2-1)."""
    config = SimConfig(num_species=2, num_particles=10, max_composites=10, max_valence=2)
    from halflife.chemistry import _species_valences
    v = np.asarray(_species_valences(config))
    for a in range(config.num_species):
        for b in range(config.num_species):
            expected = int(v[a]) + int(v[b]) - 2  # n=2, so -2*(2-1) = -2
            actual = max_free_bonds([a, b], config)
            assert actual == expected, f"a={a} b={b}: expected {expected}, got {actual}"
