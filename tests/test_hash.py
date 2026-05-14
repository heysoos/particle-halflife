"""
tests/test_hash.py — Hash distribution & binding energy diagnostics.

Primary goal: expose the binding-energy-collapse bug where _hash_to_binding_energy
returns ~0 for all species pairs because (h // 1000) % 1000 always reads digits that
are zero when h is a multiple of ~10^6.

Run standalone:  python tests/test_hash.py
Run under pytest: pytest tests/test_hash.py -v
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
# CPU pin disabled for this session — GPU available, live sim not running.
# Restore (uncomment) if integration tests start contending with the live sim.
# jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import numpy as np

from halflife.config import SimConfig
from halflife.state import initialize_physics_params
from halflife.chemistry import _entity_hash_val, _hash_to_binding_energy

_config = SimConfig()
_physics = initialize_physics_params(_config)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_species_entity_hashes(config=_config):
    """Return list of _entity_hash_val outputs for species 0..num_species-1."""
    return [int(_entity_hash_val(jnp.int32(s), config)) for s in range(config.num_species)]


def _merged_hash(h_i: int, h_j: int, config=_config) -> int:
    """Commutative merged hash: (H(i) + H(j)) % modulus."""
    return (h_i + h_j) % config.hash_modulus


def _be_from_pair(s_i: int, s_j: int, config=_config, physics=_physics) -> float:
    """Binding energy for free particle pair (s_i, s_j)."""
    hi = int(_entity_hash_val(jnp.int32(s_i), config))
    hj = int(_entity_hash_val(jnp.int32(s_j), config))
    merged = _merged_hash(hi, hj, config)
    return float(_hash_to_binding_energy(jnp.uint32(merged), config, physics))


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_entity_hash_val_distinct():
    """All 12 species should produce distinct entity hash values."""
    hashes = _all_species_entity_hashes()
    assert len(set(hashes)) == len(hashes), (
        f"Expected {len(hashes)} distinct entity hashes, got {len(set(hashes))}.\n"
        f"Hashes: {hashes}"
    )


def test_binding_energy_distribution():
    """
    Compute binding energies for all num_species x num_species free-free pairs.
    At least 10% should exceed 0.05 and at least 2% should exceed fusion_threshold.

    EXPECTED TO FAIL with the current buggy hash function:
    The 12×12 matrix will be printed on failure so the degenerate pattern is visible.
    """
    config = _config
    S = config.num_species
    matrix = np.zeros((S, S), dtype=np.float32)
    for i in range(S):
        for j in range(S):
            matrix[i, j] = _be_from_pair(i, j, config)

    n_pairs = S * S
    above_005 = np.sum(matrix > 0.05)
    above_threshold = np.sum(matrix > config.fusion_threshold)

    frac_005 = above_005 / n_pairs
    frac_threshold = above_threshold / n_pairs

    print(f"\nBinding energy matrix ({S}x{S}):")
    print(np.array2string(matrix, precision=3, suppress_small=True))
    print(f"  Pairs > 0.05: {above_005}/{n_pairs} ({frac_005*100:.1f}%)")
    print(f"  Pairs > fusion_threshold ({config.fusion_threshold}): "
          f"{above_threshold}/{n_pairs} ({frac_threshold*100:.1f}%)")
    print(f"  Min BE: {matrix.min():.4f}  Max BE: {matrix.max():.4f}  "
          f"Mean BE: {matrix.mean():.4f}")

    assert frac_005 >= 0.10, (
        f"Only {frac_005*100:.1f}% of pairs have BE > 0.05 (need ≥10%). "
        f"Hash bug likely: entity hashes are multiples of ~10^6 so "
        f"(h//1000)%%1000 is always 0."
    )
    assert frac_threshold >= 0.02, (
        f"Only {frac_threshold*100:.1f}% of pairs have BE > {config.fusion_threshold} "
        f"(need ≥2%). No fusions will occur!"
    )


def test_merged_hash_distribution():
    """
    500 random merged hashes (sum of two random entity hashes) should have
    mean BE > 0.1 and stddev > 0.05.
    """
    config = _config
    rng = np.random.default_rng(42)
    entity_hashes = _all_species_entity_hashes(config)

    bes = []
    for _ in range(500):
        hi = entity_hashes[rng.integers(0, config.num_species)]
        hj = entity_hashes[rng.integers(0, config.num_species)]
        merged = _merged_hash(hi, hj, config)
        be = float(_hash_to_binding_energy(jnp.uint32(merged), config, _physics))
        bes.append(be)

    bes = np.array(bes)
    mean_be = bes.mean()
    std_be = bes.std()

    print(f"\nMerged-hash BE over 500 random pairs: mean={mean_be:.4f} std={std_be:.4f}")

    assert mean_be > 0.1, (
        f"Mean BE={mean_be:.4f} is too low (need > 0.1). "
        f"Hash extractor is likely reading always-zero digits."
    )
    assert std_be > 0.05, (
        f"Std BE={std_be:.4f} is too low (need > 0.05). "
        f"Distribution is degenerate."
    )


def test_hash_commutative():
    """H(i∪j) = H(j∪i) for 50 random pairs (commutative property)."""
    config = _config
    rng = np.random.default_rng(99)
    entity_hashes = _all_species_entity_hashes(config)

    for _ in range(50):
        si = rng.integers(0, config.num_species)
        sj = rng.integers(0, config.num_species)
        hi = entity_hashes[si]
        hj = entity_hashes[sj]
        h_ij = _merged_hash(hi, hj, config)
        h_ji = _merged_hash(hj, hi, config)
        assert h_ij == h_ji, (
            f"Hash not commutative: H({si}∪{sj})={h_ij} ≠ H({sj}∪{si})={h_ji}"
        )


# ── Standalone runner ─────────────────────────────────────────────────────────

# ── Tests for _hash_to_partition ─────────────────────────────────────────────

def test_partition_deterministic():
    """Same (h, n_members) must produce the same assignment every call."""
    from halflife.chemistry import _hash_to_partition
    config = SimConfig()
    h = jnp.uint32(123_456_789)
    n = jnp.int32(5)
    a1 = _hash_to_partition(h, n, config)
    a2 = _hash_to_partition(h, n, config)
    assert jnp.all(a1 == a2), f"non-deterministic: {a1} vs {a2}"


def test_partition_assignments_in_valid_range():
    """For valid slots i < n_members, assignment[i] in {0, 1}; else -1."""
    from halflife.chemistry import _hash_to_partition
    config = SimConfig()
    M = config.max_composite_size
    for h_val in [1, 100, 10_000, 999_999, 2**31 - 1]:
        for n_val in [2, 3, 5, 8, 16]:
            h = jnp.uint32(h_val)
            n = jnp.int32(n_val)
            a = jnp.asarray(_hash_to_partition(h, n, config))
            assert a.shape == (M,)
            for i in range(M):
                if i < n_val:
                    assert a[i] in (0, 1), f"slot {i} of n={n_val}, h={h_val}: got {a[i]}"
                else:
                    assert a[i] == -1, f"padding slot {i}: got {a[i]}"


def test_partition_both_products_nonempty():
    """For any (h, n>=2), both products must have >=1 member assigned."""
    from halflife.chemistry import _hash_to_partition
    config = SimConfig()
    for h_val in [0, 1, 100, 99_999, 2_654_435_761]:
        for n_val in [2, 3, 4, 5, 8, 16, 32]:
            h = jnp.uint32(h_val)
            n = jnp.int32(n_val)
            a = jnp.asarray(_hash_to_partition(h, n, config))
            valid = a[:n_val]
            count_0 = int(jnp.sum(valid == 0))
            count_1 = int(jnp.sum(valid == 1))
            assert count_0 + count_1 == n_val, f"missing members for n={n_val}, h={h_val}"
            assert count_0 >= 1, f"product 0 empty for n={n_val}, h={h_val}"
            assert count_1 >= 1, f"product 1 empty for n={n_val}, h={h_val}"


def test_partition_distribution_varies_with_hash():
    """Different hash values must produce different partition shapes (sometimes)."""
    from halflife.chemistry import _hash_to_partition
    config = SimConfig()
    n = jnp.int32(8)
    pivots_seen = set()
    for h_val in range(20):
        a = jnp.asarray(_hash_to_partition(jnp.uint32(h_val * 1_000_003), n, config))
        valid = a[:8]
        pivot = int(jnp.sum(valid == 0))
        pivots_seen.add(pivot)
    assert len(pivots_seen) >= 4, f"too few pivot values across 20 hashes: {pivots_seen}"


# ── Tests for _hash_to_valence ───────────────────────────────────────────────

def test_valence_deterministic():
    """Same species index must yield the same valence every call."""
    from halflife.chemistry import _hash_to_valence
    config = SimConfig(use_valence=True, max_valence=4)
    v1 = int(_hash_to_valence(jnp.int32(0), config))
    v2 = int(_hash_to_valence(jnp.int32(0), config))
    assert v1 == v2, f"non-deterministic: {v1} vs {v2}"


def test_valence_in_range():
    """For any species index, valence must be in [1, max_valence]."""
    from halflife.chemistry import _hash_to_valence
    config = SimConfig(use_valence=True, max_valence=4, num_species=64)
    for s in range(config.num_species):
        v = int(_hash_to_valence(jnp.int32(s), config))
        assert 1 <= v <= config.max_valence, (
            f"species {s} valence {v} out of range [1, {config.max_valence}]"
        )


def test_valence_vector_distribution():
    """Pre-computed valence vector should span a non-trivial portion of [1, max_valence]."""
    from halflife.chemistry import _species_valences
    config = SimConfig(use_valence=True, max_valence=4, num_species=12)
    v = jnp.asarray(_species_valences(config))
    assert v.shape == (config.num_species,)
    # With 12 species and max_valence=4, we'd be very unlucky to get only one value.
    distinct = len(set(v.tolist()))
    assert distinct >= 2, (
        f"valence vector collapsed to a single value across 12 species: {v.tolist()}"
    )


def test_valence_max_1_forces_all_ones():
    """With max_valence=1 every species must have v_s = 1 (uniformity check)."""
    from halflife.chemistry import _species_valences
    config = SimConfig(use_valence=True, max_valence=1, num_species=12)
    v = jnp.asarray(_species_valences(config))
    assert int(v.min()) == 1 and int(v.max()) == 1, (
        f"max_valence=1 should produce all-ones vector; got {v.tolist()}"
    )


if __name__ == '__main__':
    passed = failed = 0
    for name, fn in [(n, v) for n, v in sorted(globals().items()) if n.startswith('test_')]:
        try:
            fn()
            print(f'  PASS  {name}')
            passed += 1
        except Exception as e:
            print(f'  FAIL  {name}: {e}')
            traceback.print_exc()
            failed += 1
    print(f'\n{passed} passed, {failed} failed')
    sys.exit(failed)
