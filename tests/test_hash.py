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
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import numpy as np

from halflife.config import SimConfig
from halflife.chemistry import _entity_hash_val, _hash_to_binding_energy, _hash_to_half_life

_config = SimConfig()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_species_entity_hashes(config=_config):
    """Return list of _entity_hash_val outputs for species 0..num_species-1."""
    return [int(_entity_hash_val(jnp.int32(s), config)) for s in range(config.num_species)]


def _merged_hash(h_i: int, h_j: int, config=_config) -> int:
    """Commutative merged hash: (H(i) + H(j)) % modulus."""
    return (h_i + h_j) % config.hash_modulus


def _be_from_pair(s_i: int, s_j: int, config=_config) -> float:
    """Binding energy for free particle pair (s_i, s_j)."""
    hi = int(_entity_hash_val(jnp.int32(s_i), config))
    hj = int(_entity_hash_val(jnp.int32(s_j), config))
    merged = _merged_hash(hi, hj, config)
    return float(_hash_to_binding_energy(jnp.uint32(merged), config))


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
        be = float(_hash_to_binding_energy(jnp.uint32(merged), config))
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


def test_half_life_distribution():
    """
    _hash_to_half_life over 100 random hashes should span most of
    [half_life_min, half_life_max] * composite_half_life_scale.
    """
    config = _config
    rng = np.random.default_rng(7)
    raw_hashes = rng.integers(0, config.hash_modulus, size=100, dtype=np.int64)

    hls = np.array([
        float(_hash_to_half_life(jnp.uint32(int(h)), config))
        for h in raw_hashes
    ])

    expected_min = config.half_life_min * config.composite_half_life_scale
    expected_max = config.half_life_max * config.composite_half_life_scale

    print(f"\nHalf-life distribution over 100 random hashes:")
    print(f"  min={hls.min():.1f}  max={hls.max():.1f}  "
          f"expected range=[{expected_min:.1f}, {expected_max:.1f}]")

    # Should cover at least 50% of the range
    observed_range = hls.max() - hls.min()
    expected_range = expected_max - expected_min
    assert observed_range > 0.5 * expected_range, (
        f"Half-life range {observed_range:.1f} < 50% of expected {expected_range:.1f}. "
        f"Distribution collapsed."
    )
    assert hls.min() >= expected_min * 0.95, (
        f"Min half-life {hls.min():.1f} below expected min {expected_min:.1f}"
    )
    assert hls.max() <= expected_max * 1.05, (
        f"Max half-life {hls.max():.1f} above expected max {expected_max:.1f}"
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
