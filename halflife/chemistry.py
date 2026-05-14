"""
Hash-based reaction system: fusion, decay, and fission.

This is the intellectual core of Half-Life. Instead of a lookup table,
reaction rules are implicit in a commutative additive hash over participant
species IDs. Inspired by Hiroki Sayama's Hash Chemistry.

  Same species set → same hash → same composite properties, every time.
  Different hash constants (config) → different "universe" / chemistry.

Fusion:
  Two entities (free particles or composites) within fusion_radius whose
  hash-derived binding energy exceeds fusion_threshold combine. The merged
  hash is H(i ∪ j) = (H(i) + H(j)) % modulus — single addition, no sort.

Decay:
  Only composites decay; particles are conserved. Each composite has
  P(decay in dt) = 1 - exp(-dt*ln2/half_life). The half-life is derived
  from the composite's binding energy (high BE → long life), with a size
  penalty applied in fusion_scan_body (larger composites are less stable).

Fission:
  Composite decay releases its member particles back to free state with
  their original species (no transmutation), and injects
  binding_energy * (1 - fission_cost) as radial kinetic energy.

JIT notes:
  - No Python control flow inside JAX operations
  - Conflict resolution via lax.scan over a randomly shuffled candidate set
  - Dead composite slots recycled using cumsum-based free-slot finding
"""

import jax
import jax.numpy as jnp

from halflife.state import ParticleState, CompositeState, WorldState, InteractionParams, PhysicsParams
from halflife.config import SimConfig
from halflife.utils import find_free_slots


# ── Hash Utilities ────────────────────────────────────────────────────────────

def _entity_hash_val(species: jnp.ndarray, config: SimConfig) -> jnp.ndarray:
    """
    Per-species hash value (order-independent, no sort needed).
    f(s) = (s+1)^2 * prime_a + (s+1) * prime_b  — unique per species.
    """
    s1 = species.astype(jnp.int32) + 1
    return (s1 * s1 * config.hash_prime_a + s1 * config.hash_prime_b) % config.hash_modulus


def _compute_entity_hash(pid: jnp.ndarray, particles, composites,
                          config: SimConfig) -> tuple:
    """
    Compute the commutative hash and member count for the entity containing pid.
    H(entity) = sum(_entity_hash_val(s) for s in members) % modulus
    Merged hash: H(i union j) = (H(i) + H(j)) % modulus  — no sort needed.
    """
    M = config.max_composite_size
    c = jnp.clip(particles.composite_id[pid], 0, config.max_composites - 1)
    is_free = particles.composite_id[pid] < 0

    count = jnp.where(is_free, jnp.int32(1), composites.member_count[c])

    # Free particle: single-species hash
    free_h = _entity_hash_val(particles.species[pid], config)

    # Composite: sum hashes of all members' species (parallel, no scan)
    safe_members = jnp.where(composites.members[c] >= 0, composites.members[c], 0)
    member_species = particles.species[safe_members]  # (M,)
    valid = (composites.members[c] >= 0) & (jnp.arange(M) < composites.member_count[c])
    member_hvals = jax.vmap(lambda s: _entity_hash_val(s, config))(member_species)  # (M,)
    comp_h = jnp.sum(jnp.where(valid, member_hvals, 0)) % config.hash_modulus

    h = jnp.where(is_free, free_h, comp_h).astype(jnp.uint32)
    return h, count


def _hash_to_binding_energy(h: jnp.ndarray, config: SimConfig,
                             physics: PhysicsParams) -> jnp.ndarray:
    """Derive binding energy from species hash (normalized to [0, 1])."""
    # Bug fix: (h // 1000) % 1000 always read 0 because entity hashes are
    # multiples of hash_prime_a ≈ 10^6, so decimal digits 3-5 are always zero.
    # Fix: apply a secondary Fibonacci-hash mix before extracting low digits.
    h2 = (h * jnp.uint32(2_654_435_761)) ^ (h >> jnp.uint32(13))
    frac = (h2 % jnp.uint32(1000)).astype(jnp.float32) / 999.0
    return frac * physics.binding_energy_scale


def _hash_to_valence(species: jnp.ndarray, config: SimConfig) -> jnp.ndarray:
    """
    Per-species valence in [1, config.max_valence]. Deterministic per species
    index, decorrelated from binding energy via a Fibonacci-style re-mix.

    Valence represents the maximum number of "hands" a particle of this species
    can use to hold neighbors — analogous to molecular valence (H=1, O=2, etc).
    A composite's free bonds = Σ v_s − 2 × (n−1), tracked per composite.

    Args:
        species: scalar int32 — species index in [0, num_species)
        config:  SimConfig (static)

    Returns:
        v: scalar int32 in [1, max_valence]
    """
    # Hash the species index via the same per-species mixer used elsewhere
    # (_entity_hash_val), then re-mix to decorrelate from BE.
    h = _entity_hash_val(species, config).astype(jnp.uint32)
    h2 = (h * jnp.uint32(0x9E3779B1)) ^ (h >> jnp.uint32(13))
    return (h2 % jnp.uint32(config.max_valence) + jnp.uint32(1)).astype(jnp.int32)



def _hash_to_rest_length(s_i: jnp.ndarray, s_j: jnp.ndarray,
                         config: SimConfig) -> jnp.ndarray:
    """
    Hash-derived bond rest length for species pair (s_i, s_j).

    Order-independent (uses the same commutative additive hash as composites)
    so r_rest[i, j] == r_rest[j, i] without an explicit symmetry pass.
    Re-mixed with a Fibonacci hash to decorrelate from binding energy and
    valence so universes with the same num_species but different hash_modulus
    get genuinely different bond chemistries.

    Returns:
        scalar float32 in [config.r_rest_min, config.r_rest_max]
    """
    h_i = _entity_hash_val(s_i, config).astype(jnp.uint32)
    h_j = _entity_hash_val(s_j, config).astype(jnp.uint32)
    h = (h_i + h_j) % jnp.uint32(config.hash_modulus)
    # Fibonacci re-mix to decorrelate from BE / valence streams.
    h_mix = (h * jnp.uint32(0x9E3779B1)) ^ (h >> jnp.uint32(11))
    frac = (h_mix % jnp.uint32(1000)).astype(jnp.float32) / 999.0
    return jnp.float32(config.r_rest_min) + frac * jnp.float32(
        config.r_rest_max - config.r_rest_min
    )

def _species_valences(config: SimConfig) -> jnp.ndarray:
    """Pre-compute the (num_species,) valence vector. Fixed for a given config."""
    species_idx = jnp.arange(config.num_species, dtype=jnp.int32)
    return jax.vmap(lambda s: _hash_to_valence(s, config))(species_idx)


def compute_degree(composites, config: SimConfig) -> jnp.ndarray:
    """
    Per-particle edge-incidence count, summed across all alive composites.

    For each valid edge (i, j) in every alive composite, increment degree[i]
    and degree[j] by 1. Returns (N,) int32. Used by the per-particle valence
    gate in fusion and by the ring-closure scan.

    Args:
        composites: CompositeState
        config:     SimConfig (static)

    Returns:
        (N,) int32 — degree[i] = number of edges incident to particle i
    """
    N = config.num_particles
    C = config.max_composites
    E = config.e_max

    # Mask edges by alive composite AND valid slot index (<= edge_count[c]).
    # Each edge slot contributes 2 scatter-adds (one per endpoint).
    e_idx = jnp.arange(E, dtype=jnp.int32)  # (E,)
    valid = composites.alive[:, None] & (e_idx[None, :] < composites.edge_count[:, None])  # (C, E)

    pid_a = composites.edges[:, :, 0]  # (C, E)
    pid_b = composites.edges[:, :, 1]  # (C, E)

    # Route invalid entries to index N (OOB, dropped via mode='drop').
    drop_a = jnp.where(valid, pid_a, N)
    drop_b = jnp.where(valid, pid_b, N)

    degree = jnp.zeros(N, dtype=jnp.int32)
    degree = degree.at[drop_a.reshape(-1)].add(1, mode='drop')
    degree = degree.at[drop_b.reshape(-1)].add(1, mode='drop')
    return degree


def compute_composite_free_bonds(particles, composites, degree: jnp.ndarray,
                                  species_valences: jnp.ndarray,
                                  config: SimConfig) -> jnp.ndarray:
    """
    Per-composite free-bond cache.

    composite_free_bonds[c] = Σ (v_{species[m]} − degree[m]) over m in members[c]
                            = Σ v_{species[m]} − 2 · edge_count[c]

    (Equivalent because each edge contributes 1 to two endpoint degrees.)

    Used as the cheap (C,) skip mask for the ring-closure scan: composites with
    free_bonds < 2 contribute zero work because they can't add another edge.

    Args:
        particles, composites: state
        degree:                (N,) int32 from compute_degree
        species_valences:      (S,) int32 from _species_valences
        config:                SimConfig (static)

    Returns:
        (C,) int32 — composite-level free bonds
    """
    M = config.max_composite_size
    C = config.max_composites
    m_idx = jnp.arange(M, dtype=jnp.int32)

    def per_composite(c):
        members = composites.members[c]  # (M,)
        n = composites.member_count[c]
        valid = composites.alive[c] & (members >= 0) & (m_idx < n)
        safe_m = jnp.where(valid, members, 0)
        per_particle_free = species_valences[particles.species[safe_m]] - degree[safe_m]
        return jnp.sum(jnp.where(valid, per_particle_free, 0))

    return jax.vmap(per_composite)(jnp.arange(C, dtype=jnp.int32))


def _entity_free_bonds(pid: jnp.ndarray, particles, composites,
                        species_valences: jnp.ndarray,
                        config: SimConfig) -> jnp.ndarray:
    """
    Remaining bond capacity for the entity (free particle or composite)
    containing particle `pid`. Returns a scalar int32.

      free_bonds(free particle s)         = v_s
      free_bonds(composite of size n)     = Σ v_s_i  −  2 × (n − 1)

    Spanning-tree accounting: every fusion adds one edge, consuming one bond
    on each side, so an n-member tree uses 2*(n-1) bonds total. The remainder
    is "free" — available for further mergers.
    """
    M = config.max_composite_size
    c = jnp.clip(particles.composite_id[pid], 0, config.max_composites - 1)
    is_free = particles.composite_id[pid] < 0

    # Free particle: v_s for its own species.
    free_val = species_valences[particles.species[pid]]

    # Composite: sum of member valences − 2 × (n − 1).
    safe_members = jnp.where(composites.members[c] >= 0, composites.members[c], 0)
    member_species = particles.species[safe_members]  # (M,)
    valid = (composites.members[c] >= 0) & (jnp.arange(M) < composites.member_count[c])
    member_valences = species_valences[member_species]  # (M,)
    total_v = jnp.sum(jnp.where(valid, member_valences, jnp.int32(0)))
    n = composites.member_count[c]
    comp_val = total_v - jnp.int32(2) * (n - jnp.int32(1))

    return jnp.where(is_free, free_val, comp_val)


def _product_free_bonds(member_ids: jnp.ndarray, count: jnp.ndarray,
                         particles, species_valences: jnp.ndarray,
                         config: SimConfig) -> jnp.ndarray:
    """
    Free bonds for a fission product's (already-compacted) member list.
    Identical math to the composite branch of _entity_free_bonds, but takes
    the member list directly rather than dereferencing a composite slot.
    Returns a scalar int32 (may be negative for structurally unsound partitions).
    """
    M = config.max_composite_size
    safe = jnp.where(member_ids >= 0, member_ids, 0)
    sp = particles.species[safe]
    valid = (member_ids >= 0) & (jnp.arange(M) < count)
    vs = species_valences[sp]
    total_v = jnp.sum(jnp.where(valid, vs, jnp.int32(0)))
    return total_v - jnp.int32(2) * (count - jnp.int32(1))


def _hash_to_partition(h: jnp.ndarray, n_members: jnp.ndarray,
                       config: SimConfig) -> jnp.ndarray:
    """
    Determine a binary partition of composite members for fission.

    For each member slot i ∈ [0, n_members), compute a sort key from
    hash_mix(h, i) and rank slots by that key. The first `pivot` slots
    in sorted order go to product 0, the rest to product 1. Both products
    are guaranteed non-empty because pivot ∈ [1, n_members - 1].

    Args:
        h:          scalar uint32 — composite's species hash
        n_members:  scalar int32 — number of valid members (>= 2)
        config:     SimConfig (static)

    Returns:
        assignment: (max_composite_size,) int32 — values in {0, 1} for slots
                    i < n_members, else -1.
    """
    M = config.max_composite_size

    # Per-slot sort key: mix h with slot index using Fibonacci-style hash.
    slot_idx = jnp.arange(M, dtype=jnp.uint32)
    sort_keys = (h.astype(jnp.uint32) * jnp.uint32(2_654_435_761)
                 + slot_idx * jnp.uint32(1_000_003))
    # Mark padding slots (i >= n_members) with max key so they sort last.
    sort_keys = jnp.where(
        jnp.arange(M, dtype=jnp.int32) < n_members,
        sort_keys,
        jnp.uint32(0xFFFFFFFF),
    )

    # Argsort: order[k] = slot whose sort_key is k-th smallest.
    order = jnp.argsort(sort_keys)  # (M,)

    # Pivot in [1, n_members - 1] from a different region of the hash.
    # n_members >= 2 guarantees this range is non-empty.
    pivot = jnp.int32(1) + (
        ((h >> jnp.uint32(12)).astype(jnp.int32)) % jnp.maximum(n_members - 1, jnp.int32(1))
    )

    # In sorted order, assign first `pivot` to product 0, rest to product 1.
    sorted_assignment = jnp.where(
        jnp.arange(M, dtype=jnp.int32) < pivot,
        jnp.int32(0),
        jnp.int32(1),
    )
    # Mark padding slots (rank >= n_members in sorted order) with -1.
    sorted_assignment = jnp.where(
        jnp.arange(M, dtype=jnp.int32) < n_members,
        sorted_assignment,
        jnp.int32(-1),
    )

    # Scatter back to original slot order: assignment[order[k]] = sorted_assignment[k].
    assignment = jnp.full(M, -1, dtype=jnp.int32).at[order].set(sorted_assignment)
    return assignment


# ── Composite Decay / Fission ─────────────────────────────────────────────────

def apply_composite_decay(state: WorldState, config: SimConfig,
                           physics: PhysicsParams) -> WorldState:
    """
    Apply binary fission decay to all alive composites.

    A decaying composite is partitioned into two products by _hash_to_partition.
    Product 0 reuses the parent's composite slot. Product 1 takes a fresh free
    slot from the composite pool. Products of size 1 become free particles;
    products of size >= 2 become new composites with hash-derived properties.

    Particle species are never modified — only their composite_id and velocity.

    Energy: parent.binding_energy * (1 - fission_cost) is split equally between
    the two products as kinetic energy, applied as a momentum-conserving kick
    along the COM-vs-COM axis (product 0 → +direction, product 1 → -direction).
    Each product's members all get the same kick (the product moves as a unit).

    Args:
        state:   WorldState
        config:  SimConfig (static)
        physics: PhysicsParams — provides dt for the per-step decay probability

    Returns:
        Updated WorldState
    """
    particles = state.particles
    composites = state.composites
    key, subkey = jax.random.split(state.rng_key)
    N = config.num_particles
    M = config.max_composite_size
    C = config.max_composites

    # ── Roll for which composites decay this step ───────────────────────────
    rand = jax.random.uniform(subkey, (C,))
    ln2 = jnp.log(2.0)
    decay_prob = 1.0 - jnp.exp(-physics.dt * ln2 / (composites.half_life + 1e-10))
    fissions = composites.alive & (rand < decay_prob)  # (C,) bool

    # Pre-allocate fresh composite slots for product 1 of each fissioning composite.
    free_slots = find_free_slots(composites.alive, C)  # (C,) int32

    # Assign each fissioning composite a "fission rank" via cumsum so it
    # picks free_slots[rank] as its product-1 target.
    fission_rank = jnp.cumsum(fissions.astype(jnp.int32)) - 1  # (C,) — -1 for non-fissioning

    # ── Per-composite: compute partition assignment and COMs ────────────────
    def per_composite(c):
        n = composites.member_count[c]
        h = composites.species_hash[c]
        assignment = _hash_to_partition(h, n, config)  # (M,) ∈ {-1, 0, 1}

        # Compute each product's COM using min-image displacement from member 0.
        member_ids = composites.members[c]  # (M,)
        safe_ids = jnp.where(member_ids >= 0, member_ids, 0)
        valid = (member_ids >= 0) & (jnp.arange(M) < n)
        ref = particles.position[safe_ids[0]]

        def disp_from_ref(idx):
            d = particles.position[safe_ids[idx]] - ref
            if config.boundary_mode == "periodic":
                d = d - config.world_width  * jnp.round(d[0] / config.world_width)  * jnp.array([1., 0.])
                d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
            return d

        rels = jax.vmap(disp_from_ref)(jnp.arange(M))  # (M, 2)

        in_p0 = valid & (assignment == 0)
        in_p1 = valid & (assignment == 1)
        n0 = jnp.sum(in_p0.astype(jnp.float32))
        n1 = jnp.sum(in_p1.astype(jnp.float32))
        com0 = ref + jnp.sum(rels * in_p0[:, None].astype(jnp.float32), axis=0) / (n0 + 1e-8)
        com1 = ref + jnp.sum(rels * in_p1[:, None].astype(jnp.float32), axis=0) / (n1 + 1e-8)

        rank = fission_rank[c]
        target_p1 = free_slots[jnp.clip(rank, 0, C - 1)]

        return assignment, com0, com1, target_p1, n0.astype(jnp.int32), n1.astype(jnp.int32)

    all_assignment, all_com0, all_com1, all_target_p1, all_n0, all_n1 = jax.vmap(per_composite)(
        jnp.arange(C, dtype=jnp.int32)
    )
    # Shapes: (C, M), (C, 2), (C, 2), (C,), (C,), (C,)

    # ── Compact each product's members & compute its hash ───────────────────
    # Runs before per_member so that the cap-validity check below has the
    # product hashes available; per_member then consults the validity flags
    # when deciding whether each member rejoins a composite or goes free.
    def per_product(c):
        does_fission = fissions[c]
        assignment = all_assignment[c]
        member_ids = composites.members[c]
        n = composites.member_count[c]

        # Compact members of each product to front using cumsum (same trick as fusion).
        in_p0 = (assignment == 0) & (member_ids >= 0) & (jnp.arange(M) < n)
        in_p1 = (assignment == 1) & (member_ids >= 0) & (jnp.arange(M) < n)

        pos_p0 = jnp.cumsum(in_p0.astype(jnp.int32)) - 1
        out_pos_p0 = jnp.where(in_p0, pos_p0, M)
        members_p0 = jnp.full(M, -1, dtype=jnp.int32).at[out_pos_p0].set(member_ids, mode='drop')
        count_p0 = jnp.sum(in_p0.astype(jnp.int32))

        pos_p1 = jnp.cumsum(in_p1.astype(jnp.int32)) - 1
        out_pos_p1 = jnp.where(in_p1, pos_p1, M)
        members_p1 = jnp.full(M, -1, dtype=jnp.int32).at[out_pos_p1].set(member_ids, mode='drop')
        count_p1 = jnp.sum(in_p1.astype(jnp.int32))

        # Species hashes via commutative sum over each product's members.
        def hash_for_product(members_arr, count_arr):
            safe = jnp.where(members_arr >= 0, members_arr, 0)
            sp = particles.species[safe]
            valid_m = (members_arr >= 0) & (jnp.arange(M) < count_arr)
            hvals = jax.vmap(lambda s: _entity_hash_val(s, config))(sp)
            return jnp.sum(jnp.where(valid_m, hvals, 0)) % config.hash_modulus

        h_p0 = hash_for_product(members_p0, count_p0).astype(jnp.uint32)
        h_p1 = hash_for_product(members_p1, count_p1).astype(jnp.uint32)

        return members_p0, count_p0, h_p0, members_p1, count_p1, h_p1

    p0_members, p0_count, p0_hash, p1_members, p1_count, p1_hash = jax.vmap(per_product)(
        jnp.arange(C, dtype=jnp.int32)
    )

    # ── Per-product free bonds and structural validity ──────────────────────
    # free_bonds(product) = Σ v_s − 2 × (count − 1). A product with free_bonds
    # < 0 is structurally unsound (a tree with that many nodes would require
    # more bonds than the members collectively offer), so it cannot form a
    # composite — its members shatter into free particles (the fission kick
    # still fires). Gated by config.use_valence so the toggle is zero-cost
    # when off. Size-1 products are handled by the count >= 2 gate downstream.
    species_valences_decay = _species_valences(config)
    p0_free_bonds = jax.vmap(
        lambda members, count: _product_free_bonds(
            members, count, particles, species_valences_decay, config
        )
    )(p0_members, p0_count)
    p1_free_bonds = jax.vmap(
        lambda members, count: _product_free_bonds(
            members, count, particles, species_valences_decay, config
        )
    )(p1_members, p1_count)

    if config.use_valence:
        p0_valid = p0_free_bonds >= 0
        p1_valid = p1_free_bonds >= 0
    else:
        p0_valid = jnp.ones(C, dtype=bool)
        p1_valid = jnp.ones(C, dtype=bool)

    # ── Update each member particle's composite_id and velocity ────────────
    def per_member(c, m):
        does_fission = fissions[c]
        n = composites.member_count[c]
        member_id = composites.members[c, m]
        valid = does_fission & (m < n) & (member_id >= 0)

        a = all_assignment[c, m]
        com0 = all_com0[c]
        com1 = all_com1[c]
        n0 = all_n0[c]
        n1 = all_n1[c]
        target_p1 = all_target_p1[c]

        # Direction along COM-COM axis (min-image).
        d = com0 - com1
        if config.boundary_mode == "periodic":
            d = d - config.world_width  * jnp.round(d[0] / config.world_width)  * jnp.array([1., 0.])
            d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
        d_norm = jnp.linalg.norm(d) + 1e-8
        d_hat = d / d_norm

        # Energy split: half to each product.
        e_per = composites.binding_energy[c] * (1.0 - config.fission_cost) * 0.5

        # Mass per product (mass=1 per particle).
        mass_p0 = n0.astype(jnp.float32)
        mass_p1 = n1.astype(jnp.float32)
        v0 = jnp.sqrt(jnp.maximum(0.0, 2.0 * e_per / (mass_p0 + 1e-8)))
        v1 = jnp.sqrt(jnp.maximum(0.0, 2.0 * e_per / (mass_p1 + 1e-8)))

        # Kick: product 0 → +d_hat * v0, product 1 → -d_hat * v1.
        # Note: the kick always fires (even for shattered products), because
        # the binding-energy release happens regardless of whether the pieces
        # then bind into a sub-composite or fly apart as free particles.
        kick = jnp.where(
            a == 0,
            d_hat * v0,
            jnp.where(a == 1, -d_hat * v1, jnp.zeros(2)),
        )

        # Composite-formation flags incorporate structural free-bond validity.
        # When valence is off, p0_valid/p1_valid are all-True, so behavior is
        # unchanged from the no-saturation baseline.
        forms_p0 = (n0 >= 2) & p0_valid[c]
        forms_p1 = (n1 >= 2) & p1_valid[c]

        # New composite_id:
        #   a==0 and forms_p0 → c (reuse parent slot)
        #   a==0 and not forms_p0 → -1 (free particle — size-1 or structurally unsound)
        #   a==1 and forms_p1 → target_p1
        #   a==1 and not forms_p1 → -1 (free)
        #   a==-1 (padding) → preserve original (the scatter is gated on `valid`,
        #                     so this value is never written; use a safe default).
        safe_member = jnp.where(member_id >= 0, member_id, 0)
        new_cid = jnp.where(
            a == 0,
            jnp.where(forms_p0, c, jnp.int32(-1)),
            jnp.where(a == 1,
                      jnp.where(forms_p1, target_p1, jnp.int32(-1)),
                      particles.composite_id[safe_member]),
        )

        return member_id, valid, new_cid, kick

    pid_grid, valid_grid, cid_grid, kick_grid = jax.vmap(
        lambda c: jax.vmap(lambda m: per_member(c, m))(jnp.arange(M, dtype=jnp.int32))
    )(jnp.arange(C, dtype=jnp.int32))
    # Shapes: (C, M), (C, M), (C, M), (C, M, 2)

    flat_pid   = pid_grid.reshape(-1)
    flat_valid = valid_grid.reshape(-1)
    flat_cid   = cid_grid.reshape(-1)
    flat_kick  = kick_grid.reshape(-1, 2)

    # Route invalid entries to OOB index N (dropped). Without mode='drop',
    # JAX scatters with duplicate indices have indeterminate behavior, so
    # M-1 invalid slots writing the read-back value to index 0 would race
    # against any real write to particle 0.
    drop_pids = jnp.where(flat_valid, flat_pid, N)
    new_composite_id = particles.composite_id.at[drop_pids].set(flat_cid, mode='drop')

    # Velocity adds — duplicates accumulate, invalid entries add 0, so safe form is fine.
    safe_pids = jnp.where(flat_valid, flat_pid, 0)
    new_velocity = particles.velocity.at[safe_pids].add(
        jnp.where(flat_valid[:, None], flat_kick, 0.0)
    )

    # ── Write product 0 into parent slot c (in place) ──
    # AND with p0_valid so structurally unsound products leave the parent slot
    # dead (its members already got composite_id=-1 above), reclaiming the slot.
    p0_alive = fissions & (p0_count >= 2) & p0_valid

    # Half-life from BE + size penalty. Same formula as fusion_scan_body.
    # Take both args explicitly so both can be vmapped (closing over p0_count
    # would mis-broadcast against scalar `be` under vmap).
    def _hl_from_be_and_n(be, n):
        t = jnp.clip((be - physics.fusion_threshold) / (1.0 - physics.fusion_threshold + 1e-8), 0.0, 1.0)
        hl_base = config.half_life_min + (config.half_life_max - config.half_life_min) * t
        size_penalty = 1.0 + config.composite_size_decay_scale * jnp.maximum(
            0.0, n.astype(jnp.float32) - 2.0
        )
        return hl_base / size_penalty

    p0_be_all = jax.vmap(lambda h: _hash_to_binding_energy(h, config, physics))(p0_hash)
    p0_hl_all = jax.vmap(_hl_from_be_and_n)(p0_be_all, p0_count)

    new_alive = jnp.where(fissions, p0_alive, composites.alive)
    new_members = jnp.where(fissions[:, None], p0_members, composites.members)
    new_member_count = jnp.where(fissions, p0_count, composites.member_count)
    new_species_hash = jnp.where(fissions, p0_hash, composites.species_hash)
    new_binding_energy = jnp.where(fissions, p0_be_all, composites.binding_energy)
    new_half_life = jnp.where(fissions, p0_hl_all, composites.half_life)
    new_free_bonds = jnp.where(fissions, p0_free_bonds, composites.free_bonds)
    # Reset age on the parent slot (it's now a fresh product).
    new_age = jnp.where(fissions, jnp.float32(0.0), composites.age)

    # ── Write product 1 into all_target_p1[c] when fissions[c] AND p1_count[c] >= 2 ──
    # AND with p1_valid so structurally unsound products are not written to a
    # fresh composite slot (the free slot remains available for next step).
    p1_writes = fissions & (p1_count >= 2) & p1_valid

    p1_be_all = jax.vmap(lambda h: _hash_to_binding_energy(h, config, physics))(p1_hash)
    p1_hl_all = jax.vmap(_hl_from_be_and_n)(p1_be_all, p1_count)

    # Scatter-write product 1 to all_target_p1[c]. Guard against negative
    # indices: find_free_slots returns -1 when there aren't enough free slots,
    # and JAX's negative-index default would wrap to [C-1] — clobbering the
    # last composite. Route those to C (OOB) so mode='drop' actually drops them.
    drop_targets = jnp.where(
        p1_writes & (all_target_p1 >= 0),
        all_target_p1,
        C,  # OOB → drop
    )

    new_alive          = new_alive.at[drop_targets].set(p1_writes, mode='drop')
    new_members        = new_members.at[drop_targets].set(p1_members, mode='drop')
    new_member_count   = new_member_count.at[drop_targets].set(p1_count, mode='drop')
    new_species_hash   = new_species_hash.at[drop_targets].set(p1_hash, mode='drop')
    new_binding_energy = new_binding_energy.at[drop_targets].set(p1_be_all, mode='drop')
    new_half_life      = new_half_life.at[drop_targets].set(p1_hl_all, mode='drop')
    new_free_bonds     = new_free_bonds.at[drop_targets].set(p1_free_bonds, mode='drop')
    new_age            = new_age.at[drop_targets].set(jnp.float32(0.0), mode='drop')

    new_composites = composites._replace(
        members=new_members,
        member_count=new_member_count,
        alive=new_alive,
        binding_energy=new_binding_energy,
        half_life=new_half_life,
        age=new_age,
        species_hash=new_species_hash,
        free_bonds=new_free_bonds,
    )

    new_particles = particles._replace(
        composite_id=new_composite_id,
        velocity=new_velocity,
    )

    return state._replace(
        particles=new_particles,
        composites=new_composites,
        rng_key=key,
    )


# ── Fusion ───────────────────────────────────────────────────────────────────

def attempt_fusion(state: WorldState, neighbors: jnp.ndarray,
                   params: InteractionParams, config: SimConfig,
                   physics: PhysicsParams,
                   degree: jnp.ndarray = None,
                   species_valences: jnp.ndarray = None,
                   metrics=None) -> tuple:
    """
    Unified entity-entity fusion: any entity (free particle or composite) can
    fuse with any neighboring entity.

    An entity is represented by its lowest-index member (the "representative").
    Only representatives participate in the fusion scan to avoid double-counting.

    Three cases handled uniformly:
      - free + free   → create new composite
      - composite + free / free + composite → grow existing composite
      - composite + composite → merge smaller into larger (lower index wins)

    Args:
        state:     WorldState
        neighbors: (N, max_neighbors) int32
        config:    SimConfig (static)

    Returns:
        Updated WorldState with new/grown composites
    """
    # If callers didn't pass degree (legacy path), compute it locally so the
    # function works standalone too. New step.py path always passes it.
    if degree is None:
        degree = compute_degree(state.composites, config)
    if species_valences is None:
        species_valences = _species_valences(config)

    particles = state.particles
    composites = state.composites
    key, subkey = jax.random.split(state.rng_key)
    N = config.num_particles
    M = config.max_composite_size

    fusion_r2 = config.fusion_radius ** 2

    # ── Step 1: Identify representatives ──────────────────────────────────────
    def get_rep(i):
        c = jnp.clip(particles.composite_id[i], 0, config.max_composites - 1)
        is_free = particles.composite_id[i] < 0
        return jnp.where(is_free, i, composites.members[c, 0])

    all_reps = jax.vmap(get_rep)(jnp.arange(N, dtype=jnp.int32))  # (N,)
    is_rep = (all_reps == jnp.arange(N))  # (N,)

    # ── Pre-cache entity hashes (computed once, reused in check_neighbor) ─────
    # Commutative hash: H(i union j) = (H(i) + H(j)) % modulus — no sort needed.
    all_entity_hash, all_entity_cnt = jax.vmap(
        lambda i: _compute_entity_hash(i, particles, composites, config)
    )(jnp.arange(N, dtype=jnp.int32))  # (N,) uint32, (N,) int32

    # ── Pre-cache per-entity free bonds (only when valence gate is active) ───
    # free_bonds(entity) = Σ v_s − 2 × (n − 1). Fusion requires both entities
    # to have free_bonds ≥ 1 (one bond consumed on each side of the new edge).
    # Static-`if` on the config flag keeps this branch out of the trace when
    # the feature is off, so the toggle is genuinely zero-cost in that case.
    species_valences = _species_valences(config)  # (S,) int32, cheap to recompute
    if config.use_valence:
        all_entity_free_bonds = jax.vmap(
            lambda i: _entity_free_bonds(i, particles, composites,
                                          species_valences, config)
        )(jnp.arange(N, dtype=jnp.int32))  # (N,) int32
    else:
        # Unused stub. Same shape so closure types are stable across toggle.
        all_entity_free_bonds = jnp.zeros(N, dtype=jnp.int32)

    # ── Step 2: For each representative, find its best fusion partner ──────────
    def find_entity_partner(i):
        """
        For representative particle i, scan its neighbors to find the best
        entity partner. Returns (partner_rep, be_eff, merged_h, merged_count).
        """
        i_is_rep = is_rep[i]

        h_i   = all_entity_hash[i]
        cnt_i = all_entity_cnt[i]
        c_i   = jnp.clip(particles.composite_id[i], 0, config.max_composites - 1)

        def check_neighbor(j):
            j_is_rep = is_rep[j]
            valid = (
                (j >= 0) & (j != i) &
                j_is_rep &
                # Don't fuse same composite with itself
                ~((particles.composite_id[i] >= 0) &
                  (particles.composite_id[i] == particles.composite_id[j]))
            )
            # Distance check (between representative particles i and j)
            d = particles.position[i] - particles.position[j]
            if config.boundary_mode == "periodic":
                d = d - config.world_width  * jnp.round(d[0] / config.world_width) * jnp.array([1., 0.])
                d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
            dist2 = jnp.dot(d, d)
            in_range = dist2 < fusion_r2

            cnt_j = all_entity_cnt[j]
            merged_count = jnp.minimum(cnt_i + cnt_j, M)

            # Commutative merged hash — single addition, no sort, no scan
            merged_h = ((h_i.astype(jnp.int32) + all_entity_hash[j].astype(jnp.int32))
                        % config.hash_modulus).astype(jnp.uint32)

            be_eff = _hash_to_binding_energy(merged_h, config, physics)

            # Size cap: don't grow beyond buffer
            would_overflow = (cnt_i + cnt_j) > M

            # Valence gate: each entity must have at least one free bond, since
            # the new edge consumes one bond on each side. The merged composite's
            # free_bonds = free_bonds(i) + free_bonds(j) − 2, which is ≥ 0 whenever
            # both sides have ≥ 1. Gated by use_valence (zero-cost when off).
            if config.use_valence:
                has_free_bonds = (
                    (all_entity_free_bonds[i] >= 1)
                    & (all_entity_free_bonds[j] >= 1)
                )
            else:
                has_free_bonds = jnp.bool_(True)

            can_fuse = (
                valid & in_range
                & (be_eff > physics.fusion_threshold)
                & has_free_bonds
                & ~would_overflow
            )
            return (
                jnp.where(can_fuse, j,            jnp.int32(-1)),
                jnp.where(can_fuse, be_eff,        jnp.float32(0.0)),
                jnp.where(can_fuse, merged_h,      jnp.uint32(0)),
                jnp.where(can_fuse, merged_count,  jnp.int32(0)),
            )

        # vmap over neighbors
        nbrs = neighbors[i]
        results = jax.vmap(check_neighbor)(nbrs)
        partners, bes, hs, mcounts = results
        # (max_neighbors,), (max_neighbors,), ...

        best_idx = jnp.argmax(bes)
        best_j   = partners[best_idx]
        best_be  = bes[best_idx]
        best_h   = hs[best_idx]
        best_mc  = mcounts[best_idx]

        # Gate on i being a representative
        final_j  = jnp.where(i_is_rep, best_j,  jnp.int32(-1))
        final_be = jnp.where(i_is_rep, best_be,  jnp.float32(0.0))
        final_h  = jnp.where(i_is_rep, best_h,   jnp.uint32(0))
        final_mc = jnp.where(i_is_rep, best_mc,  jnp.int32(0))

        return final_j, final_be, final_h, final_mc

    all_partners, all_be, all_hashes, all_merged_counts = jax.vmap(
        find_entity_partner
    )(jnp.arange(N, dtype=jnp.int32))
    # (N,), (N,), (N,), (N,)

    # ── Step 3: Conflict resolution via sequential scan ────────────────────────
    max_fusions = config.max_fusions_per_step
    has_partner = all_partners >= 0
    cumsum_p    = jnp.cumsum(has_partner.astype(jnp.int32))
    candidate_i = jnp.where(
        has_partner & (cumsum_p <= max_fusions),
        jnp.arange(N, dtype=jnp.int32),
        N,
    )
    # NOTE: biased version (low-index particles always get priority):
    # scan_indices = jnp.sort(candidate_i)[:max_fusions]
    #
    # Fair random sample — randomly selects up to max_fusions candidates with no index bias.
    # Uses the same cumsum filter as the biased version, but over a shuffled ordering
    # so the first max_fusions valid candidates are a random draw, not lowest-index-first.
    perm = jax.random.permutation(subkey, N)
    shuffled_has_partner = has_partner[perm]
    cumsum_s = jnp.cumsum(shuffled_has_partner.astype(jnp.int32))
    candidate_i_shuffled = jnp.where(
        shuffled_has_partner & (cumsum_s <= max_fusions), perm, N
    )
    scan_indices = jnp.sort(candidate_i_shuffled)[:max_fusions]

    # Pre-compute free composite slots once (O(1) lookup in scan vs O(C) argmin)
    free_comp_slots = find_free_slots(composites.alive, max_fusions)  # (max_fusions,) int32

    def fusion_scan_body(carry, i):
        claimed, new_composite_id, composites_state, comp_count, free_slot_ptr = carry

        valid_i = i < N
        safe_i  = jnp.minimum(i, N - 1)

        j      = jnp.where(valid_i, all_partners[safe_i],      jnp.int32(-1))
        be_eff = jnp.where(valid_i, all_be[safe_i],            jnp.float32(0.0))
        h  = jnp.where(valid_i, all_hashes[safe_i],        jnp.uint32(0))
        mc = jnp.where(valid_i, all_merged_counts[safe_i], jnp.int32(0))

        safe_j = jnp.where(j >= 0, j, 0)

        can_fuse = (
            (j >= 0) &
            ~claimed[safe_i] &
            ~claimed[safe_j] &
            (comp_count < config.max_composites)
        )

        # Determine i/j free vs composite
        i_is_free = new_composite_id[safe_i] < 0
        j_is_free = new_composite_id[safe_j] < 0

        ci = jnp.clip(new_composite_id[safe_i], 0, config.max_composites - 1)
        cj = jnp.clip(new_composite_id[safe_j], 0, config.max_composites - 1)

        # Target composite slot (O(1) lookup instead of O(C) argmin)
        safe_ptr = jnp.clip(free_slot_ptr, 0, max_fusions - 1)
        free_comp_slot = free_comp_slots[safe_ptr]
        # Use jnp.where instead of jax.lax.switch — avoids GPU conditionals (2 switch
        # calls previously dominated fusion at 33ms/step in profiler traces).
        # target: free+free→new slot, i-comp+free→ci, free+j-comp→cj, comp+comp→min
        target = jnp.where(
            i_is_free,
            jnp.where(j_is_free, free_comp_slot, cj),
            jnp.where(j_is_free, ci, jnp.minimum(ci, cj)),
        )
        # absorbed: only comp+comp kills the higher-index composite
        absorbed = jnp.where(~i_is_free & ~j_is_free, jnp.maximum(ci, cj), jnp.int32(-1))

        # Energy-based half-life: high binding energy → stable, low → unstable
        t = jnp.clip(
            (be_eff - physics.fusion_threshold) / (1.0 - physics.fusion_threshold + 1e-8),
            0.0, 1.0
        )
        hl_base = config.half_life_min + (config.half_life_max - config.half_life_min) * t
        size_penalty = 1.0 + config.composite_size_decay_scale * jnp.maximum(
            0.0, mc.astype(jnp.float32) - 2.0
        )
        hl_eff = hl_base / size_penalty

        # New composite's free bonds: free_bonds(i) + free_bonds(j) − 2
        # (one bond consumed on each side of the new joining edge).
        merged_free_bonds = (
            all_entity_free_bonds[safe_i] + all_entity_free_bonds[safe_j] - jnp.int32(2)
        )

        # Build the merged member list: gather all member particle indices
        # i-side members
        i_members_comp = composites_state.members[ci]  # (M,)
        i_members_free = jnp.full(M, -1, dtype=jnp.int32).at[0].set(safe_i)
        i_members = jnp.where(i_is_free, i_members_free, i_members_comp)

        # j-side members
        j_members_comp = composites_state.members[cj]  # (M,)
        j_members_free = jnp.full(M, -1, dtype=jnp.int32).at[0].set(safe_j)
        j_members = jnp.where(j_is_free, j_members_free, j_members_comp)

        # Concat full member lists into a (2M,) buffer; compaction below trims to M.
        # would_overflow guarantees cnt_i + cnt_j <= M so no valid entries are lost.
        merged_members = jnp.concatenate([i_members, j_members])  # (2M,)

        # Compact valid IDs to front using cumsum — O(M), no separate argsort kernel.
        # Invalid entries are routed to index M (OOB) and dropped, preventing
        # write-conflicts with valid entries that land at index 0.
        valid_mask = merged_members >= 0
        pos     = jnp.cumsum(valid_mask.astype(jnp.int32)) - 1  # [0, n_valid)
        out_pos = jnp.where(valid_mask, pos, M)                  # invalid → OOB
        merged_members = jnp.full(M, -1, dtype=jnp.int32).at[out_pos].set(
            merged_members, mode='drop'
        )

        # Write to target composite
        safe_target = jnp.where(can_fuse, target, 0)
        safe_absorbed = jnp.where((absorbed >= 0) & can_fuse, absorbed, 0)

        new_members = composites_state.members.at[safe_target].set(
            jnp.where(can_fuse, merged_members, composites_state.members[safe_target])
        )
        new_comp_alive = composites_state.alive.at[safe_target].set(
            jnp.where(can_fuse, True, composites_state.alive[safe_target])
        )
        # Kill absorbed composite (comp+comp case)
        kill_absorbed = can_fuse & (absorbed >= 0)
        new_comp_alive = new_comp_alive.at[safe_absorbed].set(
            jnp.where(kill_absorbed, False, new_comp_alive[safe_absorbed])
        )
        new_comp_be = composites_state.binding_energy.at[safe_target].set(
            jnp.where(can_fuse, be_eff, composites_state.binding_energy[safe_target])
        )
        new_comp_hl = composites_state.half_life.at[safe_target].set(
            jnp.where(can_fuse, hl_eff, composites_state.half_life[safe_target])
        )
        new_comp_count_arr = composites_state.member_count.at[safe_target].set(
            jnp.where(can_fuse, mc, composites_state.member_count[safe_target])
        )
        new_comp_hash = composites_state.species_hash.at[safe_target].set(
            jnp.where(can_fuse, h, composites_state.species_hash[safe_target])
        )
        new_comp_free_bonds = composites_state.free_bonds.at[safe_target].set(
            jnp.where(can_fuse, merged_free_bonds, composites_state.free_bonds[safe_target])
        )
        new_composites = composites_state._replace(
            members=new_members,
            alive=new_comp_alive,
            binding_energy=new_comp_be,
            half_life=new_comp_hl,
            member_count=new_comp_count_arr,
            species_hash=new_comp_hash,
            free_bonds=new_comp_free_bonds,
        )

        # Update composite_id for all merged members
        # i-side: assign to target
        def assign_i_member(m_idx):
            pid = jnp.where(i_is_free, safe_i, i_members[m_idx])
            valid = can_fuse & (pid >= 0) & (
                jnp.where(i_is_free, m_idx == 0, m_idx < composites_state.member_count[ci])
            )
            return pid, valid

        # Route invalid entries to OOB index N (dropped) — see comment in
        # apply_composite_decay above. Without this, the M-1 invalid slots in
        # each scan iteration would all write to index 0 with the read-back
        # value, racing against any real write to particle 0 and clobbering it
        # non-deterministically.
        i_pids, i_valid = jax.vmap(assign_i_member)(jnp.arange(M, dtype=jnp.int32))
        drop_i_pids = jnp.where(i_valid, i_pids, N)
        new_composite_id = new_composite_id.at[drop_i_pids].set(target, mode='drop')

        # j-side: assign to target
        def assign_j_member(m_idx):
            pid = jnp.where(j_is_free, safe_j, j_members[m_idx])
            valid = can_fuse & (pid >= 0) & (
                jnp.where(j_is_free, m_idx == 0, m_idx < composites_state.member_count[cj])
            )
            return pid, valid

        j_pids, j_valid = jax.vmap(assign_j_member)(jnp.arange(M, dtype=jnp.int32))
        drop_j_pids = jnp.where(j_valid, j_pids, N)
        new_composite_id = new_composite_id.at[drop_j_pids].set(target, mode='drop')

        # Mark both representatives as claimed
        new_claimed = claimed.at[safe_i].set(claimed[safe_i] | can_fuse)
        new_claimed = new_claimed.at[safe_j].set(new_claimed[safe_j] | can_fuse)

        # Only increment comp_count for free+free (new composite created)
        new_comp_count = comp_count + jnp.where(
            can_fuse & i_is_free & j_is_free, jnp.int32(1), jnp.int32(0)
        )

        # Advance free-slot pointer only when a new composite slot is consumed
        new_free_slot_ptr = free_slot_ptr + jnp.where(
            can_fuse & i_is_free & j_is_free, jnp.int32(1), jnp.int32(0)
        )

        return (new_claimed, new_composite_id, new_composites, new_comp_count, new_free_slot_ptr), None

    claimed_init       = jnp.zeros(N, dtype=bool)
    composite_id_init  = particles.composite_id
    comp_count_init    = jnp.sum(composites.alive.astype(jnp.int32))
    free_slot_ptr_init = jnp.int32(0)

    (_, final_composite_id, final_composites, _, _), _ = jax.lax.scan(
        fusion_scan_body,
        (claimed_init, composite_id_init, composites, comp_count_init, free_slot_ptr_init),
        scan_indices,
    )

    new_particles = particles._replace(composite_id=final_composite_id)

    return state._replace(
        particles=new_particles,
        composites=final_composites,
        rng_key=key,
    ), degree
