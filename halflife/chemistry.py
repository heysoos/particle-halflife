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
  Composite decay fractures along the bond cut that maximizes total product
  binding energy (bond-cut fission). Fragments keep their internal bonds;
  species are conserved (no transmutation). The kick is the Q-value
  max(BE(p0) + BE(p1) − BE(parent), 0).

JIT notes:
  - No Python control flow inside JAX operations
  - Conflict resolution via lax.scan over a randomly shuffled candidate set
  - Dead composite slots recycled using cumsum-based free-slot finding
"""

import functools

import jax
import jax.numpy as jnp

from halflife.state import ParticleState, CompositeState, WorldState, InteractionParams, PhysicsParams, ReactionEvent
from halflife.config import SimConfig
from halflife.utils import find_free_slots
from halflife.graph import bfs_tree, subtree_sums, descendant_mask, reachable_mask
from halflife.analysis.events import KIND_FUSION, KIND_FISSION, KIND_NONE


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
                         config: SimConfig,
                         fusion_radius=None, repulsion_radius=None) -> jnp.ndarray:
    """
    Hash-derived bond rest length for species pair (s_i, s_j).

    Order-independent (uses the same commutative additive hash as composites)
    so r_rest[i, j] == r_rest[j, i] without an explicit symmetry pass.
    Re-mixed with a Fibonacci hash to decorrelate from binding energy and
    valence so universes with the same num_species but different hash_modulus
    get genuinely different bond chemistries.

    The rest length is a hash-derived fraction of fusion_radius, with the floor
    clamped to the hard core: r_rest ∈ [repulsion_radius, fusion_radius]. Bonds
    therefore never rest inside the hard core nor beyond the distance at which
    fusion fires, and the whole band auto-rescales when fusion_radius is tuned.

    fusion_radius / repulsion_radius default to the static config values (init
    time). The runtime path passes the live PhysicsParams scalars so the band
    tracks the fusion_radius (and repulsion_radius) sliders without a recompile.

    Returns:
        scalar float32 in [repulsion_radius, fusion_radius]
    """
    h_i = _entity_hash_val(s_i, config).astype(jnp.uint32)
    h_j = _entity_hash_val(s_j, config).astype(jnp.uint32)
    h = (h_i + h_j) % jnp.uint32(config.hash_modulus)
    # Fibonacci re-mix to decorrelate from BE / valence streams.
    h_mix = (h * jnp.uint32(0x9E3779B1)) ^ (h >> jnp.uint32(11))
    frac = (h_mix % jnp.uint32(1000)).astype(jnp.float32) / 999.0
    rep = config.repulsion_radius if repulsion_radius is None else repulsion_radius
    fus = config.fusion_radius    if fusion_radius    is None else fusion_radius
    lo = jnp.float32(rep)                          # clamp floor to hard core
    hi = jnp.maximum(jnp.float32(fus), lo)         # guard: never invert the band
    return lo + frac * (hi - lo)


@functools.partial(jax.jit, static_argnums=(0,))
def compute_r_rest_matrix(config: SimConfig, fusion_radius, repulsion_radius) -> jnp.ndarray:
    """
    Build the (num_species, num_species) hash-derived bond rest-length matrix
    spanning [repulsion_radius, fusion_radius] for the GIVEN radii.

    Called at init with the config values, and on every fusion_radius /
    repulsion_radius slider change with the live PhysicsParams scalars, so bond
    rest lengths track the sliders. The radii are dynamic args, so re-running it
    with a new slider value does NOT recompile; it recompiles only if the static
    `config` (num_species / hash constants) changes. Cost is O(num_species²) —
    ~4 orders of magnitude below one simulation step (see CLAUDE.md).
    """
    species_idx = jnp.arange(config.num_species, dtype=jnp.int32)
    return jax.vmap(
        lambda i: jax.vmap(
            lambda j: _hash_to_rest_length(i, j, config, fusion_radius, repulsion_radius)
        )(species_idx)
    )(species_idx)  # (S, S)

def _hash_to_bond_energy(s_i: jnp.ndarray, s_j: jnp.ndarray,
                         config: SimConfig) -> jnp.ndarray:
    """
    Hash-derived bond dissociation energy for species pair (s_i, s_j).

    Order-independent (commutative additive pair hash) and re-mixed with a
    Fibonacci-style constant DIFFERENT from the BE (2654435761, >>13),
    valence (0x9E3779B1, >>13) and rest-length (0x9E3779B1, >>11) streams so
    the four per-pair properties are mutually decorrelated.

    Returns: scalar float32 in [0, config.bond_energy_scale]
    """
    h_i = _entity_hash_val(s_i, config).astype(jnp.uint32)
    h_j = _entity_hash_val(s_j, config).astype(jnp.uint32)
    h = (h_i + h_j) % jnp.uint32(config.hash_modulus)
    h_mix = (h * jnp.uint32(0x85EBCA6B)) ^ (h >> jnp.uint32(9))
    frac = (h_mix % jnp.uint32(1000)).astype(jnp.float32) / 999.0
    return frac * config.bond_energy_scale


@functools.partial(jax.jit, static_argnums=(0,))
def compute_bond_energy_matrix(config: SimConfig) -> jnp.ndarray:
    """(num_species, num_species) dissociation-energy matrix. Static per
    config (like valence) — part of the universe, not of the run seed."""
    species_idx = jnp.arange(config.num_species, dtype=jnp.int32)
    return jax.vmap(
        lambda i: jax.vmap(
            lambda j: _hash_to_bond_energy(i, j, config)
        )(species_idx)
    )(species_idx)


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



def initialize_edges_for_existing_composites(composites: 'CompositeState',
                                              config: SimConfig) -> 'CompositeState':
    """
    For every alive composite, replace its edge list with a fresh path-
    spanning tree through its members in slot order. Used when toggling into
    'edges' bond mode from a state that didn't have edges populated.

    Free particles (composite_id == -1) are unaffected.

    Returns: updated CompositeState (composites only).
    """
    C = config.max_composites
    M = config.max_composite_size
    E = config.e_max
    e_idx = jnp.arange(E, dtype=jnp.int32)

    def per_composite(c):
        is_alive = composites.alive[c]
        n = composites.member_count[c]
        members = composites.members[c]
        valid_edge = (e_idx < jnp.maximum(n - jnp.int32(1), jnp.int32(0)))
        safe_k = jnp.minimum(e_idx, jnp.int32(M - 2))
        a = members[safe_k]
        b = members[safe_k + 1]
        a_out = jnp.where(valid_edge & is_alive, a, jnp.int32(-1))
        b_out = jnp.where(valid_edge & is_alive, b, jnp.int32(-1))
        new_edges = jnp.stack([a_out, b_out], axis=-1)  # (E, 2)
        new_count = jnp.where(
            is_alive, jnp.maximum(n - jnp.int32(1), jnp.int32(0)), jnp.int32(0)
        )
        return new_edges, new_count

    new_edges, new_edge_count = jax.vmap(per_composite)(
        jnp.arange(C, dtype=jnp.int32)
    )
    return composites._replace(edges=new_edges, edge_count=new_edge_count)


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


def _hl_from_be_and_size(be: jnp.ndarray, n: jnp.ndarray,
                         config: SimConfig, physics: PhysicsParams) -> jnp.ndarray:
    """
    Half-life from BE + size penalty. Same formula as fusion_scan_body /
    _fusion_apply_matching use inline; promoted to module level so the
    binary-split applier can reuse it for fission products.
    """
    t = jnp.clip((be - physics.fusion_threshold) / (1.0 - physics.fusion_threshold + 1e-8), 0.0, 1.0)
    hl_base = config.half_life_min + (config.half_life_max - config.half_life_min) * t
    size_penalty = 1.0 + config.composite_size_decay_scale * jnp.maximum(
        0.0, n.astype(jnp.float32) - 2.0
    )
    return hl_base / size_penalty


# ── Binary Split Applier (shared by fission and bond scission) ────────────────

def _apply_binary_splits(particles, composites, split_slots, fires, assignment,
                         kick_energy, config: SimConfig, physics: PhysicsParams):
    """
    Apply a batch of binary composite splits — the shared back half of
    half-life fission (apply_composite_decay) and chemical bond scission
    (apply_bond_scission, Phase B).

    For each batch row k with fires[k] True, the composite in slot
    split_slots[k] divides into product 0 (assignment[k] == 0) and product 1
    (assignment[k] == 1):

      - Product 0 reuses the parent slot; product 1 claims a fresh free slot.
      - Each product keeps exactly the parent edges internal to its member
        set — crossing edges break, and NO edges are ever minted. Bond
        lengths can therefore only come from bonds that already existed:
        this is the long-bond-bug fix.
      - Products of size 1 become free particles. Under valence, a product
        whose edge-based free bonds (Σ v_s − 2·edge_count) go negative
        shatters into free particles. Splits only remove edges, so degrees
        never grow and this is unreachable from a valid parent — kept as a
        cheap invariant guard.
      - kick_energy[k] splits equally between the two products as a kick
        along the product COM-COM axis (each product moves as a unit).

    Args:
        particles, composites: pre-split state pieces (authoritative)
        split_slots: (K,) int32 — parent composite slot per row; C = padding
        fires:       (K,) bool  — whether row k actually splits
        assignment:  (K, M) int32 ∈ {-1, 0, 1} — fragment label per member slot
        kick_energy: (K,) float32 — total kinetic energy released by row k
        config, physics: statics / runtime scalars

    Returns:
        (new_particles, new_composites, events) — events has leading dim K
        with kind == KIND_FISSION where fires. Callers discard events when
        config.emit_events is False; XLA DCEs the build code.
    """
    N = config.num_particles
    M = config.max_composite_size
    C = config.max_composites
    E_max = config.e_max
    K = split_slots.shape[0]
    m_idx = jnp.arange(M, dtype=jnp.int32)

    safe_slots = jnp.minimum(split_slots, C - 1)

    # Pre-allocate fresh composite slots for product 1: the k-th split takes
    # free_slots[k]. No collision with parents — parents are alive,
    # find_free_slots only returns dead slots.
    target_p1 = find_free_slots(composites.alive, K)  # (K,) int32, -1 = exhausted

    # ── Fragment lookup by particle id ───────────────────────────────────────
    # Batch rows are member-disjoint (a particle lives in one composite), so a
    # single (N,) array serves the whole batch.
    member_grid = composites.members[safe_slots]                       # (K, M)
    count_grid = composites.member_count[safe_slots]                   # (K,)
    valid_grid = (member_grid >= 0) & (m_idx[None, :] < count_grid[:, None]) \
                 & fires[:, None]
    flat_pids = jnp.where(valid_grid, member_grid, N).reshape(-1)
    frag_of_pid = jnp.full(N, -1, dtype=jnp.int32).at[flat_pids].set(
        assignment.reshape(-1), mode='drop')

    # ── Per-split: fragment COMs (min-image from member 0) ───────────────────
    def per_split(k):
        c = safe_slots[k]
        n = composites.member_count[c]
        member_ids = composites.members[c]
        safe_ids = jnp.where(member_ids >= 0, member_ids, 0)
        valid = (member_ids >= 0) & (m_idx < n)
        ref = particles.position[safe_ids[0]]

        def disp_from_ref(idx):
            d = particles.position[safe_ids[idx]] - ref
            if config.boundary_mode == "periodic":
                d = d - config.world_width  * jnp.round(d[0] / config.world_width)  * jnp.array([1., 0.])
                d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
            return d

        rels = jax.vmap(disp_from_ref)(jnp.arange(M))  # (M, 2)
        a = assignment[k]
        in_p0 = valid & (a == 0)
        in_p1 = valid & (a == 1)
        n0 = jnp.sum(in_p0.astype(jnp.float32))
        n1 = jnp.sum(in_p1.astype(jnp.float32))
        com0 = ref + jnp.sum(rels * in_p0[:, None].astype(jnp.float32), axis=0) / (n0 + 1e-8)
        com1 = ref + jnp.sum(rels * in_p1[:, None].astype(jnp.float32), axis=0) / (n1 + 1e-8)
        return com0, com1, n0.astype(jnp.int32), n1.astype(jnp.int32)

    all_com0, all_com1, all_n0, all_n1 = jax.vmap(per_split)(
        jnp.arange(K, dtype=jnp.int32))

    # ── Compact each product's members & compute its hash ────────────────────
    def per_product(k):
        c = safe_slots[k]
        member_ids = composites.members[c]
        n = composites.member_count[c]
        a = assignment[k]

        in_p0 = (a == 0) & (member_ids >= 0) & (m_idx < n)
        in_p1 = (a == 1) & (member_ids >= 0) & (m_idx < n)

        # Compact members of each product to front using cumsum (same trick
        # as fusion). Invalid entries route to OOB index M and drop.
        def compact(mask):
            pos = jnp.cumsum(mask.astype(jnp.int32)) - 1
            out = jnp.where(mask, pos, M)
            mem = jnp.full(M, -1, dtype=jnp.int32).at[out].set(member_ids, mode='drop')
            return mem, jnp.sum(mask.astype(jnp.int32))

        members_p0, count_p0 = compact(in_p0)
        members_p1, count_p1 = compact(in_p1)

        # Species hashes via commutative sum over each product's members.
        def hash_for_product(members_arr, count_arr):
            safe = jnp.where(members_arr >= 0, members_arr, 0)
            sp = particles.species[safe]
            valid_m = (members_arr >= 0) & (m_idx < count_arr)
            hvals = jax.vmap(lambda s: _entity_hash_val(s, config))(sp)
            return (jnp.sum(jnp.where(valid_m, hvals, 0)) % config.hash_modulus).astype(jnp.uint32)

        return (members_p0, count_p0, hash_for_product(members_p0, count_p0),
                members_p1, count_p1, hash_for_product(members_p1, count_p1))

    p0_members, p0_count, p0_hash, p1_members, p1_count, p1_hash = jax.vmap(per_product)(
        jnp.arange(K, dtype=jnp.int32))

    # ── Per-product edges: keep parent edges internal to each fragment ───────
    def split_edges(k):
        c = safe_slots[k]
        ga = composites.edges[c, :, 0]
        gb = composites.edges[c, :, 1]
        evalid = (jnp.arange(E_max) < composites.edge_count[c]) & (ga >= 0)
        fa = frag_of_pid[jnp.where(ga >= 0, ga, 0)]
        fb = frag_of_pid[jnp.where(gb >= 0, gb, 0)]

        def compact(keep):
            pos = jnp.cumsum(keep.astype(jnp.int32)) - 1
            out = jnp.where(keep, pos, E_max)
            e = jnp.full((E_max, 2), -1, dtype=jnp.int32).at[out].set(
                composites.edges[c], mode='drop')
            return e, jnp.sum(keep.astype(jnp.int32))

        e0, n_e0 = compact(evalid & (fa == 0) & (fb == 0))
        e1, n_e1 = compact(evalid & (fa == 1) & (fb == 1))
        return e0, n_e0, e1, n_e1

    p0_edges, p0_edge_count_all, p1_edges, p1_edge_count_all = jax.vmap(split_edges)(
        jnp.arange(K, dtype=jnp.int32))

    # ── Per-product free bonds (edge-based) and structural validity ──────────
    species_valences_split = _species_valences(config)

    def product_free_bonds(members_arr, count_arr, e_cnt):
        safe = jnp.where(members_arr >= 0, members_arr, 0)
        vs = species_valences_split[particles.species[safe]]
        valid_m = (members_arr >= 0) & (m_idx < count_arr)
        return jnp.sum(jnp.where(valid_m, vs, 0)) - jnp.int32(2) * e_cnt

    p0_free_bonds = jax.vmap(product_free_bonds)(p0_members, p0_count, p0_edge_count_all)
    p1_free_bonds = jax.vmap(product_free_bonds)(p1_members, p1_count, p1_edge_count_all)

    if config.use_valence:
        p0_valid = p0_free_bonds >= 0
        p1_valid = p1_free_bonds >= 0
    else:
        p0_valid = jnp.ones(K, dtype=bool)
        p1_valid = jnp.ones(K, dtype=bool)

    # ── Update each member particle's composite_id and velocity ──────────────
    def per_member(k, m):
        c = safe_slots[k]
        n = composites.member_count[c]
        member_id = composites.members[c, m]
        valid = fires[k] & (m < n) & (member_id >= 0)

        a = assignment[k, m]
        com0 = all_com0[k]
        com1 = all_com1[k]
        n0 = all_n0[k]
        n1 = all_n1[k]

        # Direction along COM-COM axis (min-image).
        d = com0 - com1
        if config.boundary_mode == "periodic":
            d = d - config.world_width  * jnp.round(d[0] / config.world_width)  * jnp.array([1., 0.])
            d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
        d_hat = d / (jnp.linalg.norm(d) + 1e-8)

        # Energy split: half of the released kick energy to each product.
        e_per = kick_energy[k] * 0.5
        v0 = jnp.sqrt(jnp.maximum(0.0, 2.0 * e_per / (n0.astype(jnp.float32) + 1e-8)))
        v1 = jnp.sqrt(jnp.maximum(0.0, 2.0 * e_per / (n1.astype(jnp.float32) + 1e-8)))

        # Kick: product 0 → +d_hat * v0, product 1 → -d_hat * v1.
        # Note: the kick always fires (even for shattered products), because
        # the energy release happens regardless of whether the pieces then
        # bind into a sub-composite or fly apart as free particles.
        kick = jnp.where(
            a == 0,
            d_hat * v0,
            jnp.where(a == 1, -d_hat * v1, jnp.zeros(2)),
        )

        # Composite-formation flags incorporate structural free-bond validity.
        forms_p0 = (n0 >= 2) & p0_valid[k]
        forms_p1 = (n1 >= 2) & p1_valid[k]

        # New composite_id:
        #   a==0 and forms_p0 → c (reuse parent slot)
        #   a==0 and not forms_p0 → -1 (free particle — size-1 or unsound)
        #   a==1 and forms_p1 → target_p1[k]
        #   a==1 and not forms_p1 → -1 (free)
        #   a==-1 (padding) → preserve original (the scatter is gated on
        #                     `valid`, so this value is never written).
        safe_member = jnp.where(member_id >= 0, member_id, 0)
        new_cid = jnp.where(
            a == 0,
            jnp.where(forms_p0, c, jnp.int32(-1)),
            jnp.where(a == 1,
                      jnp.where(forms_p1, target_p1[k], jnp.int32(-1)),
                      particles.composite_id[safe_member]),
        )
        return member_id, valid, new_cid, kick

    pid_grid, valid_grid_m, cid_grid, kick_grid = jax.vmap(
        lambda k: jax.vmap(lambda m: per_member(k, m))(jnp.arange(M, dtype=jnp.int32))
    )(jnp.arange(K, dtype=jnp.int32))
    # Shapes: (K, M), (K, M), (K, M), (K, M, 2)

    flat_pid   = pid_grid.reshape(-1)
    flat_valid = valid_grid_m.reshape(-1)
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

    # ── Write product 0 into the parent slot (in place) ──
    # AND with p0_valid so structurally unsound products leave the parent slot
    # dead (its members already got composite_id=-1 above), reclaiming the slot.
    p0_alive = fires & (p0_count >= 2) & p0_valid

    p0_be_all = jax.vmap(lambda h: _hash_to_binding_energy(h, config, physics))(p0_hash)
    p0_hl_all = jax.vmap(lambda be, n: _hl_from_be_and_size(be, n, config, physics))(
        p0_be_all, p0_count)

    # Scatter product-0 results into the parent slots (padding rows route to
    # OOB index C and drop).
    split_drop = jnp.where(fires, split_slots, C)
    new_alive = composites.alive.at[split_drop].set(p0_alive, mode='drop')
    new_members = composites.members.at[split_drop].set(p0_members, mode='drop')
    new_member_count = composites.member_count.at[split_drop].set(p0_count, mode='drop')
    new_species_hash = composites.species_hash.at[split_drop].set(p0_hash, mode='drop')
    new_binding_energy = composites.binding_energy.at[split_drop].set(p0_be_all, mode='drop')
    new_half_life = composites.half_life.at[split_drop].set(p0_hl_all, mode='drop')
    new_free_bonds = composites.free_bonds.at[split_drop].set(p0_free_bonds, mode='drop')
    # Reset age on the parent slot (it's now a fresh product).
    new_age = composites.age.at[split_drop].set(jnp.float32(0.0), mode='drop')
    new_edges = composites.edges.at[split_drop].set(p0_edges, mode='drop')
    new_edge_count = composites.edge_count.at[split_drop].set(p0_edge_count_all, mode='drop')

    # ── Write product 1 into target_p1[k] when it forms a composite ──
    p1_writes = fires & (p1_count >= 2) & p1_valid

    p1_be_all = jax.vmap(lambda h: _hash_to_binding_energy(h, config, physics))(p1_hash)
    p1_hl_all = jax.vmap(lambda be, n: _hl_from_be_and_size(be, n, config, physics))(
        p1_be_all, p1_count)

    # Guard against negative indices: find_free_slots returns -1 when there
    # aren't enough free slots, and JAX's negative-index default would wrap
    # to [C-1] — clobbering the last composite. Route those to C (OOB) so
    # mode='drop' actually drops them.
    drop_targets = jnp.where(
        p1_writes & (target_p1 >= 0),
        target_p1,
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
    new_edges          = new_edges.at[drop_targets].set(p1_edges, mode='drop')
    new_edge_count     = new_edge_count.at[drop_targets].set(p1_edge_count_all, mode='drop')

    new_composites = composites._replace(
        members=new_members,
        member_count=new_member_count,
        alive=new_alive,
        binding_energy=new_binding_energy,
        half_life=new_half_life,
        age=new_age,
        species_hash=new_species_hash,
        free_bonds=new_free_bonds,
        edges=new_edges,
        edge_count=new_edge_count,
    )

    new_particles = particles._replace(
        composite_id=new_composite_id,
        velocity=new_velocity,
    )

    # ── Per-split event emission (kind=2 fission; callers discard when off) ──
    # source = the parent composite BEFORE the state update; products may be
    # size 1 (shattered free particle) or — for a cycle-edge scission with no
    # actual split — product 1 may be empty (size 0, hash 0).
    ev_kind = jnp.where(fires, jnp.int32(KIND_FISSION), jnp.int32(KIND_NONE))
    ev_src_slots = jnp.stack([
        jnp.where(fires, split_slots, jnp.int32(-1)),
        jnp.full((K,), -1, dtype=jnp.int32),
    ], axis=1)
    ev_src_hashes = jnp.stack([
        jnp.where(fires, composites.species_hash[safe_slots], jnp.uint32(0)),
        jnp.zeros((K,), dtype=jnp.uint32),
    ], axis=1)
    ev_src_sizes = jnp.stack([
        jnp.where(fires, composites.member_count[safe_slots], jnp.int32(0)),
        jnp.zeros((K,), dtype=jnp.int32),
    ], axis=1)
    ev_prod_slots = jnp.stack([
        jnp.where(fires, split_slots, jnp.int32(-1)),
        jnp.where(fires, target_p1, jnp.int32(-1)),
    ], axis=1)
    ev_prod_hashes = jnp.stack([
        jnp.where(fires, p0_hash, jnp.uint32(0)),
        jnp.where(fires & (p1_count > 0), p1_hash, jnp.uint32(0)),
    ], axis=1)
    ev_prod_sizes = jnp.stack([
        jnp.where(fires, p0_count, jnp.int32(0)),
        jnp.where(fires, p1_count, jnp.int32(0)),
    ], axis=1)
    events = ReactionEvent(
        kind=ev_kind,
        source_slots=ev_src_slots,
        source_hashes=ev_src_hashes,
        source_sizes=ev_src_sizes,
        product_slots=ev_prod_slots,
        product_hashes=ev_prod_hashes,
        product_sizes=ev_prod_sizes,
    )

    return new_particles, new_composites, events


# ── Composite Decay / Fission ─────────────────────────────────────────────────

def apply_composite_decay(state: WorldState, config: SimConfig,
                           physics: PhysicsParams):
    """
    Half-life ("nuclear") fission with bond-cut fracture (2026-06-12).

    A decaying composite no longer partitions by hashing slot indices — it
    fractures along a BOND CUT: among the edges of its BFS spanning tree, the
    cut that maximizes total product binding energy (the hash-BE landscape
    acting as the shell-structure / magic-number analog) defines the two
    fragments. The additive commutative hash lets every cut be scored from
    one subtree-sum pass: frag hashes are (subtree, total − subtree). Each
    fragment keeps the parent edges internal to it, so fission never mints
    new bonds — slot order is irrelevant and the long-bond bug is gone.

    Energy: the kick is the Q-value max(BE(p0) + BE(p1) − BE(parent), 0),
    split equally between products along the COM-COM axis (replaces the old
    binding_energy * (1 − fission_cost) release). With
    config.forbid_endothermic_fission (default True), a roll whose best cut
    has Q < 0 is suppressed — the composite survives (fission barrier).

    Particle species are never modified — only composite_id and velocity.

    Perf: per-fission work runs over a compacted batch of at most
    config.max_fissions_per_step fissioning composites, not all C slots.
    Fissions beyond the budget defer to the next step (the composite stays
    alive and re-rolls). Graph sweeps are capped at
    config.fission_label_iters. When emit_events is on, the fission
    ReactionEvent batch has leading dim min(max_fissions_per_step, C).

    Args:
        state:   WorldState
        config:  SimConfig (static)
        physics: PhysicsParams — provides dt for the per-step decay probability

    Returns:
        Updated WorldState (and the ReactionEvent batch when emit_events)
    """
    particles = state.particles
    composites = state.composites
    key, subkey = jax.random.split(state.rng_key)
    N = config.num_particles
    M = config.max_composite_size
    C = config.max_composites
    E_max = config.e_max
    iters = config.fission_label_iters
    m_idx = jnp.arange(M, dtype=jnp.int32)

    # ── Roll for which composites decay this step ───────────────────────────
    rand = jax.random.uniform(subkey, (C,))
    ln2 = jnp.log(2.0)
    decay_prob = 1.0 - jnp.exp(-physics.dt * ln2 / (composites.half_life + 1e-10))
    fissions = composites.alive & (rand < decay_prob)  # (C,) bool

    # ── Compact fissioning composites to a fixed batch (perf, 2026-06-12) ───
    # Gather the fissioning slots into a (K_f,) batch and run the heavy math
    # only there. Fissions beyond the budget are deferred: the composite
    # stays alive, unchanged, and simply re-rolls its decay next step.
    K_f = min(config.max_fissions_per_step, C)
    fission_rank = jnp.cumsum(fissions.astype(jnp.int32)) - 1  # rank among fissioning
    selected = fissions & (fission_rank < K_f)
    cand = jnp.where(selected, jnp.arange(C, dtype=jnp.int32), C)
    fiss_idx = jnp.sort(cand)[:K_f]      # (K_f,) fissioning slot ids, C = padding
    fiss_valid = fiss_idx < C            # (K_f,)
    safe_fiss = jnp.minimum(fiss_idx, C - 1)

    # ── pid → local member slot, for the whole batch ─────────────────────────
    # Batch composites are member-disjoint, so one (N,) array serves all K_f.
    member_grid = composites.members[safe_fiss]                          # (K_f, M)
    count_grid = composites.member_count[safe_fiss]                      # (K_f,)
    valid_grid = (member_grid >= 0) & (m_idx[None, :] < count_grid[:, None]) \
                 & fiss_valid[:, None]
    flat = jnp.where(valid_grid, member_grid, N).reshape(-1)
    slot_of = jnp.zeros(N, dtype=jnp.int32).at[flat].set(
        jnp.tile(m_idx, K_f), mode='drop')

    # ── Per-fission: BFS tree → subtree sums → best bond cut ─────────────────
    def choose_cut(k):
        c = safe_fiss[k]
        n = composites.member_count[c]
        members = composites.members[c]
        valid_m = (members >= 0) & (m_idx < n)

        ga = composites.edges[c, :, 0]
        gb = composites.edges[c, :, 1]
        evalid = (jnp.arange(E_max) < composites.edge_count[c]) & (ga >= 0)
        la = slot_of[jnp.where(ga >= 0, ga, 0)]
        lb = slot_of[jnp.where(gb >= 0, gb, 0)]

        dist, parent = bfs_tree(la, lb, evalid, M, iters)

        # Per-slot hash values (masked to valid members) for the subtree pass.
        safe_members = jnp.where(members >= 0, members, 0)
        hvals = jax.vmap(lambda s: _entity_hash_val(s, config))(
            particles.species[safe_members]).astype(jnp.uint32)
        base_h = jnp.where(valid_m, hvals, jnp.uint32(0))
        base_c = valid_m.astype(jnp.int32)
        sub_h, sub_c = subtree_sums(parent, base_h, base_c, M, iters)

        # uint32 wraparound sum — same convention as the product hashes, so
        # the complement (total − subtree) is exact before the modulus.
        total_h = jnp.sum(base_h)

        # Candidate cut v ⇔ the spanning-tree edge (v, parent[v]).
        cand_v = valid_m & (parent >= 0)
        h1 = sub_h % jnp.uint32(config.hash_modulus)
        h0 = (total_h - sub_h) % jnp.uint32(config.hash_modulus)
        be1 = jax.vmap(lambda h: _hash_to_binding_energy(h, config, physics))(h1)
        be0 = jax.vmap(lambda h: _hash_to_binding_energy(h, config, physics))(h0)

        # Shell-effect analog: fracture along the cut that maximizes total
        # product binding energy. Deterministic (argmax ties → lowest slot).
        score = jnp.where(cand_v, be0 + be1, -jnp.inf)
        v = jnp.argmax(score)
        q = score[v] - composites.binding_energy[c]

        in_p1 = descendant_mask(parent, v.astype(jnp.int32), M, iters) & valid_m
        a = jnp.where(valid_m,
                      jnp.where(in_p1, jnp.int32(1), jnp.int32(0)),
                      jnp.int32(-1))
        has_cut = jnp.any(cand_v)  # size>=2 composites always have an edge; guard anyway
        return a, q, has_cut

    assignment, q_all, has_cut = jax.vmap(choose_cut)(jnp.arange(K_f, dtype=jnp.int32))

    fires = fiss_valid & has_cut
    if config.forbid_endothermic_fission:
        fires = fires & (q_all >= 0.0)
    kick = jnp.maximum(q_all, 0.0)

    new_particles, new_composites, events = _apply_binary_splits(
        particles, composites, fiss_idx, fires, assignment, kick, config, physics)

    new_state = state._replace(
        particles=new_particles,
        composites=new_composites,
        rng_key=key,
    )

    if config.emit_events:
        return new_state, events
    return new_state




# ── Fusion ───────────────────────────────────────────────────────────────────

def _fusion_apply_matching(state, particles, composites, key, subkey,
                           all_reps, all_entity_hash, all_entity_cnt,
                           all_partners, all_be, all_hashes, all_merged_counts,
                           degree, species_valences,
                           config: SimConfig, physics: PhysicsParams):
    """
    Parallel conflict resolution + batched apply for attempt_fusion
    (fusion_mode="matching").

    Inputs are the Step 1-2.5 products of attempt_fusion: after the per-entity
    dedup, each entity has at most one outgoing proposal (its best member-level
    candidate, identified by initiator particle i with all_partners[i] = j ≥ 0).

    Mutual-best handshake: entity A's proposal to entity B is accepted iff B's
    proposal targets A. Because each entity has exactly one outgoing proposal,
    mutual pairs are node-disjoint — no entity can appear in two accepted
    fusions — so the whole batch applies in one parallel pass with disjoint
    scatter targets, replacing the sequential `claimed` scan. Candidates whose
    handshake fails (A's best is B but B's best is C) simply retry next step,
    the same kind of one-step deferral the max_fusions budget already imposes.

    Merged-composite math (member/edge compaction, hash, BE, half-life,
    free bonds) is identical to fusion_scan_body, just vmapped over the K
    accepted pairs instead of run once per scan iteration.

    Note one intentional difference from the scan path: the scan blocked ALL
    fusions once `comp_count` reached max_composites (even comp+comp mergers,
    which reduce the count). Here only free+free pairs are blocked when no
    free composite slot exists; mergers into existing slots always proceed.

    Returns (new_state, final_degree, events) — events shaped like the scan
    path's stacked output (leading dim = max_fusions_per_step) so step.py's
    event-log concatenation is shape-compatible in both modes. The caller
    gates whether events are returned on config.emit_events; when discarded,
    XLA DCEs the event-build code.
    """
    N = config.num_particles
    M = config.max_composite_size
    C = config.max_composites
    E_max = config.e_max
    # Effective batch width. min() with N because the budget-selection sort
    # below slices an (N,) array — with N < max_fusions (tiny test worlds) the
    # slice yields (N,) and every downstream (K,) array must agree. The scan
    # path's scan_indices has the same effective length, so event-log shapes
    # stay consistent between modes.
    K = min(config.max_fusions_per_step, N)

    idx_n = jnp.arange(N, dtype=jnp.int32)

    # ── Entity-level proposal table, keyed by rep ────────────────────────────
    # Step 2.5 guarantees ≤1 winner per entity, so these scatters never race.
    winner_valid = all_partners >= 0
    safe_partner = jnp.where(winner_valid, all_partners, 0)
    drop_rep = jnp.where(winner_valid, all_reps, N)
    # out_init[r] = winning initiator particle of entity rep r (-1 = none)
    out_init = jnp.full(N, -1, dtype=jnp.int32).at[drop_rep].set(
        idx_n, mode='drop')
    # out_rep[r] = rep of the entity that r proposes to (-1 = none)
    out_rep = jnp.full(N, -1, dtype=jnp.int32).at[drop_rep].set(
        all_reps[safe_partner], mode='drop')

    # ── Handshake: keep pairs that chose each other ──────────────────────────
    safe_out = jnp.where(out_rep >= 0, out_rep, 0)
    mutual = (out_rep >= 0) & (out_rep[safe_out] == idx_n)
    # Canonical root = lower rep of the pair, so each pair is counted once.
    # The root's own proposal supplies the contact particles and merged
    # properties (hash/BE/size are symmetric in the pair, so the direction
    # only picks which contact edge is recorded).
    is_root = mutual & (idx_n < out_rep)

    # ── Budget selection: fair random sample of up to K pairs ────────────────
    # Same shuffled-cumsum trick as the scan path so no index bias.
    perm = jax.random.permutation(subkey, N)
    rooted = is_root[perm]
    cums = jnp.cumsum(rooted.astype(jnp.int32))
    sel = jnp.where(rooted & (cums <= K), perm, N)
    pair_roots = jnp.sort(sel)[:K]  # (K,) rep ids, N = padding

    valid_pair = pair_roots < N
    safe_root = jnp.minimum(pair_roots, N - 1)
    init_i = out_init[safe_root]
    safe_i = jnp.where(valid_pair & (init_i >= 0), init_i, 0)
    j = jnp.where(valid_pair, all_partners[safe_i], jnp.int32(-1))
    safe_j = jnp.where(j >= 0, j, 0)
    can_fuse = valid_pair & (init_i >= 0) & (j >= 0)

    be_eff = all_be[safe_i]
    h = all_hashes[safe_i]
    mc = all_merged_counts[safe_i]

    # Entity slots (pre-fusion state is authoritative: pairs are disjoint, so
    # no sequential composite_id updates are needed mid-batch).
    i_is_free = particles.composite_id[safe_i] < 0
    j_is_free = particles.composite_id[safe_j] < 0
    ci = jnp.clip(particles.composite_id[safe_i], 0, C - 1)
    cj = jnp.clip(particles.composite_id[safe_j], 0, C - 1)

    # ── Target slots ─────────────────────────────────────────────────────────
    # free+free → fresh slot (rank-th free slot), comp+free → the comp's slot,
    # comp+comp → min slot wins, max slot is absorbed (killed).
    free_comp_slots = find_free_slots(composites.alive, K)  # (min(K, C),) int32
    n_slots = free_comp_slots.shape[0]  # find_free_slots returns (min(K, C),)
    is_newslot = can_fuse & i_is_free & j_is_free
    slot_rank = jnp.cumsum(is_newslot.astype(jnp.int32)) - 1
    fresh = free_comp_slots[jnp.clip(slot_rank, 0, n_slots - 1)]
    # find_free_slots pads with -1 when the pool is exhausted → block the pair.
    # slot_rank < n_slots also guards the C < K case, where an unclipped rank
    # would silently re-read the last slot and alias two pairs to one target.
    fresh_ok = (slot_rank >= 0) & (slot_rank < n_slots) & (fresh >= 0)
    can_fuse = can_fuse & (~(i_is_free & j_is_free) | fresh_ok)

    target = jnp.where(
        i_is_free,
        jnp.where(j_is_free, fresh, cj),
        jnp.where(j_is_free, ci, jnp.minimum(ci, cj)),
    )
    absorbed = jnp.where(can_fuse & ~i_is_free & ~j_is_free,
                         jnp.maximum(ci, cj), jnp.int32(-1))

    # ── Energy-based half-life (same formula as fusion_scan_body) ────────────
    t = jnp.clip(
        (be_eff - physics.fusion_threshold) / (1.0 - physics.fusion_threshold + 1e-8),
        0.0, 1.0
    )
    hl_base = config.half_life_min + (config.half_life_max - config.half_life_min) * t
    size_penalty = 1.0 + config.composite_size_decay_scale * jnp.maximum(
        0.0, mc.astype(jnp.float32) - 2.0
    )
    hl_eff = hl_base / size_penalty

    # ── Per-pair merged members / edges (same math as fusion_scan_body) ──────
    def merge_pair(k):
        # i-side members
        i_members = jnp.where(
            i_is_free[k],
            jnp.full(M, -1, dtype=jnp.int32).at[0].set(safe_i[k]),
            composites.members[ci[k]],
        )
        # j-side members
        j_members = jnp.where(
            j_is_free[k],
            jnp.full(M, -1, dtype=jnp.int32).at[0].set(safe_j[k]),
            composites.members[cj[k]],
        )
        # Concat into (2M,); compact valid IDs to front using cumsum.
        # would_overflow in check_neighbor guarantees cnt_i + cnt_j <= M.
        mm = jnp.concatenate([i_members, j_members])
        vmask = mm >= 0
        pos = jnp.cumsum(vmask.astype(jnp.int32)) - 1
        outp = jnp.where(vmask, pos, M)  # invalid → OOB
        merged_members = jnp.full(M, -1, dtype=jnp.int32).at[outp].set(
            mm, mode='drop')

        # Merged edge list: i-edges + j-edges + the new contact edge
        # (safe_i, safe_j) — the actual nearest member-pair, as in the scan.
        i_edges = jnp.where(i_is_free[k],
                            jnp.full((E_max, 2), -1, dtype=jnp.int32),
                            composites.edges[ci[k]])
        j_edges = jnp.where(j_is_free[k],
                            jnp.full((E_max, 2), -1, dtype=jnp.int32),
                            composites.edges[cj[k]])
        new_edge = jnp.array([safe_i[k], safe_j[k]], dtype=jnp.int32)[None, :]
        eraw = jnp.concatenate([i_edges, j_edges, new_edge], axis=0)
        evalid = eraw[:, 0] >= 0
        epos = jnp.cumsum(evalid.astype(jnp.int32)) - 1
        eout = jnp.where(evalid, epos, E_max)
        merged_edges = jnp.full((E_max, 2), -1, dtype=jnp.int32).at[eout].set(
            eraw, mode='drop')
        merged_edge_count = jnp.sum(evalid.astype(jnp.int32))

        # free_bonds = Σ v_s − 2·edge_count over the merged member list
        mvalid = merged_members >= 0
        msp = particles.species[jnp.where(mvalid, merged_members, 0)]
        sum_v = jnp.sum(jnp.where(mvalid, species_valences[msp], 0))
        merged_free_bonds = sum_v - jnp.int32(2) * merged_edge_count
        return merged_members, merged_edges, merged_edge_count, merged_free_bonds

    merged_members, merged_edges, merged_edge_counts, merged_free_bonds = jax.vmap(
        merge_pair)(jnp.arange(K, dtype=jnp.int32))
    # Shapes: (K, M), (K, E_max, 2), (K,), (K,)

    # ── Batched composite writes ─────────────────────────────────────────────
    # Targets are distinct across accepted pairs (entity-disjoint matching ⇒
    # disjoint slots; fresh slots unique by rank), so scatter order is moot.
    drop_target = jnp.where(can_fuse, target, C)        # invalid → OOB, dropped
    drop_absorbed = jnp.where(absorbed >= 0, absorbed, C)

    new_alive = composites.alive.at[drop_target].set(True, mode='drop')
    new_alive = new_alive.at[drop_absorbed].set(False, mode='drop')
    new_composites = composites._replace(
        members=composites.members.at[drop_target].set(merged_members, mode='drop'),
        alive=new_alive,
        binding_energy=composites.binding_energy.at[drop_target].set(be_eff, mode='drop'),
        half_life=composites.half_life.at[drop_target].set(hl_eff, mode='drop'),
        member_count=composites.member_count.at[drop_target].set(mc, mode='drop'),
        species_hash=composites.species_hash.at[drop_target].set(h, mode='drop'),
        free_bonds=composites.free_bonds.at[drop_target].set(merged_free_bonds, mode='drop'),
        edges=composites.edges.at[drop_target].set(merged_edges, mode='drop'),
        edge_count=composites.edge_count.at[drop_target].set(merged_edge_counts, mode='drop'),
    )

    # ── composite_id for all merged members ─────────────────────────────────
    # The merged member list already contains both sides' members, so one
    # scatter covers free+free, comp+free, and comp+comp (absorbed members
    # included). Invalid rows route to OOB index N and drop.
    flat_pids = jnp.where(
        (merged_members >= 0) & can_fuse[:, None], merged_members, N
    ).reshape(-1)
    new_composite_id = particles.composite_id.at[flat_pids].set(
        jnp.repeat(target, M), mode='drop')

    # ── Degree update for the two new edge endpoints ─────────────────────────
    final_degree = degree.at[jnp.where(can_fuse, safe_i, N)].add(1, mode='drop')
    final_degree = final_degree.at[jnp.where(can_fuse, safe_j, N)].add(1, mode='drop')

    # ── Event batch (same per-event fields as the scan path) ─────────────────
    ev_i = jnp.where(can_fuse, safe_i, 0)
    ev_j = jnp.where(can_fuse, safe_j, 0)
    events = ReactionEvent(
        kind=jnp.where(can_fuse, jnp.int32(KIND_FUSION), jnp.int32(KIND_NONE)),
        source_slots=jnp.stack([
            jnp.where(can_fuse, safe_i, jnp.int32(-1)),
            jnp.where(can_fuse, safe_j, jnp.int32(-1)),
        ], axis=1),
        source_hashes=jnp.stack([
            jnp.where(can_fuse, all_entity_hash[ev_i], jnp.uint32(0)),
            jnp.where(can_fuse, all_entity_hash[ev_j], jnp.uint32(0)),
        ], axis=1),
        source_sizes=jnp.stack([
            jnp.where(can_fuse, all_entity_cnt[ev_i], jnp.int32(0)),
            jnp.where(can_fuse, all_entity_cnt[ev_j], jnp.int32(0)),
        ], axis=1),
        product_slots=jnp.stack([
            jnp.where(can_fuse, target, jnp.int32(-1)),
            jnp.full((K,), -1, dtype=jnp.int32),
        ], axis=1),
        product_hashes=jnp.stack([
            jnp.where(can_fuse, h, jnp.uint32(0)),
            jnp.zeros((K,), dtype=jnp.uint32),
        ], axis=1),
        product_sizes=jnp.stack([
            jnp.where(can_fuse, mc, jnp.int32(0)),
            jnp.zeros((K,), dtype=jnp.int32),
        ], axis=1),
    )

    new_state = state._replace(
        particles=particles._replace(composite_id=new_composite_id),
        composites=new_composites,
        rng_key=key,
    )
    return new_state, final_degree, events


def attempt_fusion(state: WorldState, neighbors: jnp.ndarray,
                   params: InteractionParams, config: SimConfig,
                   physics: PhysicsParams,
                   degree: jnp.ndarray = None,
                   species_valences: jnp.ndarray = None,
                   metrics=None) -> tuple:
    """
    Unified entity-entity fusion: any entity (free particle or composite) can
    fuse with any neighboring entity.

    Any member of an entity may initiate or accept a fusion: proximity is judged
    on the minimum member-member distance between two entities, not on their
    representatives. The lowest-index member (the "representative") survives only
    as a stable per-entity key — for the per-entity candidate dedup and for the
    `claimed` bookkeeping that still allows each entity at most one fusion per
    step (the double-counting guard).

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

    E_max = config.e_max
    fusion_r2 = physics.fusion_radius ** 2

    # Bit layout for the per-entity dedup key (Step 2.5): the low INDEX_BITS hold
    # an initiator particle index, the remaining bits hold a quantized BE rank.
    # Sign-safe in int32 (one bit reserved). Computed host-side from the static N.
    INDEX_BITS = max(1, N.bit_length())
    INDEX_MASK = (1 << INDEX_BITS) - 1
    BE_MAXQ    = (1 << (31 - INDEX_BITS - 1)) - 1

    # ── Step 1: Identify representatives ──────────────────────────────────────
    # The representative (lowest-index member) no longer gates the fusion *scan* —
    # any member may initiate/accept a fusion so composite-composite proximity is
    # judged on the minimum member-member distance, not rep-to-rep. The rep
    # survives only as a stable per-entity key for (a) the per-entity dedup below
    # and (b) the `claimed` bookkeeping in the conflict-resolution scan.
    def get_rep(i):
        c = jnp.clip(particles.composite_id[i], 0, config.max_composites - 1)
        is_free = particles.composite_id[i] < 0
        return jnp.where(is_free, i, composites.members[c, 0])

    all_reps = jax.vmap(get_rep)(jnp.arange(N, dtype=jnp.int32))  # (N,)

    # ── Pre-cache entity hashes (computed once, reused in check_neighbor) ─────
    # Commutative hash: H(i union j) = (H(i) + H(j)) % modulus — no sort needed.
    all_entity_hash, all_entity_cnt = jax.vmap(
        lambda i: _compute_entity_hash(i, particles, composites, config)
    )(jnp.arange(N, dtype=jnp.int32))  # (N,) uint32, (N,) int32

    # ── Pre-cache per-particle free bonds ─────────────────────────────────────
    # Per-particle: free_bond[i] = v_{species[i]} − degree[i].
    # For free particles degree[i] = 0 so free_bond[i] = v_{species[i]}.
    # For composite members this is stricter than the previous composite-level
    # check: requires the specific rep doing the fusion to have unused valence.
    all_particle_free_bonds = species_valences[particles.species] - degree  # (N,) int32

    # ── Step 2: For each member, find its best fusion partner ──────────────────
    def find_entity_partner(i):
        """
        For particle i (any member of any entity), scan its neighbors to find the
        best entity partner. Returns (partner_j, be_eff, merged_h, merged_count);
        partner_j is the contacting particle in the other entity.
        """
        h_i   = all_entity_hash[i]
        cnt_i = all_entity_cnt[i]
        c_i   = jnp.clip(particles.composite_id[i], 0, config.max_composites - 1)

        def check_neighbor(j):
            valid = (
                (j >= 0) & (j != i) &
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

            # Valence gate: each particle (NOT each composite) must have at
            # least one free bond. The new edge consumes one bond on each
            # endpoint, so both reps must individually have free_bond ≥ 1.
            if config.use_valence:
                has_free_bonds = (
                    (all_particle_free_bonds[i] >= 1)
                    & (all_particle_free_bonds[j] >= 1)
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

        # Any member may initiate — no rep gate. The per-entity dedup after this
        # vmap collapses each entity's members down to one candidate.
        return best_j, best_be, best_h, best_mc

    all_partners, all_be, all_hashes, all_merged_counts = jax.vmap(
        find_entity_partner
    )(jnp.arange(N, dtype=jnp.int32))
    # (N,), (N,), (N,), (N,)

    # ── Step 2.5: Per-entity dedup ─────────────────────────────────────────────
    # Dropping the rep gate means every member touching another entity is a
    # candidate, so a pair of overlapping composites emits many duplicate A↔B
    # candidates. Collapse them to one-per-entity BEFORE the budget-limited
    # selection below; otherwise duplicates burn `max_fusions` slots that the
    # `claimed` gate then no-ops (correct result, wasted throughput).
    #
    # Segmented arg-max keyed by the initiator's entity rep, as a single
    # scatter-max over a packed int32 key: high bits = quantized BE (the rank),
    # low bits = the initiator particle index (the payload we recover). be_q is
    # offset by +1 so any real candidate has key > 0, distinguishing it from the
    # no-partner sentinel (key 0) even when BE quantizes to 0 at particle 0.
    be_scale = jnp.maximum(physics.binding_energy_scale, jnp.float32(1e-6))
    be_q = (jnp.clip(all_be / be_scale, 0.0, 1.0) * BE_MAXQ).astype(jnp.int32) + 1
    cand_key = jnp.where(
        all_partners >= 0,
        (be_q << INDEX_BITS) | jnp.arange(N, dtype=jnp.int32),
        jnp.int32(0),
    )
    rep_best_key = jnp.zeros(N, dtype=jnp.int32).at[all_reps].max(cand_key)
    my_rep_best  = rep_best_key[all_reps]                          # (N,)
    is_winner = (
        (my_rep_best > 0)
        & ((my_rep_best & INDEX_MASK) == jnp.arange(N, dtype=jnp.int32))
    )
    all_partners = jnp.where(is_winner, all_partners, jnp.int32(-1))

    # ── Step 3 (matching mode): parallel mutual-best matching ──────────────────
    # Static dispatch on config.fusion_mode — XLA traces only the live branch.
    # See _fusion_apply_matching for the algorithm; the sequential scan below
    # is the legacy path ("scan"), kept for A/B comparison and rollback.
    if config.fusion_mode == "matching":
        new_state, final_degree, events = _fusion_apply_matching(
            state, particles, composites, key, subkey,
            all_reps, all_entity_hash, all_entity_cnt,
            all_partners, all_be, all_hashes, all_merged_counts,
            degree, species_valences, config, physics,
        )
        if config.emit_events:
            return new_state, final_degree, events
        return new_state, final_degree

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
        claimed, new_composite_id, composites_state, comp_count, free_slot_ptr, degree_carry = carry

        valid_i = i < N
        safe_i  = jnp.minimum(i, N - 1)

        j      = jnp.where(valid_i, all_partners[safe_i],      jnp.int32(-1))
        be_eff = jnp.where(valid_i, all_be[safe_i],            jnp.float32(0.0))
        h  = jnp.where(valid_i, all_hashes[safe_i],        jnp.uint32(0))
        mc = jnp.where(valid_i, all_merged_counts[safe_i], jnp.int32(0))

        safe_j = jnp.where(j >= 0, j, 0)

        # `claimed` tracks original entities (keyed by their rep) so an entity
        # fuses at most once per step even though any member can now initiate.
        rep_i = all_reps[safe_i]
        rep_j = all_reps[safe_j]

        can_fuse = (
            (j >= 0) &
            ~claimed[rep_i] &
            ~claimed[rep_j] &
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

        # ── Build merged edge list ─────────────────────────────────────────
        # i-side edges
        i_edges_comp = composites_state.edges[ci]                          # (E_max, 2)
        i_edges_free = jnp.full((E_max, 2), -1, dtype=jnp.int32)
        i_edges = jnp.where(i_is_free, i_edges_free, i_edges_comp)

        # j-side edges
        j_edges_comp = composites_state.edges[cj]
        j_edges_free = jnp.full((E_max, 2), -1, dtype=jnp.int32)
        j_edges = jnp.where(j_is_free, j_edges_free, j_edges_comp)

        # The new fusion edge connects the two contacting particles (safe_i,
        # safe_j) — the actual nearest member-pair that brought the entities
        # within fusion_radius, not their slot-0 reps. For free+free these are i
        # and j themselves; for free+comp the free particle on one side and the
        # touching member on the other. Both are already in the merged member
        # list via the merged-members construction.
        new_edge = jnp.array([safe_i, safe_j], dtype=jnp.int32)[None, :]   # (1, 2)

        # Concatenate (2·E_max + 1, 2) buffer
        merged_edges_raw = jnp.concatenate([i_edges, j_edges, new_edge], axis=0)
        edge_valid = merged_edges_raw[:, 0] >= 0  # both -1 or both valid
        # Compact valid entries to front, drop overflow
        epos    = jnp.cumsum(edge_valid.astype(jnp.int32)) - 1
        eout    = jnp.where(edge_valid, epos, E_max)  # invalid → OOB
        merged_edges = jnp.full((E_max, 2), -1, dtype=jnp.int32).at[eout].set(
            merged_edges_raw, mode='drop'
        )
        merged_edge_count = jnp.sum(edge_valid.astype(jnp.int32))

        # New composite's free bonds: Σ v_s − 2 · edge_count.
        # Member species sum (same logic as compute_composite_free_bonds but
        # against the just-computed merged_members).
        merged_member_species = particles.species[
            jnp.where(merged_members >= 0, merged_members, 0)
        ]
        merged_member_valid = merged_members >= 0
        sum_v = jnp.sum(jnp.where(
            merged_member_valid, species_valences[merged_member_species], 0
        ))
        merged_free_bonds = sum_v - jnp.int32(2) * merged_edge_count

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
        new_comp_edges = composites_state.edges.at[safe_target].set(
            jnp.where(can_fuse, merged_edges, composites_state.edges[safe_target])
        )
        new_comp_edge_count = composites_state.edge_count.at[safe_target].set(
            jnp.where(can_fuse, merged_edge_count, composites_state.edge_count[safe_target])
        )
        new_composites = composites_state._replace(
            members=new_members,
            alive=new_comp_alive,
            binding_energy=new_comp_be,
            half_life=new_comp_hl,
            member_count=new_comp_count_arr,
            species_hash=new_comp_hash,
            free_bonds=new_comp_free_bonds,
            edges=new_comp_edges,
            edge_count=new_comp_edge_count,
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

        # Mark both source entities (by rep) as claimed
        new_claimed = claimed.at[rep_i].set(claimed[rep_i] | can_fuse)
        new_claimed = new_claimed.at[rep_j].set(new_claimed[rep_j] | can_fuse)

        # Only increment comp_count for free+free (new composite created)
        new_comp_count = comp_count + jnp.where(
            can_fuse & i_is_free & j_is_free, jnp.int32(1), jnp.int32(0)
        )

        # Advance free-slot pointer only when a new composite slot is consumed
        new_free_slot_ptr = free_slot_ptr + jnp.where(
            can_fuse & i_is_free & j_is_free, jnp.int32(1), jnp.int32(0)
        )

        # Increment degree for the two new edge endpoints when fusion fires
        delta = can_fuse.astype(jnp.int32)
        degree_carry = degree_carry.at[safe_i].add(delta)
        degree_carry = degree_carry.at[safe_j].add(delta)

        # ── Per-iteration event emission (zero-cost when emit_events=False
        # because the outer attempt_fusion discards events_stack and XLA DCEs
        # the unused build code). ────────────────────────────────────────────
        # All inputs already exist in scan-body scope: safe_i, safe_j,
        # all_entity_hash, all_entity_cnt, h (merged hash), mc (merged size),
        # target (product slot). all_entity_hash/cnt are the per-rep ENTITY
        # caches (line 826) — for a composite rep they hold the whole
        # composite's hash/size, not just the rep particle's species. So
        # ev_src_hashes/sizes correctly encode the source entities for
        # both free+free and comp+free / comp+comp cases.
        # can_fuse gates emission to real events.
        ev_kind = jnp.where(can_fuse, jnp.int32(KIND_FUSION), jnp.int32(KIND_NONE))
        ev_src_slots = jnp.where(
            can_fuse,
            jnp.array([safe_i, safe_j], dtype=jnp.int32),
            jnp.array([-1, -1], dtype=jnp.int32),
        )
        ev_src_hashes = jnp.where(
            can_fuse,
            jnp.array([all_entity_hash[safe_i], all_entity_hash[safe_j]], dtype=jnp.uint32),
            jnp.array([0, 0], dtype=jnp.uint32),
        )
        ev_src_sizes = jnp.where(
            can_fuse,
            jnp.array([all_entity_cnt[safe_i], all_entity_cnt[safe_j]], dtype=jnp.int32),
            jnp.array([0, 0], dtype=jnp.int32),
        )
        ev_prod_slots = jnp.where(
            can_fuse,
            jnp.array([target, -1], dtype=jnp.int32),
            jnp.array([-1, -1], dtype=jnp.int32),
        )
        ev_prod_hashes = jnp.where(
            can_fuse,
            jnp.array([h, jnp.uint32(0)], dtype=jnp.uint32),
            jnp.array([0, 0], dtype=jnp.uint32),
        )
        ev_prod_sizes = jnp.where(
            can_fuse,
            jnp.array([mc, jnp.int32(0)], dtype=jnp.int32),
            jnp.array([0, 0], dtype=jnp.int32),
        )
        ev = ReactionEvent(
            kind=ev_kind,
            source_slots=ev_src_slots,
            source_hashes=ev_src_hashes,
            source_sizes=ev_src_sizes,
            product_slots=ev_prod_slots,
            product_hashes=ev_prod_hashes,
            product_sizes=ev_prod_sizes,
        )

        return (new_claimed, new_composite_id, new_composites, new_comp_count,
                new_free_slot_ptr, degree_carry), ev

    claimed_init       = jnp.zeros(N, dtype=bool)
    composite_id_init  = particles.composite_id
    comp_count_init    = jnp.sum(composites.alive.astype(jnp.int32))
    free_slot_ptr_init = jnp.int32(0)
    degree_init        = degree  # passed in from step.py via Task G

    (_, final_composite_id, final_composites, _, _, final_degree), events_stack = jax.lax.scan(
        fusion_scan_body,
        (claimed_init, composite_id_init, composites, comp_count_init, free_slot_ptr_init, degree_init),
        scan_indices,
    )

    new_particles = particles._replace(composite_id=final_composite_id)

    new_state = state._replace(
        particles=new_particles,
        composites=final_composites,
        rng_key=key,
    )
    # Gate the return on config.emit_events. When False the events_stack is
    # built (single trace path inside the scan body) but discarded here, and
    # XLA's DCE removes the unused build code so the live path stays cost-free.
    if config.emit_events:
        return new_state, final_degree, events_stack
    return new_state, final_degree


# ── Ring Closure (Phase 6b) ───────────────────────────────────────────────────

def attempt_ring_closure(state: WorldState, neighbors: jnp.ndarray,
                          params: InteractionParams, config: SimConfig,
                          physics: PhysicsParams,
                          degree: jnp.ndarray,
                          species_valences: jnp.ndarray) -> tuple:
    """
    Phase 6b: same-composite ring closure.

    For each pair of same-composite members within fusion_radius where both
    have per-particle free bonds (degree < v_s), add one new edge between them.
    Touches ONLY edges, edge_count, and degree — no member-list / composite_id
    changes.

    Conflict resolution follows config.fusion_mode: "matching" pairs mutual
    nearest neighbors and applies all closures in one batched pass;
    "scan" is the legacy sequential greedy scan.

    Gated by config.allow_ring_closure AND config.bond_mode == "edges" AND
    config.use_valence. In star_spring / off modes the edges array is
    physics-inert, so firing ring closure there would silently consume
    free_bonds and starve subsequent legitimate fusions — preserving the
    legacy mode's dynamics requires skipping it entirely. With use_valence
    off, ring closure is skipped because the mechanic is *defined* by
    free-bond accounting ("both members have a spare hand"); running it
    anyway made max_valence leak into valence-off dynamics (caught by
    test_valence_off_unchanged, 2026-06-12).

    Returns:
        (new_state, new_degree)
    """
    if (not config.allow_ring_closure or config.bond_mode != "edges"
            or not config.use_valence):
        return state, degree

    particles = state.particles
    composites = state.composites
    key, subkey = jax.random.split(state.rng_key)
    N = config.num_particles
    C = config.max_composites
    E_max = config.e_max
    fusion_r2 = physics.fusion_radius ** 2

    # ── Skip mask: only particles in composites with free_bonds ≥ 2 can host ─
    # a new ring edge. Composite-level free bonds = Σ free_bond[m] over members.
    composite_free_bonds = compute_composite_free_bonds(
        particles, composites, degree, species_valences, config
    )
    # Per-particle: free_bond[i] = v_{species[i]} - degree[i]
    particle_free_bonds = species_valences[particles.species] - degree  # (N,) int32
    # Per-particle skip: must be in a composite with ≥2 free bonds AND have ≥1.
    safe_cid = jnp.clip(particles.composite_id, 0, C - 1)
    can_attempt = (particles.composite_id >= 0) & \
                  (composite_free_bonds[safe_cid] >= 2) & \
                  (particle_free_bonds >= 1)  # (N,)

    # ── Find best ring partner per particle ─────────────────────────────────
    # require_j_gt_i is a static Python bool: the legacy scan considers each
    # pair once via the (j > i) filter; the matching path needs symmetric
    # proposals (i→j AND j→i) so the mutual handshake below can fire.
    def find_ring_partner(i, require_j_gt_i=True):
        nbrs = neighbors[i]
        i_attempt = can_attempt[i]
        cid_i = particles.composite_id[i]
        pos_i = particles.position[i]

        def check(j):
            valid = (
                (j >= 0) & (j != i) &
                ((j > i) if require_j_gt_i else jnp.bool_(True))  # consider each pair once (scan path)
                & can_attempt[j]
                & (particles.composite_id[j] == cid_i)  # same composite
            )
            d = pos_i - particles.position[j]
            if config.boundary_mode == "periodic":
                d = d - config.world_width  * jnp.round(d[0] / config.world_width) * jnp.array([1., 0.])
                d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
            dist2 = jnp.dot(d, d)
            in_range = dist2 < fusion_r2
            ok = valid & in_range
            return jnp.where(ok, j, jnp.int32(-1)), jnp.where(ok, dist2, jnp.float32(jnp.inf))

        partners, dists = jax.vmap(check)(nbrs)
        best_idx = jnp.argmin(dists)
        return jnp.where(i_attempt, partners[best_idx], jnp.int32(-1))

    # ── Matching mode: mutual-nearest handshake + batched edge append ───────
    # Same parallelization as _fusion_apply_matching, but at particle level:
    # a pair closes a ring iff each particle's nearest eligible same-composite
    # partner is the other. Mutual pairs are particle-disjoint, so the input
    # `degree` is exact for the valence recheck (no sequential carry needed)
    # and all edge appends can scatter in one pass. Losers retry next step.
    if config.fusion_mode == "matching":
        idx_n = jnp.arange(N, dtype=jnp.int32)
        all_partners = jax.vmap(
            lambda i: find_ring_partner(i, require_j_gt_i=False))(idx_n)  # (N,)
        safe_p = jnp.where(all_partners >= 0, all_partners, 0)
        mutual = (all_partners >= 0) & (all_partners[safe_p] == idx_n)
        is_root = mutual & (idx_n < all_partners)  # canonical: count pairs once

        # Budget: fair random sample of up to R pairs (same trick as fusion).
        R = min(config.max_ring_closures_per_step, N)
        perm = jax.random.permutation(subkey, N)
        rooted = is_root[perm]
        cums = jnp.cumsum(rooted.astype(jnp.int32))
        sel = jnp.where(rooted & (cums <= R), perm, N)
        pair_i = jnp.sort(sel)[:R]  # (R,) root particle ids, N = padding

        valid_pair = pair_i < N
        safe_i = jnp.minimum(pair_i, N - 1)
        j = jnp.where(valid_pair, all_partners[safe_i], jnp.int32(-1))
        safe_j = jnp.where(j >= 0, j, 0)
        can_close = valid_pair & (j >= 0)

        cid = jnp.clip(particles.composite_id[safe_i], 0, C - 1)

        # Dedup: is (safe_i, safe_j) already in edges[cid]? (Catches e.g. the
        # contact edge a fusion created earlier this same step.)
        def has_edge(k):
            ce = composites.edges[cid[k]]  # (E_max, 2)
            return jnp.any(
                ((ce[:, 0] == safe_i[k]) & (ce[:, 1] == safe_j[k])) |
                ((ce[:, 0] == safe_j[k]) & (ce[:, 1] == safe_i[k]))
            )
        already = jax.vmap(has_edge)(jnp.arange(R, dtype=jnp.int32))
        can_close = can_close & ~already

        # Per-particle valence recheck against the (post-fusion) input degree.
        # Pairs are particle-disjoint, so no within-batch compounding is
        # possible — the static check is exact, unlike the scan's live carry.
        free_i = species_valences[particles.species[safe_i]] - degree[safe_i]
        free_j = species_valences[particles.species[safe_j]] - degree[safe_j]
        can_close = can_close & (free_i >= 1) & (free_j >= 1)

        # Slot assignment: multiple accepted pairs in the SAME composite must
        # land in consecutive edge slots. rank_in_cid = how many earlier
        # accepted pairs share this cid (O(R²) mask, R ≤ 16 — trivial).
        k_idx = jnp.arange(R, dtype=jnp.int32)
        same_cid_before = (
            (cid[:, None] == cid[None, :])
            & can_close[:, None] & can_close[None, :]
            & (k_idx[None, :] < k_idx[:, None])
        )
        rank_in_cid = jnp.sum(same_cid_before, axis=1)
        slot = composites.edge_count[cid] + rank_in_cid
        can_close = can_close & (slot < E_max)  # no append past a full edge buffer

        # Batched writes: flat-index edge scatter, counted appends, degree.
        flat_idx = jnp.where(can_close, cid * E_max + slot, C * E_max)  # OOB → drop
        new_pair = jnp.stack([safe_i, safe_j], axis=1)  # (R, 2)
        edges_flat = composites.edges.reshape(-1, 2).at[flat_idx].set(
            new_pair, mode='drop')
        drop_cid = jnp.where(can_close, cid, C)
        new_composites = composites._replace(
            edges=edges_flat.reshape(C, E_max, 2),
            edge_count=composites.edge_count.at[drop_cid].add(1, mode='drop'),
            free_bonds=composites.free_bonds.at[drop_cid].add(-2, mode='drop'),
        )
        new_degree = degree.at[jnp.where(can_close, safe_i, N)].add(1, mode='drop')
        new_degree = new_degree.at[jnp.where(can_close, safe_j, N)].add(1, mode='drop')

        return state._replace(composites=new_composites, rng_key=key), new_degree

    all_partners = jax.vmap(find_ring_partner)(jnp.arange(N, dtype=jnp.int32))  # (N,)

    # ── Conflict resolution: take up to max_ring_closures candidates ────────
    has_partner = all_partners >= 0
    perm = jax.random.permutation(subkey, N)
    shuffled = has_partner[perm]
    cum = jnp.cumsum(shuffled.astype(jnp.int32))
    cand = jnp.where(shuffled & (cum <= config.max_ring_closures_per_step), perm, N)
    scan_indices = jnp.sort(cand)[:config.max_ring_closures_per_step]

    def ring_body(carry, i):
        composites_state, degree_carry, claimed = carry
        valid_i = i < N
        safe_i = jnp.minimum(i, N - 1)
        j = jnp.where(valid_i, all_partners[safe_i], jnp.int32(-1))
        safe_j = jnp.where(j >= 0, j, 0)

        # Recheck per-particle valence using the live degree.
        free_i_now = species_valences[particles.species[safe_i]] - degree_carry[safe_i]
        free_j_now = species_valences[particles.species[safe_j]] - degree_carry[safe_j]

        # Dedup: is (safe_i, safe_j) already in edges[cid]?
        cid = jnp.clip(particles.composite_id[safe_i], 0, C - 1)
        c_edges = composites_state.edges[cid]  # (E_max, 2)
        already = jnp.any(
            ((c_edges[:, 0] == safe_i) & (c_edges[:, 1] == safe_j)) |
            ((c_edges[:, 0] == safe_j) & (c_edges[:, 1] == safe_i))
        )

        can_close = (
            (j >= 0)
            & ~claimed[safe_i] & ~claimed[safe_j]
            & (free_i_now >= 1) & (free_j_now >= 1)
            & ~already
        )

        # Append (safe_i, safe_j) to edges[cid] at slot edge_count[cid].
        slot = composites_state.edge_count[cid]
        safe_slot = jnp.where(can_close, slot, jnp.int32(E_max))  # OOB → drop
        new_edge = jnp.where(can_close, jnp.array([safe_i, safe_j], dtype=jnp.int32),
                              jnp.array([-1, -1], dtype=jnp.int32))
        c_edges_new = c_edges.at[safe_slot].set(new_edge, mode='drop')
        composites_state = composites_state._replace(
            edges=composites_state.edges.at[cid].set(c_edges_new),
            edge_count=composites_state.edge_count.at[cid].set(
                jnp.where(can_close, slot + 1, composites_state.edge_count[cid])
            ),
            free_bonds=composites_state.free_bonds.at[cid].set(
                jnp.where(can_close, composites_state.free_bonds[cid] - 2,
                           composites_state.free_bonds[cid])
            ),
        )

        # Update degree
        delta = can_close.astype(jnp.int32)
        degree_carry = degree_carry.at[safe_i].add(delta)
        degree_carry = degree_carry.at[safe_j].add(delta)

        # Mark claimed
        claimed = claimed.at[safe_i].set(claimed[safe_i] | can_close)
        claimed = claimed.at[safe_j].set(claimed[safe_j] | can_close)

        return (composites_state, degree_carry, claimed), None

    (final_composites, final_degree, _), _ = jax.lax.scan(
        ring_body,
        (composites, degree, jnp.zeros(N, dtype=bool)),
        scan_indices,
    )

    return state._replace(composites=final_composites, rng_key=key), final_degree


# ── Chemical Bond Scission (per-bond breaking channel) ────────────────────────

def apply_bond_scission(state: WorldState, params: InteractionParams,
                        config: SimConfig, physics: PhysicsParams):
    """
    Chemical (per-bond) breaking — the kinetic/thermal counterpart to
    half-life fission. Makes the harmonic well finite: every edge carries a
    hash-derived dissociation energy E_b (_hash_to_bond_energy), and each step

      kinetic: stretch strain 0.5·k_bond·max(r − r_rest, 0)² >= E_b snaps the
               bond deterministically;
      thermal: below threshold, the bond snaps with Arrhenius probability
               P = 1 − exp(−dt · ν0 · exp(−(E_b − strain)/kT)).

    Compression never breaks a bond. At most ONE bond per composite breaks
    per step (the most-overstretched breaking edge), and at most
    config.max_scissions_per_step composites break per step (excess defers a
    step, like the fusion/fission budgets). If the broken bond was a bridge,
    the composite splits into its two connected halves via
    _apply_binary_splits with zero kick (the spring's stored energy simply
    stops acting — the pairwise forces take over); if it was a ring edge,
    only the edge is removed (members and composite_id untouched, though the
    slot's hash-derived properties and age are refreshed by the applier).

    Requires bond_mode == "edges" (step.py gates statically; the early
    return below covers standalone use).

    Returns:
        Updated WorldState (and a ReactionEvent batch of leading dim
        min(max_scissions_per_step, C) when config.emit_events).
    """
    if not (config.enable_bond_scission and config.bond_mode == "edges"):
        return state

    particles = state.particles
    composites = state.composites
    key, subkey = jax.random.split(state.rng_key)
    N = config.num_particles
    M = config.max_composite_size
    C = config.max_composites
    E_max = config.e_max
    iters = config.fission_label_iters
    m_idx = jnp.arange(M, dtype=jnp.int32)
    e_idx = jnp.arange(E_max, dtype=jnp.int32)

    # ── Per-edge strain vs dissociation energy, over the (C, E) grid ────────
    ga = composites.edges[:, :, 0]   # (C, E)
    gb = composites.edges[:, :, 1]
    evalid = composites.alive[:, None] & (e_idx[None, :] < composites.edge_count[:, None]) & (ga >= 0)
    safe_a = jnp.where(ga >= 0, ga, 0)
    safe_b = jnp.where(gb >= 0, gb, 0)
    pa = particles.position[safe_a]  # (C, E, 2)
    pb = particles.position[safe_b]
    d = pa - pb
    if config.boundary_mode == "periodic":
        d = d - config.world_width  * jnp.round(d[..., 0:1] / config.world_width)  * jnp.array([1., 0.])
        d = d - config.world_height * jnp.round(d[..., 1:2] / config.world_height) * jnp.array([0., 1.])
    r = jnp.linalg.norm(d, axis=-1)  # (C, E)
    sa = particles.species[safe_a]
    sb = particles.species[safe_b]
    r_rest = params.r_rest[sa, sb] * physics.r_rest_scale
    # Only stretch strains a bond; compression never breaks it.
    stretch = jnp.maximum(r - r_rest, 0.0)
    strain = 0.5 * physics.k_bond * stretch ** 2

    bond_e = compute_bond_energy_matrix(config)[sa, sb]  # (C, E)

    kT = jnp.maximum(jnp.float32(config.bond_temperature), 1e-8)
    barrier = jnp.maximum(bond_e - strain, 0.0)
    rate = config.bond_break_attempt_rate * jnp.exp(-barrier / kT)
    p_thermal = 1.0 - jnp.exp(-physics.dt * rate)
    u = jax.random.uniform(subkey, (C, E_max))
    breaks = evalid & ((strain >= bond_e) | (u < p_thermal))

    # ── One break per composite: the most-overstretched breaking edge ───────
    over = jnp.where(breaks, strain - bond_e, -jnp.inf)
    chosen_e = jnp.argmax(over, axis=1).astype(jnp.int32)  # (C,)
    has_break = jnp.any(breaks, axis=1)                    # (C,)

    # ── Budget-compact to a (K_s,) batch (same trick as fission) ────────────
    K_s = min(config.max_scissions_per_step, C)
    rank = jnp.cumsum(has_break.astype(jnp.int32)) - 1
    sel = has_break & (rank < K_s)
    cand = jnp.where(sel, jnp.arange(C, dtype=jnp.int32), C)
    sciss_idx = jnp.sort(cand)[:K_s]
    sciss_valid = sciss_idx < C
    safe_sc = jnp.minimum(sciss_idx, C - 1)
    cut_e = chosen_e[safe_sc]                              # (K_s,)

    # ── Remove the chosen edge from each selected composite (compact) ───────
    def drop_edge(k):
        c = safe_sc[k]
        keep = (e_idx < composites.edge_count[c]) & (composites.edges[c, :, 0] >= 0) \
               & (e_idx != cut_e[k])
        pos = jnp.cumsum(keep.astype(jnp.int32)) - 1
        out = jnp.where(keep, pos, E_max)
        new_e = jnp.full((E_max, 2), -1, dtype=jnp.int32).at[out].set(
            composites.edges[c], mode='drop')
        return new_e, jnp.sum(keep.astype(jnp.int32))

    new_edges_k, new_ecnt_k = jax.vmap(drop_edge)(jnp.arange(K_s, dtype=jnp.int32))
    drop_slots = jnp.where(sciss_valid, sciss_idx, C)
    composites_cut = composites._replace(
        edges=composites.edges.at[drop_slots].set(new_edges_k, mode='drop'),
        edge_count=composites.edge_count.at[drop_slots].set(new_ecnt_k, mode='drop'),
    )

    # ── pid → local slot for the batch (member-disjoint rows) ───────────────
    member_grid = composites.members[safe_sc]
    count_grid = composites.member_count[safe_sc]
    valid_grid = (member_grid >= 0) & (m_idx[None, :] < count_grid[:, None]) \
                 & sciss_valid[:, None]
    flat = jnp.where(valid_grid, member_grid, N).reshape(-1)
    slot_of = jnp.zeros(N, dtype=jnp.int32).at[flat].set(
        jnp.tile(m_idx, K_s), mode='drop')

    # ── Bipartition by reachability over the remaining edges ────────────────
    # Fragment 0 = everything still reachable from the removed edge's "a"
    # endpoint; fragment 1 = the rest. If the removed edge was a ring edge,
    # everything stays reachable → fragment 1 is empty → the applier writes
    # product 0 (the whole composite, minus the edge) back to the parent slot.
    def label_split(k):
        c = safe_sc[k]
        n = composites.member_count[c]
        members = composites.members[c]
        valid_m = (members >= 0) & (m_idx < n)

        rga = composites_cut.edges[c, :, 0]
        rgb = composites_cut.edges[c, :, 1]
        revalid = (e_idx < composites_cut.edge_count[c]) & (rga >= 0)
        la = slot_of[jnp.where(rga >= 0, rga, 0)]
        lb = slot_of[jnp.where(rgb >= 0, rgb, 0)]

        # local slot of the removed edge's first endpoint (from ORIGINAL edges)
        cut_a_pid = composites.edges[c, cut_e[k], 0]
        start = slot_of[jnp.where(cut_a_pid >= 0, cut_a_pid, 0)]

        reach = reachable_mask(la, lb, revalid, start, M, iters)
        a = jnp.where(valid_m,
                      jnp.where(reach, jnp.int32(0), jnp.int32(1)),
                      jnp.int32(-1))
        return a

    assignment = jax.vmap(label_split)(jnp.arange(K_s, dtype=jnp.int32))

    # No kick: the snapped spring just stops acting; pairwise forces take over.
    kick = jnp.zeros(K_s, dtype=jnp.float32)

    new_particles, new_composites, events = _apply_binary_splits(
        particles, composites_cut, sciss_idx, sciss_valid, assignment, kick,
        config, physics)

    new_state = state._replace(
        particles=new_particles,
        composites=new_composites,
        rng_key=key,
    )

    if config.emit_events:
        return new_state, events
    return new_state
