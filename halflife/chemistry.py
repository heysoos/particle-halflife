"""
Hash-based reaction system: fusion, decay, and fission.

This is the intellectual core of Half-Life. Instead of a lookup table,
reaction rules are implicit in a polynomial rolling hash applied to the
sorted multiset of participant species IDs. This is inspired by Hiroki
Sayama's Hash Chemistry.

  Same species set → same hash → same composite properties, every time.
  Different hash constants (config) → different "universe" / chemistry.

Fusion:
  Two free particles within fusion_radius whose hash-derived binding energy
  exceeds fusion_threshold become a composite.

Decay:
  Each particle and composite has P(decay in dt) = 1 - exp(-dt*ln2/half_life).
  Decaying entities are replaced by 1-3 product particles whose species are
  derived from the same hash.

Fission:
  Composite decay releases member particles back to free state and injects
  binding_energy (minus fission_cost) as kinetic energy.

JIT notes:
  - No Python control flow inside JAX operations
  - Conflict resolution via lax.scan in particle-index order
  - Dead slots recycled using cumsum-based free-slot finding
"""

import jax
import jax.numpy as jnp

from halflife.state import ParticleState, CompositeState, WorldState, InteractionParams
from halflife.config import SimConfig
from halflife.utils import find_free_slots


# ── Hash Utilities ────────────────────────────────────────────────────────────

def _hash_sorted_species(species_sorted: jnp.ndarray, count: jnp.ndarray,
                          config: SimConfig) -> jnp.ndarray:
    """
    Polynomial rolling hash over a sorted species multiset.
    Result is a uint32 that uniquely (modulo collisions) identifies
    the chemical composition.

    species_sorted: (max_composite_size,) int32, padded with -1
    count:          scalar int32 — valid entries
    """
    def body(h, i):
        s = species_sorted[i]
        valid = (i < count) & (s >= 0)
        new_h = (h * config.hash_prime_a + s + config.hash_prime_b) % config.hash_modulus
        return jnp.where(valid, new_h, h), None

    h0 = jnp.array(1, dtype=jnp.int32)
    h_final, _ = jax.lax.scan(body, h0, jnp.arange(config.max_composite_size))
    return h_final.astype(jnp.uint32)


def _hash_to_half_life(h: jnp.ndarray, config: SimConfig) -> jnp.ndarray:
    """Derive composite half-life from its species hash."""
    # Use bits [0..9] of h (normalized to [0,1]) to scale the half-life range
    frac = ((h % 1000).astype(jnp.float32)) / 999.0
    base = (config.half_life_min + config.half_life_max) * 0.5
    spread = (config.half_life_max - config.half_life_min) * 0.5
    return (base + spread * (frac * 2.0 - 1.0)) * config.composite_half_life_scale


def _hash_to_binding_energy(h: jnp.ndarray, config: SimConfig) -> jnp.ndarray:
    """Derive binding energy from species hash (normalized to [0, 1])."""
    frac = (((h // 1000) % 1000).astype(jnp.float32)) / 999.0
    return frac * config.binding_energy_scale


def _hash_to_decay_products(h: jnp.ndarray, parent_species: jnp.ndarray,
                              config: SimConfig):
    """
    Derive decay product species from a hash value.

    Returns:
        product_species: (max_decay_products,) int32 — species of each product
        product_count:   scalar int32 — how many products (1 to max_decay_products)
    """
    # Number of products: 1..max_decay_products, biased toward 1-2
    n_prods = jnp.array(1, dtype=jnp.int32) + (
        ((h >> 20) % config.max_decay_products).astype(jnp.int32)
    )

    # Species of each product: derived from hash bits
    def get_product_species(k):
        bits = (h * jnp.uint32(1000003 * (k + 1))) % jnp.uint32(config.hash_modulus)
        sp = (bits % config.num_species).astype(jnp.int32)
        return sp

    product_species = jax.vmap(get_product_species)(
        jnp.arange(config.max_decay_products, dtype=jnp.uint32)
    )
    return product_species, n_prods


# ── Decay ─────────────────────────────────────────────────────────────────────

def apply_particle_decay(state: WorldState, config: SimConfig) -> WorldState:
    """
    Apply probabilistic decay to all alive free particles.

    P(decay in dt) = 1 - exp(-dt * ln2 / half_life)

    Decaying particles are killed and replaced by product particles
    in nearby positions with a share of the parent's kinetic energy.

    Args:
        state:  WorldState
        config: SimConfig (static)

    Returns:
        Updated WorldState
    """
    particles = state.particles
    key, subkey = jax.random.split(state.rng_key)
    N = config.max_particles

    # Only free particles decay here (composites handled separately)
    is_free = particles.alive & (particles.composite_id < 0)

    # Draw uniform random for each particle
    rand = jax.random.uniform(subkey, (N,))

    # Decay probability
    ln2 = jnp.log(2.0)
    decay_prob = 1.0 - jnp.exp(-config.dt * ln2 / (particles.half_life + 1e-10))

    # Which particles decay this step?
    decays = is_free & (rand < decay_prob)

    # ── Kill decaying particles ──────────────────────────────────────────────
    new_alive = particles.alive & ~decays

    # ── Spawn products ────────────────────────────────────────────────────────
    # For each decaying particle, compute its hash and products
    # We process all N particles but mask with `decays`

    def compute_decay_products(i):
        h = jnp.array(
            (config.hash_prime_a * (particles.species[i].astype(jnp.int32) + 1)
             + config.hash_prime_b) % config.hash_modulus,
            dtype=jnp.uint32
        )
        # Sorted species multiset for single particle = just [species]
        species_sorted = jnp.array([-1] * config.max_composite_size, dtype=jnp.int32)
        species_sorted = species_sorted.at[0].set(particles.species[i])
        return _hash_to_decay_products(h, particles.species[i], config)

    # vmap over all particles — products for all, we'll mask later
    all_product_species, all_product_counts = jax.vmap(
        lambda i: compute_decay_products(i)
    )(jnp.arange(N, dtype=jnp.int32))
    # all_product_species: (N, max_decay_products) int32
    # all_product_counts:  (N,) int32

    # Find free slots for product spawning.
    # Each decaying particle can produce up to max_decay_products offspring.
    # We'll use find_free_slots on the updated alive array.
    # For simplicity in Phase 4, limit spawning to 1 product per decaying particle.
    # (Multi-product spawning is a Phase 5 enhancement.)
    n_decaying = jnp.sum(decays.astype(jnp.int32))

    # Find free slots — capped at max_decay_per_step to keep this O(N log N)
    max_spawns = config.max_decay_per_step
    free_slots = find_free_slots(new_alive, max_spawns)  # (max_spawns,)

    # Assign decaying particles to spawn slots sequentially
    decay_cumsum = jnp.cumsum(decays.astype(jnp.int32))  # 1-indexed

    # For each decaying particle i with ordinal k, spawn into free_slots[k-1]
    def spawn_into_slot(i):
        """Given decaying particle i, return (target_slot, new_species, new_pos, new_vel)"""
        k = decay_cumsum[i] - 1  # 0-indexed ordinal among decaying particles
        slot = jnp.where((k >= 0) & (k < max_spawns), free_slots[k], -1)
        slot = jnp.where(slot >= N, -1, slot)  # -1 if no free slot

        # Product species: first product from decay hash
        new_species = all_product_species[i, 0]

        # Position: near parent (small random scatter added in post)
        new_pos = particles.position[i]

        # Velocity: inherit parent's direction, share energy
        parent_speed = jnp.linalg.norm(particles.velocity[i]) + 1e-8
        # Give product a random direction but similar speed
        new_vel = particles.velocity[i]  # same velocity (products fly off together)

        return slot, new_species, new_pos, new_vel

    spawn_slots, spawn_species, spawn_pos, spawn_vel = jax.vmap(spawn_into_slot)(
        jnp.arange(N, dtype=jnp.int32)
    )
    # spawn_slots: (N,) int32 — destination slot for each particle (if it decays)

    # Apply spawns: update new_alive, species, position, velocity for spawned slots
    # We write into the arrays using .at[].set() with masking
    spawn_mask = decays & (spawn_slots >= 0)

    def update_array_at_slots(arr, values, slots, mask):
        """arr[slots[i]] = values[i] for all i where mask[i]"""
        safe_slots = jnp.where(mask, slots, 0)
        # Build updated array: scatter values into safe_slots
        # Use a loop-free approach: for each slot, check if any spawner targets it
        # This is simplified: we use .at[].set() which handles duplicate writes via last-wins
        return arr.at[safe_slots].set(
            jnp.where(mask, values, arr[safe_slots])
        )

    # Add random scatter to spawn positions
    key, k_scatter = jax.random.split(key)
    scatter = jax.random.normal(k_scatter, (N, 2)) * 0.1
    spawn_pos = spawn_pos + scatter

    new_alive       = update_array_at_slots(new_alive, jnp.ones(N, bool),    spawn_slots, spawn_mask)
    new_species     = update_array_at_slots(particles.species,  spawn_species, spawn_slots, spawn_mask)
    new_pos         = particles.position.copy()
    new_vel_arr     = particles.velocity.copy()
    # Scatter positions and velocities
    safe_slots = jnp.where(spawn_mask, spawn_slots, 0)
    new_pos = new_pos.at[safe_slots].set(
        jnp.where(spawn_mask[:, None], spawn_pos, new_pos[safe_slots])
    )
    new_vel_arr = new_vel_arr.at[safe_slots].set(
        jnp.where(spawn_mask[:, None], spawn_vel, new_vel_arr[safe_slots])
    )

    # Reset age and composite_id for new particles
    new_age = particles.age.at[safe_slots].set(
        jnp.where(spawn_mask, 0.0, particles.age[safe_slots])
    )
    new_cid = particles.composite_id.at[safe_slots].set(
        jnp.where(spawn_mask, -1, particles.composite_id[safe_slots])
    )
    # New half-life: sample from parent's half-life with some variation
    new_hl  = particles.half_life.at[safe_slots].set(
        jnp.where(spawn_mask, particles.half_life[safe_slots], particles.half_life[safe_slots])
    )

    new_particles = particles._replace(
        position=new_pos,
        velocity=new_vel_arr,
        species=new_species,
        alive=new_alive,
        age=new_age,
        composite_id=new_cid,
        half_life=new_hl,
    )

    return state._replace(particles=new_particles, rng_key=key)


# ── Composite Decay / Fission ─────────────────────────────────────────────────

def apply_composite_decay(state: WorldState, config: SimConfig) -> WorldState:
    """
    Apply decay to all alive composites (fission).

    A decaying composite:
      1. Marks itself dead
      2. Releases all member particles back to free state
      3. Injects (binding_energy - fission_cost * binding_energy) as kinetic energy,
         distributed as radial velocity among released members

    Args:
        state:  WorldState
        config: SimConfig (static)

    Returns:
        Updated WorldState
    """
    particles = state.particles
    composites = state.composites
    key, subkey = jax.random.split(state.rng_key)

    rand = jax.random.uniform(subkey, (config.max_composites,))
    ln2 = jnp.log(2.0)
    decay_prob = 1.0 - jnp.exp(
        -config.dt * ln2 / (composites.half_life + 1e-10)
    )

    fissions = composites.alive & (rand < decay_prob)  # (C,) bool

    # Mark decayed composites as dead
    new_comp_alive = composites.alive & ~fissions

    # For each composite that fissions, release its members
    # and give them radial outward velocities

    def release_members_from_composite(c):
        """Process composite c: if fissioning, free all its members."""
        does_fission = fissions[c]
        n_members = composites.member_count[c]
        energy_per_member = jnp.where(
            n_members > 0,
            composites.binding_energy[c] * (1.0 - config.fission_cost) / (n_members + 1e-8),
            0.0
        )

        def release_member(m_idx):
            """Release member at slot m_idx of composite c."""
            pid = composites.members[c, m_idx]
            valid = does_fission & (m_idx < n_members) & (pid >= 0)
            # Compute outward velocity: radial from center of mass
            member_ids = composites.members[c]  # (M,)
            valid_mask = (
                (member_ids >= 0) & (jnp.arange(config.max_composite_size) < n_members)
            ).astype(jnp.float32)  # (M,)
            safe_member_ids = jnp.where(member_ids >= 0, member_ids, 0)  # (M,)
            member_positions = particles.position[safe_member_ids]  # (M, 2)
            com = jnp.sum(member_positions * valid_mask[:, None], axis=0) / (n_members + 1e-8)
            d = particles.position[pid] - com
            d_norm = jnp.linalg.norm(d) + 1e-8
            d_hat = d / d_norm
            kick_speed = jnp.sqrt(jnp.maximum(0.0, 2.0 * energy_per_member))
            kick = d_hat * kick_speed
            return pid, valid, kick

        member_pids, member_valid, member_kicks = jax.vmap(release_member)(
            jnp.arange(config.max_composite_size, dtype=jnp.int32)
        )
        return member_pids, member_valid, member_kicks

    # vmap over composites
    all_pids, all_valid, all_kicks = jax.vmap(release_members_from_composite)(
        jnp.arange(config.max_composites, dtype=jnp.int32)
    )
    # all_pids:   (C, M) int32
    # all_valid:  (C, M) bool
    # all_kicks:  (C, M, 2) float32

    # Flatten and apply to particle arrays
    flat_pids  = all_pids.reshape(-1)   # (C*M,)
    flat_valid = all_valid.reshape(-1)  # (C*M,)
    flat_kicks = all_kicks.reshape(-1, 2)  # (C*M, 2)

    # Release particles: set composite_id to -1, add velocity kick
    safe_pids = jnp.where(flat_valid, flat_pids, 0)
    new_composite_id = particles.composite_id.at[safe_pids].set(
        jnp.where(flat_valid, -1, particles.composite_id[safe_pids])
    )
    new_velocity = particles.velocity.at[safe_pids].add(
        jnp.where(flat_valid[:, None], flat_kicks, 0.0)
    )

    new_particles = particles._replace(
        composite_id=new_composite_id,
        velocity=new_velocity,
    )
    new_composites = composites._replace(alive=new_comp_alive)

    return state._replace(
        particles=new_particles,
        composites=new_composites,
        rng_key=key,
    )


# ── Fusion ───────────────────────────────────────────────────────────────────

def _build_entity_species(pid: jnp.ndarray, particles, composites,
                           config: SimConfig) -> tuple:
    """
    Return the sorted species array and member count for the entity containing
    particle pid.  If free → single-element; if in composite → all members.

    Returns:
        species_buf: (max_composite_size,) int32, sorted, padded with -1
        count:       scalar int32
        rep:         representative particle index (members[c,0] or pid)
    """
    M = config.max_composite_size
    c = jnp.clip(particles.composite_id[pid], 0, config.max_composites - 1)
    is_free = particles.composite_id[pid] < 0

    # Representative = self if free, else first member of composite
    rep = jnp.where(is_free, pid, composites.members[c, 0])

    # Count
    count = jnp.where(is_free, jnp.int32(1), composites.member_count[c])

    # Species buffer: free → [species[pid], -1, ...]; composite → species of members
    free_buf = jnp.full(M, -1, dtype=jnp.int32).at[0].set(particles.species[pid])

    # For composite: gather species of each member slot
    safe_members = jnp.where(composites.members[c] >= 0, composites.members[c], 0)
    member_species = particles.species[safe_members]  # (M,)
    valid_mask = (composites.members[c] >= 0) & (jnp.arange(M) < composites.member_count[c])
    comp_buf = jnp.where(valid_mask, member_species, jnp.int32(-1))

    raw_buf = jnp.where(is_free, free_buf, comp_buf)
    # Sort: move -1 sentinels to end by sorting (valid species < sentinel if using large value)
    # Replace -1 with large value for sort, then restore
    sort_key = jnp.where(raw_buf < 0, jnp.int32(999999), raw_buf)
    sorted_buf = jnp.sort(sort_key)
    # Restore -1 for values that were sentinel
    final_buf = jnp.where(sorted_buf >= 999999, jnp.int32(-1), sorted_buf)

    return final_buf, count, rep


def attempt_fusion(state: WorldState, neighbors: jnp.ndarray,
                   params: InteractionParams, config: SimConfig) -> WorldState:
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
    particles = state.particles
    composites = state.composites
    key, subkey = jax.random.split(state.rng_key)
    N = config.max_particles
    M = config.max_composite_size

    fusion_r2 = config.fusion_radius ** 2

    # ── Step 1: Identify representatives ──────────────────────────────────────
    def get_rep(i):
        c = jnp.clip(particles.composite_id[i], 0, config.max_composites - 1)
        is_free = particles.composite_id[i] < 0
        return jnp.where(is_free, i, composites.members[c, 0])

    all_reps = jax.vmap(get_rep)(jnp.arange(N, dtype=jnp.int32))  # (N,)
    is_rep = (all_reps == jnp.arange(N)) & particles.alive  # (N,)

    # ── Step 2: For each representative, find its best fusion partner ──────────
    def find_entity_partner(i):
        """
        For representative particle i, scan its neighbors to find the best
        entity partner. Returns (partner_rep, be_eff, merged_h, merged_count).
        """
        i_is_rep = is_rep[i]

        sp_i, cnt_i, _ = _build_entity_species(i, particles, composites, config)
        c_i = jnp.clip(particles.composite_id[i], 0, config.max_composites - 1)

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

            # Build merged species multiset (up to 2*M, clamped to M)
            sp_j, cnt_j, _ = _build_entity_species(j, particles, composites, config)
            merged_count = jnp.minimum(cnt_i + cnt_j, M)

            # Concatenate and sort into one buffer of size M
            # Use a fixed-size trick: place sp_i in [0..M/2) and sp_j in [M/2..M)
            half = M // 2
            merged_raw = jnp.concatenate([sp_i[:half], sp_j[:half]])  # (M,)
            sort_key = jnp.where(merged_raw < 0, jnp.int32(999999), merged_raw)
            merged_sorted = jnp.sort(sort_key)
            merged_buf = jnp.where(merged_sorted >= 999999, jnp.int32(-1), merged_sorted)

            h = _hash_sorted_species(merged_buf, merged_count, config)
            be = _hash_to_binding_energy(h, config)

            # Polarity bonus
            c_j = jnp.clip(particles.composite_id[j], 0, config.max_composites - 1)
            pi = jnp.where(
                particles.composite_id[i] < 0,
                params.polarity[particles.species[i]],
                composites.net_polarity[c_i]
            )
            pj = jnp.where(
                particles.composite_id[j] < 0,
                params.polarity[particles.species[j]],
                composites.net_polarity[c_j]
            )
            be_eff = be + config.polarity_fusion_scale * (-pi * pj)

            # Size cap: don't grow beyond buffer
            would_overflow = (cnt_i + cnt_j) > M

            can_fuse = valid & in_range & (be_eff > config.fusion_threshold) & ~would_overflow
            return (
                jnp.where(can_fuse, j, jnp.int32(-1)),
                jnp.where(can_fuse, be_eff, jnp.float32(0.0)),
                jnp.where(can_fuse, h, jnp.uint32(0)),
                jnp.where(can_fuse, merged_count, jnp.int32(0)),
                jnp.where(can_fuse, merged_buf, jnp.full(M, -1, jnp.int32)),
            )

        # vmap over neighbors
        nbrs = neighbors[i]
        results = jax.vmap(check_neighbor)(nbrs)
        partners, bes, hs, mcounts, mbufs = results
        # (max_neighbors,), (max_neighbors,), ...

        best_idx    = jnp.argmax(bes)
        best_j      = partners[best_idx]
        best_be     = bes[best_idx]
        best_h      = hs[best_idx]
        best_mc     = mcounts[best_idx]
        best_mbuf   = mbufs[best_idx]

        # Gate on i being a representative
        final_j    = jnp.where(i_is_rep, best_j,    jnp.int32(-1))
        final_be   = jnp.where(i_is_rep, best_be,   jnp.float32(0.0))
        final_h    = jnp.where(i_is_rep, best_h,    jnp.uint32(0))
        final_mc   = jnp.where(i_is_rep, best_mc,   jnp.int32(0))
        final_mbuf = jnp.where(i_is_rep, best_mbuf, jnp.full(M, -1, jnp.int32))

        return final_j, final_be, final_h, final_mc, final_mbuf

    all_partners, all_be, all_hashes, all_merged_counts, all_merged_bufs = jax.vmap(
        find_entity_partner
    )(jnp.arange(N, dtype=jnp.int32))
    # (N,), (N,), (N,), (N,), (N, M)

    # ── Step 3: Conflict resolution via sequential scan ────────────────────────
    max_fusions = config.max_fusions_per_step
    has_partner = all_partners >= 0
    cumsum_p    = jnp.cumsum(has_partner.astype(jnp.int32))
    candidate_i = jnp.where(
        has_partner & (cumsum_p <= max_fusions),
        jnp.arange(N, dtype=jnp.int32),
        N,
    )
    scan_indices = jnp.sort(candidate_i)[:max_fusions]

    def fusion_scan_body(carry, i):
        claimed, new_composite_id, composites_state, comp_count = carry

        valid_i = i < N
        safe_i  = jnp.minimum(i, N - 1)

        j        = jnp.where(valid_i, all_partners[safe_i],      jnp.int32(-1))
        be       = jnp.where(valid_i, all_be[safe_i],            jnp.float32(0.0))
        h        = jnp.where(valid_i, all_hashes[safe_i],        jnp.uint32(0))
        mc       = jnp.where(valid_i, all_merged_counts[safe_i], jnp.int32(0))
        mbuf     = jnp.where(valid_i, all_merged_bufs[safe_i],
                             jnp.full(M, -1, dtype=jnp.int32))

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

        # Target composite slot:
        #   free+free  → new free slot
        #   comp+free  → ci (i's composite grows)
        #   free+comp  → cj (j's composite grows)
        #   comp+comp  → lower index absorbs higher
        free_comp_slot = jnp.argmin(
            composites_state.alive.astype(jnp.int32) * config.max_composites
            + jnp.arange(config.max_composites)
        )
        target = jax.lax.switch(
            (i_is_free.astype(jnp.int32) * 2 + j_is_free.astype(jnp.int32)),
            [
                # case 0: comp+comp → lower index absorbs
                lambda: jnp.minimum(ci, cj),
                # case 1: comp+free → ci
                lambda: ci,
                # case 2: free+comp → cj
                lambda: cj,
                # case 3: free+free → new slot
                lambda: free_comp_slot,
            ]
        )
        absorbed = jax.lax.switch(
            (i_is_free.astype(jnp.int32) * 2 + j_is_free.astype(jnp.int32)),
            [
                lambda: jnp.maximum(ci, cj),  # comp+comp: higher index killed
                lambda: jnp.int32(-1),          # comp+free: no composite killed
                lambda: jnp.int32(-1),          # free+comp: no composite killed
                lambda: jnp.int32(-1),          # free+free: no composite killed
            ]
        )

        # Derive composite properties from merged hash
        hl = _hash_to_half_life(h, config)

        # Mean polarity of merged entity
        pi = jnp.where(i_is_free,
                        params.polarity[particles.species[safe_i]],
                        composites_state.net_polarity[ci])
        pj = jnp.where(j_is_free,
                        params.polarity[particles.species[safe_j]],
                        composites_state.net_polarity[cj])
        cnt_i_scalar = jnp.where(i_is_free, jnp.int32(1), composites_state.member_count[ci])
        cnt_j_scalar = jnp.where(j_is_free, jnp.int32(1), composites_state.member_count[cj])
        net_pol = (pi * cnt_i_scalar.astype(jnp.float32) +
                   pj * cnt_j_scalar.astype(jnp.float32)) / (mc.astype(jnp.float32) + 1e-8)
        neutrality = 1.0 - jnp.abs(net_pol)
        hl_eff = hl * (1.0 + config.polarity_stability_scale * neutrality)

        # Build the merged member list: gather all member particle indices
        # i-side members
        i_members_comp = composites_state.members[ci]  # (M,)
        i_members_free = jnp.full(M, -1, dtype=jnp.int32).at[0].set(safe_i)
        i_members = jnp.where(i_is_free, i_members_free, i_members_comp)

        # j-side members
        j_members_comp = composites_state.members[cj]  # (M,)
        j_members_free = jnp.full(M, -1, dtype=jnp.int32).at[0].set(safe_j)
        j_members = jnp.where(j_is_free, j_members_free, j_members_comp)

        # Concat first half of each into a (M,) buffer
        half = M // 2
        merged_members = jnp.concatenate([i_members[:half], j_members[:half]])  # (M,)
        # Replace -1 sentinels beyond mc with -1 (already set), ensure count is mc
        # (The concat gives us up to M valid slots; any excess are -1 already)

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
            jnp.where(can_fuse, be, composites_state.binding_energy[safe_target])
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
        new_comp_net_pol = composites_state.net_polarity.at[safe_target].set(
            jnp.where(can_fuse, net_pol, composites_state.net_polarity[safe_target])
        )

        new_composites = composites_state._replace(
            members=new_members,
            alive=new_comp_alive,
            binding_energy=new_comp_be,
            half_life=new_comp_hl,
            member_count=new_comp_count_arr,
            species_hash=new_comp_hash,
            net_polarity=new_comp_net_pol,
        )

        # Update composite_id for all merged members
        # i-side: assign to target
        def assign_i_member(m_idx):
            pid = jnp.where(i_is_free, safe_i, i_members[m_idx])
            valid = can_fuse & (pid >= 0) & (
                jnp.where(i_is_free, m_idx == 0, m_idx < composites_state.member_count[ci])
            )
            return pid, valid

        i_pids, i_valid = jax.vmap(assign_i_member)(jnp.arange(half, dtype=jnp.int32))
        safe_i_pids = jnp.where(i_valid, i_pids, 0)
        new_composite_id = new_composite_id.at[safe_i_pids].set(
            jnp.where(i_valid, target, new_composite_id[safe_i_pids])
        )

        # j-side: assign to target
        def assign_j_member(m_idx):
            pid = jnp.where(j_is_free, safe_j, j_members[m_idx])
            valid = can_fuse & (pid >= 0) & (
                jnp.where(j_is_free, m_idx == 0, m_idx < composites_state.member_count[cj])
            )
            return pid, valid

        j_pids, j_valid = jax.vmap(assign_j_member)(jnp.arange(half, dtype=jnp.int32))
        safe_j_pids = jnp.where(j_valid, j_pids, 0)
        new_composite_id = new_composite_id.at[safe_j_pids].set(
            jnp.where(j_valid, target, new_composite_id[safe_j_pids])
        )

        # Mark both representatives as claimed
        new_claimed = claimed.at[safe_i].set(claimed[safe_i] | can_fuse)
        new_claimed = new_claimed.at[safe_j].set(new_claimed[safe_j] | can_fuse)

        # Only increment comp_count for free+free (new composite created)
        new_comp_count = comp_count + jnp.where(
            can_fuse & i_is_free & j_is_free, jnp.int32(1), jnp.int32(0)
        )

        return (new_claimed, new_composite_id, new_composites, new_comp_count), None

    claimed_init      = jnp.zeros(N, dtype=bool)
    composite_id_init = particles.composite_id
    comp_count_init   = jnp.sum(composites.alive.astype(jnp.int32))

    (_, final_composite_id, final_composites, _), _ = jax.lax.scan(
        fusion_scan_body,
        (claimed_init, composite_id_init, composites, comp_count_init),
        scan_indices,
    )

    new_particles = particles._replace(composite_id=final_composite_id)

    return state._replace(
        particles=new_particles,
        composites=final_composites,
        rng_key=key,
    )
