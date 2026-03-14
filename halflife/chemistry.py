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

from halflife.state import ParticleState, CompositeState, WorldState
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

    # Find slots for new particles
    free_slots = find_free_slots(new_alive, N)  # (N,) indices, -1 = no free slot

    # Assign decaying particles to spawn slots sequentially
    # decay_indices: which particles are decaying (as sequential list)
    decay_cumsum = jnp.cumsum(decays.astype(jnp.int32))  # 1-indexed

    # For each decaying particle i with ordinal k, spawn into free_slots[k-1]
    def spawn_into_slot(i):
        """Given decaying particle i, return (target_slot, new_species, new_pos, new_vel)"""
        k = decay_cumsum[i] - 1  # 0-indexed ordinal among decaying particles
        slot = jnp.where((k >= 0) & (k < N), free_slots[k], -1)
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

def attempt_fusion(state: WorldState, neighbors: jnp.ndarray,
                   config: SimConfig) -> WorldState:
    """
    Try to fuse pairs of nearby free particles into composites.

    For each particle i:
      - Check neighbors within fusion_radius
      - Both must be free (composite_id == -1)
      - Compute hash of sorted [species_i, species_j]
      - If hash-derived binding_energy > fusion_threshold: fuse

    Conflict resolution: scan particles in ascending index order.
    If particle i claims particle j, j is marked as "claimed" and cannot
    be claimed again by a later particle.

    Args:
        state:     WorldState
        neighbors: (N, max_neighbors) int32
        config:    SimConfig (static)

    Returns:
        Updated WorldState with new composites formed
    """
    particles = state.particles
    composites = state.composites
    key, subkey = jax.random.split(state.rng_key)
    N = config.max_particles

    fusion_r2 = config.fusion_radius ** 2

    # --- Find fusion candidates ---
    # For each particle i, find its best fusion partner (lowest index valid neighbor)
    def find_fusion_partner(i):
        """
        Returns (partner_j, binding_energy) for particle i's best fusion candidate.
        Returns (-1, 0.0) if no valid partner found.
        """
        is_free_i = particles.alive[i] & (particles.composite_id[i] < 0)

        def check_neighbor(j):
            valid = (
                (j >= 0) & (j > i) &  # only consider j > i to avoid double-processing
                particles.alive[j] &
                (particles.composite_id[j] < 0)
            )
            # Distance check
            d = particles.position[i] - particles.position[j]
            if config.boundary_mode == "periodic":
                d = d - config.world_width  * jnp.round(d[0] / config.world_width) * jnp.array([1., 0.])
                d = d - config.world_height * jnp.round(d[1] / config.world_height) * jnp.array([0., 1.])
            dist2 = jnp.dot(d, d)
            in_range = dist2 < fusion_r2

            # Hash of sorted species pair
            si = particles.species[i]
            sj = particles.species[j]
            s_lo = jnp.minimum(si, sj)
            s_hi = jnp.maximum(si, sj)
            species_sorted = jnp.array([-1] * config.max_composite_size, dtype=jnp.int32)
            species_sorted = species_sorted.at[0].set(s_lo)
            species_sorted = species_sorted.at[1].set(s_hi)
            h = _hash_sorted_species(species_sorted, jnp.array(2), config)
            be = _hash_to_binding_energy(h, config)

            can_fuse = valid & in_range & (be > config.fusion_threshold)
            return jnp.where(can_fuse, j, -1), jnp.where(can_fuse, be, 0.0), h

        results = jax.vmap(check_neighbor)(neighbors[i])
        partner_candidates = results[0]  # (max_neighbors,)
        be_candidates      = results[1]  # (max_neighbors,)
        h_candidates       = results[2]  # (max_neighbors,)

        # Pick the partner with highest binding energy
        best_idx = jnp.argmax(be_candidates)
        best_partner = partner_candidates[best_idx]
        best_be      = be_candidates[best_idx]
        best_h       = h_candidates[best_idx]

        # No partner if particle is not free
        final_partner = jnp.where(is_free_i, best_partner, -1)
        final_be      = jnp.where(is_free_i, best_be, 0.0)
        final_h       = jnp.where(is_free_i, best_h, jnp.uint32(0))

        return final_partner, final_be, final_h

    # vmap to get fusion candidates for all particles
    all_partners, all_be, all_hashes = jax.vmap(find_fusion_partner)(
        jnp.arange(N, dtype=jnp.int32)
    )
    # all_partners: (N,) int32 — proposed partner for each particle (-1 = none)

    # --- Conflict resolution via sequential scan ---
    # Scan in ascending particle index order.
    # Maintain a "claimed" array: once particle j is claimed, it can't be fused again.

    def fusion_scan_body(carry, i):
        claimed, new_composite_id, composites_state, comp_count = carry

        partner = all_partners[i]
        be      = all_be[i]
        h       = all_hashes[i]

        can_fuse = (
            (partner >= 0) &
            ~claimed[i] &
            ~claimed[partner] &
            (comp_count < config.max_composites)
        )

        # If can_fuse, allocate a new composite slot
        # Find first free composite slot
        free_comp_slot = jnp.argmin(composites_state.alive.astype(jnp.int32) +
                                     jnp.arange(config.max_composites))
        # More robust: use the cumulative count
        comp_slot = jnp.where(can_fuse, comp_count, -1)

        # Build species array for the new composite
        si = particles.species[i]
        sj = jnp.where(partner >= 0, particles.species[partner], si)
        s_lo = jnp.minimum(si, sj)
        s_hi = jnp.maximum(si, sj)
        species_sorted = jnp.full(config.max_composite_size, -1, dtype=jnp.int32)
        species_sorted = species_sorted.at[0].set(s_lo)
        species_sorted = species_sorted.at[1].set(s_hi)
        hl = _hash_to_half_life(h, config)

        # Update composite arrays
        new_members = composites_state.members.at[comp_slot].set(
            jnp.where(can_fuse,
                       jnp.array([i, partner] + [-1] * (config.max_composite_size - 2), dtype=jnp.int32),
                       composites_state.members[comp_slot])
        )
        new_comp_alive = composites_state.alive.at[comp_slot].set(
            jnp.where(can_fuse, True, composites_state.alive[comp_slot])
        )
        new_comp_be = composites_state.binding_energy.at[comp_slot].set(
            jnp.where(can_fuse, be, composites_state.binding_energy[comp_slot])
        )
        new_comp_hl = composites_state.half_life.at[comp_slot].set(
            jnp.where(can_fuse, hl, composites_state.half_life[comp_slot])
        )
        new_comp_count_arr = composites_state.member_count.at[comp_slot].set(
            jnp.where(can_fuse, 2, composites_state.member_count[comp_slot])
        )
        new_comp_hash = composites_state.species_hash.at[comp_slot].set(
            jnp.where(can_fuse, h, composites_state.species_hash[comp_slot])
        )

        new_composites = composites_state._replace(
            members=new_members,
            alive=new_comp_alive,
            binding_energy=new_comp_be,
            half_life=new_comp_hl,
            member_count=new_comp_count_arr,
            species_hash=new_comp_hash,
        )

        # Update particle composite_id
        new_composite_id = new_composite_id.at[i].set(
            jnp.where(can_fuse, comp_slot, new_composite_id[i])
        )
        new_composite_id = new_composite_id.at[
            jnp.where(can_fuse, partner, 0)
        ].set(jnp.where(can_fuse, comp_slot, new_composite_id[jnp.where(can_fuse, partner, 0)]))

        # Mark both as claimed
        new_claimed = claimed.at[i].set(claimed[i] | can_fuse)
        new_claimed = new_claimed.at[jnp.where(can_fuse, partner, 0)].set(
            new_claimed[jnp.where(can_fuse, partner, 0)] | can_fuse
        )

        new_comp_count = comp_count + jnp.where(can_fuse, 1, 0)

        return (new_claimed, new_composite_id, new_composites, new_comp_count), None

    claimed_init = jnp.zeros(N, dtype=bool)
    composite_id_init = particles.composite_id
    comp_count_init = jnp.sum(composites.alive.astype(jnp.int32))

    (final_claimed, final_composite_id, final_composites, _), _ = jax.lax.scan(
        fusion_scan_body,
        (claimed_init, composite_id_init, composites, comp_count_init),
        jnp.arange(N, dtype=jnp.int32)
    )

    new_particles = particles._replace(composite_id=final_composite_id)

    return state._replace(
        particles=new_particles,
        composites=final_composites,
        rng_key=key,
    )
