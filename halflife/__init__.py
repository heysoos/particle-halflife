"""
Half-Life Particle Simulator
=============================
A JAX-based particle simulator where everything decays.

Particles interact via species-dependent force kernels, fuse into composites
with hash-determined properties, and constantly face probabilistic decay/fission.
Emergent autocatalytic sets, self-maintaining organizations, and evolutionary
dynamics arise from this simple premise.

Modules (import order / dependency order):
  config        - SimConfig frozen dataclass, all tunable parameters
  state         - ParticleState, CompositeState, WorldState NamedTuples
  utils         - JAX helpers: hashing, free-slot finding, boundaries
  spatial       - Cell-list spatial indexing and neighbor queries
  interactions  - Pairwise force kernels (Particle Life-style)
  chemistry     - Hash-based fusion, fission, decay
  energy        - Energy accounting and conservation
  step          - Main simulation_step() JIT-compiled orchestrator
  renderer      - ModernGL + pygame real-time visualization
  main          - Entry point and event loop
"""
