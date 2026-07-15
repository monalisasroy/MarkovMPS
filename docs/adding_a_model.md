# Adding a model

A new Markov problem is a new struct in one file under `src/models/`, and nothing else changes: the generator assembly, the solvers, and the baselines are model-blind. This page states the interface a model implements and walks the shortest path from an idea to a cross-checked observable, using the two models in the package as worked precedent. The interface is small because the exact engine and Monte Carlo consume the state-space half while the tensor-train path consumes the operator half, and the twin check between them is what certifies a new model's construction.

## Contents

- [The interface](#the-interface)
- [The state-space half](#the-state-space-half)
- [The operator half](#the-operator-half)
- [The twin check](#the-twin-check)
- [Wiring a problem kind](#wiring-a-problem-kind)

## The interface

A model is a struct whose constructor validates every parameter and rejects out-of-range input with a message naming the offender. It then implements two halves of one process definition.

The state-space half, consumed by the exact engine and kinetic Monte Carlo:

    local_dims(m)          # local dimension per site
    state_count(m)         # how many states enumerate_states would return, without allocating them
    enumerate_states(m)    # every reachable configuration (small systems)
    is_absorbing(m, s)     # does the dynamics stop at s
    moves(m, s)            # outgoing (rate, destination) pairs from s
    initial_state(m)       # the starting configuration

The operator half, consumed by the generator assembly and the tensor-train solvers:

    terms(m)               # off-diagonal transitions: (rate, site => matrix) or (rate, operator)
    initial_tt(m)          # the starting distribution as a rank-1 tensor train

plus whatever projectors and observables the model's physics needs; the walk carries transient and absorbing projectors and four observables, the sandpile carries an activity vector, 2 projectors, and four observables.

## The state-space half

Write `moves()` first, without matrices: it is the definition of the process, and everything else is checked against it. Include every rule of the dynamics here, including blocked branches at boundaries or caps, because the escape rates of the exact engine and the trajectories of Monte Carlo both come from this one function. Keep `enumerate_states()` restricted to the reachable sector when a conserved quantity exists, as the sandpile does with its fixed grain number; the dense engine then scales with the physical sector, not the product space. Implement `state_count()` by counting, never by enumerating, because the dense-limit guard compares it before any allocation; the test suite pins the counter against the enumeration on a small instance, and the sandpile's counter uses arbitrary-precision integers so an oversized model is refused with the true count rather than a silently wrapped one.

## The operator half

Write `terms()` second, as a list of (rate, move) pairs. A site-local move is a mapping from site indices to local matrices and becomes a rank-1 product operator; a move that is not a site-local product, such as the carry-chain position shifts of the walk, is supplied as a prebuilt operator. The escape diagonal is never written by a model: `markov_generator` computes it from the off-diagonal part by applying the transpose to the all-ones vector, so a blocked branch is excluded automatically and the columns sum to 0 by construction. The one rule that matters is that `terms()` and `moves()` must encode the same process, including the same blocking; the twin check exists to catch any divergence.

## The twin check

Before any observable, densify the assembled generator on a small instance and compare it entry by entry against the dense generator built from `moves()`; then check the column sums and, where a conserved quantity exists, the sector identities. The committed tests in `test/generator.jl` (and `Python/tests/test_generator.py`) are the template: copy one, point it at the new model, and keep it in the suite permanently. A model whose two halves agree on a dense instance can then be trusted where only the tensor-train half reaches, up to truncation error, which the exact engine bounds separately on small systems.

## Wiring a problem kind

To make the model runnable from the entry points, add its parameters to the allowed field set in `utils`, add a kind name and a branch in `build_model`, and add one function in `plot_utils` that writes the model's table and figure. Commit one TOML instance under `problems/` at the repository root, with an anchor stated in its comment: a closed form where one exists, the exact engine where one does not, and Monte Carlo always. A model enters the repository only when its instance runs from a fresh clone and its twin check passes.
