"""MarkovMPS: tensor-train propagators for classical Markov generators.

The package computes first-passage statistics and Manna sandpile avalanche
distributions by representing a classical probability distribution as a
tensor train, selecting the solver each observable needs, and checking
every tensor-train number against 2 independent baselines: an exact dense
engine on small systems and kinetic Monte Carlo at every size. This Python
implementation and the Julia implementation beside it hold the same
structure file for file; the committed problem instances under problems/
are shared between them and define the cross-validation contract.
"""

from .generator import escape_rates, markov_generator, offdiagonal_operator
from .models.firstpassage import BiasedWalk
from .models.manna import MannaSandpile
from .solvers import (
    exact_absorption_probability,
    exact_avalanche_pgf,
    exact_avalanche_size_distribution,
    exact_generator,
    exact_mfpt,
    exact_survival,
    kmc_avalanche_sizes,
    kmc_mfpt,
    kmc_survival,
    kmc_trajectory,
    max_abs_deviation,
    tt_eigen_max,
    tt_evolve,
    tt_linear_solve,
)
from .tensor_train import TT, TTOperator, mpo_sum, qtt_shift, tt_sum
from .utils import ProblemInstance, load_problem, problem_paths

__version__ = "0.1.0"
