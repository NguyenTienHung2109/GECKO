"""FedFST algorithm package.

The package owns both client HHKR mechanics and the federated/server HLST
strategy.  UEFA registry and runtime modules contain only thin integration
points so future method work remains localized here.
"""

from gecko.algorithms.federated.fedfst.core import FEDFST_IMPLEMENTATION_VERSION
from gecko.algorithms.federated.fedfst.core import PAPER_DOI
from gecko.algorithms.federated.fedfst.core import RAYLEIGH_DENOMINATOR_EPSILON
from gecko.algorithms.federated.fedfst.core import REFERENCE_CODE_COMMIT
from gecko.algorithms.federated.fedfst.core import ConditionalFeatureGenerator
from gecko.algorithms.federated.fedfst.core import FedFSTParameters
from gecko.algorithms.federated.fedfst.core import SpectralEnergy
from gecko.algorithms.federated.fedfst.core import TopologyAdjustment
from gecko.algorithms.federated.fedfst.core import adjust_homophily
from gecko.algorithms.federated.fedfst.core import adjust_spectral_energy
from gecko.algorithms.federated.fedfst.core import edgewise_low_frequency_kl
from gecko.algorithms.federated.fedfst.core import graph_homophily
from gecko.algorithms.federated.fedfst.core import hhkr_loss
from gecko.algorithms.federated.fedfst.core import high_frequency_energy
from gecko.algorithms.federated.fedfst.core import hlst_loss
from gecko.algorithms.federated.fedfst.core import smooth_probabilities
from gecko.algorithms.federated.fedfst.core import transition_matrix
from gecko.algorithms.federated.fedfst.core import weighted_generator_average
from gecko.algorithms.federated.fedfst.core import weighted_spectral_target
from gecko.algorithms.federated.fedfst.strategy import FedFSTStrategy

__all__ = [
    "FEDFST_IMPLEMENTATION_VERSION",
    "PAPER_DOI",
    "RAYLEIGH_DENOMINATOR_EPSILON",
    "REFERENCE_CODE_COMMIT",
    "ConditionalFeatureGenerator",
    "FedFSTParameters",
    "FedFSTStrategy",
    "SpectralEnergy",
    "TopologyAdjustment",
    "adjust_homophily",
    "adjust_spectral_energy",
    "edgewise_low_frequency_kl",
    "graph_homophily",
    "hhkr_loss",
    "high_frequency_energy",
    "hlst_loss",
    "smooth_probabilities",
    "transition_matrix",
    "weighted_generator_average",
    "weighted_spectral_target",
]
