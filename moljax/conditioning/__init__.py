"""Matrix-free conditioning diagnostics."""

from .field_of_values import FieldOfValuesResult, numerical_range
from .figures import (
    plot_numerical_range,
    plot_pseudospectrum,
    plot_rate_scaling,
    plot_residual_envelope,
)
from .linearization import LinearizedOperator, adjoint_identity, linearized_operator
from .non_normality import (
    PreconditionerAssessment,
    RateEstimates,
    assess_preconditioner,
    clustering_rate,
    crouzeix_palencia_envelope,
    enclosing_disk_rate,
    estimate_rates,
    right_real_outliers,
    traced_boundary_rate,
)
from .pseudospectra import (
    PseudospectraResult,
    arnoldi,
    epsilon_zero,
    pseudospectrum_dense,
    reduced_pseudospectrum,
    ritz_values,
)

__all__ = [
    "FieldOfValuesResult",
    "LinearizedOperator",
    "PseudospectraResult",
    "PreconditionerAssessment",
    "RateEstimates",
    "arnoldi",
    "adjoint_identity",
    "assess_preconditioner",
    "clustering_rate",
    "crouzeix_palencia_envelope",
    "epsilon_zero",
    "enclosing_disk_rate",
    "estimate_rates",
    "numerical_range",
    "plot_numerical_range",
    "plot_pseudospectrum",
    "plot_rate_scaling",
    "plot_residual_envelope",
    "linearized_operator",
    "pseudospectrum_dense",
    "reduced_pseudospectrum",
    "right_real_outliers",
    "ritz_values",
    "traced_boundary_rate",
]
