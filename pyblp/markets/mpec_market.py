"""Market-level MPEC equilibrium constraint functionality."""

from typing import List, Tuple

import numpy as np

from .market import Market
from .. import exceptions, options
from ..utilities.basics import Array, Error, NumericalErrorHandler


class MPECMarket(Market):
    """A market underlying the MPEC formulation of the BLP problem.

    Unlike :class:`ProblemMarket`, this market never solves the contraction mapping that inverts market shares for
    delta. Instead, delta is treated as a free variable, and the market share equations are evaluated (and
    differentiated) at whatever delta the outer NLP solver currently proposes -- this is exactly what MPEC needs to
    impose them as equilibrium constraints instead of solving them every objective evaluation.
    """

    @NumericalErrorHandler(exceptions.MPECConstraintNumericalError)
    def solve_constraint(self, delta: Array, compute_jacobians: bool) -> Tuple[Array, Array, Array, List[Error]]:
        """Compute simulated shares at a free delta (the equilibrium constraint residual is these shares minus
        observed shares, computed by the caller). If compute_jacobians is True, also compute the Jacobian of shares
        with respect to delta (equivalently, xi) and with respect to theta, both holding beta fixed. Neither Jacobian
        accounts for any implicit dependence of delta on theta, since under MPEC, delta is a free variable and has no
        such dependence.
        """
        errors: List[Error] = []
        probabilities, conditionals = self.compute_probabilities(delta)
        shares = probabilities @ self.agents.weights

        shares_by_delta_jacobian = np.zeros((self.J, self.J), options.dtype)
        shares_by_theta_jacobian = np.zeros((self.J, self.parameters.P), options.dtype)
        if compute_jacobians:
            shares_by_delta_jacobian = self.compute_shares_by_xi_jacobian(probabilities, conditionals)
            probabilities_tangent_mapping, _ = self.compute_probabilities_by_parameter_tangent_mapping(
                probabilities, conditionals, delta, keep_conditionals=False,
            )
            shares_by_theta_jacobian = self.compute_shares_by_theta_jacobian(probabilities_tangent_mapping)

        return shares, shares_by_delta_jacobian, shares_by_theta_jacobian, errors
