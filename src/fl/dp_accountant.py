"""Differential Privacy Accountant — Formal (ε, δ) privacy budget tracking.

Implements Rényi Differential Privacy (RDP) accounting for federated learning.
Tracks cumulative privacy expenditure across rounds and provides certifiable
(ε, δ)-DP guarantees.

Based on:
- Mironov (2017): "Rényi Differential Privacy"
- Abadi et al. (2016): "Deep Learning with Differential Privacy"
- Wang et al. (2019): "Subsampled Rényi Differential Privacy"

Usage:
    accountant = RDPAccountant(noise_multiplier=1.0, sample_rate=0.1)
    for round in range(20):
        accountant.step()
    eps = accountant.get_epsilon(delta=1e-5)
    print(f"Total privacy cost: eps={eps:.2f} at delta=1e-5")
"""

import math
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

import numpy as np
from scipy import special

logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudget:
    """Represents a consumed privacy budget."""
    epsilon: float
    delta: float
    rounds: int
    noise_multiplier: float
    sample_rate: float
    mechanism: str = "gaussian"

    def summary(self) -> str:
        return (
            f"Privacy Budget: eps={self.epsilon:.4f}, delta={self.delta}, "
            f"sigma={self.noise_multiplier}, q={self.sample_rate}, "
            f"rounds={self.rounds}"
        )


class RDPAccountant:
    """Rényi Differential Privacy accountant for federated learning.

    Uses Rényi DP composition which provides tighter bounds than
    standard (ε, δ)-DP composition, especially for many rounds.

    The accountant tracks the noise multiplier, sampling rate, and
    number of rounds to compute the final (ε, δ) guarantee.

    Args:
        noise_multiplier: sigma = noise_multiplier * max_norm.
            Controls the amount of Gaussian noise added per round.
        sample_rate: q = clients_per_round / total_clients.
            The probability that any given client participates.
        delta: Target delta for (ε, δ)-DP conversion.
        orders: RDP orders to track (default: 1.25 to 256).
    """

    def __init__(
        self,
        noise_multiplier: float = 0.0,
        sample_rate: float = 1.0,
        delta: float = 1e-5,
        orders: Optional[List[float]] = None,
    ):
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.delta = delta
        self.orders = orders or list(
            np.concatenate([
                np.linspace(1.25, 2.0, 9),
                np.linspace(2.0, 4.0, 20),
                np.linspace(4.0, 8.0, 45),
                np.linspace(8.0, 32.0, 60),
                np.linspace(32.0, 256.0, 20),
            ])
        )
        self.orders = [float(o) for o in self.orders if o > 1.0]

        self._num_steps = 0
        self._rdp: Dict[float, float] = {alpha: 0.0 for alpha in self.orders}
        self._history: List[Dict[str, Any]] = []

    def step(
        self,
        noise_multiplier: Optional[float] = None,
        sample_rate: Optional[float] = None,
    ):
        """Record one step of the DP mechanism."""
        sigma = noise_multiplier if noise_multiplier is not None else self.noise_multiplier
        q = sample_rate if sample_rate is not None else self.sample_rate

        if sigma <= 0:
            self._num_steps += 1
            self._history.append({
                "step": self._num_steps,
                "sigma": sigma,
                "q": q,
                "epsilon": None,
                "note": "No noise added",
            })
            return

        for alpha in self.orders:
            rdp_step = self._compute_rdp(q, sigma, alpha)
            self._rdp[alpha] += rdp_step

        self._num_steps += 1
        current_eps = self._get_epsilon_from_rdp(self.delta)

        self._history.append({
            "step": self._num_steps,
            "sigma": sigma,
            "q": q,
            "epsilon": current_eps,
            "delta": self.delta,
        })

        logger.debug(
            "DP step %d: sigma=%.3f, q=%.3f, eps(%.0e)=%.4f",
            self._num_steps, sigma, q, self.delta, current_eps
        )

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """Compute the total eps consumed at a given delta."""
        target_delta = delta if delta is not None else self.delta
        return self._get_epsilon_from_rdp(target_delta)

    def get_privacy_budget(self, delta: Optional[float] = None) -> PrivacyBudget:
        """Get the complete privacy budget summary."""
        target_delta = delta if delta is not None else self.delta
        epsilon = self._get_epsilon_from_rdp(target_delta)

        return PrivacyBudget(
            epsilon=epsilon,
            delta=target_delta,
            rounds=self._num_steps,
            noise_multiplier=self.noise_multiplier,
            sample_rate=self.sample_rate,
        )

    def get_remaining_budget(
        self,
        max_epsilon: float,
        delta: Optional[float] = None,
        noise_multiplier: Optional[float] = None,
        sample_rate: Optional[float] = None,
    ) -> int:
        """Estimate how many more rounds can be run before exceeding max eps."""
        target_delta = delta if delta is not None else self.delta
        sigma = noise_multiplier if noise_multiplier is not None else self.noise_multiplier
        q = sample_rate if sample_rate is not None else self.sample_rate

        if sigma <= 0:
            return 0

        low, high = 0, max(10000, self._num_steps * 10)
        while low < high:
            mid = (low + high + 1) // 2
            test_eps = self._estimate_epsilon_for_steps(
                self._num_steps + mid, sigma, q, target_delta
            )
            if test_eps <= max_epsilon:
                low = mid
            else:
                high = mid - 1

        return low

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the full history of privacy accounting."""
        return list(self._history)

    def reset(self):
        """Reset the accountant to initial state."""
        self._num_steps = 0
        self._rdp = {alpha: 0.0 for alpha in self.orders}
        self._history = []

    def _compute_rdp(self, q: float, sigma: float, alpha: float) -> float:
        """Compute the RDP at order alpha for one step of subsampled Gaussian mechanism."""
        if q == 0:
            return 0.0

        if q == 1.0:
            return self._gaussian_rdp(sigma, alpha)

        alpha = float(alpha)

        rdp = self._compute_log_a_for_finite_sample(q, sigma, alpha)

        return rdp

    def _gaussian_rdp(self, sigma: float, alpha: float) -> float:
        """RDP of the Gaussian mechanism (no subsampling).

        For Gaussian noise N(0, sigma^2):
            RDP(alpha) = alpha / (2 * sigma^2)
        """
        return alpha / (2 * sigma ** 2)

    def _compute_log_a_for_finite_sample(
        self, q: float, sigma: float, alpha: float
    ) -> float:
        """Compute RDP for subsampled Gaussian using the tight bound."""
        if alpha == float("inf"):
            return float("inf")

        alpha = float(alpha)

        log_q = math.log(q)
        log_1mq = math.log1p(-q) if q < 1 else -30.0

        terms = []
        for k in range(1, int(alpha) + 1):
            if k == 0:
                continue
            log_binom = self._log_choose(int(alpha), k)
            term = log_binom + k * log_q + (int(alpha) - k) * log_1mq
            term += self._gaussian_rdp(sigma, float(k * alpha))
            terms.append(term)

        if terms:
            max_t = max(terms)
            return (
                math.log(sum(math.exp(t - max_t) for t in terms)) / (alpha - 1)
                + max_t / (alpha - 1)
            )

        return 0.0

    def _log_choose(self, n: int, k: int) -> float:
        """Compute log(C(n, k)) using log-gamma for numerical stability."""
        return special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)

    def _get_epsilon_from_rdp(self, delta: float) -> float:
        """Convert Rényi DP to (ε, δ)-DP."""
        if self._num_steps == 0:
            return 0.0

        eps_candidates = []
        for alpha, rdp_value in self._rdp.items():
            if rdp_value == 0:
                continue
            eps = rdp_value + math.log(1.0 / delta) / (alpha - 1)
            eps_candidates.append(eps)

        if not eps_candidates:
            return 0.0

        return min(eps_candidates)

    def _estimate_epsilon_for_steps(
        self, total_steps: int, sigma: float, q: float, delta: float
    ) -> float:
        """Estimate eps for a given total number of steps."""
        additional_steps = total_steps - self._num_steps
        if additional_steps <= 0:
            return self._get_epsilon_from_rdp(delta)

        rdp_per_step = {}
        for alpha in self.orders:
            rdp_per_step[alpha] = self._compute_rdp(q, sigma, alpha)

        eps_candidates = []
        for alpha in self.orders:
            total_rdp = self._rdp[alpha] + additional_steps * rdp_per_step[alpha]
            eps = total_rdp + math.log(1.0 / delta) / (alpha - 1)
            eps_candidates.append(eps)

        return min(eps_candidates)

    def report(self) -> str:
        """Generate a human-readable privacy budget report."""
        budget = self.get_privacy_budget()
        lines = [
            "=" * 60,
            "DIFFERENTIAL PRIVACY BUDGET REPORT",
            "=" * 60,
            f"Mechanism:          Gaussian (subsampled)",
            f"Noise multiplier:   {budget.noise_multiplier:.4f}",
            f"Sampling rate:      {budget.sample_rate:.4f}",
            f"Total rounds:       {budget.rounds}",
            f"Target delta:       {budget.delta}",
            "-" * 60,
            f"Total epsilon:      {budget.epsilon:.4f}",
            "=" * 60,
        ]

        if self.noise_multiplier > 0:
            remaining_1 = self.get_remaining_budget(max_epsilon=1.0)
            remaining_5 = self.get_remaining_budget(max_epsilon=5.0)
            remaining_10 = self.get_remaining_budget(max_epsilon=10.0)
            lines.extend([
                f"Rounds remaining at eps <= 1.0:  {remaining_1}",
                f"Rounds remaining at eps <= 5.0:  {remaining_5}",
                f"Rounds remaining at eps <= 10.0: {remaining_10}",
                "=" * 60,
            ])

        if self._history:
            last = self._history[-1]
            lines.append(f"Last step: eps={last.get('epsilon', 'N/A')}, sigma={last['sigma']:.3f}")

        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=== Test 1: No noise (DP disabled) ===")
    acct = RDPAccountant(noise_multiplier=0.0, sample_rate=1.0)
    for _ in range(20):
        acct.step()
    print(acct.report())
    print()

    print("=== Test 2: sigma=2.0, full participation ===")
    acct = RDPAccountant(noise_multiplier=2.0, sample_rate=1.0)
    for _ in range(20):
        acct.step()
    print(acct.report())
    print()

    print("=== Test 3: sigma=1.0, subsampled q=0.3 ===")
    acct = RDPAccountant(noise_multiplier=1.0, sample_rate=0.3)
    for _ in range(50):
        acct.step()
    print(acct.report())
