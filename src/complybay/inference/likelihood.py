"""Joint Gaussian likelihood with one sampled noise scale per data group.

For data group k with residuals r_i = d_i - g_i(m), standard deviations s_i and
noise scale h_k (a sampled hyperparameter, ``log10 h_k``):

    log L = sum_k [ -1/2 sum_i (r_i / (h_k s_i))^2 - N_k ln h_k ]

Terms that depend on neither the model nor h_k (2 pi, the s_i) are dropped.
The N_k ln h_k term keeps h_k from growing to absorb any misfit, so each group
gets the weight its residuals support instead of a hand-tuned one; at fixed
residuals log L is largest for h_k^2 = chi2_k / N_k. Samples are independent
unless an :class:`ErrorCorrelation` covers them.

Correlated errors (ACF lags, and Z and P ACFs of the same events): with z the
residuals divided by s over the fitted samples of the members of a block, R
their correlation, and only the k leading eigenvectors V (eigenvalues l) of R
kept (``variance_fraction`` of the trace), the block contributes the Gaussian
log density of y = Q^T z, Q = V diag(l)^-1/2:

    log L_block = -1/2 y^T M^-1 y - 1/2 ln det M,   M = Q^T D R D Q,

where D holds the noise scale h of each member on its samples. With one
member, M = h^2 I and this is -1/2 |y|^2 / h^2 - k ln h: the k-dimensional
version of the independent formula. Keeping all eigenvectors would give the
full Gaussian, but these correlations are nearly singular and the discarded
directions carry no error variance, so including them would put N instead of
k in the h term and drive h too low (the calibration is in
``docs/inference.md``).

A forward failure (an exception from a solver, or a non-finite prediction such
as a missing dispersion mode) gives log L = -inf with the reason.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_triangular

from complybay.forward import synthetic_acfs
from complybay.inference.observations import ACFObservation, Observation
from complybay.model import OBSModel


@dataclass(frozen=True)
class GroupFit:
    name: str
    prediction: np.ndarray  # full prediction (all ACF lags)
    chi2: float  # sum (r / s)^2 over the samples in the likelihood, before scaling
    n: int
    scale: float  # h
    log_likelihood: float


@dataclass(frozen=True)
class BlockFit:
    name: str
    members: tuple[str, ...]
    chi2: float  # y^T M^-1 y at the sampled noise scales
    k: int  # retained eigenvectors: the effective number of data
    log_likelihood: float


@dataclass(frozen=True)
class LikelihoodEvaluation:
    log_likelihood: float
    groups: tuple[GroupFit, ...] = ()
    error: str | None = None
    blocks: tuple[BlockFit, ...] = ()

    @property
    def success(self) -> bool:
        return self.error is None


def predict(observations: Sequence[Observation], model: OBSModel) -> list[np.ndarray]:
    """Predictions for every observation; ACFs sharing a key share one response."""
    predictions: list[np.ndarray | None] = [None] * len(observations)
    acf_groups: dict[tuple, list[int]] = defaultdict(list)
    for index, obs in enumerate(observations):
        if isinstance(obs, ACFObservation):
            acf_groups[obs.synthetic_key()].append(index)
        else:
            predictions[index] = obs.predict(model)
    for indices in acf_groups.values():
        first = observations[indices[0]]
        components = tuple(dict.fromkeys(observations[i].component for i in indices))
        _, acfs = synthetic_acfs(
            model,
            first.ray_p,
            first.processing,
            components,
            dt=first.dt,
            duration=first.duration,
            snr=first.snr,
        )
        for i in indices:
            predictions[i] = acfs[observations[i].component]
    return predictions


# Assumed: the error eigenvectors holding 99 % of the variance are kept; in Monte
# Carlo tests of kim2023 stacks this gave chi2 / k = 1.0-1.1 (docs/inference.md).
DEFAULT_VARIANCE_FRACTION = 0.99


@dataclass(frozen=True)
class ErrorCorrelation:
    """Correlation of the errors of ``members`` over their fitted samples.

    ``correlation`` is joint over the members' fitted samples (the ACF lags
    inside the window), concatenated in the order of ``members``.
    """

    members: tuple[str, ...]
    correlation: np.ndarray
    variance_fraction: float = DEFAULT_VARIANCE_FRACTION
    name: str = ""
    _basis: np.ndarray = field(init=False, repr=False, compare=False)
    _pieces: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        members = tuple(self.members)
        if not members or len(set(members)) != len(members):
            raise ValueError("An error correlation needs distinct members.")
        corr = np.array(self.correlation, dtype=float)
        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            raise ValueError("correlation must be a square matrix.")
        if not np.allclose(corr, corr.T, atol=1e-10) or not np.allclose(np.diag(corr), 1.0):
            raise ValueError("correlation must be symmetric with a unit diagonal.")
        if not 0 < self.variance_fraction <= 1:
            raise ValueError("variance_fraction must be in (0, 1].")
        values, vectors = np.linalg.eigh(corr)
        values, vectors = values[::-1], vectors[:, ::-1]
        if values[0] <= 0:
            raise ValueError("correlation has no positive eigenvalue.")
        cumulative = np.cumsum(np.clip(values, 0.0, None)) / np.clip(values, 0.0, None).sum()
        k = int(np.searchsorted(cumulative, self.variance_fraction - 1e-12)) + 1
        corr.flags.writeable = False
        basis = vectors[:, :k] / np.sqrt(values[:k])
        basis.flags.writeable = False
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "correlation", corr)
        object.__setattr__(self, "_basis", basis)
        object.__setattr__(self, "_pieces", {})
        if not self.name:
            object.__setattr__(self, "name", "+".join(members))

    @property
    def basis(self) -> np.ndarray:
        """Q (n_samples, k): y = Q^T z has unit covariance when all h = 1."""
        return self._basis

    @property
    def k(self) -> int:
        return self._basis.shape[1]

    def log_likelihood(self, z: np.ndarray, scales: np.ndarray) -> tuple[float, float]:
        """(log L, y^T M^-1 y) for scaled-by-sigma residuals ``z`` and per-sample h."""
        y = self._basis.T @ z
        if np.all(scales == scales[0]):
            chi2 = float(y @ y) / scales[0] ** 2
            return -0.5 * chi2 - self.k * np.log(scales[0]), chi2
        # M = Q^T D R D Q = sum_ab h_a h_b Q_a^T R_ab Q_b over the groups of samples
        # sharing one h; the k x k pieces are computed once per grouping, which
        # avoids an (n x n) product per call.
        values, labels = np.unique(scales, return_inverse=True)
        m = np.einsum("a,b,abij->ij", values, values, self._grouped(labels, values.size))
        chol = np.linalg.cholesky(m)
        w = solve_triangular(chol, y, lower=True)
        chi2 = float(w @ w)
        return -0.5 * chi2 - float(np.sum(np.log(np.diag(chol)))), chi2

    def _grouped(self, labels: np.ndarray, n_groups: int) -> np.ndarray:
        key = labels.tobytes()
        pieces = self._pieces.get(key)
        if pieces is None:
            q = [np.where((labels == a)[:, None], self._basis, 0.0) for a in range(n_groups)]
            pieces = np.array([[qa.T @ self.correlation @ qb for qb in q] for qa in q])
            if len(self._pieces) >= 8:  # assumed bound; a run uses one or two groupings
                self._pieces.clear()
            self._pieces[key] = pieces
        return pieces


@dataclass(frozen=True)
class JointLikelihood:
    observations: tuple[Observation, ...]
    correlations: tuple[ErrorCorrelation, ...] = ()

    def __post_init__(self):
        observations = tuple(self.observations)
        if not observations:
            raise ValueError("A likelihood needs at least one observation.")
        names = [o.name for o in observations]
        if len(names) != len(set(names)):
            raise ValueError(f"Observation names must be unique: {names}")
        correlations = tuple(self.correlations)
        covered = [m for c in correlations for m in c.members]
        if len(covered) != len(set(covered)):
            raise ValueError("An observation can be in one error correlation only.")
        by_name = dict(zip(names, observations, strict=True))
        for block in correlations:
            missing = [m for m in block.members if m not in by_name]
            if missing:
                raise ValueError(
                    f"Error correlation {block.name!r}: unknown observations {missing}."
                )
            n = sum(int(np.count_nonzero(_fitted(by_name[m]))) for m in block.members)
            if block.correlation.shape[0] != n:
                raise ValueError(
                    f"Error correlation {block.name!r} is {block.correlation.shape[0]} x "
                    f"{block.correlation.shape[0]}, but its members have {n} fitted samples."
                )
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "correlations", correlations)

    @property
    def noise_parameters(self) -> tuple[str, ...]:
        return tuple(o.noise_parameter for o in self.observations if o.noise_parameter)

    @property
    def terms(self) -> tuple[str, ...]:
        """Names of the additive log-likelihood terms: independent groups, then blocks."""
        in_block = {m for c in self.correlations for m in c.members}
        independent = tuple(o.name for o in self.observations if o.name not in in_block)
        return independent + tuple(c.name for c in self.correlations)

    @property
    def sizes(self) -> dict[str, int]:
        """Number of fitted samples per observation (N of chi2 / N)."""
        return {o.name: int(np.count_nonzero(_fitted(o))) for o in self.observations}

    def details(self, evaluation: LikelihoodEvaluation) -> np.ndarray:
        """chi2 per observation (stated sigma, h = 1), then log L per term; NaN on failure."""
        n_obs, n_terms = len(self.observations), len(self.terms)
        if not evaluation.success:
            return np.full(n_obs + n_terms, np.nan)
        chi2 = [g.chi2 for g in evaluation.groups]
        by_name = {g.name: g.log_likelihood for g in evaluation.groups}
        by_name.update({b.name: b.log_likelihood for b in evaluation.blocks})
        return np.array(chi2 + [by_name[t] for t in self.terms], dtype=float)

    def evaluate(
        self, model: OBSModel, parameters: Mapping[str, float] | None = None
    ) -> LikelihoodEvaluation:
        """``parameters`` must hold every ``log10`` noise parameter the observations name."""
        parameters = parameters or {}
        try:
            predictions = predict(self.observations, model)
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            return LikelihoodEvaluation(-np.inf, error=f"forward: {exc}")
        in_block = {m for c in self.correlations for m in c.members}
        groups, total, z_of, scale_of = [], 0.0, {}, {}
        for obs, prediction in zip(self.observations, predictions, strict=True):
            mask = _fitted(obs)
            residual = (obs.values - prediction)[mask]
            if not np.all(np.isfinite(residual)):
                return LikelihoodEvaluation(-np.inf, error=f"forward: non-finite {obs.name}")
            z = residual / obs.sigma[mask]
            chi2 = float(z @ z)
            n = residual.size
            if obs.noise_parameter is None:
                scale = 1.0
            else:
                if obs.noise_parameter not in parameters:
                    raise KeyError(f"Missing noise parameter {obs.noise_parameter!r}.")
                scale = 10.0 ** float(parameters[obs.noise_parameter])
            if obs.name in in_block:
                z_of[obs.name], scale_of[obs.name] = z, scale
                log_l = np.nan  # counted in its block
            else:
                log_l = -0.5 * chi2 / scale**2 - n * np.log(scale)
                total += log_l
            groups.append(GroupFit(obs.name, prediction, chi2, n, scale, log_l))
        blocks = []
        for block in self.correlations:
            z = np.concatenate([z_of[m] for m in block.members])
            scales = np.concatenate([np.full(z_of[m].size, scale_of[m]) for m in block.members])
            log_l, chi2 = block.log_likelihood(z, scales)
            blocks.append(BlockFit(block.name, block.members, chi2, block.k, log_l))
            total += log_l
        return LikelihoodEvaluation(total, tuple(groups), blocks=tuple(blocks))


def _fitted(obs: Observation) -> np.ndarray:
    """Mask of the samples that enter the likelihood."""
    if isinstance(obs, ACFObservation):
        return obs.mask
    return np.ones(obs.values.size, dtype=bool)
