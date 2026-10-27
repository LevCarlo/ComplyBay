"""Random-walk Metropolis sampling of a posterior target.

Adapted from SeisForge (``seisforge.inv.samplers``, commit 019d568): Gaussian
random-walk proposals, an optional uniform "global jump" every
``global_jump_interval`` steps (an independence proposal, symmetric inside the
bounds, from the ``step4uwalk`` idea of pymcinv), burn-in and thinning, and
independent chains run serially, in threads or in processes. With
``update="one"`` a global jump redraws one random parameter uniformly within
its bounds and keeps the others, so a chain stuck in a wrong mode of, say, an
interface depth can leave it; with ``"all"`` it redraws the whole vector as in
SeisForge, which is rarely accepted in more than a few dimensions.

Changes from SeisForge:

- ``update="one"`` perturbs one randomly chosen parameter per step
  (random-scan Metropolis-within-Gibbs); ``"all"`` perturbs all at once.
- ``adapt=True`` tunes the proposal sigmas during burn-in only (Robbins-Monro
  on log sigma towards ``target_acceptance``), then freezes them, so the stored
  samples come from a fixed, valid Markov chain. With ``update="one"`` each
  sigma adapts on its own; with ``"all"`` one accept/reject says nothing about
  which parameter was too far, so only a common factor adapts and the sigma
  ratios stay as given.
- The target returns (log prior, log likelihood) separately and a chain runs
  at ``temperature`` T, accepting with
  log alpha = d(log prior) + d(log likelihood) / T. T = 1 is the posterior;
  this is the hook for parallel tempering (chains at several T plus swaps).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

ProgressCallback = Callable[[int, int, float, float], None]  # chain, step, acceptance, log post


class SplitTarget(Protocol):
    """A target that returns (log prior, log likelihood) for a parameter vector."""

    def log_prior_and_likelihood(self, theta: np.ndarray) -> tuple[float, float]: ...


@dataclass(frozen=True)
class LogProbTarget:
    """Wrap a plain log-density as a target with zero likelihood term (no tempering)."""

    log_prob: Callable[[np.ndarray], float]

    def log_prior_and_likelihood(self, theta) -> tuple[float, float]:
        return float(self.log_prob(theta)), 0.0


@dataclass(frozen=True)
class MetropolisConfig:
    n_steps: int
    proposal_sigma: Sequence[float] | float
    bounds: tuple[Sequence[float], Sequence[float]]
    burn_in: int = 0
    thin: int = 1
    update: Literal["one", "all"] = "one"
    global_jump_interval: int | None = None
    adapt: bool = False
    target_acceptance: float | None = None  # default 0.44 for "one", 0.234 for "all"
    temperature: float = 1.0

    def __post_init__(self):
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if not 0 <= self.burn_in < self.n_steps:
            raise ValueError("Need 0 <= burn_in < n_steps.")
        if self.thin <= 0:
            raise ValueError("thin must be positive.")
        if self.update not in ("one", "all"):
            raise ValueError("update must be 'one' or 'all'.")
        if self.global_jump_interval is not None and self.global_jump_interval <= 0:
            raise ValueError("global_jump_interval must be positive.")
        if self.temperature < 1:
            raise ValueError("temperature must be >= 1.")
        if self.adapt and self.burn_in == 0:
            raise ValueError("Adaptation runs during burn-in; set burn_in > 0.")

    @property
    def acceptance_goal(self) -> float:
        if self.target_acceptance is not None:
            return self.target_acceptance
        return 0.44 if self.update == "one" else 0.234


@dataclass(frozen=True)
class ChainResult:
    """Stored (post burn-in, thinned) samples of one chain and its diagnostics."""

    samples: np.ndarray  # (n_kept, n_parameters)
    log_prior: np.ndarray
    log_likelihood: np.ndarray
    accepted: np.ndarray  # every step, burn-in included
    proposal_sigma: np.ndarray  # final (adapted) sigmas
    initial_theta: np.ndarray
    seed: int | None
    chain_id: int
    temperature: float

    @property
    def log_posterior(self) -> np.ndarray:
        return self.log_prior + self.log_likelihood

    def acceptance_rate(self, *, after_burn_in: int = 0) -> float:
        return float(np.mean(self.accepted[after_burn_in:]))


@dataclass(frozen=True)
class MCMCResult:
    samples: np.ndarray  # (n_chains, n_kept, n_parameters)
    log_prior: np.ndarray  # (n_chains, n_kept)
    log_likelihood: np.ndarray
    chains: tuple[ChainResult, ...]
    config: MetropolisConfig
    names: tuple[str, ...] = ()

    @property
    def log_posterior(self) -> np.ndarray:
        return self.log_prior + self.log_likelihood

    @property
    def acceptance_rates(self) -> np.ndarray:
        """Per chain, after burn-in."""
        return np.array([c.acceptance_rate(after_burn_in=self.config.burn_in) for c in self.chains])

    def flat_samples(self) -> np.ndarray:
        return self.samples.reshape(-1, self.samples.shape[-1])


def _safe(target: SplitTarget, theta: np.ndarray) -> tuple[float, float]:
    try:
        log_prior, log_likelihood = target.log_prior_and_likelihood(theta)
    except (FloatingPointError, OverflowError, ValueError):
        return -np.inf, -np.inf
    if not np.isfinite(log_prior) or np.isnan(log_likelihood):
        return -np.inf, -np.inf
    return float(log_prior), float(log_likelihood)


def run_chain(
    target: SplitTarget,
    theta0,
    config: MetropolisConfig,
    *,
    seed: int | None = None,
    chain_id: int = 0,
    progress: ProgressCallback | None = None,
    progress_every: int | None = None,
) -> ChainResult:
    """One Metropolis chain from ``theta0``."""
    theta = np.asarray(theta0, dtype=float).copy()
    lower, upper = (np.asarray(b, dtype=float) for b in config.bounds)
    if lower.shape != theta.shape or upper.shape != theta.shape or np.any(upper <= lower):
        raise ValueError("bounds must be two arrays like theta with lower < upper.")
    sigma = np.broadcast_to(np.asarray(config.proposal_sigma, dtype=float), theta.shape).copy()
    if np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
        raise ValueError("proposal_sigma must be positive and finite.")
    rng = np.random.default_rng(seed)
    temperature = config.temperature

    current = _safe(target, theta)
    if not np.isfinite(current[0] + current[1]):
        raise ValueError(f"Chain {chain_id}: initial theta is outside the target support.")
    initial = theta.copy()
    accepted = np.zeros(config.n_steps, dtype=bool)
    kept, kept_prior, kept_like = [], [], []
    log_sigma = np.log(sigma)
    goal = config.acceptance_goal

    for step in range(config.n_steps):
        global_jump = (
            config.global_jump_interval is not None
            and (step + 1) % config.global_jump_interval == 0
        )
        index = None
        if global_jump and config.update == "one":
            # Redraw one parameter uniformly in its bounds, keep the others.
            jump = int(rng.integers(theta.size))
            proposal = theta.copy()
            proposal[jump] = rng.uniform(lower[jump], upper[jump])
        elif global_jump:
            proposal = rng.uniform(lower, upper)
        elif config.update == "one":
            index = int(rng.integers(theta.size))
            proposal = theta.copy()
            proposal[index] += sigma[index] * rng.standard_normal()
        else:
            proposal = theta + sigma * rng.standard_normal(theta.size)

        if np.all((proposal >= lower) & (proposal <= upper)):
            proposed = _safe(target, proposal)
        else:
            proposed = (-np.inf, -np.inf)
        log_alpha = (proposed[0] - current[0]) + (proposed[1] - current[1]) / temperature
        if np.isfinite(log_alpha) and (log_alpha >= 0 or np.log(rng.random()) < log_alpha):
            theta, current = proposal, proposed
            accepted[step] = True

        if config.adapt and step < config.burn_in and not global_jump:
            # Robbins-Monro: grow sigma after acceptances, shrink after rejections.
            gain = (step + 1) ** -0.6
            if index is None:
                log_sigma += gain * (accepted[step] - goal)
            else:
                log_sigma[index] += gain * (accepted[step] - goal)
            sigma = np.minimum(np.exp(log_sigma), upper - lower)
            log_sigma = np.log(sigma)

        if step >= config.burn_in and (step - config.burn_in) % config.thin == 0:
            kept.append(theta.copy())
            kept_prior.append(current[0])
            kept_like.append(current[1])
        if (
            progress is not None
            and progress_every
            and ((step + 1) % progress_every == 0 or step + 1 == config.n_steps)
        ):
            progress(chain_id, step + 1, float(accepted[: step + 1].mean()), sum(current))

    return ChainResult(
        samples=np.asarray(kept),
        log_prior=np.asarray(kept_prior),
        log_likelihood=np.asarray(kept_like),
        accepted=accepted,
        proposal_sigma=sigma,
        initial_theta=initial,
        seed=seed,
        chain_id=chain_id,
        temperature=temperature,
    )


def chain_seeds(seed: int | None, n_chains: int) -> list[int | None]:
    """Independent per-chain seeds spawned from one seed."""
    if seed is None:
        return [None] * n_chains
    children = np.random.SeedSequence(seed).spawn(n_chains)
    return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]


def _worker(args) -> ChainResult:
    target, theta0, config, seed, chain_id, progress, progress_every = args
    return run_chain(
        target,
        theta0,
        config,
        seed=seed,
        chain_id=chain_id,
        progress=progress,
        progress_every=progress_every,
    )


def run_metropolis(
    target: SplitTarget,
    initial_thetas,
    config: MetropolisConfig,
    *,
    seed: int | None = None,
    executor: Literal["serial", "thread", "process"] = "serial",
    max_workers: int | None = None,
    names: Sequence[str] = (),
    progress: ProgressCallback | None = None,
    progress_every: int | None = None,
) -> MCMCResult:
    """Independent chains, one per row of ``initial_thetas``.

    ``process`` runs chains in separate processes (the target must be
    picklable; progress callbacks are not supported there). Use exact prior
    samples (:func:`complybay.parameterization.sample_prior`) as diverse starts.
    """
    starts = np.atleast_2d(np.asarray(initial_thetas, dtype=float))
    seeds = chain_seeds(seed, starts.shape[0])
    if executor == "process" and progress is not None:
        raise ValueError("progress callbacks are not supported with the process executor.")
    args = [
        (target, starts[i], config, seeds[i], i, progress, progress_every)
        for i in range(starts.shape[0])
    ]
    if executor == "serial" or len(args) == 1:
        chains = tuple(_worker(a) for a in args)
    elif executor == "thread":
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            chains = tuple(pool.map(_worker, args))
    elif executor == "process":
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            chains = tuple(pool.map(_worker, args))
    else:
        raise ValueError(f"Unknown executor {executor!r}.")
    return MCMCResult(
        samples=np.stack([c.samples for c in chains]),
        log_prior=np.stack([c.log_prior for c in chains]),
        log_likelihood=np.stack([c.log_likelihood for c in chains]),
        chains=chains,
        config=config,
        names=tuple(names),
    )
