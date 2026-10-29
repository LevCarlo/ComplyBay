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
- ``update="covariance"`` is adaptive Metropolis (Haario, Saksman & Tamminen
  2001, Bernoulli 7(2):223-242) restricted to burn-in: until
  ``covariance_start`` it runs the adaptive one-parameter scheme; from then on
  it proposes all parameters at once from N(0, s^2 C), where C is the sample
  covariance of the chain's own states since ``covariance_start // 2``
  (updated every step) and s starts at the optimal 2.38 / sqrt(d) (Gelman,
  Roberts & Gilks 1996) and is tuned towards ``target_acceptance`` (0.234).
  At the end of burn-in C and s are frozen, so the stored samples again come
  from a fixed random-walk Metropolis chain and no ergodicity condition for
  adaptive MCMC is needed. Correlated parameters (a trade-off ridge) are then
  crossed along the ridge instead of one axis at a time. Global jumps redraw
  one parameter, as for ``"one"``.
- The target returns (log prior, log likelihood) separately and a chain runs
  at ``temperature`` T, accepting with
  log alpha = d(log prior) + d(log likelihood) / T. T = 1 is the posterior.
- Parallel tempering (Geyer 1991, Computing Science and Statistics 23:156-163;
  Earl & Deem 2005, Phys. Chem. Chem. Phys. 7:3910-3916): with
  ``temperatures`` = (1, T_2, ..., T_max) every chain is a ladder of replicas,
  one per temperature, stepped together in one process. Each replica keeps and
  adapts its own proposal. Every ``swap_interval`` steps adjacent replicas
  i, i+1 (even pairs and odd pairs in turn) exchange their states with
  log alpha = (1/T_i - 1/T_{i+1}) (L_{i+1} - L_i), L the log likelihood; the
  prior cancels. This leaves the product of the tempered posteriors invariant,
  so the T = 1 replica, the only one stored, samples the posterior; the hot
  replicas cross between modes that the cold one cannot leave and hand them
  down. Without ``temperatures`` a chain draws exactly the random numbers it
  drew before tempering existed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

ProgressCallback = Callable[[int, int, float, float], None]  # chain, step, acceptance, log post


# Assumed: a geometric ladder with ratios of about 2-3 between adjacent temperatures
# keeps swaps accepted for the 5-10 parameters used here (swap acceptance is logged).
def geometric_temperatures(n_temperatures: int, max_temperature: float) -> tuple[float, ...]:
    """T_i = T_max^(i / (n - 1)), i = 0..n-1: 1 to ``max_temperature``."""
    if n_temperatures < 2 or max_temperature <= 1:
        raise ValueError("Need n_temperatures >= 2 and max_temperature > 1.")
    exponents = np.arange(n_temperatures) / (n_temperatures - 1)
    return tuple(float(t) for t in max_temperature**exponents)


# Assumed: the covariance proposal adds (1e-4 * bound width)^2 to each variance, so
# a parameter that never moved before covariance_start can still move. It is
# negligible against any variance a sampled parameter reaches in practice.
COVARIANCE_FLOOR = 1e-4


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
    update: Literal["one", "all", "covariance"] = "one"
    global_jump_interval: int | None = None
    adapt: bool = False
    target_acceptance: float | None = None  # default 0.44 for "one", else 0.234
    temperature: float = 1.0
    covariance_start: int | None = None  # "covariance" only; default burn_in // 2
    temperatures: Sequence[float] | None = None  # parallel tempering ladder, first = 1
    swap_interval: int = 1

    def __post_init__(self):
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if not 0 <= self.burn_in < self.n_steps:
            raise ValueError("Need 0 <= burn_in < n_steps.")
        if self.thin <= 0:
            raise ValueError("thin must be positive.")
        if self.update not in ("one", "all", "covariance"):
            raise ValueError("update must be 'one', 'all' or 'covariance'.")
        if self.global_jump_interval is not None and self.global_jump_interval <= 0:
            raise ValueError("global_jump_interval must be positive.")
        if self.temperature < 1:
            raise ValueError("temperature must be >= 1.")
        if self.adapt and self.burn_in == 0:
            raise ValueError("Adaptation runs during burn-in; set burn_in > 0.")
        if self.update == "covariance":
            if not self.adapt:
                raise ValueError("update='covariance' learns its proposal; set adapt=True.")
            if not 2 <= self.covariance_start_step < self.burn_in:
                raise ValueError("Need 2 <= covariance_start < burn_in.")
        elif self.covariance_start is not None:
            raise ValueError("covariance_start applies to update='covariance' only.")
        if self.temperatures is not None:
            ladder = np.asarray(self.temperatures, dtype=float)
            if (
                ladder.ndim != 1
                or ladder.size < 2
                or ladder[0] != 1
                or np.any(np.diff(ladder) <= 0)
            ):
                raise ValueError("temperatures must increase from 1 with at least two values.")
            if self.temperature != 1:
                raise ValueError("Give either temperature or temperatures, not both.")
            object.__setattr__(self, "temperatures", tuple(float(t) for t in ladder))
        if self.swap_interval < 1:
            raise ValueError("swap_interval must be positive.")

    @property
    def ladder(self) -> tuple[float, ...]:
        """Temperatures of the replicas of each chain; the first one is stored."""
        return self.temperatures if self.temperatures is not None else (self.temperature,)

    @property
    def covariance_start_step(self) -> int:
        if self.covariance_start is not None:
            return self.covariance_start
        return self.burn_in // 2

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
    proposal_sigma: np.ndarray  # final (adapted) sigmas; sqrt(diag) for "covariance"
    initial_theta: np.ndarray
    seed: int | None
    chain_id: int
    temperature: float
    proposal_covariance: np.ndarray | None = None  # frozen s^2 C (+ floor), "covariance" only
    swap_acceptance: np.ndarray | None = None  # per adjacent pair of the ladder, all steps
    details: np.ndarray | None = None  # (n_kept, n_details): the target's detail_names
    trace_steps: np.ndarray | None = None  # every thin-th step from 0, burn-in included
    trace_log_posterior: np.ndarray | None = None
    trace_details: np.ndarray | None = None  # (n_trace, n_details)

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
    detail_names: tuple[str, ...] = ()

    @property
    def log_posterior(self) -> np.ndarray:
        return self.log_prior + self.log_likelihood

    @property
    def acceptance_rates(self) -> np.ndarray:
        """Per chain, after burn-in."""
        return np.array([c.acceptance_rate(after_burn_in=self.config.burn_in) for c in self.chains])

    def flat_samples(self) -> np.ndarray:
        return self.samples.reshape(-1, self.samples.shape[-1])


def _n_details(target) -> int:
    return len(getattr(target, "detail_names", ()))


def _safe(target: SplitTarget, theta: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(log prior, log likelihood, details); details are the target's optional
    per-group values (``detail_names`` and ``log_prior_likelihood_and_details``)."""
    n = _n_details(target)
    failed = (-np.inf, -np.inf, np.full(n, np.nan))
    try:
        if n:
            log_prior, log_likelihood, details = target.log_prior_likelihood_and_details(theta)
        else:
            log_prior, log_likelihood = target.log_prior_and_likelihood(theta)
            details = np.zeros(0)
    except (FloatingPointError, OverflowError, ValueError):
        return failed
    if not np.isfinite(log_prior) or np.isnan(log_likelihood):
        return failed
    return float(log_prior), float(log_likelihood), np.asarray(details, dtype=float)


class _Replica:
    """One Metropolis chain at one temperature: state, proposal and its adaptation."""

    def __init__(self, target, theta, current, sigma, bounds, temperature, config):
        self.target, self.config, self.temperature = target, config, temperature
        self.theta, self.current = theta.copy(), current
        self.lower, self.upper = bounds
        self.sigma = sigma.copy()
        self.log_sigma = np.log(self.sigma)
        self.accepted = np.zeros(config.n_steps, dtype=bool)
        self.covariance = config.update == "covariance"
        if self.covariance:
            dim = theta.size
            self.start = config.covariance_start_step
            self.floor = np.diag((COVARIANCE_FLOOR * (self.upper - self.lower)) ** 2)
            self.log_scale = np.log(2.38 / np.sqrt(dim))
            self.n_seen, self.mean, self.scatter = 0, np.zeros(dim), np.zeros((dim, dim))
            self.cholesky = None

    def step(self, step: int, rng: np.random.Generator) -> None:
        config, theta, lower, upper = self.config, self.theta, self.lower, self.upper
        goal = config.acceptance_goal
        global_jump = (
            config.global_jump_interval is not None
            and (step + 1) % config.global_jump_interval == 0
        )
        joint = self.covariance and step >= self.start  # full-vector covariance proposal
        if joint and (self.cholesky is None or step < config.burn_in):
            cov = np.exp(2 * self.log_scale) * self.scatter / max(self.n_seen - 1, 1) + self.floor
            self.cholesky = np.linalg.cholesky(cov)
        index = None
        if global_jump and config.update != "all":
            # Redraw one parameter uniformly in its bounds, keep the others.
            jump = int(rng.integers(theta.size))
            proposal = theta.copy()
            proposal[jump] = rng.uniform(lower[jump], upper[jump])
        elif global_jump:
            proposal = rng.uniform(lower, upper)
        elif joint:
            proposal = theta + self.cholesky @ rng.standard_normal(theta.size)
        elif config.update == "all":
            proposal = theta + self.sigma * rng.standard_normal(theta.size)
        else:
            index = int(rng.integers(theta.size))
            proposal = theta.copy()
            proposal[index] += self.sigma[index] * rng.standard_normal()

        if np.all((proposal >= lower) & (proposal <= upper)):
            proposed = _safe(self.target, proposal)
        else:
            proposed = (-np.inf, -np.inf, None)
        current = self.current
        log_alpha = (proposed[0] - current[0]) + (proposed[1] - current[1]) / self.temperature
        if np.isfinite(log_alpha) and (log_alpha >= 0 or np.log(rng.random()) < log_alpha):
            self.theta, self.current = proposal, proposed
            self.accepted[step] = True

        if config.adapt and step < config.burn_in and not global_jump:
            # Robbins-Monro: grow the step after acceptances, shrink after rejections.
            if joint:
                gain = (step - self.start + 1) ** -0.6
                self.log_scale += gain * (self.accepted[step] - goal)
            else:
                gain = (step + 1) ** -0.6
                local_goal = 0.44 if self.covariance else goal
                if index is None:
                    self.log_sigma += gain * (self.accepted[step] - local_goal)
                else:
                    self.log_sigma[index] += gain * (self.accepted[step] - local_goal)
                self.sigma = np.minimum(np.exp(self.log_sigma), upper - lower)
                self.log_sigma = np.log(self.sigma)
        if self.covariance and self.start // 2 <= step < config.burn_in:
            # Welford update of the mean and scatter matrix of the visited states.
            self.n_seen += 1
            delta = self.theta - self.mean
            self.mean += delta / self.n_seen
            self.scatter += np.outer(delta, self.theta - self.mean)


def _swap(replicas: list[_Replica], parity: int, rng, attempts, accepts) -> None:
    """Exchange states between adjacent replicas (pairs starting at ``parity``)."""
    for i in range(parity, len(replicas) - 1, 2):
        cold, hot = replicas[i], replicas[i + 1]
        log_alpha = (1 / cold.temperature - 1 / hot.temperature) * (
            hot.current[1] - cold.current[1]
        )
        attempts[i] += 1
        if log_alpha >= 0 or np.log(rng.random()) < log_alpha:
            cold.theta, hot.theta = hot.theta, cold.theta
            cold.current, hot.current = hot.current, cold.current
            accepts[i] += 1


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
    """One Metropolis chain from ``theta0`` (a ladder of replicas with ``temperatures``)."""
    theta = np.asarray(theta0, dtype=float).copy()
    lower, upper = (np.asarray(b, dtype=float) for b in config.bounds)
    if lower.shape != theta.shape or upper.shape != theta.shape or np.any(upper <= lower):
        raise ValueError("bounds must be two arrays like theta with lower < upper.")
    sigma = np.broadcast_to(np.asarray(config.proposal_sigma, dtype=float), theta.shape).copy()
    if np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
        raise ValueError("proposal_sigma must be positive and finite.")
    rng = np.random.default_rng(seed)

    current = _safe(target, theta)
    if not np.isfinite(current[0] + current[1]):
        raise ValueError(f"Chain {chain_id}: initial theta is outside the target support.")
    replicas = [
        _Replica(target, theta, current, sigma, (lower, upper), t, config) for t in config.ladder
    ]
    cold = replicas[0]
    n_pairs = len(replicas) - 1
    attempts, accepts = np.zeros(n_pairs, dtype=int), np.zeros(n_pairs, dtype=int)
    kept, kept_prior, kept_like, kept_details = [], [], [], []
    trace_steps, trace_lp, trace_details = [], [], []

    for step in range(config.n_steps):
        for replica in replicas:
            replica.step(step, rng)
        if n_pairs and (step + 1) % config.swap_interval == 0:
            _swap(replicas, (step // config.swap_interval) % 2, rng, attempts, accepts)

        if step >= config.burn_in and (step - config.burn_in) % config.thin == 0:
            kept.append(cold.theta.copy())
            kept_prior.append(cold.current[0])
            kept_like.append(cold.current[1])
            kept_details.append(cold.current[2])
        if step % config.thin == 0:
            trace_steps.append(step)
            trace_lp.append(cold.current[0] + cold.current[1])
            trace_details.append(cold.current[2])
        if (
            progress is not None
            and progress_every
            and ((step + 1) % progress_every == 0 or step + 1 == config.n_steps)
        ):
            progress(
                chain_id,
                step + 1,
                float(cold.accepted[: step + 1].mean()),
                cold.current[0] + cold.current[1],
            )

    proposal_covariance, sigma = None, cold.sigma
    if cold.covariance:
        proposal_covariance = cold.cholesky @ cold.cholesky.T
        sigma = np.sqrt(np.diag(proposal_covariance))
    swap_acceptance = accepts / np.maximum(attempts, 1) if n_pairs else None
    return ChainResult(
        samples=np.asarray(kept),
        log_prior=np.asarray(kept_prior),
        log_likelihood=np.asarray(kept_like),
        accepted=cold.accepted,
        proposal_sigma=sigma,
        initial_theta=theta,
        seed=seed,
        chain_id=chain_id,
        temperature=config.ladder[0],
        proposal_covariance=proposal_covariance,
        swap_acceptance=swap_acceptance,
        details=np.asarray(kept_details),
        trace_steps=np.asarray(trace_steps),
        trace_log_posterior=np.asarray(trace_lp),
        trace_details=np.asarray(trace_details),
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
        detail_names=tuple(getattr(target, "detail_names", ())),
    )
