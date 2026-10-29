from dataclasses import dataclass

import numpy as np
import pytest
from test_inference import _setup
from test_prior import _prior

from complybay.inference import (
    LogProbTarget,
    MetropolisConfig,
    PosteriorTarget,
    geometric_temperatures,
    lagging_chains,
    run_chain,
    run_metropolis,
)
from complybay.parameterization import sample_prior

MEAN = np.array([1.0, -2.0])
STD = np.array([0.5, 2.0])
BOX = (np.array([-10.0, -30.0]), np.array([10.0, 30.0]))


def gaussian(theta):
    return float(-0.5 * np.sum(((theta - MEAN) / STD) ** 2))


@dataclass(frozen=True)
class GaussianLikelihood:
    """Flat prior on the box, unit Gaussian likelihood: tempering at T widens it by sqrt(T)."""

    def log_prior_and_likelihood(self, theta):
        return 0.0, float(-0.5 * np.sum(theta**2))


def _config(**kwargs):
    base = dict(n_steps=40000, proposal_sigma=[3.0, 0.1], bounds=BOX, burn_in=5000, adapt=True)
    base.update(kwargs)
    return MetropolisConfig(**base)


@pytest.mark.parametrize(
    ("update", "sigma"),
    [
        ("one", [3.0, 0.1]),  # each sigma adapts on its own: wrong ratios are fixed
        ("all", [0.05, 0.2]),  # one shared factor: only the overall scale (10x) is fixed
    ],
)
def test_recovers_a_gaussian_from_badly_scaled_proposals(update, sigma):
    chain = run_chain(
        LogProbTarget(gaussian), [0.0, 0.0], _config(update=update, proposal_sigma=sigma), seed=1
    )
    samples = chain.samples
    np.testing.assert_allclose(samples.mean(axis=0), MEAN, atol=0.1 * STD.max())
    np.testing.assert_allclose(samples.std(axis=0), STD, rtol=0.1)
    # Adaptation moved the sigmas towards the target scales and hit the acceptance goal.
    goal = 0.44 if update == "one" else 0.234
    assert chain.acceptance_rate(after_burn_in=5000) == pytest.approx(goal, abs=0.08)
    if update == "one":
        assert chain.proposal_sigma[0] < 3.0 and chain.proposal_sigma[1] > 0.1
    else:
        np.testing.assert_allclose(chain.proposal_sigma[1] / chain.proposal_sigma[0], 4.0)


def test_temperature_flattens_only_the_likelihood():
    box = (np.full(1, -20.0), np.full(1, 20.0))
    config = MetropolisConfig(40000, 1.0, box, burn_in=4000, adapt=True, temperature=4.0)
    chain = run_chain(GaussianLikelihood(), [0.0], config, seed=2)
    assert chain.samples.std() == pytest.approx(2.0, rel=0.1)  # sqrt(T) * 1
    assert np.all(chain.log_prior == 0.0)


def test_burn_in_thin_and_reproducibility():
    config = MetropolisConfig(1000, [0.5, 1.0], BOX, burn_in=200, thin=10)
    a = run_chain(LogProbTarget(gaussian), [0.0, 0.0], config, seed=5)
    b = run_chain(LogProbTarget(gaussian), [0.0, 0.0], config, seed=5)
    c = run_chain(LogProbTarget(gaussian), [0.0, 0.0], config, seed=6)
    assert a.samples.shape == (80, 2) and a.accepted.shape == (1000,)
    np.testing.assert_array_equal(a.samples, b.samples)
    assert not np.array_equal(a.samples, c.samples)


def test_samples_stay_inside_bounds():
    box = (np.array([0.0, -1.0]), np.array([1.5, 1.0]))  # cuts the Gaussian
    chain = run_chain(LogProbTarget(gaussian), [1.0, 0.0], MetropolisConfig(5000, 0.5, box), seed=3)
    assert np.all(chain.samples >= box[0]) and np.all(chain.samples <= box[1])


def test_global_jumps_cross_between_separated_modes():
    def bimodal(theta):
        return float(
            np.logaddexp(-0.5 * ((theta[0] + 5) / 0.3) ** 2, -0.5 * ((theta[0] - 5) / 0.3) ** 2)
        )

    box = (np.array([-10.0]), np.array([10.0]))
    local = run_chain(LogProbTarget(bimodal), [-5.0], MetropolisConfig(20000, 0.3, box), seed=4)
    jumps = run_chain(
        LogProbTarget(bimodal),
        [-5.0],
        MetropolisConfig(20000, 0.3, box, global_jump_interval=10),
        seed=4,
    )
    assert np.all(local.samples < 0)  # never leaves the first mode
    assert 0.3 < np.mean(jumps.samples > 0) < 0.7  # both modes, about equally


def test_configuration_and_start_errors():
    with pytest.raises(ValueError, match="burn_in"):
        MetropolisConfig(100, 1.0, BOX, adapt=True)
    with pytest.raises(ValueError, match="temperature"):
        MetropolisConfig(100, 1.0, BOX, temperature=0.5)
    with pytest.raises(ValueError, match="initial theta"):
        run_chain(LogProbTarget(lambda t: -np.inf), [0.0, 0.0], MetropolisConfig(10, 1.0, BOX))


@pytest.mark.parametrize("executor", ["thread", "process"])
def test_executors_give_the_serial_result(executor):
    config = MetropolisConfig(500, [0.5, 1.0], BOX, burn_in=100)
    starts = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0]])
    serial = run_metropolis(LogProbTarget(gaussian), starts, config, seed=11)
    other = run_metropolis(LogProbTarget(gaussian), starts, config, seed=11, executor=executor)
    assert serial.samples.shape == (3, 400, 2)
    np.testing.assert_array_equal(serial.samples, other.samples)


def test_prior_only_run_stays_in_the_prior():
    prior = _prior()
    target, _, _ = _setup()
    prior_only = PosteriorTarget(target.prior, target.likelihood, prior_only=True)
    starts = sample_prior(target.prior, 2, rng=0).theta
    lower, upper = target.prior.parameterization.bounds()
    config = MetropolisConfig(300, target.prior.parameterization.proposal_sigmas(), (lower, upper))
    result = run_metropolis(prior_only, starts, config, seed=0, names=prior_only.names)
    assert result.names == prior_only.names
    assert np.all(result.log_likelihood == 0.0)
    for theta in result.flat_samples()[::50]:
        assert target.prior.evaluate(theta).success
    assert prior.parameterization.names == prior_only.names[:5]


def test_posterior_chain_climbs_towards_the_truth():
    target, theta_true, _ = _setup()
    start = theta_true.copy()
    start[0] = 1.2  # sediment twice as thick
    lower, upper = target.prior.parameterization.bounds()
    config = MetropolisConfig(400, target.prior.parameterization.proposal_sigmas(), (lower, upper))
    chain = run_chain(target, start, config, seed=1)
    assert chain.log_posterior[-1] > target(start) + 10


# A narrow ridge: correlation 0.99 between parameters of very different scales.
RIDGE_STD = np.array([1.0, 0.01])
RIDGE_CORR = 0.99
RIDGE_COV = np.array(
    [
        [RIDGE_STD[0] ** 2, RIDGE_CORR * RIDGE_STD[0] * RIDGE_STD[1]],
        [RIDGE_CORR * RIDGE_STD[0] * RIDGE_STD[1], RIDGE_STD[1] ** 2],
    ]
)
RIDGE_PRECISION = np.linalg.inv(RIDGE_COV)
RIDGE_BOX = (np.array([-10.0, -0.1]), np.array([10.0, 0.1]))


def ridge(theta):
    return float(-0.5 * theta @ RIDGE_PRECISION @ theta)


def _ridge_chain(update, seed=0):
    config = MetropolisConfig(
        30000, [0.5, 0.005], RIDGE_BOX, burn_in=10000, update=update, adapt=True
    )
    return run_chain(LogProbTarget(ridge), [0.0, 0.0], config, seed=seed)


def test_covariance_proposal_learns_a_correlated_ridge():
    # Analytic check: the frozen proposal is 2.38^2 / d * C times a tuned factor of
    # order 1, so its correlation must match the target's and the samples must
    # reproduce the known covariance.
    chain = _ridge_chain("covariance")
    proposal = chain.proposal_covariance
    corr = proposal[0, 1] / np.sqrt(proposal[0, 0] * proposal[1, 1])
    assert corr == pytest.approx(RIDGE_CORR, abs=0.005)
    ratio = proposal / RIDGE_COV
    assert 0.3 < ratio[0, 0] / (2.38**2 / 2) < 3  # Robbins-Monro factor stays O(1)
    np.testing.assert_allclose(chain.proposal_sigma, np.sqrt(np.diag(proposal)))
    np.testing.assert_allclose(np.cov(chain.samples.T), RIDGE_COV, rtol=0.15, atol=1e-6)
    assert chain.acceptance_rate(after_burn_in=10000) == pytest.approx(0.234, abs=0.08)


def test_covariance_proposal_mixes_faster_than_one_parameter_updates():
    from complybay.inference import effective_sample_size

    one = _ridge_chain("one")
    cov = _ridge_chain("covariance")
    ess_one = effective_sample_size(one.samples[None, :, 0])
    ess_cov = effective_sample_size(cov.samples[None, :, 0])
    # Along a 0.99 ridge an axis-aligned step must be ~sqrt(1 - 0.99^2) ~ 0.14 of the
    # marginal width, so one-parameter updates are several times slower.
    assert ess_cov > 3 * ess_one


def test_covariance_proposal_is_frozen_after_burn_in_and_reproducible():
    config = MetropolisConfig(
        3000,
        [0.5, 0.005],
        RIDGE_BOX,
        burn_in=1000,
        update="covariance",
        adapt=True,
        covariance_start=400,
        global_jump_interval=20,
    )
    a = run_chain(LogProbTarget(ridge), [0.0, 0.0], config, seed=3)
    b = run_chain(LogProbTarget(ridge), [0.0, 0.0], config, seed=3)
    np.testing.assert_array_equal(a.samples, b.samples)
    np.testing.assert_array_equal(a.proposal_covariance, b.proposal_covariance)
    assert np.all(a.samples >= RIDGE_BOX[0]) and np.all(a.samples <= RIDGE_BOX[1])


def test_covariance_configuration_errors():
    with pytest.raises(ValueError, match="adapt=True"):
        MetropolisConfig(100, 1.0, BOX, burn_in=50, update="covariance")
    with pytest.raises(ValueError, match="covariance_start < burn_in"):
        MetropolisConfig(
            100, 1.0, BOX, burn_in=50, update="covariance", adapt=True, covariance_start=60
        )
    with pytest.raises(ValueError, match="'covariance' only"):
        MetropolisConfig(100, 1.0, BOX, burn_in=50, covariance_start=10)


# --- parallel tempering and lagging chains ------------------------------------------


@dataclass(frozen=True)
class DiagonalModes:
    """Flat prior; two modes (weights 0.3, 0.7) that differ in both parameters at once."""

    def log_prior_and_likelihood(self, theta):
        low = -0.5 * np.sum(((theta + 3) / 0.2) ** 2) + np.log(0.3)
        high = -0.5 * np.sum(((theta - 3) / 0.2) ** 2) + np.log(0.7)
        return 0.0, float(np.logaddexp(low, high))


def test_parallel_tempering_recovers_the_mode_weights():
    # Independent check: the exact mode weights. Leaving a mode needs both parameters
    # to move together, so one-parameter global jumps fail (as for basement and Vp).
    box = (np.full(2, -6.0), np.full(2, 6.0))
    common = dict(burn_in=4000, adapt=True, update="covariance", global_jump_interval=20)
    starts = np.full((4, 2), -3.0)
    plain = run_metropolis(
        DiagonalModes(), starts, MetropolisConfig(20000, 0.2, box, **common), seed=11
    )
    assert np.all(plain.samples[..., 0] < 0)
    ladder = geometric_temperatures(7, 300.0)  # hottest: barrier of 225 in log L down to < 1
    config = MetropolisConfig(20000, 0.2, box, temperatures=ladder, **common)
    tempered = run_metropolis(DiagonalModes(), starts, config, seed=11)
    high = tempered.samples[..., 0] > 0
    # 4 x 1600 draws with a few hundred mode switches: standard error of the weight ~0.02.
    assert np.mean(high) == pytest.approx(0.7, abs=0.06)
    np.testing.assert_allclose(tempered.samples[high].std(axis=0), 0.2, rtol=0.1)
    for chain in tempered.chains:
        assert chain.temperature == 1.0 and chain.swap_acceptance.shape == (6,)
        assert np.all(chain.swap_acceptance > 0.3)
    assert plain.chains[0].swap_acceptance is None


def test_tempering_configuration():
    np.testing.assert_allclose(geometric_temperatures(3, 9.0), [1.0, 3.0, 9.0])
    assert MetropolisConfig(100, 1.0, BOX).ladder == (1.0,)
    assert MetropolisConfig(100, 1.0, BOX, temperatures=[1, 2.0]).ladder == (1.0, 2.0)
    for ladder in ([2.0, 4.0], [1.0, 1.0], [1.0]):
        with pytest.raises(ValueError, match="increase from 1"):
            MetropolisConfig(100, 1.0, BOX, temperatures=ladder)
    with pytest.raises(ValueError, match="not both"):
        MetropolisConfig(100, 1.0, BOX, temperatures=[1.0, 2.0], temperature=2.0)
    with pytest.raises(ValueError, match="swap_interval"):
        MetropolisConfig(100, 1.0, BOX, temperatures=[1.0, 2.0], swap_interval=0)
    with pytest.raises(ValueError, match="n_temperatures"):
        geometric_temperatures(1, 10.0)


def test_lagging_chains_flags_a_chain_stuck_in_a_lower_mode():
    rng = np.random.default_rng(12)
    lp = -150 + 2 * rng.standard_normal((4, 500))  # 8 parameters: spread ~ sqrt(8 / 2)
    assert lagging_chains(lp) == ()
    lp[2] -= 380  # the CZP case of docs/synthetic_tests.md
    lp[3] -= 3  # shifted, but overlapping: not flagged
    assert lagging_chains(lp) == (2,)


@dataclass(frozen=True)
class GaussianWithDetails:
    """Flat prior, unit Gaussian likelihood, and the per-parameter chi2 as details."""

    detail_names = ("chi2.a", "chi2.b")

    def log_prior_and_likelihood(self, theta):
        return 0.0, float(-0.5 * np.sum(theta**2))

    def log_prior_likelihood_and_details(self, theta):
        return 0.0, float(-0.5 * np.sum(theta**2)), theta**2


@pytest.mark.parametrize("ladder", [None, (1.0, 3.0, 9.0)])
def test_details_follow_the_stored_states(ladder):
    # The details travel with each state, through accepts and swaps alike.
    box = (np.full(2, -10.0), np.full(2, 10.0))
    config = MetropolisConfig(3000, 0.5, box, burn_in=500, thin=7, temperatures=ladder)
    result = run_metropolis(GaussianWithDetails(), np.ones((2, 2)), config, seed=13)
    assert result.detail_names == ("chi2.a", "chi2.b")
    for chain in result.chains:
        np.testing.assert_array_equal(chain.details, chain.samples**2)
        np.testing.assert_allclose(
            chain.trace_log_posterior, -0.5 * chain.trace_details.sum(axis=1)
        )
        assert chain.trace_steps[0] == 0 and np.all(np.diff(chain.trace_steps) == 7)
        # The trace and the kept draws record the same states where their steps meet.
        common = np.nonzero(np.isin(chain.trace_steps, 500 + 7 * np.arange(len(chain.samples))))[0]
        kept = (chain.trace_steps[common] - 500) // 7
        np.testing.assert_array_equal(chain.trace_details[common], chain.details[kept])
    plain = run_chain(
        GaussianLikelihood(), [0.0], MetropolisConfig(50, 1.0, (np.full(1, -5.0), np.full(1, 5.0)))
    )
    assert plain.details.shape == (50, 0)
