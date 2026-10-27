from dataclasses import dataclass

import numpy as np
import pytest
from test_inference import _setup
from test_prior import _prior

from complybay.inference import (
    LogProbTarget,
    MetropolisConfig,
    PosteriorTarget,
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
