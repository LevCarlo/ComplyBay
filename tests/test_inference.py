import copy
from unittest import mock

import numpy as np
import pytest
from test_prior import CONSTRAINTS, DISCRETIZATION, _config

import complybay.inference.likelihood as likelihood_module
from complybay.acf import preset
from complybay.forward import compliance, phase_velocity, synthetic_acfs
from complybay.inference import (
    ACFObservation,
    ComplianceObservation,
    DispersionObservation,
    JointLikelihood,
    PosteriorTarget,
    predict,
)
from complybay.parameterization import GeophysicalPrior, parameterization_from_config

KIM = preset("kim2023")
NOISE = ["noise.compliance", "noise.dispersion", "noise.acf_z", "noise.acf_p"]
FREQS = np.geomspace(0.004, 0.025, 12)
PERIODS = np.linspace(8.0, 30.0, 8)
WINDOW = (0.3, 8.0)


def _setup(theta_true=None):
    """Prior with 5 model + 4 noise parameters and noise-free data from ``theta_true``."""
    config = copy.deepcopy(_config())
    config["Inversion"]["parameters"] += [
        {"name": n, "initial": 0.0, "prior": {"bounds": [-0.5, 0.5]}} for n in NOISE
    ]
    p = parameterization_from_config(config)
    prior = GeophysicalPrior(p, DISCRETIZATION, CONSTRAINTS)
    theta_true = p.theta0() if theta_true is None else np.asarray(theta_true)
    truth = prior.evaluate(theta_true).obs_model
    _, acfs = synthetic_acfs(truth, 0.06, KIM)
    observations = (
        ComplianceObservation(
            FREQS,
            compliance(truth, FREQS),
            0.1 * compliance(truth, FREQS),
            noise_parameter=NOISE[0],
        ),
        DispersionObservation(
            PERIODS, phase_velocity(truth, PERIODS), 0.02, noise_parameter=NOISE[1]
        ),
        ACFObservation(
            "vertical", 0.06, KIM, acfs["vertical"], 0.05, WINDOW, noise_parameter=NOISE[2]
        ),
        ACFObservation(
            "pressure", 0.06, KIM, acfs["pressure"], 0.05, WINDOW, noise_parameter=NOISE[3]
        ),
    )
    return PosteriorTarget(prior, JointLikelihood(observations)), theta_true, truth


def test_predictions_match_the_forward_solvers():
    target, theta, truth = _setup()
    obs = target.likelihood.observations
    pred = predict(obs, truth)
    np.testing.assert_array_equal(pred[0], compliance(truth, FREQS))
    np.testing.assert_array_equal(pred[1], phase_velocity(truth, PERIODS))
    _, acfs = synthetic_acfs(truth, 0.06, KIM)
    np.testing.assert_array_equal(pred[2], acfs["vertical"])
    np.testing.assert_array_equal(pred[3], acfs["pressure"])


def test_vertical_and_pressure_share_one_response():
    target, theta, truth = _setup()
    obs = list(target.likelihood.observations)
    with mock.patch.object(likelihood_module, "synthetic_acfs", wraps=synthetic_acfs) as spy:
        predict(obs, truth)
        assert spy.call_count == 1
        assert spy.call_args.args[3] == ("vertical", "pressure")
        other = ACFObservation("vertical", 0.07, KIM, obs[2].values, 0.05, WINDOW, name="acf_z_p07")
        predict([*obs, other], truth)
        assert spy.call_count == 3  # one more call each for p = 0.06 and p = 0.07


def test_noise_free_data_are_fitted_exactly_at_the_truth():
    target, theta, _ = _setup()
    result = target.evaluate(theta)
    assert result.error is None
    assert result.log_likelihood == pytest.approx(0.0, abs=1e-9)
    assert all(g.chi2 < 1e-12 for g in result.likelihood.groups)
    worse = theta.copy()
    worse[0] += 0.05  # sediment 50 m thicker
    assert target(worse) < target(theta) - 1.0


def test_noise_scale_term_and_its_optimum():
    """log L_k = -chi2 / (2 h^2) - N ln h, largest at h^2 = chi2 / N."""
    target, theta, truth = _setup()
    obs = target.likelihood.observations[1]
    k = 1.7  # residuals of exactly 1.7 sigma
    shifted = DispersionObservation(
        obs.periods, obs.values + k * obs.sigma, obs.sigma, noise_parameter="h"
    )
    lik = JointLikelihood((shifted,))
    n = shifted.periods.size
    for log10_h in (-0.3, 0.0, 0.4):
        h = 10**log10_h
        value = lik.evaluate(truth, {"h": log10_h}).log_likelihood
        assert value == pytest.approx(-0.5 * n * k**2 / h**2 - n * np.log(h), rel=1e-9)
    grid = np.linspace(-0.5, 0.8, 2601)
    best = grid[np.argmax([lik.evaluate(truth, {"h": g}).log_likelihood for g in grid])]
    assert 10**best == pytest.approx(k, rel=2e-3)


def test_prior_failures_skip_the_forward_calculation():
    target, theta, _ = _setup()
    bad = theta.copy()
    bad[4] = 1.4  # sediment Vp below water: vp_above_water
    with mock.patch.object(likelihood_module, "predict", side_effect=AssertionError("called")):
        result = target.evaluate(bad)
    assert result.log_posterior == -np.inf
    assert result.error.startswith("physical_constraints")


def test_prior_only_target_ignores_the_data():
    target, theta, _ = _setup()
    prior_only = PosteriorTarget(target.prior, target.likelihood, prior_only=True)
    with mock.patch.object(likelihood_module, "predict", side_effect=AssertionError("called")):
        assert prior_only(theta) == target.prior.log_prior(theta)


def test_forward_failure_gives_minus_infinity_with_reason():
    target, theta, truth = _setup()
    no_root = DispersionObservation(
        [100.0], [4.0], 0.02, name="mode1", mode=1
    )  # no 1st higher mode
    result = JointLikelihood((no_root,)).evaluate(truth)
    assert result.log_likelihood == -np.inf and "non-finite mode1" in result.error


def test_configuration_errors():
    target, theta, _ = _setup()
    obs = target.likelihood.observations
    with pytest.raises(ValueError, match="unique"):
        JointLikelihood((obs[0], obs[0]))
    missing = ComplianceObservation(
        FREQS, obs[0].values, obs[0].sigma, noise_parameter="noise.typo"
    )
    with pytest.raises(ValueError, match="not defined"):
        PosteriorTarget(target.prior, JointLikelihood((missing,)))
    with pytest.raises(ValueError, match="samples"):
        ACFObservation("vertical", 0.06, KIM, obs[2].values[:-1], 0.05, WINDOW)
    with pytest.raises(ValueError, match="window"):
        ACFObservation("vertical", 0.06, KIM, obs[2].values, 0.05, (40.0, 50.0))
    assert obs[2].mask.sum() == np.count_nonzero((obs[2].lags >= 0.3) & (obs[2].lags <= 8.0))
