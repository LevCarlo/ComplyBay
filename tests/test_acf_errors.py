import dataclasses

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from complybay.acf import (
    acf_from_trace,
    preset,
    processing_error_correlation,
    shrink_correlation,
    stack_errors,
)
from complybay.inference import ErrorCorrelation
from complybay.io import read_acf_errors, write_acf_errors

DT = 0.05


def _noise_acfs(processing, n_traces, seed=0, n=1400):
    rng = np.random.default_rng(seed)
    return np.array(
        [
            acf_from_trace(rng.standard_normal(n), DT, processing, taper_fraction=0.2)[1]
            for _ in range(n_traces)
        ]
    )


def _ar1(p, phi):
    i = np.arange(p)
    return phi ** np.abs(i[:, None] - i[None, :])


# --- theory against simulation ----------------------------------------------------


@pytest.mark.parametrize(
    "processing",
    [
        preset("kim2023"),  # whitening, Gaussian + high-pass, zero-lag normalization
        dataclasses.replace(preset("kim2023"), normalize="none", whitening=None),
    ],
    ids=["kim2023", "filters-only"],
)
def test_processing_correlation_matches_simulated_noise(processing):
    # Independent check: push white-noise traces through the real processing.
    acfs = _noise_acfs(processing, 3000)
    lags = np.arange(acfs.shape[1]) * DT
    fit = (lags >= 0.3) & (lags <= 8.0)
    simulated = np.corrcoef(acfs[:, fit].T)
    theory = processing_error_correlation(processing, lags[fit], DT, expected=acfs.mean(axis=0))
    # 3000 traces: standard error of a correlation ~0.018; the max over ~12000 pairs
    # lies near 4.5 standard errors.
    assert np.max(np.abs(simulated - theory)) < 0.1
    assert np.sqrt(np.mean((simulated - theory) ** 2)) < 0.03
    # About 20 independent pieces of information in 155 lags for kim2023.
    n_eff = np.trace(theory) ** 2 / np.sum(theory**2)
    assert n_eff < 0.3 * fit.sum()


def test_processing_correlation_input_errors():
    kim = preset("kim2023")
    lags = np.arange(6, 20) * DT
    with pytest.raises(ValueError, match="expected"):
        processing_error_correlation(kim, lags, DT)
    expected = np.zeros(601)
    expected[0] = 1.0
    with pytest.raises(ValueError, match="vanishes"):
        processing_error_correlation(kim, np.arange(0, 20) * DT, DT, expected=expected)
    with pytest.raises(ValueError, match="multiples"):
        processing_error_correlation(kim, lags + 0.01, DT, expected=expected)


# --- estimation from events ---------------------------------------------------------


def test_shrinkage_recovers_a_known_correlation():
    rng = np.random.default_rng(1)
    truth = _ar1(40, 0.8)
    chol = np.linalg.cholesky(truth)
    many = (chol @ rng.standard_normal((40, 5000))).T
    estimate, lam = shrink_correlation(many, np.eye(40))
    assert lam < 0.02 and np.max(np.abs(estimate - truth)) < 0.06
    few = (chol @ rng.standard_normal((40, 20))).T  # fewer samples than variables
    estimate, lam = shrink_correlation(few, truth)
    assert lam > 0.3  # the right target gets a large weight
    assert np.all(np.diag(estimate) == 1.0) and np.all(np.linalg.eigvalsh(estimate) > 0)


def test_stack_errors_keep_the_cross_component_correlation(tmp_path):
    rng = np.random.default_rng(2)
    p, n_events = 30, 4000
    within = _ar1(p, 0.7)
    joint = np.block([[within, 0.5 * within], [0.5 * within, within]])
    errors = (np.linalg.cholesky(joint) @ rng.standard_normal((2 * p, n_events))).T
    mean = np.linspace(1.0, 0.0, p)
    events = {"vertical": mean + 0.1 * errors[:, :p], "pressure": mean + 0.1 * errors[:, p:]}
    result = stack_errors(events, DT, (0.0, 10.0))
    assert result.components == ("vertical", "pressure") and result.n_events == n_events
    np.testing.assert_allclose(result.sigma["vertical"], 0.1 / np.sqrt(n_events), rtol=0.05)
    np.testing.assert_allclose(result.correlation, joint, atol=0.06)
    block = result.block(["pressure"], result.window_lags[5:10])
    np.testing.assert_allclose(block, result.correlation[p + 5 : p + 10, p + 5 : p + 10])

    path = tmp_path / "acf_errors.nc"
    write_acf_errors(result, path)
    back = read_acf_errors(path)
    assert back.components == result.components and back.n_events == n_events
    np.testing.assert_array_equal(back.correlation, result.correlation)
    np.testing.assert_array_equal(back.stack["pressure"], result.stack["pressure"])


# --- the correlated likelihood -------------------------------------------------------


def test_identity_correlation_gives_the_independent_likelihood():
    rng = np.random.default_rng(3)
    z = rng.standard_normal(50)
    block = ErrorCorrelation(("a", "b"), np.eye(50), variance_fraction=1.0)
    scales = np.r_[np.full(20, 1.3), np.full(30, 0.8)]
    log_l, chi2 = block.log_likelihood(z, scales)
    independent = -0.5 * np.sum((z / scales) ** 2) - 20 * np.log(1.3) - 30 * np.log(0.8)
    assert log_l == pytest.approx(independent) and chi2 == pytest.approx(np.sum((z / scales) ** 2))


def test_full_rank_block_matches_the_multivariate_normal():
    # Independent implementation: scipy's Gaussian with covariance D R D, compared
    # between two settings of the noise scales (constants cancel).
    rng = np.random.default_rng(4)
    corr = np.block([[_ar1(15, 0.6), 0.3 * _ar1(15, 0.6)], [0.3 * _ar1(15, 0.6), _ar1(15, 0.6)]])
    block = ErrorCorrelation(("z", "p"), corr, variance_fraction=1.0)
    z = rng.standard_normal(30)

    def both(h_z, h_p):
        scales = np.r_[np.full(15, h_z), np.full(15, h_p)]
        cov = np.outer(scales, scales) * corr
        return block.log_likelihood(z, scales)[0], multivariate_normal(cov=cov).logpdf(z)

    (a, ref_a), (b, ref_b) = both(1.0, 1.0), both(1.4, 0.7)
    assert a - b == pytest.approx(ref_a - ref_b, rel=1e-8)
    # The cached pieces follow a change of grouping (three distinct scales).
    scales = np.r_[np.full(10, 0.9), np.full(5, 1.2), np.full(15, 0.6)]
    cov = np.outer(scales, scales) * corr
    c = block.log_likelihood(z, scales)[0]
    assert a - c == pytest.approx(ref_a - multivariate_normal(cov=cov).logpdf(z), rel=1e-8)


def test_reduced_rank_block_is_calibrated_on_processing_errors():
    # Residuals drawn from the near-singular kim2023 correlation: keeping 99 % of
    # the variance gives chi2 ~ chi2_k, so h stays near 1.
    lags = np.arange(6, 161) * DT
    expected = np.exp(-np.arange(601) * DT)
    corr = processing_error_correlation(preset("kim2023"), lags, DT, expected=expected)
    block = ErrorCorrelation(("acf",), corr)
    assert block.k < 0.4 * lags.size
    values, vectors = np.linalg.eigh(corr)
    draw = vectors * np.sqrt(np.clip(values, 0, None))
    rng = np.random.default_rng(5)
    chi2 = [
        block.log_likelihood(draw @ rng.standard_normal(lags.size), np.ones(lags.size))[1]
        for _ in range(400)
    ]
    # mean of chi2_k / k is 1 with standard error sqrt(2 / k / 400) ~ 0.013
    assert np.mean(chi2) / block.k == pytest.approx(1.0, abs=0.06)


def test_error_correlation_validation():
    with pytest.raises(ValueError, match="unit diagonal"):
        ErrorCorrelation(("a",), 2 * np.eye(3))
    with pytest.raises(ValueError, match="distinct"):
        ErrorCorrelation(("a", "a"), np.eye(3))
