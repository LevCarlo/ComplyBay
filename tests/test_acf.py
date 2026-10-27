import dataclasses

import numpy as np
import pytest
from scipy.fft import irfft, next_fast_len, rfftfreq

from complybay.acf import (
    ACFProcessing,
    Bandpass,
    GaussianFilter,
    Highpass,
    Lowpass,
    Whitening,
    acf_from_spectrum,
    acf_from_trace,
    preset,
    stack_acfs,
)
from complybay.forward.acf import synthetic_acf, synthetic_acfs, wraparound_fraction
from complybay.forward.response import seafloor_response
from complybay.model import LayeredModel, OBSModel, WaterLayer

SOLID = LayeredModel(
    thickness=[0.4, 1.5, 5.0],
    vp=[1.8, 4.5, 6.5, 8.0],
    vs=[0.3, 2.5, 3.7, 4.5],
    rho=[1.8, 2.6, 2.9, 3.3],
)
PROCESSING = ACFProcessing(
    max_lag=12.0, whitening=Whitening(0.5, 0.01), filters=(Bandpass(0.5, 4.0),), zero_lag_taper=0.5
)
FREQS = np.linspace(0.01, 5.0, 300)


def _qa(v, p):
    return np.sqrt(1 / v**2 - p**2)


def _strongest(lags, acf, after):
    keep = lags > after
    i = np.argmax(np.abs(acf[keep]))
    return lags[keep][i], acf[keep][i]


# --- seafloor response ---------------------------------------------------------


def test_land_halfspace_doubles_vertical_motion():
    r = seafloor_response(OBSModel(LayeredModel([], [6.0], [3.5], [2.8])), 0.0, FREQS)
    np.testing.assert_allclose(r.displacement_z, 2.0)
    np.testing.assert_allclose(r.displacement_x, 0.0, atol=1e-14)
    np.testing.assert_allclose(r.pressure, 0.0)


@pytest.mark.parametrize("water", [None, WaterLayer(4.0)])
@pytest.mark.parametrize("ray_p", [0.0, 0.04, 0.07])
def test_energy_flux_is_conserved(water, ray_p):
    """No energy leaves through the top, so reflected P + S flux equals incident flux."""
    r = seafloor_response(OBSModel(SOLID, water), ray_p, FREQS)
    vp, vs, rho = SOLID.vp[-1], SOLID.vs[-1], SOLID.rho[-1]
    incident = rho * vp**2 * _qa(vp, ray_p)
    reflected = rho * vp**2 * _qa(vp, ray_p) * np.abs(r.reflected_p) ** 2
    reflected += rho * vs**2 * _qa(vs, ray_p) * np.abs(r.reflected_s) ** 2
    np.testing.assert_allclose(reflected / incident, 1.0, rtol=1e-10)


def _acoustic_chain(model, omega):
    """Normal incidence by an independent real 2x2 (u_z, sigma_zz) propagator."""

    def prop(h, rho, vp, w):
        k, z = w / vp, w * rho * vp
        return np.array([[np.cos(k * h), np.sin(k * h) / z], [-z * np.sin(k * h), np.cos(k * h)]])

    solid, water = model.solid, model.water
    out = []
    for w in omega:
        a = np.eye(2)
        for h, rho, vp in zip(solid.thickness, solid.rho, solid.vp, strict=False):
            a = prop(h, rho, vp, w) @ a
        pw = prop(-water.thickness, water.rho, water.vp, w)
        stress_per_u = -pw[1, 0] / pw[1, 1]
        u, s = a @ np.array([1.0, stress_per_u])
        upgoing = (u - s / (1j * w * solid.rho[-1] * solid.vp[-1])) / 2
        scale = -1.0 / upgoing  # unit upward incident motion (z is down)
        out.append((-scale, -scale * stress_per_u))  # (u_z up, pressure)
    return np.array(out)


def test_normal_incidence_matches_acoustic_chain():
    model = OBSModel(SOLID, WaterLayer(4.0))
    r = seafloor_response(model, 0.0, FREQS)
    ref = _acoustic_chain(model, 2 * np.pi * FREQS)
    np.testing.assert_allclose(r.displacement_z, ref[:, 0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(r.pressure, ref[:, 1], rtol=1e-9, atol=1e-9)


# --- synthetic ACF ----------------------------------------------------------------


def test_sediment_reverberation_time():
    h, vp, p = 1.0, 2.5, 0.06
    model = OBSModel(LayeredModel([h], [vp, 6.0], [0.8, 3.5], [2.0, 2.7]))
    lags, acf = synthetic_acf(model, p, PROCESSING)
    lag, value = _strongest(lags, acf, 0.6)
    assert lag == pytest.approx(2 * h * _qa(vp, p), abs=0.05)
    assert value < 0  # Claerbout: ACF ~ delta - reflection response


def test_water_reverberation_has_opposite_sign_on_pressure():
    hw, p = 3.0, 0.06
    model = OBSModel(LayeredModel([], [6.0], [3.5], [2.7]), WaterLayer(hw))
    expected = 2 * hw * _qa(1.5, p)
    signs = {}
    for component in ("vertical", "pressure"):
        lags, acf = synthetic_acf(model, p, PROCESSING, component=component)
        lag, value = _strongest(lags, acf, 0.6)
        assert lag == pytest.approx(expected, abs=0.05)
        signs[component] = np.sign(value)
    assert signs["vertical"] == -signs["pressure"]


def test_pressure_acf_needs_water():
    with pytest.raises(ValueError):
        synthetic_acf(OBSModel(SOLID), 0.06, PROCESSING, component="pressure")


def test_trace_and_spectrum_paths_agree():
    """A recorded impulse response processed as data gives the synthetic ACF."""
    model = OBSModel(SOLID, WaterLayer(1.0))
    dt, duration, p = 0.05, 200.0, 0.06
    n_fft = next_fast_len(int(round(duration / dt)), real=True)
    spectrum = seafloor_response(model, p, rfftfreq(n_fft, dt)).component("vertical")
    trace = irfft(np.conj(spectrum), n=n_fft)  # exp(-i omega t) convention -> numpy's
    _, from_trace = acf_from_trace(trace, dt, PROCESSING, taper_fraction=0.001)
    _, synthetic = synthetic_acf(model, p, PROCESSING, dt=dt, duration=duration)
    np.testing.assert_allclose(from_trace, synthetic, atol=0.03)


# --- processing -------------------------------------------------------------------


def test_zero_lag_taper_and_normalization():
    lags, acf = acf_from_trace(np.random.default_rng(0).standard_normal(4000), 0.05, PROCESSING)
    assert acf[0] == 0.0
    assert np.max(np.abs(acf)) == pytest.approx(1.0)
    assert lags[-1] == pytest.approx(PROCESSING.max_lag)


def test_preset_lookup():
    assert preset("tauzin2019").whitening is not None
    with pytest.raises(ValueError):
        preset("unknown")


class _Product:
    def __init__(self, *filters):
        self.filters = filters

    def power_response(self, freqs):
        return np.prod([f.power_response(freqs) for f in self.filters], axis=0)


def test_filters_chain_multiplies_responses():
    rng = np.random.default_rng(2)
    n_fft, dt = 4000, 0.05
    spectrum = rng.standard_normal(n_fft // 2 + 1) + 1j * rng.standard_normal(n_fft // 2 + 1)
    chain = (GaussianFilter(5.0), Highpass(0.5), Lowpass(3.0))
    chained = ACFProcessing(max_lag=20, filters=chain)
    single = ACFProcessing(max_lag=20, filters=_Product(*chain))
    np.testing.assert_allclose(
        acf_from_spectrum(spectrum, dt, n_fft, chained)[1],
        acf_from_spectrum(spectrum, dt, n_fft, single)[1],
        rtol=1e-12,
        atol=1e-15,
    )
    # Band-pass is the product of its high- and low-pass parts.
    f = np.linspace(0.01, 5, 200)
    np.testing.assert_allclose(
        Bandpass(0.5, 3.0).power_response(f),
        Highpass(0.5).power_response(f) * Lowpass(3.0).power_response(f),
    )


def test_filters_accepts_single_filter_and_rejects_others():
    assert ACFProcessing(filters=Highpass(0.5)).filters == (Highpass(0.5),)
    assert ACFProcessing(filters=None).filters == ()
    with pytest.raises(TypeError):
        ACFProcessing(filters=(0.5,))


def test_kim2023_preset():
    kim = preset("kim2023")
    assert kim.normalize == "zero_lag" and kim.zero_lag_taper == 0.0
    f = np.linspace(0.01, 4.0, 4000)
    response = np.prod([flt.power_response(f) for flt in kim.filters], axis=0)
    band = f[response >= response.max() / 2]
    assert band[0] == pytest.approx(0.55, abs=0.02) and band[-1] == pytest.approx(1.72, abs=0.02)
    model = OBSModel(SOLID, WaterLayer(2.0))
    for component in ("vertical", "pressure"):
        _, acf = synthetic_acf(model, 0.06, kim, component=component)
        assert acf[0] == pytest.approx(1.0)
        assert np.max(np.abs(acf[1:])) < 1.0


def test_phase_weighted_stack_suppresses_incoherent_noise():
    rng = np.random.default_rng(1)
    signal = np.zeros(400)
    signal[100] = 1.0
    acfs = signal + 0.3 * rng.standard_normal((50, 400))
    linear = stack_acfs(acfs)
    pws = stack_acfs(acfs, method="pws")
    noise = np.r_[0:90, 110:400]
    assert np.std(pws[noise]) < np.std(linear[noise])
    assert np.argmax(pws) == 100


# --- performance-related behavior -------------------------------------------------


def test_shared_response_matches_single_component_calls():
    model = OBSModel(SOLID, WaterLayer(2.0))
    lags, acfs = synthetic_acfs(model, 0.06, PROCESSING)
    for component in ("vertical", "pressure"):
        lags1, acf = synthetic_acf(model, 0.06, PROCESSING, component=component)
        np.testing.assert_array_equal(lags, lags1)
        np.testing.assert_array_equal(acfs[component], acf)


@pytest.mark.parametrize("name", ["kim2023", "tauzin2019"])
def test_band_limit_changes_acf_negligibly(name):
    """Dropping frequencies the filters suppress below 1e-6 in power leaves the ACF unchanged."""
    model = OBSModel(SOLID, WaterLayer(2.0))
    limited = preset(name)
    full = dataclasses.replace(limited, band_tolerance=0.0)
    assert np.isinf(full.frequency_limit(np.linspace(0, 10, 100)))
    for component in ("vertical", "pressure"):
        _, a = synthetic_acf(model, 0.06, limited, component=component)
        _, b = synthetic_acf(model, 0.06, full, component=component)
        np.testing.assert_allclose(a, b, atol=1e-5)


def test_frequency_limit_and_validation():
    freqs = np.linspace(0, 10, 2001)
    kim = preset("kim2023")
    limit = kim.frequency_limit(freqs)
    assert 5.0 < limit < 7.5  # Gaussian a = 5 power falls below 1e-6 near 5.9 Hz, plus 0.4 Hz
    assert np.isinf(ACFProcessing(filters=()).frequency_limit(freqs))
    with pytest.raises(ValueError):
        ACFProcessing(band_tolerance=1.0)


def test_wraparound_fraction_shrinks_with_duration():
    model = OBSModel(SOLID, WaterLayer(2.0))
    kim = preset("kim2023")
    short = wraparound_fraction(model, 0.06, kim, duration=70.0)
    long = wraparound_fraction(model, 0.06, kim, duration=200.0)
    assert 0 <= long < short
    assert long < 1e-3


# --- Kim et al. (2023) Supporting Text S2 / S4 ---------------------------------


@pytest.mark.parametrize("average", ["power", "amplitude"])
def test_whitening_uses_2n_plus_1_point_running_mean(average):
    rng = np.random.default_rng(3)
    amplitude = rng.uniform(0.5, 2.0, 400)
    df, width = 0.01, 0.2  # N = 10, window of 21 samples
    w = Whitening(width=width, water_level=0.0, average=average)
    values = amplitude**2 if average == "power" else amplitude
    i = 200
    expected = values[i] / values[i - 10 : i + 11].mean()
    if average == "amplitude":
        expected = expected**2
    got = w.whitened_power(amplitude, df, np.ones(400, bool))
    assert got[i] == pytest.approx(expected, rel=1e-12)


def test_kim2023_follows_supporting_text_s2():
    kim = preset("kim2023")
    assert kim.whitening == Whitening(width=0.8, water_level=0.0, average="power")
    assert kim.filters == (GaussianFilter(5.0), Highpass(0.5, corners=2))


def test_snr_noise_lowers_acf_amplitude():
    model = OBSModel(SOLID, WaterLayer(2.0))
    kim = preset("kim2023")
    lags, clean = synthetic_acf(model, 0.06, kim)
    _, almost_clean = synthetic_acf(model, 0.06, kim, snr=1e6)
    np.testing.assert_allclose(almost_clean, clean, atol=1e-5)
    late = lags > 0.3
    peaks = [np.abs(synthetic_acf(model, 0.06, kim, snr=s)[1][late]).max() for s in (10, 3, 1.5)]
    assert np.abs(clean[late]).max() > peaks[0] > peaks[1] > peaks[2]
    with pytest.raises(ValueError):
        synthetic_acf(model, 0.06, kim, snr=1.0)
