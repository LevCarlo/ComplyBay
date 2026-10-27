from pathlib import Path

import numpy as np
import pytest

from complybay.forward.compliance import (
    G_DEFAULT,
    compliance,
    compliance_kernel,
    halfspace_compliance,
    water_wavenumber,
)
from complybay.model import LayeredModel, OBSModel, WaterLayer

BASELINE = np.load(Path(__file__).parent / "data" / "legacy_v0_1_forward.npz")
FREQS = BASELINE["freqs"]
MODEL_NAMES = sorted(k.removeprefix("model__") for k in BASELINE.files if k.startswith("model__"))


def _obs(name, wdepth_m):
    solid = LayeredModel.from_legacy_array(BASELINE[f"model__{name}"])
    return OBSModel(solid, WaterLayer(wdepth_m * 1e-3))


@pytest.mark.parametrize("depth_m", [50.0, 779.0, 3000.0, 6000.0])
def test_exact_wavenumber_solves_dispersion_relation(depth_m):
    omega = 2 * np.pi * np.geomspace(1e-4, 0.5, 200)
    k = water_wavenumber(omega, depth_m)
    np.testing.assert_allclose(G_DEFAULT * k * np.tanh(k * depth_m), omega**2, rtol=1e-12)


def test_wavenumber_limits():
    depth = 1000.0
    low = 2 * np.pi * 1e-5
    assert water_wavenumber(low, depth)[0] == pytest.approx(low / np.sqrt(G_DEFAULT * depth), 1e-6)
    high = 2 * np.pi * 1.0
    assert water_wavenumber(high, depth)[0] == pytest.approx(high**2 / G_DEFAULT, 1e-12)
    assert water_wavenumber([0.0, high], depth)[0] == 0.0


@pytest.mark.parametrize("wdepth", BASELINE["water_depths"])
def test_wavenumbers_match_legacy(wdepth):
    omega = 2 * np.pi * FREQS
    poly = water_wavenumber(omega, wdepth, method="poly")
    np.testing.assert_allclose(poly, BASELINE[f"k__h{int(wdepth)}__poly"], rtol=1e-12)
    # Legacy Brent stops at an absolute tolerance, ~1e-9 relative here.
    exact = water_wavenumber(omega, wdepth)
    np.testing.assert_allclose(exact, BASELINE[f"k__h{int(wdepth)}__brent"], rtol=1e-7)


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("wdepth", BASELINE["water_depths"])
def test_compliance_matches_legacy(name, wdepth):
    tag = f"{name}__h{int(wdepth)}"
    model = _obs(name, wdepth)
    np.testing.assert_allclose(
        compliance(model, FREQS, wavenumber="poly"),
        BASELINE[f"ncomp_batch__{tag}__poly"],
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        compliance(model, FREQS), BASELINE[f"ncomp_batch__{tag}__brent"], rtol=1e-7
    )


def test_halfspace_approaches_static_solution():
    vp, vs, rho = 4.0, 2.0, 2.5
    model = OBSModel(LayeredModel([], [vp], [vs], [rho]), WaterLayer(3.0))
    freqs = np.array([1e-3, 5e-3, 2e-2])
    ratio = compliance(model, freqs) / halfspace_compliance(vp, vs, rho)
    # The dynamic correction is of order (c / vs)^2 with c the gravity-wave phase speed.
    c = 2 * np.pi * freqs / water_wavenumber(2 * np.pi * freqs, 3000.0)
    assert np.all(np.abs(ratio - 1) < 2 * (c / (vs * 1e3)) ** 2)
    assert np.all(np.abs(ratio - 1) < 1e-2)


def test_compliance_requires_water():
    with pytest.raises(ValueError):
        compliance(OBSModel(LayeredModel([], [4.0], [2.0], [2.5])), [0.01])


def test_kernel_is_central_difference_and_leaves_model_alone():
    model = _obs("sediment_gradient", 779.0)
    freqs = np.array([0.005, 0.02])
    vs_before = model.solid.vs.copy()
    kernel = compliance_kernel(model, freqs, "vs", rel_step=0.01)
    np.testing.assert_array_equal(model.solid.vs, vs_before)
    assert kernel.shape == (2, model.solid.n_layers + 1)

    layer = 4
    plus = model.with_solid(model.solid.scaled("vs", layer, 1.01))
    minus = model.with_solid(model.solid.scaled("vs", layer, 0.99))
    expected = (compliance(plus, freqs) - compliance(minus, freqs)) / (
        0.02 * compliance(model, freqs)
    )
    np.testing.assert_allclose(kernel[:, layer], expected, rtol=1e-12)
    # Softer shallow layers raise compliance: kernels for vs are negative.
    assert np.all(kernel[:, :5] < 0)


def test_deep_model_is_truncated_instead_of_overflowing():
    """A 150 km model overflowed the propagator at high frequency before truncation."""
    deep = LayeredModel(
        thickness=np.r_[np.full(6, 0.1), 2.0, 4.4, np.full(20, 7.0)],
        vp=np.r_[np.linspace(1.6, 2.2, 6), 5.2, 6.7, np.full(20, 8.1), 8.2],
        vs=np.r_[np.linspace(0.15, 0.6, 6), 2.9, 3.75, np.full(20, 4.5), 4.5],
        rho=np.r_[np.linspace(1.6, 2.0, 6), 2.6, 2.9, np.full(20, 3.35), 3.4],
    )
    model = OBSModel(deep, WaterLayer(2.0))
    freqs = np.array([0.003, 0.01, 0.025, 0.03])
    eta = compliance(model, freqs)
    assert np.all(np.isfinite(eta)) and np.all(eta > 0)
    # A model ending at 28 km gives the same result where the cut lies above 28 km.
    shallow = OBSModel(
        LayeredModel(deep.thickness[:11], deep.vp[:12], deep.vs[:12], deep.rho[:12]),
        WaterLayer(2.0),
    )
    np.testing.assert_allclose(compliance(shallow, freqs[2:]), eta[2:], rtol=1e-10)
    # Keeping more layers (larger cutoff) does not change the result: truncation error is tiny.
    np.testing.assert_allclose(compliance(model, freqs[:2], decay_cutoff=40.0), eta[:2], rtol=1e-9)
