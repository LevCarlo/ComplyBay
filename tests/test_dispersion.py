import numpy as np
import pytest
from scipy.optimize import brentq

from complybay.forward.dispersion import disba_arrays, phase_velocity, phase_velocity_kernel
from complybay.model import LayeredModel, OBSModel, WaterLayer

OCEANIC = LayeredModel(
    thickness=[0.5, 6.5, 53.0],
    vp=[2.0, 6.5, 8.1, 7.9],
    vs=[0.5, 3.6, 4.55, 4.3],
    rho=[1.8, 2.9, 3.35, 3.35],
)


def _scholte_speed(water_vp, water_rho, vp, vs, rho):
    """Interface-wave speed of a fluid half-space over a solid half-space."""

    def f(c):
        qa = np.sqrt(1 - c**2 / vp**2)
        qb = np.sqrt(1 - c**2 / vs**2)
        rayleigh = (2 - c**2 / vs**2) ** 2 - 4 * qa * qb
        return rayleigh + (water_rho / rho) * (c**4 / vs**4) * qa / np.sqrt(1 - c**2 / water_vp**2)

    return brentq(f, 1e-3 * vs, min(vs, water_vp) * (1 - 1e-12))


@pytest.mark.parametrize(("vp", "vs", "rho"), [(5.2, 3.0, 2.7), (1.8, 0.4, 1.8)])
def test_short_period_limit_is_scholte_wave(vp, vs, rho):
    """Validates disba's fluid layer: with deep water mode 0 becomes the Scholte wave."""
    water = WaterLayer(30.0)
    model = OBSModel(LayeredModel([], [vp], [vs], [rho]), water)
    expected = _scholte_speed(water.vp, water.rho, vp, vs, rho)
    np.testing.assert_allclose(phase_velocity(model, [0.5, 1.0]), expected, rtol=1e-5)


def test_thin_water_converges_to_solid_only():
    periods = [20.0, 30.0]
    wet = phase_velocity(OBSModel(OCEANIC, WaterLayer(0.01)), periods)
    dry = phase_velocity(OBSModel(OCEANIC), periods)
    np.testing.assert_allclose(wet, dry, rtol=1e-3)


def test_water_slows_short_periods():
    periods = np.array([8.0, 12.0, 20.0, 30.0])
    deep = phase_velocity(OBSModel(OCEANIC, WaterLayer(4.0)), periods)
    dry = phase_velocity(OBSModel(OCEANIC), periods)
    assert np.all(np.diff(deep) > 0)
    slowdown = 1 - deep / dry
    assert slowdown[0] > 0.3 and slowdown[-1] < 0.02


def test_output_follows_input_order_and_marks_missing_roots():
    model = OBSModel(OCEANIC, WaterLayer(4.0))
    periods = np.array([30.0, 8.0, 15.0])
    np.testing.assert_allclose(
        phase_velocity(model, periods), phase_velocity(model, [8, 15, 30])[[2, 0, 1]]
    )
    # The first higher mode has no root at long periods.
    assert np.isnan(phase_velocity(model, [100.0], mode=1)[0])


def test_disba_arrays_put_water_first():
    thickness, vp, vs, rho = disba_arrays(OBSModel(OCEANIC, WaterLayer(4.0)))
    assert thickness[0] == 4.0 and vs[0] == 0.0 and vp[0] == 1.5
    assert thickness.size == OCEANIC.n_layers + 2


def test_kernel_drops_water_layer():
    model = OBSModel(OCEANIC.resampled(2.0), WaterLayer(4.0))
    kernel = phase_velocity_kernel(model, 20.0)
    assert kernel.shape == (model.solid.n_layers + 1,)
    # At 20 s the Vs sensitivity peaks in the upper mantle, not in the sediment.
    tops = model.solid.tops
    assert 10.0 < tops[np.argmax(kernel[:-1])] < 40.0
