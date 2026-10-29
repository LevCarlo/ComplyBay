import numpy as np
import pytest

from complybay.forward.attenuation import (
    complex_velocity,
    dispersed,
    kjartansson_gamma,
    phase_velocity_at,
)
from complybay.model import LayeredModel

FREQS = np.geomspace(1e-3, 10.0, 50)


@pytest.mark.parametrize("q", [5.0, 20.0, 300.0])
def test_constant_q_identities(q):
    v = complex_velocity(0.4, q, FREQS, 1.0)
    modulus = v**2  # M / rho
    # Q = Re M / -Im M at every frequency (definition; exp(-i omega t) convention).
    np.testing.assert_allclose(modulus.real / -modulus.imag, q, rtol=1e-12)
    # Phase velocity omega / Re k with k = omega / v: v_ref at f_ref, power law elsewhere.
    phase = 1.0 / (1.0 / v).real
    np.testing.assert_allclose(phase, 0.4 * FREQS ** kjartansson_gamma(q), rtol=1e-12)
    assert phase[np.argmin(np.abs(FREQS - 1.0))] == pytest.approx(0.4, rel=0.2)
    np.testing.assert_allclose(phase_velocity_at(0.4, q, 1.0, 1.0), 0.4)
    # Amplitude decay per unit distance, Im k, against the high-Q formula omega / (2 Q c).
    omega = 2 * np.pi * FREQS
    decay = (omega / v).imag
    assert np.all(decay > 0)
    np.testing.assert_allclose(decay, omega / (2 * q * phase), rtol=1.5 / q**2 + 1e-12)


def test_elastic_limit_and_validation():
    np.testing.assert_array_equal(complex_velocity(2.0, np.inf, FREQS, 1.0), 2.0)
    assert kjartansson_gamma(np.inf) == 0.0
    with pytest.raises(ValueError):
        kjartansson_gamma(0.0)
    with pytest.raises(ValueError):
        complex_velocity(2.0, 10.0, [0.0, 1.0], 1.0)


def test_dispersed_model():
    solid = LayeredModel(
        [0.5], [1.8, 6.0], [0.3, 3.5], [1.8, 2.7], qp=[100.0, np.inf], qs=[20.0, np.inf]
    )
    low = dispersed(solid, 0.01)
    assert not low.attenuating and low.qs is None
    # About ln(f / f_ref) / (pi Q) slower: 7 % for Qs = 20 two decades below 1 Hz.
    assert low.vs[0] / solid.vs[0] == pytest.approx(1 + np.log(0.01) / (np.pi * 20), abs=3e-3)
    assert low.vs[1] == solid.vs[1] and low.vp[1] == solid.vp[1]
    assert dispersed(solid, 1.0).vs[0] == solid.vs[0]
