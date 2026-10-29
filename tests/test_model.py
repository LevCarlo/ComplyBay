import numpy as np
import pytest

from complybay.model import LayeredModel, OBSModel, WaterLayer


def _solid():
    return LayeredModel(
        thickness=[0.5, 2.0], vp=[1.8, 4.5, 6.5], vs=[0.3, 2.5, 3.7], rho=[1.8, 2.6, 2.9]
    )


def test_shapes_and_tops():
    solid = _solid()
    assert solid.n_layers == 2
    np.testing.assert_allclose(solid.tops, [0.0, 0.5, 2.5])


def test_arrays_are_read_only():
    solid = _solid()
    with pytest.raises(ValueError):
        solid.vs[0] = 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thickness": [0.5], "vp": [1.8, 4.5, 6.5], "vs": [0.3, 2.5, 3.7], "rho": [1.8, 2.6, 2.9]},
        {"thickness": [0.5, 0.0], "vp": [1.8, 4.5, 6.5], "vs": [0.3, 2.5, 3.7], "rho": [1, 2, 3]},
        {"thickness": [0.5, 2.0], "vp": [1.5, 4.5, 6.5], "vs": [0.0, 2.5, 3.7], "rho": [1, 2, 3]},
        {"thickness": [0.5, 2.0], "vp": [0.3, 4.5, 6.5], "vs": [0.3, 2.5, 3.7], "rho": [1, 2, 3]},
    ],
)
def test_invalid_models_raise(kwargs):
    with pytest.raises(ValueError):
        LayeredModel(**kwargs)


def test_water_layer_validation():
    with pytest.raises(ValueError):
        WaterLayer(0.0)
    assert OBSModel(_solid(), WaterLayer(4.0)).water_depth == 4.0
    assert OBSModel(_solid()).water_depth == 0.0


def test_legacy_array_round_trip():
    legacy = np.array([[500.0, 1.8, 1.8, 0.3], [2000.0, 2.6, 4.5, 2.5], [0.0, 2.9, 6.5, 3.7]])
    solid = LayeredModel.from_legacy_array(legacy)
    np.testing.assert_allclose(solid.thickness, [0.5, 2.0])
    np.testing.assert_allclose(solid.rho, legacy[:, 1])
    np.testing.assert_allclose(solid.vs, legacy[:, 3])


def test_scaled_returns_copy():
    solid = _solid()
    scaled = solid.scaled("vs", 1, 1.1)
    assert scaled.vs[1] == pytest.approx(2.75)
    assert solid.vs[1] == 2.5


def test_resampled_preserves_depths_and_values():
    solid = _solid().resampled(0.3)
    assert np.all(solid.thickness <= 0.3 + 1e-12)
    assert solid.tops[-1] == pytest.approx(2.5)
    assert solid.vs[-1] == 3.7
    assert set(solid.vs[:-1]) == {0.3, 2.5}


def test_attenuation_fields():
    base = dict(thickness=[0.5], vp=[1.8, 6.0], vs=[0.3, 3.5], rho=[1.8, 2.7])
    elastic = LayeredModel(**base)
    assert not elastic.attenuating and np.all(np.isinf(elastic.q("qs")))
    model = LayeredModel(**base, qs=[20.0, np.inf])
    assert model.attenuating and model.q("qp").tolist() == [np.inf, np.inf]
    assert not LayeredModel(**base, qs=[np.inf, np.inf]).attenuating
    resampled = model.resampled(0.1)
    assert resampled.qs.tolist() == [20.0] * 5 + [np.inf] and resampled.qp is None
    for bad in ([20.0], [0.0, np.inf], [np.nan, 1.0]):
        with pytest.raises(ValueError, match="qs needs"):
            LayeredModel(**base, qs=bad)
    with pytest.raises(ValueError, match="q_reference_frequency"):
        LayeredModel(**base, qs=[20.0, np.inf], q_reference_frequency=0.0)
