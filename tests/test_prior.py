import copy

import numpy as np
import pytest

from complybay.parameterization import (
    BoundaryNonDecreasingVs,
    ConstraintSuite,
    DiscretizationConfig,
    GeophysicalPrior,
    MaxVsGradient,
    MonotonicVs,
    VpAboveWater,
    VpVsRange,
    VsRange,
    constraints_from_config,
    parameterization_from_config,
    sample_prior,
)

DISCRETIZATION = DiscretizationConfig(
    layers_per_segment={"sediment": 10, "crust": 10, "mantle": 10}, transition_layers=6
)
NAMES = ["basement.depth", "basement.width", "sed.vs_top", "sed.vs_bottom", "sed.vp"]


def _config():
    return {
        "Model": {
            "water": {"depth_km": 2.0},
            "zmax_km": 100.0,
            "interfaces": [
                {
                    "name": "basement",
                    "depth_km": {"parameter": "basement.depth"},
                    "width_km": {"parameter": "basement.width"},
                },
                {"name": "moho", "depth_km": 7.0},
            ],
            "segments": [
                {
                    "name": "sediment",
                    "vs": {
                        "type": "gradient",
                        "top": {"parameter": "sed.vs_top"},
                        "bottom": {"parameter": "sed.vs_bottom"},
                    },
                    "vp": {"type": "constant", "value": {"parameter": "sed.vp"}},
                    "scaling": {"rho": {"method": "gardner1974"}},
                },
                {"name": "crust", "vs": {"type": "gradient", "top": 3.0, "bottom": 3.9}},
                {
                    "name": "mantle",
                    "vs": {"type": "constant", "value": 4.5},
                    "scaling": {
                        "vp": {"method": "constant_vp_vs", "vp_vs": 1.8},
                        "rho": {"method": "constant", "rho": 3.35},
                    },
                },
            ],
        },
        "Inversion": {
            "parameters": [
                {"name": "basement.depth", "initial": 0.6, "prior": {"bounds": [0.2, 1.5]}},
                {
                    "name": "basement.width",
                    "initial": 0.0,
                    "prior": {"type": "half_normal", "scale": 0.05, "bounds": [0.0, 0.15]},
                },
                {"name": "sed.vs_top", "initial": 0.15, "prior": {"bounds": [0.05, 0.6]}},
                {"name": "sed.vs_bottom", "initial": 0.6, "prior": {"bounds": [0.2, 1.5]}},
                {"name": "sed.vp", "initial": 1.75, "prior": {"bounds": [1.3, 2.6]}},
            ]
        },
    }


CONSTRAINTS = constraints_from_config(
    [
        {"type": "vp_vs_range", "bounds": [1.6, 15.0]},
        {"type": "vs_range", "bounds": [0.05, 1.5], "segments": ["sediment"]},
        {"type": "boundary_non_decreasing_vs"},
        {"type": "vp_above_water"},
    ]
)


def _prior(constraints=CONSTRAINTS, config=None):
    p = parameterization_from_config(config or _config())
    return GeophysicalPrior(p, DISCRETIZATION, constraints)


def _theta(**values):
    base = dict(zip(NAMES, [0.6, 0.0, 0.15, 0.6, 1.75], strict=True))
    base.update({k.replace("__", "."): v for k, v in values.items()})
    return np.array([base[n] for n in NAMES])


def test_valid_model_keeps_the_built_obs_model():
    result = _prior().evaluate(_theta())
    assert result.success and result.log_prior == 0.0
    assert result.obs_model.water_depth == 2.0
    assert result.obs_model.solid.n_layers == 30


@pytest.mark.parametrize(
    ("theta", "reason"),
    [
        (_theta(sed__vp=2.7), "parameter_prior"),  # outside its bounds
        (
            _theta(basement__depth=0.2, basement__width=0.15),
            "model_mapping",
        ),  # support hits seafloor
        (_theta(sed__vp=1.4), "physical_constraints: vp_above_water"),
        (_theta(sed__vs_bottom=1.2, sed__vp=1.85), "physical_constraints: vp_vs_range"),  # 1.54
    ],
)
def test_invalid_vectors_get_minus_infinity_and_a_reason(theta, reason):
    result = _prior().evaluate(theta)
    assert result.log_prior == -np.inf and not result.success
    assert result.error.startswith(reason)


def test_negative_bulk_modulus_is_rejected_as_model_mapping():
    # Vp/Vs below sqrt(4/3): LayeredModel refuses it; no vp_vs_range constraint here.
    result = _prior(ConstraintSuite()).evaluate(_theta(sed__vs_bottom=1.5, sed__vp=1.6))
    assert result.error.startswith("model_mapping") and "sqrt(4/3)" in result.error


def test_segment_constraints_follow_a_moving_interface():
    # Sediment Vs <= 1.5 must not apply to the crust, wherever the basement is.
    prior = _prior(ConstraintSuite((VsRange((0.05, 1.5), ("sediment",)),)))
    for depth in (0.3, 0.9, 1.4):
        assert prior.evaluate(_theta(basement__depth=depth)).success


def test_profile_constraints():
    model = parameterization_from_config(_config()).vector_to_model(_theta())
    assert MonotonicVs().check(model, None)
    assert BoundaryNonDecreasingVs().check(model, None)
    assert not MaxVsGradient(0.5, ("sediment",)).check(model, None)  # 0.75 km/s per km
    assert MaxVsGradient(1.0, ("sediment",)).check(model, None)
    assert VpVsRange((1.7, 12.0), ("sediment",)).check(model, None)
    decreasing = parameterization_from_config(_config()).vector_to_model(
        _theta(sed__vs_top=0.5, sed__vs_bottom=0.3)
    )
    assert not MonotonicVs(("sediment",)).check(decreasing, None)


def test_configuration_errors():
    with pytest.raises(ValueError, match="Unknown constraint"):
        constraints_from_config([{"type": "vp_below_water"}])
    with pytest.raises(ValueError, match="unknown segments"):
        _prior(ConstraintSuite((VpAboveWater(segments=("sediments",)),)))


def test_prior_samples_are_valid_independent_and_reproducible():
    prior = _prior()
    samples = sample_prior(prior, 300, rng=1)
    assert samples.theta.shape == (300, 5)
    assert all(prior.evaluate(t).success for t in samples.theta[:50])
    assert 0 < samples.acceptance < 1
    assert set(samples.failures) <= {"model_mapping", "physical_constraints"}
    np.testing.assert_array_equal(
        sample_prior(prior, 20, rng=7).theta, sample_prior(prior, 20, rng=7).theta
    )
    # Half-normal width prior: most mass near a hard step, never outside bounds.
    width = samples.theta[:, 1]
    assert width.min() >= 0 and width.max() <= 0.15 and np.median(width) < 0.05
    # Water-velocity floor removes sediment Vp below 1.5 km/s from the [1.3, 2.6] box.
    assert samples.theta[:, 4].min() >= 1.5


def test_contradictory_prior_fails_loudly():
    config = copy.deepcopy(_config())
    config["Inversion"]["parameters"][4]["prior"]["bounds"] = [1.3, 1.45]
    config["Inversion"]["parameters"][4]["initial"] = 1.4
    with pytest.raises(RuntimeError, match="physical_constraints"):
        sample_prior(_prior(config=config), 5, rng=0, max_attempts=200)
