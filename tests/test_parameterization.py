"""Segment/interface parameterization; adapted from SeisForge tests/test_inv_interfaces.py."""

import copy

import numpy as np
import pytest

from complybay.acf import preset
from complybay.forward import compliance, phase_velocity, synthetic_acfs
from complybay.model import LayeredModel
from complybay.parameterization import (
    ConstantProfile,
    DiscretizationConfig,
    GradientProfile,
    HalfNormalPrior,
    Interface,
    NormalPrior,
    ParameterizedModel,
    ScalingConfig,
    Segment,
    discretize,
    layer_geometry,
    layering_report,
    model_from_config,
    parameterization_from_config,
    prior_from_config,
    transition_half_support_km,
    transition_weight,
)

MANTLE = ScalingConfig(
    vp="constant_vp_vs", vp_kwargs={"vp_vs": 1.8}, rho="constant", rho_kwargs={"rho": 3.35}
)
SEDIMENT = ScalingConfig(rho="gardner1974")


def _model(*, basement=1.0, moho=8.0, basement_width=0.0, moho_width=0.0, shape="sigmoid"):
    return ParameterizedModel(
        segments=(
            Segment(
                0.0, basement, GradientProfile(0.2, 0.9), ConstantProfile(3.0), "sediment", SEDIMENT
            ),
            Segment(basement, moho, GradientProfile(3.0, 3.9), name="crust"),
            Segment(moho, 80.0, GradientProfile(4.4, 4.6), name="mantle", scaling=MANTLE),
        ),
        interfaces=(
            Interface(basement, basement_width, shape, "basement"),
            Interface(moho, moho_width, shape, "moho"),
        ),
    )


# --- transitions ------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["sigmoid", "smootherstep"])
def test_transition_weight_is_a_compact_monotone_step(shape):
    depth, width = 10.0, 2.0
    half = transition_half_support_km(width, shape)
    z = np.linspace(depth - 2 * half, depth + 2 * half, 2001)
    weight = transition_weight(z, depth, width, shape)
    assert transition_weight(np.array([depth]), depth, width, shape)[0] == pytest.approx(0.5)
    assert transition_weight(np.array([depth - width / 2]), depth, width, shape)[
        0
    ] == pytest.approx(0.1, abs=1e-3)
    assert transition_weight(np.array([depth + width / 2]), depth, width, shape)[
        0
    ] == pytest.approx(0.9, abs=1e-3)
    assert np.all(weight[z <= depth - half] == 0.0)
    assert np.all(weight[z >= depth + half] == 1.0)
    assert np.all(np.diff(weight) >= 0)


def test_interface_validation():
    with pytest.raises(ValueError, match="depth"):
        Interface(0.0)
    with pytest.raises(ValueError, match="width"):
        Interface(1.0, width_km=-0.1)
    with pytest.raises(ValueError, match="shape"):
        Interface(1.0, shape="erf")


@pytest.mark.parametrize("shape", ["sigmoid", "smootherstep"])
def test_gradational_interface_blends_smoothly_and_matches_hard_outside(shape):
    model = _model(moho_width=2.0, shape=shape)
    top, bottom = model.interfaces[1].support_km
    hard = _model()
    outside = np.r_[np.linspace(1.5, top, 50), np.linspace(bottom, 79.0, 50)]
    np.testing.assert_allclose(model.evaluate(outside), hard.evaluate(outside))
    z = np.linspace(top - 0.5, bottom + 0.5, 4001)
    vs = model.evaluate(z)
    assert np.all(np.diff(vs) > 0)
    assert np.max(np.abs(np.diff(vs))) < 0.01


def test_transition_supports_may_not_overlap_seafloor_neighbours_or_bottom():
    with pytest.raises(ValueError, match="seafloor"):
        _model(basement_width=0.8)
    with pytest.raises(ValueError, match="previous interface"):
        _model(basement=5.0, moho=6.0, basement_width=0.5, moho_width=0.5)
    with pytest.raises(ValueError, match="model bottom"):
        _model(moho=75.0, moho_width=4.0)


# --- properties ---------------------------------------------------------------------


def test_sampled_vp_vs_and_density_from_vp():
    model = _model()
    vs, vp, rho = model.evaluate_properties(np.array([0.5, 4.0, 40.0]))
    assert vp[0] == pytest.approx(3.0 * vs[0])  # sediment: sampled Vp/Vs profile
    assert rho[0] == pytest.approx(1.741 * vp[0] ** 0.25)  # Gardner on that Vp
    assert vp[2] == pytest.approx(1.8 * vs[2]) and rho[2] == pytest.approx(3.35)


def test_relation_errors_name_the_segment():
    model = ParameterizedModel(
        segments=(
            Segment(0.0, 5.0, ConstantProfile(3.5), name="crust"),
            Segment(5.0, 50.0, ConstantProfile(4.6), name="mantle"),
        )
    )
    with pytest.raises(ValueError, match="mantle: Brocher"):
        model.evaluate_properties(np.array([20.0]))


def test_gradational_vp_vs_stays_between_neighbours():
    model = _model(basement=1.0, basement_width=0.4, moho_width=0.0)
    top, bottom = model.interfaces[0].support_km
    z = np.linspace(top, bottom, 200)
    vs, vp, _ = model.evaluate_properties(z)
    ratio = vp / vs
    _, vp_crust, _ = model.segments[1].properties(z, ScalingConfig())
    assert np.all(ratio <= 3.0 + 1e-9)
    assert np.all(
        ratio >= np.min(vp_crust / model.segments[1].properties(z, ScalingConfig())[0]) - 1e-9
    )


# --- layering -----------------------------------------------------------------------


def test_hard_interfaces_are_exact_layer_boundaries():
    model = _model(basement=0.737, moho=7.31)
    layered = discretize(model, DiscretizationConfig(dz=0.5))
    report = layering_report(model, layered)
    assert report.hard_interfaces_preserved
    assert np.any(layered.tops == 0.737) and np.any(layered.tops == 7.31)


def test_transition_support_gets_the_configured_layers():
    model = _model(moho_width=1.5)
    layered = discretize(model, DiscretizationConfig(dz=2.0, transition_layers=12))
    assert layering_report(model, layered).transition_layer_counts == (12,)


def test_fixed_layer_counts_keep_geometry_continuous_in_interface_depth():
    config = DiscretizationConfig(layers_per_segment={"sediment": 6, "crust": 20, "mantle": 8})
    a = layer_geometry(_model(basement=1.000), config)
    b = layer_geometry(_model(basement=1.001), config)
    assert a.tops_km.size == b.tops_km.size
    assert np.max(np.abs(a.tops_km - b.tops_km)) <= 1.001e-3
    by_dz = DiscretizationConfig(dz=0.25)
    assert layer_geometry(_model(basement=1.0), by_dz).tops_km.size != (
        layer_geometry(_model(basement=1.04), by_dz).tops_km.size
    )


def test_layering_report_converges_and_detects_straddled_interfaces():
    model = _model(moho_width=2.0)
    coarse = layering_report(model, discretize(model, DiscretizationConfig(dz=2.0)))
    fine = layering_report(model, discretize(model, DiscretizationConfig(dz=0.05)))
    assert abs(fine.vs_vertical_time_relative_error) < abs(coarse.vs_vertical_time_relative_error)
    assert abs(fine.vs_vertical_time_relative_error) < 1e-4
    straddled = LayeredModel(
        thickness=[0.75, 0.5, 6.75],
        vs=[0.5, 2.0, 3.5, 4.5],
        vp=[1.8, 3.8, 6.0, 8.1],
        rho=[1.9, 2.2, 2.7, 3.35],
    )
    report = layering_report(model, straddled)
    assert not report.hard_interfaces_preserved
    assert report.max_interface_offset_km == pytest.approx(0.25)


def test_layer_mean_equals_midpoint_for_linear_profiles():
    model = _model()
    a = discretize(model, DiscretizationConfig(dz=0.5))
    b = discretize(model, DiscretizationConfig(dz=0.5, layer_value="mean"))
    np.testing.assert_allclose(b.vs, a.vs, rtol=1e-12)


def test_halfspace_takes_bottom_properties():
    model = _model()
    layered = discretize(model, DiscretizationConfig(dz=5.0, zmax=60.0))
    assert layered.tops[-1] == pytest.approx(60.0)
    assert layered.vs[-1] == pytest.approx(model.evaluate(np.array([60.0]))[0])


# --- config and theta mapping -------------------------------------------------------


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
                    "shape": "sigmoid",
                },
                {"name": "moho", "depth_km": 7.0},
            ],
            "segments": [
                {
                    "name": "sediment",
                    "vs": {"type": "gradient", "top": {"parameter": "sed.vs_top"}, "bottom": 0.8},
                    "vp_vs": {"type": "constant", "value": {"parameter": "sed.vp_vs"}},
                    "scaling": {"rho": {"method": "gardner1974"}},
                },
                {
                    "name": "crust",
                    "vs": {"type": "bspline", "coefficients": [3.0, 3.4, 3.8], "degree": 2},
                },
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
                {"name": "basement.depth", "initial": 0.5, "prior": {"bounds": [0.2, 1.5]}},
                {
                    "name": "basement.width",
                    "initial": 0.0,
                    "prior": {"type": "half_normal", "scale": 0.05, "bounds": [0.0, 0.2]},
                },
                {"name": "sed.vs_top", "initial": 0.2, "prior": {"bounds": [0.05, 0.5]}},
                {
                    "name": "sed.vp_vs",
                    "initial": 4.0,
                    "prior": {"bounds": [1.8, 12.0]},
                    "proposal": {"sigma": 0.2},
                },
            ]
        },
    }


def test_config_maps_theta_to_depths_widths_and_vp_vs():
    p = parameterization_from_config(_config())
    assert p.water.thickness == 2.0 and p.water.vp == 1.5
    model = p.vector_to_model([0.6, 0.1, 0.3, 5.0])
    np.testing.assert_allclose(model.interface_depths(), [0.6, 7.0])
    assert model.interfaces[0].width_km == 0.1
    vs, vp, _ = model.evaluate_properties(np.array([0.0]))
    assert vs[0] == pytest.approx(0.3) and vp[0] == pytest.approx(1.5)
    sigmas = p.proposal_sigmas()
    assert sigmas[3] == 0.2 and sigmas[0] == pytest.approx(0.05 * 1.3)


def test_theta0_is_checked_and_references_must_exist():
    config = _config()
    config["Inversion"]["parameters"][0]["initial"] = 6.9  # basement 6.9 km, transition 0
    config["Inversion"]["parameters"][0]["prior"]["bounds"] = [0.2, 7.5]
    config["Inversion"]["parameters"][1]["initial"] = 0.2
    config["Inversion"]["parameters"][1]["prior"]["bounds"] = [0.0, 0.3]
    with pytest.raises(ValueError, match="Initial parameter values"):
        parameterization_from_config(config)  # support 6.5-7.3 km overlaps the 7 km Moho
    config = _config()
    config["Model"]["segments"][0]["vs"]["bottom"] = {"parameter": "missing"}
    with pytest.raises(ValueError, match="undefined parameters"):
        parameterization_from_config(config)


def test_layout_errors_are_explicit():
    model_cfg = copy.deepcopy(_config()["Model"])
    model_cfg["segments"][0]["top_km"] = 0.0
    with pytest.raises(ValueError, match="remove top_km"):
        model_from_config(
            model_cfg,
            {"basement.depth": 0.5, "basement.width": 0.0, "sed.vs_top": 0.2, "sed.vp_vs": 4},
        )
    model_cfg = copy.deepcopy(_config()["Model"])
    del model_cfg["zmax_km"]
    with pytest.raises(ValueError, match="zmax_km"):
        model_from_config(model_cfg)
    model_cfg = copy.deepcopy(_config()["Model"])
    model_cfg["interfaces"][0]["depth_km"] = 8.0
    model_cfg["interfaces"][0]["width_km"] = 0.0
    with pytest.raises(ValueError, match="increase downward"):
        model_from_config(model_cfg, {"sed.vs_top": 0.2, "sed.vp_vs": 4})


def test_priors():
    normal = prior_from_config({"type": "normal", "mean": 35.0, "sigma": 2.0, "bounds": [28, 45]})
    assert isinstance(normal, NormalPrior) and normal.logpdf(37.0) == pytest.approx(-0.5)
    assert normal.logpdf(50.0) == -np.inf
    half = prior_from_config({"type": "half_normal", "scale": 2.0, "bounds": [0, 8]})
    assert isinstance(half, HalfNormalPrior) and half.logpdf(2.0) == pytest.approx(-0.5)
    with pytest.raises(ValueError, match="bounds"):
        prior_from_config({"type": "normal", "mean": 1.0, "sigma": 1.0})
    p = parameterization_from_config(_config())
    assert p.log_parameter_prior(p.theta0()) == 0.0
    assert p.log_parameter_prior([0.5, 0.0, 0.2, 13.0]) == -np.inf


def test_theta_to_obs_model_runs_every_forward_solver():
    p = parameterization_from_config(_config())
    config = DiscretizationConfig(
        layers_per_segment={"sediment": 8, "crust": 10, "mantle": 12}, transition_layers=8
    )
    obs = p.vector_to_obs_model([0.5, 0.05, 0.2, 4.0], config)
    assert obs.water_depth == 2.0
    assert obs.solid.tops[-1] == pytest.approx(100.0)
    assert np.all(np.isfinite(compliance(obs, [0.005, 0.01, 0.02])))
    assert np.all(np.isfinite(phase_velocity(obs, [8.0, 15.0, 30.0])))
    _, acfs = synthetic_acfs(obs, 0.06, preset("kim2023"))
    assert acfs["vertical"][0] == pytest.approx(1.0) and acfs["pressure"][0] == pytest.approx(1.0)


def test_direct_vp_profile_for_sediment():
    seg = Segment(
        0.0, 0.6, GradientProfile(0.15, 0.6), vp=GradientProfile(1.52, 2.0), name="sediment"
    )
    model = ParameterizedModel(
        segments=(seg, Segment(0.6, 10.0, ConstantProfile(3.5), name="crust"))
    )
    vs, vp, rho = model.evaluate_properties(np.array([0.0, 0.3, 0.6 - 1e-9]))
    np.testing.assert_allclose(vp, [1.52, 1.76, 2.0])
    np.testing.assert_allclose(vp / vs, [1.52 / 0.15, 1.76 / 0.375, 2.0 / 0.6], rtol=1e-6)
    with pytest.raises(ValueError, match="not both"):
        Segment(0.0, 1.0, ConstantProfile(0.3), ConstantProfile(5.0), vp=ConstantProfile(1.6))
    config = _config()
    sediment = config["Model"]["segments"][0]
    del sediment["vp_vs"]
    sediment["vp"] = {"type": "constant", "value": {"parameter": "sed.vp"}}
    config["Inversion"]["parameters"][3] = {
        "name": "sed.vp",
        "initial": 1.6,
        "prior": {"bounds": [1.5, 2.5]},
    }
    p = parameterization_from_config(config)
    _, vp, _ = p.vector_to_model([0.5, 0.0, 0.2, 1.7]).evaluate_properties(np.array([0.1]))
    assert vp[0] == pytest.approx(1.7)


# --- attenuation ------------------------------------------------------------------


def test_sampled_sediment_q_reaches_the_layers():
    config = _config()
    config["Model"]["q_reference_frequency_hz"] = 2.0
    sediment = config["Model"]["segments"][0]
    sediment["qs"] = {"type": "constant", "value": {"parameter": "sed.qs"}}
    sediment["qp"] = {"type": "constant", "value": 80.0}
    config["Inversion"]["parameters"].append(
        {"name": "sed.qs", "initial": 20.0, "prior": {"bounds": [5.0, 200.0]}}
    )
    p = parameterization_from_config(config)
    disc = DiscretizationConfig(
        layers_per_segment={"sediment": 6, "crust": 8, "mantle": 8}, transition_layers=6
    )
    solid = p.vector_to_obs_model([0.5, 0.1, 0.2, 4.0, 25.0], disc).solid
    assert solid.attenuating and solid.q_reference_frequency == 2.0
    mid = 0.5 * (solid.tops[:-1] + solid.tops[1:])
    half = transition_half_support_km(0.1, "sigmoid")
    inside = mid < 0.5 - half  # above the basement transition
    below = mid > 0.5 + half
    np.testing.assert_allclose(solid.qs[:-1][inside], 25.0)
    np.testing.assert_allclose(solid.qp[:-1][inside], 80.0)
    assert np.all(np.isinf(solid.qs[:-1][below])) and np.isinf(solid.qs[-1])
    # Inside the transition 1/Q blends towards the elastic crust: Q grows monotonically.
    across = ~inside & ~below
    assert np.all(np.diff(solid.qs[:-1][across]) > 0) and np.all(solid.qs[:-1][across] > 25.0)
    elastic = parameterization_from_config(_config()).vector_to_obs_model(
        [0.5, 0.1, 0.2, 4.0], disc
    )
    assert elastic.solid.qs is None and not elastic.solid.attenuating
    np.testing.assert_array_equal(elastic.solid.vs, solid.vs)
