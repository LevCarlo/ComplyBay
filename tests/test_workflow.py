import copy
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from complybay.acf import Highpass, Whitening, preset
from complybay.cli import main
from complybay.forward import synthetic_acfs
from complybay.inference.diagnostics import effective_sample_size, split_rhat
from complybay.inference.ensemble import derived_quantities
from complybay.io import dump_yaml, load_yaml, read_three_columns, write_three_columns
from complybay.parameterization import (
    ConstantProfile,
    GradientProfile,
    ParameterizedModel,
    Segment,
)
from complybay.workflow import build_inversion, run_inversion
from complybay.workflow.config import processing_from_config

EXAMPLE = Path(__file__).parents[1] / "examples" / "synthetic"


def _quick_model(tmp_path, **sampler):
    cfg = load_yaml(EXAMPLE / "model.yaml")
    cfg["Sampler"].update(
        {"n_steps": 300, "burn_in": 100, "thin": 5, "n_chains": 2, "executor": "serial"}
    )
    cfg["Sampler"].update(sampler)
    cfg["Output"].update({"n_model_samples": 20, "n_predictive_samples": 5, "n_depths": 51})
    path = tmp_path / "model.yaml"
    dump_yaml(cfg, path)
    return path


# --- files and configuration ---------------------------------------------------


def test_three_column_files_round_trip_and_validation(tmp_path):
    path = tmp_path / "d.dat"
    write_three_columns(path, [1, 2, 3], [0.1, 0.2, 0.3], [0.01] * 3, header="x v s")
    data = read_three_columns(path)
    np.testing.assert_allclose(data.value, [0.1, 0.2, 0.3])
    write_three_columns(path, [1, 3, 2], [0.1, 0.2, 0.3], [0.01] * 3, header="x v s")
    with pytest.raises(ValueError, match="increase"):
        read_three_columns(path)
    write_three_columns(path, [1, 2, 3], [0.1, 0.2, 0.3], [0.01, 0.0, 0.01], header="x v s")
    with pytest.raises(ValueError, match="sigma"):
        read_three_columns(path)


def test_example_builds_with_station_water_and_noise_parameters():
    setup = build_inversion(
        load_yaml(EXAMPLE / "model.yaml"), load_yaml(EXAMPLE / "obs.yaml"), obs_dir=EXAMPLE
    )
    assert setup.station == "SYN01" and setup.water.thickness == 2.0
    assert setup.names[:4] == ("basement.depth", "sed.vs_top", "sed.vs_bottom", "sed.vp")
    assert setup.names[4:] == (
        "noise.compliance",
        "noise.dispersion",
        "noise.acf_vertical",
        "noise.acf_pressure",
    )
    acf = setup.target.likelihood.observations[2]
    assert acf.dt == pytest.approx(0.05) and acf.processing.max_lag == pytest.approx(30.0)
    assert acf.processing.whitening == preset("kim2023").whitening
    assert setup.metropolis_config().global_jump_interval == 20


def test_configuration_errors_are_explicit():
    model, obs = load_yaml(EXAMPLE / "model.yaml"), load_yaml(EXAMPLE / "obs.yaml")
    bad = copy.deepcopy(model)
    bad["Model"]["water"] = {"depth_km": 2.0}
    with pytest.raises(ValueError, match="belongs to the station"):
        build_inversion(bad, obs, obs_dir=EXAMPLE)
    bad = copy.deepcopy(model)
    bad["Samplr"] = {}
    with pytest.raises(ValueError, match="unknown keys"):
        build_inversion(bad)
    bad = copy.deepcopy(obs)
    bad["Observations"]["dispersion"]["moed"] = 0
    with pytest.raises(ValueError, match="Observations.dispersion: unknown keys"):
        build_inversion(model, bad, obs_dir=EXAMPLE)
    bad = copy.deepcopy(obs)
    del bad["Observations"]["compliance"]["noise"]
    setup = build_inversion(model, bad, obs_dir=EXAMPLE)
    assert "noise.compliance" not in setup.names  # sigma used as given


def test_acf_error_correlation_forms():
    model, obs = load_yaml(EXAMPLE / "model.yaml"), load_yaml(EXAMPLE / "obs_events.yaml")
    joint = build_inversion(model, obs, obs_dir=EXAMPLE).target.likelihood
    (block,) = joint.correlations
    assert block.members == ("acf_vertical", "acf_pressure") and block.correlation.shape == (
        310,
        310,
    )
    assert block.k < 100  # far fewer independent pieces than 310 lags

    theory = copy.deepcopy(obs)
    theory["Observations"]["acf"]["error_correlation"] = "processing"
    blocks = build_inversion(model, theory, obs_dir=EXAMPLE).target.likelihood.correlations
    assert [b.members for b in blocks] == [("acf_vertical",), ("acf_pressure",)]

    none = copy.deepcopy(obs)
    del none["Observations"]["acf"]["error_correlation"]
    assert build_inversion(model, none, obs_dir=EXAMPLE).target.likelihood.correlations == ()

    mismatched = copy.deepcopy(obs)
    mismatched["Observations"]["acf"]["components"]["vertical"]["file"] = "acf_z.dat"
    with pytest.raises(ValueError, match="stack differs"):
        build_inversion(model, mismatched, obs_dir=EXAMPLE)
    bad = copy.deepcopy(obs)
    bad["Observations"]["acf"]["error_correlation"] = {"model": "processing", "file": "x.nc"}
    with pytest.raises(ValueError, match="exactly one"):
        build_inversion(model, bad, obs_dir=EXAMPLE)


def _acf_groups_obs(tmp_path, ray_ps, **group):
    """obs.yaml with one ACF group per ray parameter, data = synthetics of theta0."""
    setup = build_inversion(
        load_yaml(EXAMPLE / "model.yaml"), load_yaml(EXAMPLE / "obs.yaml"), obs_dir=EXAMPLE
    )
    model = setup.prior.evaluate(setup.parameterization.theta0()).obs_model
    template = next(o for o in setup.observations.observations if o.name == "acf_vertical")
    obs = load_yaml(EXAMPLE / "obs.yaml")
    del obs["Observations"]["compliance"], obs["Observations"]["dispersion"]
    groups = []
    for i, ray_p in enumerate(ray_ps):
        lags, acfs = synthetic_acfs(model, ray_p, template.processing, dt=template.dt)
        components = {}
        for component, values in acfs.items():
            path = tmp_path / f"acf_{i}_{component}.dat"
            write_three_columns(path, lags, values, np.full(lags.size, 0.01), header="synthetic")
            components[component] = {"file": str(path), "noise": True}
        groups.append(
            {"name": f"p{i}", "processing": "kim2023", "ray_p": ray_p, "window": [0.3, 8.0]}
            | {"components": components}
            | group
        )
    obs["Observations"]["acf"] = groups
    return setup, obs


def test_acf_groups_by_ray_parameter(tmp_path):
    setup, obs = _acf_groups_obs(tmp_path, [0.05, 0.075], error_correlation="processing")
    model_cfg = load_yaml(EXAMPLE / "model.yaml")
    grouped = build_inversion(model_cfg, obs)
    names = [o.name for o in grouped.observations.observations]
    assert names == ["acf_p0_vertical", "acf_p0_pressure", "acf_p1_vertical", "acf_p1_pressure"]
    assert {f"noise.{n}" for n in names} <= set(grouped.names)
    assert [c.members for c in grouped.observations.correlations] == [(n,) for n in names]
    # Each group is predicted at its own ray parameter: theta0 fits its own data exactly.
    theta = np.r_[setup.parameterization.theta0(), np.zeros(len(grouped.names) - 4)]
    theta = np.array(
        [theta[setup.names.index(n)] if n in setup.names else 0.0 for n in grouped.names]
    )
    fits = grouped.target.evaluate(theta).likelihood
    assert max(b.chi2 for b in fits.blocks) < 1e-12
    swapped = copy.deepcopy(obs)
    swapped["Observations"]["acf"][0]["ray_p"] = 0.075
    fits = build_inversion(model_cfg, swapped).target.evaluate(theta).likelihood
    assert fits.blocks[0].chi2 > 1.0 and fits.blocks[2].chi2 < 1e-12


def test_acf_group_list_errors(tmp_path):
    _, obs = _acf_groups_obs(tmp_path, [0.05, 0.075])
    model_cfg = load_yaml(EXAMPLE / "model.yaml")
    for edit, match in [
        (lambda g: g[1].update(name="p0"), "distinct"),
        (lambda g: g[0].pop("name"), "needs a name"),
        (lambda g: g[0].update(name="p 0"), "needs a name"),
        (lambda g: g[1].pop("ray_p"), r"acf\[p1\] needs ray_p"),
        (lambda g: g[1].update(raypar=0.1), r"acf\[p1\]: unknown keys"),
        (lambda g: g.clear(), "non-empty list"),
    ]:
        bad = copy.deepcopy(obs)
        edit(bad["Observations"]["acf"])
        with pytest.raises(ValueError, match=match):
            build_inversion(model_cfg, bad)


def test_setup_checks_flag_unusable_data(tmp_path):
    model_cfg = load_yaml(EXAMPLE / "model.yaml")
    obs = load_yaml(EXAMPLE / "obs.yaml")
    data = read_three_columns(EXAMPLE / "compliance.dat")
    freqs = np.r_[data.x, 0.03, 0.04]  # the limit for 2 km of water is 0.0279 Hz
    value, sigma = np.r_[data.value, data.value[-2:]], np.r_[data.sigma, data.sigma[-2:]]
    write_three_columns(tmp_path / "c.dat", freqs, value, sigma, header="compliance")
    obs["Observations"]["compliance"]["file"] = str(tmp_path / "c.dat")
    obs["Observations"]["dispersion"]["file"] = str(EXAMPLE / "vph.dat")
    for component in obs["Observations"]["acf"]["components"].values():
        component["file"] = str(EXAMPLE / component["file"])
    obs["Observations"]["acf"]["window"] = [0.0, 40.0]
    obs["Observations"]["acf"]["processing"] = {"preset": "kim2023", "normalize": "zero_lag"}
    notes = build_inversion(model_cfg, obs).check()
    warnings = [n.message for n in notes if n.level == "warning" and n.group == "compliance"]
    assert len(warnings) == 1 and warnings[0].startswith("2 of 17 frequencies above")
    acf = [n.message for n in notes if n.group == "acf_vertical"]
    assert any("beyond the data's last lag" in m for m in acf)
    assert any("normalizing zero lag" in m for m in acf)
    info = next(m for m in acf if m.startswith("water multiples"))
    assert "2.66, 5.31" in info  # 2 H sqrt(1/1.5^2 - 0.06^2) = 2.664 s per multiple
    obs["Observations"]["acf"]["window"] = [0.3, 2.0]
    notes = build_inversion(model_cfg, obs).check()
    assert any(m.message.startswith("no water multiple") for m in notes)


def test_processing_config_forms():
    assert processing_from_config("kim2023", max_lag=12.0).max_lag == 12.0
    custom = processing_from_config(
        {
            "preset": "kim2023",
            "zero_lag_taper": 0.3,
            "filters": [{"type": "highpass", "freq": 0.7}],
        },
        max_lag=20.0,
    )
    assert custom.zero_lag_taper == 0.3 and custom.filters == (Highpass(0.7),)
    assert custom.whitening == Whitening(0.8, 0.0, "power")
    with pytest.raises(ValueError, match="filter type"):
        processing_from_config({"filters": [{"type": "notch"}]}, max_lag=10.0)


# --- derived quantities and diagnostics ---------------------------------------------


def test_derived_quantities_of_constant_segments():
    model = ParameterizedModel(
        (
            Segment(0.0, 0.5, ConstantProfile(0.4), vp=ConstantProfile(1.6), name="sediment"),
            Segment(0.5, 10.0, ConstantProfile(3.5), name="crust"),
        )
    )
    d = derived_quantities(model)
    assert d["depth.interface0"] == 0.5 and d["thickness.sediment"] == 0.5
    assert d["tp2.sediment"] == pytest.approx(2 * 0.5 / 1.6)
    assert d["ts2.sediment"] == pytest.approx(2 * 0.5 / 0.4)
    assert d["vs_mean.sediment"] == pytest.approx(0.4)


def test_derived_quantities_of_a_linear_vs_gradient():
    # Analytic integrals of Vs = t + (b - t) z / H; 32-point Gauss-Legendre on these
    # smooth integrands is exact to ~1e-10, so the tolerances only allow rounding.
    t, b, h = 0.15, 0.6, 0.6
    model = ParameterizedModel(
        (
            Segment(0.0, h, GradientProfile(t, b), vp=ConstantProfile(1.75), name="sediment"),
            Segment(h, 10.0, ConstantProfile(3.5), name="crust"),
        )
    )
    d = derived_quantities(model)
    ts2 = 2 * h * np.log(b / t) / (b - t)
    assert d["ts2.sediment"] == pytest.approx(ts2, rel=1e-6)
    assert d["vs_mean.sediment"] == pytest.approx(2 * h / ts2, rel=1e-6)
    sqrt_mean = (2 / 3 * (b**1.5 - t**1.5) / (b - t)) ** 2
    assert d["vs_sqrt_mean.sediment"] == pytest.approx(sqrt_mean, rel=1e-8)
    assert d["vs_mid.sediment"] == pytest.approx((t + b) / 2)
    assert d["vs_mid.crust"] == pytest.approx(3.5)


def test_rhat_and_ess_on_known_processes():
    rng = np.random.default_rng(0)
    iid = rng.standard_normal((4, 2000))
    assert split_rhat(iid) == pytest.approx(1.0, abs=0.01)
    assert effective_sample_size(iid) == pytest.approx(8000, rel=0.1)
    phi, ar = 0.9, np.zeros((4, 4000))
    noise = rng.standard_normal((4, 4000))
    for t in range(1, 4000):
        ar[:, t] = phi * ar[:, t - 1] + noise[:, t]
    assert effective_sample_size(ar) == pytest.approx(16000 * (1 - phi) / (1 + phi), rel=0.2)
    shifted = iid.copy()
    shifted[0] += 3.0
    assert split_rhat(shifted) > 1.5
    assert split_rhat(np.stack([iid, shifted], axis=2)).shape == (2,)


# --- the run driver ---------------------------------------------------------------


def test_run_writes_the_documented_outputs(tmp_path):
    out = tmp_path / "run"
    result = run_inversion(_quick_model(tmp_path), EXAMPLE / "obs.yaml", out, with_prior=True)
    expected = {
        "resolved_model.yaml",
        "resolved_obs.yaml",
        "run.log",
        "prior_mcmc.nc",
        "prior_model.nc",
        "posterior_mcmc.nc",
        "posterior_model.nc",
        "posterior_predictive.nc",
    }
    assert expected <= {p.name for p in out.iterdir()}
    mcmc = xr.open_dataset(out / "posterior_mcmc.nc", engine="h5netcdf")
    assert mcmc.sizes["chain"] == 2 and mcmc.sizes["draw"] == 40 and mcmc.sizes["parameter"] == 8
    assert {"rhat", "ess", "log_prior", "log_likelihood"} <= set(mcmc.data_vars)
    # Per-group misfits: the stored chi2 of a draw is that of its parameters.
    assert list(mcmc["group"].values) == [
        "compliance",
        "dispersion",
        "acf_vertical",
        "acf_pressure",
    ]
    assert list(mcmc["term"].values) == list(mcmc["group"].values)
    assert int(mcmc["trace_step"][0]) == 0 and mcmc.sizes["trace_step"] == 60
    setup = build_inversion(
        load_yaml(tmp_path / "model.yaml"), load_yaml(EXAMPLE / "obs.yaml"), obs_dir=EXAMPLE
    )
    fit = setup.target.evaluate(mcmc["theta"].values[1, 7]).likelihood
    np.testing.assert_allclose(mcmc["group_chi2"].values[1, 7], [g.chi2 for g in fit.groups])
    np.testing.assert_allclose(
        mcmc["term_log_likelihood"].values[1, 7].sum(), mcmc["log_likelihood"].values[1, 7]
    )
    assert int(mcmc["group_n"].sel(group="acf_vertical")) == 155
    assert mcmc.attrs["station"] == "SYN01" and mcmc.attrs["stage"] == "posterior"
    model = xr.open_dataset(out / "posterior_model.nc", engine="h5netcdf")
    assert model["vs"].shape == (20, 51) and "tp2.sediment" in model
    predictive = xr.open_dataset(out / "posterior_predictive.nc", engine="h5netcdf")
    assert predictive["acf_vertical_predicted"].shape == (5, 601)
    assert "acf_vertical_best_chi2" in predictive.attrs
    assert load_yaml(out / "resolved_obs.yaml")["Run"]["station"] == "SYN01"
    assert "check acf_vertical: water multiples inside the window" in (out / "run.log").read_text()
    assert result.files["posterior_predictive"].exists()


def test_tempered_run_records_the_ladder(tmp_path):
    model = _quick_model(tmp_path, tempering={"n_temperatures": 3, "max_temperature": 9.0})
    out = tmp_path / "pt"
    run_inversion(model, None, out, prior_only=True)
    mcmc = xr.open_dataset(out / "prior_mcmc.nc", engine="h5netcdf")
    assert mcmc["swap_acceptance"].shape == (2, 2)
    assert mcmc.attrs["temperatures"] == "1.0, 3.0, 9.0" and mcmc.attrs["swap_interval"] == 1
    assert "tempering ladder 1 3 9" in (out / "run.log").read_text()


def test_tempering_settings():
    setup = build_inversion(load_yaml(EXAMPLE / "model.yaml"))
    assert setup.metropolis_config().ladder == (1.0,)
    cfg = load_yaml(EXAMPLE / "model.yaml")
    cfg["Sampler"]["tempering"] = {}
    ladder = build_inversion(cfg).metropolis_config().ladder
    assert len(ladder) == 5 and ladder[-1] == pytest.approx(30.0)
    cfg["Sampler"]["tempering"] = {"temperatures": [1, 4], "swap_interval": 2}
    config = build_inversion(cfg).metropolis_config()
    assert config.ladder == (1.0, 4.0) and config.swap_interval == 2
    for bad, match in [
        ({"temperatures": [1, 4], "n_temperatures": 3}, "geometric"),
        ({"n_temperature": 3}, "unknown keys"),
        ({"max_temperature": 0.5}, "max_temperature"),
    ]:
        cfg["Sampler"]["tempering"] = bad
        with pytest.raises(ValueError, match=match):
            build_inversion(cfg)


def test_run_refuses_existing_output_and_prior_only_needs_no_data(tmp_path):
    model = _quick_model(tmp_path)
    out = tmp_path / "prior"
    run_inversion(model, None, out, prior_only=True)
    assert not (out / "posterior_mcmc.nc").exists()
    with pytest.raises(FileExistsError):
        run_inversion(model, None, out, prior_only=True)
    with pytest.raises(ValueError, match="needs obs.yaml"):
        run_inversion(model, None, tmp_path / "post")


def test_cli(tmp_path, capsys):
    model = _quick_model(tmp_path)
    assert main(["invert", "--model", str(model), "-o", str(tmp_path / "cli"), "--prior-only"]) == 0
    assert "results in" in capsys.readouterr().out
    assert main(["invert", "--model", str(model), "-o", str(tmp_path / "cli"), "--prior-only"]) == 2
    assert "exists" in capsys.readouterr().err
