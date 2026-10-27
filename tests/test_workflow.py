import copy
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from complybay.acf import Highpass, Whitening, preset
from complybay.cli import main
from complybay.inference.diagnostics import effective_sample_size, split_rhat
from complybay.inference.ensemble import derived_quantities
from complybay.io import dump_yaml, load_yaml, read_three_columns, write_three_columns
from complybay.parameterization import ConstantProfile, ParameterizedModel, Segment
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
    assert mcmc.attrs["station"] == "SYN01" and mcmc.attrs["stage"] == "posterior"
    model = xr.open_dataset(out / "posterior_model.nc", engine="h5netcdf")
    assert model["vs"].shape == (20, 51) and "tp2.sediment" in model
    predictive = xr.open_dataset(out / "posterior_predictive.nc", engine="h5netcdf")
    assert predictive["acf_vertical_predicted"].shape == (5, 601)
    assert "acf_vertical_best_chi2" in predictive.attrs
    assert load_yaml(out / "resolved_obs.yaml")["Run"]["station"] == "SYN01"
    assert result.files["posterior_predictive"].exists()


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
