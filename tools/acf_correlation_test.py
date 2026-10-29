"""Invert event-based synthetic ACFs with and without correlated ACF errors.

Uses ``examples/synthetic/obs_events.yaml`` (Z and P ACFs stacked from 60
synthetic events, see ``tools/make_synthetic_example.py --events``) and fits
it with three settings of ``Observations.acf.error_correlation``:

- ``independent``: lags and components independent (the old likelihood),
- ``processing``: theory for additive noise, per component,
- ``events``: estimated from the event ACFs, joint over Z and P,

once with all data (CDZP) and once with the ACFs alone (ZP).

    python tools/acf_correlation_test.py examples/synthetic runs/acf_correlation

Writes one run directory per case, ``summary.nc`` and ``summary.md`` like
``tools/data_subset_test.py``; finished cases are not rerun.
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import xarray as xr
from data_subset_test import DERIVED, summarize
from make_synthetic_example import TRUTH

from complybay.inference import derived_quantities
from complybay.io import dump_yaml, load_yaml
from complybay.workflow import build_inversion, run_inversion

SETTINGS = {"independent": None, "processing": "processing", "events": {"file": "acf_errors.nc"}}
CASES = ["CDZP-independent", "CDZP-processing", "CDZP-events", "ZP-independent", "ZP-events"]


def case_obs(obs: dict, case: str, base: Path) -> dict:
    data, setting = case.split("-")
    out = copy.deepcopy(obs)
    groups = out["Observations"]
    if data == "ZP":
        groups.pop("compliance")
        groups.pop("dispersion")
    for item in groups.values():
        if "file" in item:
            item["file"] = str((base / item["file"]).resolve())
    acf = groups["acf"]
    for item in acf["components"].values():
        item["file"] = str((base / item["file"]).resolve())
    acf.pop("error_correlation", None)
    if SETTINGS[setting] is not None:
        value = copy.deepcopy(SETTINGS[setting])
        if isinstance(value, dict):
            value["file"] = str((base / value["file"]).resolve())
        acf["error_correlation"] = value
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("example", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", nargs="+", default=CASES)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    model_file = args.example / "model.yaml"
    obs = load_yaml(args.example / "obs_events.yaml")
    for case in args.cases:
        run_dir = args.output / case
        if (run_dir / "posterior_mcmc.nc").exists():
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        obs_file = run_dir / "obs_case.yaml"
        dump_yaml(case_obs(obs, case, args.example), obs_file)
        print(f"running {case}", flush=True)
        run_inversion(model_file, obs_file, run_dir, overwrite=True)

    p = build_inversion(load_yaml(model_file)).parameterization
    names = list(p.names) + DERIVED
    truth = {**TRUTH, **derived_quantities(p.vector_to_model([TRUTH[n] for n in p.names]))}
    results = [summarize(args.output / c, "posterior", names, p) for c in args.cases]
    stats = np.array([[r["stats"][n] for n in names] for r in results])
    noise = {}
    for case in args.cases:
        with xr.open_dataset(args.output / case / "posterior_mcmc.nc", engine="h5netcdf") as m:
            theta = m["theta"].values.reshape(-1, m.sizes["parameter"])
            params = list(m["parameter"].values)
        noise[case] = {
            n: float(10 ** np.median(theta[:, params.index(n)]))
            for n in params
            if n.startswith("noise.acf")
        }
    summary = xr.Dataset(
        {
            "q05": (("case", "quantity"), stats[..., 0]),
            "median": (("case", "quantity"), stats[..., 1]),
            "q95": (("case", "quantity"), stats[..., 2]),
            "rhat": (("case", "quantity"), stats[..., 3]),
            "truth": (("quantity",), [truth[n] for n in names]),
        },
        coords={"case": list(args.cases), "quantity": names},
        attrs={"description": "Event-based synthetic ACFs fitted with three ACF error models"},
    )
    summary.to_netcdf(args.output / "summary.nc", engine="h5netcdf")

    lines = [
        "90 % intervals (`*`: truth outside); h: posterior median ACF noise scales.",
        "",
        "| case | " + " | ".join(names) + " | h (Z, P) | max R-hat |",
        "|---" * (len(names) + 3) + "|",
    ]
    for i, case in enumerate(args.cases):
        cells = []
        for j, name in enumerate(names):
            lo, _, hi, _ = stats[i, j]
            mark = "" if lo <= truth[name] <= hi else "*"
            cells.append(f"{lo:.3f}-{hi:.3f}{mark}")
        h = ", ".join(f"{v:.2f}" for v in noise[case].values())
        rhat = np.nanmax(stats[i, : len(p.names), 3])
        lines.append(f"| {case} | " + " | ".join(cells) + f" | {h} | {rhat:.3f} |")
    lines.append("")
    lines.append("| truth | " + " | ".join(f"{truth[n]:.3f}" for n in names) + " | | |")
    text = "\n".join(lines) + "\n"
    (args.output / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
