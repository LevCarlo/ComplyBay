"""Invert the synthetic example with every combination of data groups.

Shows what each data type constrains: for each combination of compliance (C),
phase velocity (D) and the vertical / pressure ACFs (Z, P) the example in
``examples/synthetic`` is run with only those groups, and the posterior of
each sampled parameter and of the derived sediment quantities (two-way P and
S times, mean Vs) is compared with the prior and the true model.

    python tools/data_subset_test.py examples/synthetic runs/data_subsets
    python tools/data_subset_test.py examples/synthetic runs/data_subsets --cases C D CDZP

Writes one run directory per case, ``summary.nc`` (median, 5/95 % quantiles,
90 % width relative to the prior, R-hat, truth per case and quantity, and the
correlation of the model parameters) and ``summary.md``. Cases whose run
directory already holds ``posterior_mcmc.nc`` are not rerun.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import xarray as xr
from make_synthetic_example import TRUTH

from complybay.inference import derived_quantities
from complybay.io import dump_yaml, load_yaml
from complybay.workflow import build_inversion, run_inversion

GROUPS = {"C": "compliance", "D": "dispersion", "Z": "vertical", "P": "pressure"}
DEFAULT_CASES = ["C", "D", "Z", "P", "ZP", "CD", "CZP", "DZP", "CDZ", "CDP", "CDZP"]
DERIVED = [
    "tp2.sediment",
    "ts2.sediment",
    "vs_mean.sediment",
    "vs_sqrt_mean.sediment",
    "vs_mid.sediment",
]


def subset_obs(obs: dict, case: str, base: Path) -> dict:
    """``obs.yaml`` with only the groups of ``case``; file paths made absolute."""
    out = {key: value for key, value in obs.items() if key != "Observations"}
    groups = {}
    full = obs["Observations"]
    for letter, name in (("C", "compliance"), ("D", "dispersion")):
        if letter in case:
            groups[name] = dict(full[name], file=str((base / full[name]["file"]).resolve()))
    components = {
        name: dict(cfg, file=str((base / cfg["file"]).resolve()))
        for name, cfg in full["acf"]["components"].items()
        if {"vertical": "Z", "pressure": "P"}[name] in case
    }
    if components:
        groups["acf"] = dict(full["acf"], components=components)
    out["Observations"] = groups
    return out


def quantiles(values: np.ndarray) -> np.ndarray:
    return np.quantile(values, [0.05, 0.5, 0.95])


def summarize(run_dir: Path, stage: str, names: list[str], parameterization) -> dict:
    with xr.open_dataset(run_dir / f"{stage}_mcmc.nc", engine="h5netcdf") as mcmc:
        theta = mcmc["theta"].values.reshape(-1, mcmc.sizes["parameter"])
        params = list(mcmc["parameter"].values)
        rhat = dict(zip(params, mcmc["rhat"].values, strict=True))
    # Derived quantities from every kept sample (the model file holds a subset).
    n_model = len(parameterization.names)
    rows = [derived_quantities(parameterization.vector_to_model(t[:n_model])) for t in theta]
    derived = {name: np.array([row[name] for row in rows]) for name in DERIVED}
    out = {}
    for name in names:
        if name in params:
            out[name] = (*quantiles(theta[:, params.index(name)]), rhat[name])
        else:
            out[name] = (*quantiles(derived[name]), np.nan)
    model_params = [p for p in params if not p.startswith("noise.")]
    corr = np.corrcoef(theta[:, [params.index(p) for p in model_params]].T)
    return {"stats": out, "corr": corr, "model_params": model_params}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("example", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    args = parser.parse_args(argv)
    for case in args.cases:
        if not case or set(case) - set(GROUPS) or len(set(case)) != len(case):
            raise SystemExit(f"Bad case {case!r}: use letters from {''.join(GROUPS)}.")
    args.output.mkdir(parents=True, exist_ok=True)
    model_file = args.example / "model.yaml"
    obs = load_yaml(args.example / "obs.yaml")

    prior_dir = args.output / "prior"
    if not (prior_dir / "prior_mcmc.nc").exists():
        run_inversion(model_file, None, prior_dir, prior_only=True, overwrite=True)
    for case in args.cases:
        run_dir = args.output / case
        if (run_dir / "posterior_mcmc.nc").exists():
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        obs_file = run_dir / "obs_subset.yaml"
        dump_yaml(subset_obs(obs, case, args.example), obs_file)
        print(f"running {case}", flush=True)
        run_inversion(model_file, obs_file, run_dir, overwrite=True)

    p = build_inversion(load_yaml(model_file)).parameterization  # model parameters only
    model_names = list(p.names)
    truth_model = p.vector_to_model([TRUTH[name] for name in p.names])
    truth = {**TRUTH, **derived_quantities(truth_model)}
    names = model_names + DERIVED

    prior = summarize(prior_dir, "prior", names, p)
    cases = ["prior", *args.cases]
    results = [prior] + [summarize(args.output / c, "posterior", names, p) for c in args.cases]
    stats = np.array([[r["stats"][n] for n in names] for r in results])  # case, quantity, 4
    prior_width = stats[0, :, 2] - stats[0, :, 0]
    summary = xr.Dataset(
        {
            "q05": (("case", "quantity"), stats[..., 0]),
            "median": (("case", "quantity"), stats[..., 1]),
            "q95": (("case", "quantity"), stats[..., 2]),
            "relative_width": (("case", "quantity"), (stats[..., 2] - stats[..., 0]) / prior_width),
            "rhat": (("case", "quantity"), stats[..., 3]),
            "truth": (("quantity",), [truth[n] for n in names]),
            "correlation": (
                ("case", "parameter", "parameter_2"),
                np.stack([r["corr"] for r in results]),
            ),
        },
        coords={
            "case": cases,
            "quantity": names,
            "parameter": model_names,
            "parameter_2": model_names,
        },
        attrs={
            "description": "Synthetic example inverted with subsets of the data groups",
            "groups": ", ".join(f"{k}={v}" for k, v in GROUPS.items()),
            "relative_width": "90 % interval width divided by the prior's",
        },
    )
    summary.to_netcdf(args.output / "summary.nc", engine="h5netcdf")

    lines = [
        "90 % interval width relative to the prior (truth inside the interval: no mark; "
        "outside: *); max R-hat of the model parameters.",
        "",
        "| case | " + " | ".join(names) + " | R-hat |",
        "|---" * (len(names) + 2) + "|",
    ]
    for i, case in enumerate(cases):
        cells = []
        for j, name in enumerate(names):
            inside = stats[i, j, 0] <= truth[name] <= stats[i, j, 2]
            cells.append(f"{summary.relative_width.values[i, j]:.2f}{'' if inside else '*'}")
        rhat = np.nanmax(stats[i, : len(model_names), 3])
        lines.append(f"| {case} | " + " | ".join(cells) + f" | {rhat:.3f} |")
    lines += ["", "Correlation of the model parameters (" + ", ".join(model_names) + "):", ""]
    pairs = list(itertools.combinations(range(len(model_names)), 2))
    lines.append(
        "| case | " + " | ".join(f"{model_names[a]} / {model_names[b]}" for a, b in pairs) + " |"
    )
    lines.append("|---" * (len(pairs) + 1) + "|")
    for i, case in enumerate(cases):
        corr = results[i]["corr"]
        lines.append(f"| {case} | " + " | ".join(f"{corr[a, b]:+.2f}" for a, b in pairs) + " |")
    text = "\n".join(lines) + "\n"
    (args.output / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
