"""Run an inversion from ``model.yaml`` / ``obs.yaml`` and write its outputs.

Output directory (refused if it exists, unless ``overwrite``)::

    resolved_model.yaml, resolved_obs.yaml   inputs, plus a Run block
    run.log                                  settings, progress, diagnostics
    prior_mcmc.nc, prior_model.nc            with --prior-only / --with-prior
    posterior_mcmc.nc, posterior_model.nc, posterior_predictive.nc

See :mod:`complybay.io.netcdf` for the file contents.
"""

from __future__ import annotations

import copy
import datetime as dt
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import complybay
from complybay.inference import predict
from complybay.inference.diagnostics import (
    effective_sample_size,
    lagging_chains,
    split_rhat,
)
from complybay.inference.ensemble import model_ensemble
from complybay.io import dump_yaml, load_yaml
from complybay.io.netcdf import mcmc_dataset, model_dataset, predictive_dataset, write_netcdf
from complybay.workflow.setup import InversionSetup, build_inversion


@dataclass
class RunResult:
    output_dir: Path
    files: dict[str, Path] = field(default_factory=dict)


def _git_commit() -> str:
    try:
        repo = Path(complybay.__file__).resolve().parent
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _logger(path: Path, level: str) -> logging.Logger:
    logger = logging.getLogger(f"complybay.run.{path}")
    logger.setLevel(level.upper())
    logger.handlers.clear()
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def _run_attrs(setup: InversionSetup, model_file, obs_file) -> dict:
    s = setup.sampler
    return {
        "complybay_version": complybay.__version__,
        "git_commit": _git_commit(),
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model_file": str(Path(model_file).resolve()),
        "obs_file": None if obs_file is None else str(Path(obs_file).resolve()),
        "station": setup.station,
        "water_depth_km": None if setup.water is None else setup.water.thickness,
        "parameter_names": list(setup.names),
        "sampler_seed": s.seed,
        "sampler_n_chains": s.n_chains,
        "sampler_executor": s.executor,
        "sampler_initial": s.initial,
    }


def _thinned(flat: np.ndarray, n: int) -> np.ndarray:
    """Up to ``n`` evenly spaced rows of ``flat``."""
    if flat.shape[0] <= n:
        return flat
    return flat[np.linspace(0, flat.shape[0] - 1, n).astype(int)]


def _write_stage(stage, setup, result, out, attrs, logger, files):
    rhat = split_rhat(result.samples)
    ess = effective_sample_size(result.samples)
    logger.info("%s: acceptance after burn-in %s", stage, np.round(result.acceptance_rates, 2))
    logger.info("%-24s %10s %10s %10s %7s %8s", "parameter", "median", "5%", "95%", "R-hat", "ESS")
    flat = result.flat_samples()
    for i, name in enumerate(setup.names):
        q5, q50, q95 = np.percentile(flat[:, i], [5, 50, 95])
        logger.info("%-24s %10.4g %10.4g %10.4g %7.3f %8.0f", name, q50, q5, q95, rhat[i], ess[i])
    ladder = result.config.ladder
    if len(ladder) > 1:
        logger.info("%s: tempering ladder %s", stage, " ".join(f"{t:.3g}" for t in ladder))
        for chain in result.chains:
            logger.info(
                "chain %d swap acceptance %s", chain.chain_id, np.round(chain.swap_acceptance, 2)
            )
    worst = int(np.argmax(rhat))
    if rhat[worst] > 1.1:
        logger.warning(
            "not converged: R-hat %.2f for %s; run longer chains", rhat[worst], setup.names[worst]
        )
    lp = result.log_posterior
    lagging = lagging_chains(lp)
    best = int(np.argmax(np.median(lp, axis=1)))
    for c in lagging:
        logger.warning(
            "chain %d is stuck below the others: log posterior median %.1f against %.1f "
            "(chain %d); it is kept in the outputs. Use Sampler.tempering or more chains",
            c,
            np.median(lp[c]),
            np.median(lp[best]),
            best,
        )
    attrs = attrs | {"lagging_chains": list(lagging)}

    files[f"{stage}_mcmc"] = out / f"{stage}_mcmc.nc"
    write_netcdf(
        mcmc_dataset(
            result,
            rhat=rhat,
            ess=ess,
            attrs=attrs | {"stage": stage},
            group_sizes=None if stage == "prior" else setup.target.likelihood.sizes,
        ),
        files[f"{stage}_mcmc"],
    )

    o = setup.output
    zmax = setup.parameterization.vector_to_model(flat[0]).zmax
    depths = np.linspace(0.0, o.max_depth_km or zmax, o.n_depths)
    vs, vp, rho, derived = model_ensemble(
        setup.parameterization, _thinned(flat, o.n_model_samples), depths
    )
    files[f"{stage}_model"] = out / f"{stage}_model.nc"
    write_netcdf(
        model_dataset(depths, vs, vp, rho, derived, attrs=attrs | {"stage": stage}),
        files[f"{stage}_model"],
    )


def _write_predictive(setup, result, out, attrs, logger, files):
    flat = result.flat_samples()
    log_post = result.log_posterior.reshape(-1)
    rng = np.random.default_rng(setup.sampler.seed)
    n = min(setup.output.n_predictive_samples, flat.shape[0])
    picks = flat[rng.choice(flat.shape[0], n, replace=False)]
    observations = setup.target.likelihood.observations
    predictions = [[] for _ in observations]
    for theta in picks:
        for k, values in enumerate(predict(observations, setup.prior.evaluate(theta).obs_model)):
            predictions[k].append(values)
    best_theta = flat[int(np.argmax(log_post))]
    best_eval = setup.target.evaluate(best_theta)
    groups = best_eval.likelihood.groups
    blocks = best_eval.likelihood.blocks
    correlated = {m for b in blocks for m in b.members}
    for g in groups:
        note = "  (lags taken as independent)" if g.name in correlated else ""
        logger.info(
            "best sample, %-16s chi2/N = %6.2f  noise scale h = %.3f%s",
            g.name,
            g.chi2 / g.n,
            g.scale,
            note,
        )
    for b in blocks:
        logger.info(
            "best sample, correlated %s: chi2/k = %.2f with k = %d retained eigenvectors",
            b.name,
            b.chi2 / b.k,
            b.k,
        )
    block_attrs = {}
    for b in blocks:
        block_attrs[f"block_{b.name}_best_chi2"] = float(b.chi2)
        block_attrs[f"block_{b.name}_k"] = int(b.k)
    files["posterior_predictive"] = out / "posterior_predictive.nc"
    dataset = predictive_dataset(
        observations,
        [np.asarray(p) for p in predictions],
        [g.prediction for g in groups],
        groups,
        attrs=attrs
        | block_attrs
        | {"stage": "posterior", "best_log_posterior": float(best_eval.log_posterior)},
    )
    write_netcdf(dataset, files["posterior_predictive"])


def run_inversion(
    model_file: str | Path,
    obs_file: str | Path | None,
    output: str | Path,
    *,
    prior_only: bool = False,
    with_prior: bool = False,
    overwrite: bool = False,
    progress: bool = False,
    progress_every: int | None = None,
    log_level: str = "INFO",
) -> RunResult:
    """Build the inversion, sample it, and write the run directory."""
    if prior_only and with_prior:
        raise ValueError("prior_only and with_prior exclude each other.")
    if not prior_only and obs_file is None:
        raise ValueError("A posterior run needs obs.yaml; use prior_only for prior sampling.")
    model_cfg = load_yaml(model_file)
    obs_cfg = None if obs_file is None else load_yaml(obs_file)
    setup = build_inversion(
        model_cfg, obs_cfg, obs_dir="." if obs_file is None else Path(obs_file).parent
    )

    out = Path(output)
    if out.exists() and not overwrite:
        raise FileExistsError(f"Output directory exists: {out} (use overwrite).")
    out.mkdir(parents=True, exist_ok=True)
    result = RunResult(out)
    logger = _logger(out / "run.log", log_level)
    attrs = _run_attrs(setup, model_file, obs_file)

    for name, cfg in (("model", model_cfg), ("obs", obs_cfg)):
        if cfg is not None:
            snapshot = copy.deepcopy(cfg)
            snapshot["Run"] = {k: v for k, v in attrs.items() if v is not None}
            result.files[f"resolved_{name}"] = out / f"resolved_{name}.yaml"
            dump_yaml(snapshot, result.files[f"resolved_{name}"])

    s = setup.sampler
    logger.info(
        "ComplyBay %s (%s); station %s",
        attrs["complybay_version"],
        attrs["git_commit"],
        setup.station,
    )
    logger.info("%d parameters: %s", len(setup.names), ", ".join(setup.names))
    logger.info(
        "sampler: %d chains x %d steps, burn-in %d, thin %d, update %s, executor %s, seed %s",
        s.n_chains,
        s.n_steps,
        s.burn_in,
        s.thin,
        s.update,
        s.executor,
        s.seed,
    )
    for note in setup.check():
        log = logger.warning if note.level == "warning" else logger.info
        log("check %s: %s", note.group, note.message)
    callback = None
    if progress and s.executor != "process":

        def callback(chain, step, acceptance, log_post):
            logger.info(
                "chain %d step %d acceptance %.2f log posterior %.2f",
                chain,
                step,
                acceptance,
                log_post,
            )
    elif progress:
        logger.warning("progress is not reported with the process executor")
    every = progress_every or max(1, s.n_steps // 10)

    stages = ["prior"] if prior_only else (["prior", "posterior"] if with_prior else ["posterior"])
    for stage in stages:
        logger.info("running %s sampling", stage)
        start = dt.datetime.now()
        mcmc = setup.run(prior_only=stage == "prior", progress=callback, progress_every=every)
        logger.info("%s sampling took %.0f s", stage, (dt.datetime.now() - start).total_seconds())
        _write_stage(stage, setup, mcmc, out, attrs, logger, result.files)
        if stage == "posterior":
            _write_predictive(setup, mcmc, out, attrs, logger, result.files)
    logger.info("wrote %s", ", ".join(p.name for p in result.files.values()))
    return result
