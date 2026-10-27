"""Command line: ``complybay invert --model model.yaml --obs obs.yaml -o run_dir``."""

from __future__ import annotations

import argparse
import sys

from complybay.workflow.driver import run_inversion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="complybay", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    invert = commands.add_parser("invert", help="Sample the prior or the posterior of one station.")
    invert.add_argument(
        "--model", required=True, help="model.yaml: model space, priors, sampler, output."
    )
    invert.add_argument(
        "--obs", default=None, help="obs.yaml: station, data files, forward settings."
    )
    invert.add_argument("-o", "--output", required=True, help="Output directory for this run.")
    stage = invert.add_mutually_exclusive_group()
    stage.add_argument("--prior-only", action="store_true", help="Sample the prior only (no data).")
    stage.add_argument(
        "--with-prior", action="store_true", help="Sample the prior, then the posterior."
    )
    invert.add_argument(
        "--overwrite", action="store_true", help="Write into an existing directory."
    )
    invert.add_argument(
        "--progress", action="store_true", help="Log chain progress (not with processes)."
    )
    invert.add_argument(
        "--progress-every", type=int, default=None, help="Steps between progress lines."
    )
    invert.add_argument("--log-level", default="INFO", help="Logging level (default INFO).")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "invert":
        try:
            result = run_inversion(
                args.model,
                args.obs,
                args.output,
                prior_only=args.prior_only,
                with_prior=args.with_prior,
                overwrite=args.overwrite,
                progress=args.progress,
                progress_every=args.progress_every,
                log_level=args.log_level,
            )
        except (ValueError, FileExistsError, FileNotFoundError, KeyError) as exc:
            print(f"complybay invert: error: {exc}", file=sys.stderr)
            return 2
        print(f"results in {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
