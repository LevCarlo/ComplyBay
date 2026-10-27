"""Write the data files of examples/synthetic from a known model.

The data are computed on a finer layering than the inversion uses (avoids
fitting the exact discretization) and get Gaussian noise of the stated sigma:
compliance 10 %, phase velocity 0.02 km/s, ACF 0.05 per lag (independent).

    python tools/make_synthetic_example.py examples/synthetic
"""

import argparse
from pathlib import Path

import numpy as np

from complybay.acf import preset
from complybay.forward import compliance, phase_velocity, synthetic_acfs
from complybay.io import load_yaml, write_three_columns
from complybay.model import OBSModel
from complybay.parameterization import DiscretizationConfig, discretize
from complybay.workflow.config import water_from_station
from complybay.workflow.setup import build_inversion

TRUTH = {"basement.depth": 0.6, "sed.vs_top": 0.15, "sed.vs_bottom": 0.6, "sed.vp": 1.75}
FINE = DiscretizationConfig(layers_per_segment={"sediment": 60, "crust": 40, "mantle": 30})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    setup = build_inversion(load_yaml(args.directory / "model.yaml"))
    p = setup.parameterization
    model = p.vector_to_model([TRUTH[name] for name in p.names])
    water = water_from_station(load_yaml(args.directory / "obs.yaml")["Station"])
    truth = OBSModel(discretize(model, FINE), water)
    rng = np.random.default_rng(args.seed)

    freqs = np.geomspace(0.004, 0.025, 15)
    eta = compliance(truth, freqs)
    write_three_columns(
        args.directory / "compliance.dat",
        freqs,
        eta * (1 + 0.1 * rng.standard_normal(freqs.size)),
        0.1 * eta,
        header="frequency_Hz compliance_1/Pa sigma_1/Pa (synthetic, 10 % noise)",
    )
    periods = np.linspace(8.0, 30.0, 12)
    c = phase_velocity(truth, periods)
    write_three_columns(
        args.directory / "vph.dat",
        periods,
        c + 0.02 * rng.standard_normal(periods.size),
        np.full(periods.size, 0.02),
        header="period_s phase_velocity_km/s sigma_km/s (synthetic)",
    )
    lags, acfs = synthetic_acfs(truth, 0.06, preset("kim2023"))
    for component, name in (("vertical", "acf_z.dat"), ("pressure", "acf_p.dat")):
        write_three_columns(
            args.directory / name,
            lags,
            acfs[component] + 0.05 * rng.standard_normal(lags.size),
            np.full(lags.size, 0.05),
            header=f"lag_s acf sigma ({component}, kim2023, ray_p 0.06 s/km, synthetic)",
        )
    print(f"wrote data for {TRUTH} to {args.directory}")


if __name__ == "__main__":
    main()
