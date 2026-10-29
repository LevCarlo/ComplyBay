"""Invert synthetic data of an attenuating sediment with and without Q.

The true model of ``examples/synthetic`` with sediment Qs = 20, Qp = 100
(velocities at the 1 Hz reference) gives the data, with the same noise
realization as the example (relative for compliance, additive otherwise), so
the runs compare with test 1's CDZP case. Two inversions: elastic, and with
Q fixed at its true value (``docs/synthetic_tests.md``, test 6).

    python tools/q_synthetic_test.py
"""

import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, "tools")
from make_synthetic_example import FINE, TRUTH

from complybay.acf import preset
from complybay.forward import compliance, phase_velocity, synthetic_acfs
from complybay.io import dump_yaml, load_yaml, read_three_columns, write_three_columns
from complybay.model import OBSModel
from complybay.parameterization import discretize
from complybay.workflow import build_inversion, run_inversion
from complybay.workflow.config import water_from_station

QS, QP = 20.0, 100.0


def main():
    ex, out = Path("examples/synthetic"), Path("runs/q_test")
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    setup = build_inversion(load_yaml(ex / "model.yaml"))
    p = setup.parameterization
    model = p.vector_to_model([TRUTH[n] for n in p.names])
    water = water_from_station(load_yaml(ex / "obs.yaml")["Station"])
    elastic = OBSModel(discretize(model, FINE), water)
    sed = elastic.solid.tops[:-1] < TRUTH["basement.depth"] - 1e-9

    def q(value):
        return np.r_[np.where(sed, value, np.inf), np.inf]

    attenuating = elastic.with_solid(replace(elastic.solid, qs=q(QS), qp=q(QP)))

    c = read_three_columns(ex / "compliance.dat")
    rel = c.value / compliance(elastic, c.x) - 1
    eta = compliance(attenuating, c.x)
    write_three_columns(data / "compliance.dat", c.x, eta * (1 + rel), 0.1 * eta, header="Q test")
    d = read_three_columns(ex / "vph.dat")
    write_three_columns(
        data / "vph.dat",
        d.x,
        phase_velocity(attenuating, d.x) + d.value - phase_velocity(elastic, d.x),
        d.sigma,
        header="Q test",
    )
    _, el = synthetic_acfs(elastic, 0.06, preset("kim2023"))
    _, at = synthetic_acfs(attenuating, 0.06, preset("kim2023"))
    for comp, name in (("vertical", "acf_z.dat"), ("pressure", "acf_p.dat")):
        a = read_three_columns(ex / name)
        write_three_columns(
            data / name, a.x, at[comp] + a.value - el[comp], a.sigma, header="Q test"
        )
    shutil.copy(ex / "obs.yaml", data / "obs.yaml")

    base = load_yaml(ex / "model.yaml")
    withq = load_yaml(ex / "model.yaml")
    withq["Model"]["segments"][0]["qs"] = {"type": "constant", "value": QS}
    withq["Model"]["segments"][0]["qp"] = {"type": "constant", "value": QP}
    for name, cfg in (("elastic", base), ("q_fixed", withq)):
        d = out / name
        d.mkdir(exist_ok=True)
        dump_yaml(cfg, d / "model_case.yaml")
        print("running", name, flush=True)
        run_inversion(d / "model_case.yaml", data / "obs.yaml", d, overwrite=True)


if __name__ == "__main__":
    main()
