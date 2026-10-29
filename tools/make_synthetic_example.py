"""Write the data files of examples/synthetic from a known model.

The data are computed on a finer layering than the inversion uses (avoids
fitting the exact discretization) and get Gaussian noise of the stated sigma:
compliance 10 %, phase velocity 0.02 km/s, ACF 0.05 per lag (independent).

With ``--events N`` it also writes ACFs as real data would be made: N event
traces (the true seafloor response to a random source wavelet of
``--source-duration`` s, a 70 s window from 10 s before P, optional white
noise), each processed with kim2023 and normalized, then stacked with
:func:`complybay.acf.stack_errors`. The errors of these stacks come from the
differences between source wavelets, so they are correlated between lags and
between Z and P. Writes ``acf_z_events.dat``, ``acf_p_events.dat`` and
``acf_errors.nc`` (used by ``obs_events.yaml``); the other files are unchanged
(separate random stream).

    python tools/make_synthetic_example.py examples/synthetic
    python tools/make_synthetic_example.py examples/synthetic --events 60
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from complybay.acf import acf_from_trace, preset, stack_errors
from complybay.forward import compliance, phase_velocity, seafloor_response, synthetic_acfs
from complybay.io import load_yaml, write_acf_errors, write_three_columns
from complybay.model import OBSModel
from complybay.parameterization import DiscretizationConfig, discretize
from complybay.workflow.config import water_from_station
from complybay.workflow.setup import build_inversion

TRUTH = {"basement.depth": 0.6, "sed.vs_top": 0.15, "sed.vs_bottom": 0.6, "sed.vp": 1.75}
FINE = DiscretizationConfig(layers_per_segment={"sediment": 60, "crust": 40, "mantle": 30})


def event_acfs(model, ray_p, processing, n_events, rng, *, source_duration=3.0, noise=0.0, dt=0.05):
    """Per-event Z and P ACFs (n_events, n_lags) of synthetic event traces.

    Source: white noise under a Hann window of ``source_duration`` s (assumed
    shape). Window: 70 s from 10 s before P, Tukey 0.2 (Kim et al., 2023).
    ``noise``: white noise std relative to the trace std.
    """
    n_fft = 4000  # 200 s: the response has decayed (see wraparound_fraction)
    freqs = rfftfreq(n_fft, dt)
    response = seafloor_response(model, ray_p, freqs[1:])
    out = {"vertical": [], "pressure": []}
    n_src = int(round(source_duration / dt))
    for _ in range(n_events):
        source = rfft(rng.standard_normal(n_src) * np.hanning(n_src), n=n_fft)
        for component in out:
            spectrum = np.r_[0.0, response.component(component)] * 1.0
            trace = irfft(np.conj(spectrum) * source, n=n_fft)  # exp(-i w t) -> numpy
            trace = np.roll(trace, int(round(10 / dt)))[: int(round(70 / dt))]
            trace = trace + noise * trace.std() * rng.standard_normal(trace.size)
            out[component].append(acf_from_trace(trace, dt, processing, taper_fraction=0.2)[1])
    return {c: np.array(v) for c, v in out.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--events", type=int, default=0, help="also write event-based ACFs")
    parser.add_argument("--source-duration", type=float, default=3.0)
    parser.add_argument("--event-noise", type=float, default=0.0)
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
    if args.events:
        kim = preset("kim2023")
        events = event_acfs(
            truth,
            0.06,
            kim,
            args.events,
            np.random.default_rng(args.seed + 1000),
            source_duration=args.source_duration,
            noise=args.event_noise,
        )
        window = tuple(
            load_yaml(args.directory / "obs_events.yaml")["Observations"]["acf"]["window"]
        )
        errors = stack_errors(events, 0.05, window, processing=kim)
        for component, name in (("vertical", "acf_z_events.dat"), ("pressure", "acf_p_events.dat")):
            write_three_columns(
                args.directory / name,
                errors.lags,
                errors.stack[component],
                errors.sigma[component],
                header=(
                    f"lag_s acf sigma ({component}, kim2023, ray_p 0.06 s/km, linear stack of "
                    f"{args.events} synthetic events; correlation in acf_errors.nc)"
                ),
            )
        write_acf_errors(
            errors,
            args.directory / "acf_errors.nc",
            attrs={
                "source": "tools/make_synthetic_example.py",
                "source_duration_s": args.source_duration,
                "event_noise": args.event_noise,
                "seed": args.seed + 1000,
            },
        )
        print(f"wrote {args.events}-event ACFs (shrinkage {errors.shrinkage:.2f})")
    print(f"wrote data for {TRUTH} to {args.directory}")


if __name__ == "__main__":
    main()
