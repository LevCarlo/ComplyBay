# Development guide

How ComplyBay code is organized and what every change has to satisfy. The
short version for coding agents is `CLAUDE.md`; this page is the reference.

## Environment, tests and lint

```bash
conda activate complybay                       # or any env outside the source tree
pip install meson-python meson ninja numpy
pip install --no-build-isolation -e ".[dev]"   # editable; rebuilds Fortran on import
pytest
ruff check .
ruff format <files you touched>
```

- `src/` layout: pytest imports the **installed** package, so a module missing
  from a `meson.build` file fails a test instead of being hidden by the
  source tree. Every new `.py` file must be listed in the `meson.build` of its
  directory (and a new package needs `subdir(...)` in its parent).
- Keep the Python environment outside the repository; meson refuses include
  paths inside the source tree.
- To check packaging, install non-editable (`pip install ".[dev]"`) into a
  fresh environment and run the suite.
- CI (`.github/workflows/ci.yml`) runs ruff and pytest on Python 3.10/3.12/3.13
  (Linux) and 3.12 (macOS).

## Code layout

Scientific choices must stay visible: equations live in small functions, not
inside IO or orchestration code.

| Layer | Location | Contains | Must not |
|---|---|---|---|
| model | `complybay/model` | frozen dataclasses (`WaterLayer`, `LayeredModel`, `OBSModel`); units fixed and validated on construction | know about any solver |
| forward | `complybay/forward` | pure forward solvers: model in, arrays or small result objects out | read or write files, use global state |
| processing | `complybay/acf` | processing shared by observed and synthetic data (`ACFProcessing`) | depend on a particular data source |
| inversion | `complybay/inversion` (to be rewritten) | parameterization, priors, likelihood, BayesBay adapter | put formulas in the sampler glue; the theta -> model mapping must not depend on the sampler |
| io | future `complybay/io` | file formats, paths, provenance metadata | contain science |
| driver / cli | future | read config, call the layers above, write outputs and logs | contain any formula |

The legacy v0.1 modules (`forward/ncomp_fortran.py`, `k_solver.py`,
`complykernel.py`, `ncomp_matlab_py.py`, `inversion/`, `scripts/`) are kept
only for the old inversion and its baseline tests. Do not build new code on
them; they are listed in ruff's per-file ignores and will be removed when the
inversion layer is rewritten.

## Scientific functions

- Keep the formula in the function body, close to its name; a reader should
  see the method without following several layers of indirection.
- Validate inputs at the top of the function, separately from the computation.
- Pass the scientific parameters explicitly (`decay_cutoff`, `rel_step`,
  `wavenumber=`), not a whole application config.
- State physical conventions in the module docstring: coordinate direction,
  time convention, sign of each output component, normalization. See
  `forward/response.py`.
- Cite the method's source and its range of validity in the docstring.
- Do not mutate inputs. Model arrays are read-only; build modified copies with
  `LayeredModel.scaled` / `with_property` / `OBSModel.with_solid`.
- Optimize only after the scientific result is pinned by a test, and keep the
  test passing.

## Units and naming

- Inside the model and at every public interface: thickness and depth in km,
  velocity in km/s, density in g/cm^3, time in s, frequency in Hz, slowness in
  s/km. Depth 0 is the seafloor.
- A solver that needs other units converts at its own boundary and says so
  (compliance converts to SI for the Fortran propagator).
- Docstrings give the unit of every argument and return value.
- Add a unit suffix only where the unit departs from the convention above or
  is ambiguous (`depth_m`); do not suffix every variable.
- Names carry scientific meaning: `decay_cutoff`, `zero_lag_taper`,
  `water_level`, not `factor`, `eps`, `n`.

## Numerical changes and reproducibility

- A change that alters numbers produced by existing code gets an entry under
  **Changed results** in `CHANGELOG.md`, naming the option that restores the
  old behavior if there is one.
- When old results must stay reproducible, keep the old behavior behind an
  explicitly named option (for example `wavenumber="poly"`); never change a
  default silently.
- Values that are assumptions rather than taken from a source (for example
  parts of the `kim2023` preset) are marked as such where they are defined.
- `tests/data/legacy_v0_1_forward.npz` is the v0.1 regression baseline. It is
  regenerated only with `tools/make_legacy_baseline.py` from tag
  `v0.1-legacy`, never edited to make a test pass.

## Tests

- Every forward solver has at least one independent physical check, not just
  "it runs": an analytic limit, a conservation law, an independent
  implementation, or the legacy baseline. `docs/forward.md` lists the checks
  in place.
- Every tolerance has a reason, written next to it when it is not obvious
  (for example the f2py vs meson build difference in the legacy tests).
- New behavior comes with tests in the same change. Bug fixes come with a test
  that fails without the fix.
- Tests stay fast (the whole suite runs in seconds); long checks go into
  `tools/` scripts or notebooks.

## Style

- ruff enforces style (`pyproject.toml`: line length 100, rules E, F, W, I,
  UP, B). Run `ruff format` on files you touch; do not reformat unrelated
  legacy files.
- Match the surrounding code's idiom and comment density. Comments explain
  why, not what.
- Public functions and classes have type hints and a docstring; private
  helpers start with `_`.
- Modules export their public API through `__all__` in the package
  `__init__.py`.

## Dependencies and licensing

- ComplyBay is MIT licensed. GPL code (for example rein-acfcompute) may be run
  locally as a reference but never copied or ported into this repository.
- Prefer numpy/scipy over a new dependency; a new runtime dependency needs a
  reason in the change description and an entry in `pyproject.toml`.

## Documentation

- `docs/forward.md` describes the forward solvers: conventions, options and
  how each was validated. Update it with any solver change.
- `CHANGELOG.md` records user-visible changes, newest first.
- Demo notebooks under `examples/` are not tracked (`.gitignore`); anything a
  test or another script depends on belongs in `src/`, `tests/` or `tools/`.

## Git workflow

- Develop on `dev`; `main` is a backup of the pre-refactor `dev`, and tag
  `v0.1-legacy` marks the v0.1 code.
- Commit or push only when asked. Commit messages: a short summary line, then
  what changed and why.
