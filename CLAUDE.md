# ComplyBay

Forward modelling and Bayesian inversion for ocean-bottom seismology
(compliance, Rayleigh phase velocity, teleseismic ACF). Full rules:
`docs/development.md`; solver conventions and validation: `docs/forward.md`.

## Commands

```bash
conda activate complybay
pip install --no-build-isolation -e ".[dev]"   # after adding files or touching Fortran
pytest
ruff check .
```

## Rules

- Keep layers apart: `model` (validated frozen dataclasses), `forward` / `acf`
  (pure functions, no file IO), inversion, io, driver/cli. Formulas never go in
  drivers, IO or sampler glue.
- Units at every interface: km, km/s, g/cm^3, s, Hz, s/km; depth 0 = seafloor.
  Convert at a solver's own boundary and document it.
- Put physical conventions (signs, time convention, normalization) and
  references in module docstrings. Do not mutate inputs.
- A change that alters existing numbers needs a `CHANGELOG.md` entry under
  "Changed results" and, when old results must stay reproducible, an explicit
  option that restores them. Never change a default silently.
- Every new or changed solver ships with an independent physical check
  (analytic limit, conservation law, independent implementation or the legacy
  baseline); tolerances carry a reason.
- List every new module in its directory's `meson.build`; tests run against
  the installed package.
- Do not build on the legacy v0.1 modules (`ncomp_fortran`, `k_solver`,
  `complykernel`, `ncomp_matlab_py`, `inversion/`, `scripts/`); never edit
  `tests/data/legacy_v0_1_forward.npz` by hand.
- Never copy GPL code (e.g. rein-acfcompute) into the repository; use it only
  as a locally run reference.
- Mark assumed values (not taken from a source) where they are defined.
- When designing or refactoring, load the project skill
  `research-code-architecture-trial` (`.claude/skills/`), a trial revision of
  the personal `research-code-architecture`; use it instead of the personal one
  here, and note where it helps or gets in the way.
- Work on `dev`; commit or push only when the user asks.
