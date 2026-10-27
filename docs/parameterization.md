# Model parameterization

`complybay.parameterization` turns the sampler's parameter vector into a
seafloor model for the forward solvers:

```text
theta -> ModelParameterization.vector_to_model -> ParameterizedModel (segments + interfaces)
      -> discretize(DiscretizationConfig)      -> LayeredModel
      -> OBSModel(layers, fixed WaterLayer)     -> compliance / phase velocity / ACF
```

The mapping does not depend on any sampler. It is adapted from SeisForge
(`seisforge.inv`, commit 019d568: `model`, `layering`, `transitions`,
`bspline`, `parameterization`, `model_config`); see SeisForge's
`docs/model.md` for the background. Depths are km below the seafloor.

## Configuration

```yaml
Model:
  water: {depth_km: 2.0}          # fixed bathymetry; vp 1.5 km/s, rho 1.03 g/cm^3 by default
  zmax_km: 150.0                  # top of the half-space
  scaling:                        # default relations for segments without their own
    vp: {method: brocher2005}
    rho: {method: brocher2005}
  interfaces:                     # top to bottom
    - name: basement
      depth_km: {parameter: basement.depth}
      width_km: {parameter: basement.width}   # 10-90 % width; 0 = hard step
      shape: sigmoid                          # or smootherstep
    - {name: moho, depth_km: {parameter: moho.depth}}
  segments:                       # one more than interfaces
    - name: sediment
      vs: {type: gradient, top: {parameter: sed.vs_top}, bottom: {parameter: sed.vs_bottom}}
      vp: {type: constant, value: {parameter: sed.vp}}     # or a gradient
      scaling: {rho: {method: gardner1974}}
    - name: crust
      vs: {type: bspline, coefficients: [{parameter: crust.c0}, ...], degree: 3}
    - name: mantle
      vs: {type: gradient, top: {parameter: mantle.vs_top}, bottom: 4.5}
      scaling:
        vp: {method: constant_vp_vs, vp_vs: 1.8}
        rho: {method: constant, rho: 3.35}

Inversion:
  parameters:
    - name: basement.depth
      initial: 0.5
      prior: {type: uniform, bounds: [0.1, 1.5]}
      proposal: {sigma: 0.02}
    - name: basement.width
      initial: 0.0
      prior: {type: half_normal, scale: 0.05, bounds: [0.0, 0.2]}
    ...
```

Any number may be a literal or `{parameter: name}`. Every reference must be
defined, and the initial values must give a valid model; both are checked
when the parameterization is built.

## Segments, profiles and relations

- Profiles (`vs`, and optional `vp` or `vp_vs`): `constant`, `gradient` (values at the
  segment top and bottom), `bspline` (open-clamped on the segment's normalized
  coordinate; first/last coefficients are the end values). Profiles stretch
  with the interfaces.
- Vp: the segment's own `vp` profile, or `vs * vp_vs`, or otherwise the
  segment's Vp relation. Give the sediment its own `vp` profile (constant by
  default): marine sediment Vp stays near 1.5-2 km/s while Vs grows several
  times, so a constant Vp/Vs with a Vs gradient puts Vp below the water
  velocity near the seafloor, and no empirical relation covers Vp/Vs of 3-10.
  A forward test (2 km water, 0.6 km sediment, Vs 0.15-0.6 km/s) showed that
  compliance and 8-30 s phase velocity barely see the shape of the sediment Vp
  profile (< 1 %, < 0.005 km/s); only the ACF does, and a Vp gradient of
  1.52-2.0 km/s versus a constant Vp with the same two-way time differs about
  as much as a 5 % change in sediment thickness.
- Density always comes from the segment's Vp (`brocher2005` Nafe-Drake fit,
  `gardner1974`, `constant`). Unlike SeisForge, a sampled Vp/Vs therefore also
  moves density.
- Vp relations: `brocher2005` (Vs <= 4.5 km/s), `castagna1985` (mudrock line),
  `constant_vp_vs`. Mantle segments need a relation valid above 4.5 km/s.

## Interfaces

A gradational interface blends Vs, Vp and rho of the two neighbours with the
same weight (`sigmoid` or `smootherstep`, compact support). Supports may not
reach the seafloor, overlap a neighbouring interface or pass `zmax_km`; such
parameter values raise `ValueError` when the model is built, which the prior
will turn into log prior = -inf. A `half_normal` width prior makes a hard
interface the default.

## Prior and constraints

`GeophysicalPrior(parameterization, discretization, constraints)` returns
log prior = -inf, with a reason, when

1. a parameter is outside its bounds (`parameter_prior`),
2. the vector does not give a valid model (`model_mapping`): interfaces out of
   order, overlapping transition supports, a relation outside its range, or
   Vp <= sqrt(4/3) Vs in some layer,
3. a hard constraint fails (`physical_constraints: <names>`).

It never raises, and a valid evaluation keeps the built `OBSModel` for the
likelihood.

```yaml
Constraints:
  - {type: vp_vs_range, bounds: [1.6, 15.0]}
  - {type: vs_range, bounds: [0.05, 1.5], segments: [sediment]}
  - {type: boundary_non_decreasing_vs}
  - {type: monotonic_vs, segments: [crust]}
  - {type: max_vs_gradient, max_abs: 5.0}        # km/s per km
  - {type: vp_above_water, margin: 0.0}          # Vp below the seafloor >= water Vp
```

With `segments`, a constraint checks those segments' own profiles over their
own intervals, so it moves with sampled interfaces and ignores the blend
with a neighbour inside a transition. Without `segments` it checks the whole
blended model (so `max_vs_gradient` then also bounds jump / transition width).

`sample_prior(prior, n, rng=...)` draws parameters from their priors and keeps
the vectors inside the full prior: exact independent prior samples, the
acceptance rate, and the rejection reasons. Use them to look at the model
space before any data enters and as diverse MCMC starting points; a very low
acceptance means bounds and constraints contradict each other.
SeisForge's soft (curvature) prior is not ported yet.

## Layering

`DiscretizationConfig(layers_per_segment={...}, transition_layers=16)`:

- hard interfaces are always layer boundaries;
- with sampled interface depths use `layers_per_segment` rather than `dz`, so
  the layer geometry moves continuously with the interfaces (a `dz` grid
  changes the layer count in steps and puts steps into the likelihood);
- `layering_report(model, layered)` gives the vertical travel-time error of
  the layering; keep it well below the timing resolution of the ACF.

Cost of theta -> `OBSModel` for a 38-layer model: about 0.3 ms, small next to
the forward solvers (about 5 ms for compliance + dispersion + Z/P-ACF).
