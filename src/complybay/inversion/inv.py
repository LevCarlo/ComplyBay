import bayesbay as bb
import numpy as np
from bayesbay.discretization import Voronoi1D
from disba import PhaseDispersion

from complybay.forward.ncomp_fortran import ncomp_fortran_batch
from complybay.inversion.model import RhoEstimate, VpEstimate


def init_variables(para, positions=None):
    vmin, vmax = para.get_vmin_vmax(positions)
    sorted_vals = np.sort(np.random.uniform(vmin, vmax, positions.size))
    return sorted_vals


def parametrize_model(param, vs_init=None, vp_init=None, rho_init=None):
    paramters = []
    for v, key in zip([vs_init, vp_init, rho_init], ["vs", "vp", "rho"]):
        if v is not None:
            v_depth = v[:, 0]
            v_min = v[:, 2]
            v_max = v[:, 3]
            v_perturb_std = v[:, 4]

            prior = bb.prior.UniformPrior(
                name=key,
                vmin=v_min,
                vmax=v_max,
                position=v_depth,
                perturb_std=v_perturb_std
            )
            prior.set_custom_initialize(init_variables)
            paramters.append(prior)


    voronoi_param = param["INVERSION"]["MODEL"]["VORONOI"]
    voronoi = Voronoi1D(
        name="voronoi",
        vmin=voronoi_param["min_depth"],
        vmax=voronoi_param["max_depth"],
        perturb_std=voronoi_param["perturb_std"],
        n_dimensions=voronoi_param["n_dimensions"],
        n_dimensions_min=voronoi_param["n_dimensions_min"],
        n_dimensions_max=voronoi_param["n_dimensions_max"],
        parameters=paramters,
        birth_from=voronoi_param["birth_from"]
    )

    parameterization = bb.parameterization.Parameterization(voronoi)
    return parameterization


def forward_comply(state, freqs=None, wdepth=None, param=None, k_method='poly', log_scale=False):
    voronoi = state["voronoi"]
    voronoi_sites = voronoi["discretization"]

    voronoi_param = param["INVERSION"]["MODEL"]["VORONOI"]
    thickness = Voronoi1D.compute_cell_extents(
        voronoi_sites,
        lb=voronoi_param["min_depth"],
        ub=voronoi_param["max_depth"]
    )
    vs = voronoi["vs"]
    # Custom prior constraints: Vs monotonically increasing when depth > 1000 m
    # ========================================================================
    # depths = np.cumsum(thickness)
    # mask = depths >= 800
    # if np.any(np.diff(vs[mask]) < 0):
    #     pred = np.full_like(freqs, -np.inf)
    #     return pred
    # ========================================================================


    model_param = param["INVERSION"]["MODEL"]
    if model_param["VP"]["inverse"]:
        vp = voronoi["vp"]
    else:
        vp_estimator = VpEstimate(vs=vs)
        vp = vp_estimator.estimate(model_param["VP"]["method"])


    if model_param["RHO"]["inverse"]:
        rho = voronoi["rho"]
    else:
        rho_estimator = RhoEstimate(vs=vs, vp=vp)
        rho = rho_estimator.estimate(model_param["RHO"]["method"])

    # print(np.column_stack((thickness, vs)))
    # print(np.cumsum(thickness))
    # print("=" * 50)
    model = np.column_stack((thickness, rho, vp, vs))
    pred = ncomp_fortran_batch(
        depth=wdepth,
        freqs=freqs,
        model=model,
        k_method=k_method,
        log_scale=log_scale
    )
    if log_scale:
        if np.any(~np.isfinite(pred)) or np.any(np.isnan(pred)):
            pred = np.full_like(pred, -np.inf)
    else:
        if np.any(pred <= 0) or np.any(~np.isfinite(pred)) or np.any(np.isnan(pred)):
            pred = np.full_like(pred, -np.inf)

    return pred

def forward_sw(state, param, periods=None, wave="rayleigh", mode=0, wdepth=None):
    voronoi = state["voronoi"]
    voronoi_sites = voronoi["discretization"]

    voronoi_param = param["INVERSION"]["MODEL"]["VORONOI"]
    thickness = Voronoi1D.compute_cell_extents(
        voronoi_sites,
        lb=voronoi_param["min_depth"],
        ub=voronoi_param["max_depth"]
    )
    vs = voronoi["vs"]

    model_param = param["INVERSION"]["MODEL"]
    if model_param["VP"]["inverse"]:
        vp = voronoi["vp"]
    else:
        vp_estimator = VpEstimate(vs=vs)
        vp = vp_estimator.estimate(model_param["VP"]["method"])
    if model_param["RHO"]["inverse"]:
        rho = voronoi["rho"]
    else:
        rho_estimator = RhoEstimate(vs=vs)
        rho = rho_estimator.estimate(model_param["RHO"]["method"])

    # add water layer
    thickness = np.concatenate((np.array([wdepth]), thickness))
    vs = np.concatenate((np.array([0]), vs))
    vp = np.concatenate((np.array([1.5]), vp))  # Water velocity in km/s
    rho = np.concatenate((np.array([1.025]), rho))  # Density of water in g/cm^3

    model = np.column_stack((thickness * 1e-3, vp, vs, rho))  # Convert thickness to km for disba compatibility
    pd = PhaseDispersion(*model.T)
    d_pred = pd(periods, mode=mode, wave=wave).velocity
    if np.any(d_pred <= 0) or np.any(~np.isfinite(d_pred)) or np.any(np.isnan(d_pred)):
        d_pred = np.full_like(d_pred, -np.inf)
    return d_pred
