import numpy as np
from ComplyBay.forward.ncomp_fortran import ncomp_fortran_batch
from ComplyBay.inversion.model import VpEstimate, RhoEstimate
import bayesbay as bb
from bayesbay.discretization import Voronoi1D

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
        n_dimensions=None,
        n_dimensions_min=voronoi_param["n_dimensions_min"],
        n_dimensions_max=voronoi_param["n_dimensions_max"],
        parameters=paramters,
        birth_from=voronoi_param["birth_from"]
    )

    parameterization = bb.parameterization.Parameterization(voronoi)
    return parameterization
    

def forward_comply(state, freqs=None, wdepth=None, param=None):
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

    model = np.column_stack((thickness, rho, vp, vs))
    pred = ncomp_fortran_batch(wdepth, freqs, model)
    return pred
    