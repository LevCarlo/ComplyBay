import os
from glob import glob
import yaml
import argparse
import numpy as np
from ComplyBay.forward.ncomp_fortran import ncomp_fortran_batch
from ComplyBay.inversion.model import VpEstimate, RhoEstimate, layer2profile
import bayesbay as bb
from bayesbay.discretization import Voronoi1D
from bayesbay.likelihood import Target, LogLikelihood
from functools import partial
import pickle

def read_param(filepath):
    with open(filepath, 'r') as f:
        param = yaml.safe_load(f)

    if param["IO"]["ROOT"] is None:
        param["IO"]["ROOT"] = os.path.dirname(os.path.abspath(filepath))
    
    return param

def load_comply_data(param, station=None):
    comply_datadir = os.path.join(param["IO"]["ROOT"], param["IO"]["COMPLY_DATA"])
    comply_file = glob(os.path.join(comply_datadir, f"{station}*"))
    if len(comply_file) != 1:
        raise ValueError(f"Multiple or no comply files found in {comply_datadir}")
    
    comply_dat = np.loadtxt(comply_file[0], skiprows=1)
    freqs, comply, comply_err = comply_dat[:, 0], comply_dat[:, 1], comply_dat[:, 2]

    src_rec_datadir = os.path.join(param["IO"]["ROOT"], param["IO"]["SRC_REC"])
    source_file = glob(os.path.join(src_rec_datadir, f"*sources*"))
    if len(source_file) != 1:
        raise ValueError(f"Multiple or no source files found in {src_rec_datadir}")

    sources = np.genfromtxt(source_file[0], dtype=str)
    wdepth = float(sources[sources[:, 0] == station, -1])
    wdepth = wdepth if wdepth > 0 else -wdepth

    return {
        "freqs": freqs,
        "comply": comply,
        "comply_err": comply_err,
        "wdepth": wdepth,
    }
    
def load_initial_model(param, station=None):
    src_rec_datadir = os.path.join(param["IO"]["ROOT"], param["IO"]["SRC_REC"])
    initial_model_file = glob(os.path.join(src_rec_datadir, f"{station}*init.mod"))
    if len(initial_model_file) != 1:
        raise ValueError(f"Multiple or no initial model files found in {src_rec_datadir}")
    initial_model = np.loadtxt(initial_model_file[0], skiprows=1)
    return initial_model

def init_variables(param, positions=None):
    vmin, vmax = param.get_vmin_vmax(positions)
    sorted_vals = np.sort(np.random.uniform(vmin, vmax, positions.size))
    return sorted_vals


def parametrize_model(param, station=None):
    comply_data = load_comply_data(param, station)
    initial_model = load_initial_model(param, station)

    # Prior initialization
    # thick, _vs = initial_model[:, 0], initial_model[:, 2]
    # d = np.linspace(0, np.sum(thick), 6)
    # vs = layer2profile(thickness=thick, values=_vs, z=d, interp_method="cubic")
    d = initial_model[:, 0]
    vs = initial_model[:, 2]
    
    vs_prior_param = param["INVERSION"]["MODEL"]["VS"]["PRIOR"]
    vs_min = np.maximum(vs_prior_param["vmin"], vs - vs_prior_param["vmin_offset"])
    vs_max = np.minimum(vs_prior_param["vmax"], vs + vs_prior_param["vmax_offset"])
    # vs_perturb_std = vs * vs_prior_param["perturb_in_percent"] / 100
    vs_perturb_std = 0.15
    vs_prior = bb.prior.UniformPrior(
        name="vs",
        vmin=vs_min,
        vmax=vs_max,
        position=d,
        perturb_std=vs_perturb_std,
    )
    vs_prior.set_custom_initialize(init_variables)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(3, 6))
    # depth_prior = np.linspace(0, 5000)
    # vmin_prior, vmax_prior = vs_prior.get_vmin_vmax(depth_prior)
    # ax.fill_betweenx(depth_prior, vmin_prior, vmax_prior, color='lightgray', label='Vs Prior Range')
    # ax.plot(vs, d)
    # ax.set_xlabel("Vs [km/s]")
    # ax.invert_yaxis()
    # fig.suptitle(f"Vs Prior for {station}")
    # fig.savefig(f"vs_prior_{station}.png", dpi=300)
    # exit()
    
    voronoi_param = param["INVERSION"]["MODEL"]["VORONOI"]
    voronoi = Voronoi1D(
        name="voronoi",
        vmin=0,
        vmax=voronoi_param["max_depth"],
        perturb_std=voronoi_param["perturb_std"],
        n_dimensions=None,
        n_dimensions_min=voronoi_param["n_dimensions_min"],
        n_dimensions_max=voronoi_param["n_dimensions_max"],
        parameters=[vs_prior],
        birth_from=voronoi_param["birth_from"]
    )
    parameterization = bb.parameterization.Parameterization(voronoi)

    return parameterization

def forward_comply(state, freqs=None, wdepth=None):
    voronoi = state["voronoi"]
    voronoi_sites = voronoi["discretization"]
    thickness = Voronoi1D.compute_cell_extents(voronoi_sites)
    vs = voronoi["vs"]

    z = np.concatenate(([0], np.cumsum(thickness)))
    f_vs = layer2profile(thickness=thickness, values=vs, z=z, interp_method="cubic")

    vp_estimator = VpEstimate(vs=vs)
    rho_estimator = RhoEstimate(vs=vs)
    vp = vp_estimator.estimate("Brocher2005")
    rho = rho_estimator.estimate("Brocher2005")
    model = np.column_stack((thickness, rho, vp, vs))
    pred = ncomp_fortran_batch(wdepth, freqs, model)
    return pred



def main():
    parser = argparse.ArgumentParser(description="Comply Inversion with Bayesbay")
    parser.add_argument(
        "-p", "--param", type=str, required=True, default="complybay.yaml", help="parameter file"
    )
    parser.add_argument(
        "-s", "--station", type=str, required=False, default=None, help="station"
    )
    args = parser.parse_args()
    param_file = args.param
    param = read_param(param_file)
    comply_data = load_comply_data(param, args.station)
    initial_model = load_initial_model(param, args.station)
    freqs = comply_data["freqs"]
    comply = comply_data["comply"]
    comply_err = comply_data["comply_err"]
    wdepth = comply_data["wdepth"]


    forward_func = partial(forward_comply, freqs=freqs, wdepth=wdepth)
    target = Target(name="Comply", dobs=comply, covariance_mat_inv=1 / comply_err**2)
    log_likelihood = LogLikelihood(targets=target, fwd_functions=forward_func)
    parameterization = parametrize_model(param, args.station)

    inv_hyperparam = param["INVERSION"]["HYPERPARAMETERS"]

    inversion = bb.BayesianInversion(
        parameterization=parameterization,
        log_likelihood=log_likelihood,
        n_chains= inv_hyperparam["n_chains"]
    )

    inversion.run(
        sampler=None,
        n_iterations=inv_hyperparam["n_iterations"],
        burnin_iterations= inv_hyperparam["burnin_iterations"],
        save_every=inv_hyperparam["save_every"],
        verbose=inv_hyperparam["verbose"]
    )

    savedir = os.path.join(param["IO"]["ROOT"], param["IO"]["DATA"])
    os.makedirs(savedir, exist_ok=True)
    chains = {}
    for i, chain in enumerate(inversion.chains):
        saved_states = chain.saved_states
        chains[f"chain_{i}"] = saved_states
    with open(os.path.join(savedir, f"{args.station}_chains.pkl"), "wb") as f:
        pickle.dump(chains, f)
    
    results = inversion.get_results(concatenate_chains=True)
    with open(os.path.join(savedir, f"{args.station}_results.pkl"), "wb") as f:
        pickle.dump(results, f)
    

    