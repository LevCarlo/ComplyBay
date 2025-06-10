import os
from glob import glob
import numpy as np
import yaml
import argparse
from ComplyBay.inversion.inv import parametrize_model, forward_comply
from functools import partial
from bayesbay.likelihood import Target, LogLikelihood
import bayesbay as bb
import pickle

def load_parse_args():
    parser = argparse.ArgumentParser(description="Run Comply Inversion")
    parser.add_argument(
        "-p", "--param_file", type=str, required=True, default="complybay.yaml", help="parameter file"
    )
    parser.add_argument(
        "-s", "--station", type=str, required=False, default=None, help="station"
    )
    args = parser.parse_args()
    return args

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

def inversion_init(param, station=None):
    IO_params = param["IO"]
    if IO_params["ROOT"] is None:
        IO_params["ROOT"] = os.getcwd()

    comply_data = load_comply_data(param, station)
    freqs = comply_data["freqs"]
    comply = comply_data["comply"]
    comply_err = comply_data["comply_err"]
    wdepth = comply_data["wdepth"]

    inversion_params = param["INVERSION"]
    model_params = inversion_params["MODEL"]
    vs_init, vp_init, rho_init = None, None, None
    if model_params["VS"]["inverse"]:
        vs_init_file = glob(os.path.join(IO_params["ROOT"], IO_params["SRC_REC"], f"{station}_vs_init*"))
        vs_init = np.loadtxt(vs_init_file[0], skiprows=1) if vs_init_file else None
    if model_params["VP"]["inverse"]:
        vp_init_file = glob(os.path.join(IO_params["ROOT"], IO_params["SRC_REC"], f"{station}_vp_init*"))
        vp_init = np.loadtxt(vp_init_file[0], skiprows=1) if vp_init_file else None
    if model_params["RHO"]["inverse"]:
        rho_init_file = glob(os.path.join(IO_params["ROOT"], IO_params["SRC_REC"], f"{station}_rho_init*"))
        rho_init = np.loadtxt(rho_init_file[0], skiprows=1) if rho_init_file else None
    
    parameterization = parametrize_model(
        param=param,
        vs_init=vs_init,
        vp_init=vp_init,
        rho_init=rho_init
    )

    forward_func = partial(forward_comply, freqs=freqs, wdepth=wdepth, param=param)
    target = Target(
        name="Comply", 
        dobs=comply, 
        covariance_mat_inv=1 / comply_err**2)
    log_likelihood = LogLikelihood(
        targets=target, 
        fwd_functions=forward_func)
    
    inv_hyperparams = param["INVERSION"]["HYPERPARAMETERS"]
    inversion = bb.BayesianInversion(
        parameterization=parameterization,
        log_likelihood=log_likelihood,
        n_chains=inv_hyperparams["n_chains"]
    )
    return inversion

def main():
    args = load_parse_args()
    with open(args.param_file, 'r') as f:
        param = yaml.safe_load(f)
    
    station = args.station
    
    inversion = inversion_init(param, station)
    inv_hyperparams = param["INVERSION"]["HYPERPARAMETERS"]
    inversion.run(
        sampler=None,
        n_iterations=inv_hyperparams["n_iterations"],
        burnin_iterations=inv_hyperparams["burnin_iterations"],
        save_every=inv_hyperparams["save_every"],
        verbose=inv_hyperparams["verbose"]
    )

    savedir = os.path.join(param["IO"]["ROOT"], param["IO"]["SOLVER"])
    os.makedirs(savedir, exist_ok=True)
    chains = {}
    for i, chain in enumerate(inversion.chains):
        saved_states = chain.saved_states
        chains[f"chain_{i}"] = saved_states
    with open(os.path.join(savedir, f"{station}_chains.pkl"), 'wb') as f:
        pickle.dump(chains, f)
    
    results = inversion.get_results(concatenate_chains=True)
    with open(os.path.join(savedir, f"{station}_results.pkl"), 'wb') as f:
        pickle.dump(results, f)