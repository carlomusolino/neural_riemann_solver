import os
import pickle
import pandas as pd

from rimeannML.training_utils.training_losses import *


def collect_results(base_dir="../models_new", problems=["prob_1", "prob_2", "prob_3", "prob_4"]):
    data = []

    for layer in [2]:  # Or parametrize this
        for n_neurons in [64, 128, 256, 512]:
            for dset_size in [2**x for x in range(17, 24)]:
                dset_dir = os.path.join(base_dir, f"{layer}_layers_{n_neurons}_neurons", f"{dset_size}_samples")
                if not os.path.isdir(dset_dir):
                    continue

                for seed in [42] + list(range(100, 200, 10)):
                    seed_dir = os.path.join(dset_dir, str(seed))
                    if not os.path.isdir(seed_dir):
                        continue

                    for prob in problems:
                        err_path = os.path.join(seed_dir, f"{prob}_fo", "errors.pkl")
                        time_path = os.path.join(seed_dir, f"{prob}_fo", "timers.pkl")
                        try:
                            with open(err_path, "rb") as f:
                                errors = pickle.load(f)
                            l1_err = errors["l1"]["ml"][0]  # First index corresponds to 800 zones

                            # Initialize timing info in case file is missing
                            riemann_time = None
                            total_time = None

                            if os.path.isfile(time_path):
                                with open(time_path, "rb") as f:
                                    timers = pickle.load(f)
                                tdict = timers[800]['ml']['timers']
                                cdict = timers[800]['ml']['counters']

                                riemann_time = tdict['riemann'] / cdict['riemann'] if cdict['riemann'] > 0 else None
                                total_time = sum(tdict[k] / cdict[k] for k in ['c2p', 'recon', 'flux'] if cdict[k] > 0)

                            data.append({
                                "layers": layer,
                                "neurons": n_neurons,
                                "dset_size": dset_size,
                                "seed": seed,
                                "problem": prob,
                                "l1_error": l1_err,
                                "riemann_time": riemann_time,
                                "total_time": total_time
                            })
                        except Exception as e:
                            print(f"Failed to process: {err_path} -- {e}")
                            continue

    return pd.DataFrame(data)

def evaluate_model_percentiles(bdir, solver_name, gamma=5./3., nsamples=100_000, percentiles=[50, 90, 99], seed=42, device=None):
    import torch
    import os
    import yaml
    from .generate_data import generate_dataset, generate_dataset_raref, filter_continuous
    from exact.hybrid_eos import hybrid_eos
    from exact.riemann_solver import classify_riemann_problem
    from solvers.MLP import RootfindMLP, Compactification
    import numpy as np

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    # Load normalization info
    input_min = torch.load(os.path.join(bdir, 'input_min.pt'), map_location=device)
    input_max = torch.load(os.path.join(bdir, 'input_max.pt'), map_location=device)

    # Load config
    with open(os.path.join(bdir, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    n_neurons = config['n_neurons']
    n_layers = config['n_layers']

    # Construct appropriate model
    model_path = os.path.join(bdir, solver_name, "checkpoints", "best_model.pt")
    input_dim = 6 if solver_name not in ['raref_left', 'raref_right'] else 3
    d_ff = n_neurons if input_dim == 6 else n_neurons // 2

    from solvers.MLP import RootfindMLP  # Adjust if needed
    model = RootfindMLP(input_dim=input_dim, output_dim=1, d_ff=d_ff, depth=n_layers,
                        input_min=input_min, input_max=input_max,
                        output_mapping=torch.sigmoid).to(device).to(dtype)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    wave_configs = {
        "d_shock": {
            "pjumpmin": 0.1,
            "pjumpmax": 1.5,
            "loss_fn": compute_loss__d_shock
        },
        "d_raref": {
            "pjumpmin": 0.1,
            "pjumpmax": 1,
            "loss_fn": compute_loss__d_raref
        },
        "rarefshock": {
            "pjumpmin": 0.1,
            "pjumpmax": 7.5,
            "loss_fn": compute_loss__rarefshock
        },
        "raref_left": dict(),
        "raref_right": dict()
    }
    # Generate dataset
    convfac = np.log10(np.exp(1))
    lrhomax = input_max[0].item() * convfac
    lrhomin = input_min[0].item() * convfac
    lpressmax = input_max[1].item() * convfac
    lpressmin = input_min[1].item() * convfac
    velmax = input_max[2].item()
    velmin = input_min[2].item()
    
    config = wave_configs[solver_name]
    if solver_name not in ['raref_left', 'raref_right']:
        
        dataset = generate_dataset(
            lpressmin, lpressmax,
            config['pjumpmin'], config['pjumpmax'],
            lrhomin, lrhomax, velmin, velmax,
            solver_name, nsamples, device, gamma, dtype, seed=seed
        )
        dataset = filter_continuous(dataset, 1e-12)
        eos = hybrid_eos(0,2,gamma)
        total_dset_size = dataset.size(0)
        labels, mask, p_c = classify_riemann_problem(
            torch.exp(dataset[:,0,:]), 
            torch.exp(dataset[:,1,:]), 
            dataset[:,2,:], eos, torch.float64, tol=1e-15
        )  
        labels_flat = torch.argmax(labels[mask].to(torch.int64), dim=1) 
        prims  = dataset[mask,:,:]
        p_c = p_c[mask]
        # Separate wave patterns
        if solver_name == 'd_shock':
            target_label = 0 
        elif solver_name == 'd_raref':
            target_label = 1
        elif solver_name == 'rarefshock':
            target_label = 2 
        wave_mask = (labels_flat == target_label)
        prims = prims[wave_mask,:,:]
        p_c   = p_c[wave_mask]
        
        if solver_name == 'd_shock':
            mapping = Compactification(method='compact', a=torch.max(torch.exp(prims[:,1,0]), torch.exp(prims[:,1,1])))
        elif solver_name == 'd_raref':
            mapping = Compactification(method='affine', a=torch.zeros_like(p_c, device=p_c.device, dtype=p_c.dtype), b=torch.min(torch.exp(prims[:,1,0]), torch.exp(prims[:,1,1])))
        elif solver_name == 'rarefshock':
            mapping = Compactification(method='affine', a=torch.min(torch.exp(prims[:,1,0]), torch.exp(prims[:,1,1])),b=torch.max(torch.exp(prims[:,1,0]), torch.exp(prims[:,1,1])))
        
        with torch.no_grad():
            y_pred = mapping.xi_to_x(model(prims).squeeze(-1))

        loss_values = ((torch.log(y_pred) - torch.log(p_c))**2).abs().cpu().numpy()

    else:
        dataset_L, dataset_R = generate_dataset_raref(
        lpressmin,lpressmax,lrhomin,lrhomax,velmin,velmax,nsamples,device,gamma,dtype,seed
        )
        if solver_name == 'raref_left':
            dataset = dataset_L[0]
            cs_true = dataset_L[1]
            rho   = torch.exp(dataset[:,0])
            press = torch.exp(dataset[:,1])
            csa   = get_csnd(rho,press,gamma)
            xi_pred = model(dataset).squeeze(-1)
            mapping = Compactification(method="affine", a=torch.zeros(dataset.size(0),device=cs_true.device,dtype=cs_true.dtype), b=csa)
            with torch.no_grad(): cs_pred = mapping.xi_to_x(xi_pred)
            loss_values = (cs_pred - cs_true).abs().cpu().numpy()

        else:
            dataset = dataset_R[0]
            cs_true = dataset_R[1]
            rho   = torch.exp(dataset[:,0])
            press = torch.exp(dataset[:,1])
            csa   = get_csnd(rho,press,gamma)
            xi_pred = model(dataset).squeeze(-1)
            mapping = Compactification(method="affine", a=torch.zeros(dataset.size(0),device=cs_true.device,dtype=cs_true.dtype), b=csa)
            with torch.no_grad(): cs_pred = mapping.xi_to_x(xi_pred)
            loss_values = (cs_pred - cs_true).abs().cpu().numpy()

    # Return selected percentiles
    return {
        'p50': np.percentile(loss_values, 50),
        'p90': np.percentile(loss_values, 90),
        'p99': np.percentile(loss_values, 99),
        'mean': np.mean(loss_values),
        'std': np.std(loss_values),
        'max': np.max(loss_values),
        'p99_p50': np.percentile(loss_values, 99) / np.percentile(loss_values, 50),
        'std_over_mean': np.std(loss_values) / np.mean(loss_values),
    }


def evaluate_directory_percentiles(bdir, seed, train_dset_size, neurons, layers=2, solvers=("d_shock", "d_raref", "rarefshock"), gamma=5./3., nsamples=100_000):
    import pandas as pd
    data = []

    dir = os.path.join(bdir, f'{layers}_layers_{neurons}_neurons', f'{train_dset_size}_samples', str(seed))
    
    for solver in solvers:
        try:
            percentiles = evaluate_model_percentiles(dir, solver, gamma=gamma, nsamples=nsamples, seed=seed)
            data.append({
                'layers': layers,
                "neurons": neurons,
                "dset_size": train_dset_size,
                "seed": seed,
                'solver': solver,
                'eval_dataset_size': nsamples,
                'p50': percentiles['p50'],
                'p90': percentiles['p90'],
                'p99': percentiles['p99'],
                'mean': percentiles['mean'],
                'std': percentiles['std'],
                'max': percentiles['max'],
                'p99_p50': percentiles['p99_p50'],
                'std_over_mean': percentiles['std_over_mean']
            })
        except Exception as e:
            print(f"Failed to evaluate {solver} in {bdir}: {e}")

    return pd.DataFrame(data)
