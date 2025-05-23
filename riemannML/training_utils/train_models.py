# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init 
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np
import matplotlib.pyplot as plt
from riemannML.data.generate_data import generate_dataset, generate_dataset_raref, filter_continuous, get_coverage
from riemannML.exact.riemann_solver import classify_riemann_problem
from riemannML.exact.hybrid_eos import hybrid_eos
from .train_MTL import train_model_PCGrad
from .training_losses import compute_loss__d_raref, compute_loss__d_shock, compute_loss__raref_solver, compute_loss__rarefshock
import yaml 

from riemannML.solvers.MLP import RootfindMLP

import os 
import argparse
import pickle

def train_raref_solvers(
    seed, n_layers, n_neurons, N,
    input_min, input_max,
    gamma,lr,
    clip_grads, grad_norm, batch_size,
    device, model_basepath=None):
    if model_basepath is None:
        model_basepath = os.path.join('../models',str(seed))
    dtype = torch.float64
    
    # Scaling for net input (always the same)
    convfac = np.log10(np.exp(1)) # Conversion from log to log10

    lrhomax   =  input_max[0]*convfac
    lrhomin   =  input_min[0]*convfac
    lpressmax =  input_max[1]*convfac
    lpressmin =  input_min[1]*convfac
    velmax    =  input_max[2]
    velmin    =  input_min[2]
    
    # Generate dataset using Sobol sequences
    dataset_L, dataset_R = generate_dataset_raref(
        lpressmin,lpressmax,lrhomin,lrhomax,velmin,velmax,N,device,gamma,dtype,seed
    )
    
    # ------------------ CONFIG ------------------
    epochs = 100
    # ------------------ SPLIT DATA ------------------
    dataset = TensorDataset(dataset_L[0].to(dtype), dataset_L[1].to(dtype))
    train_size = int(0.8 * len(dataset))  # 80% train, 20% test
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # ------------------ LOADERS ------------------
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    modelpath = os.path.join(model_basepath,"raref_left")
    checkpoint_dir = os.path.join(modelpath, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = RootfindMLP(input_dim=3,output_dim=1, output_mapping=torch.sigmoid, input_min=input_min, input_max=input_max, d_ff=n_neurons, depth=n_layers).to(device).to(dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    train_loss, test_loss = train_model_PCGrad(
        model,
        optimizer,
        scheduler,
        train_loader,
        test_loader,
        epochs,
        checkpoint_dir,
        2,
        compute_loss__raref_solver,
        clip_grads,
        grad_norm,
        +1,
        gamma
        )  
    
    # Save loss trajectories
    train_loss_root, train_loss_residual = [], []
    test_loss_root, test_loss_residual = [], []

    for train, test in zip(train_loss,test_loss):
        train_loss_root.append(train[1])
        train_loss_residual.append(train[0])
        test_loss_root.append(test[1])
        test_loss_residual.append(test[0])
        
    losses = {
        'training': { 
            'root': np.array(train_loss_root),
            'residual': np.array(train_loss_residual)
        },
        'test': { 
            'root': np.array(test_loss_root),
            'residual': np.array(test_loss_residual)
        },
    }
    
    with open(os.path.join(modelpath,'losses.pkl'), 'wb') as f:
        pickle.dump(losses,f)
        
    # Right going rarefactions 
    # ------------------ SPLIT DATA ------------------
    dataset = TensorDataset(dataset_R[0].to(dtype), dataset_R[1].to(dtype))
    train_size = int(0.8 * len(dataset))  # 80% train, 20% test
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # ------------------ LOADERS ------------------
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    modelpath = os.path.join(model_basepath,"raref_right")
    checkpoint_dir = os.path.join(modelpath, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = RootfindMLP(input_dim=3,output_dim=1, output_mapping=torch.sigmoid, input_min=input_min, input_max=input_max, d_ff=n_neurons, depth=n_layers).to(device).to(dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    train_loss, test_loss = train_model_PCGrad(
        model,
        optimizer,
        scheduler,
        train_loader,
        test_loader,
        epochs,
        checkpoint_dir,
        2,
        compute_loss__raref_solver,
        clip_grads,
        grad_norm,
        -1,
        gamma
        ) 
    
    # Save loss trajectories
    train_loss_root, train_loss_residual = [], []
    test_loss_root, test_loss_residual = [], []

    for train, test in zip(train_loss,test_loss):
        train_loss_root.append(train[1])
        train_loss_residual.append(train[0])
        test_loss_root.append(test[1])
        test_loss_residual.append(test[0])
        
    losses = {
        'training': { 
            'root': np.array(train_loss_root),
            'residual': np.array(train_loss_residual)
        },
        'test': { 
            'root': np.array(test_loss_root),
            'residual': np.array(test_loss_residual)
        },
    }
    
    with open(os.path.join(modelpath,'losses.pkl'), 'wb') as f:
        pickle.dump(losses,f)
        
    # All done!
    return 

def train_model(seed,
                n_layers,
                n_neurons,
                input_min,
                input_max,
                pjumpmin, 
                pjumpmax, 
                gamma,
                N,
                wave_type,
                loss_fn,
                lr,
                clip_grads,
                grad_norm,
                batch_size,
                device,
                model_basepath=None):
    
    if model_basepath is None:
        model_basepath = os.path.join('../models',str(seed))
    
    dtype = torch.float64
    # Scaling for net input (always the same)
    convfac = np.log10(np.exp(1)) # Conversion from log to log10

    lrhomax   =  input_max[0]*convfac
    lrhomin   =  input_min[0]*convfac
    lpressmax =  input_max[1]*convfac 
    lpressmin =  input_min[1]*convfac
    velmax    =  input_max[2]
    velmin    =  input_min[2]

    
    # Generate dataset using Sobol sequences
    dataset = generate_dataset(lpressmin,
                               lpressmax,
                               pjumpmin,
                               pjumpmax,
                               lrhomin,
                               lrhomax,
                               velmin,
                               velmax,
                               wave_type,
                               N,
                               device,
                               gamma,
                               dtype,
                               seed)
    
    modelpath = os.path.join(model_basepath,wave_type)
    
    # Filter out continuous data samples 
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
    
    print(f"Generated training data for {wave_type} MLP")
    print(f"Converged p_c {torch.sum(mask)}/{total_dset_size}")
    print(f"  Max log10(rho) {torch.max(prims[:,0,:])*convfac}, min log10(rho) {torch.min(prims[:,0,:])*convfac}")
    print(f"  Max log10(p) {torch.max(prims[:,1,:])*convfac}, min log10(p) {torch.min(prims[:,1,:])*convfac}")
    print(f"  Max vel {torch.max(prims[:,2,:])}, min vel {torch.min(prims[:,2,:])}")
    
    # Separate wave patterns
    if wave_type == 'd_shock':
        target_label = 0 
    elif wave_type == 'd_raref':
        target_label = 1
    elif wave_type == 'rarefshock':
        target_label = 2 
    train_mask = (labels_flat == target_label)
    prims_train = prims[train_mask,:,:]
    p_c_train   = p_c[train_mask]
    
    print(f"Final dataset size: {p_c_train.size(0)}")
    
    print(f"Sobol dataset coverage: {get_coverage(prims,10)*100}%")
    
    # ------------------ CONFIG ------------------
    epochs = 100
    # ------------------ SPLIT DATA ------------------
    dataset = TensorDataset(prims_train.to(dtype), p_c_train.to(dtype))
    train_size = int(0.8 * len(dataset))  # 80% train, 20% test
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # ------------------ LOADERS ------------------
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(dataset, shuffle=True, generator=g)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    checkpoint_dir = os.path.join(modelpath,"checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure checkpoint directory exists

    # ------------------ MODEL + OPTIMIZER ------------------
    model = RootfindMLP( input_dim=6, output_dim=1, d_ff=n_neurons, depth=n_layers, input_max=input_max.to(dtype), input_min=input_min.to(dtype), output_mapping=F.sigmoid).to(device).to(dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    train_loss, test_loss = train_model_PCGrad(model,optimizer,scheduler,train_loader,test_loader,epochs,checkpoint_dir, 2, loss_fn, clip_grads, grad_norm, gamma)
    
    # Save loss trajectories
    train_loss_root, train_loss_residual = [], []
    test_loss_root, test_loss_residual = [], []

    for train, test in zip(train_loss,test_loss):
        train_loss_root.append(train[1])
        train_loss_residual.append(train[0])
        test_loss_root.append(test[1])
        test_loss_residual.append(test[0])
        
    losses = {
        'training': { 
            'root': np.array(train_loss_root),
            'residual': np.array(train_loss_residual)
        },
        'test': { 
            'root': np.array(test_loss_root),
            'residual': np.array(test_loss_residual)
        },
    }
    
    with open(os.path.join(modelpath,'losses.pkl'), 'wb') as f:
        pickle.dump(losses,f)
    
    # All done! 
    return    

def train_models(seed, 
                 gamma=5./3., 
                 N=200_000, 
                 lrhomin=-2, lrhomax=2,
                 lpressmin=-7, lpressmax=5,
                 velmin=-0.99, velmax=0.99,
                 n_layers=2, n_neurons=128,
                 batch_size=1024, lr=1e-2,
                 models_basepath="../models", 
                 device=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if device.type == 'cuda': 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    dtype = torch.float64
    
    models_path = os.path.join(models_basepath, str(seed))
    
    os.makedirs(models_path, exist_ok=True)
    
    # Scaling for net input (always the same)
    convfac = np.log10(np.exp(1)) # Conversion from log to log10

    input_max = torch.tensor((lrhomax/convfac,lpressmax/convfac,velmax), device=device, dtype=dtype)
    input_min = torch.tensor((lrhomin/convfac,lpressmin/convfac,velmin), device=device, dtype=dtype)
    
    torch.save(input_max, os.path.join(models_path, "input_max.pt"))
    torch.save(input_min, os.path.join(models_path, "input_min.pt"))
    
    config = {'n_layers': n_layers, 'n_neurons': n_neurons}
    with open(os.path.join(models_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
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
        }
    }

    for wave_type, config in wave_configs.items():
        print(f"\nTraining model for wave pattern: {wave_type}")
        train_model(
            seed,
            n_layers, n_neurons,
            input_min, input_max,
            config["pjumpmin"],
            config["pjumpmax"],
            gamma,
            N,
            wave_type,
            config["loss_fn"],
            lr=lr,
            clip_grads=True,
            grad_norm=1.0,
            batch_size=batch_size,
            device=device,
            model_basepath=models_path,

        )
        
    # Finally we train the rarefaction solvers 
    train_raref_solvers(seed,n_layers,n_neurons//2,N,input_min,input_max,gamma,lr,True,1.0,batch_size,device,models_path)
    
    return 