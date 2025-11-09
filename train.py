import yaml
import torch
import argparse
import numpy as np
import warnings; warnings.filterwarnings("ignore")

from model.hypernet import Phi
from model.backbone import Theta
from utils import Config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MFM Lightning")
    parser.add_argument("--device", type=str, default='cuda:0')
    args = parser.parse_args()

    system = 'epidemic'
    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")    
    with open('config/epidemic.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    conf = Config(config_dict)
    
    # --- 1. Prepare Environment & Data ---
    interval = conf.interval
    horizon_max = int(conf.traj_length/interval)
    prefix = f'log'
    log_dir = f'{prefix}/{system}/s{interval}/ours/{conf.Theta_model}/NeuralODE/train' if conf.Phi.dynamics_type == 'NeuralODE' else f'{prefix}/{system}/s{interval}/ours/{conf.Theta_model}/NeuralCDE_{conf.Phi.z_method}/train'
    feature_dir = f'{prefix}/{system}/s{interval}/ours/{conf.Theta_model}/theta_path'
    traj_path = f'data/{system}/trajectories.npy'
    trajectories = np.load(traj_path).astype(np.int64)
    max_species_count = np.max(trajectories) + 1
    seen_time_points = range(0, trajectories.shape[1], interval)
    print(f"System: {system} | Interval: {interval} | Horizon: {horizon_max} | Max Species Count: {max_species_count} | Trajectories Shape: {trajectories.shape} | Z_Method: {conf.Phi.z_method} | Z_Dim: {conf.Phi.z_dim}")
    trajectories = torch.from_numpy(trajectories).to(DEVICE)

    # --- 2. Instantiate Models ---
    theta_model = Theta(
        hidden_dim=conf.Theta.hidden_dim, 
        num_heads=conf.Theta.num_heads if conf.Theta_model == 'Transformer' else None,
        num_layers=conf.Theta.num_layers, 
        mode=conf.Theta.mode, 
        max_species_count=max_species_count
    ).to(DEVICE)
    
    phi_model = Phi(
        theta_model=theta_model, 
        dynamics_type=conf.Phi.dynamics_type, 
        theta_snapshots=None, 
        z_dim=conf.Phi.z_dim, 
        z_method=conf.Phi.z_method, 
        energy_lambda=conf.Phi.energy_lambda,
        hidden_dim=conf.Phi.hidden_dim, 
        layer_num=conf.Phi.layer_num, 
        num_heads=conf.Phi.num_heads, 
        dropout_p=conf.Phi.dropout_p,
        tmp_dir=log_dir,
    ).to(DEVICE)
    print(f"State Num: {max_species_count}| Theta Size ({conf.Theta_model}): {sum(p.numel() for p in theta_model.parameters() if p.requires_grad)}")
    
    # --- 3. Train the Model ---
    phi_model.fit(
        trajectories=trajectories,
        seen_time_points=seen_time_points,
        batch_size=conf.train.batch_size,
        lr=conf.train.lr,
        horizon_min=1,
        horizon_max=horizon_max,
        total_epochs=conf.train.max_epoch,
        log_dir=log_dir,
        feature_dir=feature_dir,
        fit_steps=conf.pretrain.fit_steps,
        verbose=False
    )