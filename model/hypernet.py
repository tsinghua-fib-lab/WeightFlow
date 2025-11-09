import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchdiffeq import odeint

from model.backbone import Theta
from model.NeuralODE import NeuralOdeFunction
from model.NeuralCDE import NeuralCdeFunction


class my_trajectory:
    """
    A simple wrapper for trajectory data.
    """
    def __init__(self, trajectories, device):
        self.device = device
        self.trajectories = trajectories # Tensor: (num_traj, num_timesteps, num_nodes) or DataFrame [t*num_traj^, species1, species2, ...]
        self.time_length = self.trajectories.shape[1] if isinstance(trajectories, torch.Tensor) else len(trajectories.samples.unique())
        self.num_species = self.trajectories.shape[2] if isinstance(trajectories, torch.Tensor) else len(trajectories.columns) - 1

    def __getitem__(self, idx):
        if isinstance(self.trajectories, torch.Tensor):
            return self.trajectories[:, idx] # num_traj, num_nodes
        elif isinstance(self.trajectories, pd.DataFrame):
            data = np.array(self.trajectories[self.trajectories['samples'] == idx])
            data = torch.from_numpy(data[:, 1:]).float().to(self.device)
            return data  # num_traj, num_nodes

    def __len__(self):
        return self.trajectories.shape[0]


class Phi(nn.Module):
    """
    The Phi module encapsulates the Neural ODE. 
    It learns the dynamics 'f' and orchestrates the entire training process.
    """
    def __init__(self, theta_model: Theta, dynamics_type: str, theta_snapshots: torch.Tensor = None, z_dim: int = 8, z_method: str = 'PCA', energy_lambda: float = 1e-3, hidden_dim: int = 64, layer_num: int = 4, num_heads: int = 4, dropout_p: float = 0.1, tmp_dir=''):
        super(Phi, self).__init__()        
        self.theta = theta_model
        self.dynamics_type = dynamics_type
        self.energy_lambda = energy_lambda
        if dynamics_type == 'NeuralODE':
            self.dynamics_net = NeuralOdeFunction(theta_model, hidden_dim=hidden_dim, num_attn_layers=layer_num, num_heads=num_heads, dropout_p=dropout_p)
        elif dynamics_type == 'NeuralCDE':
            if theta_snapshots is not None:
                self.dynamics_net = NeuralCdeFunction(theta_model, theta_snapshots, z_dim, z_method, hidden_dim=hidden_dim, num_attn_layers=layer_num, num_heads=num_heads, dropout_p=dropout_p, tmp_dir=tmp_dir)
            else:
                self.z_dim = z_dim
                self.z_method = z_method
                self.hidden_dim = hidden_dim
                self.layer_num = layer_num
                self.num_heads = num_heads
                self.dropout_p = dropout_p
                self.tmp_dir = tmp_dir
                self.dynamics_net = None # wait for pre-computation of theta_snapshots

    def forward(self, y0_batch: torch.Tensor, t_span: torch.Tensor, method: str = 'rk4', adjoint_energy: bool = False) -> torch.Tensor:
        """Evolves the initial state theta_start_flat over the given time span."""
        
        initial_weights = y0_batch
        initial_energy = torch.zeros(initial_weights.shape[0], device=initial_weights.device)
        state_0 = (initial_weights, initial_energy)
        
        solution = odeint(
            self.dynamics_net,
            state_0,
            t_span,
            method=method,
        ) # Shape: (num_timesteps, batch_size, N, dynamic_dim)
        
        weight_solution = solution[0]  # (num_timesteps, batch_size, N, dynamic_dim)
        energy_solution = solution[1]  # (num_timesteps, batch_size)
        
        # Return the solution at all time points except the initial one
        if adjoint_energy:
            return weight_solution[1:], energy_solution[1:]
        else:
            return weight_solution[1:]


    def fit(self, trajectories: torch.Tensor,
            seen_time_points: list,
            total_epochs: int,
            log_dir: str = 'log',
            feature_dir: str = 'log/theta_path',
            batch_size: int = 32,
            lr: float = 1e-4,
            dynamics_align: int = True,
            horizon_min: int = 1,
            horizon_max: int = 20,
            steps_per_epoch: int = 100,
            fit_steps: int = 50,
            verbose: bool = True
        ):
        assert len(seen_time_points) > horizon_max, "The last seen time point must be greater than the maximum horizon."
        device = next(self.parameters()).device
        self.to(device)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(feature_dir, exist_ok=True)
        dataset = my_trajectory(trajectories, device=device)

        # --- Pre-computation: Cache DYNAMIC FEATURES at all time points ---
        print("Pre-computing and caching DYNAMIC FEATURES for all time points...")
        dynamic_feature_memory, weight_graph = {}, None
        for idx, abs_t in tqdm(enumerate(seen_time_points), desc="Pre-computing Features", total=len(seen_time_points)):
            fit_steps_ = fit_steps if idx > 0 else fit_steps * 2
            try:
                dynamic_feature_memory[idx] = torch.load(f'{feature_dir}/feat_t{abs_t}.pt', map_location=device)
                weight_graph = self.theta.get_weight_graph()
                weight_graph['tokens'] = dynamic_feature_memory[idx]
            except:
                data_t = dataset[abs_t] # (num_nodes, max_length)
                theta_init = weight_graph if dynamics_align else None
                self.theta.fit(data_t, theta_init=theta_init, epochs=fit_steps_, lr=1e-2, save_opt=True)
                self.theta.sample_and_plot(
                    num_samples=data_t.shape[0],
                    num_species=dataset.num_species,
                    gt=data_t,
                    save_path=f'{feature_dir}/feat_t{abs_t}.png'
                )
                weight_graph = self.theta.get_weight_graph()
                dynamic_feature_memory[idx] = weight_graph['tokens'].detach().to(device)
                torch.save(dynamic_feature_memory[idx], f'{feature_dir}/feat_t{abs_t}.pt')
        print("...Caching complete. Starting main training loop.")
        
        # --- Initialize the dynamic feature memory ---
        if self.dynamics_type == 'NeuralCDE' and self.dynamics_net is None:
            theta_flat_len = self.theta.get_flat_theta().shape[0]
            theta_snapshots = torch.zeros((len(seen_time_points), theta_flat_len))
            for i, t in enumerate(seen_time_points):
                weight_path = f'{feature_dir}/feat_t{t}.pt'
                weight_graph = self.theta.get_weight_graph()
                weight_graph['tokens'] = torch.load(weight_path, map_location=device)
                self.theta.load_weight_graph(weight_graph)
                theta_snapshots[i] = self.theta.get_flat_theta()
            self.dynamics_net = NeuralCdeFunction(self.theta, theta_snapshots, self.z_dim, self.z_method, hidden_dim=self.hidden_dim, num_attn_layers=self.layer_num, num_heads=self.num_heads, dropout_p=self.dropout_p, tmp_dir=self.tmp_dir)
            self.dynamics_net.to(device)
        

        # --- Training Loop ---
        optimizer = optim.AdamW(self.dynamics_net.parameters(), lr=lr, weight_decay=1e-4)
        mse = nn.MSELoss()
        epoch_pbar = tqdm(range(total_epochs), desc="Total Training Progress")
        loss_list, best_loss = [], float('inf')
        self.train()
        for epoch in epoch_pbar:
            segment_pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
            ep_weight_l = 0.0
            
            for step in segment_pbar:
                # 1. Sample a batch of starting points and a shared horizon
                horizon = np.random.randint(horizon_min, horizon_max + 1)
                safe_upper_bound = len(seen_time_points) - horizon
                t_start_indices = np.random.randint(0, safe_upper_bound, size=batch_size)
                
                # 2. Prepare batches of initial states (y0) and ground truth sequences
                t_eval_indices = np.arange(horizon + 1)
                rel_t_trajs = t_start_indices[:, None] + t_eval_indices[1:]
                
                y0_batch = torch.stack([dynamic_feature_memory[i] for i in t_start_indices]) # (batch_size, num_nodes, max_length)
                y_true_batch = torch.stack([
                    torch.stack([dynamic_feature_memory[rel_t] for rel_t in rel_t_traj])
                    for rel_t_traj in rel_t_trajs
                ]).permute(1, 0, 2, 3) # (horizon, batch_size, num_nodes, max_length)
                
                # 3. Perform a batched long-range evolution
                tspan = [seen_time_points[i] for i in t_eval_indices]
                t_span = torch.tensor(tspan, device=device) / dataset.time_length # Normalize time span to [0, 1]

                weight_sequence, energy_sequence = self.forward(y0_batch, t_span, adjoint_energy=True) # (horizon, batch_size, num_nodes, max_length)

                # 4. Calculate loss
                weight_loss = mse(weight_sequence, y_true_batch)
                energy_loss = energy_sequence.mean()
                loss = weight_loss + self.energy_lambda * energy_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                segment_pbar.set_postfix({
                    'Horizon': f'{horizon}',
                    'Weight_L': f'{weight_loss.item():.4f}',
                    'Energy_L': f'{energy_loss.item():.4f}',
                    'Total_L': f'{loss.item():.4f}',
                })
                ep_weight_l += weight_loss.item()
                
                if step % 10 == 0 and epoch % 10 == 0 and verbose:
                    t_a = seen_time_points[t_start_indices[0].item()]
                    t_b = seen_time_points[t_start_indices[0].item()] + horizon
                    self._visualize_prediction(
                        t_a=t_a,
                        t_b=t_b,
                        data_ta=dataset[t_a],
                        data_tb=dataset[t_b],
                        theta_ta=dynamic_feature_memory[t_start_indices[0].item()],
                        theta_tb=weight_sequence[-1, 0],
                        log_dir=log_dir+'/figures',
                        k=step,
                        epoch=epoch,
                        weight_loss=weight_loss.item()
                    )

            epoch_pbar.set_postfix({'Avg_Weight_L': f'{ep_weight_l / steps_per_epoch:.4f}'})
            loss_list.append(ep_weight_l / steps_per_epoch)
            
            if best_loss > loss_list[-1]:
                best_loss = loss_list[-1]
                torch.save(self.dynamics_net.state_dict(), f'{log_dir}/phi.pth')
                if hasattr(self.dynamics_net, 'path'):
                    with open(f'{log_dir}/phi_path.pkl', 'wb') as f:
                        pickle.dump(self.dynamics_net.path, f)

        plt.figure(figsize=(10, 5))
        plt.plot(loss_list)
        plt.xlabel("Epoch")
        plt.ylabel("Average Weight Loss")
        plt.title("Training Loss Over Epochs")
        plt.yscale('log')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{log_dir}/loss_plot.png')
        
    
    def _visualize_prediction(self, t_a, t_b, data_ta: torch.Tensor, data_tb: torch.Tensor, theta_ta: torch.Tensor, theta_tb: torch.Tensor, log_dir: str, k: int, epoch: int, weight_loss: float, species_to_plot: int = 5, num_viz_samples: int = 1000):
        """Visualizes the predicted vs. actual distributions using stochastic sampling."""        
        num_species_total = data_tb.shape[1]
        species_to_plot = min(species_to_plot, num_species_total)        
        max_count = self.theta.max_species_count.item()
        
        graph_data_template = self.theta.get_weight_graph()
        
        # Get all predicted distributions at time t_a
        graph_data_for_loading = graph_data_template.copy()
        graph_data_for_loading['dynamic_features'] = theta_ta
        self.theta.load_weight_graph(graph_data_for_loading)
        generated_ta_samples = self.theta.sample(num_samples=num_viz_samples, num_species=num_species_total)
        
        # Get all predicted distributions at time t_b
        graph_data_for_loading = graph_data_template.copy()
        graph_data_for_loading['dynamic_features'] = theta_tb
        self.theta.load_weight_graph(graph_data_for_loading)
        generated_tb_samples = self.theta.sample(num_samples=num_viz_samples, num_species=num_species_total)

        fig, axes = plt.subplots(species_to_plot, 2, figsize=(12, 2 * species_to_plot), sharex=True)
        if species_to_plot == 1: axes = [axes]
        
        for i in range(species_to_plot):
            ax1 = axes[i, 0]
            ax2 = axes[i, 1]

            pred_ta_counts = generated_ta_samples[:, i].cpu().numpy()
            pred_ta_bins = np.bincount(pred_ta_counts, minlength=max_count)
            predicted_ta_freq = pred_ta_bins / len(pred_ta_counts)
            
            actual_ta_counts = data_ta[:, i].cpu().numpy()
            actual_ta_bins = np.bincount(actual_ta_counts, minlength=max_count)
            actual_ta_freq = actual_ta_bins / len(actual_ta_counts) if len(actual_ta_counts) > 0 else np.zeros_like(actual_ta_bins)
            
            pred_tb_counts = generated_tb_samples[:, i].cpu().numpy()
            pred_tb_bins = np.bincount(pred_tb_counts, minlength=max_count)
            predicted_tb_freq = pred_tb_bins / len(pred_tb_counts)
            
            actual_tb_counts = data_tb[:, i].cpu().numpy()
            actual_tb_bins = np.bincount(actual_tb_counts, minlength=max_count)
            actual_tb_freq = actual_tb_bins / len(actual_tb_counts)
            
            freq_max = max(predicted_ta_freq.max(), actual_ta_freq.max(), predicted_tb_freq.max(), actual_tb_freq.max())
            
            ax2.bar(range(max_count), predicted_tb_freq, color='skyblue', alpha=0.7, label='Predicted Freq. (at $t_b$)')
            ax2.plot(range(max_count), actual_tb_freq, 'o-', color='coral', label=f'Actual Data Freq. (at $t_b$)')
            ax2.set_title(f"Marginal Distribution @ $t_b={t_b}$", fontsize=14)
            ax2.set_ylabel("Frequency", fontsize=12)
            ax2.legend(frameon=False)
            ax2.set_ylim(0, freq_max + 0.05)
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.set_xlim(left=-1, right=max_count)
            
            ax1.bar(range(max_count), predicted_ta_freq, color='lightgreen', alpha=0.7, label='Predicted Freq. (at $t_a$)')
            ax1.plot(range(max_count), actual_ta_freq, 'o-', color='orange', label=f'Actual Data Freq. (at $t_a$)')
            ax1.set_title(f"Marginal Distribution @ $t_a={t_a}$", fontsize=14)
            ax1.set_ylabel("Frequency", fontsize=12)
            ax1.legend(frameon=False)
            ax1.set_ylim(0, freq_max + 0.05)
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.set_xlim(left=-1, right=max_count)

        axes[-1, 0].set_xlabel("Species Count", fontsize=12)
        axes[-1, 1].set_xlabel("Species Count", fontsize=12)
        fig.suptitle(f"Epoch {epoch+1} | Step {k} | Weight Loss: {weight_loss:.4f}", fontsize=16, y=0.98)
        plt.tight_layout()
        os.makedirs(f"{log_dir}/e{epoch+1}", exist_ok=True)
        plt.savefig(f"{log_dir}/e{epoch+1}/k{k}.png", dpi=300)
        plt.close(fig)