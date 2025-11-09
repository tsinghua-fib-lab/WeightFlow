import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import torchcde
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap

from model.backbone import Theta
from utils import local_seed


def MLP(input_dim, output_dim, hidden_dim=128, num_layers=2, norm=False, end_nonlinear=False):
    layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim) if norm else nn.Identity(), nn.Tanh()]
    for _ in range(num_layers-2): 
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim) if norm else nn.Identity(), nn.Tanh()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    if end_nonlinear:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)

class Z_Encode(nn.Module):
    def __init__(self, input_dim: int, z_dim: int = 2, hidden_dim: int = 128, method: str = 'PCA'):
        super().__init__()
        self.method = method
        if method == 'AE':
            self.encoder = MLP(input_dim, z_dim, hidden_dim, num_layers=2, end_nonlinear=True)
            self.decoder = MLP(z_dim, input_dim, hidden_dim, num_layers=2)
        elif method == 'PCA':
            self.pca = PCA(n_components=z_dim)
        elif method == 'MDS':
            self.mds = MDS(n_components=z_dim)
        elif method == 'Isomap':
            self.isomap = Isomap(n_components=z_dim)

    def fit_transform(self, x: torch.Tensor, epochs: int = 50, lr: float = 0.01, opt='Adam') -> torch.Tensor:
        if self.method == 'PCA':
            x_flat = x.view(x.shape[0], -1).cpu().detach().numpy()
            self.pca.fit(x_flat)
            z = torch.tensor(self.pca.transform(x_flat), device=x.device, dtype=torch.float32)
            print(f"Encode Z by PCA | Explained variance ratio: {self.pca.explained_variance_ratio_.sum()*100:.2f}%")
            z = (z - z.mean(dim=0, keepdim=True)) / z.std(dim=0, keepdim=True)  # Standardize Z
            return z
        elif self.method == 'AE':
            if opt == 'Adam':
                optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            elif opt == 'SGD':
                optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
            x_flat = x.view(x.shape[0], -1).detach()
            for _ in range(epochs):
                optimizer.zero_grad()
                z = self.encoder(x_flat)
                x_recon = self.decoder(z)
                loss = F.mse_loss(x_recon, x_flat)
                loss.backward()
                optimizer.step()
                print(f"\rFitting Path | Epoch {_+1}/{epochs} | MSE Loss: {loss.item():.4f}", end='')
            print()
            with torch.no_grad():
                return self.encoder(x_flat)
        elif self.method == 'MDS':
            x_flat = x.view(x.shape[0], -1).cpu().detach().numpy()
            z = torch.tensor(self.mds.fit_transform(x_flat), device=x.device, dtype=torch.float32)
            print(f"Encode Z by MDS | Number of components: {self.mds.n_components}")
            z = (z - z.mean(dim=0, keepdim=True)) / z.std(dim=0, keepdim=True)  # Standardize Z
            return z
        elif self.method == 'Isomap':
            x_flat = x.view(x.shape[0], -1).cpu().detach().numpy()
            z = torch.tensor(self.isomap.fit_transform(x_flat), device=x.device, dtype=torch.float32)
            print(f"Encode Z by Isomap | Number of components: {self.isomap.n_components}")
            z = (z - z.mean(dim=0, keepdim=True)) / z.std(dim=0, keepdim=True)  # Standardize Z
            return z

class NeuralCdeFunction(nn.Module):
    def __init__(self, initial_theta_model: Theta, snapshots, z_dim: int = 2, z_method: str = 'PCA', hidden_dim: int = 64, num_attn_layers: int = 2, num_heads: int = 4, dropout_p: float = 0.1, tmp_dir=''):
        super().__init__()
        
        initial_graph = initial_theta_model.get_weight_graph()
        metadata = initial_graph["metadata"]
        
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.tokens_info = metadata["tokens_info_by_layer"]
        self.max_feature_len = metadata["max_feature_len"]
        self.layer_names = list(self.tokens_info.keys())

        with local_seed(3):
            self.path = self._create_path(snapshots, z_dim, z_method, tmp_dir)

        self.input_mlps = nn.ModuleDict()
        self.output_mlps = nn.ModuleDict()

        for name, info in self.tokens_info.items():
            original_len = info["original_feature_len"]
            clean_name = name.replace('.', '_')
            
            self.input_mlps[clean_name] = MLP(original_len, hidden_dim, num_layers=1)
            self.output_mlps[clean_name] = MLP(hidden_dim, original_len * self.z_dim, num_layers=1)
            
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, batch_first=True, dropout=dropout_p)
        self.attention_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_attn_layers)

    def _create_path(self, snapshots, z_dim: int, z_method: str, tmp_dir: str) -> torchcde.CubicSpline:
        if isinstance(snapshots, torchcde.CubicSpline):
            z_path = snapshots
            return z_path

        num_snapshots = snapshots.shape[0]
        snapshots_flat = snapshots.view(num_snapshots, -1).float()
        
        ae = Z_Encode(input_dim=snapshots_flat.shape[1], z_dim=z_dim, method=z_method).to(snapshots.device)
        if 'veres' in tmp_dir or 'embryoid' in tmp_dir:
            z_seen = ae.fit_transform(snapshots_flat, epochs=10000, lr=0.1, opt='SGD')
        else:
            z_seen = ae.fit_transform(snapshots_flat, epochs=10000, lr=0.001, opt='Adam')
        
        t_points = torch.linspace(0, 1, num_snapshots, device=z_seen.device)
        coeffs = torchcde.natural_cubic_spline_coeffs(z_seen, t=t_points)
        path = torchcde.CubicSpline(coeffs, t=t_points)
        if z_dim == 1:
            import os
            import matplotlib.pyplot as plt
            eval_t = torch.linspace(t_points.min(), t_points.max(), 1000)
            x_eval = path.evaluate(eval_t)
            z_seen = z_seen.detach().cpu().numpy()
            x_eval_np = x_eval.detach().cpu().numpy()
            plt.figure(figsize=(10, 8))
            plt.scatter(t_points, z_seen, c=t_points.numpy(), cmap='viridis', s=80, zorder=3, label='X Seen Points')
            plt.plot(eval_t, x_eval_np, color='orangered', linewidth=2, zorder=2, label='Cubic Spline Path')
            plt.xlabel('Time', fontsize=14)
            plt.ylabel('Z', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.colorbar(label='t')
            plt.tight_layout()
            os.makedirs(tmp_dir, exist_ok=True)
            plt.savefig(tmp_dir+'/cubic_spline_path.png', dpi=300)
        
        return path

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # y shape: (batch, total_tokens, max_feature_len)
        y, _ = state
        batch_size = y.shape[0]
        
        aligned_tokens_list: List[torch.Tensor] = []
        current_offset = 0
        for name in self.layer_names:
            info, num_tokens, original_len = self.tokens_info[name], self.tokens_info[name]["num_tokens"], self.tokens_info[name]["original_feature_len"]
            layer_original_tokens = y[:, current_offset : current_offset + num_tokens, :original_len]
            aligned_tokens = self.input_mlps[name.replace('.', '_')](layer_original_tokens)
            aligned_tokens_list.append(aligned_tokens)
            current_offset += num_tokens
        
        all_tokens_seq = torch.cat(aligned_tokens_list, dim=1)
        processed_tokens_seq = self.attention_layers(all_tokens_seq)
        
        dZdt = self.path.derivative(t) # Shape: (z_dim)
        num_tokens_split = [info["num_tokens"] for info in self.tokens_info.values()]
        split_processed_tokens = torch.split(processed_tokens_seq, num_tokens_split, dim=1)
        
        output_dXdt_list: List[torch.Tensor] = []
        d_energy_dt = torch.zeros(batch_size, device=y.device)
        for i, name in enumerate(self.layer_names):
            info = self.tokens_info[name]
            original_len, num_tokens = info["original_feature_len"], info["num_tokens"]
            
            dXdZ_flat = self.output_mlps[name.replace('.', '_')](split_processed_tokens[i])
            dXdZ_part = dXdZ_flat.view(batch_size, num_tokens, original_len, self.z_dim)
            dXdt_part = dXdZ_part @ dZdt # Shape: (batch, num_tokens, original_len)
            
            layer_kinetic_energy = torch.sum(dXdt_part**2, dim=[1, 2])
            d_energy_dt += layer_kinetic_energy
            
            padding_needed = self.max_feature_len - original_len
            dX_dt_padded = F.pad(dXdt_part, (0, padding_needed))
            output_dXdt_list.append(dX_dt_padded)
        
        dXdt = torch.cat(output_dXdt_list, dim=1) # Shape: (batch, total_tokens, max_feature_len)
        d_energy_dt *= 0.5
        return dXdt, d_energy_dt
