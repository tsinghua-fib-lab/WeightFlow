import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from model.backbone import Theta


def MLP(input_dim, output_dim, hidden_dim=128, num_layers=2, norm=False):
    layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim) if norm else nn.Identity(), nn.Tanh()]
    for _ in range(num_layers-2): 
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim) if norm else nn.Identity(), nn.Tanh()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)

class NeuralOdeFunction(nn.Module):
    def __init__(self, initial_theta_model: Theta, hidden_dim: int = 64, num_attn_layers: int = 2, num_heads: int = 4, dropout_p: float = 0.1):
        super().__init__()
        
        initial_graph = initial_theta_model.get_weight_graph()
        metadata = initial_graph["metadata"]
        
        self.hidden_dim = hidden_dim
        self.tokens_info = metadata["tokens_info_by_layer"]
        self.max_feature_len = metadata["max_feature_len"]
        self.layer_names = list(self.tokens_info.keys())

        self.input_mlps = nn.ModuleDict()
        self.output_mlps = nn.ModuleDict()

        for name, info in self.tokens_info.items():
            original_len = info["original_feature_len"]
            
            clean_name = name.replace('.', '_')
            
            self.input_mlps[clean_name] = MLP(original_len, hidden_dim, num_layers=1)
            self.output_mlps[clean_name] = MLP(hidden_dim, original_len, num_layers=1)
            
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, batch_first=True, dropout=dropout_p)
        self.attention_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_attn_layers)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        y, _ = state
        batch_size = y.shape[0]
        
        aligned_tokens_list: List[torch.Tensor] = []
        current_offset = 0
        for name in self.layer_names:
            info = self.tokens_info[name]
            num_tokens, original_len = info["num_tokens"], info["original_feature_len"]
            
            layer_padded_tokens = y[:, current_offset : current_offset + num_tokens]
            layer_original_tokens = layer_padded_tokens[:, :, :original_len]
            
            clean_name = name.replace('.', '_')
            aligned_tokens = self.input_mlps[clean_name](layer_original_tokens)
            aligned_tokens_list.append(aligned_tokens)
            
            current_offset += num_tokens
            
        all_tokens_seq = torch.cat(aligned_tokens_list, dim=1)
        processed_tokens_seq = self.attention_layers(all_tokens_seq)
        
        num_tokens_split = [info["num_tokens"] for info in self.tokens_info.values()]
        split_processed_tokens = torch.split(processed_tokens_seq, num_tokens_split, dim=1)
        
        output_padded_derivatives_list: List[torch.Tensor] = []
        d_energy_dt = torch.zeros(batch_size, device=y.device)
        for i, name in enumerate(self.layer_names):
            original_len = self.tokens_info[name]["original_feature_len"]
            
            clean_name = name.replace('.', '_')
            derivatives_original = self.output_mlps[clean_name](split_processed_tokens[i])
            
            layer_kinetic_energy = torch.sum(derivatives_original**2, dim=[1, 2])
            d_energy_dt += layer_kinetic_energy
            
            padding_needed = self.max_feature_len - original_len
            derivatives_padded = F.pad(derivatives_original, (0, padding_needed))
            output_padded_derivatives_list.append(derivatives_padded)
            
        dXdt = torch.cat(output_padded_derivatives_list, dim=1)
        d_energy_dt *= 0.5
        return dXdt, d_energy_dt
    
