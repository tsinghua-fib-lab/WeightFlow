import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from tqdm import tqdm
from typing import Dict, Tuple, Any
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from utils import set_seed, wasserstein


class MDN:
    @staticmethod
    def nll_loss(pi_logits, mus, log_sigmas, target) -> torch.Tensor:
            """Calculates the Negative Log-Likelihood for a batch under a GMM."""
            sigmas = torch.exp(log_sigmas)
            
            target_expanded = target.unsqueeze(1)

            # log PDF of a 1D Gaussian: -log(sigma) - 0.5*log(2pi) - 0.5*((x-mu)/sigma)^2
            log_probs_unnormalized = -log_sigmas - 0.5 * math.log(2 * math.pi) - 0.5 * (((target_expanded - mus) / sigmas) ** 2)
            log_probs_unnormalized = log_probs_unnormalized.squeeze(2)

            log_pi = F.log_softmax(pi_logits, dim=1)
            log_likelihood = torch.logsumexp(log_pi + log_probs_unnormalized, dim=1)
            
            return -torch.mean(log_likelihood)

    @staticmethod
    def sample(pi_logits, mus, log_sigmas) -> torch.Tensor:
        """Samples from the GMM for each item in the batch."""
        sigmas = torch.exp(log_sigmas)

        pis = F.softmax(pi_logits, dim=1)
        # component_indices has shape (B, 1)
        component_indices = torch.multinomial(pis, 1)

        # Expand index to 3 dimensions (B, 1, 1) to match the input tensors (mus, sigmas)
        expanded_indices = component_indices.unsqueeze(-1)

        # mu_selected and sigma_selected will have the shape of the index: (B, 1, 1)
        mu_selected = torch.gather(mus, 1, expanded_indices)
        sigma_selected = torch.gather(sigmas, 1, expanded_indices)
        
        # torch.normal returns shape (B, 1, 1). .squeeze() removes all dims of size 1 -> (B,)
        return torch.normal(mu_selected, sigma_selected).squeeze()


class Theta(nn.Module):
    """
    Represents the probability distribution p(x|θ) at a single time slice.
    
    This module is an autoregressive GRU that models the joint probability of species counts.
    Its parameters, collectively denoted as θ, are the state variables that will be evolved by the Neural ODE φ.
    """
    def __init__(self, hidden_dim: int = 8,num_heads: int = None, num_layers: int = 1, mode: str = 'discrete', max_species_count: int = 10, num_mixtures: int = 5):
        super(Theta, self).__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures

        if self.mode == 'discrete':
            self.max_species_count = max_species_count # Each species state is a max_species_count-dim vector
            self.input_dim = (int(max_species_count) - 1).bit_length()
        elif self.mode == 'continuous':
            self.input_dim = 1 # Each species state is a 1D scalar

        self.gru_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = self.input_dim if i == 0 else hidden_dim
            layer = nn.ModuleDict({
                # Input to Hidden linear layers
                'ir_i': nn.Linear(input_dim, hidden_dim),
                'iz_i': nn.Linear(input_dim, hidden_dim),
                'in_i': nn.Linear(input_dim, hidden_dim),
                # Hidden to Hidden linear layers
                'hr_h': nn.Linear(hidden_dim, hidden_dim),
                'hz_h': nn.Linear(hidden_dim, hidden_dim),
                'hn_h': nn.Linear(hidden_dim, hidden_dim),
            })
            self.gru_layers.append(layer)
        
        self.output_layer = nn.Linear(hidden_dim, max_species_count if self.mode == 'discrete' else 3 * num_mixtures)
            
    def _gru_cell(self, x: torch.Tensor, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        layer = self.gru_layers[layer_idx]
        
        r_t = torch.sigmoid(layer['ir_i'](x) + layer['hr_h'](h))
        z_t = torch.sigmoid(layer['iz_i'](x) + layer['hz_h'](h))
        n_t = torch.tanh(layer['in_i'](x) + r_t * layer['hn_h'](h))
        
        h_next = (1 - z_t) * n_t + z_t * h
        return h_next

    def reset_parameters(self) -> None:
        """Initializes parameters with a standard Kaiming-like uniform distribution."""
        set_seed(42)
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for i in range(self.num_layers):
            layer = self.gru_layers[i]
            for name, param in layer.items():
                if isinstance(param, nn.Linear):
                    nn.init.uniform_(param.weight, -stdv, stdv)
                    nn.init.zeros_(param.bias)
        
        nn.init.uniform_(self.output_layer.weight, -stdv, stdv)
        nn.init.zeros_(self.output_layer.bias)
        
    # --- Forward Pass ---
    def _autoregressive_step(self, x_binary: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor]:
        """A single forward step through all GRU layers for one species."""
        layer_input = x_binary
        new_hidden_list = []
        for i in range(self.num_layers):
            h_prev = hidden[i]
            h_next = self._gru_cell(layer_input, h_prev, i)
            layer_input = h_next
            new_hidden_list.append(h_next)
        
        last_hidden_state = new_hidden_list[-1]
        params = self.output_layer(last_hidden_state)
        new_hidden = torch.stack(new_hidden_list)

        if self.mode == 'discrete':
            log_probs = F.log_softmax(params, dim=1)
            return log_probs, new_hidden
        else: # continuous
            pi_logits, mus, log_sigmas = torch.chunk(params, 3, dim=1)
            return (pi_logits, mus.unsqueeze(2), log_sigmas.unsqueeze(2)), new_hidden

    def forward(self, x_batch: torch.Tensor, p_teacher_force: float = 1.0) -> torch.Tensor:
        if self.mode == 'discrete':
            return self._discrete_forward(x_batch, p_teacher_force)
        else:
            return self._continuous_forward(x_batch, p_teacher_force)
        
    def _discrete_forward(self, x_batch: torch.Tensor, p_teacher_force: float) -> torch.Tensor:
        batch_size, num_species = x_batch.shape
        device = x_batch.device
        criterion = nn.NLLLoss()
        
        total_loss = 0.0
        hidden = self.init_hidden(batch_size, device)
        input_count = torch.zeros(batch_size, dtype=torch.long, device=device)

        for j in range(num_species):
            input_binary = self._to_binary(input_count, device)
            log_probs, hidden = self._autoregressive_step(input_binary, hidden)
            
            target = x_batch[:, j]
            total_loss += criterion(log_probs, target)
            
            use_teacher_force = torch.rand(1).item() < p_teacher_force
            if use_teacher_force:
                input_count = target
            else:
                probs = torch.exp(log_probs)
                input_count = torch.multinomial(probs, num_samples=1).squeeze(1)
            
        return total_loss / num_species
    
    def _continuous_forward(self, x_batch: torch.Tensor, p_teacher_force: float) -> torch.Tensor:
        batch_size, num_species = x_batch.shape
        device = x_batch.device
        
        total_loss = 0.0
        hidden = self.init_hidden(batch_size, device)
        input_scalar = torch.zeros(batch_size, 1, device=device)

        for j in range(num_species):
            params, hidden = self._autoregressive_step(input_scalar, hidden)
            pi_logits, mus, log_sigmas = params
            
            target = x_batch[:, j].unsqueeze(1)
            total_loss += MDN.nll_loss(pi_logits, mus, log_sigmas, target)
            
            use_teacher_force = torch.rand(1).item() < p_teacher_force
            if use_teacher_force:
                input_scalar = target
            else:
                input_scalar = MDN.sample(pi_logits, mus, log_sigmas).unsqueeze(1)

        return total_loss / num_species
    
    # --- Parameter Management ---
    def get_flat_theta(self) -> torch.Tensor:
        """Flattens all model parameters into a single 1D tensor."""
        return torch.cat([p.flatten() for p in self.parameters()])
    
    @torch.no_grad()
    def get_weight_graph(self) -> Dict[str, Any]:
        tokens_by_layer: Dict[str, torch.Tensor] = {}
        tokens_info_by_layer: Dict[str, Dict[str, int]] = {}
        max_feature_len = 0

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                weights, biases = module.weight, module.bias
                tokens_for_layer = torch.cat([weights, biases.unsqueeze(1)], dim=1)
                tokens_by_layer[name] = tokens_for_layer
                
                original_len = tokens_for_layer.shape[1]
                if original_len > max_feature_len:
                    max_feature_len = original_len
                
                tokens_info_by_layer[name] = {
                    "num_tokens": tokens_for_layer.shape[0],
                    "original_feature_len": original_len
                }
        
        padded_tokens_list = []
        for name in tokens_info_by_layer.keys():
            tokens = tokens_by_layer[name]
            padding_needed = max_feature_len - tokens.shape[1]
            padded_tokens = F.pad(tokens, (0, padding_needed))
            padded_tokens_list.append(padded_tokens)

        final_tokens = torch.cat(padded_tokens_list, dim=0)

        return {
            "tokens": final_tokens,
            "metadata": {
                "tokens_info_by_layer": tokens_info_by_layer,
                "max_feature_len": max_feature_len,
                "total_tokens": final_tokens.shape[0]
            }
        }
        
    @torch.no_grad()
    def load_weight_graph(self, graph_data: Dict[str, Any]):
        tokens = graph_data["tokens"]
        metadata = graph_data["metadata"]
        tokens_info = metadata["tokens_info_by_layer"]
        
        current_offset = 0
        for name, info in tokens_info.items():
            num_tokens = info["num_tokens"]
            original_len = info["original_feature_len"]
            
            target_module = self.get_submodule(name)
            
            layer_padded_tokens = tokens[current_offset : current_offset + num_tokens]
            layer_original_tokens = layer_padded_tokens[:, :original_len]
            
            weights_data = layer_original_tokens[:, :-1]
            bias_data = layer_original_tokens[:, -1]
            
            target_module.weight.data.copy_(weights_data)
            target_module.bias.data.copy_(bias_data)
            
            current_offset += num_tokens

    # --- Utility and Helper Functions ---
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initializes the hidden state tensor."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
    
    def _to_binary(self, numbers: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Vectorized conversion of a tensor of numbers to their binary representations."""
        powers_of_2 = 2 ** torch.arange(self.input_dim - 1, -1, -1, device=device)
        binary_matrix = (numbers.unsqueeze(1) & powers_of_2) > 0
        return binary_matrix.float()

    # --- Training and Fitting Methods ---
    def fit(self, data: torch.Tensor, theta_init: Dict[str, torch.Tensor] = None, 
            epochs: int = 100, batch_size: int = 64, lr: float = 0.001, save_opt: bool = False) -> None:
        """
        Fits the model to a given data distribution D_t. This function is used for the 'Teacher Forcing & Fitting' step in the main training algorithm.
        
        Args:
            data: The dataset for the current time slice, shape (num_samples, num_species).
            theta_init: An optional initial state_dict to start from (warm start).
            epochs, batch_size, lr: Training hyperparameters.
        """
        device = next(self.parameters()).device
        if theta_init is not None:
            self.load_weight_graph(theta_init)
        else:
            self.reset_parameters()
        
        if save_opt:
            if getattr(self, 'optimizer', None) is not None:
                optimizer = self.optimizer
            else:
                # optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
                optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        
        pbar = tqdm(range(epochs), desc=f"Fitting p(x|θ)", unit="epoch", leave=False)
        for _ in pbar:
            self.train()
            epoch_loss = 0
            # Shuffle data each epoch
            perm = torch.randperm(data.size(0))
            for i in range(0, len(data), batch_size):
                batch_indices = perm[i : i + batch_size]
                x_batch = data[batch_indices].to(device)
                
                optimizer.zero_grad()
                # p_teacher_force = max(0.0, 1.0 - (epoch / epochs)) # Decaying teacher forcing probability (linear schedule)
                loss = self.forward(x_batch, p_teacher_force=1.0)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(x_batch)

            avg_loss = epoch_loss / len(data)
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})
    
    # --- Application Methods ---
    def sample(self, num_samples: int, num_species: int) -> torch.Tensor:
        """
        Autoregressively generates samples from the learned distribution.

        Args:
            num_samples: The number of sample trajectories to generate (batch_size).
            num_species: The number of species in a trajectory (N).

        Returns:
            A tensor of generated samples of shape (num_samples, num_species).
            Dtype will be torch.long for 'discrete' mode and torch.float for 'continuous' mode.
        """
        self.eval()
        with torch.no_grad():
            if self.mode == 'discrete':
                return self._discrete_sample(num_samples, num_species)
            else: # continuous
                return self._continuous_sample(num_samples, num_species)

    def _discrete_sample(self, num_samples: int, num_species: int) -> torch.Tensor:
        """Generates samples for the discrete mode."""
        device = next(self.parameters()).device
        hidden = self.init_hidden(num_samples, device)
        input_count = torch.zeros(num_samples, dtype=torch.long, device=device)
        
        samples = torch.zeros(num_samples, num_species, dtype=torch.long, device=device)

        for j in range(num_species):
            input_binary = self._to_binary(input_count, device)
            log_probs, hidden = self._autoregressive_step(input_binary, hidden)
            
            probs = torch.exp(log_probs)
            
            next_input_count = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            samples[:, j] = next_input_count
            input_count = next_input_count
            
        return samples

    def _continuous_sample(self, num_samples: int, num_species: int) -> torch.Tensor:
        """Generates samples for the continuous mode."""
        device = next(self.parameters()).device
        hidden = self.init_hidden(num_samples, device)
        input_scalar = torch.zeros(num_samples, 1, device=device)
        
        samples = torch.zeros(num_samples, num_species, dtype=torch.float, device=device)

        for j in range(num_species):
            params, hidden = self._autoregressive_step(input_scalar, hidden)
            pi_logits, mus, log_sigmas = params
            
            next_input_scalar = MDN.sample(pi_logits, mus, log_sigmas)
            
            samples[:, j] = next_input_scalar
            input_scalar = next_input_scalar.unsqueeze(1)
            
        return samples

    def sample_and_plot(self, num_samples: int, num_species: int, gt: torch.Tensor, save_path=None):
        if self.mode == 'continuous':
            samples = self.sample(num_samples, num_species)
            w1, w2 = wasserstein(samples, gt)
            
            pca = PCA(n_components=2)
            gt_2d = pca.fit_transform(gt.cpu().numpy())
            samples_2d = pca.transform(samples.cpu().numpy())
            
            plt.figure(figsize=(6, 6))
            plt.scatter(gt_2d[:, 0], gt_2d[:, 1], color='blue', label='Ground Truth', alpha=0.5)
            plt.scatter(samples_2d[:, 0], samples_2d[:, 1], color='red', label='Generated Samples', alpha=0.5)
            plt.title(f'Samples vs Ground Truth | W1: {w1:.4f}, W2: {w2:.4f}')
            plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
            plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
            plt.legend()
            plt.savefig(f'{save_path}.png')
        
