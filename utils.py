import ot
import torch
import numpy as np
from tqdm import tqdm
import random
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import contextlib
from scipy.stats import entropy
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@contextlib.contextmanager
def local_seed(seed: int):
    """
    A context manager to temporarily set a random seed for random, numpy, and torch.
    After the block is exited, the random states are restored to what they were before.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    
    if torch.cuda.is_available():
        torch_cuda_states = torch.cuda.get_rng_state_all()

    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark

    try:
        set_seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(torch_cuda_states)
            
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    
        
def create_gif_from_jpgs(input_dir, output_gif_path, duration=200, loop=0):
    """
    Reads all JPG files from a directory, sorts them, and creates a GIF.

    Args:
        input_dir (str): The directory containing the .jpg files.
        output_gif_path (str): The path to save the output GIF file.
        duration (int): The duration (in milliseconds) for each frame.
        loop (int): The number of loops for the GIF. 0 means infinite loop.
    """
    print(f"Searching for JPG images in: {input_dir}")

    # 查找所有.jpg和.jpeg文件
    image_paths = [f'{input_dir}distribution_t{t}.png' for t in range(1, 100+1)]

    if not image_paths:
        print("No JPG images found in the specified directory.")
        return

    print(f"Found {len(image_paths)} images. Starting GIF creation...")

    # 读取所有图片到PIL Image对象列表
    try:
        images = [Image.open(fp) for fp in image_paths]
    except Exception as e:
        print(f"Error opening images: {e}")
        return
        
    # 获取第一张图片
    first_image = images[0]

    # 将第一张图片作为基础，并追加剩余的图片作为帧
    try:
        first_image.save(
            output_gif_path,
            save_all=True,
            append_images=images[1:],  # 剩余的图片列表
            duration=duration,         # 每帧的持续时间 (ms)
            loop=loop                  # 循环次数，0表示无限循环
        )
        print(f"GIF successfully saved to: {output_gif_path}")
    except Exception as e:
        print(f"Error saving GIF: {e}")
        
        
        
class Config:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)



def my_dataset(system: str, batch_size: int, max_state: int, interval: int, device: str) -> DataLoader:
    file_path = f'data/{system}/trajectories.npy'
        
    trajectories = np.load(file_path).astype(np.int64)

    assert max_state >= trajectories.max() + 1, f"max_state {max_state} should be greater than max state in trajectories {trajectories.max() + 1}"
    T = trajectories.shape[1]
    trajectories = trajectories[:, ::interval]
    one_hot_trajectories = np.eye(max_state)[trajectories]
    
    class StateTransitionDataset(Dataset):
        def __init__(self, one_hot_data):
            N_traj, _, num_nodes, max_state_dim = one_hot_data.shape
            
            times_single_traj = np.arange(0, T, interval) / T  # Normalize time to [0, 1]
            self.t_step = times_single_traj[1] - times_single_traj[0]
            self.t_stamps = np.tile(times_single_traj, (N_traj, 1))
            self.t = torch.from_numpy(self.t_stamps).reshape(-1).to(device)
            
            self.one_hot_data = torch.from_numpy(one_hot_data).float().to(device)

            self.x = self.one_hot_data[:, :-1]
            self.y = self.one_hot_data[:, 1:]

            self.x = self.x.reshape(-1, num_nodes, max_state_dim)
            self.y = self.y.reshape(-1, num_nodes, max_state_dim)

        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx], self.t[idx]

    dataset = StateTransitionDataset(one_hot_trajectories)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader, dataset.t_step



def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.constant_(m.bias, 0)
        


def wasserstein(predicted_samples, actual_samples):
    p_np = predicted_samples.detach().cpu().numpy() if isinstance(predicted_samples, torch.Tensor) else predicted_samples
    q_np = actual_samples.detach().cpu().numpy() if isinstance(actual_samples, torch.Tensor) else actual_samples

    num_samples = p_np.shape[0]

    # Uniform weights for both distributions
    p_weights = ot.unif(num_samples)
    q_weights = ot.unif(num_samples)

    # W1
    cost_matrix_w1 = ot.dist(p_np, q_np, metric='euclidean')
    w1_distance = ot.emd2(p_weights, q_weights, cost_matrix_w1)

    # W2
    cost_matrix_w2 = ot.dist(p_np, q_np, metric='sqeuclidean')
    wasserstein_squared_w2 = ot.emd2(p_weights, q_weights, cost_matrix_w2)
    w2_distance = np.sqrt(wasserstein_squared_w2)
    
    return w1_distance, w2_distance


def jensen_shannon(
    predicted_dists: Union[torch.Tensor, np.ndarray],
    actual_dists: Union[torch.Tensor, np.ndarray]
) -> float:
    p = predicted_dists.detach().cpu().numpy() if isinstance(predicted_dists, torch.Tensor) else predicted_dists
    q = actual_dists.detach().cpu().numpy() if isinstance(actual_dists, torch.Tensor) else actual_dists
    
    m = 0.5 * (p + q)
    
    epsilon = 1e-10
    p_smooth = p + epsilon
    q_smooth = q + epsilon
    m_smooth = m + epsilon

    p_norm = p_smooth / np.sum(p_smooth, axis=1, keepdims=True)
    q_norm = q_smooth / np.sum(q_smooth, axis=1, keepdims=True)
    m_norm = m_smooth / np.sum(m_smooth, axis=1, keepdims=True)
    
    kl_p_m = entropy(pk=p_norm, qk=m_norm, axis=1)
    kl_q_m = entropy(pk=q_norm, qk=m_norm, axis=1)
    
    jsd_per_row = 0.5 * (kl_p_m + kl_q_m)
    
    mean_jsd = np.mean(jsd_per_row)
    
    return float(mean_jsd)


def calculate_mmd(p, q, sigmas=[0.1, 1.0, 10.0]):
    p_dist = torch.cdist(p, p, p=2).pow(2)
    q_dist = torch.cdist(q, q, p=2).pow(2)
    pq_dist = torch.cdist(p, q, p=2).pow(2)
    mmd_sq = torch.tensor(0.0, device=p.device)

    for sigma in sigmas:
        # --- 高斯核计算 ---
        # k(x, y) = exp(-||x-y||^2 / (2 * sigma^2))
        
        # p和p之间的核矩阵
        k_p = torch.exp(-p_dist / (2 * sigma**2))
        # q和q之间的核矩阵
        k_q = torch.exp(-q_dist / (2 * sigma**2))
        # p和q之间的核矩阵
        k_pq = torch.exp(-pq_dist / (2 * sigma**2))

        # --- MMD项计算 ---
        # 计算k(p, p)的均值，排除对角线元素以获得无偏估计
        # E[k(x, x')] for x, x' ~ P
        k_p_mean = (k_p.sum() - k_p.trace()) / (p.shape[0] * (p.shape[0] - 1)) if p.shape[0] > 1 else torch.tensor(0.0)
        
        # 计算k(q, q)的均值，排除对角线元素
        # E[k(y, y')] for y, y' ~ Q
        k_q_mean = (k_q.sum() - k_q.trace()) / (q.shape[0] * (q.shape[0] - 1)) if q.shape[0] > 1 else torch.tensor(0.0)

        # 计算k(p, q)的均值
        # E[k(x, y)] for x ~ P, y ~ Q
        k_pq_mean = k_pq.mean()

        # 累加当前sigma下的MMD平方值
        mmd_sq += k_p_mean + k_q_mean - 2 * k_pq_mean

    return mmd_sq.mean().item()



def estimate_cumulative_path_energy(traj: np.ndarray, delta_t: float = 1.0) -> np.ndarray:
    """
    从离散的轨迹快照估算累积路径能量开销。

    该函数通过计算连续快照之间的最优传输代价来近似路径能量，
    并返回一个随时间单调递增的累积能量曲线。

    核心公式： E_interval ≈ (Wasserstein-2 Distance)^2 / delta_t
    """
    T, N, D = traj.shape
    
    interval_energies = []
    
    print(f"Calculating path energy for {T-1} intervals...")
    for k in tqdm(range(T - 1), desc="Calculating Energy Intervals"):
        S_k = traj[k, :, :]
        S_k_plus_1 = traj[k+1, :, :]
        cost_matrix = ot.dist(S_k, S_k_plus_1, metric='sqeuclidean')
        
        total_squared_dist = ot.emd2(np.ones(N) / N, np.ones(N) / N, cost_matrix, numItermax=1000000)
        
        energy_k = total_squared_dist / delta_t
        interval_energies.append(energy_k)
    
    cumulative_energy = np.zeros((T, 1))
    cumulative_energy[1:, 0] = np.cumsum(interval_energies)
    
    return cumulative_energy



def trace_optimal_transport_paths(traj: np.ndarray, n_paths: int) -> np.ndarray:
    """
    使用最优传输(OT)追踪并计算一部分粒子的完整演化路径。

    Args:
        traj (np.ndarray): 输入的轨迹数据，形状为 (T, N, D)。
        n_paths (int): 要追踪的路径数量。

    Returns:
        np.ndarray: 被追踪粒子的完整路径坐标，形状为 (T, n_paths, D)。
    """
    T, N, D = traj.shape
    if n_paths > N:
        n_paths = N
    
    # 随机选择 n_paths 个初始粒子
    initial_indices = np.random.choice(N, n_paths, replace=False)
    
    # path_indices 用来存储每个时刻被追踪粒子的索引
    path_indices = np.zeros((T, n_paths), dtype=int)
    path_indices[0, :] = initial_indices
    
    # 迭代计算OT并链接路径
    for k in tqdm(range(T - 1), desc="Chaining OT to trace paths"):
        S_k = traj[k, :, :]
        S_k_plus_1 = traj[k+1, :, :]
        
        # 求解 S_k 到 S_k+1 的最优传输问题
        gamma = ot.emd(np.ones(N) / N, np.ones(N) / N, ot.dist(S_k, S_k_plus_1))
        
        # 找到当前追踪的粒子在下一时刻最匹配的粒子索引
        current_indices = path_indices[k, :]
        next_indices = np.argmax(N * gamma[current_indices, :], axis=1)
        path_indices[k+1, :] = next_indices

    # 根据追踪到的索引，构建完整的路径坐标
    full_paths = np.zeros((T, n_paths, D))
    for t in range(T):
        full_paths[t, :, :] = traj[t, path_indices[t, :], :]
        
    return full_paths
