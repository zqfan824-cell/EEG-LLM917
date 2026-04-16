"""
重建损失函数实现
参考NeuroLM的VQ机制，为EEGLLM添加重建约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def l2norm(t):
    """L2标准化"""
    return F.normalize(t, p=2, dim=-1)


def ema_inplace(moving_avg, new, decay):
    """EMA原地更新"""
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


class ReconstructionLosses(nn.Module):
    """重建损失函数集合"""
    
    def __init__(self, use_smooth_l1=False, freq_weight=1.0, raw_weight=1.0):
        super().__init__()
        self.loss_fn = F.smooth_l1_loss if use_smooth_l1 else F.mse_loss
        self.freq_weight = freq_weight
        self.raw_weight = raw_weight
    
    def compute_freq_domain_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算频域目标
        Args:
            x: (batch_size, seq_len, n_channels) - 原始EEG数据
        Returns:
            freq_target: (batch_size, seq_len, n_channels//2) - 频域幅度谱
        """
        # 对每个通道进行FFT
        x_fft = torch.fft.fft(x, dim=1)  # (B, L, C)
        amplitude = torch.abs(x_fft)
        
        # 只取前一半频率（对称性）
        freq_target = amplitude[:, :amplitude.shape[1]//2, :]
        
        # 标准化
        freq_target = self._normalize_tensor(freq_target)
        
        return freq_target
    
    def compute_raw_domain_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算时域目标（标准化后的原始数据）
        Args:
            x: (batch_size, seq_len, n_channels) - 原始EEG数据
        Returns:
            raw_target: (batch_size, seq_len, n_channels) - 标准化的时域数据
        """
        return self._normalize_tensor(x)
    
    def _normalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """标准化张量"""
        mean = torch.mean(x, dim=(1, 2), keepdim=True)
        std = torch.std(x, dim=(1, 2), keepdim=True)
        return (x - mean) / (std + 1e-8)
    
    def compute_reconstruction_loss(self, 
                                  freq_pred: torch.Tensor,
                                  raw_pred: torch.Tensor,
                                  freq_target: torch.Tensor,
                                  raw_target: torch.Tensor,
                                  mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        计算重建损失
        Args:
            freq_pred: 频域预测 (B, L, C//2)
            raw_pred: 时域预测 (B, L, C)
            freq_target: 频域目标 (B, L, C//2)
            raw_target: 时域目标 (B, L, C)
            mask: 掩码 (B, L) - 可选
        Returns:
            total_loss: 总重建损失
            loss_dict: 各项损失的详细信息
        """
        if mask is not None:
            # 应用掩码
            freq_mask = mask.unsqueeze(-1).expand_as(freq_pred)
            raw_mask = mask.unsqueeze(-1).expand_as(raw_pred)
            
            freq_pred_masked = freq_pred * freq_mask
            raw_pred_masked = raw_pred * raw_mask
            freq_target_masked = freq_target * freq_mask
            raw_target_masked = raw_target * raw_mask
        else:
            freq_pred_masked = freq_pred
            raw_pred_masked = raw_pred
            freq_target_masked = freq_target
            raw_target_masked = raw_target
        
        # 计算各项损失
        freq_loss = self.loss_fn(freq_pred_masked, freq_target_masked)
        raw_loss = self.loss_fn(raw_pred_masked, raw_target_masked)
        
        # 加权总损失
        total_loss = self.freq_weight * freq_loss + self.raw_weight * raw_loss
        
        loss_dict = {
            'freq_reconstruction_loss': freq_loss.item(),
            'raw_reconstruction_loss': raw_loss.item(),
            'total_reconstruction_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


class NormEMAVectorQuantizer(nn.Module):
    """
    标准化EMA向量量化器
    参考NeuroLM的实现，适配EEGLLM
    """
    
    def __init__(self, n_embed=8192, embedding_dim=128, beta=1.0, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.decay = decay
        self.eps = eps
        
        # 初始化codebook
        weight = torch.randn(n_embed, embedding_dim)
        weight = l2norm(weight)
        
        self.register_buffer('embedding', weight)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', weight.clone())
        self.register_buffer('initted', torch.tensor(False))
        
    def forward(self, z):
        """
        Args:
            z: (batch_size, seq_len, embedding_dim)
        Returns:
            z_q: 量化后的特征
            loss: 量化损失
            encoding_indices: 编码索引
        """
        # 保存原始输入以保持梯度
        z_input = z

        # L2标准化
        z = l2norm(z)  # (B, L, D)
        z_flattened = z.reshape(-1, self.codebook_dim)  # (B*L, D)

        # 计算距离
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding)

        # 找到最近的codebook向量
        encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(encoding_indices, self.embedding).view(z.shape)

        # 计算量化损失
        loss = self.beta * F.mse_loss(z_q.detach(), z)

        # EMA更新（训练时）
        if self.training:
            encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)

            # 更新cluster size
            bins = encodings.sum(0)
            ema_inplace(self.cluster_size, bins, self.decay)

            # 更新embedding
            embed_sum = z_flattened.t() @ encodings
            embed_normalized = (embed_sum / (bins.unsqueeze(0) + self.eps)).t()
            embed_normalized = l2norm(embed_normalized)

            zero_mask = (bins == 0)
            embed_normalized = torch.where(zero_mask.unsqueeze(-1),
                                         self.embedding, embed_normalized)

            ema_inplace(self.embed_avg, embed_normalized, self.decay)
            self.embedding.data.copy_(l2norm(self.embed_avg))

        # 直通估计器 - 使用原始输入保持梯度
        z_q = z_input + (z_q - z_input).detach()

        return z_q, loss, encoding_indices


class AdaptiveLossWeighter(nn.Module):
    """自适应损失权重调整器"""
    
    def __init__(self, num_losses=5):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
    
    def forward(self, losses):
        """
        Args:
            losses: [classification_loss, domain_loss, vq_loss, freq_loss, raw_loss]
        """
        precision = torch.exp(-self.log_vars)
        weighted_losses = []
        
        for i, loss in enumerate(losses):
            if isinstance(loss, (int, float)) and loss == 0:
                weighted_losses.append(0)
            else:
                weighted_loss = precision[i] * loss + self.log_vars[i]
                weighted_losses.append(weighted_loss)
        
        total_loss = sum(weighted_losses)
        return total_loss, precision
