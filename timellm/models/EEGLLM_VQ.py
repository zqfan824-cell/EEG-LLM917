"""
EEGLLM_VQ - 在 EEGLLM 基础上集成 Vector Quantization、重建损失和增强域对抗学习
参考 NeuroLM 的 VQ 机制，提升 EEG 与 LLM 词空间的模态对齐效果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.EEGLLM import Model as EEGLLMBase, ReverseLayerF
from utils.reconstruction_losses import NormEMAVectorQuantizer, ReconstructionLosses
import math


class ModalAlignmentScheduler:
    """
    模态对齐权重调度器 - 参考NeuroLM的alpha调度策略
    """
    def __init__(self, max_alpha=1.0, schedule_type='sigmoid'):
        self.max_alpha = max_alpha
        self.schedule_type = schedule_type

    def get_alpha(self, current_step, total_steps):
        """获取当前步骤的alpha值"""
        progress = current_step / max(total_steps, 1)

        if self.schedule_type == 'sigmoid':
            # Sigmoid调度：从小值开始，中期快速增长，后期趋于稳定
            # 修改：确保从非零值开始，避免前期完全没有域对抗学习
            alpha = self.max_alpha * (2.0 / (1.0 + math.exp(-10 * (progress - 0.1))) - 1.0)
            alpha = max(0.01 * self.max_alpha, alpha)  # 确保最小值为max_alpha的1%
        elif self.schedule_type == 'linear':
            # 线性调度
            alpha = self.max_alpha * progress
        elif self.schedule_type == 'cosine':
            # 余弦调度
            alpha = self.max_alpha * (1 - math.cos(progress * math.pi)) / 2
        else:
            alpha = self.max_alpha * progress

        return max(0.0, min(alpha, self.max_alpha))


class ModalContrastiveLearning(nn.Module):
    """
    模态对比学习模块 - 增强EEG和LLM特征的对齐
    支持不同维度的特征对齐
    """
    def __init__(self, eeg_dim, llm_dim, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.eeg_dim = eeg_dim
        self.llm_dim = llm_dim

        # 统一的投影维度
        self.proj_dim = min(eeg_dim, llm_dim) // 2

        # EEG特征投影层
        self.eeg_projector = nn.Sequential(
            nn.Linear(eeg_dim, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim // 2)
        )

        # LLM特征投影层
        self.llm_projector = nn.Sequential(
            nn.Linear(llm_dim, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim // 2)
        )

    def forward(self, eeg_features, llm_features):
        """
        计算对比学习损失
        Args:
            eeg_features: [B, L, D] EEG特征
            llm_features: [B, L, D] LLM特征
        Returns:
            contrastive_loss: 对比学习损失
        """
        # 投影到统一的低维空间
        eeg_proj = self.eeg_projector(eeg_features)  # [B, L, proj_dim//2]
        llm_proj = self.llm_projector(llm_features)  # [B, L, proj_dim//2]

        # L2归一化
        eeg_proj = F.normalize(eeg_proj, dim=-1)
        llm_proj = F.normalize(llm_proj, dim=-1)

        # 计算相似度矩阵
        # 将时间维度展平：[B*L, D//4]
        B, L, D = eeg_proj.shape
        eeg_flat = eeg_proj.view(-1, D)
        llm_flat = llm_proj.view(-1, D)

        # 计算余弦相似度 - 修复：EEG与LLM的匹配
        # 正样本：同一batch同一时间步的EEG和LLM特征应该匹配
        similarity = torch.matmul(eeg_flat, llm_flat.T) / self.temperature  # [B*L, B*L]

        # 正样本：对角线元素（同一位置的EEG和LLM特征）
        labels = torch.arange(B * L, device=eeg_features.device)

        # 对比学习损失（InfoNCE）- 确保梯度流动
        if similarity.requires_grad:
            contrastive_loss = F.cross_entropy(similarity, labels)
        else:
            contrastive_loss = torch.tensor(0.0, device=eeg_features.device, requires_grad=True)

        return contrastive_loss


class EEGLLM_VQ(EEGLLMBase):
    """
    集成 Vector Quantization 和重建损失的 EEGLLM 模型
    """
    
    def __init__(self, configs):
        super().__init__(configs)
        
        # VQ相关参数
        self.vq_enabled = getattr(configs, 'enable_vq', True)
        self.reconstruction_enabled = getattr(configs, 'enable_reconstruction', True)
        
        if self.vq_enabled:
            # Vector Quantizer参数
            self.vq_embed_dim = getattr(configs, 'vq_embed_dim', 128)
            self.vq_n_embed = getattr(configs, 'vq_n_embed', 8192)
            self.vq_beta = getattr(configs, 'vq_beta', 1.0)
            
            self.quantizer = NormEMAVectorQuantizer(
                n_embed=self.vq_n_embed,
                embedding_dim=self.vq_embed_dim,
                beta=self.vq_beta
            )
            
            # 编码器：将patch embedding映射到VQ空间
            self.vq_encoder = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model),
                nn.Tanh(),
                nn.Linear(configs.d_model, self.vq_embed_dim)
            )
            
            # 映射回d_model维度用于重编程
            self.vq_to_model = nn.Linear(self.vq_embed_dim, configs.d_model)
        
        if self.reconstruction_enabled:
            # 重建解码器 - 在 __init__ 里按 configs.seq_len 一次性定尺寸
            # FFT 频域只取前一半频率分量，所以 freq_dim = seq_len // 2
            freq_dim = configs.seq_len // 2
            raw_dim = configs.seq_len

            self.freq_decoder = nn.Sequential(
                nn.Linear(self.vq_embed_dim, self.vq_embed_dim),
                nn.Tanh(),
                nn.Linear(self.vq_embed_dim, freq_dim),
            )

            self.raw_decoder = nn.Sequential(
                nn.Linear(self.vq_embed_dim, self.vq_embed_dim),
                nn.Tanh(),
                nn.Linear(self.vq_embed_dim, raw_dim),
            )
            
            # 重建损失函数
            self.reconstruction_losses = ReconstructionLosses(
                use_smooth_l1=getattr(configs, 'use_smooth_l1', False),
                freq_weight=getattr(configs, 'freq_weight', 1.0),
                raw_weight=getattr(configs, 'raw_weight', 1.0)
            )
        
        # 增强的模态对抗学习模块（参考NeuroLM的VQ_Align）
        if hasattr(self, 'reprogramming_layer') and hasattr(self.reprogramming_layer, 'domain_classifier'):
            # 替换为更强的域分类器架构
            self.reprogramming_layer.domain_classifier = nn.Sequential(
                nn.Linear(self.d_llm, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 2)  # EEG域(0) vs LLM域(1)
            )

            # 添加模态对齐损失权重调度器
            self.modal_alignment_scheduler = ModalAlignmentScheduler(
                max_alpha=getattr(configs, 'max_alpha', 1.0),
                schedule_type=getattr(configs, 'alpha_schedule', 'sigmoid')
            )

            # 添加模态特征对比学习模块
            self.modal_contrastive = ModalContrastiveLearning(
                eeg_dim=configs.d_model,  # VQ后映射回d_model的EEG特征维度
                llm_dim=self.d_llm,       # LLM特征维度
                temperature=getattr(configs, 'contrastive_temp', 0.1)
            )

            # 初始化增强域分类器
            self._init_enhanced_domain_classifier()

            print(f"[EEGLLM_VQ] 增强模态对抗学习:")
            print(f"  - 深层域分类器: {self.d_llm} -> 512 -> 256 -> 128 -> 2")
            print(f"  - 模态对齐调度器: {getattr(configs, 'alpha_schedule', 'sigmoid')}")
            print(f"  - 对比学习温度: {getattr(configs, 'contrastive_temp', 0.1)}")
            print(f"  - 使用LayerNorm和渐进Dropout正则化")
        
        print(f"[EEGLLM_VQ] 初始化完成:")
        print(f"  - VQ启用: {self.vq_enabled}")
        print(f"  - 重建损失启用: {self.reconstruction_enabled}")
        if self.vq_enabled:
            print(f"  - VQ嵌入维度: {self.vq_embed_dim}")
            print(f"  - VQ码本大小: {self.vq_n_embed}")

    def _init_enhanced_domain_classifier(self):
        """初始化增强域分类器权重 - 参考NeuroLM的初始化策略"""
        for m in self.reprogramming_layer.domain_classifier.modules():
            if isinstance(m, nn.Linear):
                # 使用截断正态分布初始化权重
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None,
                mask=None, alpha=0.0, return_reconstruction_loss=False):
        """
        增强版前向传播，支持VQ和重建损失

        Args:
            x_enc: (batch_size, seq_len, n_channels) - 输入EEG数据（与数据加载器、基类 EEGLLM 一致）
            x_mark_enc: 时间标记
            alpha: 域对抗学习的alpha参数
            return_reconstruction_loss: 是否返回重建损失
        """
        B, seq_len, N = x_enc.shape

        # 1. RevIN 标准化（末轴 = channel，与 Normalize(configs.enc_in, ...) 构造一致）
        if self.normalize_layers is not None:
            x_enc = self.normalize_layers(x_enc, 'norm')

        # 2. 单次转置得到 (B, N, seq_len)，既作为 FFT 重建目标，也作为 PatchEmbedding 输入
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        x_enc_original = x_enc.clone()  # (B, N, seq_len)

        # PatchEmbedding 契约要求 (B, N, T)，输出 (B*N, num_patches, d_model)
        enc_out, n_vars = self.patch_embedding(x_enc.float())
        num_patches = enc_out.shape[1]

        # 3. VQ 编码和量化（输入输出均保持 (B*N, num_patches, *) 形状）
        vq_loss = 0
        quantized_features = enc_out

        if self.vq_enabled:
            vq_features = self.vq_encoder(enc_out)  # (B*N, num_patches, vq_embed_dim)
            quantized_features, vq_loss, encoding_indices = self.quantizer(vq_features)
            quantized_for_reprog = self.vq_to_model(quantized_features)  # (B*N, num_patches, d_model)
        else:
            quantized_for_reprog = enc_out
        
        # 4. 重建损失计算
        reconstruction_loss = 0
        reconstruction_dict = {}
        
        if self.reconstruction_enabled and return_reconstruction_loss and self.vq_enabled:
            # 重建目标（参考 NeuroLM 的 Y_freq / Y_raw）
            # x_enc_original: (B, N, seq_len) —— FFT 沿时间轴
            x_enc_fft = torch.fft.fft(x_enc_original, dim=2)
            freq_target = torch.abs(x_enc_fft)[:, :, :seq_len // 2]  # (B, N, seq_len // 2)
            raw_target = x_enc_original                               # (B, N, seq_len)

            # 按通道分组：(B*N, num_patches, vq_embed_dim) → (B, N, vq_embed_dim)
            quantized_for_recon = quantized_features.view(B, N, num_patches, self.vq_embed_dim)
            vq_pooled = quantized_for_recon.mean(dim=2)

            freq_pred = self.freq_decoder(vq_pooled)  # (B, N, seq_len // 2)
            raw_pred = self.raw_decoder(vq_pooled)    # (B, N, seq_len)

            freq_loss = F.mse_loss(freq_pred, freq_target)
            raw_loss = F.mse_loss(raw_pred, raw_target)
            reconstruction_loss = freq_loss + raw_loss

            reconstruction_dict = {
                'reconstruction_loss': reconstruction_loss.item(),
                'freq_loss': freq_loss.item(),
                'raw_loss': raw_loss.item(),
            }
        
        # 5. 重编程层处理
        prompt = self.reprogramming_layer(
            quantized_for_reprog, 
            self.word_embeddings, 
            self.word_embeddings,
            alpha=alpha,
            return_domain_loss=True
        )
        
        # 如果重编程层返回元组（包含域损失）
        if isinstance(prompt, tuple):
            prompt, domain_loss = prompt
        else:
            domain_loss = 0
        
        # 6. LLM处理
        llm_enc_out = self.llm_model(inputs_embeds=prompt).last_hidden_state

        # 7. 模态对比学习（参考 NeuroLM 的 VQ_Align）
        #    eeg_features: (B*N, num_patches, d_model)
        #    llm_enc_out : (B*N, num_patches, d_llm)
        #    ModalContrastiveLearning 内部分别投影 eeg_dim/llm_dim，允许维度不同
        contrastive_loss = 0
        if hasattr(self, 'modal_contrastive') and self.training:
            eeg_features = quantized_for_reprog
            contrastive_loss = self.modal_contrastive(eeg_features, llm_enc_out)

            if isinstance(domain_loss, torch.Tensor):
                domain_loss = domain_loss + 0.1 * contrastive_loss
            else:
                domain_loss = 0.1 * contrastive_loss

        # 8. 分类头：llm_enc_out 形状为 (B*N, num_patches, d_llm)，按基类模式整形
        if self.task_name == 'classification':
            llm_enc_out = llm_enc_out.reshape(B, N, num_patches, self.d_llm)
            llm_enc_out = llm_enc_out.permute(0, 1, 3, 2).contiguous()  # (B, N, d_llm, num_patches)
            output = self.output_projection(llm_enc_out)
        else:
            output = self.output_projection(llm_enc_out)
        
        # 8. 反标准化（如果需要）
        if self.normalize_layers is not None and self.task_name != 'classification':
            output = self.normalize_layers(output, 'denorm')
        
        # 返回结果
        if return_reconstruction_loss:
            return output, {
                'vq_loss': vq_loss,
                'domain_loss': domain_loss,
                'contrastive_loss': contrastive_loss,
                'reconstruction_loss': reconstruction_loss,
                'reconstruction_dict': reconstruction_dict
            }
        else:
            return output
