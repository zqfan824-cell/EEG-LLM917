"""
集成Vector Quantization和重建损失的TimeLLM模型
参考NeuroLM的VQ机制，增强域对抗学习效果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TimeLLM import Model as TimeLLMBase, ReverseLayerF
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


class TimeLLM_VQ(TimeLLMBase):
    """
    集成Vector Quantization和重建损失的TimeLLM模型
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
            # 重建解码器 - 输出到原始序列长度和通道数
            # 注意：实际通道数会在运行时动态更新，这里使用占位符
            self.freq_decoder = nn.Sequential(
                nn.Linear(self.vq_embed_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256)  # 先输出固定大小，后续动态调整
            )

            self.raw_decoder = nn.Sequential(
                nn.Linear(self.vq_embed_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 512)  # 先输出固定大小，后续动态调整
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

            print(f"[TimeLLM_VQ] 增强模态对抗学习:")
            print(f"  - 深层域分类器: {self.d_llm} -> 512 -> 256 -> 128 -> 2")
            print(f"  - 模态对齐调度器: {getattr(configs, 'alpha_schedule', 'sigmoid')}")
            print(f"  - 对比学习温度: {getattr(configs, 'contrastive_temp', 0.1)}")
            print(f"  - 使用LayerNorm和渐进Dropout正则化")
        
        print(f"[TimeLLM_VQ] 初始化完成:")
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
            x_enc: (batch_size, n_channels, seq_len) - 输入EEG数据
            x_mark_enc: 时间标记
            alpha: 域对抗学习的alpha参数
            return_reconstruction_loss: 是否返回重建损失
        """

        # 数据加载器返回 (B, seq_len, n_channels)，需要转置为 (B, n_channels, seq_len)
        if x_enc.shape[1] > x_enc.shape[2]:  # seq_len > n_channels
            x_enc = x_enc.transpose(1, 2)  # (B, n_channels, seq_len)

        # 保存原始输入用于重建损失计算
        x_enc_original = x_enc.clone()  # (B, n_channels, seq_len)

        # 获取基本维度信息
        B = x_enc.shape[0]  # batch_size
        N = x_enc.shape[1]  # n_channels
        seq_len = x_enc.shape[2]  # sequence length



        # 1. 数据标准化
        if self.normalize_layers is not None:
            x_enc = self.normalize_layers(x_enc, 'norm')

        # 2. Patch Embedding
        x_enc = x_enc.permute(0, 2, 1).contiguous()  # (B, seq_len, N)
        enc_out, n_vars = self.patch_embedding(x_enc.float())  # (B, patch_nums, d_model)
        patch_nums = enc_out.shape[1]  # patch数量

        # enc_out 已经是 (B, patch_nums, d_model) 格式
        enc_out_reshaped = enc_out
        
        # 3. VQ编码和量化
        vq_loss = 0
        quantized_features = enc_out_reshaped
        
        if self.vq_enabled:
            # 映射到VQ空间
            vq_features = self.vq_encoder(enc_out_reshaped)  # (B, L, vq_embed_dim)
            
            # Vector Quantization
            quantized_features, vq_loss, encoding_indices = self.quantizer(vq_features)
            
            # 映射回d_model维度用于重编程
            quantized_for_reprog = self.vq_to_model(quantized_features)
        else:
            quantized_for_reprog = enc_out_reshaped
        
        # 4. 重建损失计算
        reconstruction_loss = 0
        reconstruction_dict = {}
        
        if self.reconstruction_enabled and return_reconstruction_loss and self.vq_enabled:
            # 基于 NeuroLM 的重建损失实现
            # 注意：quantized_features 可能有错误的 batch 维度，需要修正

            # 1. 修正 VQ 特征的维度
            if quantized_features.shape[0] != B:
                # 重塑 VQ 特征到正确的 batch 维度
                total_elements = quantized_features.numel()
                expected_elements = B * patch_nums * self.vq_embed_dim
                if total_elements == expected_elements:
                    quantized_features = quantized_features.view(B, patch_nums, self.vq_embed_dim)
                else:
                    # 如果元素总数不匹配，使用自适应方法
                    quantized_features = quantized_features.view(B, -1, quantized_features.shape[-1])

            # 2. 生成重建目标（类似 NeuroLM 的 Y_freq 和 Y_raw）
            # 使用原始输入数据 x_enc_original: (B, N, seq_len)
            # 频域目标：使用 FFT 获取频域表示
            x_enc_fft = torch.fft.fft(x_enc_original, dim=2)  # (B, N, seq_len)
            x_enc_freq_mag = torch.abs(x_enc_fft)  # 频域幅度
            # 取前一半频率分量
            freq_target = x_enc_freq_mag[:, :, :x_enc_freq_mag.shape[2]//2]  # (B, N, seq_len//2)

            # 时域目标：直接使用原始信号
            raw_target = x_enc_original  # (B, N, seq_len)



            # 3. 使用 VQ 特征进行重建预测
            # 将 quantized_features 投影到重建维度
            vq_dim = quantized_features.shape[-1]
            freq_dim = freq_target.shape[2]  # 频域维度
            raw_dim = raw_target.shape[2]    # 时域维度

            if not hasattr(self, 'freq_decoder') or self.freq_decoder[-1].out_features != freq_dim:
                # 动态创建或重新创建解码器（类似 NeuroLM 的 decode_task_layer）
                self.freq_decoder = nn.Sequential(
                    nn.Linear(vq_dim, vq_dim),
                    nn.Tanh(),
                    nn.Linear(vq_dim, freq_dim)  # 输出频域维度
                ).to(quantized_features.device)

            if not hasattr(self, 'raw_decoder') or self.raw_decoder[-1].out_features != raw_dim:
                self.raw_decoder = nn.Sequential(
                    nn.Linear(vq_dim, vq_dim),
                    nn.Tanh(),
                    nn.Linear(vq_dim, raw_dim)   # 输出时域维度
                ).to(quantized_features.device)

            # 4. 解码重建
            # 将 VQ 特征从 patch 维度池化到通道维度
            # quantized_features: (B, patch_nums, vq_dim)
            # 需要池化到 (B, N, vq_dim) 其中 N = x_enc.shape[1]

            # 计算每个通道对应的 patch 数量
            patches_per_channel = quantized_features.shape[1] // x_enc_original.shape[1]
            if patches_per_channel * x_enc_original.shape[1] == quantized_features.shape[1]:
                # 可以整除，直接重塑和平均
                vq_reshaped = quantized_features.view(B, x_enc_original.shape[1], patches_per_channel, vq_dim)
                vq_pooled = vq_reshaped.mean(dim=2)  # (B, N, vq_dim)
            else:
                # 不能整除，使用自适应池化
                vq_pooled = F.adaptive_avg_pool1d(
                    quantized_features.transpose(1, 2), x_enc_original.shape[1]
                ).transpose(1, 2)  # (B, N, vq_dim)

            # 频域重建
            freq_pred = self.freq_decoder(vq_pooled)  # (B, N, freq_dim)
            # 时域重建
            raw_pred = self.raw_decoder(vq_pooled)    # (B, N, seq_len)



            # 4. 计算重建损失（类似 NeuroLM 的 calculate_rec_loss）
            freq_loss = F.mse_loss(freq_pred, freq_target)
            raw_loss = F.mse_loss(raw_pred, raw_target)

            # 总重建损失
            reconstruction_loss = freq_loss + raw_loss

            reconstruction_dict = {
                'reconstruction_loss': reconstruction_loss.item(),
                'freq_loss': freq_loss.item(),
                'raw_loss': raw_loss.item()
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

        # 7. 增强模态对抗学习（参考NeuroLM的VQ_Align）
        contrastive_loss = 0
        if hasattr(self, 'modal_contrastive') and self.training:
            # 获取EEG特征（VQ量化后的特征）
            eeg_features = quantized_for_reprog  # [B, patch_nums, d_llm]

            # 确保LLM特征维度匹配
            if llm_enc_out.shape[1] != eeg_features.shape[1]:
                # 自适应池化到相同长度
                llm_features = F.adaptive_avg_pool1d(
                    llm_enc_out.transpose(1, 2), eeg_features.shape[1]
                ).transpose(1, 2)
            else:
                llm_features = llm_enc_out

            # 计算模态对比学习损失（内部处理维度对齐）
            contrastive_loss = self.modal_contrastive(eeg_features, llm_features)

            # 更新域损失（结合对比学习）
            if isinstance(domain_loss, torch.Tensor):
                domain_loss = domain_loss + 0.1 * contrastive_loss  # 对比学习权重
            else:
                domain_loss = 0.1 * contrastive_loss

        # 7. 输出层
        if self.task_name == 'classification':
            # 简化的维度处理：确保输出是 (B, patch_nums, d_llm)
            current_shape = llm_enc_out.shape

            # 如果维度不匹配，重塑到正确的形状
            if len(current_shape) == 3:
                if current_shape[0] == B and current_shape[1] == patch_nums:
                    # 已经是正确的形状
                    pass
                elif current_shape[0] == B * patch_nums:
                    # 被展平了，重塑回来
                    llm_enc_out = llm_enc_out.view(B, patch_nums, -1)
                else:
                    # 其他情况，使用自适应重塑
                    llm_enc_out = llm_enc_out.view(B, -1, current_shape[-1])
                    if llm_enc_out.shape[1] != patch_nums:
                        # 使用平均池化调整序列长度
                        llm_enc_out = F.adaptive_avg_pool1d(
                            llm_enc_out.transpose(1, 2), patch_nums
                        ).transpose(1, 2)
            else:
                # 如果不是3维，强制重塑
                llm_enc_out = llm_enc_out.view(B, patch_nums, -1)

            # 简化的分类头处理：直接使用平均池化
            # llm_enc_out: (B, patch_nums, d_llm)
            # 目标：(B, N, d_llm, patch_nums_per_var)

            # 计算每个通道的平均表示
            patch_nums_per_var = patch_nums // N
            if patch_nums_per_var * N == patch_nums:
                # 可以整除，直接重塑
                llm_enc_out = llm_enc_out.view(B, N, patch_nums_per_var, self.d_llm)
                llm_enc_out = llm_enc_out.permute(0, 1, 3, 2)  # (B, N, d_llm, patch_nums_per_var)
            else:
                # 不能整除，使用平均池化


                # 如果 patch_nums < N，说明patch数量少于通道数，需要特殊处理
                if patch_nums < N:
                    # 使用自适应池化扩展到N个通道
                    llm_enc_out = F.adaptive_avg_pool1d(
                        llm_enc_out.transpose(1, 2), N
                    ).transpose(1, 2)  # (B, N, d_llm)
                    llm_enc_out = llm_enc_out.unsqueeze(-1)  # (B, N, d_llm, 1)
                else:
                    # 正常情况：patch_nums >= N
                    patches_per_channel = patch_nums // N
                    llm_enc_out = llm_enc_out.view(B, N, patches_per_channel, self.d_llm)
                    llm_enc_out = llm_enc_out.permute(0, 1, 3, 2)  # (B, N, d_llm, patches_per_channel)


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
