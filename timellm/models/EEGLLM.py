"""
EEGLLM 模型 - 面向 EEG 情绪分类
基于 Time-LLM (ICLR 2024) 的重编程框架改造，把冻结的 LLM 用作 EEG 特征抽取的骨干。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Function
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
from layers.Embed import PatchEmbedding
from layers.StandardNorm import Normalize
import transformers

# 设置Hugging Face镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/.cache/huggingface"

transformers.logging.set_verbosity_error()


class ReverseLayerF(Function):
    """
    反向梯度层 - 参考NeuroLM实现
    用于域对抗训练，在前向传播时保持不变，在反向传播时反转梯度
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class ClassificationHead(nn.Module):
    """新增：分类头，用于EEG情绪分类"""
    def __init__(self, n_vars, d_model, patch_nums, num_class, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.patch_nums = patch_nums
        self.num_class = num_class
        
        # 方案1：最简单的平均池化 + 线性层
        # 输入: [batch, n_vars, d_model, patch_nums]
        # 目标: [batch, num_class]
        
        # 计算输入特征维度
        self.input_dim = n_vars * d_model
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.input_dim // 2, num_class)
        )
        
        print(f"[ClassificationHead] 初始化完成:")
        print(f"  - 输入维度: {self.input_dim} (n_vars={n_vars} × d_model={d_model})")
        print(f"  - 输出类别: {num_class}")
        
    def forward(self, x):
        """
        输入: x shape [batch, n_vars, d_model, patch_nums]
        输出: [batch, num_class]
        """
        batch_size = x.shape[0]
        
        # 打印输入维度（调试用）
        # print(f"[ClassificationHead] 输入形状: {x.shape}")
        
        # Step 1: 对所有patch取平均 (最简单的聚合方式)
        # [batch, n_vars, d_model, patch_nums] -> [batch, n_vars, d_model]
        x = torch.mean(x, dim=-1)
        
        # Step 2: 展平通道维度
        # [batch, n_vars, d_model] -> [batch, n_vars * d_model]
        x = x.reshape(batch_size, -1)
        
        # Step 3: 分类
        # [batch, n_vars * d_model] -> [batch, num_class]
        x = self.classifier(x)
        
        return x


class Model(nn.Module):
    """EEGLLM 基础模型 - 用于 EEG 情绪分类"""

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # 打印任务信息
        print(f"\n[EEGLLM] 初始化模型 - 任务类型: {self.task_name}")
        
        # LLM配置（保持原样）
        if configs.llm_model == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                )
            except EnvironmentError:
                print("下载Llama模型...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )
                
            self.tokenizer = LlamaTokenizer.from_pretrained(
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=True,
            )
            
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token


        elif configs.llm_model == 'GPT2':
            from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

            # 使用在线GPT2模型
            gpt2_model_id = 'gpt2'

            # 加载配置
            self.gpt2_config = GPT2Config.from_pretrained(gpt2_model_id)
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_hidden_states = True
            self.gpt2_config.output_attentions = True

            # 加载模型
            self.llm_model = GPT2Model.from_pretrained(
                gpt2_model_id,
                config=self.gpt2_config
            )

            # 加载分词器
            self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_id)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # 冻结LLM
        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.description = 'EEG signals for emotion recognition'
        self.dropout = nn.Dropout(configs.dropout)
        
        # Patch嵌入层（保持原样）
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)
        
        # 词嵌入和重编程层（保持原样）
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        
        # 启用域对抗学习的重编程层
        enable_adversarial = getattr(configs, 'enable_adversarial', True)
        self.reprogramming_layer = ReprogrammingLayer(
            configs.d_model, configs.n_heads, self.d_ff, self.d_llm,
            enable_adversarial=enable_adversarial)
        
        # 计算patch数量
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        # 分类头
        if self.task_name != 'classification':
            raise NotImplementedError(
                f"EEGLLM 仅支持分类任务 (classification)，收到: {self.task_name}")

        self.output_projection = ClassificationHead(
            n_vars=configs.enc_in,
            d_model=self.d_llm,  # 使用LLM的输出维度而不是d_ff
            patch_nums=self.patch_nums,
            num_class=configs.num_class,
            dropout=configs.dropout
        )
        print(f"[EEGLLM] 分类任务配置:")
        print(f"  - 类别数: {configs.num_class}")
        print(f"  - 通道数: {configs.enc_in}")
        print(f"  - Patch数: {self.patch_nums}")

        # 标准化层
        self.normalize_layers = Normalize(configs.enc_in, affine=False)
    
    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None, alpha=0.0, return_domain_loss=False):
        """
        EEG 分类 forward 接口（x_dec / x_mark_dec / mask 仅为保持签名兼容，未使用）

        Args:
            x_enc: 输入数据 [batch_size, seq_len, n_channels]
            x_mark_enc: 时间标记
            alpha: 域对抗训练权重
            return_domain_loss: 是否返回域分类损失

        Returns:
            return_domain_loss=False: [batch_size, num_class] 分类 logits
            return_domain_loss=True : (分类 logits, 域分类损失)
        """
        return self.classification(
            x_enc, x_mark_enc, alpha=alpha, return_domain_loss=return_domain_loss)
    
    def classification(self, x_enc, x_mark_enc, alpha=0.0, return_domain_loss=False):
        """
        增强版分类任务前向传播 - 支持域对抗学习

        Args:
            x_enc: EEG输入数据 [batch_size, seq_len, n_channels]
            x_mark_enc: 时间标记（暂未使用）
            alpha: 域对抗训练权重
            return_domain_loss: 是否返回域分类损失

        Returns:
            如果return_domain_loss=False: 分类logits [batch_size, num_class]
            如果return_domain_loss=True: (分类logits, 域分类损失)
        """
        # 打印输入维度（调试用）
        # print(f"[classification] 输入 x_enc 形状: {x_enc.shape}")
        
        # Step 1: 标准化
        x_enc = self.normalize_layers(x_enc, 'norm')
        
        # Step 2: 重塑数据（与forecast相同）
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        
        # Step 3: 计算统计信息（简化版提示词）
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)
        
        # Step 4: 生成提示词（基于DEAP.txt的专业提示词）
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            
            # 基于DEAP.txt内容的专业提示词
            if N == 32:  # DEAP数据集（32通道）
                prompt_ = (
                    f"<|start_prompt|>Dataset Description: The DEAP dataset contains 32-channel EEG signals recorded from participants experiencing emotional stimuli through music videos. Participants rated each video on continuous scales of valence (negative to positive) and arousal (calm to excited)."
                    f"Task Instruction: This is a binary emotion classification task. Analyze the EEG signal patterns to determine whether the participant is experiencing positive (high valence) or negative (low valence) emotional states."
                    f"EEG Channel Configuration: 32 channels following the International 10-20 system. Frontal channels (Fp1, Fp2, F3, F4, F7, F8, Fz) associated with emotional processing and executive control. Central channels (C3, C4, Cz) related to sensorimotor integration. Parietal channels (P3, P4, P7, P8, Pz) involved in attention and spatial processing. Occipital channels (O1, O2) for visual processing. Temporal channels (T7, T8) for auditory and memory processing."
                    f"Emotion-Related EEG Patterns to analyze:"
                    f"Positive Emotions (High Valence): Increased left frontal alpha power (8-13 Hz) relative to right frontal, enhanced gamma band activity (30-45 Hz) in frontal regions, higher beta power (13-30 Hz) indicating active cognitive engagement, greater coherence between frontal and parietal regions."
                    f"Negative Emotions (Low Valence): Increased right frontal alpha activity, reduced overall gamma power, enhanced theta activity (4-8 Hz) in frontal and central regions, lower beta/alpha ratio, increased delta power (0.5-4 Hz) in some cases."
                    f"Input statistics for current sample: min value {min_values_str}, max value {max_values_str}, median value {median_values_str}, signal trend {'increasing' if trends[b] > 0 else 'decreasing'}."
                    f"Key Features to Analyze: Power spectral density in each frequency band, frontal asymmetry patterns, inter-channel coherence, temporal dynamics and transitions, signal complexity and entropy."
                    f"Classification: Based on the 32-channel EEG signal patterns, classify whether the emotional state is positive (high valence) or negative (low valence).<|end_prompt|>"
                )
            elif N == 62:  # SEED数据集（62通道）
                prompt_ = (
                    f"<|start_prompt|>Dataset Description: The SEED dataset contains 62-channel EEG signals recorded during emotion elicitation using carefully selected film clips. The dataset targets three distinct emotional states: positive (happiness), neutral, and negative (sadness/fear)."
                    f"Task Instruction: This is a three-class emotion classification task. Analyze the comprehensive EEG signal patterns to classify the participant's emotional state into one of three categories: positive, neutral, or negative."
                    f"EEG Channel Configuration: 62 channels with extended 10-20 montage providing high spatial resolution. Prefrontal channels (Fp1, Fp2, AFz, AF3, AF4) for executive control and emotion regulation. Frontal channels (F1-F8, Fz, FCz) for emotional processing and motor planning. Central channels (C1-C6, Cz, CPz) for sensorimotor integration. Parietal channels (P1-P8, Pz, POz) for attention and spatial awareness. Occipital channels (O1, O2, Oz) for visual processing. Temporal channels (T7, T8, TP7, TP8, FT7, FT8) for auditory and memory."
                    f"Emotion-Specific EEG Signatures:"
                    f"Positive Emotions: Significantly increased gamma power (30-50 Hz) in left temporal and frontal regions, enhanced beta activity (14-30 Hz) across frontal cortex, characteristic frontal alpha asymmetry (left < right), high-frequency oscillations in temporal-parietal networks, increased connectivity between frontal and temporal regions."
                    f"Neutral State: Balanced alpha power (8-13 Hz) between hemispheres, moderate beta activity without significant lateralization, lower gamma power compared to emotional states, stable theta rhythms (4-8 Hz) in frontal-central areas, reduced long-range connectivity."
                    f"Negative Emotions: Increased right frontal and temporal alpha power, enhanced theta activity in frontal and central regions, reduced gamma power especially in left hemisphere, increased delta power (1-4 Hz) in frontal areas, greater alpha-theta coupling, heightened connectivity in right hemisphere networks."
                    f"Input statistics for current sample: min value {min_values_str}, max value {max_values_str}, median value {median_values_str}, signal trend {'increasing' if trends[b] > 0 else 'decreasing'}."
                    f"Critical Analysis Points: Differential entropy features across bands, hemispheric power asymmetry, phase-locking values between regions, spectral power distribution, temporal evolution of emotional patterns, cross-frequency coupling."
                    f"Classification: Based on the 62-channel EEG signal patterns, classify the emotional state into positive, neutral, or negative categories.<|end_prompt|>"
                )
            else:  # 通用提示词
                prompt_ = (
                    f"<|start_prompt|>Dataset description: {self.description};"
                    f"Task description: classify the emotional state based on EEG signals;"
                    f"Input statistics: min value {min_values_str}, max value {max_values_str}, median value {median_values_str}, signal trend {'increasing' if trends[b] > 0 else 'decreasing'};"
                    f"Analyze the multi-channel EEG signal patterns to determine the emotional state classification.<|end_prompt|>"
                )
            prompt.append(prompt_)
        
        # 其余代码保持不变...
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        
        # 获取提示词嵌入
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, 
                              truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))
        
        # 其余处理流程保持不变...
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.float())

        # 域对抗学习的重编程
        if return_domain_loss:
            enc_out, domain_loss = self.reprogramming_layer(
                enc_out, source_embeddings, source_embeddings,
                alpha=alpha, return_domain_loss=True)
        else:
            enc_out = self.reprogramming_layer(
                enc_out, source_embeddings, source_embeddings,
                alpha=alpha, return_domain_loss=False)
            domain_loss = None

        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)

        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = dec_out[:, :, :, -self.patch_nums:]

        output = self.output_projection(dec_out)

        # 返回结果
        if return_domain_loss:
            return output, domain_loss
        else:
            return output
    
    def calcute_lags(self, x_enc):
        """计算滞后相关性（用于 prompt 统计）"""
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    """
    增强版重编程层 - 集成域对抗学习
    参考NeuroLM的VQ_Align模块，添加域分类器实现模态对齐
    """
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1, enable_adversarial=True):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        self.d_llm = d_llm
        self.enable_adversarial = enable_adversarial

        # 原始的重编程组件
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

        # 新增：域分类器 - 参考NeuroLM实现
        if self.enable_adversarial:
            self.domain_classifier = nn.Sequential(
                nn.Linear(d_llm, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 2)  # 2个域：EEG域(0) 和 LLM域(1)
            )

            # 初始化域分类器权重
            self._init_domain_classifier()

            print(f"[ReprogrammingLayer] 启用域对抗学习:")
            print(f"  - 域分类器输入维度: {d_llm}")
            print(f"  - 域分类器隐藏维度: 256")
            print(f"  - 域数量: 2 (EEG域 vs LLM域)")
        else:
            self.domain_classifier = None
            print(f"[ReprogrammingLayer] 未启用域对抗学习")

    def _init_domain_classifier(self):
        """初始化域分类器权重"""
        for m in self.domain_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, target_embedding, source_embedding, value_embedding, alpha=0.0, return_domain_loss=False):
        """
        增强版前向传播，支持域对抗学习

        Args:
            target_embedding: EEG patch嵌入 [B, L, d_model]
            source_embedding: LLM词嵌入 [S, d_llm]
            value_embedding: LLM词嵌入 [S, d_llm]
            alpha: 对抗训练权重，控制梯度反转强度
            return_domain_loss: 是否返回域分类损失

        Returns:
            如果return_domain_loss=False: 重编程后的嵌入 [B, L, d_llm]
            如果return_domain_loss=True: (重编程嵌入, 域分类损失)
        """
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        # 原始重编程过程
        target_proj = self.query_projection(target_embedding).view(B, L, H, -1)
        source_proj = self.key_projection(source_embedding).view(S, H, -1)
        value_proj = self.value_projection(value_embedding).view(S, H, -1)

        reprogrammed = self.reprogramming(target_proj, source_proj, value_proj)
        reprogrammed = reprogrammed.reshape(B, L, -1)

        # 输出投影
        out = self.out_projection(reprogrammed)

        # 域对抗学习
        if self.enable_adversarial and return_domain_loss:
            domain_loss = self._compute_domain_loss(out, source_embedding, alpha)
            return out, domain_loss
        else:
            return out

    def _compute_domain_loss(self, eeg_features, llm_features, alpha):
        """
        计算域分类损失，实现模态对齐

        Args:
            eeg_features: EEG特征 [B, L, d_llm]
            llm_features: LLM特征 [S, d_llm]
            alpha: 梯度反转权重

        Returns:
            domain_loss: 域分类损失
        """
        if not self.enable_adversarial or self.domain_classifier is None:
            return torch.tensor(0.0, device=eeg_features.device)

        # 准备EEG域数据
        B, L, D = eeg_features.shape
        eeg_flat = eeg_features.view(-1, D)  # [B*L, d_llm]

        # 准备LLM域数据 (随机采样与EEG相同数量的token)
        S, _ = llm_features.shape
        if S >= B * L:
            # 如果LLM token足够，随机采样
            indices = torch.randperm(S, device=llm_features.device)[:B * L]
            llm_sample = llm_features[indices]  # [B*L, d_llm]
        else:
            # 如果LLM token不够，重复采样
            repeat_times = (B * L + S - 1) // S
            llm_repeated = llm_features.repeat(repeat_times, 1)
            llm_sample = llm_repeated[:B * L]  # [B*L, d_llm]

        # 合并两个域的特征
        combined_features = torch.cat([eeg_flat, llm_sample], dim=0)  # [2*B*L, d_llm]

        # 创建域标签：EEG域=0, LLM域=1
        eeg_labels = torch.zeros(B * L, dtype=torch.long, device=eeg_features.device)
        llm_labels = torch.ones(B * L, dtype=torch.long, device=eeg_features.device)
        domain_labels = torch.cat([eeg_labels, llm_labels], dim=0)  # [2*B*L]

        # 应用梯度反转层
        reversed_features = ReverseLayerF.apply(combined_features, alpha)

        # 域分类
        domain_logits = self.domain_classifier(reversed_features)  # [2*B*L, 2]

        # 计算域分类损失
        domain_loss = F.cross_entropy(domain_logits, domain_labels)

        return domain_loss

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


