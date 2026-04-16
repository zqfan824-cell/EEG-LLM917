"""
EEGLLM 训练脚本
集成 NeuroLM 的 VQ 机制和重建损失，增强 EEG 情绪分类的域对抗学习效果
"""

import os
import sys

# 设置Hugging Face镜像（在导入其他模块之前）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/.cache/huggingface"

import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from models.EEGLLM import Model as EEGLLM
from models.EEGLLM_VQ import EEGLLM_VQ
from utils.reconstruction_losses import AdaptiveLossWeighter

from utils.tools import EarlyStopping, adjust_learning_rate


parser = argparse.ArgumentParser(description='EEGLLM with Reconstruction Loss')

# 基础配置
parser.add_argument('--task_name', type=str, required=True, default='classification',
                    help='task name, options:[long_term_forecast, short_term_forecast, classification]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='DEAP_EEGLLM_VQ', help='model id')
parser.add_argument('--model', type=str, required=True, default='EEGLLM_VQ',
                    help='model name, options: [EEGLLM, EEGLLM_VQ]')

# 数据配置
parser.add_argument('--data', type=str, required=True, default='DEAP', help='dataset type')
parser.add_argument('--root_path', type=str, default='./datasets/DEAP/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# 预测配置
parser.add_argument('--seq_len', type=int, default=256, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# 模型定义
parser.add_argument('--enc_in', type=int, default=32, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=32, help='decoder input size')
parser.add_argument('--c_out', type=int, default=32, help='output size')
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# LLM配置
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model')
parser.add_argument('--llm_dim', type=int, default=4096, help='LLM model dimension')
parser.add_argument('--llm_layers', type=int, default=32, help='LLM model layers')

# Patch配置
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride for patching')

# VQ配置
parser.add_argument('--enable_vq', action='store_true', default=True, help='enable vector quantization')
parser.add_argument('--vq_embed_dim', type=int, default=128, help='VQ embedding dimension')
parser.add_argument('--vq_n_embed', type=int, default=8192, help='VQ codebook size')
parser.add_argument('--vq_beta', type=float, default=1.0, help='VQ beta parameter')

# 重建损失配置
parser.add_argument('--enable_reconstruction', action='store_true', default=False, help='enable reconstruction loss')
parser.add_argument('--use_smooth_l1', action='store_true', default=False, help='use smooth L1 loss for reconstruction')
parser.add_argument('--freq_weight', type=float, default=1.0, help='weight for frequency domain reconstruction loss')
parser.add_argument('--raw_weight', type=float, default=1.0, help='weight for time domain reconstruction loss')
parser.add_argument('--reconstruction_weight', type=float, default=0.5, help='overall weight for reconstruction loss')

# 模态对抗学习配置（增强版）
parser.add_argument('--enable_adversarial', action='store_true', default=True, help='enable domain adversarial learning')
parser.add_argument('--domain_weight', type=float, default=0.1, help='weight for domain adversarial loss')
parser.add_argument('--contrastive_weight', type=float, default=0.1, help='weight for modal contrastive learning loss')
parser.add_argument('--contrastive_temp', type=float, default=0.1, help='temperature for contrastive learning')
parser.add_argument('--alpha_schedule', type=str, default='sigmoid', help='alpha scheduling strategy: sigmoid, linear, constant')
parser.add_argument('--max_alpha', type=float, default=1.0, help='maximum alpha value for domain adversarial training')

# 优化配置
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--loss', type=str, default='CrossEntropyLoss', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# 分类任务配置
parser.add_argument('--n_class', type=int, default=2, help='number of classes for classification')
parser.add_argument('--classification_type', type=str, default='valence', help='classification type: valence, arousal')

# EEG数据配置
parser.add_argument('--channel_selection', type=str, default='comprehensive_emotion',
                    help='channel selection strategy: comprehensive_emotion, valence_specific, arousal_specific, custom_12_channels, auto')
parser.add_argument('--use_channel_selection', action='store_true', default=True, help='whether to use channel selection')

# GPU配置
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# 其他配置
parser.add_argument('--seed', type=int, default=2021, help='random seed')

args = parser.parse_args()

# 参数名映射（兼容性处理）
args.num_class = args.n_class

# 设置加速器（简化版，不使用DeepSpeed）
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

# 设置随机种子
fix_seed = args.seed
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)


def main():
    """主训练函数"""
    
    if args.task_name == 'classification':
        from exp.exp_classification_vq import Exp_Classification_VQ
        Exp = Exp_Classification_VQ
    else:
        raise ValueError(f"Unsupported task: {args.task_name}")
    
    if args.is_training:
        for ii in range(args.itr):
            # 设置实验名称
            setting = f'{args.model_id}_{args.model}_{args.data}_{args.seq_len}_{ii}'
            
            exp = Exp(args)  # 设置实验
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)
            
            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)
            
            if args.use_gpu:
                torch.cuda.empty_cache()
    else:
        ii = 0
        setting = f'{args.model_id}_{args.model}_{args.data}_{args.seq_len}_{ii}'
        
        exp = Exp(args)
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        
        if args.use_gpu:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
