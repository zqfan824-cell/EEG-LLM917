"""
分类任务训练辅助工具
整合学习率调度、早停、模型保存等功能
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import json
from datetime import datetime
import logging
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    带预热的余弦退火学习率调度器
    """
    
    def __init__(self, 
                 optimizer, 
                 warmup_epochs: int, 
                 max_epochs: int,
                 warmup_start_lr: float = 1e-5,
                 eta_min: float = 1e-6,
                 last_epoch: int = -1):
        """
        初始化
        
        Args:
            optimizer: 优化器
            warmup_epochs: 预热epoch数
            max_epochs: 总epoch数
            warmup_start_lr: 预热起始学习率
            eta_min: 最小学习率
            last_epoch: 上一个epoch
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """计算当前学习率"""
        if self.last_epoch < self.warmup_epochs:
            # 预热阶段：线性增长
            lr_scale = (self.last_epoch + 1) / self.warmup_epochs
            lr = [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * lr_scale 
                  for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = [self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + np.cos(np.pi * progress))
                  for base_lr in self.base_lrs]
        
        return lr


class EarlyStopping:
    """
    早停机制
    """
    
    def __init__(self, 
                 patience: int = 10, 
                 min_delta: float = 0.0001,
                 mode: str = 'min',
                 verbose: bool = True):
        """
        初始化
        
        Args:
            patience: 容忍epoch数
            min_delta: 最小改善阈值
            mode: 'min'表示越小越好，'max'表示越大越好
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
        if mode == 'min':
            self.is_better = lambda new, best: new < best - min_delta
        else:
            self.is_better = lambda new, best: new > best + min_delta
    
    def __call__(self, value: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            value: 当前指标值
        
        Returns:
            是否应该早停
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.is_better(value, self.best_value):
            self.best_value = value
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: 指标改善到 {value:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: 没有改善 ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("EarlyStopping: 停止训练")
        
        return self.early_stop


class ModelCheckpoint:
    """
    模型检查点保存
    """
    
    def __init__(self,
                 checkpoint_dir: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_last: bool = True,
                 verbose: bool = True):
        """
        初始化
        
        Args:
            checkpoint_dir: 保存目录
            monitor: 监控的指标
            mode: 'min'或'max'
            save_best_only: 是否只保存最佳模型
            save_last: 是否保存最后的模型
            verbose: 是否打印信息
        """
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.verbose = verbose
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.best_value = None
        if mode == 'min':
            self.is_better = lambda new, best: new < best
        else:
            self.is_better = lambda new, best: new > best
    
    def save_checkpoint(self,
                       epoch: int,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       metrics: Dict[str, float],
                       is_best: bool = False):
        """
        保存检查点
        
        Args:
            epoch: 当前epoch
            model: 模型
            optimizer: 优化器
            metrics: 指标字典
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存最新模型
        if self.save_last:
            last_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pth')
            torch.save(checkpoint, last_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            if self.verbose:
                print(f"保存最佳模型 (epoch {epoch}, {self.monitor}={metrics.get(self.monitor, 0):.4f})")
        
        # 保存epoch检查点
        if not self.save_best_only:
            epoch_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def __call__(self, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, metrics: Dict[str, float]):
        """
        检查并保存模型
        
        Args:
            epoch: 当前epoch
            model: 模型
            optimizer: 优化器
            metrics: 指标字典
        """
        current_value = metrics.get(self.monitor)
        
        if current_value is None:
            if self.verbose:
                print(f"警告: 指标 '{self.monitor}' 不在metrics中")
            return
        
        is_best = False
        if self.best_value is None or self.is_better(current_value, self.best_value):
            self.best_value = current_value
            is_best = True
        
        self.save_checkpoint(epoch, model, optimizer, metrics, is_best)


class GradientClipping:
    """
    梯度裁剪
    """
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        """
        初始化
        
        Args:
            max_norm: 最大梯度范数
            norm_type: 范数类型
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def __call__(self, model: nn.Module) -> float:
        """
        执行梯度裁剪
        
        Args:
            model: 模型
        
        Returns:
            裁剪前的梯度范数
        """
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            self.max_norm, 
            self.norm_type
        )
        return total_norm.item()


class TrainingLogger:
    """
    训练日志记录器
    """
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        初始化
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志文件
        log_file = os.path.join(log_dir, f'{experiment_name}.log')
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(experiment_name)
        
        # 指标历史
        self.history = {}
    
    def log(self, message: str, level: str = 'info'):
        """记录日志"""
        if level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float], prefix: str = ''):
        """
        记录指标
        
        Args:
            epoch: 当前epoch
            metrics: 指标字典
            prefix: 前缀（如'train_', 'val_'）
        """
        # 构建日志消息
        msg = f"Epoch {epoch}"
        for key, value in metrics.items():
            msg += f" | {prefix}{key}: {value:.4f}"
        
        self.log(msg)
        
        # 保存到历史
        for key, value in metrics.items():
            full_key = f"{prefix}{key}"
            if full_key not in self.history:
                self.history[full_key] = []
            self.history[full_key].append(value)
    
    def save_history(self):
        """保存训练历史"""
        history_file = os.path.join(self.log_dir, f'{self.experiment_name}_history.json')
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=4)


class MixupAugmentation:
    """
    Mixup数据增强
    """
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        初始化
        
        Args:
            alpha: Beta分布参数
            prob: 使用mixup的概率
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        应用Mixup
        
        Args:
            x: 输入数据 [batch_size, ...]
            y: 标签 [batch_size]
        
        Returns:
            mixed_x: 混合后的输入
            y_a: 第一个标签
            y_b: 第二个标签
            lam: 混合系数
        """
        if np.random.random() > self.prob:
            return x, y, y, torch.ones(1), 1.0
        
        batch_size = x.size(0)
        
        # 生成混合系数
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)
        
        # 随机打乱索引
        index = torch.randperm(batch_size).to(x.device)
        
        # 混合输入
        mixed_x = lam * x + (1 - lam) * x[index]
        
        # 返回两个标签
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, torch.tensor([lam]), lam


def create_optimizer(model: nn.Module, 
                    optimizer_type: str = 'adamw',
                    learning_rate: float = 1e-3,
                    weight_decay: float = 1e-4,
                    **kwargs) -> torch.optim.Optimizer:
    """
    创建优化器
    
    Args:
        model: 模型
        optimizer_type: 优化器类型
        learning_rate: 学习率
        weight_decay: 权重衰减
        **kwargs: 其他参数
    
    Returns:
        优化器
    """
    # 获取需要训练的参数
    params = filter(lambda p: p.requires_grad, model.parameters())
    
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            **{k: v for k, v in kwargs.items() if k != 'momentum'}
        )
    else:
        raise ValueError(f"未知的优化器类型: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_type: str = 'cosine',
                    num_epochs: int = 100,
                    warmup_epochs: int = 5,
                    **kwargs) -> _LRScheduler:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
        num_epochs: 总epoch数
        warmup_epochs: 预热epoch数
        **kwargs: 其他参数
    
    Returns:
        学习率调度器
    """
    if scheduler_type == 'cosine':
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=num_epochs,
            **kwargs
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
    elif scheduler_type == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            **{k: v for k, v in kwargs.items() if k not in ['mode', 'factor', 'patience']}
        )
    else:
        raise ValueError(f"未知的调度器类型: {scheduler_type}")
    
    return scheduler


class ClassBalancedSampler(torch.utils.data.Sampler):
    """
    类别平衡采样器
    """
    
    def __init__(self, labels: List[int], num_samples: Optional[int] = None):
        """
        初始化
        
        Args:
            labels: 标签列表
            num_samples: 每个epoch的样本数
        """
        self.labels = np.array(labels)
        self.num_samples = num_samples or len(labels)
        
        # 计算每个类别的权重
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        self.weights = class_weights[labels]
        self.weights = self.weights / self.weights.sum()
    
    def __iter__(self):
        indices = np.random.choice(
            len(self.labels),
            size=self.num_samples,
            replace=True,
            p=self.weights
        )
        return iter(indices)
    
    def __len__(self):
        return self.num_samples


# 训练状态管理
class TrainingState:
    """
    训练状态管理器
    """
    
    def __init__(self):
        self.epoch = 0
        self.global_step = 0
        self.best_metric = None
        self.metrics_history = []
    
    def update(self, metrics: Dict[str, float]):
        """更新状态"""
        self.epoch += 1
        self.metrics_history.append({
            'epoch': self.epoch,
            'step': self.global_step,
            **metrics
        })
    
    def save(self, filepath: str):
        """保存状态"""
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'metrics_history': self.metrics_history
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=4)
    
    def load(self, filepath: str):
        """加载状态"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.epoch = state['epoch']
        self.global_step = state['global_step']
        self.best_metric = state['best_metric']
        self.metrics_history = state['metrics_history']


# 使用示例
if __name__ == "__main__":
    # 创建模拟模型和数据
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        
        def forward(self, x):
            return self.fc(x)
    
    model = DummyModel()
    
    # 创建优化器和调度器
    optimizer = create_optimizer(model, 'adamw', learning_rate=1e-3)
    scheduler = create_scheduler(optimizer, 'cosine', num_epochs=100, warmup_epochs=5)
    
    # 创建训练辅助工具
    early_stopping = EarlyStopping(patience=10, mode='max')
    checkpoint = ModelCheckpoint('./checkpoints', monitor='val_acc', mode='max')
    gradient_clipper = GradientClipping(max_norm=1.0)
    logger = TrainingLogger('./logs', 'test_experiment')
    
    # 模拟训练过程
    for epoch in range(20):
        # 模拟指标
        train_metrics = {
            'loss': 1.0 - epoch * 0.04,
            'acc': 0.5 + epoch * 0.02
        }
        
        val_metrics = {
            'loss': 1.1 - epoch * 0.035,
            'acc': 0.48 + epoch * 0.018
        }
        
        # 记录日志
        logger.log_metrics(epoch, train_metrics, prefix='train_')
        logger.log_metrics(epoch, val_metrics, prefix='val_')
        
        # 检查早停
        if early_stopping(val_metrics['acc']):
            print("Early stopping triggered!")
            break
        
        # 保存检查点
        checkpoint(epoch, model, optimizer, val_metrics)
        
        # 更新学习率
        scheduler.step()
        
    # 保存训练历史
    logger.save_history()
    