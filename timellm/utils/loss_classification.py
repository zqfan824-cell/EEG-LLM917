"""
分类任务专用损失函数
包括处理类别不平衡的Focal Loss、加权交叉熵等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, List


class FocalLoss(nn.Module):
    """
    Focal Loss - 处理类别不平衡和困难样本
    Reference: "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, 
                 alpha: Optional[Union[float, List[float]]] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0):
        """
        初始化Focal Loss
        
        Args:
            alpha: 各类别的权重，可以是标量或列表
            gamma: 聚焦参数，gamma越大越关注困难样本
            reduction: 损失聚合方式 ('none', 'mean', 'sum')
            label_smoothing: 标签平滑系数
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        # 如果提供了alpha，转换为tensor
        if isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, float):
            self.alpha = torch.tensor([alpha, 1-alpha])
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        
        Args:
            inputs: 模型输出 [batch_size, num_classes]
            targets: 真实标签 [batch_size]
        
        Returns:
            损失值
        """
        # 获取类别数
        num_classes = inputs.size(1)
        
        # 标签平滑
        if self.label_smoothing > 0:
            targets = self._smooth_labels(targets, num_classes, self.label_smoothing)
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算概率
        p = F.softmax(inputs, dim=1)
        
        # 获取真实类别的概率
        if self.label_smoothing > 0:
            # 如果使用标签平滑，targets是概率分布
            pt = (p * targets).sum(dim=1)
        else:
            # 否则targets是类别索引
            pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 计算focal weight: (1-pt)^gamma
        focal_weight = (1 - pt).pow(self.gamma)
        
        # 应用focal weight
        focal_loss = focal_weight * ce_loss
        
        # 应用类别权重
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            if self.label_smoothing > 0:
                # 标签平滑情况下的alpha权重
                alpha_t = (self.alpha.unsqueeze(0) * targets).sum(dim=1)
            else:
                # 正常情况下的alpha权重
                alpha_t = self.alpha.gather(0, targets)
            
            focal_loss = alpha_t * focal_loss
        
        # 聚合损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def _smooth_labels(self, targets: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
        """
        标签平滑
        
        Args:
            targets: 原始标签 [batch_size]
            num_classes: 类别数
            smoothing: 平滑系数
        
        Returns:
            平滑后的标签 [batch_size, num_classes]
        """
        confidence = 1.0 - smoothing
        smooth_label = smoothing / (num_classes - 1)
        
        one_hot = torch.zeros((targets.size(0), num_classes), device=targets.device)
        one_hot.fill_(smooth_label)
        one_hot.scatter_(1, targets.unsqueeze(1), confidence)
        
        return one_hot


class WeightedCrossEntropyLoss(nn.Module):
    """
    加权交叉熵损失 - 处理类别不平衡
    """
    
    def __init__(self, 
                 weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0,
                 reduction: str = 'mean'):
        """
        初始化
        
        Args:
            weight: 各类别的权重
            label_smoothing: 标签平滑系数
            reduction: 损失聚合方式
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算加权交叉熵
        
        Args:
            inputs: 模型输出 [batch_size, num_classes]
            targets: 真实标签 [batch_size]
        
        Returns:
            损失值
        """
        if self.weight is not None and self.weight.device != inputs.device:
            self.weight = self.weight.to(inputs.device)
        
        loss = F.cross_entropy(
            inputs, 
            targets, 
            weight=self.weight,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )
        
        return loss


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失 - 防止过拟合
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1, reduction: str = 'mean'):
        """
        初始化
        
        Args:
            num_classes: 类别数
            smoothing: 平滑系数
            reduction: 损失聚合方式
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算标签平滑损失
        
        Args:
            inputs: 模型输出 [batch_size, num_classes]
            targets: 真实标签 [batch_size]
        
        Returns:
            损失值
        """
        # 计算log概率
        log_probs = F.log_softmax(inputs, dim=1)
        
        # 创建平滑的标签
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # KL散度损失
        loss = -torch.sum(true_dist * log_probs, dim=1)
        
        # 聚合
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CenterLoss(nn.Module):
    """
    Center Loss - 增强类内紧凑性
    通常与交叉熵损失组合使用
    """
    
    def __init__(self, num_classes: int, feat_dim: int, lambda_c: float = 0.003):
        """
        初始化
        
        Args:
            num_classes: 类别数
            feat_dim: 特征维度
            lambda_c: center loss的权重
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c
        
        # 初始化类别中心
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算Center Loss
        
        Args:
            features: 特征向量 [batch_size, feat_dim]
            labels: 标签 [batch_size]
        
        Returns:
            损失值
        """
        batch_size = features.size(0)
        
        # 计算特征到对应类别中心的距离
        centers_batch = self.centers[labels]  # [batch_size, feat_dim]
        
        # 计算L2距离
        loss = F.mse_loss(features, centers_batch, reduction='sum') / batch_size
        
        return self.lambda_c * loss
    
    def update_centers(self, features: torch.Tensor, labels: torch.Tensor, alpha: float = 0.5):
        """
        更新类别中心（可选的，用于更灵活的更新策略）
        
        Args:
            features: 特征向量
            labels: 标签
            alpha: 更新率
        """
        with torch.no_grad():
            for i in range(self.num_classes):
                mask = labels == i
                if mask.sum() > 0:
                    self.centers[i] = (1 - alpha) * self.centers[i] + alpha * features[mask].mean(dim=0)


class CombinedLoss(nn.Module):
    """
    组合损失函数 - 可以组合多个损失函数
    """
    
    def __init__(self, losses: List[nn.Module], weights: List[float] = None):
        """
        初始化
        
        Args:
            losses: 损失函数列表
            weights: 各损失函数的权重
        """
        super(CombinedLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        
        if weights is None:
            weights = [1.0] * len(losses)
        self.weights = weights
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        计算组合损失
        
        Returns:
            加权损失和
        """
        total_loss = 0
        loss_dict = {}
        
        for i, (loss_fn, weight) in enumerate(zip(self.losses, self.weights)):
            loss = loss_fn(*args, **kwargs)
            total_loss += weight * loss
            loss_dict[f'loss_{i}'] = loss.item() if torch.is_tensor(loss) else loss
        
        # 可以返回总损失和各分项损失
        # return total_loss, loss_dict
        return total_loss


def create_loss_function(loss_type: str, 
                        num_classes: int,
                        class_weights: Optional[List[float]] = None,
                        **kwargs) -> nn.Module:
    """
    创建损失函数的工厂函数
    
    Args:
        loss_type: 损失函数类型
        num_classes: 类别数
        class_weights: 类别权重
        **kwargs: 其他参数
    
    Returns:
        损失函数
    """
    if loss_type in ['ce', 'CrossEntropyLoss']:
        # 标准交叉熵
        return nn.CrossEntropyLoss()

    elif loss_type == 'weighted_ce':
        # 加权交叉熵
        if class_weights:
            weight = torch.tensor(class_weights, dtype=torch.float32)
        else:
            weight = None
        return WeightedCrossEntropyLoss(weight=weight, **kwargs)
    
    elif loss_type == 'focal':
        # Focal Loss
        # 从kwargs中移除alpha，避免冲突
        focal_kwargs = kwargs.copy()
        focal_kwargs.pop('alpha', None)  # 移除可能存在的alpha参数
        return FocalLoss(alpha=class_weights, **focal_kwargs)
    
    elif loss_type == 'label_smoothing':
        # 标签平滑
        return LabelSmoothingLoss(num_classes=num_classes, **kwargs)
    
    else:
        raise ValueError(f"未知的损失函数类型: {loss_type}")


# 计算类别权重的辅助函数
def compute_class_weights(labels: Union[List, np.ndarray, torch.Tensor], 
                         method: str = 'inverse_frequency') -> torch.Tensor:
    """
    计算类别权重
    
    Args:
        labels: 标签数组
        method: 计算方法 ('inverse_frequency', 'effective_number')
    
    Returns:
        类别权重tensor
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)
    
    # 统计各类别数量
    unique_labels, counts = np.unique(labels, return_counts=True)
    num_classes = len(unique_labels)
    
    if method == 'inverse_frequency':
        # 逆频率加权
        weights = len(labels) / (num_classes * counts)
        weights = weights / weights.sum() * num_classes  # 归一化
        
    elif method == 'effective_number':
        # Effective Number加权
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes  # 归一化
        
    else:
        raise ValueError(f"未知的权重计算方法: {method}")
    
    return torch.tensor(weights, dtype=torch.float32)


# 使用示例
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    
    # 模拟数据
    batch_size = 32
    num_classes = 3
    feat_dim = 128
    
    # 模拟不平衡数据
    labels = torch.tensor([0] * 20 + [1] * 8 + [2] * 4)
    
    # 计算类别权重
    class_weights = compute_class_weights(labels)
    print(f"类别权重: {class_weights}")
    
    # 创建不同的损失函数
    losses = {
        'CE': nn.CrossEntropyLoss(),
        'Weighted CE': create_loss_function('weighted_ce', num_classes, class_weights.tolist()),
        'Focal': create_loss_function('focal', num_classes, alpha=0.25, gamma=2.0),
        'Label Smoothing': create_loss_function('label_smoothing', num_classes, smoothing=0.1)
    }
    
    # 测试损失函数
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    print("\n不同损失函数的输出:")
    for name, loss_fn in losses.items():
        loss = loss_fn(inputs, targets)
        print(f"{name}: {loss.item():.4f}")
