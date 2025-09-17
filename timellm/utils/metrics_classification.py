"""
分类任务评估指标
包含准确率、F1分数、混淆矩阵、分类报告等
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Tuple


class ClassificationMetrics:
    """分类任务评估指标集合"""
    
    def __init__(self, num_classes: int, class_names: List[str] = None):
        """
        初始化
        
        Args:
            num_classes: 类别数量
            class_names: 类别名称列表
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """重置所有存储的预测和标签"""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, preds: torch.Tensor, labels: torch.Tensor, probs: torch.Tensor = None):
        """
        更新预测和标签
        
        Args:
            preds: 预测类别 [batch_size]
            labels: 真实标签 [batch_size]
            probs: 预测概率 [batch_size, num_classes] (可选)
        """
        # 转换为numpy
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if probs is not None and isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        
        # 存储
        self.all_preds.extend(preds)
        self.all_labels.extend(labels)
        if probs is not None:
            self.all_probs.extend(probs)
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        计算所有指标
        
        Returns:
            包含所有指标的字典
        """
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        # 基础指标
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'f1_weighted': f1_score(labels, preds, average='weighted'),
            'cohen_kappa': cohen_kappa_score(labels, preds)
        }
        
        # 每个类别的指标
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None
        )
        
        for i in range(self.num_classes):
            metrics[f'{self.class_names[i]}_precision'] = precision[i]
            metrics[f'{self.class_names[i]}_recall'] = recall[i]
            metrics[f'{self.class_names[i]}_f1'] = f1[i]
            metrics[f'{self.class_names[i]}_support'] = int(support[i])
        
        # 如果有概率，计算AUC
        if self.all_probs:
            probs = np.array(self.all_probs)
            if self.num_classes == 2:
                # 二分类
                metrics['auc'] = roc_auc_score(labels, probs[:, 1])
            else:
                # 多分类
                try:
                    metrics['auc_macro'] = roc_auc_score(
                        labels, probs, multi_class='ovr', average='macro'
                    )
                except:
                    pass  # 某些情况下可能无法计算AUC
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """获取混淆矩阵"""
        return confusion_matrix(self.all_labels, self.all_preds)
    
    def get_classification_report(self) -> str:
        """获取分类报告"""
        return classification_report(
            self.all_labels, 
            self.all_preds,
            target_names=self.class_names,
            digits=4
        )
    
    def plot_confusion_matrix(self, save_path: str = None, figsize: Tuple[int, int] = (8, 6)):
        """
        绘制混淆矩阵
        
        Args:
            save_path: 保存路径（可选）
            figsize: 图像大小
        """
        cm = self.get_confusion_matrix()
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self):
        """打印评估摘要"""
        metrics = self.compute_metrics()
        
        print("\n" + "="*60)
        print("分类任务评估摘要")
        print("="*60)
        
        # 整体指标
        print(f"\n整体指标:")
        print(f"  - 准确率: {metrics['accuracy']:.4f}")
        print(f"  - F1分数 (macro): {metrics['f1_macro']:.4f}")
        print(f"  - F1分数 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"  - Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        
        if 'auc' in metrics:
            print(f"  - AUC: {metrics['auc']:.4f}")
        elif 'auc_macro' in metrics:
            print(f"  - AUC (macro): {metrics['auc_macro']:.4f}")
        
        # 每个类别的指标
        print(f"\n各类别指标:")
        for i, class_name in enumerate(self.class_names):
            print(f"\n  {class_name}:")
            print(f"    - 精确率: {metrics[f'{class_name}_precision']:.4f}")
            print(f"    - 召回率: {metrics[f'{class_name}_recall']:.4f}")
            print(f"    - F1分数: {metrics[f'{class_name}_f1']:.4f}")
            print(f"    - 样本数: {metrics[f'{class_name}_support']}")
        
        # 混淆矩阵
        print(f"\n混淆矩阵:")
        cm = self.get_confusion_matrix()
        print(cm)
        
        # 详细分类报告
        print(f"\n详细分类报告:")
        print(self.get_classification_report())


def compute_batch_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    计算单个批次的基础指标（用于训练过程中的监控）
    
    Args:
        preds: 预测结果 [batch_size, num_classes] 或 [batch_size]
        labels: 真实标签 [batch_size]
    
    Returns:
        包含准确率和F1分数的字典
    """
    # 如果是logits，转换为预测类别
    if preds.dim() > 1:
        preds = torch.argmax(preds, dim=1)
    
    # 转换为numpy
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # 计算指标
    acc = accuracy_score(labels_np, preds_np)
    f1 = f1_score(labels_np, preds_np, average='weighted')
    
    return {
        'accuracy': acc,
        'f1_score': f1
    }


# 情绪分类专用指标
class EmotionMetrics(ClassificationMetrics):
    """EEG情绪分类专用指标"""
    
    def __init__(self, dataset_name: str = 'DEAP'):
        """
        初始化
        
        Args:
            dataset_name: 数据集名称 ('DEAP' 或 'SEED')
        """
        if dataset_name == 'DEAP':
            super().__init__(
                num_classes=2,
                class_names=['Negative', 'Positive']
            )
        elif dataset_name == 'SEED':
            super().__init__(
                num_classes=3,
                class_names=['Negative', 'Neutral', 'Positive']
            )
        else:
            raise ValueError(f"未知的数据集: {dataset_name}")
        
        self.dataset_name = dataset_name
    
    def print_emotion_summary(self):
        """打印情绪分类专用摘要"""
        print(f"\n{self.dataset_name} 情绪分类评估结果")
        self.print_summary()
        
        # 额外的情绪分类相关分析
        cm = self.get_confusion_matrix()
        
        if self.dataset_name == 'DEAP':
            # 二分类特殊分析
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            print(f"\n二分类特殊指标:")
            print(f"  - 敏感度 (Sensitivity): {sensitivity:.4f}")
            print(f"  - 特异度 (Specificity): {specificity:.4f}")


# 使用示例
if __name__ == "__main__":
    # 模拟一些预测结果
    num_samples = 100
    num_classes = 2
    
    # 创建评估器
    metrics = EmotionMetrics('DEAP')
    
    # 模拟多个批次的预测
    for _ in range(10):
        batch_size = 10
        # 模拟预测和标签
        preds = torch.randint(0, num_classes, (batch_size,))
        labels = torch.randint(0, num_classes, (batch_size,))
        probs = torch.softmax(torch.randn(batch_size, num_classes), dim=1)
        
        # 更新指标
        metrics.update(preds, labels, probs)
    
    # 打印评估结果
    metrics.print_emotion_summary()
    
    # 绘制混淆矩阵
    metrics.plot_confusion_matrix()
