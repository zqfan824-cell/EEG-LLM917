"""
分类任务专用的可视化和辅助工具
包括训练曲线、特征可视化、结果分析等
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional
import pandas as pd
from datetime import datetime


class TrainingVisualizer:
    """训练过程可视化工具"""
    
    def __init__(self, save_dir: str = './visualizations'):
        """
        初始化
        
        Args:
            save_dir: 图像保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 存储训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rate': []
        }
        self.epochs = []
    
    def update(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float = None):
        """
        更新训练历史
        
        Args:
            epoch: 当前epoch
            train_metrics: 训练指标字典
            val_metrics: 验证指标字典
            lr: 当前学习率
        """
        self.epochs.append(epoch)
        
        # 更新训练指标
        self.history['train_loss'].append(train_metrics.get('loss', 0))
        self.history['train_acc'].append(train_metrics.get('accuracy', 0))
        self.history['train_f1'].append(train_metrics.get('f1_score', 0))
        
        # 更新验证指标
        self.history['val_loss'].append(val_metrics.get('loss', 0))
        self.history['val_acc'].append(val_metrics.get('accuracy', 0))
        self.history['val_f1'].append(val_metrics.get('f1_score', 0))
        
        # 学习率
        if lr is not None:
            self.history['learning_rate'].append(lr)
    
    def plot_training_curves(self, show: bool = True):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(self.epochs, self.history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(self.epochs, self.history['train_acc'], 'b-', label='Train')
        axes[0, 1].plot(self.epochs, self.history['val_acc'], 'r-', label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1分数曲线
        axes[1, 0].plot(self.epochs, self.history['train_f1'], 'b-', label='Train')
        axes[1, 0].plot(self.epochs, self.history['val_f1'], 'r-', label='Validation')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 学习率曲线
        if self.history['learning_rate']:
            axes[1, 1].plot(self.epochs, self.history['learning_rate'], 'g-')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].grid(True)
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
        
        return save_path
    
    def save_history(self, filename: str = 'training_history.csv'):
        """保存训练历史到CSV"""
        df = pd.DataFrame(self.history)
        df['epoch'] = self.epochs
        save_path = os.path.join(self.save_dir, filename)
        df.to_csv(save_path, index=False)
        return save_path


class FeatureVisualizer:
    """特征可视化工具"""
    
    def __init__(self, save_dir: str = './visualizations'):
        """
        初始化
        
        Args:
            save_dir: 图像保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def visualize_embeddings(self, 
                           features: np.ndarray, 
                           labels: np.ndarray,
                           method: str = 'tsne',
                           class_names: List[str] = None,
                           title: str = 'Feature Embeddings',
                           save_name: str = 'embeddings.png'):
        """
        可视化特征嵌入
        
        Args:
            features: 特征数组 [n_samples, n_features]
            labels: 标签数组 [n_samples]
            method: 降维方法 ('tsne' 或 'pca')
            class_names: 类别名称
            title: 图像标题
            save_name: 保存文件名
        """
        # 降维
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(features)
        elif method == 'pca':
            reducer = PCA(n_components=2)
            embeddings_2d = reducer.fit_transform(features)
        else:
            raise ValueError(f"未知的降维方法: {method}")
        
        # 绘图
        plt.figure(figsize=(10, 8))
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # 为每个类别绘制散点
        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = class_names[label] if class_names else f'Class {label}'
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=label_name,
                alpha=0.6,
                s=50
            )
        
        plt.title(f'{title} ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        return save_path
    
    def plot_class_distribution(self, 
                              labels: np.ndarray,
                              class_names: List[str] = None,
                              title: str = 'Class Distribution'):
        """
        绘制类别分布
        
        Args:
            labels: 标签数组
            class_names: 类别名称
            title: 图像标题
        """
        # 统计每个类别的数量
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # 准备类别名称
        if class_names is None:
            class_names = [f'Class {i}' for i in unique_labels]
        
        # 绘制条形图
        plt.figure(figsize=(8, 6))
        bars = plt.bar(class_names, counts, alpha=0.8)
        
        # 在条形上添加数值
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', va='bottom')
        
        plt.title(title)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.grid(True, axis='y', alpha=0.3)
        
        # 保存图像
        save_path = os.path.join(self.save_dir, 'class_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        return save_path


class EEGVisualizer:
    """EEG数据专用可视化工具"""
    
    def __init__(self, save_dir: str = './visualizations'):
        """
        初始化
        
        Args:
            save_dir: 图像保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_eeg_samples(self,
                        eeg_data: np.ndarray,
                        labels: np.ndarray,
                        sampling_rate: int = 128,
                        num_samples: int = 4,
                        class_names: List[str] = None):
        """
        绘制EEG样本
        
        Args:
            eeg_data: EEG数据 [n_samples, n_timepoints, n_channels]
            labels: 标签数组 [n_samples]
            sampling_rate: 采样率
            num_samples: 每个类别显示的样本数
            class_names: 类别名称
        """
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        
        fig, axes = plt.subplots(num_classes, num_samples, 
                               figsize=(4*num_samples, 3*num_classes))
        
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        # 时间轴
        time_points = eeg_data.shape[1]
        time_axis = np.arange(time_points) / sampling_rate
        
        # 为每个类别绘制样本
        for i, label in enumerate(unique_labels):
            # 获取该类别的样本
            class_indices = np.where(labels == label)[0]
            sample_indices = np.random.choice(
                class_indices, 
                min(num_samples, len(class_indices)), 
                replace=False
            )
            
            # 绘制每个样本
            for j, idx in enumerate(sample_indices):
                ax = axes[i, j]
                
                # 绘制所有通道（只显示前5个通道避免太密集）
                num_channels_to_show = min(5, eeg_data.shape[2])
                for ch in range(num_channels_to_show):
                    signal = eeg_data[idx, :, ch]
                    # 标准化以便更好地显示
                    signal = (signal - signal.mean()) / signal.std()
                    ax.plot(time_axis, signal + ch*2, alpha=0.7)
                
                # 设置标题和标签
                class_name = class_names[label] if class_names else f'Class {label}'
                ax.set_title(f'{class_name} - Sample {j+1}')
                if j == 0:
                    ax.set_ylabel('Channels')
                if i == num_classes - 1:
                    ax.set_xlabel('Time (s)')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.save_dir, 'eeg_samples.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        return save_path
    
    def plot_channel_importance(self,
                              importance_scores: np.ndarray,
                              channel_names: List[str] = None,
                              top_k: int = 10):
        """
        绘制通道重要性
        
        Args:
            importance_scores: 各通道的重要性分数 [n_channels]
            channel_names: 通道名称
            top_k: 显示前k个重要的通道
        """
        n_channels = len(importance_scores)
        
        # 如果没有通道名称，使用默认名称
        if channel_names is None:
            channel_names = [f'Ch{i+1}' for i in range(n_channels)]
        
        # 排序并获取top k
        sorted_indices = np.argsort(importance_scores)[::-1][:top_k]
        top_scores = importance_scores[sorted_indices]
        top_names = [channel_names[i] for i in sorted_indices]
        
        # 绘制条形图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(top_k), top_scores, alpha=0.8)
        plt.xticks(range(top_k), top_names, rotation=45, ha='right')
        
        # 在条形上添加数值
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.title(f'Top {top_k} Most Important Channels')
        plt.xlabel('Channel')
        plt.ylabel('Importance Score')
        plt.grid(True, axis='y', alpha=0.3)
        
        # 保存图像
        save_path = os.path.join(self.save_dir, 'channel_importance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        return save_path


def create_experiment_summary(experiment_name: str,
                            config: Dict,
                            results: Dict,
                            save_dir: str = './results'):
    """
    创建实验总结报告
    
    Args:
        experiment_name: 实验名称
        config: 配置字典
        results: 结果字典
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建Markdown报告
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# 实验报告: {experiment_name}

## 实验信息
- **时间**: {timestamp}
- **数据集**: {config.get('dataset', 'Unknown')}
- **模型**: {config.get('model', 'TimeLLM')}

## 配置参数
```
"""
    
    # 添加配置参数
    for key, value in config.items():
        report += f"{key}: {value}\n"
    
    report += """```

## 实验结果

### 整体性能
"""
    
    # 添加整体指标
    for metric, value in results.items():
        if isinstance(value, float):
            report += f"- **{metric}**: {value:.4f}\n"
        else:
            report += f"- **{metric}**: {value}\n"
    
    report += """

### 详细结果
请查看生成的可视化图像和CSV文件获取更多细节。

## 文件列表
- `training_curves.png`: 训练曲线
- `confusion_matrix.png`: 混淆矩阵
- `training_history.csv`: 训练历史数据
- `final_metrics.json`: 最终评估指标
"""
    
    # 保存报告
    report_path = os.path.join(save_dir, f'{experiment_name}_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"实验报告已保存到: {report_path}")
    
    return report_path


# 使用示例
if __name__ == "__main__":
    # 创建训练可视化器
    visualizer = TrainingVisualizer()
    
    # 模拟训练过程
    for epoch in range(20):
        train_metrics = {
            'loss': 1.0 - epoch * 0.04 + np.random.random() * 0.1,
            'accuracy': 0.5 + epoch * 0.02 + np.random.random() * 0.05,
            'f1_score': 0.45 + epoch * 0.02 + np.random.random() * 0.05
        }
        
        val_metrics = {
            'loss': 1.1 - epoch * 0.035 + np.random.random() * 0.15,
            'accuracy': 0.48 + epoch * 0.018 + np.random.random() * 0.07,
            'f1_score': 0.43 + epoch * 0.018 + np.random.random() * 0.07
        }
        
        lr = 0.001 * (0.95 ** epoch)
        
        visualizer.update(epoch + 1, train_metrics, val_metrics, lr)
    
    # 绘制训练曲线
    visualizer.plot_training_curves()
    visualizer.save_history()
