"""
支持 VQ 和重建损失的 EEG 分类实验类
集成 NeuroLM 的重建机制，增强 EEGLLM 的域对抗学习效果
"""

import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from models.EEGLLM_VQ import EEGLLM_VQ
from utils.reconstruction_losses import AdaptiveLossWeighter
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics_classification import compute_batch_metrics
from utils.loss_classification import create_loss_function

warnings.filterwarnings('ignore')


class Exp_Classification_VQ:
    """支持VQ和重建损失的分类实验类"""
    
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
        # 自适应损失权重调整器
        self.adaptive_weighter = AdaptiveLossWeighter(num_losses=5).to(self.device)
        
        print(f"[Exp_Classification_VQ] 初始化完成")
        print(f"  - 设备: {self.device}")
        print(f"  - 模型: {self.args.model}")
        print(f"  - VQ启用: {self.args.enable_vq}")
        print(f"  - 重建损失启用: {self.args.enable_reconstruction}")
        print(f"  - 域对抗启用: {self.args.enable_adversarial}")
    
    def _acquire_device(self):
        """获取设备"""
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print(f'Use GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _build_model(self):
        """构建模型"""
        model = EEGLLM_VQ(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model
    
    def _get_data(self, flag):
        """获取数据"""
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        """选择优化器"""
        # 为不同组件设置不同的学习率
        param_groups = []
        
        # 基础模型参数
        base_params = []
        vq_params = []
        reconstruction_params = []
        adaptive_params = []
        
        for name, param in self.model.named_parameters():
            if 'quantizer' in name:
                vq_params.append(param)
            elif 'decoder' in name or 'reconstruction' in name:
                reconstruction_params.append(param)
            else:
                base_params.append(param)
        
        # 自适应权重参数
        adaptive_params = list(self.adaptive_weighter.parameters())
        
        # 设置不同的学习率
        param_groups.append({'params': base_params, 'lr': self.args.learning_rate})
        
        if vq_params:
            param_groups.append({'params': vq_params, 'lr': self.args.learning_rate * 0.5})
        
        if reconstruction_params:
            param_groups.append({'params': reconstruction_params, 'lr': self.args.learning_rate * 0.8})
        
        if adaptive_params:
            param_groups.append({'params': adaptive_params, 'lr': self.args.learning_rate * 2.0})
        
        model_optim = optim.Adam(param_groups)
        return model_optim
    
    def _select_criterion(self):
        """选择损失函数"""
        return create_loss_function(self.args.loss, self.args.n_class)
    
    def _compute_alpha(self, epoch):
        """计算域对抗学习的alpha值"""
        if not self.args.enable_adversarial:
            return 0.0
        
        if self.args.alpha_schedule == 'sigmoid':
            progress = epoch / self.args.train_epochs
            alpha = self.args.max_alpha * (2 / (1 + np.exp(-10 * progress)) - 1)
        elif self.args.alpha_schedule == 'linear':
            alpha = self.args.max_alpha * epoch / self.args.train_epochs
        else:  # constant
            alpha = self.args.max_alpha
        
        return alpha
    
    def vali(self, vali_data, vali_loader, criterion):
        """验证函数"""
        total_loss = []
        preds = []
        trues = []
        
        self.model.eval()
        self.adaptive_weighter.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # 前向传播
                outputs, loss_dict = self.model(
                    batch_x, batch_x_mark, 
                    alpha=0.0,  # 验证时不使用域对抗
                    return_reconstruction_loss=True
                )
                
                # 分类损失
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                classification_loss = criterion(outputs, batch_y)
                
                # 组合所有损失（包含新的对比学习损失）
                losses = [
                    classification_loss,
                    loss_dict.get('domain_loss', 0) * self.args.domain_weight,
                    loss_dict.get('vq_loss', 0),
                    loss_dict.get('reconstruction_loss', 0) * self.args.reconstruction_weight,
                    loss_dict.get('contrastive_loss', 0) * getattr(self.args, 'contrastive_weight', 0.1)
                ]
                
                total_weighted_loss, _ = self.adaptive_weighter(losses)
                total_loss.append(total_weighted_loss.item())
                
                preds.append(pred)
                trues.append(true)
        
        total_loss = np.average(total_loss)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # 计算准确率
        accuracy = (preds.argmax(axis=1) == trues).mean()
        
        self.model.train()
        self.adaptive_weighter.train()
        
        return total_loss, accuracy
    
    def train(self, setting):
        """训练函数"""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        # 训练日志
        train_losses = []
        vali_losses = []
        vali_accuracies = []
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            self.adaptive_weighter.train()
            epoch_time = time.time()
            
            # 计算当前epoch的alpha值
            alpha = self._compute_alpha(epoch)
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, loss_dict = self.model(
                            batch_x, batch_x_mark,
                            alpha=alpha,
                            return_reconstruction_loss=True
                        )
                        
                        # 分类损失
                        classification_loss = criterion(outputs, batch_y)
                        
                        # 组合所有损失（包含新的对比学习损失）
                        losses = [
                            classification_loss,
                            loss_dict.get('domain_loss', 0) * self.args.domain_weight,
                            loss_dict.get('vq_loss', 0),
                            loss_dict.get('reconstruction_loss', 0) * self.args.reconstruction_weight,
                            loss_dict.get('contrastive_loss', 0) * getattr(self.args, 'contrastive_weight', 0.1)
                        ]
                        
                        loss, weights = self.adaptive_weighter(losses)
                    
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs, loss_dict = self.model(
                        batch_x, batch_x_mark,
                        alpha=alpha,
                        return_reconstruction_loss=True
                    )
                    
                    # 分类损失
                    classification_loss = criterion(outputs, batch_y)
                    
                    # 组合所有损失（包含新的对比学习损失）
                    losses = [
                        classification_loss,
                        loss_dict.get('domain_loss', 0) * self.args.domain_weight,
                        loss_dict.get('vq_loss', 0),
                        loss_dict.get('reconstruction_loss', 0) * self.args.reconstruction_weight,
                        loss_dict.get('contrastive_loss', 0) * getattr(self.args, 'contrastive_weight', 0.1)
                    ]
                    
                    loss, weights = self.adaptive_weighter(losses)
                    
                    loss.backward()
                    model_optim.step()
                
                train_loss.append(loss.item())
                
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    print(f"\t  - Classification: {classification_loss.item():.7f}")
                    print(f"\t  - Domain: {loss_dict.get('domain_loss', 0):.7f}")
                    print(f"\t  - VQ: {loss_dict.get('vq_loss', 0):.7f}")
                    print(f"\t  - Reconstruction: {loss_dict.get('reconstruction_loss', 0):.7f}")
                    print(f"\t  - Contrastive: {loss_dict.get('contrastive_loss', 0):.7f}")
                    print(f"\t  - Alpha: {alpha:.4f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()
            
            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss, vali_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)
            
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                  f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            print(f"Vali Accuracy: {vali_accuracy:.4f} Test Accuracy: {test_accuracy:.4f}")
            
            # 记录训练历史
            train_losses.append(train_loss)
            vali_losses.append(vali_loss)
            vali_accuracies.append(vali_accuracy)
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            # 简化的学习率调整（暂时跳过）
            # adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, self.args)
        
        # 保存训练历史
        history = {
            'train_losses': train_losses,
            'vali_losses': vali_losses,
            'vali_accuracies': vali_accuracies
        }
        
        import json
        with open(os.path.join(path, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model
    
    def test(self, setting, test=0):
        """测试函数"""
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark)
                
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        print('test shape:', preds.shape, trues.shape)
        
        # 计算指标
        accuracy, precision, recall, f1 = metric(preds, trues)
        print(f'Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        
        # 保存结果
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        
        return accuracy, precision, recall, f1
