import os
import numpy as np
import torch


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True, mode='min'):
        """
        早停机制，支持最小化和最大化两种模式

        Args:
            accelerator: 加速器对象
            patience: 耐心值，多少个epoch没有改善就停止
            verbose: 是否打印详细信息
            delta: 最小改善量
            save_mode: 是否保存模型
            mode: 'min'表示监控指标越小越好（如损失），'max'表示越大越好（如准确率）
        """
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_mode = save_mode
        self.mode = mode

        # 根据模式初始化
        if mode == 'min':
            self.val_metric_best = np.inf
        elif mode == 'max':
            self.val_metric_best = -np.inf
        else:
            raise ValueError(f"mode {mode} is not supported")

    def __call__(self, val_metric, model, path):
        """
        Args:
            val_metric: 验证指标（损失或准确率）
            model: 模型
            path: 保存路径
        """
        if self.mode == 'min':
            score = -val_metric
            is_improvement = val_metric < self.val_metric_best - self.delta
        else:  # mode == 'max'
            score = val_metric
            is_improvement = val_metric > self.val_metric_best + self.delta

        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_metric, model, path)
            else:
                self.val_metric_best = val_metric
        elif not is_improvement:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_metric, model, path)
            else:
                self.val_metric_best = val_metric
            self.counter = 0

    def save_checkpoint(self, val_metric, model, path):
        """保存模型检查点"""
        if self.verbose:
            direction = 'decreased' if self.mode == 'min' else 'increased'
            message = f'Validation metric {direction} ({self.val_metric_best:.6f} --> {val_metric:.6f}). Saving model ...'
            if self.accelerator is not None:
                self.accelerator.print(message)
            else:
                print(message)

        # 真正把模型权重落盘（之前缺这一行，导致 train() 末尾 torch.load 必崩）
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_metric_best = val_metric
