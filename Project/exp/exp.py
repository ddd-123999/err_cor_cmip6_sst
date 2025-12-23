import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import random

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, Type, Tuple

from tqdm import tqdm

from Project.data.data_loader import DatasetPred, DatasetTrain, DatasetTest, DatasetVal
from Project.models.mamba_unet import MambaUNet
from Project.models.swin_unet import SwinUnet
from Project.models.swin_unet_new import SwinUnet_new
from Project.models.mamba_unet_new import MambaUNet_new
from Project.models.conv_lstm import ConvLSTMSIC
from Project.models.ConvLSTM_new import ConvLSTM_new
from Project.models.Unet import UNet
from Project.models.unet_convlstm import UNet_LSTM
from Project.models.unet_nores import UNet_nores
from Project.utils.metric import metric
from Project.utils.tools import EarlyStopping, adjust_learning_rate

class Exp:
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.model = self._build_model()
        # 加载 mask (0/1)
        mask_np = self.args.mask_data
        # 转为 tensor, float32, 放入 device
        self.mask_tensor = torch.from_numpy(mask_np).float().to(self.device)
        # 扩展维度以适配广播 (1, 1, H, W)
        self.mask_tensor = self.mask_tensor.unsqueeze(0).unsqueeze(0)

        # 从 args 中获取所有四种参数和新的开关
        if self.args.use_normalized or self.args.use_standardized:
            # Min-Max 参数
            self.data_min = getattr(args, 'data_min', None)
            self.data_max = getattr(args, 'data_max', None)
            # Z-score 参数
            self.data_mean = getattr(args, 'data_mean', None)
            self.data_std = getattr(args, 'data_std', None)
        else:
            self.data_min = None;
            self.data_max = None
            self.data_mean = None;
            self.data_std = None

    def _build_model(self) -> nn.Module:
        """根据参数构建模型"""
        model_dict: Dict[str, Type[nn.Module]] = {
            'UNet_new': UNet,
            'ConvLSTM': ConvLSTMSIC,
            'ConvLSTM_new': ConvLSTM_new,
            'UNet': UNet_nores,
            'UNet_LSTM': UNet_LSTM,
            'SwinUNet': SwinUnet,
            'MambaUNet': MambaUNet,# ✅ 新增
            'SwinUNet_new': SwinUnet_new, # ✅ 新增
            'MambaUNet_new': MambaUNet_new # ✅ 新增
        }

        if self.args.model not in model_dict:
            raise ValueError(f"模型 '{self.args.model}' 未定义！")

        model = model_dict[self.args.model](self.args).to(self.device)
        return model

    def _get_data(self, flag: str) -> Tuple[torch.utils.data.Dataset, DataLoader]:
        """根据 flag 获取数据集和数据加载器"""
        dataset_dict = {
            'train': DatasetTrain,
            'val': DatasetVal,
            'test': DatasetTest,
            'pred': DatasetPred
        }
        if flag not in dataset_dict:
            raise ValueError(f"flag '{flag}' 无效！")

        if flag == 'train':
            shuffle_flag = True
            drop_last = True
        else:
            shuffle_flag = False
            drop_last = False

        root_path_target = self.args.root_path_target
        root_path_correction = self.args.root_path_correction

        data_path_target = self.args.data_path_target
        # data_path_correction = self.args.data_path_correction

        if flag == 'train':
            data_path_correction = self.args.data_path_train
            data_path_anomaly = self.args.data_path_anomaly_train
        elif flag == 'val':
            data_path_correction = self.args.data_path_val
            data_path_anomaly = self.args.data_path_anomaly_val
        elif flag == 'test':
            data_path_correction = self.args.data_path_test
            data_path_anomaly = self.args.data_path_anomaly_test
        elif flag == 'pred':
            data_path_correction = self.args.data_path_train
            data_path_anomaly = self.args.data_path_anomaly_train
        else:
            raise ValueError(f"flag '{flag}' 无效！")

        root_path_anomaly = self.args.root_path_anomaly
        #data_path_anomaly = self.args.data_path_anomaly
        add_anomaly = self.args.add_anomaly

        if flag == 'pred':
            root_path_anomaly_in_pre = self.args.root_path_anomaly_in_pre
            data_path_anomaly_in_pre = self.args.data_path_anomaly_in_pre
            root_path_pre = self.args.root_path_pre
            data_path_pre = self.args.data_path_pre

            data_set = dataset_dict[flag](
                root_path_pre=root_path_pre,
                data_path_pre=data_path_pre,
                root_path_correction=root_path_correction,
                data_path_correction=data_path_correction,
                flag=flag,
                seq_len=self.args.seq_len,
                correction_len=self.args.correction_len,
                root_path_anomaly=root_path_anomaly,
                data_path_anomaly=data_path_anomaly,
                root_path_anomaly_in_pre=root_path_anomaly_in_pre,
                data_path_anomaly_in_pre=data_path_anomaly_in_pre,
                add_anomaly=add_anomaly,
                use_normalized = self.args.use_normalized, # 新增参数
                use_standardized = self.args.use_standardized  # ✅ 新增参数
            )
        else:
            data_set = dataset_dict[flag](
                root_path_target=root_path_target,
                root_path_correction=root_path_correction,
                data_path_target=data_path_target,
                data_path_correction=data_path_correction,
                flag=flag,
                seq_len=self.args.seq_len,
                correction_len=self.args.correction_len,
                root_path_anomaly=root_path_anomaly,
                data_path_anomaly=data_path_anomaly,
                add_anomaly=add_anomaly,
                use_normalized = self.args.use_normalized, # 新增参数
                use_standardized = self.args.use_standardized  # ✅ 新增参数
            )
        print(f"{flag} 数据集大小: {len(data_set)}")
        generator = torch.Generator()
        generator.manual_seed(42)  # 使用固定的种子
        data_loader = DataLoader(
            data_set,
            batch_size=self.args.batch_size,
            shuffle=shuffle_flag,
            drop_last=drop_last,
        )
        return data_set, data_loader

    def _select_optimizer(self) -> optim.Optimizer:
        """选择优化器"""
        if self.args.model == 'SwinUNet_new':
            return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.999), weight_decay=0.00001)
        elif self.args.model in ['MambaUNet_new', 'MambaUNet']:
            return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4) #weight_decay=1e-4
        elif self.args.model == 'HybridModel':  # ← 只添加这一段
            return optim.AdamW(self.model.parameters(),
                               lr=1e-6,  # 初始化极小学习率
                               betas=(0.9, 0.999),
                               weight_decay=1e-5)
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)


    # def _select_criterion(self) -> nn.Module:
    #     """根据传入的损失函数名称选择损失函数"""
    #     if self.args.loss_name == 'mse':
    #         return nn.MSELoss()
    #     else:
    #         raise ValueError(f"损失函数 '{self.args.loss_name}' 未定义！可选项: ['mse']")
    def _select_criterion(self) -> nn.Module:
        """选择损失函数"""
        if self.args.loss_name == 'mse':
            return MaskedMSELoss()
        else:
            raise ValueError(f"损失函数 '{self.args.loss_name}' 未定义！")

    def _move_to_device_train(self, batch_x, batch_y, batch_anomaly):
        """将数据移动到设备并转换为 float32"""
        if isinstance(batch_anomaly, list):
            batch_anomaly = torch.tensor([0], device=self.device, dtype=torch.float32)
        else:
            batch_anomaly = batch_anomaly.to(self.device).float()

        return batch_x.to(self.device).float(), batch_y.to(self.device).float(), batch_anomaly

    def _move_to_device_pred(self, batch_x, batch_anomaly):
        """将数据移动到设备并转换为 float32"""
        if isinstance(batch_anomaly, list):
            batch_anomaly = torch.tensor([0], device=self.device, dtype=torch.float32)
        else:
            batch_anomaly = batch_anomaly.to(self.device).float()

        return batch_x.to(self.device).float(), batch_anomaly

    def _adjust_warmup_lr(self, optimizer, current_step, warmup_steps):
        """调整学习率预热"""
        initial_lr = self.args.learning_rate * 0.1
        new_lr = initial_lr + (self.args.learning_rate - initial_lr) * (current_step / warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        return new_lr

    # ========================================模型验证========================================
    def vali(self, vali_loader: DataLoader, criterion: nn.Module, flag, epoch) -> float:
        """验证模型性能"""
        self.model.eval()
        total_loss = []
        vali_loader = tqdm(vali_loader, file=sys.stdout)
        total_rmse = []  # ✅ 改为列表收集所有RMSE

        with torch.no_grad():
            for batch_x, batch_y, batch_anomaly in vali_loader:
                batch_x, batch_y, batch_anomaly = self._move_to_device_train(batch_x, batch_y, batch_anomaly)
                pred = self.model(batch_x, batch_anomaly)
                # loss = criterion(pred, batch_y)
                loss = criterion(pred, batch_y, self.mask_tensor)
                total_loss.append(loss.item())

                total_rmse.append(self.rmse(pred, batch_y))  # ✅ 收集RMSE
                if flag == 'vali':
                    vali_loader.desc = "[vali epoch {}] loss: {:.5f} rmse: {:.4f}".format(epoch, sum(total_loss) / len(
                        total_loss), sum(total_rmse) / len(
                        total_rmse))
                elif flag == 'test':
                    vali_loader.desc = "[test epoch {}] loss: {:.5f} rmse: {:.4f}".format(epoch, sum(total_loss) / len(
                        total_loss), sum(total_rmse) / len(
                        total_rmse))
                else:
                    sys.exit(1)
                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss)
                    sys.exit(1)

        avg_loss = torch.mean(torch.tensor(total_loss)).item()
        avg_rmse = sum(total_rmse) / len(total_rmse)  # ✅ 计算平均RMSE
        self.model.train()
        return avg_loss, avg_rmse  # ✅ 返回损失和RMS

    # ========================================模型训练========================================
    def train(self, setting: str):
        """训练模型"""
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')
        _, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        # ✅ 获取配置文件路径
        config_file = os.path.join(path, 'args_config.txt')

        # ✅ 在训练开始前写入表头
        with open(config_file, 'a', encoding='utf-8') as f:
            f.write("\n\n=== 训练过程记录 ===\n")
            f.write("Epoch | Train_Loss | Train_RMSE | Val_Loss | Val_RMSE | Test_Loss | Test_RMSE | Learning_Rate\n")
            f.write("-" * 100 + "\n")

        early_stopping = EarlyStopping(patience=self.args.patience, delta=self.args.delta)  # 针对MSE损失函数
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        adjust = False
        current_step = 0
        warmup_steps = len(train_loader) * self.args.warmup_epochs
        for param_group in model_optim.param_groups:
            param_group['lr'] = self.args.learning_rate * 0.1

        train_start = time.time()

        for epoch in range(self.args.train_epochs):
            train_loader = tqdm(train_loader, file=sys.stdout)
            self.model.train()
            train_loss = []
            train_rmse_list = []  # ✅ 重命名避免混淆 原来： rmse = []

            for step, (batch_x, batch_y, batch_anomaly) in enumerate(train_loader, 1):
                batch_x, batch_y, batch_anomaly = self._move_to_device_train(batch_x, batch_y, batch_anomaly)
                pred = self.model(batch_x, batch_anomaly)
                # loss = criterion(pred, batch_y)
                loss = criterion(pred, batch_y, self.mask_tensor)
                loss.backward()
                # =======================================================
                # 💡 仅针对 MambaUNet_new 进行梯度裁剪
                # =======================================================
                if self.args.model == 'MambaUNet_new':
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # =======================================================
                model_optim.step()
                train_loss.append(loss.item())

                current_step += 1
                if current_step <= warmup_steps:
                    self._adjust_warmup_lr(model_optim, current_step, warmup_steps)
                    adjust = False
                    if current_step == warmup_steps:
                        adjust = True
                else:
                    adjust = True

                train_rmse_list.append(self.rmse(pred, batch_y))  # ✅ 记录训练RMSE
                current_lr = model_optim.param_groups[0]['lr']
                train_loader.desc = "[train epoch {}] loss: {:.5f} rmse: {:.4f} Learning_rate {:.7f}".format(epoch,
                                                                                                             sum(train_loss) / len(
                                                                                                                 train_loss),
                                                                                                             sum(train_rmse_list) / len(
                                                                                                                 train_rmse_list),
                                                                                                             current_lr)

                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss)
                    sys.exit(1)

            if adjust:
                adjust_learning_rate(model_optim, epoch + 2, self.args)

            # ✅ 记录训练结果
            avg_train_loss = sum(train_loss) / len(train_loss)
            avg_train_rmse = sum(train_rmse_list) / len(train_rmse_list)
            current_lr = model_optim.param_groups[0]['lr']

            # ✅ 修改验证函数以返回RMSE
            vali_loss, vali_rmse = self.vali(vali_loader, criterion, 'vali', epoch)
            test_loss, test_rmse = self.vali(test_loader, criterion, 'test', epoch)

            # ✅ 简洁记录到配置文件（包含RMSE）
            with open(config_file, 'a', encoding='utf-8') as f:
                f.write(f"{epoch:5d} | "
                        f"{avg_train_loss:10.5f} | "
                        f"{avg_train_rmse:10.5f} | "
                        f"{vali_loss:8.5f} | "
                        f"{vali_rmse:8.5f} | "  # ✅ 新增验证RMSE
                        f"{test_loss:9.5f} | "
                        f"{test_rmse:9.5f} | "  # ✅ 新增测试RMSE
                        f"{current_lr:12.8f}\n")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                break

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device, weights_only=True))

        train_end = time.time()
        use_time = train_end - train_start
        print('Train use time: {:.2f}s'.format(use_time))
        return use_time, test_loader

    # ========================================模型测试========================================
    def test(self, setting: str, test_loader) -> None:
        """测试模型性能"""
        self.model.eval()
        preds, trues = self._run_test(test_loader, setting)
        rmse, mae, mse, nse, pcc, ssim = self._calculate_metrics(preds, trues)
        self._save_results(setting, rmse, mae, mse, nse, pcc, ssim)

    def _run_test(self, test_loader: DataLoader, setting):
        folder_path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(folder_path, exist_ok=True)
        save_path_preds = os.path.join(self.args.checkpoints, setting, 'test_corrections.npy')
        save_path_trues = os.path.join(self.args.checkpoints, setting, 'test_trues.npy')

        preds = []
        trues = []
        rmse = []
        test_loader = tqdm(test_loader, file=sys.stdout)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_anomaly) in enumerate(test_loader):
                batch_x, batch_y, batch_anomaly = self._move_to_device_train(batch_x, batch_y, batch_anomaly)
                pred = self.model(batch_x, batch_anomaly)

                rmse.append(self.rmse(pred, batch_y))
                pred = pred.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                test_loader.desc = "[test] rmse: {:.4f}".format(sum(rmse) / len(rmse))

                pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
                true = true.reshape(-1, true.shape[-2], true.shape[-1])

                # 新增：如果使用归一化或标准化数据，进行反向转换
                if self.args.use_normalized or self.args.use_standardized:  # ✅ 修正判断条件
                    pred = self._denormalize(pred)
                    true = self._denormalize(true)

                preds.append(pred)
                trues.append(true)

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
        np.save(save_path_preds, preds)
        np.save(save_path_trues, trues)

        return preds, trues

    def _denormalize(self, normalized_data):
        """反归一化/反标准化: 根据 args.use_normalized/use_standardized 切换模式"""

        # 1. 优先检查 Z-score 反标准化
        if self.args.use_standardized and self.data_mean is not None and self.data_std is not None:
            # print("▶️ 使用 Z-score 反标准化 (X = Z * σ + μ)")
            # Z-score 反标准化: X = Z * std + mean
            # numpy/torch 的广播机制会自动处理 (T, H, W) 与 (H, W) 的操作
            return normalized_data * self.data_std + self.data_mean

        # 2. 其次检查 Min-Max 反归一化
        elif self.args.use_normalized and self.data_min is not None and self.data_max is not None:
            # print("▶️ 使用 Min-Max 反归一化 (X = Z * (Max - Min) + Min)")
            # Min-Max 反归一化: X = Z * (Max - Min) + Min
            return normalized_data * (self.data_max - self.data_min) + self.data_min

        # 3. 默认返回原始数据
        else:
            return normalized_data

    def _calculate_metrics(self, preds, trues):
        """计算评估指标"""
        rmse, mae, mse, nse, pcc, ssim = metric(self.args.mask_data, preds, trues)

        print(f'RMSE: {rmse:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}')
        print(f'NSE: {nse:.4f}, PCC: {pcc:.4f}, SSIM: {ssim:.4f}')

        return rmse, mae, mse, nse, pcc, ssim

    def _save_results(self, setting, rmse, mae, mse, nse, pcc, ssim):
        """保存测试结果"""
        folder_path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(folder_path, exist_ok=True)

        results = {
            'rmse': rmse, 'mae': mae, 'mse': mse,
            'nse': nse, 'pcc': pcc, 'ssim': ssim
        }

        save_file = os.path.join(folder_path, 'test_criteria.npz')
        np.savez(save_file, results=results)

    # ========================================模型预测========================================
    def predict(self, setting: str) -> float:
        """加载模型并进行预测"""
        pred_start = time.time()

        _, pred_loader = self._get_data(flag='pred')
        self._load_best_model(setting)
        preds = self._run_prediction(pred_loader)
        self._save_predictions(setting, preds)

        pred_end = time.time()
        use_time = pred_end - pred_start
        print(f'Prediction use time: {use_time:.2f}s')
        return use_time

    def _load_best_model(self, setting: str) -> None:
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = os.path.join(path, 'checkpoint.pth')
        try:
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device, weights_only=True))
        except FileNotFoundError:
            raise FileNotFoundError(f"模型文件 {best_model_path} 未找到！")
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {str(e)}")

    def _run_prediction(self, pred_loader: DataLoader):
        """运行预测并返回结果"""
        self.model.eval()
        preds, trues = [], []
        pred_loader = tqdm(pred_loader, file=sys.stdout)
        with torch.no_grad():
            for batch_x, batch_anomaly in pred_loader:
                batch_x, batch_anomaly = self._move_to_device_pred(batch_x, batch_anomaly)

                pred = self.model(batch_x, batch_anomaly)
                pred = pred.detach().cpu().numpy()
                pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])

                # 新增：如果使用归一化数据，进行反归一化
                if self.args.use_normalized:
                    pred = self._denormalize(pred)

                preds.append(pred)
                pred_loader.desc = "[prediction]"

        preds = np.concatenate(preds, axis=0)
        return preds

    def _save_predictions(self, setting: str, preds) -> None:
        """保存预测结果"""
        folder_path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(folder_path, exist_ok=True)

        save_path_preds = os.path.join(folder_path, 'preds.npy')
        np.save(save_path_preds, preds)

    def rmse(self, pred, true):
        mask = self.args.mask_data  # (96, 1440)
        assert pred.shape == true.shape, "预测值与真实值形状不一致"
        assert mask.shape == (pred.shape[-2], pred.shape[-1]), "掩码形状不匹配"

        # 新增：如果使用归一化数据，进行反归一化
        if self.args.use_normalized or self.args.use_standardized:
            # 将tensor转换为numpy进行反归一化
            pred_np = pred.detach().cpu().numpy()
            true_np = true.detach().cpu().numpy()

            # 反归一化
            pred_denorm = self._denormalize(pred_np)
            true_denorm = self._denormalize(true_np)

            # 转换回tensor
            pred = torch.from_numpy(pred_denorm).to(self.device)
            true = torch.from_numpy(true_denorm).to(self.device)

        mask = mask.astype(bool)

        B, T, H, W = pred.shape
        pred_2d = pred.reshape(B * T, H, W)
        true_2d = true.reshape(B * T, H, W)

        valid_pred = pred_2d[:, mask]  # shape: (B*T, N_valid)
        valid_true = true_2d[:, mask]  # shape: (B*T, N_valid)

        # --- RMSE ---
        error = valid_pred - valid_true
        rmse = torch.sqrt(torch.mean(error ** 2))
        return rmse.item()


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target, mask):
        """
        pred: (B, T, H, W)
        target: (B, T, H, W)
        mask: (H, W) or (B, T, H, W)
        """
        # 确保 mask 广播到与 pred 相同的形状
        # 假设传入的 mask 是 (H, W)，需要扩展维度
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # 变成 (1, 1, H, W)

        # 计算平方误差
        squared_error = (pred - target) ** 2

        # 只保留 mask 为 1 (海洋) 的部分的误差
        masked_error = squared_error * mask

        # 计算平均值： 总误差 / 有效像素数
        loss = masked_error.sum() / (mask.sum() * pred.shape[0] * pred.shape[1] + 1e-8)

        return loss

