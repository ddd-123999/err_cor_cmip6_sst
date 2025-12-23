from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from datetime import date

# ---------------- 数据频率配置 ----------------
DATA_FREQUENCY = 'daily'  # 'daily' 或 'monthly'

# 完整的观测数据 (sst.npz) 的总起始年份
START_YEAR_FULL = 1982

# 训练集结束年份
END_YEAR_TRAIN = 2014  # 训练到 2014 年底
# 验证集起始年份
START_YEAR_VAL = 2015
# 测试集起始年份
START_YEAR_TEST = 2020


def _get_raw_data_points(start_year: int, end_year: int) -> int:
    """计算原始数据点的数量，支持日数据和月数据"""
    if DATA_FREQUENCY == 'daily':
        # 日数据：计算天数
        start_date = date(start_year, 1, 1)
        end_date = date(end_year, 12, 31)
        return (end_date - start_date).days + 1
    else:
        # 月数据：计算月份数
        return (end_year - start_year + 1) * 12


# 计算索引边界
TRAIN_POINTS = _get_raw_data_points(START_YEAR_FULL, END_YEAR_TRAIN)  # 日数据: 12053, 月数据: 396
VAL_POINTS = _get_raw_data_points(START_YEAR_VAL, 2019)  # 日数据: 1826, 月数据: 60
TEST_POINTS = _get_raw_data_points(START_YEAR_TEST, 2024)  # 日数据: 1826, 月数据: 60

# 观测数据切片索引
IDX_TRAIN_END = TRAIN_POINTS
IDX_VAL_START = IDX_TRAIN_END
IDX_VAL_END = IDX_VAL_START + VAL_POINTS
IDX_TEST_START = IDX_VAL_END

# 总时间长度
TOTAL_POINTS = IDX_TEST_START + TEST_POINTS

print(f"数据频率: {DATA_FREQUENCY}")
print(f"训练集数据点: {TRAIN_POINTS}")
print(f"验证集数据点: {VAL_POINTS}")
print(f"测试集数据点: {TEST_POINTS}")


class DatasetTrain(Dataset):
    def __init__(self, root_path_target, data_path_target, root_path_correction, data_path_correction, flag, seq_len,
                 correction_len, root_path_anomaly, data_path_anomaly, add_anomaly, use_normalized, use_standardized):
        self.root_path_target = root_path_target
        self.data_path_target = data_path_target
        self.path_target = os.path.join(self.root_path_target, self.data_path_target)

        self.root_path_correction = root_path_correction
        self.data_path_correction = data_path_correction
        self.path_correction = os.path.join(self.root_path_correction, self.data_path_correction)

        self.add_anomaly = add_anomaly
        if self.add_anomaly:
            self.root_path_anomaly = root_path_anomaly
            self.data_path_anomaly = data_path_anomaly
            self.path_anomaly = os.path.join(self.root_path_anomaly, self.data_path_anomaly)

        self.seq_len = seq_len
        self.correction_len = correction_len
        self.use_normalized = use_normalized
        self.use_standardized = use_standardized
        self.flag = flag

        assert flag in ['train']

        self.__read_data__()

    def __read_data__(self):
        # 1. 模式数据 - 直接加载，已经是 (Time, Lat, Lon)
        self.correction_data = np.load(self.path_correction)['sst']
        self.len = self.correction_data.shape[0]  # 训练集长度

        # 2. 观测数据 - 直接加载并切片
        raw_target_data = np.load(self.path_target)['sst']
        self.target_data = raw_target_data[:self.len, :, :]

        # 3. Anomaly数据 - 直接加载
        if self.add_anomaly:
            self.anomaly_data = np.load(self.path_anomaly)['sst_anomaly']

        # 验证数据形状
        assert self.target_data.shape[0] == self.len, f"观测数据长度不匹配: {self.target_data.shape[0]} != {self.len}"
        assert self.correction_data.shape[
                   0] == self.len, f"模式数据长度不匹配: {self.correction_data.shape[0]} != {self.len}"
        if self.add_anomaly:
            assert self.anomaly_data.shape[
                       0] == self.len, f"anomaly数据长度不匹配: {self.anomaly_data.shape[0]} != {self.len}"

            # ⚠️ 修改：打印数据使用状态
            if self.use_standardized:
                data_type = "Z-score 标准化"
            elif self.use_normalized:
                data_type = "Min-Max 归一化"
            else:
                data_type = "原始"
            print(
                f"训练集({data_type}): 模式数据形状 {self.correction_data.shape}, 观测数据形状 {self.target_data.shape}")

    def __getitem__(self, index):
        e_begin = index
        e_end = e_begin + self.seq_len
        d_begin = e_end - self.correction_len
        d_end = e_begin + self.seq_len

        x = self.correction_data[e_begin:e_end, :, :]
        y = self.target_data[d_begin:d_end, :, :]
        anomaly = []
        if self.add_anomaly:
            anomaly = self.anomaly_data[e_begin:e_end, :, :]

        return x, y, anomaly

    def __len__(self):
        return self.len - self.seq_len + 1


class DatasetVal(Dataset):
    def __init__(self, root_path_target, data_path_target, root_path_correction, data_path_correction, flag, seq_len,
                 correction_len, root_path_anomaly, data_path_anomaly, add_anomaly, use_normalized, use_standardized):
        self.root_path_target = root_path_target
        self.data_path_target = data_path_target
        self.path_target = os.path.join(self.root_path_target, self.data_path_target)

        self.root_path_correction = root_path_correction
        self.data_path_correction = data_path_correction
        self.path_correction = os.path.join(self.root_path_correction, self.data_path_correction)

        self.add_anomaly = add_anomaly
        if self.add_anomaly:
            self.root_path_anomaly = root_path_anomaly
            self.data_path_anomaly = data_path_anomaly
            self.path_anomaly = os.path.join(self.root_path_anomaly, self.data_path_anomaly)

        self.seq_len = seq_len
        self.correction_len = correction_len
        self.use_normalized = use_normalized
        self.use_standardized = use_standardized

        assert flag in ['val']
        self.__read_data__()

    def __read_data__(self):
        # 1. 模式数据 - 直接加载
        self.correction_data = np.load(self.path_correction)['sst']
        self.len = self.correction_data.shape[0]  # 验证集长度

        # 2. 观测数据 - 直接加载并切片
        raw_target_data = np.load(self.path_target)['sst']
        self.target_data = raw_target_data[IDX_VAL_START:IDX_VAL_END, :, :]

        # 3. Anomaly数据 - 直接加载
        if self.add_anomaly:
            self.anomaly_data = np.load(self.path_anomaly)['sst_anomaly']

        # 验证数据形状
        assert self.target_data.shape[0] == self.len, f"观测数据长度不匹配: {self.target_data.shape[0]} != {self.len}"
        assert self.correction_data.shape[
                   0] == self.len, f"模式数据长度不匹配: {self.correction_data.shape[0]} != {self.len}"
        if self.add_anomaly:
            assert self.anomaly_data.shape[
                       0] == self.len, f"anomaly数据长度不匹配: {self.anomaly_data.shape[0]} != {self.len}"

            # ⚠️ 修改：打印数据使用状态
            if self.use_standardized:
                data_type = "Z-score 标准化"
            elif self.use_normalized:
                data_type = "Min-Max 归一化"
            else:
                data_type = "原始"
            print(
                f"训练集({data_type}): 模式数据形状 {self.correction_data.shape}, 观测数据形状 {self.target_data.shape}")

    def __getitem__(self, index):
        e_begin = index
        e_end = e_begin + self.seq_len
        d_begin = e_end - self.correction_len
        d_end = e_begin + self.seq_len

        x = self.correction_data[e_begin:e_end, :, :]
        y = self.target_data[d_begin:d_end, :, :]
        anomaly = []
        if self.add_anomaly:
            anomaly = self.anomaly_data[e_begin:e_end, :, :]

        return x, y, anomaly

    def __len__(self):
        return self.len - self.seq_len + 1


class DatasetTest(Dataset):
    def __init__(self, root_path_target, data_path_target, root_path_correction, data_path_correction, flag, seq_len,
                 correction_len, root_path_anomaly, data_path_anomaly, add_anomaly, use_normalized, use_standardized):
        self.root_path_target = root_path_target
        self.data_path_target = data_path_target
        self.path_target = os.path.join(self.root_path_target, self.data_path_target)

        self.root_path_correction = root_path_correction
        self.data_path_correction = data_path_correction
        self.path_correction = os.path.join(self.root_path_correction, self.data_path_correction)

        self.add_anomaly = add_anomaly
        if self.add_anomaly:
            self.root_path_anomaly = root_path_anomaly
            self.data_path_anomaly = data_path_anomaly
            self.path_anomaly = os.path.join(self.root_path_anomaly, self.data_path_anomaly)

        self.seq_len = seq_len
        self.correction_len = correction_len
        self.use_normalized = use_normalized
        self.use_standardized = use_standardized

        assert flag in ['test']
        self.__read_data__()

    def __read_data__(self):
        # 1. 模式数据 - 直接加载
        self.correction_data = np.load(self.path_correction)['sst']
        self.len = self.correction_data.shape[0]  # 测试集长度

        # 2. 观测数据 - 直接加载并切片
        raw_target_data = np.load(self.path_target)['sst']
        self.target_data = raw_target_data[IDX_TEST_START:, :, :]

        # 3. Anomaly数据 - 直接加载
        if self.add_anomaly:
            self.anomaly_data = np.load(self.path_anomaly)['sst_anomaly']

        # 验证数据形状
        assert self.target_data.shape[0] == self.len, f"观测数据长度不匹配: {self.target_data.shape[0]} != {self.len}"
        assert self.correction_data.shape[
                   0] == self.len, f"模式数据长度不匹配: {self.correction_data.shape[0]} != {self.len}"
        if self.add_anomaly:
            assert self.anomaly_data.shape[
                       0] == self.len, f"anomaly数据长度不匹配: {self.anomaly_data.shape[0]} != {self.len}"

            # ⚠️ 修改：打印数据使用状态
            if self.use_standardized:
                data_type = "Z-score 标准化"
            elif self.use_normalized:
                data_type = "Min-Max 归一化"
            else:
                data_type = "原始"
            print(
                f"训练集({data_type}): 模式数据形状 {self.correction_data.shape}, 观测数据形状 {self.target_data.shape}")

    def __getitem__(self, index):
        e_begin = index
        e_end = e_begin + self.seq_len
        d_begin = e_end - self.correction_len
        d_end = e_begin + self.seq_len

        x = self.correction_data[e_begin:e_end, :, :]
        y = self.target_data[d_begin:d_end, :, :]
        anomaly = []
        if self.add_anomaly:
            anomaly = self.anomaly_data[e_begin:e_end, :, :]

        return x, y, anomaly

    def __len__(self):
        return self.len - self.seq_len + 1


class DatasetPred(Dataset):
    def __init__(self, root_path_pre, data_path_pre, root_path_correction, data_path_correction, flag, seq_len,
                 correction_len, root_path_anomaly, data_path_anomaly, root_path_anomaly_in_pre,
                 data_path_anomaly_in_pre, add_anomaly, use_normalized, use_standardized):
        self.root_path_pre = root_path_pre
        self.data_path_pre = data_path_pre
        self.path_pre = os.path.join(self.root_path_pre, self.data_path_pre)

        self.root_path_correction = root_path_correction
        self.data_path_correction = data_path_correction
        self.path_correction = os.path.join(self.root_path_correction, self.data_path_correction)

        self.add_anomaly = add_anomaly
        if self.add_anomaly:
            # anomaly0: 训练集尾部的 anomaly 数据
            self.root_path_anomaly0 = root_path_anomaly
            self.data_path_anomaly0 = data_path_anomaly
            self.path_anomaly0 = os.path.join(self.root_path_anomaly0, self.data_path_anomaly0)

            # anomaly1: 预测集本身的 anomaly 数据
            self.root_path_anomaly1 = root_path_anomaly_in_pre
            self.data_path_anomaly1 = data_path_anomaly_in_pre
            self.path_anomaly1 = os.path.join(self.root_path_anomaly1, self.data_path_anomaly1)

        self.seq_len = seq_len
        self.correction_len = correction_len
        self.use_normalized = use_normalized
        self.use_standardized = use_standardized

        assert flag in ['pred']
        self.__read_data__()

    def __read_data__(self):
        # 预测任务：拼接历史数据尾部
        start_date = 2025
        end_date = 2100
        if DATA_FREQUENCY == 'daily':
            num_total = (date(end_date, 12, 31) - date(start_date, 1, 1)).days + 1 + self.seq_len - self.correction_len
        else:
            num_total = (end_date - start_date + 1) * 12 + self.seq_len - self.correction_len

        self.len = num_total

        # 模式数据拼接 - 直接加载
        self.pred_data0 = np.load(self.path_correction)['sst'][-(self.seq_len - self.correction_len):, :, :]
        self.pred_data1 = np.load(self.path_pre)['sst']
        self.pred_data = np.concatenate((self.pred_data0, self.pred_data1), axis=0)

        # Anomaly数据拼接 - 直接加载
        if self.add_anomaly:
            self.pred_anomaly_data0 = np.load(self.path_anomaly0)['sst_anomaly'][-(self.seq_len - self.correction_len):,
                                      :, :]
            self.pred_anomaly_data1 = np.load(self.path_anomaly1)['sst_anomaly']
            self.pred_anomaly_data = np.concatenate((self.pred_anomaly_data0, self.pred_anomaly_data1), axis=0)

        # 验证数据形状
        assert self.pred_data.shape[0] == self.len, f"预测数据长度不匹配: {self.pred_data.shape[0]} != {self.len}"
        if self.add_anomaly:
            assert self.pred_anomaly_data.shape[
                       0] == self.len, f"anomaly数据长度不匹配: {self.pred_anomaly_data.shape[0]} != {self.len}"

            # ⚠️ 修改：打印数据使用状态
            if self.use_standardized:
                data_type = "Z-score 标准化"
            elif self.use_normalized:
                data_type = "Min-Max 归一化"
            else:
                data_type = "原始"
            print(
                f"训练集({data_type}): 模式数据形状 {self.correction_data.shape}, 观测数据形状 {self.target_data.shape}")

    def __getitem__(self, index):
        e_begin = index
        e_end = e_begin + self.seq_len

        x = self.pred_data[e_begin:e_end, :, :]
        anomaly = []
        if self.add_anomaly:
            anomaly = self.pred_anomaly_data[e_begin:e_end, :, :]

        return x, anomaly

    def __len__(self):
        return self.len - self.seq_len + 1


if __name__ == "__main__":
    pass