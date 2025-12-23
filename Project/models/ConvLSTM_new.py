import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint  # 引入 checkpoint
from functools import partial  # 引入偏函数处理


# --- 辅助函数 ---
def _extend_for_multilayer(param, num_layers):
    if not isinstance(param, list):
        param = [param] * num_layers
    if len(param) != num_layers:
        raise ValueError(f"参数列表长度 ({len(param)}) 必须等于 num_layers ({num_layers})")
    return param


# --- 核心：定义用于 checkpoint 的闭包函数 ---
# 这是一个静态函数，用于告诉 checkpoint 如何运行单个 Cell
def run_cell_step(input_tensor, h_prev, c_prev, cell_module):
    return cell_module(input_tensor, (h_prev, c_prev))


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv1 = nn.Conv2d(in_channels=self.in_channels + self.out_channels * 2,
                               out_channels=2 * self.out_channels,
                               kernel_size=self.kernel_size,
                               padding=self.padding,
                               bias=self.bias)
        self.conv2 = nn.Conv2d(in_channels=self.in_channels + self.out_channels,
                               out_channels=self.out_channels,
                               kernel_size=self.kernel_size,
                               padding=self.padding,
                               bias=self.bias)
        self.conv3 = nn.Conv2d(in_channels=self.in_channels + self.out_channels * 2,
                               out_channels=self.out_channels,
                               kernel_size=self.kernel_size,
                               padding=self.padding,
                               bias=self.bias)

    def forward(self, input_tensor, last_state):
        h_last, c_last = last_state
        combined_input = torch.cat([input_tensor, h_last, c_last], dim=1)
        combined_conv = self.conv1(combined_input)
        cc_i, cc_f = torch.split(combined_conv, self.out_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)

        combined_input = torch.cat([input_tensor, h_last], dim=1)
        cc_g = self.conv2(combined_input)
        g = torch.tanh(cc_g)
        c = f * c_last + i * g

        combined_input = torch.cat([input_tensor, h_last, c], dim=1)
        cc_o = self.conv3(combined_input)
        o = torch.sigmoid(cc_o)
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.out_channels, height, width, device=self.conv1.weight.device),
                torch.zeros(batch_size, self.out_channels, height, width, device=self.conv1.weight.device))


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels_list, kernel_size_list, num_layers):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        self.rnn_layers = nn.ModuleList()

        out_channels_list = _extend_for_multilayer(out_channels_list, num_layers)
        kernel_size_list = _extend_for_multilayer(kernel_size_list, num_layers)

        for i in range(num_layers):
            input_dim = in_channels if i == 0 else out_channels_list[i - 1]
            output_dim = out_channels_list[i]
            k_size = kernel_size_list[i]
            if isinstance(k_size, int): k_size = (k_size, k_size)

            stride = 1 if i == 0 else 2
            padding = (k_size[0] // 2, k_size[1] // 2)

            self.conv_layers.append(
                nn.Conv2d(input_dim, output_dim, kernel_size=k_size, stride=stride, padding=padding)
            )
            self.rnn_layers.append(
                ConvLSTMCell(in_channels=output_dim, out_channels=output_dim, kernel_size=k_size, bias=True)
            )

    def forward(self, x):
        b, t, _, h, w = x.size()
        states = []
        current_h, current_w = h, w
        # 初始化状态
        for i in range(self.num_layers):
            if i > 0:
                current_h //= 2
                current_w //= 2
            ch = self.rnn_layers[i].out_channels
            h_init = torch.zeros(b, ch, current_h, current_w).to(x.device)
            c_init = torch.zeros(b, ch, current_h, current_w).to(x.device)
            states.append((h_init, c_init))

        # 时间步循环
        for step in range(t):
            input_t = x[:, step, :, :, :]
            for i in range(self.num_layers):
                conv_out = self.conv_layers[i](input_t)
                h_i, c_i = states[i]

                # ========================================================
                # ✅ 优化：使用 Checkpoint 节省显存 (仿照老代码逻辑)
                # ========================================================
                # 只有当 input 或者是 hidden state 需要梯度时，checkpoint 才有意义
                if conv_out.requires_grad or h_i.requires_grad:
                    # 使用 partial 固定 cell_module 参数
                    run_fn = partial(run_cell_step, cell_module=self.rnn_layers[i])
                    # 执行 checkpoint
                    h_new, c_new = checkpoint(run_fn, conv_out, h_i, c_i, use_reentrant=False)
                else:
                    # 如果不需要梯度 (例如验证阶段)，直接前向传播，不使用 checkpoint
                    h_new, c_new = self.rnn_layers[i](conv_out, (h_i, c_i))

                states[i] = (h_new, c_new)
                input_t = h_new
        return states


class Decoder(nn.Module):
    def __init__(self, out_channels_list, kernel_size_list, num_layers, final_out_channels=1):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.rnn_layers = nn.ModuleList()
        self.deconv_layers = nn.ModuleList()

        out_channels_list = _extend_for_multilayer(out_channels_list, num_layers)
        kernel_size_list = _extend_for_multilayer(kernel_size_list, num_layers)
        decoder_channels = out_channels_list[::-1]
        decoder_kernels = kernel_size_list[::-1]

        for i in range(num_layers):
            ch = decoder_channels[i]
            k_size = decoder_kernels[i]
            if isinstance(k_size, int): k_size = (k_size, k_size)

            self.rnn_layers.append(
                ConvLSTMCell(in_channels=ch, out_channels=ch, kernel_size=k_size, bias=True)
            )
            if i < num_layers - 1:
                next_ch = decoder_channels[i + 1]
                self.deconv_layers.append(
                    nn.ConvTranspose2d(ch, next_ch, kernel_size=4, stride=2, padding=1)
                )
            else:
                self.final_conv = nn.Conv2d(ch, final_out_channels, kernel_size=1)

    def forward(self, encoder_states, future_steps):
        states = list(encoder_states)
        states.reverse()
        outputs = []
        b, _, h, w = states[0][0].size()
        current_input = torch.zeros(b, states[0][0].shape[1], h, w).to(states[0][0].device)

        for step in range(future_steps):
            for i in range(self.num_layers):
                h_i, c_i = states[i]

                # ========================================================
                # ✅ 优化：Decoder 也加入 Checkpoint
                # ========================================================
                if current_input.requires_grad or h_i.requires_grad:
                    run_fn = partial(run_cell_step, cell_module=self.rnn_layers[i])
                    h_new, c_new = checkpoint(run_fn, current_input, h_i, c_i, use_reentrant=False)
                else:
                    h_new, c_new = self.rnn_layers[i](current_input, (h_i, c_i))

                states[i] = (h_new, c_new)

                if i < self.num_layers - 1:
                    current_input = self.deconv_layers[i](h_new)
                else:
                    current_input = h_new

            pred = self.final_conv(current_input)
            outputs.append(pred)
            # 重置输入 (注意：纯生成模式下，下一个时刻的输入通常是0或者是上一个时刻的预测，这里保持原逻辑为0)
            current_input = torch.zeros(b, states[0][0].shape[1], h, w).to(states[0][0].device)

        # 同样需要 squeeze 逻辑
        preds = torch.stack(outputs, dim=1)
        return preds


class ConvLSTM_new(nn.Module):
    def __init__(self, args):
        super(ConvLSTM_new, self).__init__()

        if hasattr(args, 'in_channels'):
            in_channels = args.in_channels
        else:
            in_channels = 2 if getattr(args, 'add_anomaly', False) else 1

        out_channels_list = args.out_channels
        kernel_size_list = args.kernel_size
        num_layers = args.num_layers
        self.correction_len = getattr(args, 'correction_len', 1)
        self.add_anomaly = getattr(args, 'add_anomaly', False)

        self.encoder = Encoder(in_channels, out_channels_list, kernel_size_list, num_layers)
        self.decoder = Decoder(out_channels_list, kernel_size_list, num_layers)

    def forward(self, x, anomaly_data=None):
        # 1. 维度修正
        if self.add_anomaly:
            if anomaly_data is not None:
                if x.dim() == 4: x = x.unsqueeze(2)
                if anomaly_data.dim() == 4: anomaly_data = anomaly_data.unsqueeze(2)
                x = torch.cat([x, anomaly_data], dim=2)
        else:
            if x.dim() == 4:
                x = x.unsqueeze(2)

        states = self.encoder(x)
        preds = self.decoder(states, self.correction_len)

        # 2. 输出降维
        if preds.dim() == 5 and preds.size(2) == 1:
            preds = preds.squeeze(2)

        return preds