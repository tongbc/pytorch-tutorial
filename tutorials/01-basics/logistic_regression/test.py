import torch
import torch.nn as nn
# 输入维度 50，隐层100维，两层
lstm_seq = nn.LSTM(50, 100, num_layers=2)
# 查看网络的权重，ih和hh，共2层，所以有四个要学习的参数
print(lstm_seq.weight_hh_l0.size(), lstm_seq.weight_hh_l1.size(),lstm_seq.weight_ih_l0.size(),lstm_seq.weight_ih_l1.size() )
# q1： 输出的size是多少？ 都是torch.Size([400, 100]