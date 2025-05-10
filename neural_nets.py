import torch
import torch.nn as nn
import torch.nn.functional as F
####-------------------------------CWRU 数据集 网络---------STU 数据集 网络-----------------
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        # 卷积层1
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=20, stride=1, padding=0),
            # nn.LeakyReLU(),
            nn.Tanh(),
            nn.BatchNorm1d(10),
            nn.MaxPool1d(kernel_size=4, stride=2)
        )
        # 卷积层2
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=20, kernel_size=10, stride=1, padding=0),
            # nn.LeakyReLU(),
            nn.Tanh(),
            nn.BatchNorm1d(20),
            nn.MaxPool1d(kernel_size=4, stride=2)
        )
        # 卷积层3
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=40, kernel_size=5, stride=1, padding=0),
            # nn.LeakyReLU(),
            nn.Tanh(),
            nn.BatchNorm1d(40),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.mlp = nn.Sequential(
            nn.Linear(40*17, 160),         #------CWRU 使用----------
            # nn.Linear(40*72, 160),          #-------STU 使用----------
            # nn.Linear(40 * 17, 160),
            # nn.LeakyReLU(),
            nn.Tanh(),
            nn.Linear(160, 80),
            # nn.LeakyReLU(),
            nn.Tanh(),
            nn.Linear(80, 10),
            # nn.LeakyReLU(),
            # nn.Tanh(),
        )

    def forward(self, x):
        # print("="*100)
        # print("x", x.shape)
        x = x.unsqueeze(1)
        # print("x", x.shape)
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        # print("output", output.shape)
        output = output.view(-1, 40 * 17)  #------CWRU 使用------
        # output = output.view(-1, 40 * 72)    #------STU  使用------
        # output = output.view(output.size(0), -1)  # 将形状调整为 [batch_size, -1]
        # print("output", output.shape)
        output = self.mlp(output)
        # output = F.softmax(output,dim=-1)
        return output

# class cnn(nn.Module):
#     def __init__(self):
#         super(cnn, self).__init__()
#         # 卷积层1
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=16, kernel_size=20, stride=1, padding=0),  # 保持与原网络一致
#             nn.ReLU(),
#             nn.BatchNorm1d(16),
#             nn.MaxPool1d(kernel_size=4, stride=2)  # 保持与原网络一致
#         )
#         # 卷积层2
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(in_channels=16, out_channels=32, kernel_size=10, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm1d(32),
#             nn.MaxPool1d(kernel_size=4, stride=2)
#         )
#         # 卷积层3
#         self.conv3 = nn.Sequential(
#             nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm1d(64),
#             nn.MaxPool1d(kernel_size=2, stride=2)
#         )
#         # 全连接层
#         self.mlp = nn.Sequential(
#             nn.Linear(64 * 17, 256),  # 保持与原网络一致的输入维度
#             nn.ReLU(),
#             nn.Dropout(0.5),  # 添加Dropout防止过拟合
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, 10)  # 输出维度保持为10
#         )
#
#     def forward(self, x):
#         x = x.unsqueeze(1)  # 增加通道维度
#         output = self.conv1(x)
#         output = self.conv2(output)
#         output = self.conv3(output)
#         output = output.view(output.size(0), -1)  # 展平
#         output = self.mlp(output)
#         return output

# ## -------------------------------HUST 数据集 网络----------------------------
# class cnn(nn.Module):
#     def __init__(self):
#         super(cnn, self).__init__()
#         # 卷积层1
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=10, kernel_size=20, stride=1, padding=0),
#             nn.Tanh(),
#             nn.BatchNorm1d(10),
#             nn.MaxPool1d(kernel_size=4, stride=2)
#         )
#         # 卷积层2
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(in_channels=10, out_channels=20, kernel_size=10, stride=1, padding=0),
#             nn.Tanh(),
#             nn.BatchNorm1d(20),
#             nn.MaxPool1d(kernel_size=4, stride=2)
#         )
#         # 卷积层3
#         self.conv3 = nn.Sequential(
#             nn.Conv1d(in_channels=20, out_channels=40, kernel_size=5, stride=1, padding=0),
#             nn.Tanh(),
#             nn.BatchNorm1d(40),
#             nn.MaxPool1d(kernel_size=2, stride=2)
#         )
#         # 全连接层
#         self.mlp = nn.Sequential(
#             nn.Linear(40 * 120, 160),  # 修改为正确的输入特征数量
#             nn.Tanh(),
#             nn.Linear(160, 80),
#             nn.Tanh(),
#             nn.Linear(80, 9)  # 输出层，9个类别
#         )
#     def forward(self, x):
#         x = x.unsqueeze(1)  # 添加通道维度
#         output = self.conv1(x)
#         output = self.conv2(output)
#         output = self.conv3(output)
#         # print("Conv3 output shape:", output.shape)  # 打印卷积层的输出形状
#         output = output.view(output.size(0), -1)  # 展平
#         # print("Flattened output shape:", output.shape)  # 打印展平后的形状
#         output = self.mlp(output)
#         return output




