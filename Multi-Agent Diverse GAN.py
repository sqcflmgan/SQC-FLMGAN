
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import scipy.io
import scipy.io as io
import time
import os
import random


#-------余弦相似度损失-------
def cosinematrix(A,B):
    prod = torch.mm(A,B.t())
    norm_A = torch.norm(A,p=2,dim=1).unsqueeze(0)
    norm_B = torch.norm(B, p=2, dim=1).unsqueeze(0)
    cos = prod.div(torch.mm(norm_A.t(),norm_B))
    cos_epitri = np.triu(cos.cpu().detach().numpy(),1)    #-1:下三角；  0：对角线；  1：上三角
    cos_epitri_sum = cos_epitri.sum()*2/(len(A)*(len(A)-1))
    return cos_epitri_sum
#-------生成器1-------
class Generator1(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(Generator1, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size1)
        self.map2 = nn.Linear(hidden_size1, hidden_size2)
        self.map3 = nn.Linear(hidden_size2, hidden_size3)
        self.map4 = nn.Linear(hidden_size3, output_size)
        self.f = torch.sigmoid
        self.f1 = torch.relu
        self.f2 = torch.nn.LeakyReLU()
        self.f3 = torch.tanh
        self.f4 = torch.nn.ELU(alpha=1.0, inplace=False)

    def forward(self, x):
        x = self.f3(self.map1(x))
        x = self.f3(self.map2(x))
        x = self.f3(self.map3(x))
        x = self.f(self.map4(x))
        return x
#-------生成器2-------
class Generator2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(Generator2, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size1)
        self.map2 = nn.Linear(hidden_size1, hidden_size2)
        self.map3 = nn.Linear(hidden_size2, hidden_size3)
        self.map4 = nn.Linear(hidden_size3, output_size)
        self.f = torch.sigmoid
        self.f1 = torch.relu
        self.f2 = torch.nn.LeakyReLU()
        self.f3 = torch.tanh
        self.f4 = torch.nn.ELU(alpha=1.0, inplace=False)

    def forward(self, x):
        x = self.f3(self.map1(x))
        x = self.f3(self.map2(x))
        x = self.f3(self.map3(x))
        x = self.f(self.map4(x))
        return x
#-------生成器3-------
class Generator3(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(Generator3, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size1)
        self.map2 = nn.Linear(hidden_size1, hidden_size2)
        self.map3 = nn.Linear(hidden_size2, hidden_size3)
        self.map4 = nn.Linear(hidden_size3, output_size)
        self.f = torch.sigmoid
        self.f1 = torch.relu
        self.f2 = torch.nn.LeakyReLU()
        self.f3 = torch.tanh
        self.f4 = torch.nn.ELU(alpha=1.0, inplace=False)

    def forward(self, x):
        x = self.f3(self.map1(x))
        x = self.f3(self.map2(x))
        x = self.f3(self.map3(x))
        x = self.f(self.map4(x))
        return x
#-------判别器-------
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size1)
        self.map2 = nn.Linear(hidden_size1, hidden_size2)
        self.map3 = nn.Linear(hidden_size2, hidden_size3)
        self.map4 = nn.Linear(hidden_size3, output_size)
        self.f1 = torch.sigmoid
        self.f2 = torch.nn.LeakyReLU()
        self.f3 = torch.tanh
        self.f4 = torch.nn.ELU(alpha=1.0, inplace=False)

    def forward(self, x):
        x = self.f1(self.map1(x))
        x = self.f1(self.map2(x))
        x = self.f1(self.map3(x))
        return self.map4(x)
#-------网络训练-------
def GAN_train(Epoch,G1,G2,G3,D,d_steps,g_steps,d_real_data,d_fake_data, cuda_gpu, gpus,cls):
    #GPU运行数据
    if (cuda_gpu):
        d_real_data = Variable(torch.Tensor(d_real_data).cuda())
        d_fake_data = Variable(torch.Tensor(d_fake_data).cuda())
    else:
        d_real_data = Variable(torch.Tensor(d_real_data))
        d_fake_data = Variable(torch.Tensor(d_fake_data))
    #GPU运行模型
    if (cuda_gpu):
        G1 = torch.nn.DataParallel(G1, device_ids=gpus).cuda()
        G2 = torch.nn.DataParallel(G2, device_ids=gpus).cuda()
        G3 = torch.nn.DataParallel(G3, device_ids=gpus).cuda()
        D = torch.nn.DataParallel(D, device_ids=gpus).cuda()
    #优化器
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # d_optimizer = nn.DataParallel(d_optimizer, device_ids=gpus)
    # d_optimizer = torch.optim.SGD(D.parameters(), lr=0.001, weight_decay=1e-6, momentum=0.9, nesterov=False)
    # g_optimizer = torch.optim.SGD(G.parameters(), lr=0.01,  weight_decay=1e-6, momentum=0.9, nesterov=True)
    g_optimizer1 = torch.optim.Adam(G1.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    g_optimizer2 = torch.optim.Adam(G2.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,amsgrad=False)
    g_optimizer3 = torch.optim.Adam(G3.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,amsgrad=False)
    # g_optimizer = nn.DataParallel(g_optimizer, device_ids=gpus)

    for epoch in range(Epoch):
        start_time = time.time()
        # d_steps = np.int(d_steps*(Epoch/50-np.ceil(epoch/50))/(Epoch/50))
        for d_index in range(d_steps):
            # 1. Train D on real+fake
            D.zero_grad()
            #  1A: Train D on real
            d_real_decision = D(d_real_data)
            label_real = Variable(torch.zeros([len(d_real_decision)]))
            label_real = torch.tensor(label_real,dtype=torch.int64)
            d_real_error = criterion(d_real_decision, label_real.cuda())  # ones = true
            d_real_error.backward()  # compute/store gradients, but don't change params

            #  1B: Train D on fake
            d_fake_data1 = G1(d_fake_data)
            d_fake_decision1 = D(d_fake_data1)
            label_fake1 = Variable(torch.ones([len(d_fake_decision1)]))
            label_fake1 = torch.tensor(label_fake1, dtype=torch.int64)
            d_fake_error1 = criterion(d_fake_decision1, label_fake1.cuda())  # zeros = fake
            d_fake_error1.backward()

            d_fake_data2 = G2(d_fake_data)
            d_fake_decision2 = D(d_fake_data2)
            label_fake2 = Variable(2*torch.ones([len(d_fake_decision2)]))
            label_fake2 = torch.tensor(label_fake2, dtype=torch.int64)
            d_fake_error2 = criterion(d_fake_decision2,label_fake2.cuda())  # zeros = fake
            d_fake_error2.backward()

            d_fake_data3 = G3(d_fake_data)
            d_fake_decision3 = D(d_fake_data3)
            label_fake3 = Variable(3*torch.ones([len(d_fake_decision3)]))
            label_fake3 = torch.tensor(label_fake3, dtype=torch.int64)
            d_fake_error3 = criterion(d_fake_decision3,label_fake3.cuda())  # zeros = fake
            d_fake_error3.backward()

            d_fake_error = d_fake_error1 + d_fake_error2 + d_fake_error3
            d_error=d_fake_error + d_real_error
            # d_error.backward()
            d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

            dre, dfe1, dfe2, dfe3 = d_real_error.item(), d_fake_error1.item(), d_fake_error2.item(), d_fake_error3.item(),

        for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G1.zero_grad()
            g_fake_data1 = G1(d_fake_data)
            g_fake_data2 = G2(d_fake_data)
            g_fake_data3 = G3(d_fake_data)
            dg_fake_decision1 = D(g_fake_data1)
            g_loss1 = criterion(dg_fake_decision1, label_real.cuda())  # Train G to pretend it's genuine
            Loss_cos1 = cosinematrix(g_fake_data1, g_fake_data2) + cosinematrix(g_fake_data1, g_fake_data3)
            # g_error1 = g_loss1 - 0.1 * (g_loss1 - 1 / (3 - 1) * (g_loss2 + g_loss3 + Loss_cos1))
            g_error1 = g_loss1
            g_error1.backward()
            g_optimizer1.step()  # Only optimizes G's parameters
            ge1 = g_error1.item()

        # for g_index in range(g_steps):
            G2.zero_grad()
            g_fake_data1 = G1(d_fake_data)
            g_fake_data2 = G2(d_fake_data)
            g_fake_data3 = G3(d_fake_data)
            dg_fake_decision2 = D(g_fake_data2)
            g_loss2 = criterion(dg_fake_decision2, label_real.cuda())  # Train G to pretend it's genuine
            Loss_cos2 = cosinematrix(g_fake_data2, g_fake_data1) + cosinematrix(g_fake_data2, g_fake_data3)
            g_error2 = g_loss2
            g_error2.backward()
            g_optimizer2.step()  # Only optimizes G's parameters
            ge2 = g_error2.item()

        # for g_index in range(g_steps):
            G3.zero_grad()
            g_fake_data1 = G1(d_fake_data)
            g_fake_data2 = G2(d_fake_data)
            g_fake_data3 = G3(d_fake_data)
            dg_fake_decision3 = D(g_fake_data3)
            g_loss3 = criterion(dg_fake_decision3, label_real.cuda())  # Train G to pretend it's genuine
            Loss_cos3 = cosinematrix(g_fake_data3, g_fake_data1) + cosinematrix(g_fake_data3, g_fake_data2)
            g_error3 = g_loss3
            g_error3.backward()
            g_optimizer3.step()  # Only optimizes G's parameters
            ge3 = g_error3.item()

        end_time = time.time()
        time_len = end_time - start_time
        print("Class:[{}] Epoch [{}/{}]:  D[real_err:{:.6f}, fake_err1:{:.6f}, fake_err2:{:.6f}, fake_err3:{:.6f}]  G[err1:{:.6f},err2:{:.6f},err3:{:.6f}]; Real Dist:[mean:{:.6f}  std:{:.6f}], Fake Dist1:[mean1:{:.6f}  std1:{:.6f}]; run time: {:.6f}" .format
                (cls, epoch, Epoch, dre, dfe1,dfe2,dfe3, ge1,ge2,ge3, torch.mean(d_real_data), torch.std(d_real_data), torch.mean(d_fake_data1), torch.std(d_fake_data1), time_len))
        if ((torch.abs(torch.std(d_real_data)-torch.std(d_fake_data1))<0.05*torch.std(d_real_data)) and
           (torch.abs(torch.mean(d_real_data)-torch.mean(d_fake_data1))<0.05*torch.mean(d_real_data))):
            return
if __name__ == '__main__':
#--------加载数据----------
    src, tar = 'CWRU\\CWRU_1797\\HP0_1797\\仿真\\SimulationData_1797_HP0_DE_FFT.mat',\
               'CWRU\\CWRU_1797\\HP0_1797\\实验\\ExperimentalData_1797_HP0_DE_FFT.mat'
    src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    train_x, train_y, val_x, val_y = torch.Tensor(src_domain['signal_FFT']) , torch.LongTensor(src_domain['labels']).squeeze(), \
                                     torch.Tensor(tar_domain['signal_FFT']) , torch.LongTensor(tar_domain['labels']).squeeze()
    #-----选取数据量----
    num_sample = 5
    a1, a2, a3 = torch.tensor(random.sample(range(0,300),num_sample)), torch.tensor(random.sample(range(300,600),num_sample)), torch.tensor(random.sample(range(600,900),num_sample))
    a4, a5, a6 = torch.tensor(random.sample(range(900,1200),num_sample)), torch.tensor(random.sample(range(1200,1500),num_sample)), torch.tensor(random.sample(range(1500,1800),num_sample))
    a7, a8, a9, a10 = torch.tensor(random.sample(range(1800,2100),num_sample)), torch.tensor(random.sample(range(2100,2400),num_sample)), torch.tensor(random.sample(range(2400,2700),num_sample)),torch.tensor(random.sample(range(2700,3000),num_sample))

    train_x, val_x= np.array(train_x), np.array(val_x)

    train_x_max, val_x_max = np.max(train_x.T, axis=0), np.max(val_x.T, axis=0)
    print('max',train_x_max)
    train_x_min, val_x_min = np.min(train_x.T, axis=0), np.min(val_x.T, axis=0)
    print('min', train_x_min)
    train_x, val_x = (train_x.T - train_x_min) / (train_x_max - train_x_min), (val_x.T - val_x_min) / (val_x_max - val_x_min)

    df_train = pd.DataFrame(train_x.T)

    train_x, val_x = torch.tensor(train_x.T, dtype=torch.float32), torch.tensor(val_x.T, dtype=torch.float32)


    # -----加噪声-----
    noise = torch.tensor(np.random.random((3000, 200)), dtype=torch.float32)
    # train_x = train_x + 0.1 * noise
    train_x = 0.1 * noise
    #-----cuda运行-----
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda_gpu = torch.cuda.is_available()
    gpus = [0]
    #-------数据划分-------
    Xs1_1, Ys1_1, Xs1_2, Ys1_2, Xs1_3, Ys1_3 = train_x[a1, 0:200], train_y[a1], train_x[a2, 0:200], train_y[a2], train_x[a3,0:200], train_y[a3]
    Xs1_4, Ys1_4, Xs1_5, Ys1_5, Xs1_6, Ys1_6 = train_x[a4, 0:200], train_y[a4], train_x[a5, 0:200], train_y[a5], train_x[a6,0:200], train_y[a6]
    Xs1_7, Ys1_7, Xs1_8, Ys1_8, Xs1_9, Ys1_9, Xs1_10, Ys1_10 = train_x[a7, 0:200], train_y[a7], train_x[a8, 0:200], train_y[a8], train_x[a9,0:200], train_y[a9], train_x[a10,0:200], train_y[a10]

    Xt1_1, Yt1_1, Xt1_2, Yt1_2, Xt1_3, Yt1_3 = val_x[a1, 0:200], val_y[a1], val_x[a2, 0:200], val_y[a2], val_x[a3,0:200], val_y[a3]
    Xt1_4, Yt1_4, Xt1_5, Yt1_5, Xt1_6, Yt1_6 = val_x[a4, 0:200], val_y[a4], val_x[a5, 0:200], val_y[a5], val_x[a6,0:200], val_y[a6]
    Xt1_7, Yt1_7, Xt1_8, Yt1_8, Xt1_9, Yt1_9, Xt1_10, Yt1_10 = val_x[a7, 0:200], val_y[a7], val_x[a8, 0:200], val_y[a8], val_x[a9,0:200], val_y[a9], val_x[a10,0:200], val_y[a10]
    # ### ---第一类---
    # G1_1 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G2_1 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G3_1 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # D1_1 = Discriminator(input_size=200, hidden_size1=150, hidden_size2=80, hidden_size3=40, output_size=4)
    # GAN_train(Epoch=1000, G1=G1_1, G2=G2_1, G3=G3_1, D=D1_1, d_steps=20, g_steps=20, d_fake_data=Xs1_1, d_real_data=Xt1_1, cuda_gpu= cuda_gpu, gpus=gpus, cls='S1')
    # torch.save(D1_1.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\D1_1.pth')
    # torch.save(G1_1.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G1_1.pth')
    # torch.save(G2_1.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G2_1.pth')
    # torch.save(G3_1.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G3_1.pth')
    # #---第二类---
    # G1_2 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G2_2 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G3_2 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # D1_2 = Discriminator(input_size=200, hidden_size1=150, hidden_size2=80, hidden_size3=40, output_size=4)
    # GAN_train(Epoch=1000, G1=G1_2, G2=G2_2, G3=G3_2, D=D1_2, d_steps=20, g_steps=20, d_fake_data=Xs1_2, d_real_data=Xt1_2, cuda_gpu= cuda_gpu, gpus=gpus, cls='S2')
    # torch.save(D1_2.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\D1_2.pth')
    # torch.save(G1_2.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G1_2.pth')
    # torch.save(G2_2.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G2_2.pth')
    # torch.save(G3_2.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G3_2.pth')
    # # ---第三类---
    # G1_3 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G2_3 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G3_3 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # D1_3 = Discriminator(input_size=200, hidden_size1=150, hidden_size2=80, hidden_size3=40, output_size=4)
    # GAN_train(Epoch=1000, G1=G1_3, G2=G2_3, G3=G3_3, D=D1_3, d_steps=20, g_steps=20, d_fake_data=Xs1_3, d_real_data=Xt1_3, cuda_gpu= cuda_gpu, gpus=gpus, cls='S3')
    # torch.save(D1_3.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\D1_3.pth')
    # torch.save(G1_3.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G1_3.pth')
    # torch.save(G2_3.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G2_3.pth')
    # torch.save(G3_3.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G3_3.pth')
    # # ---第四类---
    # G1_4 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G2_4 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G3_4 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # D1_4 = Discriminator(input_size=200, hidden_size1=150, hidden_size2=80, hidden_size3=40, output_size=4)
    # GAN_train(Epoch=1000, G1=G1_4, G2=G2_4, G3=G3_4, D=D1_4, d_steps=20, g_steps=20, d_fake_data=Xs1_4, d_real_data=Xt1_4, cuda_gpu=cuda_gpu,gpus=gpus, cls='S4')
    # torch.save(D1_4.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\D1_4.pth')
    # torch.save(G1_4.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G1_4.pth')
    # torch.save(G2_4.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G2_4.pth')
    # torch.save(G3_4.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G3_4.pth')
    # # ---第五类---
    # G1_5 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G2_5 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G3_5 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # D1_5 = Discriminator(input_size=200, hidden_size1=150, hidden_size2=80, hidden_size3=40, output_size=4)
    # GAN_train(Epoch=1000, G1=G1_5, G2=G2_5, G3=G3_5, D=D1_5, d_steps=20, g_steps=20, d_fake_data=Xs1_5, d_real_data=Xt1_5, cuda_gpu=cuda_gpu,gpus=gpus, cls='S5')
    # torch.save(D1_5.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\D1_5.pth')
    # torch.save(G1_5.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G1_5.pth')
    # torch.save(G2_5.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G2_5.pth')
    # torch.save(G3_5.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G3_5.pth')
    # # ---第六类---
    # G1_6 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G2_6 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G3_6 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # D1_6 = Discriminator(input_size=200, hidden_size1=150, hidden_size2=80, hidden_size3=40, output_size=4)
    # GAN_train(Epoch=1000, G1=G1_6, G2=G2_6, G3=G3_6, D=D1_6, d_steps=20, g_steps=20, d_fake_data=Xs1_6, d_real_data=Xt1_6, cuda_gpu=cuda_gpu,gpus=gpus, cls='S6')
    # torch.save(D1_6.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\D1_6.pth')
    # torch.save(G1_6.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G1_6.pth')
    # torch.save(G2_6.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G2_6.pth')
    # torch.save(G3_6.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G3_6.pth')
    # # ---第七类---
    # G1_7 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G2_7 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G3_7 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # D1_7 = Discriminator(input_size=200, hidden_size1=150, hidden_size2=80, hidden_size3=40, output_size=4)
    # GAN_train(Epoch=1000, G1=G1_7, G2=G2_7, G3=G3_7, D=D1_7, d_steps=20, g_steps=20, d_fake_data=Xs1_7, d_real_data=Xt1_7, cuda_gpu=cuda_gpu,gpus=gpus, cls='S7')
    # torch.save(D1_7.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\D1_7.pth')
    # torch.save(G1_7.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G1_7.pth')
    # torch.save(G2_7.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G2_7.pth')
    # torch.save(G3_7.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G3_7.pth')
    # # ---第八类---
    # G1_8 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G2_8 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G3_8 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # D1_8 = Discriminator(input_size=200, hidden_size1=150, hidden_size2=80, hidden_size3=40, output_size=4)
    # GAN_train(Epoch=1000, G1=G1_8, G2=G2_8, G3=G3_8, D=D1_8, d_steps=20, g_steps=20, d_fake_data=Xs1_8, d_real_data=Xt1_8, cuda_gpu=cuda_gpu,gpus=gpus, cls='S8')
    # torch.save(D1_8.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\D1_8.pth')
    # torch.save(G1_8.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G1_8.pth')
    # torch.save(G2_8.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G2_8.pth')
    # torch.save(G3_8.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G3_8.pth')
    # # ---第九类---
    # G1_9 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G2_9 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G3_9 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # D1_9 = Discriminator(input_size=200, hidden_size1=150, hidden_size2=80, hidden_size3=40, output_size=4)
    # GAN_train(Epoch=1000, G1=G1_9, G2=G2_9, G3=G3_9, D=D1_9, d_steps=20, g_steps=20, d_fake_data=Xs1_9, d_real_data=Xt1_9, cuda_gpu=cuda_gpu,gpus=gpus, cls='S9')
    # torch.save(D1_9.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\D1_9.pth')
    # torch.save(G1_9.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G1_9.pth')
    # torch.save(G2_9.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G2_9.pth')
    # torch.save(G3_9.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G3_9.pth')
    # # ---第十类---
    # G1_10 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G2_10 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # G3_10 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    # D1_10 = Discriminator(input_size=200, hidden_size1=150, hidden_size2=80, hidden_size3=40, output_size=4)
    # GAN_train(Epoch=1000, G1=G1_10, G2=G2_10, G3=G3_10, D=D1_10, d_steps=20, g_steps=20, d_fake_data=Xs1_10, d_real_data=Xt1_10, cuda_gpu=cuda_gpu,gpus=gpus, cls='S10')
    # torch.save(D1_10.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\D1_10.pth')
    # torch.save(G1_10.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G1_10.pth')
    # torch.save(G2_10.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G2_10.pth')
    # torch.save(G3_10.state_dict(), 'E:\迁移学习\GAN_多源域对抗\同工况扩展数据集\MAD_GAN_parameter(3G)\参数HP0_1797\G3_10.pth')

    #--------模型加载--------
    G1_1 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G1_1.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G1_1.pth"))
    G1_1 = torch.nn.DataParallel(G1_1, device_ids=gpus).cuda()
    G2_1 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G2_1.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G2_1.pth"))
    G2_1 = torch.nn.DataParallel(G2_1, device_ids=gpus).cuda()
    G3_1 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G3_1.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G3_1.pth"))
    G3_1 = torch.nn.DataParallel(G3_1, device_ids=gpus).cuda()

    G1_2 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G1_2.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G1_2.pth"))
    G1_2 = torch.nn.DataParallel(G1_2, device_ids=gpus).cuda()
    G2_2 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G2_2.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G2_2.pth"))
    G2_2 = torch.nn.DataParallel(G2_2, device_ids=gpus).cuda()
    G3_2 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G3_2.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G3_2.pth"))
    G3_2 = torch.nn.DataParallel(G3_2, device_ids=gpus).cuda()

    G1_3 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G1_3.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G1_3.pth"))
    G1_3 = torch.nn.DataParallel(G1_3, device_ids=gpus).cuda()
    G2_3 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G2_3.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G2_3.pth"))
    G2_3 = torch.nn.DataParallel(G2_3, device_ids=gpus).cuda()
    G3_3 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G3_3.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G3_3.pth"))
    G3_3 = torch.nn.DataParallel(G3_3, device_ids=gpus).cuda()

    G1_4 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G1_4.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G1_4.pth"))
    G1_4 = torch.nn.DataParallel(G1_4, device_ids=gpus).cuda()
    G2_4 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G2_4.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G2_4.pth"))
    G2_4 = torch.nn.DataParallel(G2_4, device_ids=gpus).cuda()
    G3_4 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G3_4.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G3_4.pth"))
    G3_4 = torch.nn.DataParallel(G3_4, device_ids=gpus).cuda()

    G1_5 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G1_5.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G1_5.pth"))
    G1_5 = torch.nn.DataParallel(G1_5, device_ids=gpus).cuda()
    G2_5 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G2_5.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G2_5.pth"))
    G2_5 = torch.nn.DataParallel(G2_5, device_ids=gpus).cuda()
    G3_5 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G3_5.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G3_5.pth"))
    G3_5 = torch.nn.DataParallel(G3_5, device_ids=gpus).cuda()

    G1_6 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G1_6.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G1_6.pth"))
    G1_6 = torch.nn.DataParallel(G1_6, device_ids=gpus).cuda()
    G2_6 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G2_6.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G2_6.pth"))
    G2_6 = torch.nn.DataParallel(G2_6, device_ids=gpus).cuda()
    G3_6 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G3_6.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G3_6.pth"))
    G3_6 = torch.nn.DataParallel(G3_6, device_ids=gpus).cuda()

    G1_7 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G1_7.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G1_7.pth"))
    G1_7 = torch.nn.DataParallel(G1_7, device_ids=gpus).cuda()
    G2_7 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G2_7.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G2_7.pth"))
    G2_7 = torch.nn.DataParallel(G2_7, device_ids=gpus).cuda()
    G3_7 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G3_7.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G3_7.pth"))
    G3_7 = torch.nn.DataParallel(G3_7, device_ids=gpus).cuda()

    G1_8 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G1_8.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G1_8.pth"))
    G1_8 = torch.nn.DataParallel(G1_8, device_ids=gpus).cuda()
    G2_8 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G2_8.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G2_8.pth"))
    G2_8 = torch.nn.DataParallel(G2_8, device_ids=gpus).cuda()
    G3_8 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G3_8.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G3_8.pth"))
    G3_8 = torch.nn.DataParallel(G3_8, device_ids=gpus).cuda()

    G1_9 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G1_9.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G1_9.pth"))
    G1_9 = torch.nn.DataParallel(G1_9, device_ids=gpus).cuda()
    G2_9 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G2_9.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G2_9.pth"))
    G2_9 = torch.nn.DataParallel(G2_9, device_ids=gpus).cuda()
    G3_9 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G3_9.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G3_9.pth"))
    G3_9 = torch.nn.DataParallel(G3_9, device_ids=gpus).cuda()

    G1_10 = Generator1(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G1_10.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G1_10.pth"))
    G1_10 = torch.nn.DataParallel(G1_10, device_ids=gpus).cuda()
    G2_10 = Generator2(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G2_10.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G2_10.pth"))
    G2_10 = torch.nn.DataParallel(G2_10, device_ids=gpus).cuda()
    G3_10 = Generator3(input_size=200, hidden_size1=200, hidden_size2=200, hidden_size3=200, output_size=200)
    G3_10.load_state_dict(torch.load("CWRU\\CWRU_1797\\参数HP0_1797\\G3_10.pth"))
    G3_10 = torch.nn.DataParallel(G3_10, device_ids=gpus).cuda()

    #--------形成数据--------
    noise2=0.1*torch.tensor(np.random.random((300,200)),dtype=torch.float32)
    Xs1,Xs2,Xs3 = Variable(torch.Tensor(train_x[0:300, 0:200]).cuda()),Variable(torch.Tensor(train_x[300:600, 0:200]).cuda()),Variable(torch.Tensor(train_x[600:900, 0:200]).cuda())
    Xs4,Xs5,Xs6 = Variable(torch.Tensor(train_x[900:1200, 0:200]).cuda()),Variable(torch.Tensor(train_x[1200:1500, 0:200]).cuda()),Variable(torch.Tensor(train_x[1500:1800, 0:200]).cuda())
    Xs7,Xs8,Xs9,Xs10 = Variable(torch.Tensor(train_x[1800:2100, 0:200]).cuda()),Variable(torch.Tensor(train_x[2100:2400, 0:200]).cuda()),Variable(torch.Tensor(train_x[2400:2700, 0:200]).cuda()),Variable(torch.Tensor(train_x[2700:3000, 0:200]).cuda())

    print('Xs1',Xs1.shape)

    Xs1_1_fake, Xs1_2_fake, Xs1_3_fake, Xs1_4_fake, Xs1_5_fake  = G1_1(Xs1 + noise2.cuda()), G1_2(Xs2 + noise2.cuda()), G1_3(Xs3 + noise2.cuda()), G1_4(Xs4 + noise2.cuda()), G1_5(Xs5 + noise2.cuda())
    Xs1_6_fake, Xs1_7_fake, Xs1_8_fake, Xs1_9_fake, Xs1_10_fake = G1_6(Xs6 + noise2.cuda()), G1_7(Xs7 + noise2.cuda()), G1_8(Xs8 + noise2.cuda()), G1_9(Xs9 + noise2.cuda()), G1_10(Xs10 + noise2.cuda())

    Xs2_1_fake, Xs2_2_fake, Xs2_3_fake, Xs2_4_fake, Xs2_5_fake = G2_1(Xs1 + noise2.cuda()), G2_2(Xs2 + noise2.cuda()), G2_3(Xs3 + noise2.cuda()), G2_4(Xs4 + noise2.cuda()), G2_5(Xs5 + noise2.cuda())
    Xs2_6_fake, Xs2_7_fake, Xs2_8_fake, Xs2_9_fake, Xs2_10_fake = G2_6(Xs6 + noise2.cuda()), G2_7(Xs7 + noise2.cuda()), G2_8(Xs8 + noise2.cuda()), G2_9(Xs9 + noise2.cuda()), G2_10(Xs10 + noise2.cuda())

    Xs3_1_fake, Xs3_2_fake, Xs3_3_fake, Xs3_4_fake, Xs3_5_fake = G3_1(Xs1 + noise2.cuda()), G3_2(Xs2 + noise2.cuda()), G3_3(Xs3 + noise2.cuda()), G3_4(Xs4 + noise2.cuda()), G3_5(Xs5 + noise2.cuda())
    Xs3_6_fake, Xs3_7_fake, Xs3_8_fake, Xs3_9_fake, Xs3_10_fake = G3_6(Xs6 + noise2.cuda()), G3_7(Xs7 + noise2.cuda()), G3_8(Xs8 + noise2.cuda()), G3_9(Xs9 + noise2.cuda()), G3_10(Xs10 + noise2.cuda())

    #-------合并数据集-------
    Xs_fake = torch.cat([Xs1_1_fake[0:100,:],Xs2_1_fake[0:100,:],Xs3_1_fake[0:100,:],Xs1_2_fake[0:100,:],Xs2_2_fake[0:100,:],Xs3_2_fake[0:100,:],Xs1_3_fake[0:100,:],Xs2_3_fake[0:100,:],Xs3_3_fake[0:100,:],Xs1_4_fake[0:100,:],Xs2_4_fake[0:100,:],Xs3_4_fake[0:100,:],
                         Xs1_5_fake[0:100,:],Xs2_5_fake[0:100,:],Xs3_5_fake[0:100,:],Xs1_6_fake[0:100,:],Xs2_6_fake[0:100,:],Xs3_6_fake[0:100,:],Xs1_7_fake[0:100,:],Xs2_7_fake[0:100,:],Xs3_7_fake[0:100,:],Xs1_8_fake[0:100,:],Xs2_8_fake[0:100,:],Xs3_8_fake[0:100,:],
                         Xs1_9_fake[0:100,:],Xs2_9_fake[0:100,:],Xs3_9_fake[0:100,:],Xs1_10_fake[0:100,:],Xs2_10_fake[0:100,:],Xs3_10_fake[0:100,:],], 0)
    Xs_fake1 = Xs_fake.cpu().detach().numpy()
    train_y1 = train_y.view(3000,1).detach().numpy()
    io.savemat('CWRU\\CWRU_1797\\生成数据CWRU_1797\\GenerationData_1797_HP0_DE_FFT_Tanh3_sigmoid_5.mat', {'signal_FFT': Xs_fake1,'labels':train_y1})

    #-------分开数据集-------
    Xs_fake_G1 = torch.cat([Xs1_1_fake, Xs1_2_fake, Xs1_3_fake,Xs1_4_fake, Xs1_5_fake, Xs1_6_fake,Xs1_7_fake, Xs1_8_fake, Xs1_9_fake, Xs1_10_fake], 0)
    Xs_fake1 = Xs_fake_G1.cpu().detach().numpy()
    train_y1 = train_y.view(3000,1).detach().numpy()
    io.savemat('CWRU\\CWRU_1797\\生成数据CWRU_1797\\GenerationData_G1_1797_HP0_DE_FFT_Tanh3_sigmoid_5.mat', {'signal_FFT': Xs_fake1,'labels':train_y1})

    #-------分开数据集-------
    Xs_fake_G2 = torch.cat([Xs2_1_fake, Xs2_2_fake, Xs2_3_fake,Xs2_4_fake, Xs2_5_fake, Xs2_6_fake,Xs2_7_fake, Xs2_8_fake, Xs2_9_fake, Xs2_10_fake], 0)
    Xs_fake2 = Xs_fake_G2.cpu().detach().numpy()
    train_y2 = train_y.view(3000,1).detach().numpy()
    io.savemat('CWRU\\CWRU_1797\\生成数据CWRU_1797\\GenerationData_G2_1797_HP0_DE_FFT_Tanh3_sigmoid_5.mat', {'signal_FFT': Xs_fake2,'labels':train_y2})

    #-------分开数据集-------
    Xs_fake_G3 = torch.cat([Xs3_1_fake, Xs3_2_fake, Xs3_3_fake,Xs3_4_fake, Xs3_5_fake, Xs3_6_fake,Xs3_7_fake, Xs3_8_fake, Xs3_9_fake, Xs3_10_fake], 0)
    Xs_fake3 = Xs_fake_G3.cpu().detach().numpy()
    train_y3 = train_y.view(3000,1).detach().numpy()
    io.savemat('CWRU\\CWRU_1797\\生成数据CWRU_1797\\GenerationData_G3_1797_HP0_DE_FFT_Tanh3_sigmoid_5.mat', {'signal_FFT': Xs_fake3,'labels':train_y3})

