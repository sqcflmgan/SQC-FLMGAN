#!/usr/bin/python3
# coding=utf-8
import socket
import pickle
import json
import math
import socket
import pickle
import struct  # 用于处理固定长度的数据（如数据长度）
import time
import os
import argparse
import random
import numpy as np
import torch, torchvision
import torch.optim as optim
import data_utils  # 文件夹中第二个py文件
import neural_nets  # 文件夹中倒数第二个py文件
import distributed_training_utils as dst
from distributed_training_utils import Client, Server  # 文件夹中第四个py文件
import experiment_manager as xpm  # 文件夹中第五个py文件
import default_hyperparameters as dhp  # 文件夹中第三个py文件

import copy
import urllib3
http = urllib3.PoolManager(num_pools=50)
# 创建一个解析器（使用argparse的第一步就是创建一个ArgumentParser对象）
parser = argparse.ArgumentParser()
# 添加程序参数信息是通过调用 add_argument() 方法完成的
parser.add_argument("--schedule", default="main", type=str)  # 指定运行哪一批实验
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=None, type=int)
parser.add_argument("--reverse_order", default=False, type=bool)

# 1.10+cpu,申明Torch的版本
print("Torch Version: ", torch.__version__)
# 判断你电脑的GPU 能否被PyTorch 调用，能否完成GPU加速
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.cuda.set_device(1)
# 判断当前设备是否为第二块 GPU
if torch.cuda.current_device() == 1:
    print("Current device is set to GPU 1.")
print(torch.cuda.device_count())  # 查看CDUD可用数量
print(torch.cuda.is_available())  #
# 这些信息在 parse_args() 调用时被存储和使用
args = parser.parse_args()

# Load the Hyperparameters of all Experiments to be performed and set up the Experiments
# 加载所有要执行的实验的超参数，并设置实验
with open('federated_learning.json') as data_file:
    experiments_raw = json.load(data_file)[args.schedule]

hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(x)][args.start:args.end]
if args.reverse_order:
    # 翻转读取
    hp_dicts = hp_dicts[::-1]
experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]

# 读取实验参数，不懂的是
def run_experiments(experiments):
    print("Running {} Experiments..\n".format(len(experiments)))
    for xp_count, xp in enumerate(experiments):
        hp = dhp.get_hp(xp.hyperparameters)
        xp.prepare(hp)
        print(xp)
        clients_split = []
        client_loaders, train_loader, test_loader, stats , split , split_info = data_utils.get_data_loaders(hp)
        # client_loaders, train_loader, test_loader, stats = data_utils.get_data_loaders(hp)
        print("stats",stats)

        # 打印的东西是个对象
        print("client_loaders=", len(client_loaders))
        # Instantiate Clients and Server with Neural Net 用神经网络实例化客户端和服务器
        net = getattr(neural_nets, hp['net'])
        clients = [Client(loader, net().to(device), hp, xp, id_num=i) for i, loader in enumerate(client_loaders)]
        print("clients=", len(clients))
        server = Server(test_loader, net().to(device), hp, xp, stats)

        print("server", server)
        # Print optimizer specs 打印优化器规格
        print_model(device=clients[0])
        print_optimizer(device=clients[0])
        # Start Distributed Training Process 启动分布式训练流程
        print("Start Distributed Training..\n")
        t1 = time.time()

        for c_round in range(1, hp['communication_rounds'] + 1):
            # 随机选择1000个客户端进行培训
            participating_clients = random.sample(clients, int(len(clients) * hp['participation_rate']))
            participating_clients_id = [client.id for client in participating_clients]  # 获取参与训练客户端的id
            clients_data = []
            for client in participating_clients:
                # 客户端从服务器下载当前最新模型参数W(与服务器同步参数W)
                global_W=client.synchronize_with_server(server)
                client.compute_weight_update(hp['local_iterations'],c_round=c_round,global_W=global_W)
                client.compress_weight_update_up(compression=hp['compression_up'], accumulate=hp['accumulation_up'],count_bits=hp["count_bits"])
                # 多机部署时，实现客户端参数发送并赋值给server.dw
                # server.dW = client_send(client)
            server.aggregate_weight_updates(participating_clients, aggregation=hp['aggregation'])
            server.compress_weight_update_down(compression=hp['compression_down'], accumulate=hp['accumulation_down'],count_bits=hp["count_bits"])

            if xp.is_log_round(c_round):
                # 在第一次和 c_round 时执行的代码
                print("-----------------------------------------------------------------------------")
                print("Experiment: {} ({}/{})".format(args.schedule, xp_count + 1, len(experiments)))
                print("Evaluate...")
                results_train = server.evaluate(max_samples=10000, loader=train_loader)
                results_test = server.evaluate(max_samples=10000)

                xp.log({'communication_round': c_round, 'lr': clients[0].optimizer.__dict__['param_groups'][0]['lr'],
                        'epoch': clients[0].epoch, 'iteration': c_round * hp['local_iterations']})

                xp.log({key + '_train': value for key, value in results_train.items()})
                xp.log({key + '_test': value for key, value in results_test.items()})

                if hp["count_bits"]:
                    xp.log({'bits_sent_up_client0': sum(participating_clients[0].bits_sent),
                            'bits_sent_down_server': sum(server.bits_sent)}, printout=True)

                xp.log({'time': time.time() - t1}, printout=False)

                # Save results to Disk
                if 'log_path' in hp and hp['log_path']:
                    xp.save_to_disc(path=hp['log_path'])

                # Timing
                total_time = time.time() - t1
                avrg_time_per_c_round = (total_time) / c_round
                e = int(avrg_time_per_c_round * (hp['communication_rounds'] - c_round))
                print("Remaining Time (approx.):", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60),
                      "[{:.2f}%]\n".format(c_round / hp['communication_rounds'] * 100))

        del server
        clients.clear()
        torch.cuda.empty_cache()

def print_optimizer(device):
    try:
        print("Optimizer:", device.hp['optimizer'])
        for key, value in device.optimizer.__dict__['defaults'].items():
            print(" -", key, ":", value)

        hp = device.hp
        base_batchsize = hp['batch_size']
        if hp['fix_batchsize']:
            client_batchsize = base_batchsize // hp['n_clients']
        else:
            client_batchsize = base_batchsize
        total_batchsize = client_batchsize * hp['n_clients']
        print(" - batchsize (/ total): {} (/ {})".format(client_batchsize, total_batchsize))
        print()
    except:
        pass

#------------分布式部署，根据自身配置host 以及port----------
#------采用多机部署时，需根据客户端数量配置各客户端的训练数据集---
def client_send(client, host='localhost', port=65432):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        # 准备要发送的数据
        data = client.dW_compressed
        serialized_data = pickle.dumps(data)
        # 发送数据长度（使用struct模块来将整数转换为固定长度的字节）
        data_length = len(serialized_data)
        s.sendall(struct.pack('>I', data_length))  # '>I' 表示无符号整数（4字节），大端字节序
        # 发送实际的数据
        s.sendall(serialized_data)
        print("client1模型权重发送成功")
        # 接收服务器发送的数据长度
        response_length_bytes = s.recv(4)  # 服务器发送4字节来表示数据长度
        if len(response_length_bytes) != 4:
            raise ConnectionError("接收响应长度时出错，连接可能已中断")
        response_length = struct.unpack('>I', response_length_bytes)[0]
        # 根据接收到的长度来接收实际的数据
        buffer = b''
        bytes_recd = 0
        while bytes_recd < response_length:
            chunk = s.recv(min(response_length - bytes_recd, 4096))  # 每次最多接收4096字节
            if not chunk:
                raise ConnectionError("接收响应数据时出错，连接可能已中断")
            buffer += chunk
            bytes_recd += len(chunk)
        # 反序列化接收到的数据
        response_data = pickle.loads(buffer)
        # 打印反序列化后的数据
        # print("接收到的服务器数据：", response_data)
        print(f"成功接收到全局模型参数")
        return response_data

def print_model(device):
    print("Model {}:".format(device.hp['net']))
    n = 0
    for key, value in device.model.named_parameters():
        print(' -', '{:30}'.format(key), list(value.shape))
        n += value.numel()
    print("Total number of Parameters: ", n)
    print()

if __name__ == "__main__":
    run_experiments(experiments)
