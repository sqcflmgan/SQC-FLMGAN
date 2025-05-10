import torch.utils.data as Data
import os
import torch, torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import pandas as pd

#-------------------------------------------------------------------------------------------------------
# DATASETS 数据集
#-------------------------------------------------------------------------------------------------------
os.environ['TRAINING_DATA']="date"
DATA_PATH = os.path.join(os.environ['TRAINING_DATA'], 'PyTorch')

def print_table_data_stats(data_train, labels_train, data_test, labels_test):
  print("训练数据标签数据的大小，特征范围，标签范围")
  print("开始加载Data: ")
  print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
    np.min(labels_train), np.max(labels_train)))
  print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
    np.min(labels_test), np.max(labels_test)))

def print_image_data_stats(data_train, labels_train, data_test, labels_test):
  print("训练数据标签数据的大小，特征范围，标签范围")
  print("开始加载Data: ")
  print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
      np.min(labels_train), np.max(labels_train)))
  # 获取non-iid数据是要排序获取数据  获取iid数据是打乱数据集获取数据
  print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
    data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
      np.min(labels_test), np.max(labels_test)))

#-------------------------------------------------------------------------------------------------------
# SPLIT DATA AMONG CLIENTS 划分数据
#-------------------------------------------------------------------------------------------------------
def split_image_data(data, labels, n_clients=10, classes_per_client=10, shuffle=True, verbose=True, balancedness=None):

  n_data = data.shape[0]
  n_labels = np.max(labels) + 1
  if balancedness == 1.0:
    data_per_client = [n_data // n_clients]*n_clients
    data_per_client_per_class = [data_per_client[0] // classes_per_client]*n_clients
  else:
    fracs = balancedness**np.linspace(0,n_clients-1, n_clients)
    fracs /= np.sum(fracs)
    fracs = 0.1/n_clients + (1-0.1)*fracs
    data_per_client = [np.floor(frac*n_data).astype('int') for frac in fracs]
    data_per_client = data_per_client[::-1]
    data_per_client_per_class = [np.maximum(1,nd // classes_per_client) for nd in data_per_client]
  if sum(data_per_client) > n_data:
    print("Impossible Split 此时因为分配的客户端大于样本数，失败")
    exit()
  
  # sort for labels 对标签排序 就是想获取
  data_idcs = [[] for i in range(n_labels)]
  for j, label in enumerate(labels):
    data_idcs[label] += [j]

  if shuffle:
    for idcs in data_idcs:
      np.random.shuffle(idcs)

  # split data among clients 在客户端之间分割数据
  clients_split = []
  c = 0
  print("n_clients",n_clients)
  for i in range(n_clients):
    client_idcs = []
    budget = data_per_client[i] # budget=30
    c = np.random.randint(n_labels)
    while budget > 0:
      take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)

      client_idcs += data_idcs[c][:take]
      data_idcs[c] = data_idcs[c][take:]
      
      budget -= take
      c = (c + 1) % n_labels
    clients_split += [(data[client_idcs], labels[client_idcs])]
  print("clients_split第一个客户端的标签", clients_split[0][1])

  def print_split(clients_split):
    print("数据划分")
    print("Data split:")
    for i, client in enumerate(clients_split):
      split = np.sum(client[1].reshape(1,-1)==np.arange(n_labels).reshape(-1,1), axis=1)
      print(" - Client {}: {}".format(i,split))
    print()
  if verbose:
    print_split(clients_split)
  return clients_split


def split_table_data(data, labels, n_clients=10, classes_per_client=10, shuffle=True, verbose=True, balancedness=None):
  n_data = data.shape[0]
  n_labels = np.max(labels) + 1

  # n_labels = torch.max(labels).item() + 1

  # data_per_client = 240
  # data_per_client_per_class = 48

  if balancedness == 1.0:
    data_per_client = [n_data // n_clients] * n_clients
    print('data_per_client = [n_data // n_clients] * n_clients',data_per_client,n_data, n_clients, n_clients)
    data_per_client_per_class = [data_per_client[0] // classes_per_client] * n_clients
    print('data_per_client_per_class = [data_per_client[0] // classes_per_client] * n_clients',data_per_client_per_class,
          data_per_client[0],classes_per_client,n_clients)
  else:
    fracs = balancedness ** np.linspace(0, n_clients - 1, n_clients)
    fracs /= np.sum(fracs)
    fracs = 0.1 / n_clients + (1 - 0.1) * fracs
    data_per_client = [np.floor(frac * n_data).astype('int') for frac in fracs]
    data_per_client = data_per_client[::-1]
    data_per_client_per_class = [np.maximum(1, nd // classes_per_client) for nd in data_per_client]

  if sum(data_per_client) > n_data:
    print("Impossible Split: The allocated clients exceed the number of samples")
    exit()

  # Sort for labels
  data_idcs = [[] for _ in range(n_labels)]
  for j, label in enumerate(labels):
    data_idcs[label] += [j]

  if shuffle:
    for idcs in data_idcs:
      np.random.shuffle(idcs)

  # Split data among clients
  clients_split = []
  c = 0
  print("n_clients:", n_clients)
  for i in range(n_clients):
    client_idcs = []
    budget = data_per_client[i]
    c = np.random.randint(n_labels)
    while budget > 0:
      take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)
      client_idcs += data_idcs[c][:take]
      data_idcs[c] = data_idcs[c][take:]
      budget -= take
      c = (c + 1) % n_labels
    clients_split += [(data[client_idcs], labels[client_idcs])]
    # clients_split += [(data.iloc[client_idcs], labels.iloc[client_idcs])]
  print("Labels of the first client in clients_split:", clients_split[0][1])
  # split1 = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)

  def print_split(clients_split):
    print("Data split:")
    print("客户端真实数据数量：")
    split_info = []
    split_diff = []
    for i, client in enumerate(clients_split):
      split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
      # split = np.sum(np.array(client[1]).reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
      total_samples = np.sum(split)  # 计算样本总量
      num_classes = np.count_nonzero(split)  # 计算具有几类样本
      print(f"sum:{total_samples} , class:{num_classes},", " - Client {}: {}".format(i, split))
      split_info.append((i, total_samples, num_classes))# 将 total_samples 和 num_classes 添加到数组中
    return split_info

  if verbose:
    split_info = print_split(clients_split)

  return clients_split,split_info
#-------------------------------------------------------------------------------------------------------
# TABLE DATASET CLASS 图像数据集类
#-------------------------------------------------------------------------------------------------------
class CustomDataset(Dataset):
  """
  A custom Dataset class for table data
  inputs: numpy array [n_data x n_features]
  labels: numpy array [n_data]
  """

  def __init__(self, inputs, labels):
    assert inputs.shape[0] == labels.shape[0]
    self.inputs = torch.Tensor(inputs)
    self.labels = torch.Tensor(labels).long()

  def __getitem__(self, index):
    data, label = self.inputs[index], self.labels[index]
    return (data, label)

  def __len__(self):
    return self.inputs.shape[0]
# -------------------------------------------------------------------------------------------------------
# IMAGE DATASET CLASS 图像数据集类
# -------------------------------------------------------------------------------------------------------
class CustomImageDataset(Dataset):
  '''
  A custom Dataset class for images
  inputs : numpy array [n_data x shape]
  labels : numpy array [n_data (x 1)]
  '''
  def __init__(self, inputs, labels, transforms=None):
      assert inputs.shape[0] == labels.shape[0]
      self.inputs = torch.Tensor(inputs)
      self.labels = torch.Tensor(labels).long()
      self.transforms = transforms

  def __getitem__(self, index):
      img, label = self.inputs[index], self.labels[index]

      if self.transforms is not None:
        img = self.transforms(img)

      return (img, label)

  def __len__(self):
      return self.inputs.shape[0]

def get_data_loaders(hp,clients_split = [], verbose=True,):

  # # -------------------------------------CWRU-数据-----------------
  data = pd.read_excel('MAD-GAN\\CWRU\\CWRU_1797\\train_data_7200.xlsx')
  x_train = data.iloc[:, :-1].values
  y_train = data.iloc[:, -1].values
  data1 = pd.read_excel('MAD-GAN\\CWRU\\CWRU_1797\\test_data_1800.xlsx')
  x_test = data1.iloc[:, :-1].values
  y_test = data1.iloc[:, -1].values


  print("x_train.shape",x_train.shape)
  print("y_train.shape",y_train.shape)

  split,split_info= split_table_data(x_train, y_train, n_clients=hp['n_clients'],
          classes_per_client=hp['classes_per_client'], balancedness=hp['balancedness'],verbose=verbose)

  # 划分好数据  客户端进行下载
  client_loaders = [torch.utils.data.DataLoader(CustomDataset(x, y),
                                                batch_size=hp['batch_size'], shuffle=True) for x, y in split]

  train_loader = torch.utils.data.DataLoader(CustomDataset(x_train, y_train), batch_size=100, shuffle=True)
  test_loader = torch.utils.data.DataLoader(CustomDataset(x_test, y_test), batch_size=100, shuffle=False)

  stats = {"split" : [x.shape[0] for x, y in split]}

  print("stats",stats)
  print("stats的长度", len(stats))
  print("split_info", split_info)
  return client_loaders, train_loader, test_loader, stats , split, split_info






