import numpy as np
import scipy.sparse as sp
import os
from scipy.io import loadmat
import pickle

def get_data_info(data_path,dataset):
    '''
    获取数据信息
    '''
    DATA_PATH = os.path.join(data_path,dataset,"images")

    if dataset == "Paris6k":
        DATA_path_list = []
        for dir_ in os.listdir(DATA_PATH):
            DATA_path_list.extend([os.path.join(DATA_PATH,dir_,item) for item in os.listdir(os.path.join(DATA_PATH,dir_))])
    elif dataset == "Oxford5k":
        DATA_path_list = [os.path.join(DATA_PATH,item) for item in os.listdir(DATA_PATH)]

    DATA_file_list = [os.path.split(item)[1].split(".")[0] for item in DATA_path_list]
    return DATA_PATH, DATA_file_list


def get_data(data_path,dataset):
    '''
    获得数据
    '''
    features_path = os.path.join(data_path,dataset, 'features', '{}_resnet_rsfm120k_gem.mat'.format(dataset))
    ground_true_path = os.path.join(data_path,dataset, 'ground_true','revisited','gnd_' + dataset + '.pkl')

    features = loadmat(features_path)
    Q = features["Q"].T
    X = features["X"].T
    with open(ground_true_path, 'rb') as f:
        gt = pickle.load(f)
    return Q,X,gt

def get_gt(gt):
  '''
  获取ground_true标注
  '''
  gnd = gt['gnd']
  gnd_total = {}

  gnd_E = []
  for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['easy']])
    g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
    gnd_E.append(g)
  gnd_total['E'] = gnd_E

  gnd_M = []
  for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
    g['junk'] = np.concatenate([gnd[i]['junk']])
    gnd_M.append(g)
  gnd_total['M'] = gnd_M

  gnd_H = []
  for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['hard']])
    g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
    gnd_H.append(g)
  gnd_total['H'] = gnd_H
  return gnd_total