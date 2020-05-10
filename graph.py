import numpy as np
import os
import scipy.sparse as sp
import tensorflow as tf

def preprocess_graph(adj):
  '''
  标准化图
  '''
  adj_ = adj + sp.eye(adj.shape[0])
  rowsum = np.array(adj_.sum(1))
  degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten()) #(m,m)维对角矩阵
  adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).transpose()
  return sp.csr_matrix(adj_normalized)

def convert_sparse_matrix_to_sparse_tensor(X):
  '''
  将稀疏图转化为tf形式
  '''
  coo = X.tocoo()  # 以坐标形式存储稀疏矩阵
  indices = np.mat([coo.row, coo.col]).transpose()  # 导出非零位置的坐标
  return tf.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)

def generate_graph(Q,X,args,q_RANSAC_graph=None,x_RANSAC_graph=None):
  # 生成query的图
  q_sim = np.matmul(Q, X.T) # 相似向量
  if q_RANSAC_graph is not None:
    q_sim_top = q_RANSAC_graph[:, 0:args['kq']]
  else:
    q_sim_top = np.argpartition(q_sim, -args['kq'], 1)[:,-args['kq']:] # 找到每个query最近kq个点，不排序
  q_adj = np.zeros(q_sim.shape)
  for i in range(q_adj.shape[0]): # 只记录最邻近向量的相似值
    q_adj[i,q_sim_top[i]] = q_sim[i,q_sim_top[i]]
  q_adj = sp.csr_matrix(q_adj) # 转为稀疏数据

  # 生成数据库的图
  x_sim = np.matmul(X, X.T)
  if x_RANSAC_graph is not None:
    x_sim_top = x_RANSAC_graph[:, 0:args['k']]
  else:
    x_sim_top = np.argpartition(x_sim, -args['k'], 1)[:, -args['k']:]
  x_adj = np.zeros(x_sim.shape)
  for i in range(x_adj.shape[0]):
    x_adj[i, x_sim_top[i]] = x_sim[i, x_sim_top[i]]
    x_adj[x_sim_top[i], i] = x_sim[i, x_sim_top[i]]
    x_adj[i, i] = 0
  x_adj = sp.csr_matrix(x_adj)

  # 拼接稀疏图
  all_adj = sp.vstack((q_adj, x_adj))  #(mq+mx,mx)
  zeros = sp.csr_matrix((all_adj.shape[0], q_adj.shape[0]))  #(mq+mx,mq)
  all_adj = sp.hstack((zeros, all_adj))  #(mq+mx,mx+mq) 要凑成方阵，便于之后计算
  all_adj = sp.csr_matrix(all_adj)
  return all_adj, x_adj