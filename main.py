from graph import *
from data import *
from evaluate import *
from training import *
import setting

from matplotlib import pyplot as plt
import tensorflow as tf
import scipy.sparse as sp
import numpy as np
import time

args = setting.args

# 导入数据
DATA_PATH, DATA_file_list = get_data_info(args.data_path,args.dataset)
Q,X,gt = get_data(args.data_path,args.dataset)

# 拼接数据
all_features = np.concatenate([Q, X])
# 训练数据整理
training_dataset = tf.data.Dataset.from_tensor_slices(X).batch(X.shape[0])
validation_dataset = tf.data.Dataset.from_tensor_slices(all_features).batch(all_features.shape[0])
# 获得ground_true
gnd_total = get_gt(gt)

# 图构建
# 生成图
if args.pre_graph:
  q_RANSAC_graph = np.load(os.path.join(args.data_path,args.dataset,
                'features','{}_query_ransac_graph.npy'.format(args.dataset)))
  x_RANSAC_graph = np.load(os.path.join(args.data_path,args.dataset,
                'features','{}_index_ransac_graph.npy'.format(args.dataset)))
  all_adj, x_adj = generate_graph(Q,X,args,q_RANSAC_graph,x_RANSAC_graph)
else:
  all_adj, x_adj = generate_graph(Q,X,args)

# 正则化图
all_adj_normed = preprocess_graph(all_adj)
x_adj_normed = preprocess_graph(x_adj)

# 建立mask
all_mask_coo = all_adj_normed.tocoo() # 包含自连
all_mask_indices = np.array([all_mask_coo.row, all_mask_coo.col]).transpose()
x_mask_coo = x_adj_normed.tocoo()
x_mask_indices = np.array([x_mask_coo.row, x_mask_coo.col]).transpose()

# 转换为tf格式/神经网络可用的格式
x_adj_normed_sparse_tensor = convert_sparse_matrix_to_sparse_tensor(x_adj_normed)
all_adj_normed_sparse_tensor = convert_sparse_matrix_to_sparse_tensor(all_adj_normed)

# # 评价测试
# start_time = time.time()
# protocols = quick_get_mAP(Q,X,gnd_total)
# for protocol in protocols:
#   print(str(round(protocol,5)))
# end_time = time.time() - start_time
# print(end_time)

# 模型参数配置
args.log_name = 'strandard'

# 模型生成
MyModel,optimizer,summary_writer = create_model(args)

# 初始化
init_model(MyModel,summary_writer,
           training_dataset,validation_dataset,
           x_adj_normed_sparse_tensor,all_adj_normed_sparse_tensor,
           x_mask_indices,all_mask_indices,
           gnd_total,args)
print(args)

# 训练
training_model(MyModel,optimizer,summary_writer,
        training_dataset,validation_dataset,
        x_adj_normed_sparse_tensor,all_adj_normed_sparse_tensor,
        x_mask_indices,all_mask_indices,
        gnd_total,args,print_log=True)