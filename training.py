import numpy as np
import os
import tensorflow as tf
from model import *
from evaluate import quick_get_mAP
import time

def get_TV(hidden_emb,mask):
  hidden_emb=np.array(hidden_emb)
  first_node = hidden_emb[mask[:,0]]
  second_node = hidden_emb[mask[:,1]]
  TV = ((first_node-second_node)**2).sum()
  return TV

def update_beta(model, datas, adj_normed_sparse_tensor,args):
  hidden_emb = model(datas, adj_normed_sparse_tensor,training=False)
  hidden_sim = tf.matmul(hidden_emb, tf.transpose(hidden_emb))
  beta_score = np.percentile(tf.reshape(hidden_sim, [-1]),args.beta_percentile)
  return beta_score

# 训练步骤
# @tf.function 
def train_step(model,datas,adj_normed_sparse_tensor,optimizer,x_mask_indices,args):
  with tf.GradientTape() as tape:
    hidden_emb = model(datas, adj_normed_sparse_tensor,training=True)  #training用于类似于dropout层，这里可以不加
    if args.record_all:
      hidden_emb = hidden_emb[-1]
    hidden_sim = tf.matmul(hidden_emb, tf.transpose(hidden_emb))  # 求各结点相似性
    TV_train = get_TV(hidden_emb,x_mask_indices)

    logits = tf.nn.relu(hidden_sim)
    loss = GSS_loss(logits, args.alpha, args.beta)
    loss += sum(model.losses)
    
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss,TV_train

# 测试步骤
def test_step(model,datas,adj_normed_sparse_tensor,all_mask_indices,gnd_total,args,feature_id=-1):
  hidden_emb = model(datas, adj_normed_sparse_tensor, training=False)
  if args.record_all:
    hidden_emb = hidden_emb[feature_id]
  TV_test = get_TV(hidden_emb, all_mask_indices)
  predictions = hidden_emb.numpy()
  Q_ = predictions[:args.query_num]
  X_ = predictions[args.query_num:]
  total_protocol = quick_get_mAP(Q_,X_,gnd_total)
  return total_protocol,TV_test

# 初始更新步骤
def init_model(model,summary_writer,
        training_dataset,validation_dataset,
        x_adj_normed_sparse_tensor,all_adj_normed_sparse_tensor,
        x_mask_indices,all_mask_indices,
        gnd_total,args,hparams=None):
  for training_data in training_dataset:
    # 获取初始beta
    hidden_emb = model(training_data, x_adj_normed_sparse_tensor,training=False)
    if args.record_all:
      hidden_emb = hidden_emb[-1]
    TV_train = get_TV(hidden_emb,x_mask_indices)
    hidden_sim = tf.matmul(hidden_emb, tf.transpose(hidden_emb))
    args.beta = np.percentile(tf.reshape(hidden_sim, [-1]),args.beta_percentile)
    # 计算损失
    logits = tf.nn.relu(hidden_sim)
    loss = GSS_loss(logits, args.alpha, args.beta)
    loss += sum(model.losses)
    
  
  for testing_data in validation_dataset:
    total_protocol,TV_test = test_step(model,testing_data, 
                      all_adj_normed_sparse_tensor,all_mask_indices,
                      gnd_total,args)

  with summary_writer.as_default():
    if hparams:
      hp.hparams(hparams)
    tf.summary.scalar('loss_train', loss, step=0)
    tf.summary.scalar('beta', args.beta, step=0)
    tf.summary.scalar('TV_train', TV_train, step=0)
    tf.summary.scalar('TV_test', TV_test, step=0)
    tf.summary.scalar('mAP_E', total_protocol[0], step=0)
    tf.summary.scalar('mAP_M', total_protocol[1], step=0)
    tf.summary.scalar('mAP_H', total_protocol[2], step=0)
    args.best_mAP_E = total_protocol[0]
    args.best_mAP_M = total_protocol[1]
    args.best_mAP_H = total_protocol[2]

    for i in range(len(model.trainable_variables)):
      tf.summary.histogram(model.trainable_variables[i].name,model.trainable_variables[i],step=0)

def training_model(model,optimizer,summary_writer,
          training_dataset,validation_dataset,
          x_adj_normed_sparse_tensor,all_adj_normed_sparse_tensor,
          x_mask_indices,all_mask_indices,
          gnd_total,args,print_log=False,star_epoch=1):
  total_start_time = start_time = time.time()
  for epoch in range(star_epoch,args.epochs+star_epoch):  
    # 训练模型
    for training_data in training_dataset:
      # 数据噪声
      if args.train_noise:
        data_mean = np.mean(X)
        data_var = np.var(X)
        data_noise = np.random.randn(training_data.shape[0],training_data.shape[1])*data_var+data_mean
        training_data = training_data + data_noise*args.train_noise
      loss_train,TV_train = train_step(model, training_data, x_adj_normed_sparse_tensor, 
                        optimizer, x_mask_indices, args)
    # 测试模型
    if epoch %10 ==0 or epoch<10:
      train_time = time.time() - start_time
      for testing_data in validation_dataset:
        total_protocol,TV_test = test_step(model, testing_data, 
                          all_adj_normed_sparse_tensor, all_mask_indices,
                          gnd_total, args)
      test_time = time.time() - start_time - train_time
      # tensorboard记录
      if total_protocol[0]>args.best_mAP_E:
        args.best_mAP_E=total_protocol[0]
      if total_protocol[1]>args.best_mAP_M:
        args.best_mAP_M=total_protocol[1]
      if total_protocol[2]>args.best_mAP_H:
        args.best_mAP_H=total_protocol[2]
      with summary_writer.as_default():
        tf.summary.scalar('loss_train', loss_train, step=epoch)
        tf.summary.scalar('beta', args.beta, step=epoch)
        tf.summary.scalar('TV_train', TV_train, step=epoch)
        tf.summary.scalar('TV_test', TV_test, step=epoch)
        tf.summary.scalar('mAP_E', total_protocol[0], step=epoch)
        tf.summary.scalar('mAP_M', total_protocol[1], step=epoch)
        tf.summary.scalar('mAP_H', total_protocol[2], step=epoch)
        tf.summary.scalar('best_mAP_E', args.best_mAP_E, step=epoch)
        tf.summary.scalar('best_mAP_M', args.best_mAP_M, step=epoch)
        tf.summary.scalar('best_mAP_H', args.best_mAP_H, step=epoch)

        for i in range(len(model.trainable_variables)):
          tf.summary.histogram(model.trainable_variables[i].name,model.trainable_variables[i],step=epoch)
      if print_log:
        print("Epoch: {}, Train Time: {:.4}, Test Time: {:.4}\n E: {:.4}, M: {:.4}, H: {:.4}"
          .format(epoch, train_time, test_time, total_protocol[0],total_protocol[1], total_protocol[2]))
      start_time = time.time()
  total_end_time = time.time()
  print("Model: {}\nTime : {:.5}".format(args.log_name,total_end_time-total_start_time))
  with summary_writer.as_default():
    tf.summary.scalar('training_time', total_end_time-total_start_time,step=0)

def create_model(args):
  regularizer = tf.keras.regularizers.l2(args.regularizer_scale)
  if args.attention:
    model = ResidualGraphConvolutionalNetwork_Attention(args=args,regularizer=regularizer)
  else:
    model = ResidualGraphConvolutionalNetwork(args=args,regularizer=regularizer)
  optimizer = tf.keras.optimizers.Adam(args.lr)
  if args.hparam_tuning:
    log_dir = os.path.join('logs','hparam_tuning',args.log_name)
  else:
    if args.pre_graph:
      log_dir = os.path.join('logs',args.dataset,'pre_graph',args.log_name,str(args.num_layers))
    else:
      log_dir = os.path.join('logs',args.dataset,'standard',args.log_name,str(args.num_layers))
  summary_writer = tf.summary.create_file_writer(log_dir)
  return model,optimizer,summary_writer