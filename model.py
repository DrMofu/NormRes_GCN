import tensorflow as tf 

def GSS_loss(logits, alpha, beta): # 用个mask,只考虑有连接的边
  losses = -0.5 * alpha * (logits - beta)**2
  return tf.reduce_mean(losses)

class GCN_layer(tf.keras.layers.Layer):
  def __init__(self,layer_id,args,regularizer=None):
    super(GCN_layer, self).__init__()
    self.layer_id = layer_id
    self.hidden_units = args.hidden_units[layer_id]
    self.init_weights = args.init_weights
    self.regularizer = regularizer
    self.random = args.random
    self.dropout = args.dropout 
  
  def init_w(self,shape,dtype=tf.float32):
    if self.random:
      initializer_w = tf.random_normal_initializer()
      init_w_ = initializer_w(shape=shape)
    else:
      initializer_w = tf.random_normal_initializer(stddev=self.init_weights)
      init_w_ = initializer_w(shape=shape)
      init_w_ += tf.eye(shape[0],shape[-1])
    return init_w_

  def build(self, input_shape):
    self.W = self.add_weight(name='W_' + str(self.layer_id),
                          shape=[input_shape[-1], self.hidden_units],
                          dtype=tf.float32,
                          initializer=self.init_w,
                          regularizer=self.regularizer,
                          trainable=True)

    self.b = self.add_weight(name='b_' + str(self.layer_id),
                    shape=(self.hidden_units,),
                    dtype=tf.float32,
                    initializer=tf.zeros,
                    trainable=True)
    
  def call(self, inputs, adj_normed_sparse_tensor, training=None):
    x = tf.matmul(inputs, self.W) + self.b
    x = tf.sparse.sparse_dense_matmul(adj_normed_sparse_tensor, x) 
    pre_nonlinearity = x
    if training:
      pre_nonlinearity = tf.nn.dropout(pre_nonlinearity,
                        self.dropout,noise_shape=[1,2048])
    output = tf.nn.elu(pre_nonlinearity)

    return pre_nonlinearity, output

class ResidualGraphConvolutionalNetwork(tf.keras.Model):
  def __init__(self,args,regularizer=None):
    super(ResidualGraphConvolutionalNetwork,self).__init__(name='Residual_Graph_Convolutional_Network')
    self.num_layers = len(args.hidden_units)
    self.layer_decay = args.layer_decay
    self.regularizer = regularizer
    self.record_all = args.record_all
    self.gcn_layers = []

    for i in range(self.num_layers):
      new_gcn = GCN_layer(layer_id=i,args=args,regularizer=self.regularizer)
      self.gcn_layers.append(new_gcn)
          
  def call(self, x, adj_normed_sparse_tensor,training=None):
    residual = None
    recoder = []
    for i in range(self.num_layers): 
      pre_nonlinearity, x = self.gcn_layers[i](x,adj_normed_sparse_tensor,training=training)
      if residual is not None and self.layer_decay:  # 额外残差
        x = (1-self.layer_decay)*residual + self.layer_decay * x
        x = tf.nn.l2_normalize(x, axis=1)
      residual = pre_nonlinearity

      if self.record_all:
        recoder.append(tf.nn.l2_normalize(x, axis=1))

    # 最后计算相似
    if self.record_all:
      return recoder
    else:
      hidden_emb = tf.nn.l2_normalize(x, axis=1)  # 特征向量正则
      return hidden_emb