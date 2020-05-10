import numpy as np
import tensorflow as tf

def query_sort_tf(Q,X):
  dot_result_tf = tf.matmul(Q,tf.transpose(X))
  sorted_list_tf = tf.argsort(-dot_result_tf)
  return sorted_list_tf
  
def compute_map(ranks, gnd, kappas=[], evaluate_way=1):
  map = 0.
  nq = len(gnd) # number of queries
  aps = np.zeros(nq)
  pr = np.zeros(len(kappas))
  prs = np.zeros((nq, len(kappas)))
  nempty = 0
  for i in np.arange(nq):
    qgnd = np.array(gnd[i]['ok'])

    # no positive images, skip from the average
    if qgnd.shape[0] == 0:
      aps[i] = float('nan')
      prs[i, :] = float('nan')
      nempty += 1
      continue

    try:
      qgndj = np.array(gnd[i]['junk'])
    except:
      qgndj = np.empty(0)

    # sorted positions of positive and junk images (0 based)
    pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
    junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

    k = 0;
    ij = 0;
    if len(junk):
      # decrease positions of positives based on the number of
      # junk images appearing before them
      ip = 0
      while (ip < len(pos)):
        while (ij < len(junk) and pos[ip] > junk[ij]):
          k += 1
          ij += 1
        pos[ip] = pos[ip] - k
        ip += 1

    # compute ap
    ap = compute_ap(pos, len(qgnd), evaluate_way=evaluate_way)
    map = map + ap
    aps[i] = ap

    # compute precision @ k
    pos += 1 # get it to 1-based
    for j in np.arange(len(kappas)):
      kq = min(max(pos), kappas[j]); 
      prs[i, j] = (pos <= kq).sum() / kq
    pr = pr + prs[i, :]

  map = map / (nq - nempty)
  pr = pr / (nq - nempty)

  return map, aps, pr, prs

def compute_ap(ranks, nres, evaluate_way=1):
  nimgranks = len(ranks)
  ap = 0
  recall_step = 1. / nres

  for j in np.arange(nimgranks):
    rank = ranks[j]
    precision_1 = float(j + 1) / (rank + 1) # 猜对了几次/猜了总共几次
    if evaluate_way:
      ap += precision_1 * recall_step
    else:
      if rank == 0:
        precision_0 = 1.
      else:
        precision_0 = float(j) / rank
      ap += (precision_0 + precision_1) * recall_step / 2.
  return ap

def quick_get_mAP(Q,X,gnd_total):
  '''
  快速计算mAP得分
  '''
  sorted_list = query_sort_tf(Q, X)
  total_protocol = []
  for protocol in ['E','M','H']:
    mAP = compute_map(sorted_list.numpy().T,gnd_total[protocol],[])
    total_protocol.append(mAP[0])
  return total_protocol

