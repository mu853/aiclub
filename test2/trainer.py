import numpy as np
import math
from chainer import Variable

class Trainer():
  def __init__(self, model, optimizer):
    self.model = model
    self.optimizer = optimizer

  def train(self, x, y, bs=10, epoch=5000, display=True, drop_ratio=0.2):
    loss = []
    for e in range(epoch):
      l = self.train_one_epoch(x, y, bs, drop_ratio=drop_ratio).data
      if e % 100 == 0:
        if display: print("epoch:%5d, loss:%10.3f" % (e, l))
        loss.append(l)
    return loss

  def train_one_epoch(self, x, y, bs, drop_ratio):
    loss_total = None
    n = x.shape[0]

    for i in range(0, n, bs):
      ind = np.random.permutation(n)
      xv = Variable(x[ind[i:(i+bs) if (i+bs) < n else n]])
      yv = Variable(y[ind[i:(i+bs) if (i+bs) < n else n]])
  
      self.model.zerograds()
      loss = self.model(xv, yv, drop_ratio)
      loss.backward()
      self.optimizer.update()
  
      #if math.isnan(loss.data): continue

      if loss_total is None:
        loss_total = loss
      else:
        loss_total += loss

    return loss_total
    
