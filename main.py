import matplotlib.pyplot as plt
import numpy as np
import csv, sys, math
import chainer
from chainer import cuda, Function, gradient_check, \
                    Variable, optimizers, serializers, \
                    utils, Link, Chain, ChainList
from prepare import get_typemap, read_data
from trainer import Trainer
from model import MyModel, create_model

# read data
inputdata_file = sys.argv[1]
typemap = get_typemap(inputdata_file)
x, t = read_data(inputdata_file, typemap)

# train
epoch = 5000
min_dim = 5
max_dim = x.shape[1]
num_of_layers = 4
r_map = {}

for i in range(3):
  ind = np.random.permutation(x.shape[0])
  x_train = x[ind[:420]]
  x_test  = x[ind[420:]]
  t_train = t[ind[:420]]
  t_test  = t[ind[420:]]

  #hidden_dim = np.random.randint(min_dim, max_dim, num_of_layers - 1)
  hidden_dim = np.array([180, 120, 60])
  dim = [x.shape[1]] + hidden_dim.tolist() + [1]
  print("dim:{}".format(dim))
  m, o = create_model(dim)
  tr = Trainer(m, o)
  loss = tr.train(x_train, t_train, bs=200, display=True, epoch=epoch, drop_ratio=0.25)
  
  y = m.fwd(Variable(x_test)).data
  ac = (1 - abs(y - t_test) / t_test).mean()
  r_map[",".join(map(str, dim))] = [ac, loss[-1]]

  print("acc: %3.3f, loss: %10.3f\n" % (ac, loss[-1]))
  plt.plot(np.arange(len(loss)), loss)

for k, v in sorted(r_map.items(), key=lambda x:x[1]):
  print("acc: %3.3f, loss: %10.3f, dim: %s" % (v[0], v[1], k))

plt.show()

