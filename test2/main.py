import matplotlib.pyplot as plt
import numpy as np
import sys
import chainer
from chainer import Variable, serializers
from prepare import get_typemap, read_data
from trainer import Trainer
from model import create_model2

inputdata_file = sys.argv[1]
typemap = get_typemap(inputdata_file)
x, t = read_data(inputdata_file, typemap)

# use 2 of 3 for train, 1 of 3 for test
ind = np.random.permutation(x.shape[0])
ind_offset = len(ind) * 2 // 3
x_train = x[ind[:ind_offset]]
x_test  = x[ind[ind_offset:]]
t_train = t[ind[:ind_offset]]
t_test  = t[ind[ind_offset:]]

dim = [x.shape[1], 120, 50, 1]
model, optimizer = create_model2(dim)
tr = Trainer(model, optimizer)
for e in range(1000000):
  loss = tr.train_one_epoch(x_train, t_train, bs=200, drop_ratio=0.30).data
  if e % 100 == 0:
    y = model.fwd(Variable(x_test)).data
    ac = (1 - abs(y - t_test) / t_test).mean()
    f = open('plotdata.csv', 'a')
    f.write("{},{},{}\n".format(e, loss, ac))
    f.close()
  if e % 1000 == 0:
    serializers.save_npz("my.model_{}".format(e), model)
    serializers.save_npz("my.state_{}".format(e), optimizer)

