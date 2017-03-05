import matplotlib.pyplot as plt
import numpy as np
import sys
import chainer
from chainer import Function, Variable, serializers
from prepare import get_typemap, read_data
from trainer import Trainer
from model import create_model, create_model2

# read data
inputdata_file = sys.argv[1]
typemap = get_typemap(inputdata_file)
x, t = read_data(inputdata_file, typemap)

# train
dim = [x.shape[1], 120, 50, 1]
model, optimizer = create_model2(dim)

if len(sys.argv) >= 4:
  model_file_name = sys.argv[2]
  state_file_name = sys.argv[3]
  serializers.load_npz(model_file_name, model)
  serializers.load_npz(state_file_name, optimizer)

  y = model.fwd(Variable(x)).data
  print("expect actual  diff  acc")
  for r in np.hstack([t, y, y - t, 1 - abs((y - t) / t)]):
    print("%6d %6d %5d %.2f" % (r[0], r[1], r[2], r[3]))
else:
  ind = np.random.permutation(x.shape[0])
  x_train = x[ind[:420]]
  x_test  = x[ind[420:]]
  t_train = t[ind[:420]]
  t_test  = t[ind[420:]]

  tr = Trainer(model, optimizer)
  loss = tr.train(
      x_train, t_train,
      bs=200,
      display=True,
      epoch=1000,
      drop_ratio=0.30
      )

  serializers.save_npz("my.model", model)
  serializers.save_npz("my.state", optimizer)

  y = model.fwd(Variable(x_test)).data
  ac = (1 - abs(y - t_test) / t_test).mean()
  print("acc: %3.3f, loss: %10.3f\n" % (ac, loss[-1]))
  plt.plot(np.arange(len(loss)), loss)
  plt.show()

