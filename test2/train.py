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

dim = [x.shape[1], 120, 50, 1]
model, optimizer = create_model2(dim)

model_file_name = "my.model"
state_file_name = "my.state"
if len(sys.argv) >= 4:
  model_file_name = sys.argv[2]
  state_file_name = sys.argv[3]
  serializers.load_npz(model_file_name, model)
  serializers.load_npz(state_file_name, optimizer)

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
    epoch=200000,
    drop_ratio=0.30
    )

serializers.save_npz(model_file_name + ".new", model)
serializers.save_npz(state_file_name + ".new", optimizer)

y = model.fwd(Variable(x_test)).data
ac = (1 - abs(y - t_test) / t_test).mean()
print("acc: %3.3f, loss: %10.3f\n" % (ac, loss[-1]))
plt.plot(np.arange(len(loss)), loss)
plt.show()

