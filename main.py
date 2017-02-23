import matplotlib.pyplot as plt
import numpy as np
import csv, sys, math
import chainer
from chainer import cuda, Function, gradient_check, \
                    Variable, optimizers, serializers, \
                    utils, Link, Chain, ChainList
import chainer.functions  as F
import chainer.links as L
from prepare import get_typemap, read_data
from trainer import Trainer
from model import MyModel1, MyModel2, MyModel3, MyModel4, MyModel5

def create_model(i, h, load=False, model_file = None, state_file = None):
  #if i == 1: model = MyModel1(x.shape[1], 1)
  #if i == 2: model = MyModel2(x.shape[1], 1)
  #if i == 3: model = MyModel3(x.shape[1], 1)
  if i == 4: model = MyModel4(x.shape[1], 1, h)
  if i == 5: model = MyModel5(x.shape[1], 1, h)
  optimizer = optimizers.Adam()
  optimizer.setup(model)
  if load:
    serializers.load_npz(model_file, model)
    serializers.load_npz(state_file, optimizer)
  return model, optimizer
  
#load = False
#save = False

inputdata_file = sys.argv[1]
#if len(sys.argv) > 2:
#  save = (sys.argv[2] == "t")
#if len(sys.argv) > 3:
#  model_file = sys.argv[3]
#  state_file = sys.argv[4]
#  load = True

typemap = get_typemap(inputdata_file)
x, t = read_data(inputdata_file, typemap)

ind = np.random.permutation(x.shape[0])
x_train = x[ind[:420]]
x_test  = x[ind[420:]]
t_train = t[ind[:420]]
t_test  = t[ind[420:]]

print(x.shape)       # (470, 191)
print(t.shape)       # (470, 1)
print(x_train.shape) # (420, 191)
print(x_test.shape)  # (50, 191)
print(t_train.shape) # (420, 1)
print(t_test.shape)  # (50, 1)
#exit()

epoch = 2000
min_dim = 5
max_dim = x.shape[1]
r_map = {}
for i in range(10):
  hidden_dim = np.random.randint(min_dim, max_dim, 4)
  print("hidden dim:{}".format(hidden_dim))
  m, o = create_model(4, h = hidden_dim)
  tr = Trainer(m, o)
  loss = tr.train(x_train, t_train, bs=200, display=False, epoch=epoch)
  y = m.fwd(Variable(x_test)).data
  ac = ((1 - abs(y - t_test) / t_test) * 100).mean()
  print("accuracy(avg): %3.1f" % ac)
  plt.plot(np.arange(len(loss)), loss)
  r_map[",".join(map(str, hidden_dim))] = ac

for k, v in sorted(r_map.items(), key=lambda x:x[1]):
  print("acc: %3.1f, hidden: %s" % (v, k))

plt.show()


#m1, o1 = create_model(1, False, None, None)
#m2, o2 = create_model(2, False, None, None)
#m3, o3 = create_model(3, False, None, None)

#tr1 = Trainer(m1, o1)
#tr2 = Trainer(m2, o2)
#tr3 = Trainer(m3, o3)
#loss1 = tr1.train(x_train, t_train, bs=200, display=True, epoch=epoch)
#loss2 = tr2.train(x_train, t_train, bs=200, display=True, epoch=epoch)
#loss3 = tr3.train(x_train, t_train, bs=200, display=True, epoch=epoch)
#y1 = m1.fwd(Variable(x_test)).data
#y2 = m2.fwd(Variable(x_test)).data
#y3 = m3.fwd(Variable(x_test)).data

#result = np.hstack([t_test, y1, y2, y3])
#for r in result:
#  arr = np.array([abs(r[1]-r[2]), abs(r[2]-r[3]), abs(r[3]-r[1])])
#  i = arr.argmin()
#  if i == 0: ans = (r[1]+r[2])/2
#  if i == 1: ans = (r[2]+r[3])/2
#  if i == 2: ans = (r[3]+r[1])/2
#  print("%5d, %5d : (%5d, %5d, %5d)" % (r[0], ans, r[1], r[2], r[3]))

#t = t_test
#result = np.hstack([t, y, y-t, (100 - abs(y-t)*100/t)])
#print("answer, predi, diff , accur")
#for i in result:
#  print("%5d, %5d, %5d, %3.1f" % (i[0], i[1], i[2], i[3]))
#print("accuracy(avg): %3.1f" % (result[:,3]).mean())
#diff_total = sum(abs(result[:, 2]))
#answer_total = sum(result[:, 0])
#print("accuracy(avg): %3.1f" % (100 - diff_total*100/answer_total))

#if save: 
#  serializers.save_npz('my.model', model)
#  serializers.save_npz('my.state', optimizer)

#plt.plot(np.arange(len(loss1)), loss1)
#plt.plot(np.arange(len(loss2)), loss2)
#plt.plot(np.arange(len(loss3)), loss3)
#plt.show()

