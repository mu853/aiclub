import matplotlib.pyplot as plt
import numpy as np
import csv, sys, math
import chainer
from chainer import cuda, Function, gradient_check, \
                    Variable, optimizers, serializers, \
                    utils, Link, Chain, ChainList
import chainer.functions  as F
import chainer.links as L
from prepare import read_data
from trainer import Trainer
from model import MyAE, create_model

def plot(lo, la, y, num_of_hidden_layers, plt):
  markers = ["^", "o", "*", ".", ",", "v", ">", "<", "+", "1", "2"]
  colors = ["b", "g", "r", "c", "m", "y", "k", "w", "teal", "darkred", "indigo"]
  for i in range(num_of_hidden_layers):
    index = np.where(y == i)
    plt.scatter(lo[index], la[index], marker=markers[i], color=colors[i])

# parameters
mode = sys.argv[1]
datafile = sys.argv[2]
background_image = None
if len(sys.argv) > 3: background_image = sys.argv[3]
outputfile = None
if len(sys.argv) > 4: outputfile = sys.argv[4]

# read csv data
lo, la, data = read_data(datafile)

# plot
min_x = math.floor(min(lo) * 10) / 10
max_x = math.ceil(max(lo)  * 10) / 10
min_y = math.floor(min(la) * 10) / 10
max_y = math.ceil(max(la)  * 10) / 10
x_ticks = np.arange(min_x, max_x, 0.2) # x label
y_ticks = np.arange(min_y, max_y, 0.2) # y label
extent  = [min_x, max_x, min_y, max_y] # image size
plt.figure(figsize=(18,10)) # plot area size

subplot_rows = 2
subplot_cols = 4

num_of_hidden_layers_list = [2, 3, 4, 5, 6, 7, 8, 9]

for z in range(subplot_rows * subplot_cols):
  h_dim = num_of_hidden_layers_list[z]
  model_file_name = "my_{}.model".format(h_dim)
  state_file_name = "my_{}.state".format(h_dim)

  dim = [24, h_dim, 24]
  print("dim:{}".format(dim))

  ind = np.random.permutation(len(data))
  x_train = data[ind[:60]]
  t_train = x_train[:]
  x_test = data[ind[60:]]
  t_test = x_test[:]

  model, optimizer = create_model(dim)
  loss = None
  if mode == "loss":
    tr = Trainer(model, optimizer)
    loss = tr.train(x_train, t_train, bs=30, epoch=3000, display=False, ratio=0.2, bn=True)
    serializers.save_npz(model_file_name, model)
    serializers.save_npz(state_file_name, optimizer)
  else:
    model, optimizer = create_model(dim, model_file_name, state_file_name)
    tr = Trainer(model, optimizer)

  if mode == "acc":
    x_restore = model.fwd(Variable(x_test)).data
    y = F.sigmoid(model.l1(Variable(x_test))).data.argmax(axis=1)
    #print("y:{}".format(y))
    v_sum = 0
    for i in range(h_dim):
      subp = plt.subplot(2, 5, i + 1)
      ind = np.where(y == i)[0]
      if len(ind) > 0:
        avg = x_restore[ind].mean(axis=0)
        v = ((x_restore[ind] - avg) ** 2).mean()
        v_sum += v
      subp.set_title("no:%d, v:%.2f" % (i + 1, v))
      print("no:%d, v:%.2f" % (i + 1, v))
      for j in ind:
        subp.plot(np.arange(0, 24), x_restore[j])
    plt.savefig("acc_%02d" % h_dim)
    plt.close()
    print("v_sum:%.2f" % (v_sum))
  else:
    subp = plt.subplot(subplot_rows, subplot_cols, z + 1)
    if mode == "loss":
      subp.set_title("k = %d, loss = %.2f" % (h_dim, loss[-1]))
      subp.plot(np.arange(len(loss)), loss)
    else:
      y = F.sigmoid(model.l1(Variable(data))).data.argmax(axis=1)
      subp.set_title("k = %d" % (h_dim))
      subp.xaxis.get_major_formatter().set_useOffset(False) # disable E notation
      subp.xaxis.set_ticks(x_ticks)
      subp.yaxis.set_ticks(y_ticks)
      subp.imshow(plt.imread(background_image), extent=extent)
      plot(lo, la, y, h_dim, plt)

if mode != "acc":
  if outputfile is None:
    plt.show()
  else:
    plt.savefig(outputfile)

