import matplotlib.pyplot as plt
import numpy as np
import csv, sys, math
import chainer
from chainer import cuda, Function, gradient_check, \
                    Variable, optimizers, serializers, \
                    utils, Link, Chain, ChainList
import chainer.functions  as F
import chainer.links as L

class MyAE(Chain):
  def __init__(self, num_of_hidden_layers):
    super(MyAE, self).__init__(
      l1=L.Linear(719,num_of_hidden_layers),
      l2=L.Linear(num_of_hidden_layers,719),
    )

  def __call__(self, x):
    bv = self.fwd(x)
    return F.mean_squared_error(bv, x)

  def fwd(self, x):
    fv = F.sigmoid(self.l1(x))
    #fv = F.dropout(F.sigmoid(self.l1(x)))
    bv = self.l2(fv)
    return bv

def train(model, optimizer, data, batch_size):
  data_size = data.shape[0]
  perm = np.random.permutation(data_size)
  loss = None

  for i in range(data_size // batch_size + 1):
    index = []
    if (i + 1) * batch_size < data_size:
      index = perm[i * batch_size:(i + 1) * batch_size - 1]
    else:
      index = perm[i * batch_size:data_size - 1]
    x = data[index, :]

    # denoising
    noise_ratio = 0.2
    np.random.permutation(x.shape[1])[:int(x.shape[1] * noise_ratio)]

    model.zerograds()
    loss = model(Variable(x))
    loss.backward()
    optimizer.update()
  return loss.data

def predict(data):
  x = Variable(data)
  yt = F.sigmoid(model.l1(x))
  c = np.argmax(yt.data, axis=1)
  return c

def create_model(num_of_hidden_layers):
  model = MyAE(num_of_hidden_layers)
  optimizer = optimizers.Adam()
  optimizer.setup(model)
  return model, optimizer

def read_data(datafile):
  f = open(datafile, 'r')
  r = csv.reader(f)
  header = next(r)
  print(header)
  lo = np.array(list(map(float, next(r)[1:])))
  la = np.array(list(map(float, next(r)[1:])))
  data = np.loadtxt(datafile,
                    delimiter=",",
                    skiprows=3,
                    usecols=np.arange(1,len(header)))
  data = data.astype(np.float32).T
  return lo, la, data

def plot(lo, la, y, num_of_hidden_layers, plt):
  markers = ["^", "o", "*", ".", ",", "v", ">", "<", "+", "1", "2"]
  colors = ["b", "g", "r", "c", "m", "y", "k", "w", "teal", "darkred", "indigo"]
  for i in range(num_of_hidden_layers):
    index = np.where(y == i)
    plt.scatter(lo[index], la[index], marker=markers[i], color=colors[i])

# parameters
batch_size = 3
datafile = sys.argv[1]
background_image = sys.argv[2]
outputfile = None
if len(sys.argv) > 3: outputfile = sys.argv[3]

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
subplot_cols = 1

num_of_hidden_layers_list = [3, 4]

for z in range(subplot_rows * subplot_cols):
  model, optimizer = create_model(num_of_hidden_layers_list[z])
  for n in range(200): loss = train(model, optimizer, data, batch_size)

  y = predict(data)
  subp = plt.subplot(subplot_rows, subplot_cols, z + 1)
  subp.xaxis.get_major_formatter().set_useOffset(False) # disable E notation
  subp.xaxis.set_ticks(x_ticks)
  subp.yaxis.set_ticks(y_ticks)
  subp.imshow(plt.imread(background_image), extent=extent)
  subp.set_title("k = %d, loss = %.2f" % (num_of_hidden_layers_list[z], loss))
  plot(lo, la, y, num_of_hidden_layers_list[z], plt)

if outputfile is None:
  plt.show()
else:
  plt.savefig(outputfile)

