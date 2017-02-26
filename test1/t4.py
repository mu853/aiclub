import matplotlib.pyplot as plt
import numpy as np
import csv, sys, math
from sklearn.cluster import KMeans

def read_data(datafile):
  f = open(datafile, 'r')
  r = csv.reader(f)
  header = next(r)
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
subplot_cols = 4

num_of_hidden_layers_list = [2, 3, 4, 5, 6, 7, 8, 9]
x = np.arange(24)

for i in range(subplot_rows * subplot_cols):
  model = KMeans(n_clusters=num_of_hidden_layers_list[i]).fit(data)

  subp = plt.subplot(subplot_rows, subplot_cols, i + 1)
  subp.set_title("k = %d" % num_of_hidden_layers_list[i])
  subp.xaxis.get_major_formatter().set_useOffset(False)
  subp.xaxis.set_ticks(x_ticks)
  subp.yaxis.set_ticks(y_ticks)
  subp.imshow(plt.imread(background_image), extent=extent)
  y = model.labels_
  plot(lo, la, y, num_of_hidden_layers_list[i], plt)

#plt.tight_layout()
if outputfile is None:
  plt.show()
else:
  plt.savefig(outputfile)

