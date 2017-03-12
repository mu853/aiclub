import matplotlib.pyplot as plt
import numpy as np
import sys

def getdata(datafile):
  data = np.loadtxt(datafile, delimiter=",")
  x = data[:, 0]
  loss = data[:, 1]
  acc = data[:, 2]
  return x, loss, acc

datafile = sys.argv[1]

x, loss, acc = getdata(datafile)
fig, ax1 = plt.subplots()
l1, = ax1.plot(x, loss, color='b')
ax2 = ax1.twinx()
ax2.set_ylim(0, 1)
l2, = ax2.plot(x, acc, color='g')

while True:
  x, loss, acc = getdata(datafile)
  ax1.set_xlim(x.min(), x.max())
  l1.set_data(x, loss)
  ax2.set_xlim(x.min(), x.max())
  l2.set_data(x, acc)
  plt.title("loss:%.3f, acc:%.3f" % (loss[-1], acc[-1]))
  plt.pause(10)

