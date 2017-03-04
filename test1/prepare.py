import numpy as np
import csv

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

