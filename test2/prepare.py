import numpy as np
import csv, sys, math

def is_float(val):
  try:
    float(val)
  except ValueError:
    return False
  return True

def get_typemap(datafile):
  f = open(datafile, 'r')
  r = csv.reader(f)
  header1 = next(r)
  header2 = next(r)

  typemap = {}

  for row in r:
    parameters = row[4:]

    j = 0
    for p in parameters:
      if p in ['True', 'False']: pass
      elif p == '': pass
      elif is_float(p): pass
      else:
        if j not in typemap: typemap[j] = {}
        tmp = typemap[j]
        if p not in tmp: tmp[p] = len(tmp)
        parameters[j] = tmp[p]
      j += 1

  f.close()
  return typemap

def read_data(datafile, typemap):
  f = open(datafile, 'r')
  r = csv.reader(f)
  header1 = next(r)
  header2 = next(r)

  data = []
  t = []

  ext_ind_offset = {}
  num_of_ext_cols = 0
  for key in sorted(typemap):
    ext_ind_offset[key] = num_of_ext_cols
    num_of_ext_cols += len(typemap[key])
    #print("key:{}".format(key))
    #for k, v in sorted(typemap[key].items(), key=lambda x:x[1]):
      #print("  {}:{}".format(v, k))
  #print("ext_ind_offset:{}".format(ext_ind_offset))

  #rn = 0
  for row in r:
    price = float(row[1])
    delivery_date = row[2]
    supplier = row[3]
    parameters = row[4:] + [0] * num_of_ext_cols

    j = 0
    for p in parameters:
      if p in ['True', 'False']:
        parameters[j] = bool(p)
      elif p == '':
        parameters[j] = 0
      elif is_float(p):
        parameters[j] = float(p)
      else:
        # use j (column index) as key
        if j not in typemap:
          print("{} is not in typemap".format(j))
          continue
        tmp = typemap[j]
        if p not in tmp:
          print("{} is not in typemap".format(p))
          continue
        ext_index = len(row[4:]) + tmp[p] + ext_ind_offset[j]
        #if rn == 10:
          #print("j:{}, p:{}, len(row[4:]):{}, tmp[p]:{}, ext_ind_offset[j]:{}, ext_index:{}".format(j, p, len(row[4:]), tmp[p], ext_ind_offset[j], ext_index))

        parameters[ext_index] = 100
        parameters[j] = 0 # clear original column
      j += 1
    data.append(parameters)
    t.append(price)
    #rn += 1

  f.close()
  #f = open("./data2.csv", 'w')
  #w = csv.writer(f, lineterminator='\n')
  #w.writerows(data)
  #f.close()
  return (np.array(data).astype(np.float32),
          np.array(t).astype(np.float32).reshape(-1, 1)
          #np.array(t).astype(np.int32).reshape(-1, 1)
          )

if __name__ == '__main__':
  filename = sys.argv[1]
  typemap = get_typemap(filename)
  data, t = read_data(filename, typemap)

