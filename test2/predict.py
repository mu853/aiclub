import numpy as np
import sys
import chainer
from chainer import Variable, serializers
from prepare import get_typemap, read_data
from model import create_model2

inputdata_file = sys.argv[1]
model_file_name = sys.argv[2]
state_file_name = sys.argv[3]

typemap = get_typemap(inputdata_file)
x, t = read_data(inputdata_file, typemap)

dim = [x.shape[1], 120, 50, 1]
model, optimizer = create_model2(dim)
serializers.load_npz(model_file_name, model)
serializers.load_npz(state_file_name, optimizer)

y = model.fwd(Variable(x)).data
print("expect actual  diff  acc")
for r in np.hstack([t, y, y - t, 1 - abs((y - t) / t)]):
  print("%6d %6d %5d %.2f" % (r[0], r[1], r[2], r[3]))

ac = (1 - abs(y - t) / t).mean()
print("total acc = %.3f" % ac)

d = 1 - abs(y - t) / t
d.sort(axis=0)
ac = d[:50].mean()
print("total acc = %.3f" % ac)

