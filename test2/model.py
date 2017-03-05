import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer import cuda, Function, gradient_check, \
                    Variable, optimizers, serializers, \
                    utils, Link, Chain, ChainList

def create_model(dim, model_file = None, state_file = None):
  model = MyModel(dim)
  optimizer = optimizers.Adam()
  optimizer.setup(model)
  if model_file is not None: serializers.load_npz(model_file, model)
  if state_file is not None: serializers.load_npz(state_file, optimizer)
  return model, optimizer
  
def create_model2(dim, model_file = None, state_file = None):
  model = MyModel2(dim)
  #optimizer = optimizers.MomentumSGD()
  optimizer = optimizers.Adam()
  #optimizer = optimizers.SGD()
  optimizer.setup(model)
  if model_file is not None: serializers.load_npz(model_file, model)
  if state_file is not None: serializers.load_npz(state_file, optimizer)
  return model, optimizer
  
class MyModel(Chain):
  def __init__(self, dim):

    super(MyModel, self).__init__(
      l1=L.Linear(dim[0], dim[1]),
      l2=L.Linear(dim[1], dim[2]),
      l3=L.Linear(dim[2], dim[3]),
      l4=L.Linear(dim[3], dim[4]),
      bn1=L.BatchNormalization(dim[1]),
    )

  def __call__(self, x, t, drop_ratio):
    bv = self.fwd(x, ratio=drop_ratio)
    return F.mean_squared_error(bv, t)

  def fwd(self, x, train=False, ratio=0.2):
    fv1 = F.dropout(F.sigmoid(self.bn1(self.l1(x))), train=train, ratio=ratio)
    fv2 = F.dropout(F.sigmoid(self.l2(fv1)), train=train, ratio=ratio)
    fv3 = self.l3(fv2)
    fv4 = self.l4(fv3)
    return fv4

class MyModel2(Chain):
  def __init__(self, dim):

    super(MyModel2, self).__init__(
      l1=L.Linear(dim[0], dim[0]),
      l2=L.Linear(dim[0], dim[0]),
      l3=L.Linear(dim[0], dim[1]),
      l4=L.Linear(dim[1], dim[1]),
      l5=L.Linear(dim[1], dim[1]),
      l6=L.Linear(dim[1], dim[2]),
      l7=L.Linear(dim[2], dim[2]),
      l8=L.Linear(dim[2], dim[2]),
      l9=L.Linear(dim[2], dim[3]),
      #l10=L.Linear(dim[3], dim[4]),
      bn1=L.BatchNormalization(dim[0]),
      bn2=L.BatchNormalization(dim[1]),
      bn3=L.BatchNormalization(dim[2]),
    )

  def __call__(self, x, t, drop_ratio):
    bv = self.fwd(x, ratio=drop_ratio)
    return F.mean_squared_error(bv, t)

  def fwd(self, x, train=False, ratio=0.2):
    fv1_bn = self.bn1(x)
    fv1_re = F.relu(fv1_bn)
    fv1 = self.l1(fv1_re)

    fv2_bn = self.bn1(fv1)
    fv2_re = F.relu(fv2_bn)
    fv2_dp = F.dropout(fv2_re, train=train, ratio=ratio)
    fv2 = self.l2(fv2_dp)
    fv2_r = fv2 + x

    fv3 = self.l3(fv2_r)
    
    fv4_bn = self.bn2(fv3)
    fv4_re = F.relu(fv4_bn)
    fv4 = self.l4(fv4_re)

    fv5_bn = self.bn2(fv4)
    fv5_re = F.relu(fv5_bn)
    fv5_dp = F.dropout(fv5_re, train=train, ratio=ratio)
    fv5 = self.l5(fv5_dp)
    fv5_r = fv5 + fv3

    fv6 = self.l6(fv5_r)

    fv7_bn = self.bn3(fv6)
    fv7_re = F.relu(fv7_bn)
    fv7 = self.l7(fv7_re)

    fv8_bn = self.bn3(fv7)
    fv8_re = F.relu(fv8_bn)
    fv8_dp = F.dropout(fv8_re, train=train, ratio=ratio)
    fv8 = self.l8(fv8_dp)
    fv8_r = fv8 + fv6

    fv9 = self.l9(fv8_r)
    return fv9

