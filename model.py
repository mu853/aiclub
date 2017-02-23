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
  
class MyModel(Chain):
  def __init__(self, dim):

    super(MyModel, self).__init__(
      l1=L.Linear(dim[0], dim[1]),
      l2=L.Linear(dim[1], dim[2]),
      l3=L.Linear(dim[2], dim[3]),
      l4=L.Linear(dim[3], dim[4]),
      bn1=L.BatchNormalization(dim[1]),
    )

  def __call__(self, x, t):
    bv = self.fwd(x)
    return F.mean_squared_error(bv, t)

  def fwd(self, x, train=False, ratio=0.2):
    fv1 = F.dropout(F.sigmoid(self.bn1(self.l1(x))), train=train, ratio=ratio)
    fv2 = F.dropout(F.sigmoid(self.l2(fv1)), train=train, ratio=ratio)
    fv3 = self.l3(fv2)
    fv4 = self.l4(fv3)
    return fv4

