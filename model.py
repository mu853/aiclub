import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L

class MyModel4(Chain):
  def __init__(self, input_dim, output_dim, hidden_dim):
    super(MyModel4, self).__init__(
      l1=L.Linear(input_dim, hidden_dim[0]),
      l2=L.Linear(hidden_dim[0], hidden_dim[1]),
      l3=L.Linear(hidden_dim[1], hidden_dim[2]),
      l4=L.Linear(hidden_dim[2], output_dim),
      bn=L.BatchNormalization(hidden_dim[0]),
    )

  def __call__(self, x, t):
    bv = self.fwd(x)
    return F.mean_squared_error(bv, t)

  def fwd(self, x, train=False, ratio=0.2):
    fv1 = F.dropout(F.sigmoid(self.bn(self.l1(x))), train=train, ratio=ratio)
    fv2 = F.dropout(F.sigmoid(self.l2(fv1)), train=train, ratio=ratio)
    fv3 = self.l3(fv2)
    fv4 = self.l4(fv3)
    return fv4

class MyModel5(Chain):
  def __init__(self, input_dim, output_dim, hidden_dim):
    super(MyModel5, self).__init__(
      l1=L.Linear(input_dim, hidden_dim[0]),
      l2=L.Linear(hidden_dim[0], hidden_dim[1]),
      l3=L.Linear(hidden_dim[1], output_dim),
      bn=L.BatchNormalization(hidden_dim[0]),
    )

  def __call__(self, x, t):
    bv = self.fwd(x)
    return F.mean_squared_error(bv, t)

  def fwd(self, x, train=False, ratio=0.2):
    fv1 = F.dropout(F.sigmoid(self.bn(self.l1(x))), train=train, ratio=ratio)
    fv2 = F.dropout(F.sigmoid(self.l2(fv1)), train=train, ratio=ratio)
    fv3 = self.l3(fv2)
    return fv3

class MyModel1(Chain):
  def __init__(self, input_dim, output_dim):
    super(MyModel1, self).__init__(
      l1=L.Linear(input_dim, 120),
      l2=L.Linear(120, 80),
      l3=L.Linear(80, 20),
      l4=L.Linear(20, output_dim),
      bn=L.BatchNormalization(120),
    )

  def __call__(self, x, t):
    bv = self.fwd(x)
    return F.mean_squared_error(bv, t)

  def fwd(self, x, train=False, ratio=0.2):
    fv1 = F.dropout(F.sigmoid(self.bn(self.l1(x))), train=train, ratio=ratio)
    fv2 = F.dropout(F.sigmoid(self.l2(fv1)), train=train, ratio=ratio)
    fv3 = self.l3(fv2)
    fv4 = self.l4(fv3)
    return fv4

class MyModel2(Chain):
  def __init__(self, input_dim, output_dim):
    super(MyModel2, self).__init__(
      l1=L.Linear(input_dim, 100),
      l2=L.Linear(100, 70),
      l3=L.Linear(70, 15),
      l4=L.Linear(15, output_dim),
      bn=L.BatchNormalization(100),
    )

  def __call__(self, x, t):
    bv = self.fwd(x)
    return F.mean_squared_error(bv, t)

  def fwd(self, x, train=False, ratio=0.2):
    fv1 = F.dropout(F.sigmoid(self.bn(self.l1(x))), train=train, ratio=ratio)
    fv2 = F.dropout(F.sigmoid(self.l2(fv1)), train=train, ratio=ratio)
    fv3 = self.l3(fv2)
    fv4 = self.l4(fv3)
    return fv4

class MyModel3(Chain):
  def __init__(self, input_dim, output_dim):
    super(MyModel3, self).__init__(
      l1=L.Linear(input_dim, 150),
      l2=L.Linear(150, 90),
      l3=L.Linear(90, 30),
      l4=L.Linear(30, output_dim),
      bn=L.BatchNormalization(150),
    )

  def __call__(self, x, t):
    bv = self.fwd(x)
    return F.mean_squared_error(bv, t)

  def fwd(self, x, train=False, ratio=0.2):
    fv1 = F.dropout(F.sigmoid(self.bn(self.l1(x))), train=train, ratio=ratio)
    fv2 = F.dropout(F.sigmoid(self.l2(fv1)), train=train, ratio=ratio)
    fv3 = self.l3(fv2)
    fv4 = self.l4(fv3)
    return fv4

