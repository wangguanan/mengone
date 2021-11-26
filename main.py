import numpy as np
import time

import mengone as m1

class Model:

  def __init__(self):
    in_dim = 2
    h1_dim = 8
    h2_dim = 16
    out_dim = 1

    self.linear1 = m1.Linear(cin=in_dim, cout=h1_dim)
    self.relu1 = m1.ReLU()
    self.linear2 = m1.Linear(cin=h1_dim, cout=h2_dim)
    self.relu2 = m1.ReLU()
    self.linear3 = m1.Linear(cin=h2_dim, cout=out_dim)

  def __call__(self, x):
    return self.forward(x)

  def forward(self, x):
    """
      Args:
        x(np.array): bs x in_dim
      Return:
        out(np.array): bs x 1 
    """
    y = self.linear1(x)
    ry = self.relu1(y)
    z = self.linear2(ry)
    rz = self.relu2(z)
    out = self.linear3(rz)
    return out

  def backward(self, D):
    """
      Args:
        D(np.array): bs x 1
    """
    drz = self.linear3.backward(D)
    dz = self.relu2.backward(drz)
    dry = self.linear2.backward(dz)
    dy = self.relu1.backward(dry)
    dx = self.linear1.backward(dy)

  def update(self, lr):
    self.linear3.update(lr)
    self.linear2.update(lr)
    self.linear1.update(lr)                                                                                                                                                                                                                                   


if __name__ == '__main__':

  model = Model()
  loss = m1.L2()
  EPOCH = 10000

  bs = 128
  x1 = np.random.randn(bs, 2) + 10
  y1 = np.ones([bs, 1])
  x2 = np.random.randn(bs, 2) - 10
  y2 = np.ones([bs, 1]) * (-1)
  x = np.concatenate([x1, x2], axis=0)
  y = np.concatenate([y1, y2], axis=0)

  for e in range(EPOCH):
    p = model(x)
    l = loss(p, y)
    dp, _ = loss.backward()
    model.backward(dp)
    model.update(0.000000001)
    print(e, l)
    time.sleep(0.01)