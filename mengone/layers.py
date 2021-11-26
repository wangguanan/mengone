import numpy as np
from .ops import *

class Linear:

  def __init__(self, cin, cout):
    self.mat_mul = MalMul()
    self.add = Add()
    self.w = np.random.randn(cin, cout)
    self.b = np.random.randn(cout)

  def __call__(self, x):
    return self.forward(x)

  def forward(self, x):
    """
      Args:
        x(np.array): bs x cin
      Returns:
        y(np.array): bs x cout
    """
    xw = self.mat_mul(x, self.w)
    y = self.add(xw, self.b)
    return y

  def backward(self, D):
    """
      Args:
        D(np.array): bs x cout
    """
    dxw, db = self.add.backward(D)
    dx, dw = self.mat_mul.backward(dxw)
    self.dw = dw
    self.db = db
    return dx

  def update(self, lr):
    self.w -= lr*self.dw
    self.b -= lr*self.db
