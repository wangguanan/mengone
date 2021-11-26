import numpy as np

class Tensor:

  def __init__(self, val, trainable=False, name=""):
    self.name = name
    self.trainable = None
    self.val = val
    self.grad = np.zeros(val)
  
  def zero_grad(self):
    self.grad = np.zeros(self.val)