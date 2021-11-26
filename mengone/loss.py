import numpy as np

class L2:

  def __init__(self, reduce="mean"):
    self.reduce = reduce

  def __call__(self, x, y):
    return self.forward(x, y)

  def forward(self, x, y):
    """
    Args:
      x(np.array): bs x d
      y(np.array): bs x d
    """
    
    self.x = x
    self.y = y
    loss = np.sum((x-y)**2, axis=1)
    if self.reduce == "mean":
      return np.mean(loss)
    elif self.reduce == "sum":
      return np.sum(loss)

  def backward(self):
    return 2*(self.x-self.y), 2*(self.y-self.x)
