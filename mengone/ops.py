import numpy as np

class MalMul:

  def __init__(self,):
    pass

  def __call__(self, A, B):
    return self.forward(A, B)

  def forward(self, A, B):
    """
      Args:
        A(np.array): m x d
        B(np.array): d x n
      return:
        C(np.array): m x n
    """
    self.A = A
    self.B = B
    return np.matmul(A, B)

  def backward(self, D):
    """
      D(np.array): m x n
    """
    dA = np.matmul(D, self.B.transpose())
    dB = np.matmul(self.A.transpose(), D)
    return dA, dB

class Add:

  def __init__(self,):
    pass

  def __call__(self, A, B):
    return self.forward(A, B)

  def forward(self, A, B):
    """
      Args:
        A(np.array2d): bs, c
        B(np.array1d): c
    """
    if self._check_input(A, B):
      print("dim error")
      return
    self.A = A
    self.B = B
    return A + B

  def backward(self, D):
    """
    """
    return D, np.sum(D, axis=0)

  def _check_input(self, A, B):
    if len(A.shape) != 2: return False
    if len(B.shape) != 1: return False 

class ReLU:

  def __init__(self, ):
    pass

  def __call__(self, x):
    return self.forward(x)

  def forward(self, x):
    self.x = x
    return np.maximum(x, 0)

  def backward(self, D):
    return np.maximum(np.sign(self.x), 0) * D