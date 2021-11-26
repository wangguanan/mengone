import numpy as np

class XavierInitializer:

  def __init__(self, mode="uniform"):
    """
      Args:
        mode(str): uniform or normal
    """
    self.mode = mode

  def __call__(self, cin, cout):
    val = np.sqrt(6 / (cin + cout))
    return np.random.uniform(low=-val, high=val, size=[cin, cout])


class KaimingInitializer:

  def __init__(self,):
    pass

  def __call__(self, cin, cout):
    val = np.sqrt(6 / (cin + cout))
    return np.random.uniform(low=-val, high=val, size=[cin, cout])
