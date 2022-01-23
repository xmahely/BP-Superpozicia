import torch.nn as nn
import torch


class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(2,4),
      nn.ReLU(),
      nn.Linear(4,1)
      # nn.Linear(13, 64),
      # nn.ReLU(),
      # nn.Linear(64, 32),
      # nn.ReLU(),
      # nn.Linear(32, 1)
    )


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)