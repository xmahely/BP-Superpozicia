from torch import nn


class MLP(nn.Module):
    '''
      Multilayer Perceptron for regression.
    '''

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 1, bias=True)
        )

    def forward(self, x):
        '''
          Forward pass
        '''
        return self.layers(x)