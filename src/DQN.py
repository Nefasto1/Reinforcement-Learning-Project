import torch as th

class DQN(th.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        self.linear = th.nn.Sequential(
                          th.nn.Linear(in_features=7, out_features=128,      bias=True),
                          th.nn.ReLU(),
                          th.nn.Linear(in_features=128, out_features=128,      bias=True),
                          th.nn.ReLU(),
                          th.nn.Linear(in_features=128, out_features=4, bias=True),
                          # th.nn.Linear(in_features=128, out_features=11 ** 3, bias=True),
                      )

    def forward(self, x):
        out = self.linear(x)

        # out = out.reshape(-1, 3, 11)
        # out = out.reshape(-1, 11, 11, 11)
        return out

