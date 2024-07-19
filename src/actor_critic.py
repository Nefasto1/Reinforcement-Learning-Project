import torch as th
import torch.nn as nn

class ActorCritic(th.nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self, state):
        action_probs = nn.functional.softmax(self.actor(state)/5, dim =-1 )
        # minimum  = action_probs.min()
        # action_probs += abs(minimum)
        # action_probs /= action_probs.sum() 
        state_value = self.critic(state)

    
        return action_probs, state_value


        