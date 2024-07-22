import torch as th
import torch.nn as nn

class ActorCritic(th.nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
            nn.ReLU()
            
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16,1)
        )

    def forward(self, state):
        # action_probs = nn.functional.softmax(self.actor(state), dim =-1)
        # minimum  = action_probs.min()
        # action_probs += abs(minimum)
        # action_probs /= action_probs.sum() 
        x = self.actor(state)
        x = x / x.sum()
        state_value = self.critic(state)

    
        return x, state_value


class Actor(th.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.ReLU()
        )

    def forward(self, state):
        x = self.actor(state)
        x = x / x.sum()
        return x


class Critic(th.nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self, state):
        state_value = self.critic(state)
        return state_value


        