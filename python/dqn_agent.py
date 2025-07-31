import torch
import torch.nn as nn
import torch.optim as optim

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, lr=1e-3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1)
        )
        self.adv   = nn.Sequential(
            nn.Linear(128,64), nn.ReLU(), nn.Linear(64,action_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.shared(x)
        v = self.value(x)
        a = self.adv(x)
        return v + (a - a.mean(dim=1,keepdim=True))

    def train_step(self, batch, target_net, gamma, is_weights):
        s,a,r,s_p,done = batch
        q = self(s).gather(1,a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = target_net(s_p).max(1)[0]
            y = r + gamma * q_next * (1-done)
        td_error = y - q
        loss = (is_weights * td_error.pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return td_error.detach().cpu().numpy()
