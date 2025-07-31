import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, transition, td_error):
        p = (abs(td_error) + 1e-6) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(p)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = p
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if not self.buffer:
            return [],[],[],[],[],[],[]
        ps = np.array(self.priorities)
        probs = ps / ps.sum()
        idx = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in idx]
        weights = (len(self.buffer) * probs[idx]) ** (-beta)
        weights /= weights.max()
        s,a,r,s_p,done = zip(*samples)
        return (np.stack(s), np.array(a), np.array(r),
                np.stack(s_p), np.array(done), weights, idx)

    def update_priorities(self, idx, td_errors):
        for i,e in zip(idx, td_errors):
            self.priorities[i] = (abs(e) + 1e-6) ** self.alpha
