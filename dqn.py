import os
import time
import random
import argparse
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from env import SnakeEnv

# Hyperparameters
LEARNING_RATE   = 0.0005
GAMMA           = 0.99
BUFFER_LIMIT    = 50000
BATCH_SIZE      = 64
GRID_SIZE       = (10, 10)
NUM_EPISODES    = 500000
NUM_ACTIONS     = 4
USE_ACTION_MASK = True
WEIGHT_DIR      = 'weight'
WEIGHT_PATH     = os.path.join(WEIGHT_DIR, 'dqn.pth')


class ReplayBuffer:
    def __init__(self, device):
        self.buffer = deque(maxlen=BUFFER_LIMIT)
        self.device = device

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, m_lst, a_lst, r_lst, s_prime_lst, m_prime_lst, done_mask_lst = map(np.array, zip(*mini_batch))

        return (
            torch.tensor(s_lst, dtype=torch.float).to(self.device),
            torch.tensor(m_lst, dtype=torch.int).to(self.device),
            torch.tensor(a_lst, dtype=torch.int64).to(self.device),
            torch.tensor(r_lst, dtype=torch.float).to(self.device),
            torch.tensor(s_prime_lst, dtype=torch.float).to(self.device),
            torch.tensor(m_prime_lst, dtype=torch.int).to(self.device),
            torch.tensor(done_mask_lst, dtype=torch.float).to(self.device)
        )

    def __len__(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, grid_size, num_actions, device):
        super(QNet, self).__init__()
        n, m = grid_size
        self.num_actions = num_actions
        self.device = device
        self.layers = nn.Sequential(
            nn.Conv2d(2,  32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (n - 2) * (m - 2), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions)
        )
        self.to(device)

    def forward(self, x, mask=None):
        x = self.layers(x)
        if mask is not None:
            x.masked_fill_(mask == 0, float('-inf'))
        return x

    def _to_tensor(self, x):
        if x is not None:
            x = x[np.newaxis, :]
            x = torch.from_numpy(x)
            x = x.to(self.device)
        return x

    def get_values(self, obs, mask=None, print_values=False):
        obs, mask = self._to_tensor(obs), self._to_tensor(mask)
        values = self(obs, mask)
        if print_values:
            print('Values:', values)
        return values

    def sample_action(self, obs, mask=None, epsilon=-1, print_values=False):
        if random.random() < epsilon:
            mask = np.ones(self.num_actions) if mask is None else mask
            valid_actions = [a for a in range(self.num_actions) if mask[a]]
            return random.choice(valid_actions)
        else:
            out = self.get_values(obs, mask, print_values)
            return out.argmax().item()

def train_model(q, q_target, memory, optimizer):
    for i in range(10):
        s, m, a, r, s_prime, m_prime, done_mask = memory.sample(BATCH_SIZE)

        q_val = q(s, m).gather(1, a)
        max_q_prime = q_target(s_prime, m_prime).max(1)[0].unsqueeze(1)
        target = r + GAMMA * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_val, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=1.0)

        # if i == 0:
        #     for name, param in q.named_parameters():
        #         if param.grad is not None:
        #             grad_norm = param.grad.norm().item()
        #             print(f"Gradient norm for {name}: {grad_norm}")

        optimizer.step()

def train():

    env = SnakeEnv(grid_size=GRID_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q = QNet(GRID_SIZE, NUM_ACTIONS, device)
    q_target = QNet(GRID_SIZE, NUM_ACTIONS, device)
    q_target.load_state_dict(q.state_dict())

    memory = ReplayBuffer(device)
    optimizer = optim.Adam(q.parameters(), lr=LEARNING_RATE)
    max_score = -1

    for n_epi in range(NUM_EPISODES):

        epsilon = max(0.0001, 0.20 - 0.01 * (n_epi / 10000))
        s, m, done = env.reset(), np.ones(4), False
        score = 0

        with torch.no_grad():
            while not done:
                a = q.sample_action(s, m, epsilon)
                s_prime, r, done, info = env.step(a)
                m_prime = info.get('mask', np.ones(4)) if USE_ACTION_MASK else np.ones(4)
                memory.put((s, m, [a], [r / 1.0], s_prime, m_prime, [float(not done)]))
                s, m = s_prime, m_prime
                score += r > 0

        if score >= max_score:
            max_score = score
            torch.save(q.state_dict(), WEIGHT_PATH)
            
        if len(memory) > 10000:
            train_model(q, q_target, memory, optimizer)

        if n_epi % 20 == 0:
            q_target.load_state_dict(q.state_dict())
            print(f"n_episode: {n_epi}, score: {score:.1f}, n_buffer: {len(memory)}, eps: {epsilon * 100:.1f}%")

    env.close()


def play():
    if not os.path.exists(WEIGHT_PATH):
        print("No trained weights found!")
        return

    device = torch.device('cpu')
    q = QNet(GRID_SIZE, NUM_ACTIONS, device)
    q.load_state_dict(torch.load(WEIGHT_PATH))
    q.eval()

    env = SnakeEnv(grid_size=GRID_SIZE, gui=True)
    s, m, done, score = env.reset(), np.ones(4), False, 0
    epsilon = 0

    with torch.no_grad():
        while not done:
            a = q.sample_action(s, m, epsilon, True)
            s, r, done, info = env.step(a)
            m = info.get('mask', np.ones(4)) if USE_ACTION_MASK else np.ones(4)
            score += r > 0
            env.render()
            time.sleep(0.05)

    env.close()
    print(f'score: {score}')


if __name__ == '__main__':
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', dest='is_training', action='store_true', help="Set training mode")
    args = parser.parse_args()
    train() if args.is_training else play()
