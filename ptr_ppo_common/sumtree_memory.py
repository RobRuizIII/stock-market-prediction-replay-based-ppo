import random
import heapq
import numpy as np
import torch as th


# https://github.com/rlcode/per/blob/master/SumTree.py
# https://github.com/rlcode/per/blob/master/prioritized_memory.py


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.heap = []
        heapq.heapify(self.heap)

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):

        if (self.n_entries >= self.capacity and self.heap[0][0] < p):
            
            lowest_priority = heapq.heappop(self.heap)
            idx, self.write = lowest_priority[1]
            
            self.data[self.write] = data
            self.update(idx, p)
        
        elif (self.n_entries < self.capacity):
            
            idx = self.write + self.capacity - 1
            
            self.data[self.write] = data
            self.update(idx, p)
            
            heapq.heappush(self.heap, (p, (idx, self.write)))

            self.write += 1
            if self.write >= self.capacity:
                self.write = 0

            if self.n_entries < self.capacity:
                self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    

class SumTreeMemory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity: int, max_advantage: bool = True):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.max_advantage = max_advantage

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a



    def add(self, n_envs: int, n_steps: int, log_probs: th.Tensor, advantages: th.Tensor, observations: th.Tensor, actions: th.Tensor, values: th.Tensor, next_non_terminal: th.Tensor, returns: th.Tensor, obs_shape: tuple, action_dim: int) -> None:
        

        log_probs = np.reshape(log_probs.clone().cpu().numpy().flatten(), ((n_steps, n_envs)), order='F')
        advantages = np.reshape(advantages.clone().cpu().numpy().flatten(), ((n_steps, n_envs)), order='F')
        observations = np.reshape(observations.clone().cpu().numpy().flatten(), ((n_steps, n_envs) + obs_shape), order='F')
        actions = np.reshape(actions.clone().cpu().numpy().flatten(), ((n_steps, n_envs, action_dim)), order='F')
        values = np.reshape(values.clone().cpu().numpy().flatten(), ((n_steps, n_envs)), order='F')
        next_non_terminal = np.reshape(next_non_terminal.clone().cpu().numpy().flatten(), ((n_steps, n_envs)), order='F')
        returns = np.reshape(returns.clone().cpu().numpy().flatten(), ((n_steps, n_envs)), order='F')

        for n_env in range(n_envs):
            traj_log_probs = log_probs[:, n_env]
            traj_advantages = advantages[:, n_env]
            traj_observations = observations[:, n_env, :]
            traj_actions = actions[:, n_env, :]
            traj_values = values[:, n_env]
            traj_next_non_terminal = next_non_terminal[:, n_env]
            traj_returns = returns[:, n_env]

            traj_priority = np.max(np.abs(traj_advantages)) if self.max_advantage else np.mean(np.abs(traj_advantages))
            traj_priority = self._get_priority(traj_priority)
            self.tree.add(traj_priority, (traj_log_probs, traj_advantages, traj_observations, traj_actions, traj_values, traj_next_non_terminal, traj_returns))


    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        error = np.max(np.abs(error)) if self.max_advantage else np.mean(np.abs(error))
        p = self._get_priority(error)
        self.tree.update(idx, p)