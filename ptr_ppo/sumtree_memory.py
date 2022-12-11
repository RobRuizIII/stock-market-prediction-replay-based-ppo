import random
import heapq
import numpy as np


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0
    heap = heapq.heapify([])

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

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


    # Need to see how I will calculate advantage and ensure I used everything---> might only need log probs and advantages but need to consider importance sampling
    def add(self, episode_starts: np.ndarray, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray, log_probs: np.ndarray, advantages: np.ndarray):
        
        # get number of steps per environment and number of environments
        (n_steps, n_envs) = observations.shape

        # get idx of episode starts
        idx_episode_starts = (episode_starts > 0.5).nonzero()
        idx_episode_starts = sorted(zip(idx_episode_starts[0], idx_episode_starts[1]), key = lambda x: (x[1], x[0]))


        # check if episode_starts 1.0 values are start or end of episode
        cur_idx = 0
        for n_env in range(n_envs):
            start = 0
            end = idx_episode_starts[cur_idx][0] if idx_episode_starts[cur_idx][1] == n_env else n_steps
            steps = 0
            while (steps < n_steps):
                traj_obs = observations[start:end, n_env]
                traj_act = actions[start:end, n_env]
                traj_rew = rewards[start:end, n_env]
                traj_prob = log_probs[start:end, n_env]
                traj_adv = advantages[start:end, n_env]
                steps += end - start
                
                adv_error = np.amax(np.abs(traj_adv.copy()), axis = 0) if self.max_advantage else np.mean(np.abs(traj_adv.copy()), axis = 0)
                priority = self._get_priority(adv_error)

                # [(observation, actions, reward, log_prob, advantage)...]
                traj_sample = zip(traj_obs, traj_act, traj_rew, traj_prob, traj_adv)

                self.tree.add(priority, traj_sample)

                if (end != n_steps):
                    cur_idx += 1

                start = end
                end = idx_episode_starts[cur_idx][0] if idx_episode_starts[cur_idx][1] == n_env else n_steps


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
        p = self._get_priority(error)
        self.tree.update(idx, p)