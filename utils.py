import numpy as np

class ReplayBuffer(object):
    """
    经验回放池
    """
    def __init__(self, capacity, observation_shape, action_dim):
        self.capacity = capacity
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)
        
        self.index = 0
        self.isfull = False
    
    def add(self, observation, action, reward, done):
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        
        if self.index == self.capacity - 1:
            self.isfull = True
        self.index = (self.index + 1) % self.capacity # 环形缓冲区，能够实现新数据的覆盖，避免频繁的内存分配与释放
    
    def sample(self, batch_size, chunk_length):
        """
        从经验回放池中近似均匀地采样，采样结果的形式是 (batch_size, chunk_length)
        每个 batch 的 chunk 是连续的序列
        """
        # 先划分出每一个 episode 的终止时刻
        episode_border = np.where(self.dones)[0]
        # 保存采样序列的 index
        sampled_indexes = []
        for _ in range(batch_size):
            """
            确保采样的 chunk 在同一个 episode 中，否则没有状态转移的意义
            """
            cross_border = True
            while cross_border:
                initial_index = np.random.randint(len(self) - chunk_length + 1) # 确保 initial_index 最大值开始的 chunk 不会超出 Replay_Buffer 的范围
                final_index = initial_index + chunk_length - 1
                cross_border = np.logical_and(initial_index <= episode_border, 
                                              final_index > episode_border).any() # 对两个 condition 逻辑或，只要有任意一处为 True，则超出 episode 范围
            
            sampled_indexes += list(range(initial_index, final_index + 1))
            
        sampled_observations = self.observations[sampled_indexes].reshape(
            batch_size, chunk_length, *self.observations.shape[1:])
        sampled_actions = self.actions[sampled_indexes].reshape(
            batch_size, chunk_length, self.actions.shape[1])
        sampled_rewards = self.rewards[sampled_indexes].reshape(
            batch_size, chunk_length, 1)
        sampled_donns = self.dones[sampled_indexes].reshape(
            batch_size, chunk_length, 1)
        return sampled_observations, sampled_actions, sampled_rewards, sampled_donns
    
    def __len__(self):
        return self.capacity if self.isfull else self.index

def preprocess_obs(obs, bit_depth=5):
    """
    降低图像的位深度，归一化像素值范围到 [-0.5, 0.5]
    并添加按照原文设定的均匀分布的随机噪声
    """
    obs = obs.astype(np.float32)
    reduced_obs = np.floor(obs / 2 ** (8 - bit_depth)) # 等价于低通滤波器，忽略高频细节，像素范围变为 [0, 31]
    normalized_obs = reduced_obs / 2**bit_depth - 0.5 # 归一化像素值到 [-0.5, 0.5]
    # 噪声设定为 1 / delta，其中 delta 为量化后的空间
    normalized_obs += np.random.uniform(0.0, 1.0 / 2 ** bit_depth, normalized_obs.shape)
    return normalized_obs