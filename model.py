import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 对原始观察图像进行卷积，降低维度
        self.cv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2) # (64 - 4)/2 + 1 = 31
        self.cv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # (31 - 4)/2 + 1 = 14
        self.cv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2) # (14 - 4)/2 + 1 = 6
        self.cv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2) # (6 - 4)/2 + 1 = 2
        
    def forward(self, obs):
        hidden = F.relu(self.cv1(obs))
        hidden = F.relu(self.cv2(hidden))
        hidden = F.relu(self.cv3(hidden))
        hidden = F.relu(self.cv4(hidden))
        # print(hidden.size()) 
        # - torch.Size([1, 256, 2, 2])
        embedded_obs = hidden.reshape(hidden.size(0), -1) # 自动调整 hidden 形状为 (1, 1024)
        return embedded_obs

class RecurrentStateSpaceModel(nn.Module):
    """
    RSSM 模型中包含了多个部分
    Deterministic state model: h_{t+1} = f(h_t, s_t, a_t)
    Stochastic state model(prior): p(s_{t+1} | h_{t+1})
    State posterior: q(s_t | h_t, o_t)
    Note: 这里的 hidden_dim 是通用的合并特征的压缩空间
    """
    def __init__(self, state_dim, action_dim, deterministic_state_dim, hidden_dim=200, min_stddev=0.1, act=F.relu):
        super().__init__()
        self.state_dim = state_dim # Stochastic state
        self.action_dim = action_dim 
        self.deterministic_dim = deterministic_state_dim
        self.fc_state_action = nn.Linear(state_dim + action_dim, hidden_dim) # hidden - stochastic state with action
        self.rnn = nn.GRUCell(hidden_dim, deterministic_state_dim) 
        self.fc_deterministic_stochastic_state = nn.Linear(deterministic_state_dim, hidden_dim) # p(s_{t+1} | h_{t+1})
        self.fc_state_prior_mean = nn.Linear(hidden_dim, state_dim)
        self.fc_state_prior_stddev = nn.Linear(hidden_dim, state_dim) # 先验分布 p(s_{t+1} | h_{t+1}) 的均值和标准差
        # 将 deterministic state 和 observation 映射到 hidden_dim 空间
        self.fc_deterministic_state_embedded_observation = nn.Linear(deterministic_state_dim + 1024, hidden_dim)
        self.fc_state_posterior_mean = nn.Linear(hidden_dim, state_dim)
        self.fc_state_posterior_stddev = nn.Linear(hidden_dim, state_dim) # 后验分布 q(s_{t} | h_{t}, o_{t}) 的均值和标准差
        self.min_stddev = min_stddev
        self.act = act
        
    def forward(self, state, action, deterministic_state, embedded_next_observation):
        """
        h_{t+1} = f(h_t, s_t, a_t)
        返回:
        先验分布 p(s_{t+1} | h_{t+1})
        后验分布 q(s_{t+1} | h_{t+1}, o_{t+1}) 
        """
        next_state_prior, deterministic_state = self.prior(state, action, deterministic_state)
        next_state_posterior = self.posterior(deterministic_state, embedded_next_observation)
        return next_state_prior, next_state_posterior, deterministic_state
    
    def prior(self, state, action, deterministic_state):
        """
        h_{t+1} = f(h_t, s_t, a_t)
        s_{t+1} ~ p(s_{t+1} | h_{t+1})
        Note: 最后要输出 h_{t+1} 给后验分布进行计算
        """
        hidden = self.act(self.fc_state_action(torch.cat([state, action], dim=1))) # 将 state 和 action 拼接
        deterministic_state = self.rnn(hidden) # 计算下一个 step 的确定性状态 h_{t+1}
        hidden = self.act(self.fc_deterministic_stochastic_state(deterministic_state)) # 由于 rnn 输出的是 deterministic_state 的维度，还要再通过一层全连接网络到统一的 hidden_dim
        
        mean = self.fc_state_prior_mean(hidden)
        stddev = F.softplus(self.fc_state_prior_stddev(hidden)) + self.min_stddev # p(s_{t+1} | h_{t+1}) 的分布, 这里使用 softplus 避免输出方差为负数
        return Normal(mean, stddev), deterministic_state
        
    def posterior(self, deterministic_state, embedded_next_observation):
        """
        s_{t+1} ~ q(s_{t+1} | h_{t+1}, o_{t+1})
        """
        hidden = self.act(self.fc_deterministic_state_embedded_observation(torch.cat([deterministic_state, embedded_next_observation], dim=1))) # 将 deterministic state 和 t+1 时的 observation 进行拼接
        mean = self.fc_state_posterior_mean(hidden)
        stddev = F.softplus(self.fc_state_posterior_stddev(hidden)) + self.min_stddev
        return Normal(mean, stddev)
        
class ObservationModel(nn.Module):
    """
    p(o_t | s_t, h_t)
    首先将 stochastic_state 和 deterministic_state 映射到 1024 维, 和 Encoder 保持一致
    """
    def __init__(self, stochastic_state_dim, deterministic_dim):
        super().__init__()
        self.fc = nn.Linear(stochastic_state_dim + deterministic_dim, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)
        
    def forward(self, stochastic_state, deterministic_state):
        hidden = self.fc(torch.cat([stochastic_state, deterministic_state], dim=1))
        hidden = hidden.view(hidden.size(0), 1024, 1, 1)
        hidden = F.relu(self.deconv1(hidden))
        hidden = F.relu(self.deconv2(hidden))
        hidden = F.relu(self.deconv3(hidden))
        obs = self.deconv4(hidden)
        return obs
        
        
class RewardModel(nn.Module):
    """
    p(r_t | s_t, h_t)
    和 ObservationModel 相同的操作, 但是降维到 hidden_dim
    """
    def __init__(self, stochastic_state_state_dim, deterministic_state_dim, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(stochastic_state_state_dim + deterministic_state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
    
    def forward(self, stochastic_state, deterministic_state):
        hidden = F.relu(self.fc1(torch.cat([stochastic_state, deterministic_state], dim=1)))
        hidden = F.relu(self.fc2(hidden))
        hidden = F.relu(self.fc3(hidden))
        reward = self.fc4(hidden)
        return reward
        