import torch
from torch.distributions import Normal
from utils import preprocess_obs

class CEMAgent:
    """
    交叉熵方法 (cross entropy method, CEM) 是一种进化策略方法,
    它的核心思想是维护一个带参数的分布，根据每次采样的结果来更新分布中的参数，
    使得分布中能获得较高累积奖励的动作序列的概率比较高。
    对 RSSM 模型使用 action 的决策
    """
    def __init__(self, encoder, rssm, reward_model, horizen, N_iterations, N_candidates, N_top_candidates):
        self.encoder = encoder
        self.rssm = rssm
        self.reward_model = reward_model
        
        self.horizon = horizen
        self.N_iterations = N_iterations
        self.N_candidates = N_candidates
        self.N_top_candidates = N_top_candidates
        
        self.device = next(self.reward_model.parameters()).device
        self.deterministic_state = torch.zeros(1, rssm.deterministic_dim, device=self.device)
    
    def __call__(self, obs):
        obs = preprocess_obs(obs)
        obs = torch.as_tensor(obs, device=self.device)
        # 这里的 obs 的维度是 (width, height, channel)
        obs = obs.transpose(0, 1).transpose(1, 2).unsqueeze(0) # 增加一个 batch 的维度 -> (batch, channel, width, height)
        
        with torch.no_grad(): # 不涉及模型训练
            embedded_obs = self.encoder(obs)
            stochastic_state_posterior = self.rssm(self.deterministic_state, embedded_obs)
            
            # 初始化 action 的 distribution
            action_distribution = Normal(
                torch.zeros(self.horizon, self.rssm.action_dim, device=self.device),
                torch.ones(self.horizon, self.rssm.action_dim, device=self.device)
            )
            
            # 迭代更新动作分布参数，利用 CEM 算法得到最优动作分布
            for iteration in range(self.N_iterations):
                # 从 action_distribution 中采样得到的形状为: (horizon, 1) -- N 次采样 --> (N_candidates, horizon, 1) 
                action_candidates = action_distribution.sample([self.N_candidates]).transpose(0, 1) # (horizon, N_candidate, 1)
                
                total_predicted_reward = torch.zeros(self.N_candidates, device=self.device)
                stochastic_states = stochastic_state_posterior.sample([self.N_candidates]) # 去掉 batch_size 1
                deterministic_states = self.deterministic_state.repeat([self.N_candidates, 1]) # 给每一个 candidate 创建初始 deterministic state
                
                # 计算 horizon 步的动作序列的 reward
                for t in range(self.horizon):
                    next_state_prior, deterministic_states = self.rssm(stochastic_states, action_candidates[t], deterministic_states)
                    stochastic_states = next_state_prior.sample()
                    total_predicted_reward += self.reward_model(stochastic_states, deterministic_states).squeeze() # batch_size 为 1
                
                # 用 top-N candidates 来更新 action distribution
                top_indexes = total_predicted_reward.argsort(descending=True)[: self.N_top_candidates]
                top_action_candidates = action_candidates[:, top_indexes, :]
                mean = top_action_candidates.mean(dim=1) # (horizon, 1)
                stddev = (top_action_candidates - mean.unsqueeze(1)) \
                    .abs().sum(dim=1) / (self.N_top_candidates - 1) # 这里 unsqueeze(1) 插入 N_top_candidate 维度, 才能广播相减
                # N_top_candidates - 1 的原因: 无偏估计
                action_distribution = Normal(mean, stddev)
                
        action = mean[0] # 只输出每个动作序列的第一个动作作为 CEM 的决策
        
        # 更新 deterministic state 用于下一步的 planning
        with torch.no_grad():
            _, self.deterministic_state = self.rssm(stochastic_state_posterior.sample(),
                                                action.unsqueeze[0],
                                                self.deterministic_state)
        return action.cpu().numpy()