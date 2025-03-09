import argparse
from datetime import datetime   
import json
import os
from pprint import pprint
import time
import numpy as np
import torch
from torch.distributions import kl_divergence
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from dm_control import suite
from dm_control.suite.wrappers import pixels
from wrappers import GymWrapper, RepeatAction
from utils import ReplayBuffer, preprocess_obs
from model import Encoder, RecurrentStateSpaceModel, ObservationModel, RewardModel
from agent import CEMAgent


def main():
    parser = argparse.ArgumentParser(description='pytorch implementation of PlaNet by LinboWang')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-dir', type=str, default='log')
    parser.add_argument('--test-interval', type=int, default=10)
    parser.add_argument('--domain-name', type=str, default='cheetah')
    parser.add_argument('--task-name', type=str, default='run')
    parser.add_argument('--action-repeat', type=int, default=4) # cheetah 4; cartpole 8; reacher 4; finger 2; cup 4; walker 2
    parser.add_argument('--state-dim', type=int, default=30) # Distributions in latent space are 30-dimensional diagonal Gaussians with predicted mean and standard deviation.
    parser.add_argument('--rnn-hidden-dim', type=int, default=200) # a GRU (Cho et al., 2014) with 200 units as deterministic path in the dynamics model
    parser.add_argument('--buffer-capacity', type=int, default=1000000)
    parser.add_argument('--all-episodes', type=int, default=200)
    parser.add_argument('--seed-episodes', type=int, default=5) # We start from S = 5 seed episodes with random actions
    parser.add_argument('--update-steps', type=int, default=100) # collect another episode every C = 100 update steps
    parser.add_argument('--batch-size', type=int, default=50) # batches of B = 50
    parser.add_argument('--chunk-length', type=int, default=50) # sequence chunks of length L = 50
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epsilon', type=float, default=1e-4) # The model is trained using the Adam optimizer (Kingma & Ba, 2014) with a learning rate of 1e-3, epsilon = 1e−4
    parser.add_argument('--clip-grad-norm', type=int, default=1000) # gradient clipping norm of 1000
    parser.add_argument('--free-nats', type=int, default=3) # grant the model 3 free nats by clipping the divergence loss below this value
    """
    For planning, we use:
    CEM with horizon length H = 12, 
    optimization iterations I = 10, 
    candidate samples J = 1000, 
    and refitting to the best K = 100
    """
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--N-iterations', type=int, default=10)
    parser.add_argument('--N-candidates', type=int, default=1000)
    parser.add_argument('--N-top-candidates', type=int, default=100)
    parser.add_argument('--action-noise-var', type=float, default=0.3) # under epsilon ∼ Normal(0, 0.3) action noise
    args = parser.parse_args()
    
    # 准备训练日志的输出路径
    log_dir = os.path.join(args.log_dir, args.domain_name + '_' + args.task_name)
    log_dir = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M'))
    os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f) # 写入超参数到 json 文件中
    pprint(vars(args))
    writer = SummaryWriter(log_dir=log_dir) # 初始化 TensorBoard 日志
    
    # 设定随机数种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建 dm_control 环境并用 GymWrapper 修饰，其中包含了 RepeatStep
    env = suite.load(args.domain_name, args.task_name, task_kwargs={'random': args.seed})
    env = pixels.Wrapper(env, render_kwargs={'height': 64, 
                                             'width': 64, 
                                             'camera_id': 0})
    env = RepeatAction(GymWrapper(env), skip=args.action_repeat) # 默认为 4
    
    # 定义经验回放池
    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity, 
                                 observation_shape=env.observation_space.shape,
                                 action_dim=env.action_space.shape[0])
    
    # 定义模型和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder().to(device)
    Rssm = RecurrentStateSpaceModel(args.state_dim, 
                                    env.action_space.shape[0], 
                                    args.rnn_hidden_dim).to(device)
    obs_model = ObservationModel(args.state_dim, args.rnn_hidden_dim).to(device)
    reward_model = RewardModel(args.state_dim, args.rnn_hidden_dim).to(device)
    all_params = (list(encoder.parameters()) +
                  list(Rssm.parameters()) +
                  list(obs_model.parameters()) +
                  list(reward_model.parameters()))
    optimizer = Adam(all_params, lr=args.lr, eps=args.epsilon)
    
    # seed episodes，随机策略收集经验
    for episode in range(args.seed_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            replay_buffer.add(obs, action, reward, done)
            obs = next_obs
            
    # 剩余的 episodes，固定步长更新模型参数并根据 CEM 算法进行动作规划
    for episode in range(args.seed_episodes, args.all_episodes):
        
        # 收集经验
        start = time.time()
        cem_agent = CEMAgent(encoder, Rssm, reward_model, args.horizon, args.N_iterations, args.N_candidates, args.N_top_candidates)
        
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = cem_agent(obs)
            action += np.random.normal(0, np.sqrt(args.action_noise_var),
                                       env.action_space.shape[0]) # 在动作中加入噪声
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.add(obs, action, reward, done)
            obs = next_obs
            total_reward += reward

        writer.add_scalar('total reward at train', total_reward, episode)
        print('episode [%4d/%4d] is collected. Total reward is %f' %
              (episode+1, args.all_episodes, total_reward))
        print('elasped time for interaction: %.2fs' % (time.time() - start))    
        
        # 更新模型
        start = time.time()
        for update_step in range(args.update_steps):
            observations, actions, rewards, dones = replay_buffer.sample(args.batch_size, args.chunk_length)
            # 对观察图像进行预处理，并转为 tensor 格式
            observations = preprocess_obs(obs=observations)
            # observation 的格式 (batch_size, chunk_length, width, height, channels)
            observations = torch.as_tensor(observations, device=device)
            # 将 observevation 的格式转成 RSSM 模型 (RNN) 能处理的格式 (chunk_length, batch_size, channels, width, height)
            observations = observations.transpose(3, 4).transpose(2, 3)
            observations = observations.transpose(0, 1)
            actions = torch.as_tensor(actions, device=device).transpose(0, 1)
            rewards = torch.as_tensor(rewards, device=device).transpose(0, 1)
            
            # embed observation 将观察编码到 latent space 中
            embedded_observations = encoder(observations.reshape(-1, 3, 64, 64)).view(args.chunk_length, args.batch_size, -1)
            
            # 存储 stochastic state 和 deterministic state
            stochastic_states = torch.zeros(args.chunk_length, args.batch_size, args.state_dim, device=device)
            deterministic_states = torch.zeros(args.chunk_length, args.batch_size, args.rnn_hidden_dim, device=device)
            
            # 用 zero vector 初始化 stochastic state 和 deterministic state
            stochastic_state = torch.zeros(args.batch_size, args.state_dim, device=device)
            deterministic_state = torch.zeros(args.batch_size, args.rnn_hidden_dim, device=device)
            
            # 计算损失函数: Loss = KL(q(s_t | h_t, o_t) || p(s_t | h_t)) + ObsLoss + RewardLoss
            kl_loss = 0
            for l in range(args.chunk_length - 1):
                next_state_prior, next_state_posterior, deterministic_state = Rssm(stochastic_state, actions[l], deterministic_state, embedded_observations[l+1])
                stochastic_state = next_state_prior.rsample() # rsample 支持重参数化
                stochastic_states[l+1] = stochastic_state
                deterministic_states[l+1] = deterministic_state
                kl = kl_divergence(next_state_prior, next_state_posterior) # 等价于 - kl(q(posterior) || p(prior)), 后面需要最小化 Loss, 符号一致
                kl_loss += kl.clamp(min=args.free_nats).mean() # 限制 kl 小于 3 free nats, 避免过拟合 | 对 batch 取平均
            kl_loss /= (args.chunk_length - 1) # 计算了 chunk_length - 1 次 kl loss
            
            # 计算 Reconstruction Loss 和 Reward Loss
            flatten_stochastic_states = stochastic_states.view(-1, args.state_dim)
            flatten_deterministic_states = deterministic_states.view(-1, args.rnn_hidden_dim)
            reconstruct_observation = obs_model(flatten_stochastic_states, flatten_deterministic_states).view(args.chunk_length, args.batch_size, 3, 64, 64)
            predicted_reward = reward_model(flatten_stochastic_states, flatten_deterministic_states).view(args.chunk_length, args.batch_size, 1)
            mse_output = mse_loss(reconstruct_observation[1:], observations[1:], reduction='none')
            obs_loss = 0.5 * mse_loss(reconstruct_observation[1:], observations[1:], reduction='none').mean([0, 1]).sum() # 按照 chunk 和 batch 两个维度求平均, 然后再将 3 个 channel 的所有像素的 mse loss 求和
            
            reward_loss = 0.5 * mse_loss(predicted_reward[1:], rewards[:-1])
            
            # 将所有的 Loss 求和, 更新模型参数
            loss = kl_loss + obs_loss + reward_loss
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(all_params, args.clip_grad_norm)
            optimizer.step()

            # 输出损失并同步到 tensorboard 上
            print('update_step: %3d loss: %.5f, kl_loss: %.5f, obs_loss: %.5f, reward_loss: %.5f' % (update_step+1, loss.item(), kl_loss.item(), obs_loss.item(), reward_loss.item()))
            total_update_step = episode * args.update_steps + update_step
            writer.add_scalar('overall loss', loss.item(), total_update_step)
            writer.add_scalar('kl loss', kl_loss.item(), total_update_step)
            writer.add_scalar('obs loss', obs_loss.item(), total_update_step)
            writer.add_scalar('reward loss', reward_loss.item(), total_update_step)
        
        print('elapsed time for update: %.2fs' % (time.time() - start))
        
        # 每 10 个更新步长测试一次 without exploration noise
        if (episode + 1) % args.test_interval == 0:
            start = time.time()
            cem_agent = CEMAgent(encoder, Rssm, reward_model,
                                 args.horizon, args.N_iterations,
                                 args.N_candidates, args.N_top_candidates)
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = cem_agent(obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward

            writer.add_scalar('total reward at test', total_reward, episode)
            print('Total test reward at episode [%4d/%4d] is %f' %
                  (episode+1, args.all_episodes, total_reward))
            print('elasped time for test: %.2fs' % (time.time() - start))        
    
    # 保存模型参数，供测试和重建观测视频
    torch.save(encoder.state_dict(), os.path.join(log_dir, 'encoder.pth'))
    torch.save(Rssm.state_dict(), os.path.join(log_dir, 'rssm.pth'))
    torch.save(obs_model.state_dict(), os.path.join(log_dir, 'obs_model.pth'))
    torch.save(reward_model.state_dict(), os.path.join(log_dir, 'reward_model.pth'))
    
if __name__ == '__main__':
    main()
