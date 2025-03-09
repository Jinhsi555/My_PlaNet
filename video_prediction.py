import argparse
import json
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
from dm_control import suite
from dm_control.suite.wrappers import pixels
from agent import CEMAgent
from model import Encoder, RecurrentStateSpaceModel, ObservationModel, RewardModel
from utils import preprocess_obs
from wrappers import GymWrapper, RepeatAction

def save_video_as_gif(frames):
    """
    将预测的 observation 帧存储为 gif 视频
    """
    plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        patch.set_data(frames[i])
        plt.title('Ground Truth Frame' + ' '*20 + 'Predicted Frame \n Step %d' % (i))
        
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=150)
    anim.save('Predicted Observation.gif', writer='imagemagick')
    
def main():
    parser = argparse.ArgumentParser(description='Predicted Observation vedio by learned model')
    parser.add_argument('dir', type=str, help='log directory to load learned model')
    parser.add_argument('--length', type=int, default=50, help='the frames length of predicted video')
    parser.add_argument('--domain-name', type=str, default='cheetah')
    parser.add_argument('--task-name', type=str, default='run')
    parser.add_argument('--action-repeat', type=int, default=4)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--N-iterations', type=int, default=10)
    parser.add_argument('--N-candidates', type=int, default=1000)
    parser.add_argument('--N-top-candidates', type=int, default=100)
    args = parser.parse_args()
    
    env = suite.load(args.domain_name, args.task_name)
    env = pixels.Wrapper(env, render_kwargs={'height': 64,
                                             'width': 64,
                                             'camera_id': 0})
    env = GymWrapper(env)
    env = RepeatAction(env, skip=args.action_repeat)
    
    with open(os.path.join(args.dir, 'args.json'), 'r') as f:
        train_args = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Encoder().to(device)
    rssm = RecurrentStateSpaceModel(args.state_dim, 
                                    env.action_space.shape[0],
                                    train_args['rnn_hidden_dim']).to(device)
    obs_model = ObservationModel(train_args['state_dim'],
                                 train_args['rnn_hidden_dim']).to(device)
    reward_model = RewardModel(train_args['state_dim'],
                               train_args['rnn_hidden_dim']).to(device)
    
    encoder.load_state_dict(torch.load(os.path.join(args.dir, 'encoder.pth')))
    rssm.load_state_dict(torch.load(os.path.join(args.dir, 'rssm.pth')))
    obs_model.load_state_dict(torch.load(os.path.join(args.dir, 'obs_model.pth')))
    reward_model.load_state_dict(torch.load(os.path.join(args.dir, 'reward_model.pth')))
    
    cem_agent = CEMAgent(encoder, rssm, reward_model,
                         args.horizon, args.N_iterations,
                         args.N_candidates, args.N_top_candidates)
    
    # 先确定开始预测的时间点
    starting_point = torch.randint(1000 // args.action_repeat - args.length, (1,)).item()
    obs = env.reset()
    for _ in range(starting_point):
        action = cem_agent(obs)
        obs, _, _, _ = env.step(action)
        
    preprocessed_obs = preprocess_obs(obs)
    preprocessed_obs = torch.as_tensor(preprocessed_obs, device=device)
    preprocessed_obs = preprocessed_obs.transpose(1, 2).transpose(0, 1).unsqueeze(0)
    with torch.no_grad():
        embedded_obs = encoder(preprocess_obs)
        
    # 用 embedded_obs 根据后验分布计算 stochastic state
    deterministic_state = cem_agent.deterministic_state
    stochastic_state = rssm.posterior(deterministic_state, embedded_obs)
    frame = np.zeros((64, 64*2, 3))
    frames = []
    for _ in range(args.length):
        action = cem_agent(obs)
        obs, _, _, _ = env.step(action)
        
        action = torch.as_tensor(action, device=device).unsqueeze(0)
        with torch.no_grad():
            state_prior, deterministic_state = rssm.prior(stochastic_state, action, deterministic_state)
            stochastic_state = state_prior.sample()
            predict_obs = obs_model(stochastic_state, deterministic_state)
            
        frame[:, :64, :] = preprocess_obs(obs)
        frame[:, 64:, :] = predict_obs.squeeze().transpose(0, 1).transpose(1, 2).cpu().numpy()
        frames.append((frame + 0.5).clip(0.0, 1.0)) # 像素值被归一化到 [-0.5, 0.5] 内，恢复到 [0, 1] 区间
        
    save_video_as_gif(frames)
    
if __name__ == '__main__':
    main()