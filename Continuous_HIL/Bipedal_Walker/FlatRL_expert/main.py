import numpy as np
import torch
import gym
import argparse
import os
import multiprocessing

import utils
import TD3
import SAC


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, training_step, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Seed {seed}, Step {training_step}, Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

def train(args, seed): 
    env = gym.make(args.env)
    
    # Set seeds
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    	
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = np.zeros((action_dim,))
    for a in range(action_dim):
        max_action[a] = env.action_space.high[a]
    
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "l_rate": args.l_rate,
        "discount": args.discount,
        "tau": args.tau,
        }
    
    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action[0]
        kwargs["noise_clip"] = args.noise_clip * max_action[0]
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
        
    if args.policy == "SAC":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["alpha"] = args.alpha
        kwargs["critic_freq"] = args.critic_freq
        policy = SAC.SAC(**kwargs)
    
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    	
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, seed, 0)]
    
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
		
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        elif args.policy == "TD3":
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
        elif args.policy == "SAC":
            action = (policy.select_action(np.array(state)))

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            # Reset environment
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, seed, t+1))
                
    return evaluations, policy


if __name__ == "__main__":
	
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="SAC")                   # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="BipedalWalker-v3")         # OpenAI gym environment name
    parser.add_argument("--Nseed", default=1, type=int)          # Sets Gym, PyTorch and Numpy seeds int(0.5*multiprocessing.cpu_count())
    parser.add_argument("--start_timesteps", default=25e3, type=int) # Time steps initial random policy is used 25e3
    parser.add_argument("--eval_freq", default=5e3, type=int)        # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)    # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)       # Batch size for both actor and critic
    parser.add_argument("--l_rate", default=3e-4)                    # Learning rate
    parser.add_argument("--discount", default=0.99)                  # Discount factor
    parser.add_argument("--tau", default=0.005)                      # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)               # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                 # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)        # Frequency of delayed policy updates
    parser.add_argument("--alpha", default=0.2, type=int)            # SAC entropy regularizer term
    parser.add_argument("--critic_freq", default=2, type=int)        # Frequency of delayed critic updates
    parser.add_argument("--save_model", action="store_true")         # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_Nseed_{args.Nseed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, NSeed: {args.Nseed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")
        
    # arguments = [(args, seed) for seed in range(args.Nseed)] 
    # with multiprocessing.Pool(args.Nseed) as pool:
    #     results = pool.starmap(train, arguments)
    #     pool.close()
    #     pool.join()
    
    evaluations, policy = train(args, 0)
    
    # evaluations = []
    # for i in range(args.Nseed):
    #     evaluations.append(results[i][0])
        
    np.save(f"./results/mean_{file_name}", np.mean(evaluations,0))
    np.save(f"./results/std_{file_name}", np.std(evaluations,0))
    np.save(f"./results/steps_{file_name}", np.linspace(0,args.max_timesteps,len(np.mean(evaluations,0))))
    if args.save_model: 
        # index = np.argmax(np.max(evaluations,1))
        # policy = results[index][1]
        np.save(f"./results/evaluation_{file_name}", evaluations)
        policy.save(f"./models/{file_name}")



