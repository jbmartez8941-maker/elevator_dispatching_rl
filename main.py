import argparse
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.torch_layers import CombinedExtractor
from elevator_env import ElevatorEnv
from gui import run_gui
import numpy as np

def setup_logging():
    """Create logging directory and return path"""
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def train_agent(num_floors, num_elevators, total_timesteps, log_dir):
    # Create vectorized environment
    env = make_vec_env(
        lambda: ElevatorEnv(num_floors, num_elevators),
        n_envs=4,  # Parallel environments for faster training
        seed=np.random.randint(0, 1000)
    )
    
    # Setup evaluation callback
    eval_env = ElevatorEnv(num_floors, num_elevators)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        callback_on_new_best=StopTrainingOnRewardThreshold(
            reward_threshold=-10,  # Stop if we achieve this reward
            verbose=1
        )
    )
    
    # Use MultiInputPolicy for dictionary observations
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ortho_init=True,    # Better weight initialization
        features_extractor_class=CombinedExtractor,
        # activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        "MultiInputPolicy",  # Changed from MlpPolicy
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-3,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,     # Smaller clipping
        ent_coef=0.01,      # Encourage exploration
        max_grad_norm=0.5,  # Gradient clipping
        vf_coef=0.5,        # Reduce value loss weight
        policy_kwargs=policy_kwargs
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="PPO"
    )
    
    # Save model with timestamp
    model_name = f"elevator_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model.save(os.path.join(log_dir, model_name))
    model.save(f"elevator_ppo_model_{num_floors}_{num_elevators}")  # Save a copy for GUI
    print(f"elevator_ppo_model_{num_floors}_{num_elevators}")
    return model

def evaluate_agent(model_path, num_floors, num_elevators, num_episodes, render=False):
    env = ElevatorEnv(num_floors, num_elevators)
    model = PPO.load(model_path)
    
    rewards = []
    wait_times = []
    utilizations = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            if render:
                env.render(mode='human')
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
        wait_times.append(info.get("total_wait_time", 0))
        utilizations.append(info.get("elevator_utilization", 0))
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Avg Wait Time: {info.get('total_wait_time', 0)/env.episode_length:.2f} steps")
        print(f"  Elevator Utilization: {info.get('elevator_utilization', 0)*100:.2f}%")
    
    print("\nEvaluation Summary:")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Wait Time: {np.mean(wait_times):.2f} steps")
    print(f"Average Utilization: {np.mean(utilizations)*100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Elevator Dispatch RL System")
    parser.add_argument("--gui", action="store_true", help="Run GUI simulation")
    parser.add_argument("--train", action="store_true", help="Train the RL agent")
    parser.add_argument("--evaluate", type=str, help="Evaluate model (provide path)")
    parser.add_argument("--floors", type=int, default=10, help="Number of floors")
    parser.add_argument("--elevators", type=int, default=3, help="Number of elevators")
    parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
    parser.add_argument("--episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render evaluation")
    
    args = parser.parse_args()
    log_dir = setup_logging()

    if args.train:
        print(f"Training agent for {args.timesteps} timesteps...")
        train_agent(
            args.floors,
            args.elevators,
            args.timesteps,
            log_dir
        )
    
    if args.evaluate:
        print(f"Evaluating model {args.evaluate} over {args.episodes} episodes...")
        evaluate_agent(
            args.evaluate,
            args.floors,
            args.elevators,
            args.episodes,
            args.render
        )
    
    if args.gui:
        run_gui(args.floors, args.elevators)

if __name__ == "__main__":
    main()