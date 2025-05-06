"""
Evaluation script for the RL-based taxi pricing system.
This script focuses on evaluating a trained model and generating performance metrics.
"""

import os
import argparse
import logging
import yaml
import time
import json
import numpy as np
import matplotlib.pyplot as plt

from config.config_manager import ConfigManager
from environment.taxi_env import OptimizedTaxiEnv
from utils.performance import timer, print_summary
from utils.visualization import plot_learning_curve, plot_environment_metrics
from data.ride_pool_builder import OptimizedRidePoolBuilder
from serving.model_server import ModelServer
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("taxi-rl-evaluate.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("taxi-rl.evaluate")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate RL-based Taxi Pricing Model")
    
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model file")
    parser.add_argument("--config", type=str, default="config/colab_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data", type=str, nargs='+', required=True,
                        help="Path(s) to parquet data file(s)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory for output files")
    parser.add_argument("--n_episodes", type=int, default=10,
                        help="Number of episodes for evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--use_onnx", action="store_true",
                        help="Use ONNX model for inference")
    parser.add_argument("--render", action="store_true",
                        help="Render evaluations (if environment supports it)")
    parser.add_argument("--record_actions", action="store_true",
                        help="Record all actions and rewards")
    
    return parser.parse_args()

def evaluate():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Start timer for overall execution
    start_time = time.time()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config_path = args.config
    config_manager = ConfigManager(config_path)
    
    try:
        config = config_manager.load_config()
        logger.info(f"Configuration loaded from {config_path}")
    except (FileNotFoundError, ValueError):
        logger.warning(f"Configuration file not found: {config_path}")
        logger.info("Creating default configuration")
        config = config_manager.create_default_config(config_path)
    
    # Process data
    logger.info(f"Processing data from {len(args.data)} files")
    with timer("data_processing"):
        builder = OptimizedRidePoolBuilder(args.data)
        ride_pool_df, build_metrics = builder.run()
    
    # Create evaluation environment
    logger.info("Creating evaluation environment")
    with timer("environment_creation"):
        eval_env = OptimizedTaxiEnv(
            ride_pool_df=ride_pool_df,
            api_key=config["env"]["api_key"],
            max_rides=config["env"]["max_rides"],
            seed=config["env"]["seed"] + 100,  # Use a different seed for evaluation
            use_cache=config["performance"]["use_cache"],
            use_numba=config["performance"]["use_numba"],
            use_parallel=config["performance"]["parallel_processing"]
        )
    
    # Check if the model exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    # Evaluate using ModelServer if ONNX is requested
    if args.use_onnx:
        logger.info("Using ModelServer with ONNX for evaluation")
        with timer("model_server_init"):
            model_server = ModelServer(
                model_path=args.model,
                use_onnx=True,
                device=args.device
            )
        
        # Manual evaluation using the ModelServer
        with timer("model_evaluation"):
            total_rewards = []
            episode_lengths = []
            episode_actions = []
            
            for episode in range(args.n_episodes):
                episode_reward = 0
                episode_length = 0
                actions = []
                
                obs = eval_env.reset()
                done = False
                
                while not done:
                    # Get action from model server
                    action = model_server.predict(obs, deterministic=True)
                    
                    # Step the environment
                    obs, reward, done, info = eval_env.step(action)
                    
                    # Record data
                    episode_reward += reward
                    episode_length += 1
                    
                    if args.record_actions:
                        actions.append({
                            "step": episode_length,
                            "action": int(action),
                            "reward": float(reward),
                            "info": info
                        })
                    
                    # Render if requested
                    if args.render:
                        eval_env.render()
                
                total_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                if args.record_actions:
                    episode_actions.append(actions)
                
                logger.info(f"Episode {episode+1}/{args.n_episodes}: "
                          f"Reward = {episode_reward:.2f}, Length = {episode_length}")
            
            # Calculate statistics
            mean_reward = np.mean(total_rewards)
            std_reward = np.std(total_rewards)
            
            evaluation_results = {
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
                "episode_rewards": [float(r) for r in total_rewards],
                "episode_lengths": episode_lengths,
                "server_metrics": model_server.get_performance_metrics()
            }
            
            if args.record_actions:
                evaluation_results["episode_actions"] = episode_actions
            
            logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    else:
        # Load the model directly using stable-baselines3
        logger.info(f"Loading model from {args.model}")
        with timer("model_loading"):
            model = DQN.load(args.model, device=args.device)
        
        # Evaluate using stable-baselines3
        logger.info(f"Evaluating model over {args.n_episodes} episodes")
        with timer("model_evaluation"):
            mean_reward, std_reward = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=args.n_episodes,
                deterministic=True,
                render=args.render,
                return_episode_rewards=False
            )
        
        evaluation_results = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward)
        }
        
        logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Get environment performance metrics
    env_metrics = eval_env.get_performance_metrics()
    evaluation_results["environment_metrics"] = env_metrics
    
    # Save evaluation results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_path}")
    
    # Generate plots
    # Plot environment metrics
    env_plot_path = os.path.join(args.output_dir, "environment_metrics.png")
    plot_environment_metrics(env_metrics, save_path=env_plot_path, show=False)
    
    # Plot episode rewards
    if "episode_rewards" in evaluation_results:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(evaluation_results["episode_rewards"]) + 1), 
                evaluation_results["episode_rewards"])
        plt.axhline(y=mean_reward, color='r', linestyle='-', label=f'Mean: {mean_reward:.2f}')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards')
        plt.legend()
        plt.grid(True)
        
        rewards_plot_path = os.path.join(args.output_dir, "episode_rewards.png")
        plt.savefig(rewards_plot_path)
        plt.close()
        
        logger.info(f"Episode rewards plot saved to {rewards_plot_path}")
    
    # Report overall performance
    total_time = time.time() - start_time
    logger.info(f"Total evaluation time: {total_time:.2f} seconds")
    print_summary()
    
    return evaluation_results

if __name__ == "__main__":
    evaluate()
