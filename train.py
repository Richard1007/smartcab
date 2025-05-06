"""
Training script for the RL-based taxi pricing system.
This script focuses specifically on model training, allowing for resuming from checkpoints.
"""

import os
import argparse
import logging
import yaml
import time

from config.config_manager import ConfigManager
from environment.taxi_env import OptimizedTaxiEnv
from models.custom_dqn import create_dqn_model
from models.callbacks import CustomCheckpointCallback, PerformanceMonitorCallback
from utils.performance import timer, print_summary
from utils.visualization import visualize_training_results
from data.ride_pool_builder import OptimizedRidePoolBuilder
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("taxi-rl-train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("taxi-rl.train")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RL-based Taxi Pricing Model")
    
    parser.add_argument("--config", type=str, default="config/colab_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data", type=str, nargs='+', required=True,
                        help="Path(s) to parquet data file(s)")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory for output files")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--timesteps", type=int, 
                        help="Number of training timesteps (overrides config)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--checkpoint_freq", type=int, default=1000,
                        help="Checkpoint frequency in timesteps")
    parser.add_argument("--eval_freq", type=int, default=1000,
                        help="Evaluation frequency in timesteps")
    parser.add_argument("--n_eval_episodes", type=int, default=5,
                        help="Number of episodes for evaluation")
    
    return parser.parse_args()

def train():
    """Main training function."""
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
    
    # Override config with command line arguments
    if args.timesteps:
        config_manager.update_config("train.total_timesteps", args.timesteps)
    
    if args.eval_freq:
        config_manager.update_config("train.eval_freq", args.eval_freq)
        
    if args.n_eval_episodes:
        config_manager.update_config("train.n_eval_episodes", args.n_eval_episodes)
    
    # Save updated configuration
    config_manager.save_config(os.path.join(args.output_dir, "train_config.yaml"))
    
    # Reload the final config for use
    config = config_manager.config
    
    # Process data
    logger.info(f"Processing data from {len(args.data)} files")
    with timer("data_processing"):
        builder = OptimizedRidePoolBuilder(args.data)
        ride_pool_df, build_metrics = builder.run()
    
    # Create environments
    logger.info("Creating training and evaluation environments")
    with timer("environment_creation"):
        env = OptimizedTaxiEnv(
            ride_pool_df=ride_pool_df,
            api_key=config["env"]["api_key"],
            max_rides=config["env"]["max_rides"],
            seed=config["env"]["seed"],
            use_cache=config["performance"]["use_cache"],
            use_numba=config["performance"]["use_numba"],
            use_parallel=config["performance"]["parallel_processing"]
        )
        
        eval_env = OptimizedTaxiEnv(
            ride_pool_df=ride_pool_df,
            api_key=config["env"]["api_key"],
            max_rides=config["env"]["max_rides"],
            seed=config["env"]["seed"] + 1,
            use_cache=config["performance"]["use_cache"],
            use_numba=config["performance"]["use_numba"],
            use_parallel=config["performance"]["parallel_processing"]
        )
    
    # Create or load model
    if args.checkpoint:
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        with timer("model_loading"):
            model = DQN.load(args.checkpoint, env=env, device=args.device)
            
        logger.info("Model loaded successfully")
        # Check if we should reset timesteps
        reset_num_timesteps = False
    else:
        logger.info("Creating new DQN model")
        with timer("model_creation"):
            model = create_dqn_model(env, config, device=args.device)
            
        # For new models, we reset timesteps
        reset_num_timesteps = True
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create callbacks
    checkpoint_callback = CustomCheckpointCallback(
        eval_env=eval_env,
        checkpoint_dir=checkpoint_dir,
        check_freq=args.checkpoint_freq,
        n_eval_episodes=config["train"]["n_eval_episodes"]
    )
    
    performance_callback = PerformanceMonitorCallback(
        log_freq=args.checkpoint_freq
    )
    
    # Train the model
    total_timesteps = config["train"]["total_timesteps"]
    logger.info(f"Training model for {total_timesteps} timesteps")
    with timer("model_training"):
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, performance_callback],
            reset_num_timesteps=reset_num_timesteps
        )
    
    # Save the final model
    final_model_path = os.path.join(args.output_dir, "final_model.zip")
    model.save(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Evaluate the final model
    logger.info("Evaluating final model")
    with timer("model_evaluation"):
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=10,
            deterministic=True
        )
    logger.info(f"Final mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Get environment performance metrics
    env_metrics = env.get_performance_metrics()
    
    # Visualize results
    logger.info("Generating visualization plots")
    with timer("visualization"):
        visualize_training_results(
            checkpoint_callback.get_metrics(),
            performance_callback.get_metrics(),
            env_metrics,
            output_dir=os.path.join(args.output_dir, "plots")
        )
    
    # Report overall performance
    total_time = time.time() - start_time
    logger.info(f"Total training time: {total_time:.2f} seconds")
    print_summary()
    
    return {
        "final_model_path": final_model_path,
        "reward": {
            "mean": mean_reward,
            "std": std_reward
        },
        "checkpoint_metrics": checkpoint_callback.get_metrics(),
        "performance_metrics": performance_callback.get_metrics(),
        "environment_metrics": env_metrics,
        "total_time": total_time
    }

if __name__ == "__main__":
    train()
