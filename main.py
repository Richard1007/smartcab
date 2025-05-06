"""
Main execution script for the RL-based taxi pricing system.
This script orchestrates the entire pipeline from data processing to model training.
"""

import os
import argparse
import logging
import yaml
import time

from config.config_manager import ConfigManager
from data.ride_pool_builder import OptimizedRidePoolBuilder
from environment.taxi_env import OptimizedTaxiEnv
from models.custom_dqn import create_dqn_model, export_to_onnx, benchmark_onnx
from models.callbacks import CustomCheckpointCallback, PerformanceMonitorCallback
from utils.performance import timer, print_summary
from utils.visualization import visualize_training_results
from stable_baselines3.common.evaluation import evaluate_policy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("taxi-rl.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("taxi-rl.main")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RL-based Taxi Pricing System")
    
    parser.add_argument("--config", type=str, default="config/colab_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data", type=str, nargs='+', required=True,
                        help="Path(s) to parquet data file(s)")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory for output files")
    parser.add_argument("--timesteps", type=int, 
                        help="Number of training timesteps (overrides config)")
    parser.add_argument("--seed", type=int,
                        help="Random seed (overrides config)")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Disable GPU usage")
    parser.add_argument("--no_numba", action="store_true",
                        help="Disable Numba JIT compilation")
    parser.add_argument("--no_parallel", action="store_true",
                        help="Disable parallel processing")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable caching")
    parser.add_argument("--no_onnx", action="store_true",
                        help="Disable ONNX export")
    parser.add_argument("--checkpoint_freq", type=int, default=1000,
                        help="Checkpoint frequency in timesteps")
    
    return parser.parse_args()

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Start timer for overall execution
    start_time = time.time()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or create configuration
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
        
    if args.seed:
        config_manager.update_config("env.seed", args.seed)
    
    # Update performance settings
    config_manager.update_config("performance.use_gpu", not args.no_gpu)
    config_manager.update_config("performance.use_numba", not args.no_numba)
    config_manager.update_config("performance.parallel_processing", not args.no_parallel)
    config_manager.update_config("performance.use_cache", not args.no_cache)
    config_manager.update_config("performance.onnx_optimization", not args.no_onnx)
    
    # Save updated configuration
    config_manager.save_config(os.path.join(args.output_dir, "config.yaml"))
    
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
    
    # Create model
    logger.info("Creating DQN model")
    device = "cuda" if config["performance"]["use_gpu"] else "cpu"
    with timer("model_creation"):
        model = create_dqn_model(env, config, device=device)
    
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
            callback=[checkpoint_callback, performance_callback]
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
    
    # Export to ONNX if enabled
    if config["performance"]["onnx_optimization"]:
        logger.info("Exporting model to ONNX format")
        onnx_path = os.path.join(args.output_dir, "model.onnx")
        with timer("onnx_export"):
            export_success = export_to_onnx(model, env, onnx_path, device=device)
            
        if export_success:
            logger.info("Benchmarking ONNX inference")
            with timer("onnx_benchmark"):
                benchmark_results = benchmark_onnx(model, env, onnx_path, device=device)
                
            logger.info(f"ONNX speedup: {benchmark_results.get('speedup', 0):.2f}x")
    
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
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    print_summary()
    
    # Return results as dictionary
    return {
        "final_model_path": final_model_path,
        "reward": {
            "mean": mean_reward,
            "std": std_reward
        },
        "checkpoints": checkpoint_callback.get_metrics(),
        "performance": performance_callback.get_metrics(),
        "environment": env_metrics,
        "build_metrics": build_metrics,
        "total_time": total_time
    }

if __name__ == "__main__":
    main()
