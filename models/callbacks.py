"""
Custom callbacks for model training and evaluation.
"""

import os
import time
import numpy as np
import logging
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

logger = logging.getLogger("taxi-rl.callbacks")

class CustomCheckpointCallback(BaseCallback):
    """
    Custom callback for saving models and evaluating at specified intervals.
    """
    def __init__(
        self,
        eval_env,
        checkpoint_dir: str,
        check_freq: int = 1000,
        n_eval_episodes: int = 5,
        verbose: int = 1
    ):
        """
        Initialize the callback.
        
        Args:
            eval_env: Environment for evaluation
            checkpoint_dir (str): Directory to save checkpoints
            check_freq (int): Frequency of checkpoints in timesteps
            n_eval_episodes (int): Number of episodes for evaluation
            verbose (int): Verbosity level
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.checkpoint_dir = checkpoint_dir
        self.n_eval_episodes = n_eval_episodes
        self.eval_env = eval_env
        self.best_mean_reward = -float('inf')
        self.checkpoint_metrics = {}

        # Create directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Check if we should save a checkpoint and evaluate.
        
        Returns:
            bool: Whether to continue training
        """
        if self.n_calls % self.check_freq == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_{self.n_calls}.zip"
            )

            # Save checkpoint
            self.model.save(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Evaluate model
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )

            # Update best model if needed
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_path = os.path.join(self.checkpoint_dir, "best_model.zip")
                self.model.save(best_path)
                logger.info(f"New best model with reward {mean_reward:.2f} saved to {best_path}")

            # Record metrics
            self.checkpoint_metrics[self.n_calls] = {
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "checkpoint_path": checkpoint_path,
                "timestamp": time.time()
            }

            # Print info
            if self.verbose > 0:
                print(f"\nCheckpoint at {self.n_calls} steps:")
                print(f"- Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"- Checkpoint saved to: {checkpoint_path}")
                print(f"- Best mean reward so far: {self.best_mean_reward:.2f}")

        return True

    def get_metrics(self):
        """
        Get all recorded metrics.
        
        Returns:
            dict: Checkpoint metrics
        """
        return self.checkpoint_metrics

class PerformanceMonitorCallback(BaseCallback):
    """
    Callback to monitor and log training performance metrics.
    """
    def __init__(self, log_freq=1000, verbose=1):
        """
        Initialize the performance monitor.
        
        Args:
            log_freq (int): Frequency of logging in timesteps
            verbose (int): Verbosity level
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.start_time = None
        self.last_time = None
        self.last_calls = 0
        self.fps_history = []
        self.metrics = {}

    def _on_training_start(self):
        """Initialize timers when training starts."""
        self.start_time = time.time()
        self.last_time = self.start_time

    def _on_step(self):
        """Record performance metrics at each step."""
        # Initialize if not done yet
        if self.start_time is None:
            self._on_training_start()

        # Calculate FPS periodically
        if self.n_calls % self.log_freq == 0:
            current_time = time.time()
            elapsed = current_time - self.last_time
            steps = self.n_calls - self.last_calls
            
            # Calculate frames per second
            fps = steps / elapsed if elapsed > 0 else 0
            self.fps_history.append(fps)
            
            # Update metrics
            self.metrics[self.n_calls] = {
                "fps": fps,
                "elapsed_time": current_time - self.start_time,
                "steps": self.n_calls
            }
            
            # Print metrics if verbose
            if self.verbose > 0:
                print(f"Steps: {self.n_calls}, FPS: {fps:.1f}, "
                      f"Total elapsed: {current_time - self.start_time:.2f}s")
            
            # Update for next calculation
            self.last_time = current_time
            self.last_calls = self.n_calls
            
        return True

    def get_metrics(self):
        """
        Get all recorded performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        return {
            "fps_history": self.fps_history,
            "metrics": self.metrics,
            "mean_fps": np.mean(self.fps_history) if self.fps_history else 0,
            "total_time": time.time() - self.start_time if self.start_time else 0,
            "total_steps": self.n_calls
        }
