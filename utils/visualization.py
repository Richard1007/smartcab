"""
Visualization utilities for plotting training results and performance metrics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("taxi-rl.visualization")

def plot_learning_curve(metrics, save_path=None, show=False):
    """
    Plot the learning curve from checkpoint metrics.
    
    Args:
        metrics (dict): Checkpoint metrics from the callback
        save_path (str, optional): Path to save the figure
        show (bool): Whether to display the plot
        
    Returns:
        plt.Figure: The matplotlib figure
    """
    # Extract metrics
    timesteps = sorted(metrics.keys())
    rewards = [metrics[t]["mean_reward"] for t in timesteps]
    stds = [metrics[t]["std_reward"] for t in timesteps]

    # Create figure
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    # Plot mean reward with standard deviation band
    ax.plot(timesteps, rewards, 'b-', label='Mean Reward')
    ax.fill_between(
        timesteps,
        [r - s for r, s in zip(rewards, stds)],
        [r + s for r, s in zip(rewards, stds)],
        color='blue', alpha=0.2, label='Standard Deviation'
    )
    
    # Add labels and grid
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Learning Curve')
    ax.grid(True)
    ax.legend()

    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Learning curve saved to {save_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def plot_performance_metrics(performance_metrics, save_path=None, show=False):
    """
    Plot performance metrics over time.
    
    Args:
        performance_metrics (dict): Performance metrics from the monitor callback
        save_path (str, optional): Path to save the figure
        show (bool): Whether to display the plot
        
    Returns:
        plt.Figure: The matplotlib figure
    """
    # Extract metrics
    fps_history = performance_metrics.get("fps_history", [])
    metrics = performance_metrics.get("metrics", {})
    timesteps = sorted(metrics.keys())
    fps_values = [metrics[t]["fps"] for t in timesteps]
    elapsed_times = [metrics[t]["elapsed_time"] for t in timesteps]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot FPS over time
    ax1.plot(timesteps, fps_values, 'g-')
    ax1.set_ylabel('Frames Per Second')
    ax1.set_title('Training Performance Metrics')
    ax1.grid(True)
    
    # Plot elapsed time
    ax2.plot(timesteps, elapsed_times, 'r-')
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Elapsed Time (s)')
    ax2.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Performance metrics plot saved to {save_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def plot_environment_metrics(env_metrics, save_path=None, show=False):
    """
    Plot environment performance metrics.
    
    Args:
        env_metrics (dict): Environment metrics
        save_path (str, optional): Path to save the figure
        show (bool): Whether to display the plot
        
    Returns:
        plt.Figure: The matplotlib figure
    """
    # Extract metrics
    metrics = [
        ("avg_step_time", "Average Step Time (s)"),
        ("avg_get_rides_time", "Average Get Rides Time (s)"),
        ("avg_get_obs_time", "Average Get Observation Time (s)")
    ]
    
    # Create figure
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    # Extract values
    labels = [label for _, label in metrics]
    values = [env_metrics.get(key, 0) for key, _ in metrics]
    
    # Plot bar chart
    bars = ax.bar(labels, values)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Add labels and grid
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Environment Performance Metrics')
    ax.grid(True, axis='y')
    
    # Add cache hit rate if available
    cache_stats = env_metrics.get("api_cache", {})
    if cache_stats:
        plt.figure(figsize=(8, 4))
        hit_rate = cache_stats.get("hit_rate", 0) * 100  # Convert to percentage
        plt.pie([hit_rate, 100-hit_rate], 
                labels=['Cache Hits', 'Cache Misses'],
                autopct='%1.1f%%', 
                colors=['#4CAF50', '#F44336'],
                startangle=90)
        plt.axis('equal')
        plt.title('API Cache Performance')
        
        if save_path:
            cache_path = save_path.replace('.png', '_cache.png')
            plt.savefig(cache_path)
            logger.info(f"Cache performance plot saved to {cache_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Environment metrics plot saved to {save_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def visualize_training_results(checkpoint_metrics, performance_metrics, env_metrics, 
                               output_dir='./plots', show=False):
    """
    Generate and save all visualization plots.
    
    Args:
        checkpoint_metrics (dict): Checkpoint metrics
        performance_metrics (dict): Performance metrics
        env_metrics (dict): Environment metrics
        output_dir (str): Directory to save plots
        show (bool): Whether to display the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot learning curve
    plot_learning_curve(
        checkpoint_metrics, 
        save_path=os.path.join(output_dir, 'learning_curve.png'),
        show=show
    )
    
    # Plot performance metrics
    plot_performance_metrics(
        performance_metrics,
        save_path=os.path.join(output_dir, 'performance_metrics.png'),
        show=show
    )
    
    # Plot environment metrics
    plot_environment_metrics(
        env_metrics,
        save_path=os.path.join(output_dir, 'environment_metrics.png'),
        show=show
    )
    
    logger.info(f"All visualization plots saved to {output_dir}")
