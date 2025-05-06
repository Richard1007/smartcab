"""
Custom DQN architecture with GPU acceleration support.
This module demonstrates techniques from Week 12 (Python for GPUs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger("taxi-rl.models")

class CustomDQN(nn.Module):
    """
    Custom DQN architecture with GPU acceleration support.
    Implements a dueling DQN architecture for improved performance.
    """
    def __init__(self, observation_space, action_space):
        """
        Initialize the network architecture.
        
        Args:
            observation_space: Gym observation space
            action_space: Gym action space
        """
        super().__init__()

        input_dim = observation_space.shape[0]
        output_dim = action_space.n

        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Dueling architecture for better performance
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            tensor: Q-values for each action
        """
        features = self.features(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return value + (advantages - advantages.mean(dim=1, keepdim=True))

def create_dqn_model(env, config, device="cuda"):
    """
    Create a DQN model with the specified configuration.
    
    Args:
        env: Gym environment
        config: Configuration dictionary
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        DQN model
    """
    from stable_baselines3 import DQN
    
    # Check if GPU is available
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Using CPU instead.")
        device = "cpu"
    
    logger.info(f"Creating DQN model on {device}")

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=config['model']['learning_rate'],
        buffer_size=config['model']['buffer_size'],
        learning_starts=config['model']['learning_starts'],
        batch_size=config['model']['batch_size'],
        gamma=config['model']['gamma'],
        train_freq=config['model']['train_freq'],
        gradient_steps=config['model']['gradient_steps'],
        target_update_interval=config['model']['target_update_interval'],
        exploration_fraction=config['model']['exploration_fraction'],
        exploration_initial_eps=config['model']['exploration_initial_eps'],
        exploration_final_eps=config['model']['exploration_final_eps'],
        device=device,  # Use GPU if available
        verbose=1
    )
    
    return model

def export_to_onnx(model, env, path, device="cuda"):
    """
    Export the model to ONNX format for inference optimization.
    
    Args:
        model: DQN model
        env: Gym environment
        path: Path to save ONNX model
        device: Device to use
        
    Returns:
        bool: Whether export was successful
    """
    try:
        import onnx
        import onnxruntime as ort
        
        logger.info("Exporting model to ONNX format...")
        
        # Extract policy network
        q_net = model.q_net
        
        # Create dummy input
        dummy_input = torch.randn(1, env.observation_space.shape[0], device=device)
        
        # Export model
        torch.onnx.export(
            q_net,
            dummy_input,
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )
        
        # Verify model
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"Model successfully exported to {path}")
        return True
        
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return False

def benchmark_onnx(model, env, onnx_path, device="cuda", n_runs=100):
    """
    Benchmark ONNX inference against PyTorch inference.
    
    Args:
        model: DQN model
        env: Gym environment
        onnx_path: Path to ONNX model
        device: Device to use
        n_runs: Number of runs for benchmarking
        
    Returns:
        dict: Benchmark results
    """
    try:
        import onnxruntime as ort
        import time
        
        logger.info("Benchmarking ONNX inference performance...")
        
        # Create ONNX session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = ort.InferenceSession(
            onnx_path,
            sess_options=session_options,
            providers=['CUDAExecutionProvider' if device == "cuda" else 'CPUExecutionProvider']
        )
        
        # Run inference test
        obs = env.reset()
        
        # PyTorch inference time
        torch_start = time.time()
        for _ in range(n_runs):
            with torch.no_grad():
                torch_input = torch.FloatTensor(obs).unsqueeze(0).to(device)
                _ = model.q_net(torch_input)
        torch_time = time.time() - torch_start
        
        # ONNX inference time
        onnx_start = time.time()
        for _ in range(n_runs):
            ort_inputs = {ort_session.get_inputs()[0].name: obs.reshape(1, -1).astype(np.float32)}
            _ = ort_session.run(None, ort_inputs)
        onnx_time = time.time() - onnx_start
        
        # Report performance comparison
        speedup = torch_time / onnx_time if onnx_time > 0 else 0
        
        results = {
            "torch_time": torch_time,
            "onnx_time": onnx_time,
            "speedup": speedup,
            "n_runs": n_runs
        }
        
        logger.info(f"PyTorch inference ({n_runs} runs): {torch_time:.4f} seconds")
        logger.info(f"ONNX inference ({n_runs} runs): {onnx_time:.4f} seconds")
        logger.info(f"Speedup factor: {speedup:.2f}x")
        
        return results
        
    except Exception as e:
        logger.error(f"ONNX benchmarking failed: {e}")
        return {"error": str(e)}
