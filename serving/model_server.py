"""
Model serving utilities for inference in production environments.
This module provides functionality for deploying the trained model for inference.
"""

import os
import time
import logging
import numpy as np
import json
import torch
from stable_baselines3 import DQN

# Optional ONNX support
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger("taxi-rl.serving")

class ModelServer:
    """
    Server for providing inference with trained RL models.
    Supports both PyTorch and ONNX models.
    """
    def __init__(self, model_path, use_onnx=False, device="cuda"):
        """
        Initialize the model server.
        
        Args:
            model_path (str): Path to the trained model
            use_onnx (bool): Whether to use ONNX for inference
            device (str): Device to use for PyTorch inference
        """
        self.model_path = model_path
        self.use_onnx = use_onnx and ONNX_AVAILABLE
        self.device = device
        
        # Validate paths
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Initialize models based on type
        if self.use_onnx:
            # For ONNX, we need a separate ONNX file
            onnx_path = model_path.replace(".zip", ".onnx")
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")
                
            self.onnx_path = onnx_path
            self._init_onnx_model()
        else:
            # For PyTorch, we load the model directly
            self._init_pytorch_model()
            
        self.inference_count = 0
        self.total_inference_time = 0
        
        logger.info(f"Model server initialized with {'ONNX' if self.use_onnx else 'PyTorch'} model")
        
    def _init_pytorch_model(self):
        """Initialize the PyTorch model."""
        try:
            # Check if CUDA is requested but not available
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Using CPU instead.")
                self.device = "cpu"
                
            # Load the model
            self.model = DQN.load(self.model_path, device=self.device)
            logger.info(f"PyTorch model loaded from {self.model_path} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise
            
    def _init_onnx_model(self):
        """Initialize the ONNX model."""
        try:
            # Set up ONNX runtime session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Choose provider based on device
            providers = ['CUDAExecutionProvider' if self.device == "cuda" else 'CPUExecutionProvider']
            
            # Create inference session
            self.onnx_session = ort.InferenceSession(
                self.onnx_path,
                sess_options=session_options,
                providers=providers
            )
            
            logger.info(f"ONNX model loaded from {self.onnx_path}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
            
    def predict(self, observation, deterministic=True):
        """
        Make a prediction given an observation.
        
        Args:
            observation: Environment observation
            deterministic (bool): Whether to use deterministic policy
            
        Returns:
            int: Predicted action
        """
        start_time = time.time()
        
        if self.use_onnx:
            action = self._predict_onnx(observation)
        else:
            action = self._predict_pytorch(observation, deterministic)
            
        # Update metrics
        inference_time = time.time() - start_time
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        return action
        
    def _predict_pytorch(self, observation, deterministic=True):
        """
        Make a prediction using the PyTorch model.
        
        Args:
            observation: Environment observation
            deterministic (bool): Whether to use deterministic policy
            
        Returns:
            int: Predicted action
        """
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action
        
    def _predict_onnx(self, observation):
        """
        Make a prediction using the ONNX model.
        
        Args:
            observation: Environment observation
            
        Returns:
            int: Predicted action
        """
        # Prepare the input
        ort_inputs = {self.onnx_session.get_inputs()[0].name: 
                      observation.reshape(1, -1).astype(np.float32)}
        
        # Run inference
        ort_outputs = self.onnx_session.run(None, ort_inputs)
        
        # Get the action (argmax of Q-values)
        q_values = ort_outputs[0]
        action = np.argmax(q_values)
        
        return action
        
    def get_performance_metrics(self):
        """
        Get performance metrics for the model server.
        
        Returns:
            dict: Performance metrics
        """
        if self.inference_count > 0:
            avg_inference_time = self.total_inference_time / self.inference_count
            metrics = {
                "inference_count": self.inference_count,
                "total_inference_time": self.total_inference_time,
                "avg_inference_time": avg_inference_time,
                "fps": 1.0 / avg_inference_time if avg_inference_time > 0 else 0,
                "model_type": "ONNX" if self.use_onnx else "PyTorch",
                "device": self.device
            }
        else:
            metrics = {
                "inference_count": 0,
                "model_type": "ONNX" if self.use_onnx else "PyTorch",
                "device": self.device
            }
            
        return metrics

class RESTModelServer:
    """
    REST API wrapper for the ModelServer.
    This class provides a REST API interface for the model server.
    """
    def __init__(self, model_server):
        """
        Initialize the REST API server.
        
        Args:
            model_server: ModelServer instance
        """
        self.model_server = model_server
        
    def predict_json(self, json_input):
        """
        Make a prediction from JSON input.
        
        Args:
            json_input (str): JSON input containing observation
            
        Returns:
            dict: JSON response with prediction
        """
        try:
            # Parse the input
            input_data = json.loads(json_input)
            observation = np.array(input_data["observation"])
            deterministic = input_data.get("deterministic", True)
            
            # Make prediction
            action = self.model_server.predict(observation, deterministic)
            
            # Prepare response
            response = {
                "action": int(action),
                "status": "success"
            }
            
        except Exception as e:
            response = {
                "status": "error",
                "error": str(e)
            }
            
        return json.dumps(response)
        
    def get_metrics_json(self):
        """
        Get performance metrics as JSON.
        
        Returns:
            str: JSON metrics
        """
        metrics = self.model_server.get_performance_metrics()
        return json.dumps(metrics)
