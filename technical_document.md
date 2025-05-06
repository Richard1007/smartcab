# RL-Based Taxi Pricing System: Technical Documentation

## System Architecture

The RL-Based Taxi Pricing System is a modular application that uses reinforcement learning to optimize taxi driver decision-making. The architecture is designed to be highly performant, leveraging various advanced Python optimization techniques from the DS-GA 1019 Advanced Python for Data Science course.

### Key Components

1. **Data Processing Module**
   - `OptimizedRidePoolBuilder`: Processes raw taxi data using optimized PySpark operations
   - Implements data filtering, feature engineering, and efficient transformations

2. **API Simulation Module**
   - `OptimizedUberAPI`: Simulates real-time data from a ride-sharing service
   - Implements caching, parallelization, and JIT compilation for performance

3. **RL Environment**
   - `OptimizedTaxiEnv`: Custom Gym environment with performance optimizations
   - Implements vectorized operations and precomputation strategies

4. **ML Models**
   - `CustomDQN`: Deep Q-Network with GPU acceleration
   - Implements export to ONNX for optimized inference

5. **Serving & Inference**
   - `ModelServer`: Provides a unified interface for model inference
   - Supports both PyTorch and ONNX runtime for optimal performance

## Performance Optimization Techniques

The system incorporates numerous performance optimization techniques from the course:

### 1. Python Performance Tips (Week 2)

**Implementation Location**: `api/uber_api.py`

The `OptimizedUberAPI` class uses LRU caching for repeated API calls:

```python
@lru_cache(maxsize=1024)
def simulate_real_time_data_cached(self, location_id, hour_of_day):
    """Cached version for repeated calls"""
    self.cache_hits += 1
    return self._simulate_real_time_data_impl(location_id, hour_of_day)
```

This technique drastically reduces computation time for frequently accessed data patterns.

Additional implementations:
- Precomputation of lookup tables in `taxi_env.py`
- Method chaining in `ride_pool_builder.py`
- Efficient data structures throughout the codebase

### 2. Numba JIT Compilation (Week 6)

**Implementation Location**: `api/uber_api.py`

The system uses Numba's just-in-time compilation to accelerate numerical calculations:

```python
@staticmethod
@jit(nopython=True)
def _calculate_surge_and_eta_numba(location_id, hour_of_day):
    """Numba-optimized calculation of surge and ETA"""
    # JIT-compiled implementation
```

This technique provides near-C performance for numerical operations without manual C/C++ code.

### 3. Python Concurrency (Week 9)

**Implementation Location**: `api/uber_api.py`

The system implements parallel API requests using ThreadPoolExecutor:

```python
def batch_simulate(self, requests):
    """Perform simulations in parallel using ThreadPoolExecutor"""
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(self.simulate_real_time_data, loc_id, hour)
            for loc_id, hour in requests
        ]
        return [future.result() for future in futures]
```

This technique significantly improves throughput for batch operations.

### 4. Vectorized Operations (Week 4 & 8)

**Implementation Location**: `environment/taxi_env.py`

The environment uses vectorized NumPy operations for observation construction:

```python
def _get_observation(self):
    """Create observation using vectorized operations for performance"""
    # Pre-allocate numpy array instead of list building
    ride_features = np.zeros(8 * self.max_rides, dtype=np.float32)
    
    if self.available_rides:
        # Extract ride properties in a vectorized way
        for i, r in enumerate(self.available_rides[:self.max_rides]):
            ride_features[i*8:(i+1)*8] = [
                # Feature values
            ]
```

This technique improves performance by leveraging NumPy's optimized C implementation.

### 5. PySpark Optimization (Week 13)

**Implementation Location**: `data/ride_pool_builder.py`

The data processing module implements several PySpark optimizations:

```python
# Optimized Spark session configuration
self.spark = (SparkSession.builder
             .appName("OptimizedRidePool")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .config("spark.driver.memory", "4g")
             .config("spark.sql.shuffle.partitions", "20")
             .getOrCreate())

# Efficient transformations with method chaining and early filtering
df_clean = (df.select(*needed_columns)
           .filter((col("fare_amount") > 0) & ...)
           .cache())  # Cache for repeated use
```

These techniques optimize PySpark for big data processing.

### 6. GPU Acceleration (Week 12)

**Implementation Location**: `models/custom_dqn.py`

The model implements custom neural network architecture with GPU support:

```python
class CustomDQN(nn.Module):
    """Custom DQN architecture with GPU acceleration support."""
    def __init__(self, observation_space, action_space):
        # Implementation with dueling architecture
```

The system also includes ONNX export functionality for optimized inference:

```python
def export_to_onnx(model, env, path, device="cuda"):
    """Export the model to ONNX format for inference optimization."""
    # Implementation
```

These techniques leverage GPU for faster model training and inference.

## Modular Code Organization

The codebase is organized into modules with clear separation of concerns:

### 1. Configuration Management
**File**: `config/config_manager.py`
- Handles loading, validation, and access to configuration parameters
- Provides a centralized configuration interface

### 2. Data Processing
**File**: `data/ride_pool_builder.py`
- Processes raw taxi data with PySpark
- Implements optimized data transformations and feature engineering

### 3. API Simulation
**File**: `api/uber_api.py`
- Simulates real-time data from a ride-sharing service
- Implements performance optimizations with caching, Numba, and parallelization

### 4. RL Environment
**File**: `environment/taxi_env.py`
- Custom Gym environment for taxi pricing
- Implements vectorized operations and performance monitoring

### 5. ML Models
**Files**: `models/custom_dqn.py`, `models/callbacks.py`
- Custom DQN architecture with GPU acceleration
- Training callbacks for checkpointing and monitoring

### 6. Utilities
**Files**: `utils/performance.py`, `utils/visualization.py`
- Performance monitoring and profiling
- Visualization utilities for plotting results

### 7. Model Serving
**File**: `serving/model_server.py`
- Provides a unified interface for model inference
- Supports both PyTorch and ONNX for optimal performance

### 8. Main Execution Scripts
**Files**: `main.py`, `train.py`, `evaluate.py`
- Entry points for different execution modes
- Command-line interfaces for flexibility

## Execution Flow

### Training Pipeline

1. **Configuration**:
   - Load configuration from file or create default
   - Override with command-line arguments

2. **Data Processing**:
   - Load and process taxi data with `OptimizedRidePoolBuilder`
   - Apply filtering and feature engineering

3. **Environment Creation**:
   - Create training and evaluation environments
   - Configure with performance optimizations

4. **Model Creation**:
   - Create DQN model with GPU support
   - Configure learning parameters

5. **Training**:
   - Train the model with callbacks for monitoring
   - Save checkpoints during training

6. **Evaluation**:
   - Evaluate the trained model
   - Generate performance metrics

7. **Visualization**:
   - Create learning curves and performance plots
   - Save results to output directory

### Inference Pipeline

1. **Model Loading**:
   - Load trained model (PyTorch or ONNX)
   - Initialize model server

2. **Environment Creation**:
   - Create evaluation environment
   - Configure with performance optimizations

3. **Inference**:
   - Make predictions with the model
   - Collect performance metrics

4. **Visualization**:
   - Create performance plots
   - Save results to output directory

## AWS Deployment Guidelines

For deployment to AWS, the modular architecture provides several options:

### Option 1: EC2 Deployment

1. **Package Preparation**:
   - Create a Python package from the codebase
   - Upload to S3 or use direct installation

2. **EC2 Instance Setup**:
   - Choose an appropriate instance type (e.g., g4dn for GPU support)
   - Install dependencies from requirements.txt

3. **Model Deployment**:
   - Upload trained models to S3
   - Download to EC2 for inference

4. **Execution**:
   - Run the evaluation script for inference
   - Collect and visualize results

### Option 2: SageMaker Deployment

1. **Model Preparation**:
   - Export the model to ONNX format
   - Create a SageMaker model artifact

2. **SageMaker Setup**:
   - Create a SageMaker endpoint configuration
   - Deploy the model to an endpoint

3. **Inference**:
   - Send requests to the SageMaker endpoint
   - Process responses as needed

### Option 3: Lambda + API Gateway

For serverless inference (with limitations):

1. **Model Optimization**:
   - Export to ONNX and optimize for size
   - Ensure model fits within Lambda limits

2. **Lambda Function**:
   - Create a Lambda function for inference
   - Include only necessary dependencies

3. **API Gateway**:
   - Create an API Gateway to expose the Lambda
   - Configure appropriate request/response mappings

## Performance Monitoring

The system includes comprehensive performance monitoring:

1. **Training Performance**:
   - Frames per second (FPS)
   - Training time per step
   - Learning progress

2. **Inference Performance**:
   - Inference latency
   - Throughput (requests per second)
   - GPU vs. CPU comparison
   - ONNX vs. PyTorch comparison

3. **Environment Performance**:
   - Step time
   - Observation construction time
   - API request time
   - Cache hit rate

## Conclusion

The RL-Based Taxi Pricing System demonstrates numerous advanced Python optimization techniques from the DS-GA 1019 course. The modular architecture makes it easy to deploy to AWS and adapt to different requirements. The performance optimizations provide significant speedups for both training and inference.
