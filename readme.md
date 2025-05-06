# RL-Based Taxi Pricing System

## Project Overview
This project implements a reinforcement learning-based system for optimizing taxi pricing and driver decision-making. It demonstrates advanced Python techniques for performance optimization, including:
- Numba JIT compilation for numerical operations
- Caching strategies for repeated calculations
- Parallel processing with ThreadPoolExecutor
- Vectorized operations with NumPy
- PySpark optimization for big data processing
- GPU acceleration with PyTorch
- ONNX optimization for inference

The system uses Deep Q-Learning (DQN) to train an agent that learns to make optimal decisions about which rides to accept based on factors like fare amount, distance, time of day, and surge pricing.

## Project Structure
```
smartcab_rl/
│
├── config/
│   ├── __init__.py
│   └── config_manager.py       # Configuration management
│
├── data/
│   ├── __init__.py
│   └── ride_pool_builder.py    # Data processing with PySpark
│
├── api/
│   ├── __init__.py
│   └── uber_api.py             # API wrapper with optimization
│
├── environment/
│   ├── __init__.py
│   └── taxi_env.py             # RL environment
│
├── models/
│   ├── __init__.py
│   ├── custom_dqn.py           # Custom DQN architecture
│   └── callbacks.py            # Training callbacks
│
├── utils/
│   ├── __init__.py
│   ├── performance.py          # Performance monitoring
│   └── visualization.py        # Plotting utilities
│
├── serving/
│   ├── __init__.py
│   └── model_server.py         # Model serving for inference
│
├── main.py                     # Main execution script
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for training)
- At least 8GB RAM
- 20GB free disk space

### Dependencies
Key dependencies and their minimum versions:
- numpy >= 1.20.0
- pandas >= 1.3.0
- pyspark >= 3.1.2
- torch >= 1.9.0
- stable-baselines3 >= 1.5.0
- gymnasium >= 0.28.1
- shimmy >= 2.0.0
- numba >= 0.53.1
- onnx >= 1.10.1
- onnxruntime >= 1.8.1

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/rl-taxi-pricing.git
cd rl-taxi-pricing
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create required directories:
```bash
mkdir -p data/raw
mkdir -p output/models
mkdir -p output/plots
```

## Data Preparation
The system expects parquet files containing taxi trip data with the following columns:
- `tpep_pickup_datetime`: Pickup timestamp
- `tpep_dropoff_datetime`: Dropoff timestamp
- `PULocationID`: Pickup location ID
- `DOLocationID`: Dropoff location ID
- `fare_amount`: Trip fare
- `trip_distance`: Trip distance

Place your parquet files in the `data/raw` directory.

## Usage

### Training a Model
```bash
python train.py --config config/default_config.yaml --data data/raw/taxi_data*.parquet --output_dir output/training
```

#### Training Options
- `--config`: Path to configuration file
- `--data`: Path(s) to parquet data file(s)
- `--output_dir`: Directory for output files
- `--checkpoint`: Path to checkpoint to resume from
- `--timesteps`: Number of training timesteps
- `--device`: Device to use (cuda or cpu)
- `--checkpoint_freq`: Checkpoint frequency in timesteps

### Evaluating a Model
```bash
python evaluate.py --model output/training/final_model.zip --data data/raw/taxi_data_test.parquet --output_dir output/evaluation
```

#### Evaluation Options
- `--model`: Path to trained model file
- `--data`: Path(s) to parquet data file(s)
- `--output_dir`: Directory for output files
- `--n_episodes`: Number of episodes for evaluation
- `--device`: Device to use (cuda or cpu)
- `--use_onnx`: Use ONNX model for inference
- `--render`: Render evaluations
- `--record_actions`: Record all actions and rewards

### Full Pipeline Execution
```bash
python main.py --config config/default_config.yaml --data data/raw/taxi_data*.parquet --output_dir output/pipeline
```

#### Main Options
- `--config`: Path to configuration file
- `--data`: Path(s) to parquet data file(s)
- `--output_dir`: Directory for output files
- `--timesteps`: Number of training timesteps
- `--seed`: Random seed
- `--no_gpu`: Disable GPU usage
- `--no_numba`: Disable Numba JIT compilation
- `--no_parallel`: Disable parallel processing
- `--no_cache`: Disable caching
- `--no_onnx`: Disable ONNX export

## Performance Optimizations
This project demonstrates several advanced Python performance optimization techniques:

1. **Numba JIT Compilation**:
   - Used for accelerating numerical calculations in the API simulation logic
   - Implementation in `api/uber_api.py`

2. **LRU Caching**:
   - Applied to frequently called functions like API simulations
   - Implementation in `api/uber_api.py`

3. **Parallel Processing**:
   - ThreadPoolExecutor for parallel API requests
   - Implementation in `api/uber_api.py`

4. **Vectorized Operations**:
   - NumPy vectorization for efficient observation construction
   - Pre-allocated arrays for better memory usage
   - Implementation in `environment/taxi_env.py`

5. **PySpark Optimizations**:
   - Proper Spark configuration
   - Caching strategies for repeated operations
   - Method chaining for better readability and performance
   - Implementation in `data/ride_pool_builder.py`

6. **GPU Acceleration**:
   - PyTorch with CUDA for faster model training
   - Implementation in `models/custom_dqn.py`

7. **ONNX Optimization**:
   - ONNX export for faster inference
   - Implementation in `models/custom_dqn.py` and `serving/model_server.py`

## AWS Deployment

### Prerequisites
- AWS account with appropriate permissions
- AWS CLI installed and configured
- IAM role with S3 and EC2 access

### Deployment Steps
1. Build the project as a Python package:
```bash
pip install setuptools wheel
python setup.py sdist bdist_wheel
```

2. Create an AWS S3 bucket for storing models and data:
```bash
aws s3 mb s3://rl-taxi-pricing
```

3. Upload the package and models to S3:
```bash
aws s3 cp dist/rl_taxi_pricing-0.1.0.tar.gz s3://rl-taxi-pricing/
aws s3 cp output/training/final_model.zip s3://rl-taxi-pricing/models/
```

4. Create an EC2 instance with the required dependencies:
```bash
# Use AWS Console or CloudFormation to create an EC2 instance
# Make sure to include GPU support if needed (e.g., g4dn instance)
```

5. Install the package on the EC2 instance:
```bash
pip install s3://rl-taxi-pricing/rl_taxi_pricing-0.1.0.tar.gz
```

6. Download the model:
```bash
aws s3 cp s3://rl-taxi-pricing/models/final_model.zip ./
```

7. Run inference:
```bash
python -m rl_taxi_pricing.evaluate --model final_model.zip --data s3://your-data-bucket/test_data.parquet
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **CUDA Out of Memory**
   - Reduce batch size in config file
   - Use gradient accumulation
   - Try running on CPU with `--device cpu`

2. **PySpark Memory Issues**
   - Adjust Spark memory configuration in `config/config_manager.py`
   - Increase executor memory
   - Implement data partitioning

3. **Slow Training Performance**
   - Enable Numba JIT compilation
   - Verify GPU utilization with `nvidia-smi`
   - Check data loading bottlenecks

4. **Model Convergence Issues**
   - Adjust learning rate in configuration
   - Modify reward scaling
   - Check for data quality issues

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`python -m pytest tests/`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions
- Add unit tests for new features

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- This project was developed as part of the Advanced Python for Data Science course
- Thanks to the Stable-Baselines3 team for their RL implementation
- Special thanks to all contributors

## Contact
For questions and support, please:
- Open an issue in the GitHub repository
- Contact the maintainers
- Join our Discord community (link in project wiki)
