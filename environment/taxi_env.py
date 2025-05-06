"""
Optimized Taxi Environment for Reinforcement Learning.
This module demonstrates techniques from Week 4 (Performance Tuning) 
and Week 8 (Optimization in Python).
"""

import time
import numpy as np
import gym
from gym import spaces
import logging

from api.uber_api import OptimizedUberAPI

logger = logging.getLogger("taxi-rl.environment")

class OptimizedTaxiEnv(gym.Env):
    """
    Optimized Taxi environment with vectorized operations and performance enhancements.
    Demonstrates vectorized numpy operations and performance optimization techniques.
    """
    def __init__(
        self,
        ride_pool_df,
        api_key,
        max_rides=10,
        seed=42,
        use_cache=True,
        use_numba=True,
        use_parallel=True
    ):
        """
        Initialize the Taxi environment.
        
        Args:
            ride_pool_df: PySpark DataFrame with ride data
            api_key (str): API key for the Uber API
            max_rides (int): Maximum number of rides to consider
            seed (int): Random seed for reproducibility
            use_cache (bool): Whether to use caching in the API
            use_numba (bool): Whether to use Numba in the API
            use_parallel (bool): Whether to use parallel processing
        """
        super().__init__()
        self.ride_pool = ride_pool_df
        self.max_rides = max_rides

        # Performance options
        self.use_cache = use_cache
        self.use_numba = use_numba
        self.use_parallel = use_parallel

        # Set random seed
        self.seed_value = seed
        np.random.seed(seed)

        # Initialize state variables
        self.hours = sorted(
            ride_pool_df.select("pickup_hour").distinct().rdd.flatMap(lambda x: x).collect()
        )
        self.days = sorted(
            ride_pool_df.select("pickup_day").distinct().rdd.flatMap(lambda x: x).collect()
        )

        # Precompute hour indices for easier lookup (Week 2: Performance Tips)
        self.hour_to_idx = {hour: idx for idx, hour in enumerate(self.hours)}
        self.day_to_idx = {day: idx for idx, day in enumerate(self.days)}

        # Precompute cyclical time features (Week 8: Optimization in Python)
        self._precompute_time_features()

        self.current_hour_idx = 0
        self.current_day_idx = 0
        self.current_location = None
        self.available_rides = []
        self.total_reward = 0
        self.episode_rides = 0

        # Maximum location ID
        self.max_location_id = 263

        # Initialize API wrapper
        self.api = OptimizedUberAPI(
            api_key,
            use_cache=self.use_cache,
            use_numba=self.use_numba
        )

        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.max_rides + 1)  # +1 for 'wait' action

        # Observation space
        # [hour_sin, hour_cos, day_sin, day_cos, current_location, is_high_demand_location] +
        # [ride features] * max_rides
        single_ride_features = 8
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6 + single_ride_features * self.max_rides,),
            dtype=np.float32
        )

        # Reward function parameters
        self.reward_params = {
            "fare_weight": 1.0,
            "time_cost_weight": -0.1,
            "surge_bonus_weight": 0.2,
            "high_demand_bonus": 2.0,
            "waiting_penalty": -1.0
        }

        # Performance metrics
        self.perf_metrics = {
            "get_rides_time": 0.0,
            "get_obs_time": 0.0,
            "step_time": 0.0,
            "steps_taken": 0
        }

    def _precompute_time_features(self):
        """Precompute cyclical time features for better performance."""
        # Hour features
        self.hour_sin = np.sin(2 * np.pi * np.array(self.hours) / 24)
        self.hour_cos = np.cos(2 * np.pi * np.array(self.hours) / 24)

        # Day features
        self.day_sin = np.sin(2 * np.pi * np.array(self.days) / 7)
        self.day_cos = np.cos(2 * np.pi * np.array(self.days) / 7)

    def seed(self, seed=None):
        """Set the random seed for the environment."""
        if seed is not None:
            self.seed_value = seed
            np.random.seed(seed)
        return [self.seed_value]

    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            numpy.ndarray: Initial observation
        """
        self.current_hour_idx = 0
        self.current_day_idx = 0
        self.total_reward = 0
        self.episode_rides = 0

        # Start at a random location
        self.current_location = np.random.randint(1, self.max_location_id)

        # Get available rides
        self.available_rides = self._get_rides_for_hour(
            self.hours[self.current_hour_idx],
            self.days[self.current_day_idx]
        )

        return self._get_observation()

    def step(self, action):
        """
        Take a step in the environment based on the action.
        
        Args:
            action (int): Action to take (index of ride to accept or wait)
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        start_time = time.time()

        # Initialize info dictionary
        info = {}

        # Check if action is valid
        if action > len(self.available_rides):
            action = len(self.available_rides)  # Default to wait if invalid

        # Process 'wait' action
        if action == len(self.available_rides) or action == self.max_rides:
            reward = self.reward_params["waiting_penalty"]
            info["action_type"] = "wait"
        else:
            # Accept ride
            ride = self.available_rides[action]

            # Update state
            self.current_location = ride["DOLocationID"]
            self.episode_rides += 1

            # Calculate reward
            reward = self._calculate_reward(ride)

            # Add info
            info["action_type"] = "ride"
            info["fare"] = ride["fare_amount"]
            info["duration"] = ride["trip_duration_minutes"]
            info["surge"] = ride.get("surge_multiplier", 1.0)

        # Update time
        self.current_hour_idx = (self.current_hour_idx + 1) % len(self.hours)

        # Update day if we've gone through all hours
        if self.current_hour_idx == 0:
            self.current_day_idx = (self.current_day_idx + 1) % len(self.days)

        # Check if episode is done
        done = self.current_day_idx >= len(self.days) - 1 and self.current_hour_idx == len(self.hours) - 1

        # Update available rides if not done
        if not done:
            self.available_rides = self._get_rides_for_hour(
                self.hours[self.current_hour_idx],
                self.days[self.current_day_idx]
            )

        # Update total reward
        self.total_reward += reward

        # Add additional info
        info["total_reward"] = self.total_reward
        info["episode_rides"] = self.episode_rides

        # Update performance metrics
        self.perf_metrics["step_time"] += time.time() - start_time
        self.perf_metrics["steps_taken"] += 1

        return self._get_observation(), reward, done, info

    def _get_rides_for_hour(self, hour, day):
        """
        Get available rides for the current hour and day with performance enhancements.
        
        Args:
            hour (int): Hour of day
            day (int): Day of week
            
        Returns:
            list: Available rides as dictionaries
        """
        start_time = time.time()

        # Filter rides for current hour and day
        df_hour = (
            self.ride_pool
                .filter(f"pickup_hour = {hour} AND pickup_day = {day}")
                .limit(self.max_rides)
        )

        # Convert to list of dictionaries
        rides = [row.asDict() for row in df_hour.collect()]

        # If using parallel processing, batch process all rides
        if self.use_parallel and rides:
            # Prepare request batch
            requests = [(ride["PULocationID"], hour) for ride in rides]

            # Batch simulate
            simulations = self.api.batch_simulate(requests)

            # Apply simulations to rides
            for ride, sim in zip(rides, simulations):
                ride["surge_multiplier"] = sim["surge_multiplier"]
                ride["eta"] = sim["eta"]

                # Add distance to pickup feature
                ride["distance_to_pickup"] = abs(ride["PULocationID"] - self.current_location)

                # Add profit per minute feature - Vectorized style
                ride["profit_per_minute"] = ride["fare_amount"] / max(1, ride["trip_duration_minutes"])
        else:
            # Process sequentially
            for ride in rides:
                # Use simulated data
                data = self.api.simulate_real_time_data(ride["PULocationID"], hour)
                ride["surge_multiplier"] = data["surge_multiplier"]
                ride["eta"] = data["eta"]

                # Add distance to pickup feature
                ride["distance_to_pickup"] = abs(ride["PULocationID"] - self.current_location)

                # Add profit per minute feature
                ride["profit_per_minute"] = ride["fare_amount"] / max(1, ride["trip_duration_minutes"])

        # Update performance metrics
        self.perf_metrics["get_rides_time"] += time.time() - start_time

        return rides

    # Week 4: Performance Tuning - Vectorized operations
    def _get_observation(self):
        """
        Create observation using vectorized operations for better performance.
        
        Returns:
            numpy.ndarray: Environment observation vector
        """
        start_time = time.time()

        # Use pre-computed cyclical features for better performance
        hour_sin = self.hour_sin[self.current_hour_idx]
        hour_cos = self.hour_cos[self.current_hour_idx]
        day_sin = self.day_sin[self.current_day_idx]
        day_cos = self.day_cos[self.current_day_idx]

        # Location context (simplified)
        is_high_demand = 0.0  # Simplified feature

        # Context features
        context_features = np.array([
            hour_sin, hour_cos,
            day_sin, day_cos,
            self.current_location / self.max_location_id,
            is_high_demand
        ], dtype=np.float32)

        # Vectorized computation for ride features
        # Pre-allocate numpy array instead of list building for better performance
        ride_features = np.zeros(8 * self.max_rides, dtype=np.float32)

        if self.available_rides:
            # Extract ride properties in a vectorized way
            for i, r in enumerate(self.available_rides[:self.max_rides]):
                ride_features[i*8:(i+1)*8] = [
                    r["PULocationID"] / self.max_location_id,
                    r["DOLocationID"] / self.max_location_id,
                    r["distance_to_pickup"] / self.max_location_id,
                    r["fare_amount"] / 100.0,
                    r["trip_duration_minutes"] / 60.0,
                    r["profit_per_minute"] / 10.0,
                    r["surge_multiplier"],
                    r["eta"] / 600.0
                ]

        # Combine context and ride features efficiently
        obs = np.concatenate([context_features, ride_features])

        # Update performance metrics
        self.perf_metrics["get_obs_time"] += time.time() - start_time

        return obs

    def _calculate_reward(self, ride):
        """
        Calculate reward for accepting a ride.
        
        Args:
            ride (dict): Ride information
            
        Returns:
            float: Calculated reward
        """
        params = self.reward_params

        # Base reward is fare amount
        fare = ride["fare_amount"]
        base_reward = params["fare_weight"] * fare

        # Time opportunity cost
        duration = ride["trip_duration_minutes"]
        time_cost = params["time_cost_weight"] * duration

        # Surge bonus
        surge = ride.get("surge_multiplier", 1.0)
        surge_bonus = params["surge_bonus_weight"] * fare * (surge - 1.0)

        # Calculate total reward
        total_reward = base_reward + time_cost + surge_bonus

        return total_reward

    def get_performance_metrics(self):
        """
        Return environment performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        metrics = self.perf_metrics.copy()
        if metrics["steps_taken"] > 0:
            metrics["avg_step_time"] = metrics["step_time"] / metrics["steps_taken"]
            metrics["avg_get_rides_time"] = metrics["get_rides_time"] / metrics["steps_taken"]
            metrics["avg_get_obs_time"] = metrics["get_obs_time"] / metrics["steps_taken"]

        # Add API cache statistics
        metrics["api_cache"] = self.api.get_cache_stats()

        return metrics