"""
Optimized Uber API client with performance enhancements.
This module demonstrates techniques from Week 2 (Performance Tips), 
Week 6 (Numba), and Week 9 (Python Concurrency).
"""

import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from numba import jit

class OptimizedUberAPI:
    """
    Enhanced API wrapper with performance optimizations including:
    - LRU caching for repeated calculations
    - Numba JIT compilation for numerical operations
    - ThreadPoolExecutor for parallel API requests
    """
    def __init__(self, api_key, use_cache=True, use_numba=True):
        """
        Initialize the API wrapper with optimization flags.
        
        Args:
            api_key (str): API key for authentication
            use_cache (bool): Whether to use LRU caching for repeated requests
            use_numba (bool): Whether to use Numba JIT compilation
        """
        self.api_key = api_key
        self.use_cache = use_cache
        self.use_numba = use_numba
        self.request_count = 0
        self.cache_hits = 0

    @lru_cache(maxsize=1024)  # Week 2: Performance Tips - Caching
    def simulate_real_time_data_cached(self, location_id, hour_of_day):
        """
        Cached version of the data simulation function.
        
        Args:
            location_id (int): Location ID
            hour_of_day (int): Hour of day (0-23)
            
        Returns:
            dict: Simulated surge and ETA data
        """
        self.cache_hits += 1
        return self._simulate_real_time_data_impl(location_id, hour_of_day)

    def simulate_real_time_data(self, location_id, hour_of_day):
        """
        Main interface that decides whether to use cache.
        
        Args:
            location_id (int): Location ID
            hour_of_day (int): Hour of day (0-23)
            
        Returns:
            dict: Simulated surge and ETA data
        """
        self.request_count += 1

        if self.use_cache:
            return self.simulate_real_time_data_cached(location_id, hour_of_day)
        else:
            return self._simulate_real_time_data_impl(location_id, hour_of_day)

    def _simulate_real_time_data_impl(self, location_id, hour_of_day):
        """
        Implementation of the simulation logic.
        
        Args:
            location_id (int): Location ID
            hour_of_day (int): Hour of day (0-23)
            
        Returns:
            dict: Simulated surge and ETA data
        """
        # Call the Numba-optimized version if enabled
        if self.use_numba:
            surge, eta = self._calculate_surge_and_eta_numba(location_id, hour_of_day)
        else:
            # Compute normally without Numba
            np.random.seed(location_id * 100 + hour_of_day)

            # Base values
            base_surge = 1.0
            base_eta = 180

            # Peak hours logic (7-9 AM and 5-7 PM)
            is_peak_hour = (7 <= hour_of_day <= 9) or (17 <= hour_of_day <= 19)

            # Add surge during peak hours
            if is_peak_hour and np.random.random() < 0.4:
                surge_increase = 0.2 + 0.8 * np.random.random()
                base_surge += surge_increase
            elif np.random.random() < 0.1:
                surge_increase = 0.1 + 0.4 * np.random.random()
                base_surge += surge_increase

            # Adjust ETA based on hour and location
            if is_peak_hour:
                base_eta += int(120 * np.random.random())

            location_factor = (location_id % 10) / 10.0
            location_eta = int(120 * location_factor)

            surge = base_surge
            eta = base_eta + location_eta

        return {
            "surge_multiplier": round(surge, 1),
            "eta": eta
        }

    # Week 6: Numba JIT compilation for performance
    @staticmethod
    @jit(nopython=True)
    def _calculate_surge_and_eta_numba(location_id, hour_of_day):
        """
        Numba-optimized calculation of surge and ETA.
        
        Args:
            location_id (int): Location ID
            hour_of_day (int): Hour of day (0-23)
            
        Returns:
            tuple: (surge_multiplier, eta)
        """
        # Set seed for reproducibility
        np.random.seed(location_id * 100 + hour_of_day)

        # Base values
        base_surge = 1.0
        base_eta = 180

        # Peak hours logic (7-9 AM and 5-7 PM)
        is_peak_hour = (7 <= hour_of_day <= 9) or (17 <= hour_of_day <= 19)

        # Add surge during peak hours
        if is_peak_hour and np.random.random() < 0.4:
            surge_increase = 0.2 + 0.8 * np.random.random()
            base_surge += surge_increase
        elif np.random.random() < 0.1:
            surge_increase = 0.1 + 0.4 * np.random.random()
            base_surge += surge_increase

        # Adjust ETA based on hour and location
        if is_peak_hour:
            base_eta += int(120 * np.random.random())

        location_factor = (location_id % 10) / 10.0
        location_eta = int(120 * location_factor)

        return base_surge, base_eta + location_eta

    # Week 9: Python Concurrency for parallel API requests
    def batch_simulate(self, requests):
        """
        Perform simulations in parallel using ThreadPoolExecutor.
        
        Args:
            requests: List of (location_id, hour_of_day) tuples
            
        Returns:
            list: Simulation results for each request
        """
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Map requests to executor
            futures = [
                executor.submit(self.simulate_real_time_data, loc_id, hour)
                for loc_id, hour in requests
            ]
            # Gather results
            return [future.result() for future in futures]

    # Performance statistics
    def get_cache_stats(self):
        """
        Return cache performance statistics.
        
        Returns:
            dict: Cache statistics including hit rate
        """
        if not self.request_count:
            return {"hit_rate": 0, "requests": 0, "hits": 0}

        return {
            "hit_rate": self.cache_hits / self.request_count,
            "requests": self.request_count,
            "hits": self.cache_hits
        }

    def convert_location_id_to_coordinates(self, location_id):
        """
        Convert location ID to geographic coordinates.
        
        Args:
            location_id (int): Location ID
            
        Returns:
            tuple: (latitude, longitude)
        """
        np.random.seed(location_id)
        lat = 40.7 + 0.1 * np.random.random()
        lng = -74.0 + 0.1 * np.random.random()
        return lat, lng
