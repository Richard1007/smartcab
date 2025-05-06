"""
RidePoolBuilder module handles data processing with optimized PySpark operations.
This module demonstrates techniques from Week 13 (BigData with PySpark).
"""

import time
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, unix_timestamp, dayofweek

logger = logging.getLogger("taxi-rl.data")

class OptimizedRidePoolBuilder:
    """
    Enhanced RidePoolBuilder with PySpark optimizations.
    Demonstrates advanced PySpark techniques including proper Spark configuration,
    caching strategies, and method chaining.
    """
    def __init__(self, parquet_paths, spark_config=None):
        """
        Initialize the RidePoolBuilder with paths to parquet files.
        
        Args:
            parquet_paths (str or list): Path(s) to parquet file(s)
            spark_config (dict, optional): Custom Spark configuration parameters
        """
        self.parquet_paths = parquet_paths if isinstance(parquet_paths, list) else [parquet_paths]
        
        # Default optimized Spark configuration
        default_config = {
            "spark.sql.execution.arrow.pyspark.enabled": "true",  # Enable Arrow optimization
            "spark.driver.memory": "4g",  # Optimize memory usage
            "spark.sql.shuffle.partitions": "20"  # Optimize partitioning
        }
        
        # Override defaults with custom config if provided
        if spark_config:
            default_config.update(spark_config)
        
        # Initialize Spark with optimized configuration
        spark_builder = (SparkSession.builder.appName("OptimizedRidePool"))
        
        # Apply all configurations
        for key, value in default_config.items():
            spark_builder = spark_builder.config(key, value)
            
        self.spark = spark_builder.getOrCreate()

        # Create a counter for performance metrics
        self.metrics = {"load_time": 0, "process_time": 0, "build_time": 0}

    def run(self, limit=10000):
        """
        Process the data and build a ride pool with performance optimizations.
        
        Args:
            limit (int, optional): Limit the number of rows in the final result
            
        Returns:
            tuple: (ride_pool_df, metrics_dict)
        """
        # Start timer for the entire process
        start_time = time.time()
        all_dfs = []

        # Load data from all parquet files
        for path in self.parquet_paths:
            logger.info(f"Loading data from: {path}")
            df = self.spark.read.parquet(path)
            all_dfs.append(df)

        # Combine dataframes if multiple files
        if len(all_dfs) > 1:
            df = all_dfs[0]
            for additional_df in all_dfs[1:]:
                df = df.unionAll(additional_df)
        else:
            df = all_dfs[0]

        self.metrics["load_time"] = time.time() - start_time
        logger.info(f"Data loading completed in {self.metrics['load_time']:.2f} seconds")
        logger.info(f"Combined {len(self.parquet_paths)} files with total rows: {df.count()}")

        # Process the data with optimizations
        processed_df = self._process_data(df)
        
        # Build the final ride pool
        ride_pool = self._build_ride_pool(processed_df, limit)
        
        # Performance summary
        total_time = time.time() - start_time
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"Created ride pool with {ride_pool.count()} rides")

        # Return the ride pool and metrics
        return ride_pool, self.metrics
    
    def _process_data(self, df):
        """
        Process the data with optimizations.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Processed DataFrame
        """
        # Start processing timer
        process_start = time.time()

        # Performance optimization:
        # 1. Select only needed columns early (projection pushdown)
        # 2. Chain filters for better readability and performance
        # 3. Cache the cleaned dataframe for reuse
        needed_columns = [
            "tpep_pickup_datetime", "tpep_dropoff_datetime",
            "PULocationID", "DOLocationID", "fare_amount", "trip_distance"
        ]

        df_clean = (df.select(*needed_columns)
                   .filter((col("fare_amount") > 0) &
                          (col("fare_amount") < 500) &
                          (col("trip_distance") > 0) &
                          (col("trip_distance") < 100) &
                          col("tpep_pickup_datetime").isNotNull() &
                          col("tpep_dropoff_datetime").isNotNull())
                   .cache())  # Cache for repeated use

        # Add time features using method chaining
        df_with_features = (df_clean
            .withColumn("pickup_hour", hour("tpep_pickup_datetime"))
            .withColumn("pickup_day", dayofweek("tpep_pickup_datetime"))
            .withColumn(
                "trip_duration_minutes",
                (unix_timestamp("tpep_dropoff_datetime") -
                 unix_timestamp("tpep_pickup_datetime")) / 60
            )
            .filter(
                (col("trip_duration_minutes") > 0) &
                (col("trip_duration_minutes") < 120)
            )
        )

        self.metrics["process_time"] = time.time() - process_start
        logger.info(f"Data processing completed in {self.metrics['process_time']:.2f} seconds")
        
        return df_with_features
    
    def _build_ride_pool(self, df, limit=10000):
        """
        Build the final ride pool from the processed data.
        
        Args:
            df: Processed DataFrame
            limit (int, optional): Limit the number of rows in the result
            
        Returns:
            Final ride pool DataFrame
        """
        # Start build timer
        build_start = time.time()

        # Select relevant columns for the ride pool
        ride_pool = df.select(
            "pickup_hour",
            "pickup_day",
            "PULocationID",
            "DOLocationID",
            "fare_amount",
            "trip_duration_minutes"
        )

        # Apply limit if specified
        if limit:
            ride_pool = ride_pool.limit(limit)

        self.metrics["build_time"] = time.time() - build_start
        logger.info(f"Ride pool building completed in {self.metrics['build_time']:.2f} seconds")
        
        return ride_pool
